from __future__ import annotations

import os
import json
import threading
import time
import subprocess
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Optional, Tuple, Dict, Any, Deque, Union

import cv2
import numpy as np
import httpx


# =========================
# 0) Env loader (app.env)
# =========================

def _load_env_vars() -> Optional[str]:
    """Загружает переменные окружения из app.env с override=True."""
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), "app.env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)
            return env_path
        else:
            print(f"[STREAM] WARNING: app.env not found at {env_path}, using system env vars")
            return None
    except ImportError:
        print("[STREAM] WARNING: python-dotenv not installed, using system env vars only")
        return None

_env_path = _load_env_vars()
if _env_path:
    print(f"[STREAM] Loaded environment from: {_env_path}")


def _get_env_float(key: str, fallback_key: str | None = None, default: float = 0.6) -> float:
    v = os.getenv(key)
    if v is None and fallback_key:
        v = os.getenv(fallback_key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        print(f"[STREAM] WARNING: Invalid float value for {key}: {v}, using default {default}")
        return default


def _get_env_str(key: str, fallback_key: str | None = None, default: str = "down") -> str:
    v = os.getenv(key)
    if v is None and fallback_key:
        v = os.getenv(fallback_key)
    if v is None:
        return default
    return str(v).strip().lower()


def _get_line_from_env(prefix: str, default_y: float) -> tuple[float, float, float, float, str]:
    """
    Возвращает (x1, y1, x2, y2, direction) в нормированных координатах [0..1].
    Поддерживает старый формат *_LINE_Y_POSITION как горизонтальную линию.
    """
    dir_val = _get_env_str(f"{prefix}_LINE_DIRECTION", "LINE_DIRECTION", "down")

    y_only = os.getenv(f"{prefix}_LINE_Y_POSITION")
    if y_only:
        y = float(y_only)
        return 0.0, y, 1.0, y, dir_val

    def _float_env(k: str, default: float) -> float:
        try:
            return float(os.getenv(k, f"{default:.3f}"))
        except ValueError:
            return default

    x1 = _float_env(f"{prefix}_LINE_X1", 0.0)
    y1 = _float_env(f"{prefix}_LINE_Y1", default_y)
    x2 = _float_env(f"{prefix}_LINE_X2", 1.0)
    y2 = _float_env(f"{prefix}_LINE_Y2", default_y)
    return x1, y1, x2, y2, dir_val


def _now_local_iso() -> str:
    """
    Возвращает текущее время с учётом LOCAL_TZ_OFFSET_HOURS в ISO 8601.
    Пример: offset=5 -> 2025-01-21T12:34:56+05:00
    """
    offset_hours = float(os.getenv("LOCAL_TZ_OFFSET_HOURS", "0"))
    offset = timedelta(hours=offset_hours)
    tz = timezone(offset)
    return datetime.now(tz=tz).isoformat()


# =========================
# 1) Settings
# =========================

PLATE_CAMERA_RTSP = os.getenv("PLATE_CAMERA_RTSP", "rtsp://USER:PASSWORD@HOST:554/Streaming/Channels/101")
SNOW_CAMERA_RTSP  = os.getenv("SNOW_CAMERA_RTSP",  "rtsp://USER:PASSWORD@HOST:554/Streaming/Channels/101")

PLATE_LINE_X1, PLATE_LINE_Y1, PLATE_LINE_X2, PLATE_LINE_Y2, PLATE_LINE_DIRECTION = _get_line_from_env("PLATE", 0.6)
SNOW_LINE_X1,  SNOW_LINE_Y1,  SNOW_LINE_X2,  SNOW_LINE_Y2,  SNOW_LINE_DIRECTION  = _get_line_from_env("SNOW", 0.6)

MIN_CONFIDENCE = float(os.getenv("STREAM_MIN_CONFIDENCE", "0.5"))
MIN_BBOX_AREA  = int(os.getenv("STREAM_MIN_BBOX_AREA", "10000"))
DETECTION_INTERVAL = int(os.getenv("STREAM_DETECTION_INTERVAL", "3"))

TRACK_MAX_AGE          = int(os.getenv("TRACK_MAX_AGE", "30"))
TRACK_MIN_HITS         = int(os.getenv("TRACK_MIN_HITS", "3"))
TRACK_IOU_THRESHOLD    = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))
TRACK_CROSS_COOLDOWN_S = float(os.getenv("TRACK_CROSS_COOLDOWN_S", "1.0"))

DEDUP_WINDOW_SECONDS = float(os.getenv("STREAM_DEDUP_WINDOW_SECONDS", "5.0"))

SHOW_STREAM_WINDOW = os.getenv("SHOW_STREAM_WINDOW", "false").strip().lower() == "true"

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://snowops-anpr-service.onrender.com/api/v1/anpr/events")
PLATE_CAMERA_ID = os.getenv("PLATE_CAMERA_ID", "camera-001")

FFMPEG_OUT_W = int(os.getenv("FFMPEG_OUT_W", "1280"))
FFMPEG_OUT_H = int(os.getenv("FFMPEG_OUT_H", "720"))

USE_FFMPEG_DIRECT = os.getenv("USE_FFMPEG_DIRECT", "false").strip().lower() == "true"
FFMPEG_BIN_ENV = os.getenv("FFMPEG_BIN", "").strip()

RECONNECT_BASE_DELAY_S = float(os.getenv("RECONNECT_BASE_DELAY_S", "30.0"))   # стартовая пауза
RECONNECT_MAX_DELAY_S  = float(os.getenv("RECONNECT_MAX_DELAY_S", "600.0"))  # потолок паузы
NO_FRAME_RECONNECT_THRESHOLD = int(os.getenv("NO_FRAME_RECONNECT_THRESHOLD", "1500"))

# Принудительно настраиваем FFMPEG backend: TCP, таймаут ~5с, небольшой буфер, тихий лог FFmpeg
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|buffer_size;1024000|loglevel;quiet",
)

def _silence_opencv_logs() -> None:
    """Глушим логи OpenCV/FFmpeg; учитываем разные версии OpenCV."""
    try:
        # Новый API (OpenCV >=4.5)
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        return
    except Exception:
        pass
    try:
        # Старый API (cv2.setLogLevel)
        cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
    except Exception:
        # Если нет ни одного API — просто продолжаем
        pass

_silence_opencv_logs()

print(f"[STREAM] Plate camera line: ({PLATE_LINE_X1:.3f},{PLATE_LINE_Y1:.3f})-({PLATE_LINE_X2:.3f},{PLATE_LINE_Y2:.3f}), dir={PLATE_LINE_DIRECTION}")
print(f"[STREAM] Snow camera line:  ({SNOW_LINE_X1:.3f},{SNOW_LINE_Y1:.3f})-({SNOW_LINE_X2:.3f},{SNOW_LINE_Y2:.3f}), dir={SNOW_LINE_DIRECTION}")
print(f"[STREAM] USE_FFMPEG_DIRECT={USE_FFMPEG_DIRECT}, FFMPEG_OUT={FFMPEG_OUT_W}x{FFMPEG_OUT_H}")


def _resolve_ffmpeg_bin() -> Optional[str]:
    """
    Решаем проблему 'ffmpeg виден в одном терминале, но не виден в Cursor/venv':
    - если задан FFMPEG_BIN -> используем его
    - иначе пробуем shutil.which("ffmpeg")
    """
    if FFMPEG_BIN_ENV:
        if os.path.exists(FFMPEG_BIN_ENV):
            return FFMPEG_BIN_ENV
        print(f"[FFMPEG] WARNING: FFMPEG_BIN is set but file not found: {FFMPEG_BIN_ENV}")

    p = shutil.which("ffmpeg")
    if p:
        return p
    return None


# =========================
# 2) Models
# =========================

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]      # x1,y1,x2,y2
    center: Tuple[int, int]              # cx,cy
    confidence: float
    age: int
    hits: int
    last_seen_ts: float
    crossed: bool
    direction: Optional[str]
    last_cross_ts: float = 0.0
    prev_center: Optional[Tuple[int, int]] = None  # Для визуализации направления (только текущий кадр)
    class_id: int = 7  # Класс транспорта: 2=car, 7=truck
    prev_center: Optional[Tuple[int, int]] = None  # Для визуализации направления


@dataclass
class TimestampedFrame:
    frame: np.ndarray
    timestamp: float


# =========================
# 3) Line crossing detector
# =========================

class LineCrossingDetector:
    """
    Простой IOU-трекер + детектор пересечения наклонной линии по центру bbox.
    Линия задаётся двумя точками (нормированные координаты 0..1). Направление:
    down/up/left/right/any — проверяется по вектору движения центра bbox.
    """
    def __init__(self, x1r: float, y1r: float, x2r: float, y2r: float, direction: str = "down"):
        self.line_x1 = float(x1r)
        self.line_y1 = float(y1r)
        self.line_x2 = float(x2r)
        self.line_y2 = float(y2r)
        self.direction = direction.strip().lower()
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    @staticmethod
    def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        inter = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def process_frame(self, frame: np.ndarray, detections: list[tuple[int, int, int, int, float, int]]) -> list[Track]:
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return []

        now_ts = time.time()
        x1 = int(w * self.line_x1)
        y1 = int(h * self.line_y1)
        x2 = int(w * self.line_x2)
        y2 = int(h * self.line_y2)

        lx = x2 - x1
        ly = y2 - y1
        seg_len_sq = lx * lx + ly * ly
        if seg_len_sq < 1:
            return []

        # состариваем
        for tr in self.tracks.values():
            tr.age += 1

        matched_dets: set[int] = set()
        crossed_now_tracks: list[Track] = []

        sorted_tracks = sorted(self.tracks.items(), key=lambda kv: (kv[1].hits, -kv[1].age), reverse=True)

        for track_id, tr in sorted_tracks:
            best_iou = 0.0
            best_det_idx: Optional[int] = None
            best_score = 0.0

            # Вычисляем предсказанную позицию на основе предыдущего центра (для лучшего трекинга)
            prev_cx, prev_cy = tr.center
            # Простое предсказание: предполагаем, что объект продолжает движение в том же направлении
            # Используем последние два центра для оценки скорости
            if tr.prev_center:
                vx = prev_cx - tr.prev_center[0]
                vy = prev_cy - tr.prev_center[1]
                predicted_cx = prev_cx + vx
                predicted_cy = prev_cy + vy
            else:
                predicted_cx, predicted_cy = prev_cx, prev_cy

            for det_idx, (x1_det, y1_det, x2_det, y2_det, conf, cls_id) in enumerate(detections):
                if det_idx in matched_dets:
                    continue
                iou = self._iou(tr.bbox, (x1_det, y1_det, x2_det, y2_det))
                
                # Вычисляем расстояние между центрами
                cx_det = (x1_det + x2_det) // 2
                cy_det = (y1_det + y2_det) // 2
                center_dist = ((cx_det - predicted_cx) ** 2 + (cy_det - predicted_cy) ** 2) ** 0.5
                
                # Комбинированный score: IOU + близость к предсказанной позиции
                # Нормализуем расстояние (максимальное расстояние ~1000 пикселей)
                dist_score = max(0, 1.0 - center_dist / 500.0)
                combined_score = iou * 0.7 + dist_score * 0.3
                
                # Используем более мягкий порог: либо IOU >= порог, либо комбинированный score хороший
                if (iou >= TRACK_IOU_THRESHOLD or combined_score >= 0.15) and combined_score > best_score:
                    best_iou = iou
                    best_score = combined_score
                    best_det_idx = det_idx

            if best_det_idx is None:
                continue

            x1_det, y1_det, x2_det, y2_det, conf, cls_id = detections[best_det_idx]
            cx = (x1_det + x2_det) // 2
            cy = (y1_det + y2_det) // 2

            prev_cx, prev_cy = tr.center
            crossed_now = False

            if (not tr.crossed) and (now_ts - tr.last_cross_ts >= TRACK_CROSS_COOLDOWN_S):
                side_prev = (prev_cx - x1) * ly - (prev_cy - y1) * lx
                side_curr = (cx - x1) * ly - (cy - y1) * lx

                t = ((cx - x1) * lx + (cy - y1) * ly) / seg_len_sq
                if -0.1 <= t <= 1.1:
                    sign_change = side_prev == 0 or side_curr == 0 or (side_prev * side_curr < 0)

                    mvx = cx - prev_cx
                    mvy = cy - prev_cy
                    dir_ok = True
                    if self.direction == "down":
                        dir_ok = mvy > 0  # Движение вниз (увеличение Y)
                    elif self.direction == "up":
                        dir_ok = mvy < 0  # Движение вверх (уменьшение Y)
                    elif self.direction == "right":
                        dir_ok = mvx > 0
                    elif self.direction == "left":
                        dir_ok = mvx < 0

                    crossed_now = sign_change and dir_ok

            tr.bbox = (x1_det, y1_det, x2_det, y2_det)
            tr.center = (cx, cy)
            tr.confidence = conf
            tr.class_id = cls_id  # Сохраняем класс транспорта
            tr.age = 0
            tr.hits += 1
            tr.last_seen_ts = now_ts

            if crossed_now:
                tr.crossed = True
                tr.direction = self.direction
                tr.last_cross_ts = now_ts
                crossed_now_tracks.append(tr)

            matched_dets.add(best_det_idx)

        # новые треки - проверяем, что детекция не перекрывается сильно с существующими треками
        for det_idx, (x1, y1, x2, y2, conf, cls_id) in enumerate(detections):
            if det_idx in matched_dets:
                continue

            # Проверяем, не перекрывается ли эта детекция с существующими треками
            # (даже если они не были сопоставлены из-за низкого IOU)
            overlaps_existing = False
            for existing_tr in self.tracks.values():
                iou_with_existing = self._iou((x1, y1, x2, y2), existing_tr.bbox)
                # Если IOU достаточно высокий, значит это та же машина
                if iou_with_existing > 0.3:  # Порог для предотвращения дубликатов
                    overlaps_existing = True
                    break
            
            if overlaps_existing:
                continue  # Пропускаем эту детекцию, чтобы не создавать дубликат

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            tr = Track(
                track_id=self.next_track_id,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                confidence=conf,
                age=0,
                hits=1,
                last_seen_ts=now_ts,
                crossed=False,
                direction=None,
                last_cross_ts=0.0,
                prev_center=None,
                class_id=cls_id,
            )
            self.tracks[self.next_track_id] = tr
            self.next_track_id += 1

        # чистим старые и треки, которые уехали с кадра
        to_remove = []
        for tid, tr in self.tracks.items():
            x1, y1, x2, y2 = tr.bbox
            
            # Проверяем, находится ли bbox в пределах кадра (с небольшим запасом)
            margin = 50  # Запас в пикселях
            is_in_frame = (x1 >= -margin and y1 >= -margin and 
                          x2 <= w + margin and y2 <= h + margin)
            
            # Удаляем треки, которые уехали с кадра
            if not is_in_frame:
                to_remove.append(tid)
                continue
            
            # Удаляем очень старые треки или треки с очень малым количеством попаданий
            if tr.age > TRACK_MAX_AGE:
                to_remove.append(tid)
            elif tr.hits < TRACK_MIN_HITS and tr.age > 10:
                to_remove.append(tid)
        for tid in to_remove:
            self.tracks.pop(tid, None)

        return crossed_now_tracks


# =========================
# 4) FFmpeg RTSP Reader (STABLE)
# =========================

class FFmpegRTSPReader:
    """
    Надёжный RTSP reader через ffmpeg:
      - используем явный путь к ffmpeg (FFMPEG_BIN), чтобы не зависеть от PATH Cursor/venv
      - stderr -> DEVNULL (иначе ffmpeg может зависнуть по буферу stderr)
      - stdout читаем в отдельном thread и держим last_frame
      - read() ждёт кадр ограниченное время
      - масштабируем до FFMPEG_OUT_W x FFMPEG_OUT_H
    """
    def __init__(self, rtsp_url: str, name: str):
        self.rtsp_url = rtsp_url
        self.name = name

        self.width = FFMPEG_OUT_W
        self.height = FFMPEG_OUT_H
        self.frame_size = self.width * self.height * 3  # bgr24

        self.process: Optional[subprocess.Popen] = None

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._last_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._has_frame = threading.Event()

        self._ffmpeg_bin = _resolve_ffmpeg_bin()

    def start(self) -> bool:
        if not self._ffmpeg_bin:
            print("[FFMPEG] ERROR: ffmpeg not found for this process.")
            print("[FFMPEG] Fix: set FFMPEG_BIN in app.env using: where.exe ffmpeg")
            return False

        try:
            vf = f"scale={self.width}:{self.height}"

            cmd = [
                self._ffmpeg_bin,
                "-hide_banner",
                "-loglevel", "error",
                "-nostats",

                "-rtsp_transport", "tcp",

                # делаем поток более терпимым к битому h264
                "-fflags", "+nobuffer+discardcorrupt",
                "-flags", "low_delay",
                "-err_detect", "ignore_err",

                # (опционально) уменьшить задержки анализа
                "-analyzeduration", "0",
                "-probesize", "32",

                "-i", self.rtsp_url,
                "-an",
                "-vf", vf,
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-"
            ]

            creationflags = 0
            if os.name == "nt":
                try:
                    creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
                except Exception:
                    creationflags = 0

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.frame_size * 4,
                creationflags=creationflags,
            )

            if self.process.stdout is None:
                self.release()
                return False

            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name=f"ffmpeg-reader-{self.name}")
            self._thread.start()

            print(f"[FFMPEG:{self.name}] started via: {self._ffmpeg_bin} output={self.width}x{self.height}")
            return True
        except Exception as e:
            print(f"[FFMPEG:{self.name}] start failed: {e}")
            self.release()
            return False

    def _loop(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None

        stdout = self.process.stdout
        need = self.frame_size

        while not self._stop.is_set():
            if self.process.poll() is not None:
                break

            buf = bytearray(need)
            mv = memoryview(buf)
            got = 0

            try:
                while got < need and not self._stop.is_set():
                    chunk = stdout.read(need - got)
                    if not chunk:
                        break
                    mv[got:got + len(chunk)] = chunk
                    got += len(chunk)
            except Exception:
                break

            if got != need:
                time.sleep(0.02)
                continue

            try:
                frame = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 3))
                with self._lock:
                    self._last_frame = frame
                self._has_frame.set()
            except Exception:
                continue

        self._has_frame.set()

    def read(self, timeout_s: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        if self.process is None or self.process.poll() is not None:
            return False, None

        if not self._has_frame.wait(timeout=timeout_s):
            return False, None

        with self._lock:
            if self._last_frame is None:
                return False, None
            return True, self._last_frame.copy()

    def isOpened(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def release(self) -> None:
        self._stop.set()
        try:
            self._has_frame.set()
        except Exception:
            pass

        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        self._thread = None


RTSPHandle = Union[cv2.VideoCapture, FFmpegRTSPReader]


# =========================
# 5) StreamProcessor
# =========================

class StreamProcessor:
    def __init__(self, merger):
        self.merger = merger
        _load_env_vars()

        self.use_ffmpeg_direct = USE_FFMPEG_DIRECT

        pl_x1, pl_y1, pl_x2, pl_y2, pl_dir = _get_line_from_env("PLATE", PLATE_LINE_Y1)
        sn_x1, sn_y1, sn_x2, sn_y2, sn_dir = _get_line_from_env("SNOW", SNOW_LINE_Y1)

        self.plate_detector = LineCrossingDetector(pl_x1, pl_y1, pl_x2, pl_y2, pl_dir)
        self.snow_detector  = LineCrossingDetector(sn_x1, sn_y1, sn_x2, sn_y2, sn_dir)

        print("[STREAM] Initialized detectors:")
        print(f"[STREAM]   Plate: ({pl_x1:.3f},{pl_y1:.3f})-({pl_x2:.3f},{pl_y2:.3f}) dir={pl_dir}")
        print(f"[STREAM]   Snow:  ({sn_x1:.3f},{sn_y1:.3f})-({sn_x2:.3f},{sn_y2:.3f}) dir={sn_dir}")

        self.plate_cap: Optional[RTSPHandle] = None
        self.snow_cap: Optional[RTSPHandle] = None

        self._stop_event = threading.Event()
        self._snow_thread: Optional[threading.Thread] = None
        self._plate_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None

        self._snow_frame_buffer: Deque[TimestampedFrame] = deque(maxlen=120)
        self._snow_crossing_frames: Deque[dict] = deque(maxlen=20)
        self._snow_buffer_lock = threading.Lock()
        self._snow_cross_lock = threading.Lock()
        
        # Ожидание фото с другой камеры для группировки (без привязки к track_id)
        self._pending_snow_photos: Deque[dict] = deque()  # Очередь фото со снеговой камеры: {frame, timestamp, bbox}
        self._pending_plate_photos: Deque[dict] = deque()  # Очередь фото с номерной камеры: {frame, timestamp, bbox}
        self._pending_lock = threading.Lock()
        self._pending_timeout = 2.5  # Ждем фото с другой камеры максимум 2.5 секунды

        self._processed_plates: Dict[str, float] = {}
        self._plates_lock = threading.Lock()

        self._task_queue: "deque[dict]" = deque()
        self._task_lock = threading.Lock()
        self._task_signal = threading.Event()

        # Rate limiting для Gemini и R2 (не более 1 запроса в 2 секунды)
        self._last_gemini_request_ts: float = 0.0
        self._last_r2_request_ts: float = 0.0
        self._rate_limit_seconds: float = 2.0
        self._rate_limit_lock = threading.Lock()

        # Event loop для async операций в worker thread
        self._worker_loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_loop_thread: Optional[threading.Thread] = None

        self.yolo_model = None
        self._yolo_lock = threading.Lock()
        self._load_yolo_model()

        # Окна будут созданы в потоках обработки (как в hik-anpr-wrapper)
        self._snow_window_name = "Snow Camera" if SHOW_STREAM_WINDOW else None
        self._plate_window_name = "Plate Camera" if SHOW_STREAM_WINDOW else None
        if SHOW_STREAM_WINDOW:
            print("[STREAM] Video windows enabled (SHOW_STREAM_WINDOW=true)")

    # --------- YOLO ---------

    def _load_yolo_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"[STREAM] YOLO model loaded: {YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[STREAM] ERROR: Failed to load YOLO model: {e}")
            self.yolo_model = None

    def _detect_vehicles(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float, int]]:
        """Детектирует транспорт и возвращает (x1, y1, x2, y2, conf, class_id)."""
        if self.yolo_model is None:
            return []
        try:
            with self._yolo_lock:
                # COCO: car=2, truck=7 - детектируем оба, принимаем car как грузовик для событий
                # Сниженный порог уверенности для лучшего распознавания при низкой яркости
                results = self.yolo_model(frame, classes=[2, 7], conf=MIN_CONFIDENCE, verbose=False)

            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())  # Класс: 2=car, 7=truck
                    area = (x2 - x1) * (y2 - y1)
                    if area < MIN_BBOX_AREA:
                        continue
                    detections.append((x1, y1, x2, y2, conf, cls_id))
            return detections
        except Exception as e:
            print(f"[STREAM] Error in vehicle detection: {e}")
            return []

    # --------- frame helpers ---------

    def _validate_frame(self, frame: np.ndarray) -> bool:
        try:
            if frame is None or frame.size == 0:
                return False
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                return False
            if frame.dtype != np.uint8:
                return False
            if np.all(frame == 0):
                return False
            if np.any(np.isnan(frame)):
                return False
            return True
        except Exception:
            return False

    def _encode_frame_to_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        if not self._validate_frame(frame):
            return None
        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                return None
            b = buf.tobytes()
            return b if b else None
        except Exception as e:
            print(f"[STREAM] Error encoding frame: {e}")
            return None

    def _get_snow_frame(self, prefer_crossing: bool = True) -> Optional[np.ndarray]:
        now_ts = time.time()

        if prefer_crossing:
            with self._snow_cross_lock:
                while self._snow_crossing_frames and (now_ts - self._snow_crossing_frames[0]["timestamp"] > 2.0):
                    self._snow_crossing_frames.popleft()

                if self._snow_crossing_frames:
                    item = self._snow_crossing_frames[-1]
                    fr = item["frame"]
                    if self._validate_frame(fr):
                        return fr.copy()

        with self._snow_buffer_lock:
            if not self._snow_frame_buffer:
                return None
            for i in range(len(self._snow_frame_buffer) - 1, -1, -1):
                fr = self._snow_frame_buffer[i].frame
                if self._validate_frame(fr):
                    return fr.copy()

        return None

    # --------- task queue ---------

    def _push_task(self, task: dict) -> None:
        with self._task_lock:
            self._task_queue.append(task)
            self._task_signal.set()

    def _pop_task(self) -> Optional[dict]:
        with self._task_lock:
            if not self._task_queue:
                self._task_signal.clear()
                return None
            return self._task_queue.popleft()

    def _create_plate_variants(self, plate_frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> tuple[Optional[bytes], Optional[bytes]]:
        """Создает featurePicture (ближе) и licensePlatePicture (обрезанная) из detectionPicture."""
        try:
            x1, y1, x2, y2 = bbox
            h, w = plate_frame.shape[:2]
            
            # Ограничиваем bbox размерами кадра
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            # featurePicture: увеличиваем область вокруг нижней части bbox (где обычно номер)
            # Берем нижнюю треть bbox и расширяем на 20% со всех сторон
            bbox_h = y2 - y1
            plate_y1 = y1 + int(bbox_h * 0.67)  # Нижняя треть
            plate_y2 = y2
            plate_x1 = x1
            plate_x2 = x2
            
            # Расширяем область на 20%
            expand_w = int((plate_x2 - plate_x1) * 0.2)
            expand_h = int((plate_y2 - plate_y1) * 0.2)
            feature_x1 = max(0, plate_x1 - expand_w)
            feature_y1 = max(0, plate_y1 - expand_h)
            feature_x2 = min(w, plate_x2 + expand_w)
            feature_y2 = min(h, plate_y2 + expand_h)
            
            feature_crop = plate_frame[feature_y1:feature_y2, feature_x1:feature_x2]
            if feature_crop.size > 0:
                # Увеличиваем в 2 раза для лучшего качества
                feature_resized = cv2.resize(feature_crop, (feature_crop.shape[1] * 2, feature_crop.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
                feature_jpeg = self._encode_frame_to_jpeg(feature_resized)
            else:
                feature_jpeg = None
            
            # licensePlatePicture: обрезаем только область номера (еще более узкая область)
            # Берем нижнюю четверть bbox
            license_y1 = y1 + int(bbox_h * 0.75)
            license_y2 = y2
            license_x1 = x1
            license_x2 = x2
            
            # Расширяем немного для контекста
            license_expand_w = int((license_x2 - license_x1) * 0.1)
            license_expand_h = int((license_y2 - license_y1) * 0.1)
            license_crop_x1 = max(0, license_x1 - license_expand_w)
            license_crop_y1 = max(0, license_y1 - license_expand_h)
            license_crop_x2 = min(w, license_x2 + license_expand_w)
            license_crop_y2 = min(h, license_y2 + license_expand_h)
            
            license_crop = plate_frame[license_crop_y1:license_crop_y2, license_crop_x1:license_crop_x2]
            if license_crop.size > 0:
                # Увеличиваем в 3 раза для максимального качества
                license_resized = cv2.resize(license_crop, (license_crop.shape[1] * 3, license_crop.shape[0] * 3), interpolation=cv2.INTER_LINEAR)
                license_jpeg = self._encode_frame_to_jpeg(license_resized)
            else:
                license_jpeg = None
            
            return feature_jpeg, license_jpeg
        except Exception as e:
            print(f"[STREAM] Error creating plate variants: {e}")
            return None, None

    def _crop_plate_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], padding_ratio: float = 0.3) -> Optional[np.ndarray]:
        """
        Вырезает область вокруг bbox с padding.
        padding_ratio: 0.3 означает 30% отступ со всех сторон.
        """
        if frame is None or len(frame.shape) != 3:
            print("[CROP] Invalid frame")
            return None
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Проверяем валидность bbox
        if x2 <= x1 or y2 <= y1:
            print(f"[CROP] Invalid bbox: ({x1}, {y1}, {x2}, {y2})")
            return None
        
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            print(f"[CROP] Bbox out of bounds: bbox=({x1}, {y1}, {x2}, {y2}), frame=({w}, {h})")
            # Обрезаем bbox до границ кадра
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                print(f"[CROP] Bbox became invalid after clipping")
                return None
        
        # Вычисляем padding
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        pad_w = int(bbox_w * padding_ratio)
        pad_h = int(bbox_h * padding_ratio)
        
        # Расширяем bbox с padding
        crop_x1 = max(0, x1 - pad_w)
        crop_y1 = max(0, y1 - pad_h)
        crop_x2 = min(w, x2 + pad_w)
        crop_y2 = min(h, y2 + pad_h)
        
        print(f"[CROP] Original bbox=({x1}, {y1}, {x2}, {y2}), crop=({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}), frame=({w}, {h})")
        
        # Вырезаем область
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if crop is None or crop.size == 0:
            print(f"[CROP] Empty crop result")
            return None
        
        # Проверяем минимальный размер
        if crop.shape[0] < 50 or crop.shape[1] < 50:
            print(f"[CROP] Crop too small: {crop.shape}")
            return None
        
        print(f"[CROP] Success: crop size={crop.shape[1]}x{crop.shape[0]}")
        return crop

    def _process_grouped_photos(self, snow_frame: np.ndarray, plate_frame: np.ndarray, timestamp: float, track_id: int, plate_bbox: Optional[Tuple[int, int, int, int]] = None) -> None:
        """Обрабатывает сгруппированные фото с двух камер и отправляет в очередь."""
        try:
            # Сохраняем оригинальный кадр для R2
            plate_original_jpeg = self._encode_frame_to_jpeg(plate_frame)
            snow_jpeg = self._encode_frame_to_jpeg(snow_frame)
            
            if not plate_original_jpeg or not snow_jpeg:
                print(f"[STREAM] Failed to encode original frames for track_id={track_id}")
                return
            
            # Создаем кроп для Gemini (если есть bbox)
            plate_crop_jpeg = None
            if plate_bbox:
                print(f"[STREAM] Cropping plate region for Gemini with bbox={plate_bbox}, frame size={plate_frame.shape[1]}x{plate_frame.shape[0]}")
                plate_crop = self._crop_plate_region(plate_frame, plate_bbox, padding_ratio=0.3)
                if plate_crop is not None:
                    print(f"[STREAM] Using cropped plate frame for Gemini: {plate_crop.shape[1]}x{plate_crop.shape[0]}")
                    plate_crop_jpeg = self._encode_frame_to_jpeg(plate_crop)
                else:
                    print("[STREAM] Failed to crop plate region, will use original for Gemini")
            else:
                print("[STREAM] No plate_bbox provided, will use original for Gemini")
            
            # Если кроп не создан, используем оригинал для Gemini
            if plate_crop_jpeg is None:
                plate_crop_jpeg = plate_original_jpeg
            
            # Создаем варианты из оригинального кадра для R2
            feature_jpeg = None
            license_jpeg = None
            if plate_bbox:
                feature_jpeg, license_jpeg = self._create_plate_variants(plate_frame, plate_bbox)
            
            print(f"[STREAM] Grouped photos ready (track_id={track_id}), sending to Gemini and R2...")
            
            # Отправляем задачу в очередь для обработки
            self._push_task({
                "snow_jpeg": snow_jpeg,
                "plate_original_jpeg": plate_original_jpeg,  # Оригинал для R2
                "plate_crop_jpeg": plate_crop_jpeg,  # Кроп для Gemini
                "feature_jpeg": feature_jpeg,  # Вариант для R2
                "license_jpeg": license_jpeg,  # Вариант для R2
                "timestamp": timestamp,
                "track_id": track_id
            })
        except Exception as e:
            print(f"[STREAM] Error processing grouped photos: {e}")

    async def _process_grouped_task(self, task: dict, now_ts: float) -> None:
        """Обрабатывает задачу с уже сгруппированными фото."""
        snow_jpeg = task.get("snow_jpeg")
        plate_crop_jpeg = task.get("plate_crop_jpeg")  # Кроп для Gemini
        plate_original_jpeg = task.get("plate_original_jpeg")  # Оригинал для R2
        feature_jpeg = task.get("feature_jpeg")  # Вариант для R2
        license_jpeg = task.get("license_jpeg")  # Вариант для R2
        
        if not snow_jpeg or not plate_crop_jpeg or not plate_original_jpeg:
            print("[STREAM] Missing grouped photos, skipping")
            return
        
        # Rate limiting для Gemini: не более 1 запроса в 2 секунды
        with self._rate_limit_lock:
            time_since_last_gemini = now_ts - self._last_gemini_request_ts
            if time_since_last_gemini < self._rate_limit_seconds:
                wait_time = self._rate_limit_seconds - time_since_last_gemini
                print(f"[STREAM] Rate limit: skipping Gemini request (last request {time_since_last_gemini:.2f}s ago, need {self._rate_limit_seconds}s)")
                return
            self._last_gemini_request_ts = now_ts
        
        # Отправляем в Gemini только кроп
        try:
            gemini_result = await self.merger.analyze_with_gemini(
                snow_photo=snow_jpeg,
                plate_photo_1=plate_crop_jpeg,  # Кроп для Gemini
                plate_photo_2=None,
                camera_plate=None,
            )
            print(f"[STREAM] Gemini result: {gemini_result}")
        except Exception as e:
            print(f"[STREAM] Gemini error: {e}")
            return

        plate = (gemini_result or {}).get("plate")
        plate_conf = float((gemini_result or {}).get("plate_confidence", 0.0) or 0.0)

        # Если номер не распознан — явно ставим строку "None", чтобы не было пустого значения
        if plate:
            plate = str(plate).strip().upper()
        else:
            plate = "None"
            plate_conf = 0.0

        # Дедупликация только для реальных номеров
        if plate != "None":
            with self._plates_lock:
                old = [p for p, ts in self._processed_plates.items() if (now_ts - ts) > DEDUP_WINDOW_SECONDS]
                for p in old:
                    self._processed_plates.pop(p, None)

                if plate in self._processed_plates:
                    print(f"[STREAM] Duplicate plate (Gemini): {plate}, skipping")
                    return

                self._processed_plates[plate] = now_ts

        now_iso = _now_local_iso()

        event_data = {
            "camera_id": PLATE_CAMERA_ID,
            "event_time": now_iso,
            "plate": plate,
            "confidence": plate_conf,
            "direction": self.plate_detector.direction,
            "lane": 0,
            "vehicle": {},
            "plate_source": "gemini",
            "snow_volume_percentage": float((gemini_result or {}).get("snow_percentage", 0.0) or 0.0),
            "snow_volume_confidence": float((gemini_result or {}).get("snow_confidence", 0.0) or 0.0),
            "matched_snow": True,
            "gemini_result": gemini_result,
            "timestamp": now_iso,
        }

        # Rate limiting для R2: не более 1 запроса в 2 секунды
        with self._rate_limit_lock:
            time_since_last_r2 = now_ts - self._last_r2_request_ts
            if time_since_last_r2 < self._rate_limit_seconds:
                wait_time = self._rate_limit_seconds - time_since_last_r2
                print(f"[STREAM] Rate limit: skipping R2 upload (last request {time_since_last_r2:.2f}s ago, need {self._rate_limit_seconds}s)")
                return
            self._last_r2_request_ts = now_ts

        try:
            data = {"event": json.dumps(event_data, ensure_ascii=False)}
            # Порядок фоток для R2: 0=detection (оригинал), 1=feature (ближе), 2=license (самый ближний), 3=snow
            files = [
                ("photos", ("detectionPicture.jpg", plate_original_jpeg, "image/jpeg")),  # Оригинал
            ]
            # Добавляем варианты, если они есть
            if feature_jpeg:
                files.append(("photos", ("featurePicture.jpg", feature_jpeg, "image/jpeg")))
            if license_jpeg:
                files.append(("photos", ("licensePlatePicture.jpg", license_jpeg, "image/jpeg")))
            # Снеговая фотка всегда последняя
            files.append(("photos", ("snowSnapshot.jpg", snow_jpeg, "image/jpeg")))
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(UPSTREAM_URL, data=data, files=files)
            print(f"[STREAM] Upstream: status={resp.status_code}, ok={resp.is_success}, photos={len(files)}, body={resp.text[:200]}")
        except Exception as e:
            print(f"[STREAM] Upstream send error: {e}")

    # --------- RTSP open/read ---------

    @staticmethod
    def _mask_url(rtsp_url: str) -> str:
        if "@" in rtsp_url:
            a, b = rtsp_url.split("@", 1)
            return f"rtsp://***@{b}"
        return rtsp_url

    def _open_rtsp(self, rtsp_url: str, name: str, retries: int = 5) -> Optional[RTSPHandle]:
        print(f"[STREAM] Opening {name}: {self._mask_url(rtsp_url)}")

        if self.use_ffmpeg_direct:
            for attempt in range(1, retries + 1):
                reader = FFmpegRTSPReader(rtsp_url, name=name.replace(" ", "_").lower())
                if reader.start():
                    ok, fr = reader.read(timeout_s=3.0)
                    if ok and fr is not None and fr.size > 0:
                        h, w = fr.shape[:2]
                        print(f"[STREAM] ✓ {name} FFmpeg OK: {w}x{h}")
                        return reader
                    print(f"[STREAM] ⚠ {name}: ffmpeg started but no frames yet (attempt {attempt})")
                    reader.release()
                time.sleep(1.2 * attempt)

            print(f"[STREAM] ✗ {name}: FFmpeg failed after {retries} attempts")
            return None

        # OpenCV fallback (если вдруг захочешь выключить USE_FFMPEG_DIRECT)
        for attempt in range(1, retries + 1):
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            time.sleep(1.0)
            
            opened = cap.isOpened() if not isinstance(cap, FFmpegRTSPReader) else cap.isOpened()
            if not opened:
                try:
                    cap.release()
                except Exception:
                    pass
                print(f"[STREAM] ✗ {name}: OpenCV isOpened=False (attempt {attempt})")
                time.sleep(1.2 * attempt)
                continue

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Пытаемся прочитать кадр
            ok_any = False
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    ok_any = True
                    h, w = frame.shape[:2]
                    print(f"[STREAM] ✓ {name} OpenCV OK: {w}x{h}")
                    break
                time.sleep(0.2)

            if ok_any:
                return cap

            try:
                cap.release()
            except Exception:
                pass

            print(f"[STREAM] ⚠ {name}: OpenCV opened but no frames (attempt {attempt})")
            time.sleep(1.2 * attempt)

        print(f"[STREAM] ✗ {name}: OpenCV failed after {retries} attempts")
        return None

    def _read_frame(self, handle: RTSPHandle) -> Tuple[bool, Optional[np.ndarray]]:
        if isinstance(handle, FFmpegRTSPReader):
            return handle.read(timeout_s=1.0)

        if handle is None or not handle.isOpened():
            return False, None

        ret, frame = handle.read()
        if ret and frame is not None and frame.size > 0 and len(frame.shape) == 3:
            return True, frame

        return False, None

    def _close_handle(self, handle: Optional[RTSPHandle]) -> None:
        if handle is None:
            return
        try:
            if isinstance(handle, FFmpegRTSPReader):
                handle.release()
            else:
                handle.release()
        except Exception:
            pass

    # --------- async processing ---------

    async def _process_crossing_async(self, plate_frame: np.ndarray) -> None:
        now_ts = time.time()

        snow_frame = self._get_snow_frame(prefer_crossing=True)
        if snow_frame is None:
            print("[STREAM] No snow frame available, skipping")
            return

        plate_bytes = self._encode_frame_to_jpeg(plate_frame)
        snow_bytes  = self._encode_frame_to_jpeg(snow_frame)
        if plate_bytes is None or snow_bytes is None:
            print("[STREAM] Failed to encode frames, skipping")
            return

        try:
            gemini_result = await self.merger.analyze_with_gemini(
                snow_photo=snow_bytes,
                plate_photo_1=plate_bytes,
                plate_photo_2=None,
                camera_plate=None,
            )
            print(f"[STREAM] Gemini result: {gemini_result}")
        except Exception as e:
            print(f"[STREAM] Gemini error: {e}")
            return

        plate = (gemini_result or {}).get("plate")
        plate_conf = float((gemini_result or {}).get("plate_confidence", 0.0) or 0.0)

        # Если номер не распознан — явно ставим строку "None", чтобы не было пустого значения
        if plate:
            plate = str(plate).strip().upper()
        else:
            plate = "None"
            plate_conf = 0.0

        # Дедупликация только для реальных номеров
        if plate != "None":
            with self._plates_lock:
                old = [p for p, ts in self._processed_plates.items() if (now_ts - ts) > DEDUP_WINDOW_SECONDS]
                for p in old:
                    self._processed_plates.pop(p, None)

                if plate in self._processed_plates:
                    print(f"[STREAM] Duplicate plate (Gemini): {plate}, skipping")
                    return

                self._processed_plates[plate] = now_ts

        now_iso = _now_local_iso()

        event_data = {
            "camera_id": PLATE_CAMERA_ID,
            "event_time": now_iso,
            "plate": plate,
            "confidence": plate_conf,
            "direction": self.plate_detector.direction,
            "lane": 0,
            "vehicle": {},
            "plate_source": "gemini",
            "snow_volume_percentage": float((gemini_result or {}).get("snow_percentage", 0.0) or 0.0),
            "snow_volume_confidence": float((gemini_result or {}).get("snow_confidence", 0.0) or 0.0),
            "matched_snow": True,
            "gemini_result": gemini_result,
            "timestamp": now_iso,
        }

        try:
            data = {"event": json.dumps(event_data, ensure_ascii=False)}
            files = [
                ("photos", ("detectionPicture.jpg", plate_bytes, "image/jpeg")),
                ("photos", ("snowSnapshot.jpg", snow_bytes, "image/jpeg")),
            ]
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(UPSTREAM_URL, data=data, files=files)
            print(f"[STREAM] Upstream: status={resp.status_code}, ok={resp.is_success}, body={resp.text[:200]}")
        except Exception as e:
            print(f"[STREAM] Upstream send error: {e}")

    def _worker_loop(self) -> None:
        """Worker thread с собственным event loop для async операций."""
        print("[STREAM] Worker thread started")
        import asyncio
        
        # Создаем новый event loop для этого потока
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._worker_loop = loop
        
        try:
            while not self._stop_event.is_set():
                if not self._task_signal.wait(timeout=0.2):
                    # Очищаем старые pending фото
                    self._cleanup_old_pending_photos()
                    continue

                task = self._pop_task()
                if task is None:
                    continue

                try:
                    # Проверяем, есть ли уже сгруппированные фото
                    if "snow_jpeg" in task and "plate_crop_jpeg" in task:
                        loop.run_until_complete(self._process_grouped_task(task, time.time()))
                    elif "plate_frame" in task:
                        loop.run_until_complete(self._process_crossing_async(task["plate_frame"]))
                except Exception as e:
                    print(f"[STREAM] Worker error: {e}")
                    import traceback
                    traceback.print_exc()
        finally:
            loop.close()
            self._worker_loop = None
            print("[STREAM] Worker thread stopped")

    def _cleanup_old_pending_photos(self) -> None:
        """Очищает старые pending фото, которые не были сгруппированы за 2.5 секунды."""
        now_ts = time.time()
        with self._pending_lock:
            # Очищаем старые фото снеговой камеры (пара не нашлась за 2.5 секунды)
            while self._pending_snow_photos:
                if (now_ts - self._pending_snow_photos[0]["timestamp"]) > self._pending_timeout:
                    self._pending_snow_photos.popleft()
                    print(f"[STREAM] Snow photo expired (no pair found in 2.5s)")
                else:
                    break  # Остальные фото еще свежие
            
            # Очищаем старые фото номерной камеры (пара не нашлась за 2.5 секунды)
            while self._pending_plate_photos:
                if (now_ts - self._pending_plate_photos[0]["timestamp"]) > self._pending_timeout:
                    self._pending_plate_photos.popleft()
                    print(f"[STREAM] Plate photo expired (no pair found in 2.5s)")
                else:
                    break  # Остальные фото еще свежие

    # --------- loops ---------

    def _snow_processing_loop(self) -> None:
        """Обработка потока снеговой камеры, как в hik-anpr-wrapper."""
        print("[STREAM] Starting snow stream processing...")

        # Создаем окно в этом потоке, как в hik-anpr-wrapper
        window_name = self._snow_window_name
        if window_name:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)

        delay = RECONNECT_BASE_DELAY_S
        fail = 0
        frame_counter = 0
        MAX_FAILS = 15

        while not self._stop_event.is_set():
            # 1) гарантируем, что соединение есть
            if self.snow_cap is None:
                cap = self._open_rtsp(SNOW_CAMERA_RTSP, "Snow Camera", retries=5)
                if cap is None:
                    print(f"[STREAM] Snow camera unavailable, retry in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(RECONNECT_MAX_DELAY_S, max(RECONNECT_BASE_DELAY_S, delay * 1.6))
                    continue

                self.snow_cap = cap
                delay = RECONNECT_BASE_DELAY_S
                fail = 0
                frame_counter = 0

            # 2) проверяем "жив ли" handle
            opened = self.snow_cap.isOpened() if not isinstance(self.snow_cap, FFmpegRTSPReader) else self.snow_cap.isOpened()
            if not opened:
                print("[STREAM] Snow stream closed, reconnecting...")
                self._close_handle(self.snow_cap)
                self.snow_cap = None
                time.sleep(0.5)
                continue

            # 3) читаем кадр (как в hik-anpr-wrapper)
            ret, frame = self._read_frame(self.snow_cap)
            if not ret or frame is None or frame.size == 0:
                fail += 1
                if fail >= MAX_FAILS:
                    print("[STREAM] Snow: too many failed reads, reconnecting...")
                    self._close_handle(self.snow_cap)
                    self.snow_cap = None
                    time.sleep(2)
                    fail = 0
                time.sleep(0.05)
                continue

            # 4) кадр ок
            fail = 0
            now_ts = time.time()
            
            # Логируем первый успешный кадр
            if frame_counter == 1:
                h, w = frame.shape[:2]
                print(f"[STREAM] Snow: First frame received! Size: {w}x{h}")

            # Сохраняем кадр в буфер
            with self._snow_buffer_lock:
                while self._snow_frame_buffer and (now_ts - self._snow_frame_buffer[0].timestamp > 3.0):
                    self._snow_frame_buffer.popleft()
                self._snow_frame_buffer.append(TimestampedFrame(frame=frame.copy(), timestamp=now_ts))

            frame_counter += 1
            
            # Детекция (каждый N-й кадр)
            if frame_counter % max(1, DETECTION_INTERVAL) == 0:
                dets = self._detect_vehicles(frame)
                if dets:
                    crossed = self.snow_detector.process_frame(frame, dets)
                    for tr in crossed:
                        # Делаем фото для любого транспорта (car=2 или truck=7) при пересечении линии
                        # Принимаем car как грузовик для событий пересечения
                        print(f"[STREAM] Snow crossing: track_id={tr.track_id}, center={tr.center}, class={tr.class_id}")
                        # Сохраняем фото снеговой камеры и проверяем, есть ли фото с номерной
                        # Не сравниваем по track_id - просто ждем любое фото с другой камеры
                        with self._pending_lock:
                            # Проверяем, есть ли уже фото с номерной камеры (любое, не важно какой track_id)
                            plate_photo = None
                            if self._pending_plate_photos:
                                # Берем самое свежее фото с номерной камеры
                                plate_photo = self._pending_plate_photos.popleft()
                            
                            if plate_photo and (now_ts - plate_photo["timestamp"]) <= self._pending_timeout:
                                # Есть фото с номерной камеры - группируем и отправляем
                                self._process_grouped_photos(
                                    snow_frame=frame.copy(),
                                    plate_frame=plate_photo["frame"],
                                    timestamp=now_ts,
                                    track_id=tr.track_id,
                                    plate_bbox=plate_photo.get("bbox")
                                )
                            else:
                                # Сохраняем фото снеговой камеры в очередь и ждем фото с номерной
                                self._pending_snow_photos.append({
                                    "frame": frame.copy(),
                                    "timestamp": now_ts,
                                    "bbox": tr.bbox
                                })
                                print(f"[STREAM] Snow photo saved, waiting for plate photo")

            # Обновляем окно прямо здесь, ТОЧНО как в hik-anpr-wrapper
            if window_name:
                # Рисуем линию на кадре (SNOW_LINE для снеговой камеры)
                h, w = frame.shape[:2]
                vis_frame = frame.copy()
                
                # Рисуем линию детекции
                line_x1 = int(SNOW_LINE_X1 * w)
                line_y1 = int(SNOW_LINE_Y1 * h)
                line_x2 = int(SNOW_LINE_X2 * w)
                line_y2 = int(SNOW_LINE_Y2 * h)
                cv2.line(vis_frame, (line_x1, line_y1), (line_x2, line_y2), (0, 255, 0), 2)
                
                # Визуализируем только текущие треки снегового детектора (без следов)
                for track_id, tr in self.snow_detector.tracks.items():
                    x1, y1, x2, y2 = tr.bbox
                    cx, cy = tr.center
                    
                    # Рисуем bounding box (только текущий кадр, без истории)
                    color = (0, 255, 0) if tr.crossed else (255, 0, 0)  # Зеленый если пересек, красный если нет
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Рисуем ID трека
                    cv2.putText(vis_frame, f"ID:{track_id}", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Рисуем центр точкой
                    cv2.circle(vis_frame, (cx, cy), 5, color, -1)
                    
                    # Рисуем направление движения только если есть предыдущий центр (без сохранения следов)
                    if tr.prev_center:
                        prev_cx, prev_cy = tr.prev_center
                        dx = cx - prev_cx
                        dy = cy - prev_cy
                        length = max(1, int((dx*dx + dy*dy)**0.5))
                        if length > 5:  # Только если есть заметное движение
                            end_x = cx + int(dx * 30 / length)
                            end_y = cy + int(dy * 30 / length)
                            cv2.arrowedLine(vis_frame, (cx, cy), (end_x, end_y), (255, 255, 0), 2, tipLength=0.3)
                    
                    # Обновляем prev_center для следующего кадра (но не сохраняем историю - убираем следы)
                    tr.prev_center = (cx, cy)
                
                cv2.putText(vis_frame, "SNOW", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Tracks: {len(self.snow_detector.tracks)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Показываем кадр (ТОЧНО как в hik-anpr-wrapper)
                cv2.imshow(window_name, cv2.resize(vis_frame, (960, 540)))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("[STREAM] Window closed by user (Q/ESC)")
                    self._stop_event.set()
                    break

            # Небольшая пауза, чтобы не грузить CPU (как в hik-anpr-wrapper)
            time.sleep(0.01)

        self._close_handle(self.snow_cap)
        self.snow_cap = None
        
        # Закрываем окно (как в hik-anpr-wrapper)
        if window_name:
            cv2.destroyAllWindows()
        
        print("[STREAM] Snow processing loop stopped")


    def _plate_processing_loop(self) -> None:
        """Обработка потока камеры номеров, как в hik-anpr-wrapper."""
        print("[STREAM] Starting plate stream processing...")

        # Создаем окно в этом потоке, как в hik-anpr-wrapper
        window_name = self._plate_window_name
        if window_name:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)

        delay = RECONNECT_BASE_DELAY_S
        fail = 0
        frame_counter = 0
        MAX_FAILS = 15

        while not self._stop_event.is_set():
            # 1) гарантируем, что соединение есть
            if self.plate_cap is None:
                cap = self._open_rtsp(PLATE_CAMERA_RTSP, "Plate Camera", retries=5)
                if cap is None:
                    print(f"[STREAM] Plate camera unavailable, retry in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(RECONNECT_MAX_DELAY_S, max(RECONNECT_BASE_DELAY_S, delay * 1.6))
                    continue

                self.plate_cap = cap
                delay = RECONNECT_BASE_DELAY_S
                fail = 0
                frame_counter = 0

            # 2) проверяем "жив ли" handle
            opened = self.plate_cap.isOpened() if not isinstance(self.plate_cap, FFmpegRTSPReader) else self.plate_cap.isOpened()
            if not opened:
                print("[STREAM] Plate stream closed, reconnecting...")
                self._close_handle(self.plate_cap)
                self.plate_cap = None
                time.sleep(0.5)
                continue

            # 3) читаем кадр (как в hik-anpr-wrapper)
            ret, frame = self._read_frame(self.plate_cap)
            if not ret or frame is None or frame.size == 0:
                fail += 1
                if fail >= MAX_FAILS:
                    print("[STREAM] Plate: too many failed reads, reconnecting...")
                    self._close_handle(self.plate_cap)
                    self.plate_cap = None
                    time.sleep(2)
                    fail = 0
                time.sleep(0.05)
                continue

            # 4) кадр ок
            fail = 0
            now_ts = time.time()
            frame_counter += 1

            # Детекция (каждый N-й кадр)
            if frame_counter % max(1, DETECTION_INTERVAL) == 0:
                dets = self._detect_vehicles(frame)
                if dets:
                    crossed = self.plate_detector.process_frame(frame, dets)
                    for tr in crossed:
                        # Делаем фото для любого транспорта (car=2 или truck=7) при пересечении линии
                        # Принимаем car как грузовик для событий пересечения
                        print(f"[STREAM] Plate crossing: track_id={tr.track_id}, bbox={tr.bbox}, conf={tr.confidence:.2f}, class={tr.class_id}")
                        # Сохраняем фото номерной камеры и проверяем, есть ли фото со снеговой
                        # Не сравниваем по track_id - просто ждем любое фото с другой камеры
                        with self._pending_lock:
                            # Проверяем, есть ли уже фото со снеговой камеры (любое, не важно какой track_id)
                            snow_photo = None
                            if self._pending_snow_photos:
                                # Берем самое свежее фото со снеговой камеры
                                snow_photo = self._pending_snow_photos.popleft()
                            
                            if snow_photo and (now_ts - snow_photo["timestamp"]) <= self._pending_timeout:
                                # Есть фото со снеговой камеры - группируем и отправляем
                                print(f"[STREAM] Processing grouped photos with plate_bbox={tr.bbox}, frame_size={frame.shape[1]}x{frame.shape[0]}")
                                self._process_grouped_photos(
                                    snow_frame=snow_photo["frame"],
                                    plate_frame=frame.copy(),
                                    timestamp=now_ts,
                                    track_id=tr.track_id,
                                    plate_bbox=tr.bbox
                                )
                            else:
                                # Сохраняем фото номерной камеры в очередь и ждем фото со снеговой
                                self._pending_plate_photos.append({
                                    "frame": frame.copy(),
                                    "timestamp": now_ts,
                                    "bbox": tr.bbox  # Нужен для кропа при отправке в Gemini
                                })
                                print(f"[STREAM] Plate photo saved, waiting for snow photo")

            # Обновляем окно прямо здесь, ТОЧНО как в hik-anpr-wrapper
            if window_name:
                # Рисуем линию на кадре (PLATE_LINE для номерной камеры)
                h, w = frame.shape[:2]
                vis_frame = frame.copy()
                
                # Рисуем линию детекции
                line_x1 = int(PLATE_LINE_X1 * w)
                line_y1 = int(PLATE_LINE_Y1 * h)
                line_x2 = int(PLATE_LINE_X2 * w)
                line_y2 = int(PLATE_LINE_Y2 * h)
                cv2.line(vis_frame, (line_x1, line_y1), (line_x2, line_y2), (0, 255, 0), 2)
                
                # Визуализируем только текущие треки номерного детектора (без следов)
                for track_id, tr in self.plate_detector.tracks.items():
                    x1, y1, x2, y2 = tr.bbox
                    cx, cy = tr.center
                    
                    # Рисуем bounding box (только текущий кадр, без истории)
                    color = (0, 255, 0) if tr.crossed else (255, 0, 0)  # Зеленый если пересек, красный если нет
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Рисуем ID трека
                    cv2.putText(vis_frame, f"ID:{track_id}", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Рисуем центр точкой
                    cv2.circle(vis_frame, (cx, cy), 5, color, -1)
                    
                    # Рисуем направление движения только если есть предыдущий центр (без сохранения следов)
                    if tr.prev_center:
                        prev_cx, prev_cy = tr.prev_center
                        dx = cx - prev_cx
                        dy = cy - prev_cy
                        length = max(1, int((dx*dx + dy*dy)**0.5))
                        if length > 5:  # Только если есть заметное движение
                            end_x = cx + int(dx * 30 / length)
                            end_y = cy + int(dy * 30 / length)
                            cv2.arrowedLine(vis_frame, (cx, cy), (end_x, end_y), (255, 255, 0), 2, tipLength=0.3)
                    
                    # Обновляем prev_center для следующего кадра (но не сохраняем историю - убираем следы)
                    tr.prev_center = (cx, cy)
                
                cv2.putText(vis_frame, "PLATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Tracks: {len(self.plate_detector.tracks)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Показываем кадр (ТОЧНО как в hik-anpr-wrapper)
                cv2.imshow(window_name, cv2.resize(vis_frame, (960, 540)))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("[STREAM] Plate window closed by user (Q/ESC)")
                    self._stop_event.set()
                    break

            # Небольшая пауза, чтобы не грузить CPU (как в hik-anpr-wrapper)
            time.sleep(0.01)

        self._close_handle(self.plate_cap)
        self.plate_cap = None
        
        # Закрываем окно (как в hik-anpr-wrapper)
        if window_name:
            cv2.destroyWindow(window_name)
        
        print("[STREAM] Plate processing loop stopped")


    # --------- Public API ---------

    def start(self) -> None:
        if self._plate_thread and self._plate_thread.is_alive():
            print("[STREAM] Already running")
            return

        self._stop_event.clear()

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="stream-worker")
        self._worker_thread.start()

        self._snow_thread = threading.Thread(target=self._snow_processing_loop, daemon=True, name="snow-processor")
        self._snow_thread.start()

        time.sleep(1)

        self._plate_thread = threading.Thread(target=self._plate_processing_loop, daemon=True, name="plate-processor")
        self._plate_thread.start()

        print("[STREAM] Stream processor started (snow + plate + worker)")

    def stop(self) -> None:
        self._stop_event.set()
        self._task_signal.set()

        for th in [self._plate_thread, self._snow_thread, self._worker_thread]:
            if th:
                th.join(timeout=5)

        self._plate_thread = None
        self._snow_thread = None
        self._worker_thread = None

        # Закрываем окно, если было открыто
        if SHOW_STREAM_WINDOW:
            try:
                cv2.destroyAllWindows()
            except:
                pass

        print("[STREAM] Stream processor stopped")


# =========================
# Singleton helpers
# =========================

_stream_processor: Optional[StreamProcessor] = None

def init_stream_processor(merger) -> StreamProcessor:
    global _stream_processor
    if _stream_processor is None:
        _stream_processor = StreamProcessor(merger)
    return _stream_processor

def get_stream_processor() -> Optional[StreamProcessor]:
    return _stream_processor
