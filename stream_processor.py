from __future__ import annotations

import os
import json
import time
import threading
import subprocess
import shutil
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Optional, Tuple, Dict, Any, Deque, Union

import cv2
import numpy as np
import httpx


# =========================
# 0) CPU safety (Render 1 CPU)
# =========================
# Ограничиваем лишние потоки BLAS/OMP, иначе 1 CPU может быть "100%" почти всегда.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# =========================
# 1) Env loader (app.env)
# =========================
def _load_env_vars() -> Optional[str]:
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), "app.env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)
            return env_path
        print(f"[STREAM] WARNING: app.env not found at {env_path}, using system env vars")
        return None
    except ImportError:
        print("[STREAM] WARNING: python-dotenv not installed, using system env vars only")
        return None


_env_path = _load_env_vars()
if _env_path:
    print(f"[STREAM] Loaded environment from: {_env_path}")


def _get_env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip()


def _get_env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _get_env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return int(float(v))
    except ValueError:
        return default


def _get_line_from_env(prefix: str, default_y: float) -> tuple[float, float, float, float, str]:
    """
    (x1,y1,x2,y2,direction) in [0..1]
    """
    direction = _get_env_str(f"{prefix}_LINE_DIRECTION", "down").lower()

    y_only = os.getenv(f"{prefix}_LINE_Y_POSITION")
    if y_only:
        try:
            y = float(y_only)
        except ValueError:
            y = default_y
        return 0.0, y, 1.0, y, direction

    def f(k: str, d: float) -> float:
        try:
            return float(os.getenv(k, f"{d:.3f}"))
        except ValueError:
            return d

    x1 = f(f"{prefix}_LINE_X1", 0.0)
    y1 = f(f"{prefix}_LINE_Y1", default_y)
    x2 = f(f"{prefix}_LINE_X2", 1.0)
    y2 = f(f"{prefix}_LINE_Y2", default_y)
    return x1, y1, x2, y2, direction


def _now_local_iso() -> str:
    offset_hours = float(os.getenv("LOCAL_TZ_OFFSET_HOURS", "5"))
    tz = timezone(timedelta(hours=offset_hours))
    return datetime.now(tz=tz).isoformat()


# =========================
# 2) Settings
# =========================
PLATE_CAMERA_RTSP = _get_env_str("PLATE_CAMERA_RTSP", "rtsp://USER:PASSWORD@HOST:554/Streaming/Channels/101")
SNOW_CAMERA_RTSP = _get_env_str("SNOW_CAMERA_RTSP", "rtsp://USER:PASSWORD@HOST:554/Streaming/Channels/101")

# Линия пересечения — только для plate-камеры (snow нам нужен как фон/кузов)
PLATE_LINE_X1, PLATE_LINE_Y1, PLATE_LINE_X2, PLATE_LINE_Y2, PLATE_LINE_DIRECTION = _get_line_from_env("PLATE", 0.65)

UPSTREAM_URL = _get_env_str("UPSTREAM_URL", "https://snowops-anpr-service.onrender.com/api/v1/anpr/events")
PLATE_CAMERA_ID = _get_env_str("PLATE_CAMERA_ID", "camera-001")

# YOLO
YOLO_MODEL_PATH = _get_env_str("YOLO_MODEL_PATH", "yolov8n.pt")
MIN_CONFIDENCE = _get_env_float("STREAM_MIN_CONFIDENCE", 0.35)
MIN_BBOX_AREA = _get_env_int("STREAM_MIN_BBOX_AREA", 7000)

# Как часто гонять YOLO (каждый N-й кадр)
DETECTION_EVERY_N_FRAMES = _get_env_int("STREAM_DETECTION_EVERY_N_FRAMES", 5)

# Максимальная частота чтения кадров (доп. защита CPU)
PLATE_LOOP_SLEEP_S = _get_env_float("PLATE_LOOP_SLEEP_S", 0.05)  # ~20 FPS max
SNOW_LOOP_SLEEP_S = _get_env_float("SNOW_LOOP_SLEEP_S", 0.10)    # ~10 FPS max

# Стабилизация реконнекта
RECONNECT_MIN_S = _get_env_float("RECONNECT_MIN_S", 2.0)
RECONNECT_MAX_S = _get_env_float("RECONNECT_MAX_S", 60.0)

# Буфер снежной камеры
SNOW_BUFFER_SECONDS = _get_env_float("SNOW_BUFFER_SECONDS", 4.0)
SNOW_BUFFER_MAXLEN = _get_env_int("SNOW_BUFFER_MAXLEN", 80)  # кольцевой буфер

# Окно поиска ближайшего snow-кадра к моменту пересечения
SNOW_MATCH_WINDOW_S = _get_env_float("SNOW_MATCH_WINDOW_S", 2.0)

# Дедуп: не отправлять повторно один и тот же трек слишком часто
CROSS_EVENT_COOLDOWN_S = _get_env_float("CROSS_EVENT_COOLDOWN_S", 2.0)

# FFMPEG reader output
FFMPEG_OUT_W = _get_env_int("FFMPEG_OUT_W", 960)
FFMPEG_OUT_H = _get_env_int("FFMPEG_OUT_H", 540)

# Входной FPS для ffmpeg (главный способ убрать 100% CPU на Render)
FFMPEG_INPUT_FPS = _get_env_float("FFMPEG_INPUT_FPS", 6.0)

# YOLO запускаем на уменьшенной копии (resize, не crop)
YOLO_INFER_W = _get_env_int("YOLO_INFER_W", 640)
YOLO_INFER_H = _get_env_int("YOLO_INFER_H", 360)

# Качество JPEG для Gemini/Upstream
JPEG_QUALITY = _get_env_int("JPEG_QUALITY", 85)

# Dedup по номеру из Gemini
DEDUP_WINDOW_SECONDS = _get_env_float("STREAM_DEDUP_WINDOW_SECONDS", 8.0)

# ffmpeg bin
FFMPEG_BIN_ENV = _get_env_str("FFMPEG_BIN", "").strip()


print(f"[STREAM] Plate line: ({PLATE_LINE_X1:.3f},{PLATE_LINE_Y1:.3f})-({PLATE_LINE_X2:.3f},{PLATE_LINE_Y2:.3f}), dir={PLATE_LINE_DIRECTION}")
print(f"[STREAM] FFMPEG_OUT={FFMPEG_OUT_W}x{FFMPEG_OUT_H}, FFMPEG_INPUT_FPS={FFMPEG_INPUT_FPS}")
print(f"[STREAM] YOLO infer size={YOLO_INFER_W}x{YOLO_INFER_H}, detect every N frames={DETECTION_EVERY_N_FRAMES}")


def _resolve_ffmpeg_bin() -> Optional[str]:
    if FFMPEG_BIN_ENV:
        if os.path.exists(FFMPEG_BIN_ENV):
            return FFMPEG_BIN_ENV
        print(f"[FFMPEG] WARNING: FFMPEG_BIN is set but file not found: {FFMPEG_BIN_ENV}")
    p = shutil.which("ffmpeg")
    return p


# =========================
# 3) Models
# =========================
@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2 on ORIGINAL frame coords
    center: Tuple[int, int]
    confidence: float
    age: int
    hits: int
    last_seen_ts: float
    crossed: bool
    direction: Optional[str]
    last_cross_ts: float = 0.0


@dataclass
class TimestampedFrame:
    frame: np.ndarray
    timestamp: float


# =========================
# 4) Line crossing detector (simple IOU tracker)
# =========================
class LineCrossingDetector:
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
        area2 = (x2_2 - x1_2) * (y2_2 - y2_1)
        # исправим area2 (опечатки не допускаем)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def process_frame(self, frame: np.ndarray, detections: list[tuple[int, int, int, int, float]]) -> list[Track]:
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

        for tr in self.tracks.values():
            tr.age += 1

        matched_dets: set[int] = set()
        crossed_now_tracks: list[Track] = []

        sorted_tracks = sorted(self.tracks.items(), key=lambda kv: (kv[1].hits, -kv[1].age), reverse=True)

        for track_id, tr in sorted_tracks:
            best_iou = 0.0
            best_det_idx: Optional[int] = None

            for det_idx, (x1_det, y1_det, x2_det, y2_det, conf) in enumerate(detections):
                if det_idx in matched_dets:
                    continue
                iou = self._iou(tr.bbox, (x1_det, y1_det, x2_det, y2_det))
                if iou > best_iou and iou >= 0.30:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx is None:
                continue

            x1_det, y1_det, x2_det, y2_det, conf = detections[best_det_idx]
            cx = (x1_det + x2_det) // 2
            cy = (y1_det + y2_det) // 2

            prev_cx, prev_cy = tr.center
            crossed_now = False

            if (not tr.crossed) and (now_ts - tr.last_cross_ts >= CROSS_EVENT_COOLDOWN_S):
                side_prev = (prev_cx - x1) * ly - (prev_cy - y1) * lx
                side_curr = (cx - x1) * ly - (cy - y1) * lx

                t = ((cx - x1) * lx + (cy - y1) * ly) / seg_len_sq
                if -0.1 <= t <= 1.1:
                    sign_change = (side_prev == 0) or (side_curr == 0) or (side_prev * side_curr < 0)

                    mvx = cx - prev_cx
                    mvy = cy - prev_cy
                    dir_ok = True
                    if self.direction == "down":
                        dir_ok = mvy > 0
                    elif self.direction == "up":
                        dir_ok = mvy < 0
                    elif self.direction == "right":
                        dir_ok = mvx > 0
                    elif self.direction == "left":
                        dir_ok = mvx < 0

                    crossed_now = sign_change and dir_ok

            tr.bbox = (x1_det, y1_det, x2_det, y2_det)
            tr.center = (cx, cy)
            tr.confidence = conf
            tr.age = 0
            tr.hits += 1
            tr.last_seen_ts = now_ts

            if crossed_now:
                tr.crossed = True
                tr.direction = self.direction
                tr.last_cross_ts = now_ts
                crossed_now_tracks.append(tr)

            matched_dets.add(best_det_idx)

        # новые треки
        for det_idx, (x1_det, y1_det, x2_det, y2_det, conf) in enumerate(detections):
            if det_idx in matched_dets:
                continue
            cx = (x1_det + x2_det) // 2
            cy = (y1_det + y2_det) // 2
            tr = Track(
                track_id=self.next_track_id,
                bbox=(x1_det, y1_det, x2_det, y2_det),
                center=(cx, cy),
                confidence=conf,
                age=0,
                hits=1,
                last_seen_ts=now_ts,
                crossed=False,
                direction=None,
                last_cross_ts=0.0,
            )
            self.tracks[self.next_track_id] = tr
            self.next_track_id += 1

        # чистим старые треки
        to_remove = []
        for tid, tr in self.tracks.items():
            if tr.age > 40:
                to_remove.append(tid)
            elif tr.hits < 1 and tr.age > 8:
                to_remove.append(tid)
        for tid in to_remove:
            self.tracks.pop(tid, None)

        return crossed_now_tracks


# =========================
# 5) FFmpeg RTSP Reader (FPS-limited)
# =========================
class FFmpegRTSPReader:
    """
    Читает RTSP через ffmpeg в rawvideo bgr24.
    ВАЖНО:
      - stderr -> DEVNULL (иначе ffmpeg может зависнуть/забить буфер)
      - vf включает fps=... чтобы снизить CPU на Render
    """
    def __init__(self, rtsp_url: str, name: str):
        self.rtsp_url = rtsp_url
        self.name = name

        self.width = FFMPEG_OUT_W
        self.height = FFMPEG_OUT_H
        self.frame_size = self.width * self.height * 3

        self._ffmpeg_bin = _resolve_ffmpeg_bin()
        self.process: Optional[subprocess.Popen] = None

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._has_frame = threading.Event()
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_ts: float = 0.0

    def start(self) -> bool:
        if not self._ffmpeg_bin:
            print("[FFMPEG] ERROR: ffmpeg not found. Set FFMPEG_BIN or ensure ffmpeg in PATH.")
            return False

        vf = f"fps={FFMPEG_INPUT_FPS},scale={self.width}:{self.height}"
        cmd = [
            self._ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "error",
            "-nostats",

            "-rtsp_transport", "tcp",

            "-fflags", "+nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-err_detect", "ignore_err",

            "-analyzeduration", "0",
            "-probesize", "32",

            "-i", self.rtsp_url,
            "-an",
            "-vf", vf,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-",
        ]

        creationflags = 0
        if os.name == "nt":
            try:
                creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            except Exception:
                creationflags = 0

        try:
            # ВРЕМЕННО: логируем stderr для диагностики на сервере
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Изменили с DEVNULL на PIPE для диагностики
                bufsize=self.frame_size * 4,
                creationflags=creationflags,
            )
            if self.process.stdout is None:
                self.release()
                return False

            # Запускаем поток для чтения stderr (чтобы видеть ошибки FFMPEG)
            def _read_stderr():
                if self.process and self.process.stderr:
                    try:
                        for line in iter(self.process.stderr.readline, b''):
                            if line:
                                msg = line.decode('utf-8', errors='ignore').strip()
                                if msg:  # Логируем только непустые сообщения
                                    print(f"[FFMPEG:{self.name}] stderr: {msg}")
                    except Exception as e:
                        print(f"[FFMPEG:{self.name}] stderr reader error: {e}")

            stderr_thread = threading.Thread(target=_read_stderr, daemon=True, name=f"ffmpeg-stderr-{self.name}")
            stderr_thread.start()

            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name=f"ffmpeg-reader-{self.name}")
            self._thread.start()

            print(f"[FFMPEG:{self.name}] started output={self.width}x{self.height} fps={FFMPEG_INPUT_FPS}")
            return True
        except Exception as e:
            print(f"[FFMPEG:{self.name}] start failed: {e}")
            self.release()
            return False

    def _loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        stdout = self.process.stdout
        need = self.frame_size

        while not self._stop.is_set():
            if self.process.poll() is not None:
                # Процесс упал - логируем
                returncode = self.process.returncode
                print(f"[FFMPEG:{self.name}] process exited with code {returncode}")
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
                    self._last_frame_ts = time.time()
                self._has_frame.set()
            except Exception:
                continue

        self._has_frame.set()

    def read(self, timeout_s: float = 1.0, stale_s: float = 3.0) -> Tuple[bool, Optional[np.ndarray]]:
        if self.process is None or self.process.poll() is not None:
            return False, None

        if not self._has_frame.wait(timeout=timeout_s):
            return False, None

        with self._lock:
            if self._last_frame is None:
                return False, None
            age = time.time() - float(self._last_frame_ts or 0.0)
            if age > stale_s:
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


RTSPHandle = FFmpegRTSPReader


def _mask_url(rtsp_url: str) -> str:
    if "@" in rtsp_url:
        a, b = rtsp_url.split("@", 1)
        return f"rtsp://***@{b}"
    return rtsp_url


def _backoff_sleep(attempt: int) -> None:
    # Ограничиваем attempt, чтобы избежать переполнения
    max_attempt = 50  # 1.6^50 уже очень большое число
    safe_attempt = min(attempt, max_attempt)
    t = min(RECONNECT_MAX_S, RECONNECT_MIN_S * (1.6 ** safe_attempt))
    time.sleep(t)


# =========================
# 6) StreamProcessor
# =========================
class StreamProcessor:
    def __init__(self, merger):
        self.merger = merger
        _load_env_vars()

        self.detector = LineCrossingDetector(
            PLATE_LINE_X1, PLATE_LINE_Y1, PLATE_LINE_X2, PLATE_LINE_Y2, PLATE_LINE_DIRECTION
        )

        self.plate_cap: Optional[RTSPHandle] = None
        self.snow_cap: Optional[RTSPHandle] = None

        self._stop_event = threading.Event()
        self._snow_thread: Optional[threading.Thread] = None
        self._plate_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None

        # snow frames buffer
        self._snow_frame_buffer: Deque[TimestampedFrame] = deque(maxlen=SNOW_BUFFER_MAXLEN)
        self._snow_lock = threading.Lock()

        # tasks
        self._task_queue: "deque[dict]" = deque()
        self._task_lock = threading.Lock()
        self._task_signal = threading.Event()

        # dedup by plate (Gemini)
        self._processed_plates: Dict[str, float] = {}
        self._plates_lock = threading.Lock()

        # YOLO
        self.yolo_model = None
        self._yolo_lock = threading.Lock()
        self._load_yolo_model()

        self._last_plate_det_ts = 0.0
        self._last_plate_dets = []   # type: list[tuple[int,int,int,int,float]]
        self._hold_dets_seconds = float(os.getenv("DETS_HOLD_SECONDS", "1.5"))


    # ---------- helpers ----------
    def is_running(self) -> bool:
        return bool(
            self._plate_thread and self._plate_thread.is_alive()
            and self._snow_thread and self._snow_thread.is_alive()
            and self._worker_thread and self._worker_thread.is_alive()
        )

    def snow_buffer_size(self) -> int:
        with self._snow_lock:
            return len(self._snow_frame_buffer)

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
            return True
        except Exception:
            return False

    def _encode_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        if not self._validate_frame(frame):
            return None
        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)])
            if not ok:
                return None
            b = buf.tobytes()
            return b if b else None
        except Exception as e:
            print(f"[STREAM] JPEG encode error: {e}")
            return None

    # ---------- YOLO ----------
    def _load_yolo_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"[STREAM] YOLO model loaded: {YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[STREAM] ERROR: Failed to load YOLO model: {e}")
            self.yolo_model = None

    def _detect_trucks_on_frame(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """
        Детекция на уменьшенной версии кадра (resize, не crop),
        потом масштабируем bbox обратно к исходным координатам.
        """
        if self.yolo_model is None:
            return []

        try:
            h0, w0 = frame.shape[:2]
            small = cv2.resize(frame, (YOLO_INFER_W, YOLO_INFER_H), interpolation=cv2.INTER_LINEAR)
            sx = w0 / float(YOLO_INFER_W)
            sy = h0 / float(YOLO_INFER_H)

            with self._yolo_lock:
                # COCO: truck=7 (car=2 можно добавить, но ты хочешь грузовики)
                results = self.yolo_model(small, classes=[7], conf=MIN_CONFIDENCE, verbose=False)

            dets: list[tuple[int, int, int, int, float]] = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())

                    # back to original coords
                    X1 = int(x1 * sx)
                    Y1 = int(y1 * sy)
                    X2 = int(x2 * sx)
                    Y2 = int(y2 * sy)

                    area = (X2 - X1) * (Y2 - Y1)
                    if area < MIN_BBOX_AREA:
                        continue

                    dets.append((X1, Y1, X2, Y2, conf))
            return dets
        except Exception as e:
            print(f"[STREAM] YOLO detect error: {e}")
            return []

    # ---------- RTSP ----------
    def _open_rtsp(self, rtsp_url: str, name: str, retries: int = 1) -> Optional[RTSPHandle]:
        print(f"[STREAM] Opening {name}: {_mask_url(rtsp_url)}")
        for attempt in range(1, retries + 1):
            reader = FFmpegRTSPReader(rtsp_url, name=name.replace(" ", "_").lower())
            if reader.start():
                # Увеличиваем таймаут для первого чтения (RTSP может долго подключаться на сервере)
                ok, fr = reader.read(timeout_s=15.0)  # Увеличено с 3.0 до 15.0 для медленного соединения
                if ok and fr is not None and fr.size > 0:
                    hh, ww = fr.shape[:2]
                    print(f"[STREAM] ✓ {name} OK: {ww}x{hh}")
                    return reader
                print(f"[STREAM] ⚠ {name}: started but no frames (attempt {attempt})")
                # Проверяем, не упал ли процесс FFMPEG
                if reader.process and reader.process.poll() is not None:
                    returncode = reader.process.returncode
                    print(f"[STREAM] ⚠ {name}: FFMPEG process exited with code {returncode}")
                    # Пытаемся прочитать последние строки stderr
                    if reader.process.stderr:
                        try:
                            stderr_lines = reader.process.stderr.readlines()
                            if stderr_lines:
                                print(f"[STREAM] ⚠ {name}: Last FFMPEG errors:")
                                for line in stderr_lines[-5:]:  # Последние 5 строк
                                    print(f"  {line.decode('utf-8', errors='ignore').strip()}")
                        except Exception:
                            pass
                reader.release()
            time.sleep(1.0 * attempt)

        print(f"[STREAM] ✗ {name}: failed")
        return None

    def _close_handle(self, handle: Optional[RTSPHandle]) -> None:
        if handle is None:
            return
        try:
            handle.release()
        except Exception:
            pass

    def _read_frame(self, handle: RTSPHandle) -> Tuple[bool, Optional[np.ndarray]]:
        return handle.read(timeout_s=1.0, stale_s=3.0)

    # ---------- snow buffer ----------
    def _push_snow_frame(self, frame: np.ndarray, ts: float) -> None:
        with self._snow_lock:
            # drop old
            while self._snow_frame_buffer and (ts - self._snow_frame_buffer[0].timestamp > SNOW_BUFFER_SECONDS):
                self._snow_frame_buffer.popleft()
            self._snow_frame_buffer.append(TimestampedFrame(frame=frame, timestamp=ts))

    def _get_best_snow_frame(self, target_ts: float) -> Optional[np.ndarray]:
        with self._snow_lock:
            if not self._snow_frame_buffer:
                return None
            best = None
            best_dt = 1e9
            for item in self._snow_frame_buffer:
                dt = abs(item.timestamp - target_ts)
                if dt < best_dt:
                    best_dt = dt
                    best = item
            if best is None or best_dt > SNOW_MATCH_WINDOW_S:
                return None
            if self._validate_frame(best.frame):
                return best.frame.copy()
            return None

    # ---------- task queue ----------
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

    # ---------- worker ----------
    async def _process_crossing_async(self, plate_frame: np.ndarray, cross_ts: float, client: httpx.AsyncClient) -> None:
        snow_frame = self._get_best_snow_frame(cross_ts)
        if snow_frame is None:
            print("[STREAM] No snow frame near crossing time, skipping")
            return

        plate_bytes = self._encode_jpeg(plate_frame)
        snow_bytes = self._encode_jpeg(snow_frame)
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

        # dedup by plate
        now_ts = time.time()
        if plate:
            plate = str(plate).strip().upper()
            with self._plates_lock:
                old = [p for p, ts in self._processed_plates.items() if (now_ts - ts) > DEDUP_WINDOW_SECONDS]
                for p in old:
                    self._processed_plates.pop(p, None)
                if plate in self._processed_plates:
                    print(f"[STREAM] Duplicate plate (Gemini): {plate}, skipping upstream")
                    return
                self._processed_plates[plate] = now_ts
        else:
            plate = "None"
            plate_conf = 0.0

        now_iso = _now_local_iso()

        event_data = {
            "camera_id": PLATE_CAMERA_ID,
            "event_time": now_iso,
            "plate": plate,
            "confidence": plate_conf,
            "direction": self.detector.direction,
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
            resp = await client.post(UPSTREAM_URL, data=data, files=files)
            print(f"[STREAM] Upstream: status={resp.status_code}, ok={resp.is_success}, body={resp.text[:200]}")
        except Exception as e:
            print(f"[STREAM] Upstream send error: {e}")

    def _worker_loop(self) -> None:
        print("[STREAM] Worker thread started")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        timeout_s = float(os.getenv("UPSTREAM_TIMEOUT_S", "20"))
        client = httpx.AsyncClient(timeout=timeout_s)

        try:
            while not self._stop_event.is_set():
                if not self._task_signal.wait(timeout=0.5):
                    continue

                task = self._pop_task()
                if task is None:
                    continue

                plate_frame = task.get("plate_frame")
                cross_ts = float(task.get("cross_ts", 0.0) or 0.0)
                if plate_frame is None or cross_ts <= 0:
                    continue

                try:
                    loop.run_until_complete(self._process_crossing_async(plate_frame, cross_ts, client))
                except Exception as e:
                    print(f"[STREAM] Worker error: {e}")
        finally:
            try:
                loop.run_until_complete(client.aclose())
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

        print("[STREAM] Worker thread stopped")

    # ---------- loops ----------
    def _snow_loop(self) -> None:
        print("[STREAM] Snow loop started")
        attempt = 0
        self.snow_cap = None

        while not self._stop_event.is_set():
            if self.snow_cap is None:
                self.snow_cap = self._open_rtsp(SNOW_CAMERA_RTSP, "Snow Camera", retries=1)
                if self.snow_cap is None:
                    attempt += 1
                    _backoff_sleep(attempt)
                    continue
                attempt = 0

            ok, frame = self._read_frame(self.snow_cap)
            if not ok or frame is None or not self._validate_frame(frame):
                print("[STREAM] Snow: stream closed / stale, reconnecting...")
                self._close_handle(self.snow_cap)
                self.snow_cap = None
                attempt += 1
                _backoff_sleep(attempt)
                continue

            attempt = 0
            ts = time.time()
            self._push_snow_frame(frame.copy(), ts)

            time.sleep(SNOW_LOOP_SLEEP_S)

        self._close_handle(self.snow_cap)
        self.snow_cap = None
        print("[STREAM] Snow loop stopped")

    def _plate_loop(self) -> None:
        print("[STREAM] Plate loop started")
        attempt = 0
        self.plate_cap = None

        frame_i = 0

        while not self._stop_event.is_set():
            if self.plate_cap is None:
                self.plate_cap = self._open_rtsp(PLATE_CAMERA_RTSP, "Plate Camera", retries=1)
                if self.plate_cap is None:
                    attempt += 1
                    _backoff_sleep(attempt)
                    continue
                attempt = 0
                frame_i = 0

            ok, frame = self._read_frame(self.plate_cap)
            if not ok or frame is None or not self._validate_frame(frame):
                print("[STREAM] Plate: stream closed / stale, reconnecting...")
                self._close_handle(self.plate_cap)
                self.plate_cap = None
                attempt += 1
                _backoff_sleep(attempt)
                continue

            attempt = 0
            frame_i += 1
            now = time.time()

            # 1) Решаем: запускаем YOLO или берем "удержанные" детекции
            run_det = (frame_i % max(1, DETECTION_EVERY_N_FRAMES) == 0)

            dets: list[tuple[int, int, int, int, float]] = []

            if run_det:
                new_dets = self._detect_trucks_on_frame(frame)

                if new_dets:
                    # ✅ YOLO дал детекции — сохраняем как последние стабильные
                    self._last_plate_dets = new_dets
                    self._last_plate_det_ts = now
                    dets = new_dets
                else:
                    # YOLO "моргнул": если недавно были детекции — используем их
                    if (now - self._last_plate_det_ts) <= self._hold_dets_seconds:
                        dets = self._last_plate_dets
            else:
                # Между запусками YOLO тоже можем использовать последние детекции,
                # чтобы трекер/пересечение не разваливались
                if (now - self._last_plate_det_ts) <= self._hold_dets_seconds:
                    dets = self._last_plate_dets

            # 2) Даем детекции в трекер/детектор линии
            if dets:
                crossed_tracks = self.detector.process_frame(frame, dets)
                if crossed_tracks:
                    cross_ts = now
                    for tr in crossed_tracks:
                        print(f"[STREAM] CROSS ✅ track_id={tr.track_id} bbox={tr.bbox}")
                        self._push_task({"plate_frame": frame.copy(), "cross_ts": cross_ts})

            time.sleep(PLATE_LOOP_SLEEP_S)


        self._close_handle(self.plate_cap)
        self.plate_cap = None
        print("[STREAM] Plate loop stopped")

    # ---------- public ----------
    def start(self) -> None:
        if self._plate_thread and self._plate_thread.is_alive():
            print("[STREAM] Already running")
            return

        self._stop_event.clear()

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="stream-worker")
        self._snow_thread = threading.Thread(target=self._snow_loop, daemon=True, name="snow-loop")
        self._plate_thread = threading.Thread(target=self._plate_loop, daemon=True, name="plate-loop")

        self._worker_thread.start()
        self._snow_thread.start()
        time.sleep(0.5)
        self._plate_thread.start()

        print("[STREAM] ✅ Started (plate + snow + worker)")

    def stop(self) -> None:
        self._stop_event.set()
        self._task_signal.set()

        for th in [self._plate_thread, self._snow_thread, self._worker_thread]:
            if th:
                th.join(timeout=5)

        self._plate_thread = None
        self._snow_thread = None
        self._worker_thread = None

        print("[STREAM] ✅ Stopped")


_stream_processor: Optional[StreamProcessor] = None


def init_stream_processor(merger) -> StreamProcessor:
    global _stream_processor
    if _stream_processor is None:
        _stream_processor = StreamProcessor(merger)
    return _stream_processor


def get_stream_processor() -> Optional[StreamProcessor]:
    return _stream_processor
