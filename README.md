# Snow/Plate Wrapper (RU)

Сервис принимает две RTSP-камеры (номерная и снеговая), отслеживает пересечение заданных линий, берёт по одному кадру с каждой камеры, отправляет их в Gemini (процент снега + номер), после чего шлёт multipart/form-data на внешний backend (`UPSTREAM_URL`).

## Поток обработки
- `stream_processor.py` (запускается вместе с FastAPI `api.py`):
  1) Читает `PLATE_CAMERA_RTSP` и `SNOW_CAMERA_RTSP` через FFmpeg.
  2) YOLOv8 (`yolov8n.pt`) детектит грузовики (truck=7) не на каждом кадре, а каждые `STREAM_DETECTION_EVERY_N_FRAMES`.
     Между запусками используется "удержание" последних детекций `DETS_HOLD_SECONDS`, чтобы не моргало и трекер не разваливался.
  3) Детектор пересечения наклонной линии работает по номерной камере (plate): линия задаётся `PLATE_LINE_*`.
  4) При пересечении сохраняются кадры (в память), вызывается Gemini с двумя фото: `snow_snapshot` + `plate_snapshot`.
  5) Формируется `event` и отправляется на `UPSTREAM_URL` как multipart (`event` JSON строкой + файлы `photos`).
- `api.py` — FastAPI (эндпоинты `/health`, вспомогательные), поднимает `stream_processor` при `ENABLE_STREAM_PROCESSOR=true`.

## Что отправляем на backend (multipart/form-data)
- Поле `event` (JSON строкой):
  - `camera_id` (`PLATE_CAMERA_ID`)
  - `event_time` — локальное время с `LOCAL_TZ_OFFSET_HOURS` (ISO 8601)
  - `plate` — номер из Gemini; если не распознан, строка `"None"`
  - `confidence` — `plate_confidence` из Gemini (0 если нет номера)
  - `direction`, `lane`, `vehicle` (пустой объект)
  - `snow_volume_percentage`, `snow_volume_confidence` — из Gemini (0 если не было оценки)
  - `matched_snow` — true/false
  - `gemini_result` — сырой ответ Gemini (для отладки)
- Поле `photos` (может быть несколько):
  - `detectionPicture.jpg` — кадр с номерной камеры
  - `snowSnapshot.jpg` — кадр со снеговой камеры
  - (Если есть другие кропы — также под ключом `photos`)

## Ключевые переменные окружения (`app.env`)
В проде на Render **файл `app.env` не нужен**: переменные задаются как **Environment Variables** (секреты).
Файл `app.env`/`.env` — только для локальной разработки.

```
UPSTREAM_URL=...
PLATE_CAMERA_ID=shahovskoye

PLATE_CAMERA_RTSP=rtsp://...
SNOW_CAMERA_RTSP=rtsp://...

# Линии (нормированные координаты 0..1, можно под углом)
PLATE_LINE_X1=0.000
PLATE_LINE_Y1=0.845
PLATE_LINE_X2=1.000
PLATE_LINE_Y2=0.600
PLATE_LINE_DIRECTION=down   # up/down/left/right/any

# Детект и трекинг
STREAM_DETECTION_EVERY_N_FRAMES=5
DETS_HOLD_SECONDS=1.5
STREAM_MIN_CONFIDENCE=0.28
STREAM_MIN_BBOX_AREA=3000
CROSS_EVENT_COOLDOWN_S=2.0

FFMPEG_OUT_W=1920
FFMPEG_OUT_H=1080
FFMPEG_INPUT_FPS=6
# По умолчанию в проде лучше глушить спам ffmpeg "error while decoding MB ..."
FFMPEG_LOGLEVEL=fatal
# Включать только для диагностики:
# LOG_FFMPEG_STDERR=true

GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
LOCAL_TZ_OFFSET_HOURS=5    # Астана = +5

ENABLE_STREAM_PROCESSOR=true
```

## Запуск
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python -m uvicorn api:app --host 0.0.0.0 --port 8000
```
Логи в консоли: `Plate crossing ...`, `Snow crossing ...`, `Gemini result ...`, `Upstream: status=...`.

## Линии и предпросмотр
- `preview_lines.py` — показывает две камеры (переключение 1/2 как "вкладки"), позволяет двигать наклонные линии:
  - `1/2` выбор линии, `TAB/C` выбор точки A/B, `WASD` мелкий шаг, `IJKL` крупный, `R` перечитать `app.env`.
  - После закрытия выводит координаты для `app.env`.

## Логи/ошибки
- В консоли: crossing, Gemini, отправка на backend.
- Если номер не распознан — в `plate` отправляется строка `"None"` (backend не примет, вернёт 400; видно в логах).
- 403 от backend означает, что номер не в whitelist (на стороне backend).

## Что принимать на стороне backend
- Ожидайте multipart/form-data:
  - `event` — JSON строкой (см. ключи выше).
  - `photos` — одно или несколько файлов (JPEG). Имена файлов типовые: `detectionPicture.jpg`, `snowSnapshot.jpg` (но полагайтесь на ключ `photos`, не на имя).
