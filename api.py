from __future__ import annotations

import os
import datetime
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse

# Загружаем переменные окружения из app.env
try:
    from dotenv import load_dotenv

    env_path = os.path.join(os.path.dirname(__file__), "app.env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=False)
        print(f"[API] Loaded environment from: {env_path}")
    else:
        print(f"[API] WARNING: app.env not found at {env_path}, using system env vars")
except ImportError:
    print("[API] WARNING: python-dotenv not installed, using system env vars only")

from combined_merger import init_merger  # должен быть у тебя

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://snowops-anpr-service.onrender.com/api/v1/anpr/events")
MERGE_WINDOW_SECONDS = int(os.getenv("MERGE_WINDOW_SECONDS", "20"))
MERGE_TTL_SECONDS = int(os.getenv("MERGE_TTL_SECONDS", "50"))
ENABLE_STREAM_PROCESSOR = os.getenv("ENABLE_STREAM_PROCESSOR", "true").strip().lower() == "true"

merger = init_merger(
    upstream_url=UPSTREAM_URL,
    window_seconds=MERGE_WINDOW_SECONDS,
    ttl_seconds=MERGE_TTL_SECONDS,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    stream_proc = None
    if ENABLE_STREAM_PROCESSOR:
        try:
            from stream_processor import init_stream_processor
            stream_proc = init_stream_processor(merger)
            stream_proc.start()
            print("[STARTUP] ✅ Stream processor started (RTSP mode)")
        except Exception as e:
            print(f"[STARTUP] ❌ Failed to start stream processor: {e}")
    else:
        print("[STARTUP] ⚠️ Stream processor disabled (ENABLE_STREAM_PROCESSOR=false)")

    try:
        yield
    finally:
        try:
            if stream_proc is None:
                from stream_processor import get_stream_processor
                stream_proc = get_stream_processor()
            if stream_proc is not None:
                stream_proc.stop()
                print("[SHUTDOWN] ✅ Stream processor stopped")
        except Exception as e:
            print(f"[SHUTDOWN] ❌ Failed to stop stream processor: {e}")


app = FastAPI(
    title="Hikvision ANPR Wrapper (RTSP mode)",
    description="RTSP: детекция пересечения линии + Gemini номер/снег + отправка upstream",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.datetime.now()
    print(f"\n{'='*80}")
    print(f"[REQUEST] {start_time.isoformat()} | {request.method} {request.url.path}")
    print(f"[REQUEST] Client: {request.client.host if request.client else 'unknown'}")

    try:
        response = await call_next(request)
        dt = (datetime.datetime.now() - start_time).total_seconds()
        print(f"[REQUEST] Response: {response.status_code} | Time: {dt:.3f}s")
        print(f"{'='*80}\n")
        return response
    except Exception as e:
        dt = (datetime.datetime.now() - start_time).total_seconds()
        print(f"[REQUEST] ERROR: {type(e).__name__}: {e} | Time: {dt:.3f}s")
        print(f"{'='*80}\n")
        raise


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/stream/status")
def stream_status() -> Dict[str, Any]:
    from stream_processor import get_stream_processor
    p = get_stream_processor()
    if p is None:
        return {"status": "not_initialized", "running": False}

    return {
        "status": "initialized",
        "running": (p._plate_thread is not None and p._plate_thread.is_alive()),
        "snow_buffer_size": len(p._snow_frame_buffer),
        "use_ffmpeg_direct": p.use_ffmpeg_direct,
    }


@app.post("/stream/start")
def stream_start() -> Dict[str, str]:
    from stream_processor import init_stream_processor, get_stream_processor
    p = get_stream_processor()
    if p is None:
        p = init_stream_processor(merger)
    p.start()
    return {"status": "started"}


@app.post("/stream/stop")
def stream_stop() -> Dict[str, str]:
    from stream_processor import get_stream_processor
    p = get_stream_processor()
    if p is None:
        return {"status": "not_initialized"}
    p.stop()
    return {"status": "stopped"}


# Deprecated endpoint (оставлен, чтобы не ломать интеграции)
@app.post("/anpr", summary="Deprecated")
async def recognize_plate_anpr(file: UploadFile = File(...)) -> JSONResponse:
    return JSONResponse(
        {"error": "Deprecated. Use RTSP stream processing.", "plate": None},
        status_code=200,
    )
