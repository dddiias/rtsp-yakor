from __future__ import annotations

import os
import datetime
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse

# Env loading:
# - Local dev: load from .env / app.env if present
# - Prod (Render): use Environment Variables only (no file lookup, no warnings)
def _is_prod_env() -> bool:
    # Render sets RENDER=true in runtime
    if os.getenv("RENDER", "").strip().lower() in ("1", "true", "yes"):
        return True
    env = os.getenv("ENV", "").strip().lower()
    return env in ("prod", "production")


def _try_load_dotenv() -> None:
    if _is_prod_env():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    base = os.path.dirname(__file__)
    for fname in (".env", "app.env"):
        p = os.path.join(base, fname)
        if os.path.exists(p):
            load_dotenv(p, override=False)
            print(f"[API] Loaded environment from: {p}")
            break


_try_load_dotenv()

from combined_merger import init_merger

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://snowops-anpr-service.onrender.com/api/v1/anpr/events")
ENABLE_STREAM_PROCESSOR = os.getenv("ENABLE_STREAM_PROCESSOR", "true").strip().lower() == "true"

merger = init_merger(upstream_url=UPSTREAM_URL)


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
    title="Hikvision RTSP -> LineCross -> Gemini -> Upstream",
    description="RTSP: детекция пересечения линии грузовиком + Gemini номер/снег + отправка upstream",
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


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/stream/status")
def stream_status() -> Dict[str, Any]:
    from stream_processor import get_stream_processor
    p = get_stream_processor()
    if p is None:
        return {"status": "not_initialized", "running": False}

    def _is_open(h) -> bool:
        if h is None:
            return False
        try:
            return h.isOpened()
        except Exception:
            return False

    return {
        "status": "initialized",
        "running": p.is_running(),
        "plate_connected": _is_open(p.plate_cap),
        "snow_connected": _is_open(p.snow_cap),
        "snow_buffer_size": p.snow_buffer_size(),
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
