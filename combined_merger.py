from __future__ import annotations

import io
import os
import time
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from google import genai
from PIL import Image

LOCAL_TZ = timezone(timedelta(hours=int(os.getenv("LOCAL_TZ_OFFSET_HOURS", "5"))))


def _now() -> datetime:
    return datetime.now(tz=LOCAL_TZ)


class EventMerger:
    """
    Используем в RTSP режиме только:
      - analyze_with_gemini(...)
    """
    def __init__(self, upstream_url: str):
        self.upstream_url = upstream_url

        self._gemini_client: genai.Client | None = None
        self._gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self._gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        # dedup cache (photos hash -> (result, ts))
        self._gemini_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._gemini_cache_ttl = float(os.getenv("GEMINI_CACHE_TTL_SECONDS", "300.0"))

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            if not self._gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            self._gemini_client = genai.Client(api_key=self._gemini_api_key)
        return self._gemini_client

    @staticmethod
    def _photos_hash(snow_photo: bytes, plate_photo_1: bytes, plate_photo_2: bytes | None) -> str:
        h = hashlib.md5()
        h.update(snow_photo)
        h.update(plate_photo_1)
        if plate_photo_2:
            h.update(plate_photo_2)
        return h.hexdigest()

    async def analyze_with_gemini(
        self,
        snow_photo: bytes,
        plate_photo_1: bytes,
        plate_photo_2: bytes | None,
        camera_plate: str | None = None,
    ) -> Dict[str, Any]:
        """
        Return:
        {
          "snow_percentage": 0..100,
          "snow_confidence": 0..1,
          "plate": str | None,
          "plate_confidence": 0..1
        }
        """
        ph = self._photos_hash(snow_photo, plate_photo_1, plate_photo_2)
        now_ts = time.time()

        # purge cache
        for k, (_res, ts) in list(self._gemini_cache.items()):
            if now_ts - ts > self._gemini_cache_ttl:
                del self._gemini_cache[k]

        if ph in self._gemini_cache:
            cached, _ = self._gemini_cache[ph]
            print(f"[GEMINI] Using cached result for hash={ph[:8]}...")
            return dict(cached)

        if not self._gemini_api_key:
            result = {
                "error": "GEMINI_API_KEY is not set",
                "snow_percentage": 0.0,
                "snow_confidence": 0.0,
                "plate": None,
                "plate_confidence": 0.0,
            }
            self._gemini_cache[ph] = (dict(result), now_ts)
            return result

        snow_img = Image.open(io.BytesIO(snow_photo)).convert("RGB")
        plate_img1 = Image.open(io.BytesIO(plate_photo_1)).convert("RGB")
        images = [snow_img, plate_img1]
        if plate_photo_2:
            plate_img2 = Image.open(io.BytesIO(plate_photo_2)).convert("RGB")
            images.append(plate_img2)

        image3_text = "IMAGE 3: License plate photo 2 (optional).\n" if plate_photo_2 else ""
        camera_hint = ""
        if camera_plate and str(camera_plate).strip().lower() not in ["", "none", "unknown", "null"]:
            camera_hint = (
                f"\nIMPORTANT: camera suggested plate '{camera_plate}', but it may be wrong. "
                "Use only as a hint, verify from the images.\n"
            )

        prompt = (
            "You are analyzing truck photos for snow volume and license plate recognition.\n\n"
            "IMAGE 1: Snow photo - shows cargo bed.\n"
            "IMAGE 2: Plate photo 1 - normal/wide view.\n"
            + image3_text +
            camera_hint +
            "\n"
            "CRITICAL: Analyze ONLY the truck that is CLOSEST to the camera.\n\n"
            "TASKS:\n"
            "1) Snow: estimate how full the OPEN cargo bed is with loose/bulk snow (0-100). "
            "Exclude glare/reflections, frost/ice, background, closed beds.\n"
            "If bed not visible/closed => snow_percentage=0 and snow_confidence=0.\n\n"
            "2) Plate: read plate ONLY from the closest truck.\n"
            "Kazakhstan strict formats:\n"
            " - 111AAA11 (3 digits + 3 letters + 2 digits)\n"
            " - 111AA11  (3 digits + 2 letters + 2 digits)\n"
            "Region codes 01-18 only.\n"
            "Return plate WITHOUT spaces.\n"
            "If not sure => plate=null.\n\n"
            "Return JSON only:\n"
            "{\n"
            '  "snow_percentage": 42.5,\n'
            '  "snow_confidence": 0.9,\n'
            '  "plate": "035AL115",\n'
            '  "plate_confidence": 0.85\n'
            "}\n"
        )

        client = self._get_gemini_client()
        resp = client.models.generate_content(
            model=self._gemini_model,
            contents=images + [prompt],
        )

        text = (resp.text or "").strip()
        if not text:
            result = {
                "error": "Empty response from Gemini",
                "snow_percentage": 0.0,
                "snow_confidence": 0.0,
                "plate": None,
                "plate_confidence": 0.0,
            }
            self._gemini_cache[ph] = (dict(result), now_ts)
            return result

        # strip ```json ... ```
        if text.startswith("```"):
            t = text.strip("`").strip()
            if t.lower().startswith("json"):
                t = t[4:].strip()
            text = t

        try:
            data = json.loads(text)
        except Exception as e:
            result = {
                "error": f"JSON parse error: {e}",
                "raw": (resp.text or "")[:400],
                "snow_percentage": 0.0,
                "snow_confidence": 0.0,
                "plate": None,
                "plate_confidence": 0.0,
            }
            self._gemini_cache[ph] = (dict(result), now_ts)
            return result

        def clamp01(x: Any) -> float:
            try:
                v = float(x)
                return max(0.0, min(1.0, v))
            except Exception:
                return 0.0

        def clamp100(x: Any) -> float:
            try:
                v = float(x)
                if 0.0 <= v <= 1.0:
                    v *= 100.0
                return max(0.0, min(100.0, round(v, 2)))
            except Exception:
                return 0.0

        snow_percentage = clamp100(data.get("snow_percentage", 0.0))
        snow_confidence = clamp01(data.get("snow_confidence", 0.0))
        plate_conf = clamp01(data.get("plate_confidence", 0.0))

        plate = data.get("plate")
        if plate is not None:
            plate = str(plate).strip().upper().replace(" ", "")
            if plate in ["", "NONE", "NULL"]:
                plate = None

        result = {
            "snow_percentage": snow_percentage,
            "snow_confidence": snow_confidence,
            "plate": plate,
            "plate_confidence": plate_conf,
        }

        self._gemini_cache[ph] = (dict(result), now_ts)
        return result


_merger_instance: EventMerger | None = None


def init_merger(upstream_url: str) -> EventMerger:
    global _merger_instance
    if _merger_instance is None:
        _merger_instance = EventMerger(upstream_url=upstream_url)
    return _merger_instance
