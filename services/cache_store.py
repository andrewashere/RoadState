import json
import time
from typing import Any, Optional
from .config import CACHE_DIR, RAW_CACHE_FILE, META_CACHE_FILE

def load_cache_if_fresh(ttl_seconds: int) -> Optional[list[dict[str, Any]]]:
    if not RAW_CACHE_FILE.exists() or not META_CACHE_FILE.exists():
        return None

    try:
        meta = json.loads(META_CACHE_FILE.read_text(encoding="utf-8"))
        fetched_at = float(meta.get("fetched_at", 0))
    except Exception:
        return None

    if (time.time() - fetched_at) > ttl_seconds:
        return None

    try:
        raw = json.loads(RAW_CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return None
        return raw
    except Exception:
        return None

def save_raw_cache(raw: list[dict[str, Any]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE_FILE.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    meta = {"fetched_at": time.time(), "count": len(raw)}
    META_CACHE_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")