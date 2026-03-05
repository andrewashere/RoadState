from typing import Any
import requests
from .config import GA_CAMERAS_URL

def fetch_ga_cameras_raw(key: str, fmt: str = "json") -> list[dict[str, Any]]:
    """
    Fetch raw camera list from 511GA.

    GET https://511ga.org/api/v2/get/cameras?key={key}&format={json|xml}
    """
    params = {"key": key, "format": fmt}
    resp = requests.get(GA_CAMERAS_URL, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected response type from API: {type(data)}")
    return data