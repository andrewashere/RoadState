# services/geocode.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import requests


@dataclass(frozen=True)
class GeocodeResult:
    lat: float
    lon: float
    formatted_address: str
    place_id: str


def geocode_address(
    api_key: str,
    address: str,
    *,
    region: Optional[str] = None,          # e.g. "us"
    bounds: Optional[str] = None,          # "sw_lat,sw_lng|ne_lat,ne_lng"
    timeout_s: int = 20,
) -> GeocodeResult:
    """
    Geocode an address to (lat, lon) using Google Geocoding API.

    region: ccTLD region bias (e.g. "us")
    bounds: viewport bias "sw_lat,sw_lng|ne_lat,ne_lng"
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    if region:
        params["region"] = region
    if bounds:
        params["bounds"] = bounds

    r = requests.get(url, params=params, timeout=timeout_s)
    if not r.ok:
        raise RuntimeError(f"Geocoding HTTP error {r.status_code}: {r.text[:500]}")

    data = r.json()
    status = data.get("status")
    if status != "OK":
        # Common: ZERO_RESULTS, REQUEST_DENIED, OVER_QUERY_LIMIT, INVALID_REQUEST
        err = data.get("error_message")
        raise RuntimeError(f"Geocoding failed: status={status} error={err}")

    results = data.get("results") or []
    if not results:
        raise RuntimeError("Geocoding failed: empty results with status OK (unexpected)")

    top = results[0]
    loc = top["geometry"]["location"]
    return GeocodeResult(
        lat=float(loc["lat"]),
        lon=float(loc["lng"]),
        formatted_address=str(top.get("formatted_address", "")),
        place_id=str(top.get("place_id", "")),
    )