# services/route_points.py
from __future__ import annotations

import math
from typing import List, Tuple

import requests

LatLon = Tuple[float, float]
R_EARTH_M = 6_371_000.0


def decode_polyline(encoded: str) -> List[LatLon]:
    coords: List[LatLon] = []
    index = 0
    lat = 0
    lng = 0

    while index < len(encoded):
        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coords.append((lat * 1e-5, lng * 1e-5))

    return coords


def haversine_m(a: LatLon, b: LatLon) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R_EARTH_M * math.asin(math.sqrt(h))


def lerp_latlon(a: LatLon, b: LatLon, t: float) -> LatLon:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def resample_polyline(points: List[LatLon], step_m: float) -> List[LatLon]:
    if not points:
        return []
    out: List[LatLon] = [points[0]]
    carry = 0.0
    prev = points[0]

    for cur in points[1:]:
        seg_len = haversine_m(prev, cur)
        if seg_len <= 0:
            prev = cur
            continue

        dist_along = carry
        while dist_along + step_m <= seg_len:
            dist_along += step_m
            t = dist_along / seg_len
            out.append(lerp_latlon(prev, cur, t))

        carry = seg_len - dist_along
        prev = cur

    if out[-1] != points[-1]:
        out.append(points[-1])

    return out


def compute_routes_polyline_encoded(
    api_key: str,
    origin: LatLon,
    destination: LatLon,
    travel_mode: str = "DRIVE",
    routing_preference: str = "TRAFFIC_AWARE",
    timeout_s: int = 30,
) -> str:
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "routes.polyline.encodedPolyline",
    }
    body = {
        "origin": {"location": {"latLng": {"latitude": origin[0], "longitude": origin[1]}}},
        "destination": {"location": {"latLng": {"latitude": destination[0], "longitude": destination[1]}}},
        "travelMode": travel_mode,
        "routingPreference": routing_preference,
    }

    r = requests.post(url, headers=headers, json=body, timeout=timeout_s)
    if not r.ok:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Routes API error {r.status_code}: {err}")

    data = r.json()
    return data["routes"][0]["polyline"]["encodedPolyline"]


def route_points_every_x_meters(
    api_key: str,
    origin: LatLon,
    destination: LatLon,
    step_m: float,
    *,
    travel_mode: str = "DRIVE",
    routing_preference: str = "TRAFFIC_AWARE",
    timeout_s: int = 30,
) -> tuple[List[LatLon], str]:
    """
    Returns (points, encoded_polyline).
    Points are [[lat,lon] in order starting closest to origin], spaced every step_m,
    including final destination point.
    """
    encoded = compute_routes_polyline_encoded(
        api_key=api_key,
        origin=origin,
        destination=destination,
        travel_mode=travel_mode,
        routing_preference=routing_preference,
        timeout_s=timeout_s,
    )

    poly = decode_polyline(encoded)

    # Ensure poly starts closest to origin
    if poly and haversine_m(poly[0], origin) > haversine_m(poly[-1], origin):
        poly.reverse()

    sampled = resample_polyline(poly, step_m)
    return sampled, encoded