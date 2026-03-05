from __future__ import annotations

import math
import sqlite3
from typing import Any

from .config import CACHE_DIR, DB_PATH

SCHEMA_VERSION = 2

def open_db() -> sqlite3.Connection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY,
            name TEXT,
            roadway TEXT,
            direction TEXT,
            location TEXT,
            source TEXT,
            source_id TEXT,
            sort_order INTEGER,

            latitude REAL,
            longitude REAL,

            view_id INTEGER,
            view_url TEXT,
            view_status TEXT,
            view_description TEXT
        )
        """
    )

    con.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS cameras_rtree
        USING rtree(
            id,
            min_lat, max_lat,
            min_lon, max_lon
        )
        """
    )

def rebuild_index(con: sqlite3.Connection, cameras: list[dict[str, Any]]) -> None:
    """
    Rebuild cameras table + RTree from canonical dicts.
    """
    con.execute("BEGIN;")
    try:
        con.execute("DELETE FROM cameras;")
        con.execute("DELETE FROM cameras_rtree;")

        for cam in cameras:
            cam_id = cam.get("id")
            lat = cam.get("latitude")
            lon = cam.get("longitude")

            if cam_id is None or lat is None or lon is None:
                continue

            try:
                cam_id_s = str(cam_id)
                lat_f = float(lat)
                lon_f = float(lon)
            except (TypeError, ValueError):
                continue

            con.execute(
                """
                INSERT INTO cameras (
                    id, name, roadway, direction, location, source, source_id, sort_order,
                    latitude, longitude,
                    view_id, view_url, view_status, view_description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cam_id_s,
                    cam.get("name"),
                    cam.get("roadway"),
                    cam.get("direction"),
                    cam.get("location"),
                    cam.get("source"),
                    cam.get("source_id"),
                    cam.get("sort_order"),
                    lat_f,
                    lon_f,
                    cam.get("view_id"),
                    cam.get("view_url"),
                    cam.get("view_status"),
                    cam.get("view_description"),
                ),
            )

            con.execute(
                """
                INSERT INTO cameras_rtree (id, min_lat, max_lat, min_lon, max_lon)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cam_id_s, lat_f, lat_f, lon_f, lon_f),
            )

        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise

def query_bbox(
    con: sqlite3.Connection, min_lat: float, max_lat: float, min_lon: float, max_lon: float
) -> list[dict[str, Any]]:
    cur = con.execute(
        """
        SELECT c.*
        FROM cameras c
        JOIN cameras_rtree r ON r.id = c.id
        WHERE r.min_lat >= ? AND r.max_lat <= ?
          AND r.min_lon >= ? AND r.max_lon <= ?
        """,
        (min_lat, max_lat, min_lon, max_lon),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]

def keyword_search(con: sqlite3.Connection, q: str, limit: int) -> list[dict[str, Any]]:
    like = f"%{q}%"
    cur = con.execute(
        """
        SELECT id, name, roadway, direction, location, view_status, latitude, longitude, view_url, view_id
        FROM cameras
        WHERE name LIKE ? OR roadway LIKE ? OR direction LIKE ? OR location LIKE ?
        ORDER BY roadway, name
        LIMIT ?
        """,
        (like, like, like, like, limit),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]

def sample(con: sqlite3.Connection, n: int) -> list[dict[str, Any]]:
    cur = con.execute(
        """
        SELECT id, name, roadway, direction, view_status, latitude, longitude, view_url, view_id
        FROM cameras
        ORDER BY roadway, name
        LIMIT ?
        """,
        (n,),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def near_search(con: sqlite3.Connection, lat: float, lon: float, radius_km: float) -> list[dict[str, Any]]:
    """
    Returns cameras within radius_km, sorted by distance.
    """
    radius_m = radius_km * 1000.0

    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * max(0.1, math.cos(math.radians(lat))))

    candidates = query_bbox(
        con,
        lat - lat_delta, lat + lat_delta,
        lon - lon_delta, lon + lon_delta,
    )

    results = []
    for cam in candidates:
        if cam.get("latitude") is None or cam.get("longitude") is None:
            continue
        d = haversine_m(lat, lon, float(cam["latitude"]), float(cam["longitude"]))
        if d <= radius_m:
            cam2 = dict(cam)
            cam2["distance_m"] = d
            results.append(cam2)

    results.sort(key=lambda x: x["distance_m"])
    return results

def near_k(
    con: sqlite3.Connection,
    lat: float,
    lon: float,
    k: int = 5,
    start_radius_km: float = 2.0,
    max_radius_km: float = 200.0,
    enabled_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Returns the k closest cameras by true Haversine distance.
    Expands radius until enough candidates exist (or max radius reached).
    """
    radius = start_radius_km
    best: list[dict[str, Any]] = []

    while radius <= max_radius_km:
        candidates = near_search(con, lat, lon, radius_km=radius)

        if enabled_only:
            candidates = [c for c in candidates if (c.get("view_status") or "").lower() == "enabled"]

        if len(candidates) >= k:
            best = candidates[:k]
            break

        best = candidates
        radius *= 2.0

    return best[:k]

def get_schema_version(con: sqlite3.Connection) -> int:
    con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    row = con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    return int(row[0]) if row else 0

def set_schema_version(con: sqlite3.Connection, version: int) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    con.execute(
        "INSERT INTO meta(key,value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(version),),
    )

def reset_db(con: sqlite3.Connection) -> None:
    # Drop in dependency order: rtree first, then table
    con.execute("DROP TABLE IF EXISTS cameras_rtree;")
    con.execute("DROP TABLE IF EXISTS cameras;")
    con.execute("DROP TABLE IF EXISTS meta;")