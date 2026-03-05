from typing import Any

def normalize_ga_camera(cam: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a 511GA camera object into a canonical internal structure.
    store canonical form into SQLite.
    """
    cam_id = str(cam.get("Id"))
    lat = cam.get("Latitude")
    lon = cam.get("Longitude")

    lat_f = float(lat) if lat is not None else None
    lon_f = float(lon) if lon is not None else None

    views = cam.get("Views") or []
    view_id = None
    view_url = None
    view_status = None
    view_description = None

    if isinstance(views, list) and views:
        v0 = views[0] or {}
        view_id = v0.get("Id")
        view_url = v0.get("Url")
        view_status = v0.get("Status")
        view_description = v0.get("Description")

    return {
        "id": cam_id,
        "name": cam.get("Name"),
        "roadway": cam.get("Roadway"),
        "direction": cam.get("Direction"),
        "location": cam.get("Location"),
        "source": cam.get("Source"),
        "source_id": cam.get("SourceId"),
        "sort_order": cam.get("SortOrder"),
        "latitude": lat_f,
        "longitude": lon_f,
        "view_id": view_id,
        "view_url": view_url,
        "view_status": view_status,
        "view_description": view_description,
    }