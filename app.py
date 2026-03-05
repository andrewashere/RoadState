#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from openai import OpenAI

from services.config import CACHE_TTL_SECONDS_DEFAULT
from services.secrets import read_secret
from services.api import fetch_ga_cameras_raw
from services.cache_store import load_cache_if_fresh, save_raw_cache
from services.normalize import normalize_ga_camera
from services.db import (
    DB_PATH,  # DB path belongs with db module
    open_db,
    init_db,
    rebuild_index,
    sample,
    keyword_search,
    query_bbox,
    near_search,
    near_k,
)

from services.recorder import (
    RetryPolicy,
    record_many_with_results_parallel,
    write_run_report,
    utc_stamp,
    utc_now_iso,
)

from services.route_points import route_points_every_x_meters


# ----------------------------
# Secrets helpers
# ----------------------------
def ga_511_key() -> str:
    return read_secret("511ga_api_key")


def routes_key() -> str:
    return read_secret("google_routes_api_key")


# ----------------------------
# Ensure data ready: cache -> API -> normalize -> index
# ----------------------------
def ensure_data_ready(force_refresh: bool, ttl_seconds: int) -> tuple[str, int]:
    """
    Ensures JSON cache + SQLite index exist.
    Returns (source, raw_count) where source is "cache" or "api".
    """
    if not force_refresh:
        raw = load_cache_if_fresh(ttl_seconds)
        if raw is not None:
            if not DB_PATH.exists():
                cameras = [normalize_ga_camera(c) for c in raw]
                con = open_db()
                try:
                    init_db(con)
                    rebuild_index(con, cameras)
                finally:
                    con.close()
            return "cache", len(raw)

    key = ga_511_key()
    raw = fetch_ga_cameras_raw(key, fmt="json")
    save_raw_cache(raw)

    cameras = [normalize_ga_camera(c) for c in raw]
    con = open_db()
    try:
        init_db(con)
        rebuild_index(con, cameras)
    finally:
        con.close()

    return "api", len(raw)


# ----------------------------
# Pretty JSON helpers (NEW)
# ----------------------------
def try_parse_json(text: str | None):
    """
    Attempt to parse a JSON string into a Python object.
    Returns parsed object on success, else None.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


# ----------------------------
# Small printing helpers
# ----------------------------
def print_cam_line(cam: dict, *, show_urls: bool = False, prefix: str = "- ") -> None:
    print(
        f"{prefix}{cam['id']} | {cam.get('name')} | {cam.get('roadway')} | {cam.get('direction')} | "
        f"{cam.get('view_status')} | ({float(cam['latitude']):.6f}, {float(cam['longitude']):.6f}) | "
        f"view_id={cam.get('view_id')}"
    )
    if show_urls:
        print(f"    ViewUrl: {cam.get('view_url')}")


# ----------------------------
# Prompt + file:// helpers for inference
# ----------------------------
def file_url(path: str) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {p}")
    return f"file://{p.as_posix()}"


def load_prompt(prompt: str | None, prompt_yaml: str | None) -> str:
    if prompt:
        return prompt.strip()
    if prompt_yaml:
        data = yaml.safe_load(Path(prompt_yaml).read_text())
        if isinstance(data, dict):
            for k in ("user_prompt", "prompt", "text"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        if isinstance(data, str) and data.strip():
            return data.strip()
        raise ValueError(f"Unrecognized YAML format: {prompt_yaml}")
    raise ValueError("Provide --prompt or --prompt-yaml")


def infer_videos_from_results(
    *,
    vllm_server: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    fps: float | None,
    record_results: list,
    infer_out_path: Path,
    infer_workers: int = 8,
) -> None:
    """
    Runs vLLM inference concurrently on ok outputs.
    Writes a JSON inference report.

    IDE-friendly output change:
      - inference_raw: original model output (string)
      - inference_json: parsed JSON object if valid JSON, else null
      - inference_parse_error: short error if parsing failed (optional)
    """
    extra_body = {}
    if fps is not None:
        extra_body["mm_processor_kwargs"] = {"fps": fps, "do_sample_frames": True}

    jobs = []
    for r in record_results:
        if getattr(r, "status", None) != "ok":
            continue
        out_file = getattr(r, "out_file", None)
        if not out_file:
            continue
        jobs.append(
            {
                "image_id": getattr(r, "image_id", None),
                "out_file": out_file,
            }
        )

    def worker(job: dict) -> dict:
        # One client per worker to avoid any thread-safety issues.
        client = OpenAI(api_key="EMPTY", base_url=vllm_server)
        video = file_url(job["out_file"])

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "video_url", "video_url": {"url": video}},
                    ],
                }
            ],
            max_completion_tokens=max_tokens,
            extra_body=extra_body if extra_body else None,
        )

        content = None
        if getattr(resp, "choices", None):
            content = resp.choices[0].message.content

        parsed = try_parse_json(content) if isinstance(content, str) else None
        parse_error = None
        if isinstance(content, str) and content.strip() and parsed is None:
            parse_error = "Model output was not valid JSON."

        return {
            "image_id": job["image_id"],
            "out_file": job["out_file"],
            "inference_raw": content,        # original text
            "inference_json": parsed,        # dict/list if valid JSON else None
            "inference_parse_error": parse_error,
            "error": None,
        }

    results_out: list[dict] = []

    if not jobs:
        payload = {
            "vllm_server": vllm_server,
            "model": model,
            "max_tokens": max_tokens,
            "fps": fps,
            "infer_workers": infer_workers,
            "count": 0,
            "results": [],
        }
        infer_out_path.parent.mkdir(parents=True, exist_ok=True)
        infer_out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFER REPORT] wrote: {infer_out_path}")
        return

    with ThreadPoolExecutor(max_workers=infer_workers) as ex:
        futs = {ex.submit(worker, job): job for job in jobs}
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                out = fut.result()
                print(f"[INFER OK] image_id={job['image_id']} file={Path(job['out_file']).name}")
                results_out.append(out)
            except Exception as e:
                print(f"[INFER FAIL] image_id={job['image_id']} file={Path(job['out_file']).name}: {e}")
                results_out.append(
                    {
                        "image_id": job["image_id"],
                        "out_file": job["out_file"],
                        "inference_raw": None,
                        "inference_json": None,
                        "inference_parse_error": None,
                        "error": str(e),
                    }
                )

    # Stable order for downstream consumption
    results_out.sort(key=lambda x: (x["image_id"] if x["image_id"] is not None else 10**12, x["out_file"]))

    payload = {
        "vllm_server": vllm_server,
        "model": model,
        "max_tokens": max_tokens,
        "fps": fps,
        "infer_workers": infer_workers,
        "count": len(results_out),
        "results": results_out,
    }
    infer_out_path.parent.mkdir(parents=True, exist_ok=True)
    infer_out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFER REPORT] wrote: {infer_out_path}")


# ----------------------------
# Helper: record image_ids list
# ----------------------------
def record_image_ids_parallel(
    *,
    image_ids: list[int],
    seconds: int,
    out_dir: Path,
    out_template: str,
    verbose: bool,
    policy: RetryPolicy,
    report_path: Path,
    run_meta: dict,
    targets: list[dict],
    workers: int = 5,
) -> list:
    """
    Records image_ids in parallel and writes a run report.
    Returns the RecordResult list.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    results = record_many_with_results_parallel(
        image_ids=image_ids,
        seconds=seconds,
        out_dir=out_dir,
        out_template=out_template,
        verbose=verbose,
        policy=policy,
        max_workers=workers,
    )

    run_meta = dict(run_meta)
    run_meta["counts"] = {
        "requested": len(image_ids),
        "ok": sum(1 for r in results if r.status == "ok"),
        "skip": sum(1 for r in results if r.status == "skip"),
        "fail": sum(1 for r in results if r.status == "fail"),
    }

    write_run_report(report_path=report_path, run_meta=run_meta, targets=targets, results=results)
    print(f"[REPORT] wrote: {report_path}")
    print(f"[SUMMARY] ok={run_meta['counts']['ok']} skip={run_meta['counts']['skip']} fail={run_meta['counts']['fail']}")
    return results


# ----------------------------
# CLI commands
# ----------------------------
def cmd_sample(args: argparse.Namespace) -> None:
    source, raw_count = ensure_data_ready(force_refresh=args.refresh, ttl_seconds=args.ttl)
    print(f"Data ready: {raw_count} cameras (source: {source})\n")

    con = open_db()
    try:
        rows = sample(con, args.n)
    finally:
        con.close()

    for c in rows:
        print_cam_line(c, show_urls=args.show_urls)


def cmd_search(args: argparse.Namespace) -> None:
    source, _ = ensure_data_ready(force_refresh=args.refresh, ttl_seconds=args.ttl)
    print(f"Searching (source: {source})\n")

    con = open_db()
    try:
        results = keyword_search(con, args.q, args.limit)
    finally:
        con.close()

    if not results:
        print("No matches.")
        return

    for c in results:
        print_cam_line(c, show_urls=args.show_urls)


def cmd_bbox(args: argparse.Namespace) -> None:
    source, _ = ensure_data_ready(force_refresh=args.refresh, ttl_seconds=args.ttl)
    print(f"Bounding box query (source: {source})\n")

    con = open_db()
    try:
        results = query_bbox(con, args.min_lat, args.max_lat, args.min_lon, args.max_lon)
    finally:
        con.close()

    if args.enabled_only:
        results = [c for c in results if (c.get("view_status") or "").lower() == "enabled"]

    print(f"Found {len(results)} cameras.\n")
    for cam in results[: args.limit]:
        print_cam_line(cam, show_urls=args.show_urls)


def cmd_near(args: argparse.Namespace) -> None:
    source, _ = ensure_data_ready(force_refresh=args.refresh, ttl_seconds=args.ttl)
    print(f"Near query (source: {source})\n")

    con = open_db()
    try:
        results = near_search(con, args.lat, args.lon, args.radius_km)
    finally:
        con.close()

    if args.enabled_only:
        results = [c for c in results if (c.get("view_status") or "").lower() == "enabled"]

    print(f"Found {len(results)} cameras within {args.radius_km} km.\n")
    for cam in results[: args.limit]:
        km = float(cam["distance_m"]) / 1000.0
        print(
            f"- {cam['id']} | {cam.get('name')} | {cam.get('roadway')} | {cam.get('direction')} | "
            f"{cam.get('view_status')} | {km:.3f} km | "
            f"({float(cam['latitude']):.6f}, {float(cam['longitude']):.6f}) | view_id={cam.get('view_id')}"
        )
        if args.show_urls:
            print(f"    ViewUrl: {cam.get('view_url')}")


def cmd_route_cameras(args: argparse.Namespace) -> None:
    source, _ = ensure_data_ready(force_refresh=args.refresh, ttl_seconds=args.ttl)

    origin = (args.origin_lat, args.origin_lon)
    dest = (args.dest_lat, args.dest_lon)

    points, encoded = route_points_every_x_meters(
        api_key=routes_key(),
        origin=origin,
        destination=dest,
        step_m=args.step_m,
        travel_mode=args.travel_mode,
        routing_preference=args.routing_preference,
        timeout_s=args.timeout,
    )

    con = open_db()
    try:
        results = []
        seen_camera_ids: set[int] = set()

        for i, (lat, lon) in enumerate(points):
            if args.mode == "k":
                cams = near_k(
                    con,
                    lat=lat,
                    lon=lon,
                    k=args.k,
                    start_radius_km=args.start_radius_km,
                    max_radius_km=args.max_radius_km,
                    enabled_only=args.enabled_only,
                )
            else:
                cams = near_search(con, lat, lon, args.radius_km)
                if args.enabled_only:
                    cams = [c for c in cams if (c.get("view_status") or "").lower() == "enabled"]
                cams = cams[: args.limit_per_point]

            if args.dedupe_global:
                unique = []
                for c in cams:
                    cid = int(c.get("id"))
                    if cid in seen_camera_ids:
                        continue
                    seen_camera_ids.add(cid)
                    unique.append(c)
                cams = unique

            out_cams = []
            for c in cams:
                out_cams.append(
                    {
                        "camera_id": c.get("id"),
                        "view_id": c.get("view_id"),
                        "name": c.get("name"),
                        "roadway": c.get("roadway"),
                        "direction": c.get("direction"),
                        "location": c.get("location"),
                        "view_status": c.get("view_status"),
                        "latitude": c.get("latitude"),
                        "longitude": c.get("longitude"),
                        "distance_m": c.get("distance_m"),
                        "view_url": c.get("view_url"),
                    }
                )

            results.append({"idx": i, "point": {"lat": lat, "lon": lon}, "cameras": out_cams})
    finally:
        con.close()

    out = {
        "source": source,
        "origin": {"lat": origin[0], "lon": origin[1]},
        "destination": {"lat": dest[0], "lon": dest[1]},
        "step_m": args.step_m,
        "points_count": len(points),
        "encoded_polyline": encoded,
        "results": results,
    }
    print(json.dumps(out, indent=2))


def cmd_route_record_infer(args: argparse.Namespace) -> None:
    """
    1) route -> points
    2) select 1 nearest camera per point (dedupe by view_id)
    3) record in parallel (5)
    4) infer on each successful recording via local vLLM (parallel infer-workers)
    """
    source, _ = ensure_data_ready(force_refresh=args.refresh, ttl_seconds=args.ttl)
    print(f"Route-record-infer (camera source: {source})\n")

    prompt_text = load_prompt(args.prompt, args.prompt_yaml)

    origin = (args.origin_lat, args.origin_lon)
    dest = (args.dest_lat, args.dest_lon)

    points, encoded = route_points_every_x_meters(
        api_key=routes_key(),
        origin=origin,
        destination=dest,
        step_m=args.step_m,
        travel_mode=args.travel_mode,
        routing_preference=args.routing_preference,
        timeout_s=args.timeout,
    )

    print(f"Route points: {len(points)} (step_m={args.step_m})")

    # Select 1 nearest per point + dedupe by view_id
    con = open_db()
    try:
        selected_targets = []
        view_ids: list[int] = []
        seen_view_ids: set[int] = set()

        for (lat, lon) in points:
            cams = near_k(
                con,
                lat=lat,
                lon=lon,
                k=1,
                start_radius_km=args.start_radius_km,
                max_radius_km=args.max_radius_km,
                enabled_only=args.enabled_only,
            )
            for cam in cams:
                vid = cam.get("view_id")
                if vid is None:
                    continue
                vid_int = int(vid)
                if vid_int in seen_view_ids:
                    continue
                seen_view_ids.add(vid_int)
                view_ids.append(vid_int)
                selected_targets.append(cam)
                break
            if args.max_total > 0 and len(view_ids) >= args.max_total:
                break
    finally:
        con.close()

    if not view_ids:
        print("No recordable cameras found along the route (no cameras with view_id).")
        return

    print(f"\nSelected record targets (unique view_ids): {len(view_ids)}")
    if args.preview:
        for i, vid in enumerate(view_ids[: args.preview_limit], 1):
            print(f"  {i}. view_id={vid}")
        if len(view_ids) > args.preview_limit:
            print(f"  ... +{len(view_ids) - args.preview_limit} more")
        print()

    policy = RetryPolicy(
        token_retries=args.token_retries,
        offline_retries=args.offline_retries,
        record_retries=args.record_retries,
        backoff_s=args.backoff,
    )

    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir) if args.report_dir else out_dir
    report_name = args.report_name or f"route_run_{utc_stamp()}.json"
    report_path = report_dir / report_name

    report_targets = [
        {
            "camera_id": cam.get("id"),
            "view_id": cam.get("view_id"),
            "name": cam.get("name"),
            "roadway": cam.get("roadway"),
            "direction": cam.get("direction"),
            "location": cam.get("location"),
            "view_status": cam.get("view_status"),
            "latitude": cam.get("latitude"),
            "longitude": cam.get("longitude"),
            "distance_m": cam.get("distance_m"),
            "view_url": cam.get("view_url"),
        }
        for cam in selected_targets
    ]

    run_meta = {
        "started_utc": utc_now_iso(),
        "ended_utc": None,
        "cmd": "route-record-infer",
        "camera_source": source,
        "origin": {"lat": origin[0], "lon": origin[1]},
        "destination": {"lat": dest[0], "lon": dest[1]},
        "route": {
            "step_m": args.step_m,
            "points_count": len(points),
            "travel_mode": args.travel_mode,
            "routing_preference": args.routing_preference,
            "encoded_polyline": encoded if args.include_polyline_in_report else None,
        },
        "selection": {
            "nearest_per_point": 1,
            "enabled_only": args.enabled_only,
            "start_radius_km": args.start_radius_km,
            "max_radius_km": args.max_radius_km,
            "max_total": args.max_total,
            "dedupe": "view_id (always)",
        },
        "recording": {
            "seconds": args.seconds,
            "out_dir": str(out_dir),
            "out_template": args.out_template,
            "parallel_workers": 5,
            "token_retries": args.token_retries,
            "offline_retries": args.offline_retries,
            "record_retries": args.record_retries,
            "backoff": args.backoff,
        },
        "inference": {
            "vllm_server": args.vllm_server,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "fps": args.fps,
            "infer_workers": args.infer_workers,
            "prompt_yaml": args.prompt_yaml,
        },
    }

    # Record
    record_results = record_image_ids_parallel(
        image_ids=view_ids,
        seconds=args.seconds,
        out_dir=out_dir,
        out_template=args.out_template,
        verbose=args.verbose,
        policy=policy,
        report_path=report_path,
        run_meta=run_meta,
        targets=report_targets,
        workers=5,
    )

    # Infer (parallel)
    infer_out = report_path.with_name(report_path.stem + "_infer.json")
    infer_videos_from_results(
        vllm_server=args.vllm_server,
        model=args.model,
        prompt_text=prompt_text,
        max_tokens=args.max_tokens,
        fps=args.fps,
        record_results=record_results,
        infer_out_path=infer_out,
        infer_workers=args.infer_workers,
    )


# ----------------------------
# CLI parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="511GA Cameras + Recorder + Route Inference")
    p.add_argument("--ttl", type=int, default=CACHE_TTL_SECONDS_DEFAULT, help="cache TTL seconds")

    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("sample", help="Show a few cameras (from DB)")
    ps.add_argument("-n", type=int, default=10)
    ps.add_argument("--show-urls", action="store_true")
    ps.add_argument("--refresh", action="store_true")
    ps.set_defaults(func=cmd_sample)

    psearch = sub.add_parser("search", help="Keyword search (name/roadway/direction/location)")
    psearch.add_argument("--q", required=True)
    psearch.add_argument("--limit", type=int, default=25)
    psearch.add_argument("--show-urls", action="store_true")
    psearch.add_argument("--refresh", action="store_true")
    psearch.set_defaults(func=cmd_search)

    pb = sub.add_parser("bbox", help="Bounding box search by lat/lon")
    pb.add_argument("--min-lat", type=float, required=True)
    pb.add_argument("--max-lat", type=float, required=True)
    pb.add_argument("--min-lon", type=float, required=True)
    pb.add_argument("--max-lon", type=float, required=True)
    pb.add_argument("--limit", type=int, default=50)
    pb.add_argument("--enabled-only", action="store_true")
    pb.add_argument("--show-urls", action="store_true")
    pb.add_argument("--refresh", action="store_true")
    pb.set_defaults(func=cmd_bbox)

    pn = sub.add_parser("near", help="Find cameras within radius-km of a point")
    pn.add_argument("--lat", type=float, required=True)
    pn.add_argument("--lon", type=float, required=True)
    pn.add_argument("--radius-km", type=float, required=True)
    pn.add_argument("--limit", type=int, default=25)
    pn.add_argument("--enabled-only", action="store_true")
    pn.add_argument("--show-urls", action="store_true")
    pn.add_argument("--refresh", action="store_true")
    pn.set_defaults(func=cmd_near)

    prc = sub.add_parser("route-cameras", help="Route origin->dest, sample points, find cameras near each point")
    prc.add_argument("--origin-lat", type=float, required=True)
    prc.add_argument("--origin-lon", type=float, required=True)
    prc.add_argument("--dest-lat", type=float, required=True)
    prc.add_argument("--dest-lon", type=float, required=True)
    prc.add_argument("--step-m", type=float, default=100.0)
    prc.add_argument("--travel-mode", type=str, default="DRIVE")
    prc.add_argument("--routing-preference", type=str, default="TRAFFIC_AWARE")
    prc.add_argument("--timeout", type=int, default=30)
    prc.add_argument("--enabled-only", action="store_true")
    prc.add_argument("--refresh", action="store_true")
    prc.add_argument("--mode", choices=["k", "radius"], default="k")
    prc.add_argument("--k", type=int, default=2)
    prc.add_argument("--radius-km", type=float, default=1.0)
    prc.add_argument("--limit-per-point", type=int, default=10)
    prc.add_argument("--start-radius-km", type=float, default=2.0)
    prc.add_argument("--max-radius-km", type=float, default=200.0)
    prc.add_argument("--dedupe-global", action="store_true")
    prc.set_defaults(func=cmd_route_cameras)

    # record + infer
    pri = sub.add_parser("route-record-infer", help="Route -> record (parallel 5) -> infer (parallel)")
    pri.add_argument("--origin-lat", type=float, required=True)
    pri.add_argument("--origin-lon", type=float, required=True)
    pri.add_argument("--dest-lat", type=float, required=True)
    pri.add_argument("--dest-lon", type=float, required=True)

    pri.add_argument("--step-m", type=float, default=250.0)
    pri.add_argument("--travel-mode", type=str, default="DRIVE")
    pri.add_argument("--routing-preference", type=str, default="TRAFFIC_AWARE")
    pri.add_argument("--timeout", type=int, default=30)

    pri.add_argument("--refresh", action="store_true")
    pri.add_argument("--enabled-only", action="store_true")
    pri.add_argument("--start-radius-km", type=float, default=2.0)
    pri.add_argument("--max-radius-km", type=float, default=200.0)
    pri.add_argument("--max-total", type=int, default=0)

    pri.add_argument("--seconds", type=int, default=30)
    pri.add_argument("--out-dir", type=str, default="recordings_route")
    pri.add_argument("--out-template", type=str, default="img_{image_id}_{stamp}_{seconds}s.mp4")
    pri.add_argument("--verbose", action="store_true")
    pri.add_argument("--token-retries", type=int, default=2)
    pri.add_argument("--offline-retries", type=int, default=2)
    pri.add_argument("--record-retries", type=int, default=1)
    pri.add_argument("--backoff", type=float, default=1.5)

    pri.add_argument("--report-dir", type=str, default="")
    pri.add_argument("--report-name", type=str, default="")
    pri.add_argument("--preview", action="store_true")
    pri.add_argument("--preview-limit", type=int, default=25)
    pri.add_argument("--include-polyline-in-report", action="store_true")

    # inference args
    pri.add_argument("--vllm-server", default="http://127.0.0.1:8000/v1")
    pri.add_argument("--model", default="nvidia/Cosmos-Reason2-8B")
    pri.add_argument("--prompt", help="Inline prompt text.")
    pri.add_argument("--prompt-yaml", help="YAML with user_prompt/prompt/text.")
    pri.add_argument("--fps", type=float, default=None)
    pri.add_argument("--max-tokens", type=int, default=700)
    pri.add_argument("--infer-workers", type=int, default=8, help="Concurrent vLLM requests (default: 8)")

    pri.set_defaults(func=cmd_route_record_infer)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()