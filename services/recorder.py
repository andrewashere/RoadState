from __future__ import annotations  # Postpone type-hint evaluation (lets you use | and forward refs cleanly).

import json  # Read/write the run report JSON to disk.
import time  # Sleep/backoff between retries and track simple timing.
import subprocess  # Invoke ffmpeg as an external process.
from dataclasses import dataclass, asdict  # Define lightweight data containers + convert them to dicts for JSON.
from datetime import datetime, timezone  # Generate UTC timestamps for filenames and reports.
from pathlib import Path  # OS-independent filesystem paths (output dirs/files).
from typing import Optional, List, Dict, Any  # Type hints for readability and editor support.
from urllib.parse import urljoin  # Safely build absolute URLs from relative playlist paths.

import requests  # HTTP client used for session cookies, CSRF, playlists, and segment probes.
from .config import BASE_SITE  # Base website URL (e.g., https://511ga.org) from your config.

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:147.0) Gecko/20100101 Firefox/147.0"  # Browser-like UA to match site expectations.
BASE = BASE_SITE  # Canonical base origin used in headers and URL building.

HOME_HEADERS = {"User-Agent": UA}  # Headers for initial homepage request to establish cookies.
XHR_HEADERS_BASE = {  # Headers for the site XHR endpoint that returns the HLS playlist URL.
    "User-Agent": UA,
    "Referer": f"{BASE}/",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "*/*",
}
HLS_HEADERS = {  # Headers used when fetching HLS playlists/segments (Origin/Referer matter for some servers).
    "User-Agent": UA,
    "Origin": BASE,
    "Referer": f"{BASE}/",
    "Accept": "*/*",
}

# -----------------------------
# Error taxonomy
# -----------------------------
class StreamError(Exception): pass         # Base exception for any stream/recording failure.
class CsrfOrSessionError(StreamError): pass # Session/CSRF cookie missing/expired or blocked by auth.
class GetVideoUrlError(StreamError): pass   # Failure calling /Camera/GetVideoUrl or parsing its result.
class PlaylistError(StreamError): pass      # Failure downloading/parsing M3U8 playlists (master/variant).
class OfflineError(StreamError): pass       # Stream appears offline/stalled (no TS segments or tiny segments).
class RecorderError(StreamError): pass      # ffmpeg failed or output validation failed.

# -----------------------------
# Retry policy
# -----------------------------
@dataclass
class RetryPolicy:  # Tunables controlling how many times to retry token/playlist/offline/ffmpeg failures.
    token_retries: int = 2       # Retries for CSRF/session/token issues (refresh session or reacquire token).
    offline_retries: int = 2     # Retries for offline/stalled stream indicators (no segments, tiny segments).
    record_retries: int = 1      # Retries for ffmpeg/write/validation failures.
    backoff_s: float = 1.5       # Sleep between retry attempts (simple fixed backoff).

def utc_stamp() -> str:
    # Filesystem-safe UTC timestamp (for output filenames).
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")

def utc_now_iso() -> str:
    # ISO-8601 UTC timestamp (for report metadata).
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# -----------------------------
# Structured result objects
# -----------------------------
@dataclass
class RecordResult:  # One per image_id, capturing success/skip/fail plus file/bytes/errors and timings.
    image_id: int
    status: str  # "ok" | "skip" | "fail"
    out_file: str | None
    bytes_written: int | None
    error_type: str | None
    error_message: str | None
    started_utc: str
    ended_utc: str

def make_session() -> requests.Session:
    # Creates a persistent HTTP session and primes it by loading the homepage (sets cookies/CSRF).
    s = requests.Session()
    r = s.get(f"{BASE}/", headers=HOME_HEADERS, timeout=20)
    r.raise_for_status()
    return s

def refresh_session(s: Optional[requests.Session]) -> requests.Session:
    # Discards the current session (closing sockets) and creates a fresh one (new cookies/CSRF).
    try:
        if s is not None:
            s.close()
    except Exception:
        pass
    return make_session()

def get_csrf_cookie(s: requests.Session) -> str:
    # Pulls the CSRF cookie that the site expects for the GetVideoUrl XHR call.
    token = s.cookies.get("__RequestVerificationToken")
    if not token:
        raise CsrfOrSessionError("CSRF cookie '__RequestVerificationToken' not found after loading homepage.")
    return token

def get_master_playlist_url(s: requests.Session, image_id: int, verbose: bool = False) -> str:
    # Calls /Camera/GetVideoUrl for an imageId to obtain the HLS master playlist (.m3u8) URL.
    csrf = get_csrf_cookie(s)

    headers = dict(XHR_HEADERS_BASE)
    headers["__RequestVerificationToken"] = csrf

    try:
        r = s.get(
            f"{BASE}/Camera/GetVideoUrl",
            params={"imageId": str(image_id)},
            headers=headers,
            timeout=20,
        )
    except requests.RequestException as e:
        raise GetVideoUrlError(f"GetVideoUrl request error: {e}") from e

    if r.status_code in (401, 403):
        raise CsrfOrSessionError(f"GetVideoUrl unauthorized/forbidden: {r.status_code}")
    if r.status_code != 200:
        raise GetVideoUrlError(f"GetVideoUrl status {r.status_code}")

    body = r.text.strip()
    if verbose:
        print(f"[GetVideoUrl] image_id={image_id} ct={r.headers.get('Content-Type')} body={body[:160]!r}")

    master = body.strip().strip('"')
    if "m3u8" not in master:
        raise GetVideoUrlError(f"GetVideoUrl did not return m3u8 URL. body={body[:200]!r}")

    return urljoin(f"{BASE}/", master)  # Ensures relative URLs become absolute.

def fetch_m3u8(url: str, verbose: bool = False) -> str:
    # Downloads an M3U8 playlist and verifies it looks like a real HLS playlist.
    try:
        r = requests.get(url, headers=HLS_HEADERS, timeout=20)
    except requests.RequestException as e:
        raise PlaylistError(f"m3u8 request error: {e}") from e

    if r.status_code in (401, 403):
        raise PlaylistError(f"m3u8 unauthorized/forbidden: {r.status_code}")
    if r.status_code == 404:
        raise PlaylistError("m3u8 not found (404)")
    if r.status_code != 200:
        raise PlaylistError(f"m3u8 status {r.status_code}")

    text = r.text
    if verbose:
        print(f"[m3u8] GET {url} ct={r.headers.get('Content-Type')}")
        for line in text.splitlines()[:10]:
            print("  ", line)

    if not text.lstrip().startswith("#EXTM3U"):
        raise PlaylistError("Response did not look like an M3U8 playlist")
    return text

def resolve_variant(master_url: str, master_text: str) -> str:
    # From the master playlist, finds the first variant playlist (.m3u8) URL.
    for line in master_text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and ".m3u8" in line:
            return urljoin(master_url, line)
    raise PlaylistError("No variant (.m3u8) line found in master playlist")

def resolve_first_ts(variant_url: str, variant_text: str) -> str:
    # From the variant playlist, finds the first TS media segment URL (to probe stream liveness).
    for line in variant_text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and ".ts" in line:
            return urljoin(variant_url, line)
    raise OfflineError("No TS segments found in variant playlist (camera may be offline)")

def preflight_stream(variant_url: str, variant_text: str, verbose: bool = False) -> None:
    # Quick “is it live?” check by fetching one TS segment and verifying it has real bytes.
    ts_url = resolve_first_ts(variant_url, variant_text)

    try:
        rs = requests.get(ts_url, headers=HLS_HEADERS, timeout=20, stream=True)
    except requests.RequestException as e:
        raise OfflineError(f"TS fetch failed: {e}") from e

    if rs.status_code == 404:
        raise OfflineError("TS segment not found (404) - stream likely stalled/offline")
    if rs.status_code in (401, 403):
        raise PlaylistError(f"TS unauthorized/forbidden: {rs.status_code}")
    if rs.status_code != 200:
        raise OfflineError(f"TS status {rs.status_code}")

    chunk = next(rs.iter_content(chunk_size=65536), b"")  # Read a small chunk to confirm media is flowing.
    if verbose:
        print(f"[preflight] ts={ts_url} first_chunk={len(chunk)} bytes")

    if len(chunk) < 10_000:
        raise OfflineError(f"TS segment too small ({len(chunk)} bytes) - likely offline/stalled")

def record_with_ffmpeg(variant_url: str, seconds: int, out_file: Path, verbose: bool = False) -> None:
    # Records an HLS variant playlist for N seconds into an output file by shelling out to ffmpeg.
    out_file.parent.mkdir(parents=True, exist_ok=True)

    header_blob = (  # Injects headers so ffmpeg requests resemble the browser’s cross-origin requests.
        f"Origin: {BASE}\r\n"
        f"Referer: {BASE}/\r\n"
        f"User-Agent: {UA}\r\n"
    )

    cmd = [
        "ffmpeg", "-y",                 # Overwrite output if it exists.
        "-headers", header_blob,        # Attach required HTTP headers.
        "-i", variant_url,              # Input is the HLS variant playlist.
        "-t", str(seconds),             # Limit recording duration.
        "-c", "copy",                   # No re-encode; just remux segments to a file.
        str(out_file),
    ]

    if verbose:
        print("[ffmpeg] " + " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RecorderError(f"ffmpeg failed (exit {e.returncode})") from e

def validate_output(out_file: Path, min_bytes: int = 100_000) -> int:
    # Ensures ffmpeg produced a non-trivial file, returning size in bytes as a success signal.
    if not out_file.exists():
        raise RecorderError("ffmpeg finished but output file is missing")
    size = out_file.stat().st_size
    if size < min_bytes:
        raise RecorderError(f"Output too small ({size} bytes) - likely no real media recorded")
    return size

def record_one_with_retries(
    session: requests.Session,
    image_id: int,
    seconds: int,
    out_file: Path,
    verbose: bool,
    policy: RetryPolicy,
) -> tuple[requests.Session, int]:
    """
    Records one camera with robust retry logic.
    Returns (possibly refreshed) session and bytes_written.
    """
    master_url: Optional[str] = None  # Stores the master playlist URL for this image_id.

    # 1) Token/session retries (fix CSRF/session expiry by refreshing cookies/session).
    token_attempt = 0
    while True:
        try:
            master_url = get_master_playlist_url(session, image_id, verbose=verbose)
            break
        except CsrfOrSessionError as e:
            token_attempt += 1
            if token_attempt > policy.token_retries:
                raise
            if verbose:
                print(f"[retry] image_id={image_id} CSRF/session: {e} -> refresh session")
            session = refresh_session(session)
            time.sleep(policy.backoff_s)

    assert master_url is not None

    # 2) Resolve variant + preflight (ensure variant exists and stream segments look live).
    offline_attempt = 0
    while True:
        try:
            master_text = fetch_m3u8(master_url, verbose=verbose)
            variant_url = resolve_variant(master_url, master_text)

            variant_text = fetch_m3u8(variant_url, verbose=False)
            preflight_stream(variant_url, variant_text, verbose=verbose)
            break
        except PlaylistError as e:
            token_attempt += 1
            if token_attempt > policy.token_retries:
                raise
            if verbose:
                print(f"[retry] image_id={image_id} playlist/token: {e} -> reacquire master")
            master_url = get_master_playlist_url(session, image_id, verbose=verbose)
            time.sleep(policy.backoff_s)
        except OfflineError as e:
            offline_attempt += 1
            if offline_attempt > policy.offline_retries:
                raise
            if verbose:
                print(f"[retry] image_id={image_id} offline-ish: {e} -> wait & retry preflight")
            time.sleep(policy.backoff_s)

    # 3) Record retries (ffmpeg hiccups -> reacquire variant and retry recording).
    record_attempt = 0
    while True:
        try:
            record_with_ffmpeg(variant_url, seconds, out_file, verbose=verbose)
            bytes_written = validate_output(out_file)
            return session, bytes_written
        except RecorderError as e:
            record_attempt += 1
            if record_attempt > policy.record_retries:
                raise
            if verbose:
                print(f"[retry] image_id={image_id} recorder: {e} -> reacquire master/variant & retry")
            master_url = get_master_playlist_url(session, image_id, verbose=verbose)
            master_text = fetch_m3u8(master_url, verbose=verbose)
            variant_url = resolve_variant(master_url, master_text)
            time.sleep(policy.backoff_s)

# -----------------------------
# Batch with structured results + report writing
# -----------------------------
def record_many_with_results(
    image_ids: List[int],
    seconds: int,
    out_dir: Path,
    out_template: str,
    verbose: bool,
    policy: RetryPolicy,
) -> List[RecordResult]:
    # Records multiple image_ids sequentially and returns a structured per-id result list (no hard crash).
    out_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()
    results: List[RecordResult] = []

    for image_id in image_ids:
        started = utc_now_iso()
        stamp = utc_stamp()
        out_file = out_dir / out_template.format(image_id=image_id, seconds=seconds, stamp=stamp)

        print(f"[START] image_id={image_id} -> {out_file}")

        try:
            session, bytes_written = record_one_with_retries(
                session=session,
                image_id=image_id,
                seconds=seconds,
                out_file=out_file,
                verbose=verbose,
                policy=policy,
            )
            ended = utc_now_iso()
            print(f"[DONE]  image_id={image_id} saved={out_file} bytes={bytes_written}")
            results.append(
                RecordResult(
                    image_id=image_id,
                    status="ok",
                    out_file=str(out_file),
                    bytes_written=bytes_written,
                    error_type=None,
                    error_message=None,
                    started_utc=started,
                    ended_utc=ended,
                )
            )
        except OfflineError as e:
            ended = utc_now_iso()
            print(f"[SKIP]  image_id={image_id} OFFLINE: {e}")
            results.append(
                RecordResult(
                    image_id=image_id,
                    status="skip",
                    out_file=None,
                    bytes_written=None,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    started_utc=started,
                    ended_utc=ended,
                )
            )
        except StreamError as e:
            ended = utc_now_iso()
            print(f"[FAIL]  image_id={image_id} STREAM_ERROR: {e}")
            results.append(
                RecordResult(
                    image_id=image_id,
                    status="fail",
                    out_file=None,
                    bytes_written=None,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    started_utc=started,
                    ended_utc=ended,
                )
            )
        except Exception as e:
            ended = utc_now_iso()
            print(f"[FAIL]  image_id={image_id} UNEXPECTED: {e}")
            results.append(
                RecordResult(
                    image_id=image_id,
                    status="fail",
                    out_file=None,
                    bytes_written=None,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    started_utc=started,
                    ended_utc=ended,
                )
            )

    return results

def write_run_report(
    report_path: Path,
    run_meta: Dict[str, Any],
    targets: List[Dict[str, Any]],
    results: List[RecordResult],
) -> None:
    # Writes one JSON file summarizing the run (args/targets/results) for debugging and auditing.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_meta": run_meta,
        "targets": targets,
        "results": [asdict(r) for r in results],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

from concurrent.futures import ThreadPoolExecutor, as_completed

def record_many_with_results_parallel(
    image_ids: List[int],
    seconds: int,
    out_dir: Path,
    out_template: str,
    verbose: bool,
    policy: RetryPolicy,
    max_workers: int = 5,
) -> List[RecordResult]:
    """
    Record multiple image_ids concurrently (up to max_workers at a time).
    Safe approach: each job gets its own Session (CSRF/cookies) and closes it.
    Returns results in the SAME order as image_ids.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def _job(image_id: int) -> RecordResult:
        started = utc_now_iso()
        stamp = utc_stamp()
        out_file = out_dir / out_template.format(image_id=image_id, seconds=seconds, stamp=stamp)

        print(f"[START] image_id={image_id} -> {out_file}")

        session = None
        try:
            session = make_session()
            session, bytes_written = record_one_with_retries(
                session=session,
                image_id=image_id,
                seconds=seconds,
                out_file=out_file,
                verbose=verbose,
                policy=policy,
            )
            ended = utc_now_iso()
            print(f"[DONE]  image_id={image_id} saved={out_file} bytes={bytes_written}")
            return RecordResult(
                image_id=image_id,
                status="ok",
                out_file=str(out_file),
                bytes_written=bytes_written,
                error_type=None,
                error_message=None,
                started_utc=started,
                ended_utc=ended,
            )
        except OfflineError as e:
            ended = utc_now_iso()
            print(f"[SKIP]  image_id={image_id} OFFLINE: {e}")
            return RecordResult(
                image_id=image_id,
                status="skip",
                out_file=None,
                bytes_written=None,
                error_type=type(e).__name__,
                error_message=str(e),
                started_utc=started,
                ended_utc=ended,
            )
        except StreamError as e:
            ended = utc_now_iso()
            print(f"[FAIL]  image_id={image_id} STREAM_ERROR: {e}")
            return RecordResult(
                image_id=image_id,
                status="fail",
                out_file=None,
                bytes_written=None,
                error_type=type(e).__name__,
                error_message=str(e),
                started_utc=started,
                ended_utc=ended,
            )
        except Exception as e:
            ended = utc_now_iso()
            print(f"[FAIL]  image_id={image_id} UNEXPECTED: {e}")
            return RecordResult(
                image_id=image_id,
                status="fail",
                out_file=None,
                bytes_written=None,
                error_type=type(e).__name__,
                error_message=str(e),
                started_utc=started,
                ended_utc=ended,
            )
        finally:
            try:
                if session is not None:
                    session.close()
            except Exception:
                pass

    # Submit all jobs but run at most max_workers concurrently
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for image_id in image_ids:
            futures[ex.submit(_job, image_id)] = image_id

        # Collect results
        by_id: Dict[int, RecordResult] = {}
        for fut in as_completed(futures):
            image_id = futures[fut]
            by_id[image_id] = fut.result()

    # Preserve original ordering
    return [by_id[i] for i in image_ids]