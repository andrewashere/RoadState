from pathlib import Path

GA_CAMERAS_URL = "https://511ga.org/api/v2/get/cameras"
BASE_SITE = "https://511ga.org"

SECRETS_DIR = Path("secrets")
KEY_FILE = SECRETS_DIR / "511ga_key.txt"

CACHE_DIR = Path("cache")
RAW_CACHE_FILE = CACHE_DIR / "ga_cameras_raw.json"
META_CACHE_FILE = CACHE_DIR / "meta.json"

DB_PATH = CACHE_DIR / "ga_cameras.db"

RECORDINGS_DIR = Path("recordings") 

# API throttling note: 10 calls per 60s. 
CACHE_TTL_SECONDS_DEFAULT = 70  