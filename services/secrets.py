# services/secrets.py
from pathlib import Path

SECRETS_DIR = Path(__file__).resolve().parent.parent / "secrets"

def read_secret(name: str) -> str:
    """
    Read a secret from secrets/<name>.txt
    """
    path = SECRETS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing secret file: {path}")
    return path.read_text(encoding="utf-8").strip()