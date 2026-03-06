#!/usr/bin/env bash
set -euo pipefail

umask 077  # new files default to 600/700-style perms

LOG_DIR="/root/RoadState/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
BOOT_OUT="$LOG_DIR/bootstrap.${STAMP}.out"
BOOT_ERR="$LOG_DIR/bootstrap.${STAMP}.err"

# Log everything to file + console (unique per run, no overwrites)
exec > >(tee -a "$BOOT_OUT") 2> >(tee -a "$BOOT_ERR" >&2)

echo "[boot] Starting bootstrap in: $(pwd)"
echo "[boot] UTC stamp: ${STAMP}"
echo "[boot] User: $(id -u):$(id -g)  HOME=${HOME:-<unset>}"

export DEBIAN_FRONTEND=noninteractive

if [[ "$(id -u)" -ne 0 ]]; then
  echo "[boot] ERROR: This script expects to run as root (apt-get). Exiting."
  exit 1
fi

echo "[boot] apt-get update"
apt-get update -y

echo "[boot] Installing packages: curl ffmpeg git git-lfs ca-certificates"
apt-get install -y --no-install-recommends curl ffmpeg git git-lfs ca-certificates
rm -rf /var/lib/apt/lists/*

echo "[boot] Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is available even if the env file isn't present
export PATH="${HOME:-/root}/.local/bin:$PATH"

# Source uv env if it exists (don’t fail if missing)
if [[ -f "${HOME:-/root}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME:-/root}/.local/bin/env"
fi

echo "[boot] uv path: $(command -v uv || true)"
uv --version

echo "[boot] Installing huggingface_hub CLI via uv tool (idempotent)"
uv tool install -U huggingface_hub

# -----------------------------
# Venv 
# -----------------------------
if [[ -d ".venv" ]]; then
  echo "[boot] venv exists: .venv (skipping create)"
else
  echo "[boot] Creating venv (Python 3.12) with seed"
  uv venv --python 3.12 --seed
fi

echo "[boot] Activating venv"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[boot] Installing vLLM (torch backend auto)"
uv pip install vllm --torch-backend=auto

echo "[boot] Installing Python deps: openai pyyaml hf_transfer"
uv pip install openai pyyaml hf_transfer

# -----------------------------
# Secrets 
# -----------------------------
echo "[boot] Ensuring secret files exist (non-destructive)"
SECRETS_DIR="/root/RoadState/secrets"
mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR"

# Create only if missing; 
for f in 511ga_api_key.txt google_routes_api_key.txt; do
  if [[ -e "$SECRETS_DIR/$f" ]]; then
    echo "[boot] Secret exists: $SECRETS_DIR/$f (leaving as-is)"
  else
    : > "$SECRETS_DIR/$f"
    chmod 600 "$SECRETS_DIR/$f"
    echo "[boot] Secret created: $SECRETS_DIR/$f"
  fi
done

# populate from env vars ONLY if file is empty
if [[ -n "${GA511_API_KEY:-}" && ! -s "$SECRETS_DIR/511ga_api_key.txt" ]]; then
  printf '%s' "$GA511_API_KEY" > "$SECRETS_DIR/511ga_api_key.txt"
  echo "[boot] Wrote GA511_API_KEY into 511ga_api_key.txt (was empty)"
fi

if [[ -n "${GOOGLE_ROUTES_API_KEY:-}" && ! -s "$SECRETS_DIR/google_routes_api_key.txt" ]]; then
  printf '%s' "$GOOGLE_ROUTES_API_KEY" > "$SECRETS_DIR/google_routes_api_key.txt"
  echo "[boot] Wrote GOOGLE_ROUTES_API_KEY into google_routes_api_key.txt (was empty)"
fi

echo "[boot] Secrets ready:"
echo "       - $SECRETS_DIR/511ga_api_key.txt (bytes: $(wc -c < "$SECRETS_DIR/511ga_api_key.txt"))"
echo "       - $SECRETS_DIR/google_routes_api_key.txt (bytes: $(wc -c < "$SECRETS_DIR/google_routes_api_key.txt"))"

echo "[boot] HF auth (non-interactive if HF_TOKEN is provided)"
if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential
  echo "[boot] HF login: OK"
else
  echo "[boot] HF_TOKEN not set; skipping hf auth login."
  echo "[boot] Set HF_TOKEN in RunPod environment variables if you need gated model downloads."
fi

echo "[boot] START_VLLM=${START_VLLM:-0}"
if [[ "${START_VLLM:-0}" == "1" ]]; then
  echo "[boot] Starting vLLM..."
  mkdir -p "$LOG_DIR"

  VLLM_OUT="$LOG_DIR/vllm.${STAMP}.out"
  VLLM_ERR="$LOG_DIR/vllm.${STAMP}.err"

  # Background start + log capture (unique per run)
  nohup uv run vllm serve nvidia/Cosmos-Reason2-8B \
    --port 8000 \
    --allowed-local-media-path "/root" \
    --max-model-len 8192 \
    --media-io-kwargs '{"video": {"num_frames": 20}}' \
    --reasoning-parser qwen3 \
    > "$VLLM_OUT" 2> "$VLLM_ERR" &

  echo "[boot] vLLM PID: $!"
  echo "[boot] vLLM logs: $VLLM_OUT  /  $VLLM_ERR"
else
  echo "[boot] START_VLLM not set to 1; skipping vLLM start."
fi

echo "[boot] Done."
echo "[boot] Logs: $BOOT_OUT (stdout), $BOOT_ERR (stderr)"