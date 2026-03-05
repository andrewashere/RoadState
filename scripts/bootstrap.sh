#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/root/logs"
mkdir -p "$LOG_DIR"

# Log everything to file + console
exec > >(tee -a "$LOG_DIR/bootstrap.out") 2> >(tee -a "$LOG_DIR/bootstrap.err" >&2)

echo "[boot] Starting bootstrap in: $(pwd)"
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

echo "[boot] uv version: $(command -v uv || true)"
uv --version

echo "[boot] Installing huggingface_hub CLI via uv tool"
uv tool install -U huggingface_hub

echo "[boot] Creating venv (Python 3.12) with seed"
uv venv --python 3.12 --seed

echo "[boot] Activating venv"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[boot] Installing vLLM (torch backend auto)"
uv pip install vllm --torch-backend=auto

echo "[boot] Installing Python deps: openai pyyaml hf_transfer"
uv pip install openai pyyaml hf_transfer

# -----------------------------
# Secrets (startup-safe)
# -----------------------------
echo "[boot] Creating secret files (if missing)"
SECRETS_DIR="/root/secrets"
mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR"

touch "$SECRETS_DIR/511ga_api_key.txt" "$SECRETS_DIR/google_routes_api_key.txt"
chmod 600 "$SECRETS_DIR/511ga_api_key.txt" "$SECRETS_DIR/google_routes_api_key.txt"

# Optional: populate from env vars (recommended)
# - GA511_API_KEY for secrets/511ga_api_key.txt
# - GOOGLE_ROUTES_API_KEY for secrets/google_routes_api_key.txt
if [[ -n "${GA511_API_KEY:-}" && ! -s "$SECRETS_DIR/511ga_api_key.txt" ]]; then
  printf '%s' "$GA511_API_KEY" > "$SECRETS_DIR/511ga_api_key.txt"
fi

if [[ -n "${GOOGLE_ROUTES_API_KEY:-}" && ! -s "$SECRETS_DIR/google_routes_api_key.txt" ]]; then
  printf '%s' "$GOOGLE_ROUTES_API_KEY" > "$SECRETS_DIR/google_routes_api_key.txt"
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

  # Background start + log capture
  nohup uv run vllm serve nvidia/Cosmos-Reason2-8B \
    --port 8000 \
    --allowed-local-media-path "/root" \
    --max-model-len 8192 \
    --media-io-kwargs '{"video": {"num_frames": 20}}' \
    --reasoning-parser qwen3 \
    > "$LOG_DIR/vllm.out" 2> "$LOG_DIR/vllm.err" &

  echo "[boot] vLLM PID: $!"
  echo "[boot] vLLM logs: $LOG_DIR/vllm.out  /  $LOG_DIR/vllm.err"
else
  echo "[boot] START_VLLM not set to 1; skipping vLLM start."
fi

echo "[boot] Done."
echo "[boot] Logs: $LOG_DIR/bootstrap.out (stdout), $LOG_DIR/bootstrap.err (stderr)"