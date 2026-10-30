#!/usr/bin/env bash
# Run flux klein studio on its own under `helm dev`, which keeps what it stores in ./.helm.
#
#   [FLUX_MODELS=<dir holding flux-3-image-klein, -pup, and optionally -base>] [HELM=<helm>] bash web/run.sh [helm dev flags]
#
# Base is not linked: the studio finds flux-3-image-klein-base beside flux-3-image-klein itself.
#   bash web/run.sh stop
#
# A run first stops the helm dev this script started before, so starting again is a restart.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

HELM="${HELM:-helm}"
MODELS="${FLUX_MODELS:-$HOME/GenAI/models}"
RUN="$PWD/.cache/flux-studio"
PIDFILE="$RUN/flux-studio.pid"

# Ends the helm dev this script last started, which stops the studio before it exits.
stop() {
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "$pid" ] && ps -p "$pid" -o command= | grep -q " dev -f helmstudio.yaml"; then
    echo "helm dev: stopping the flux klein studio started before ($pid)"
    kill -TERM "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null; do sleep 0.2; done
  fi
  rm -f "$PIDFILE"
}
if [ "${1:-}" = "stop" ]; then
  stop
  exit 0
fi

if ! command -v "$HELM" >/dev/null; then
  echo "helm is not installed. Install it with helmstudio's installer, or set HELM to its path." >&2
  exit 1
fi
if ! .venv/bin/python -c "import helm_runtime_sdk" 2>/dev/null; then
  echo "helm-runtime-sdk is not installed in .venv: run uv sync" >&2
  exit 1
fi

stop
mkdir -p "$RUN"
echo $$ >"$PIDFILE" # exec keeps this pid, so it is helm dev's
exec "$HELM" dev -f helmstudio.yaml -venv .venv \
  -link "klein=$MODELS/flux-3-image-klein" \
  -link "pup=$MODELS/flux-3-image-klein-pup" \
  "$@"
