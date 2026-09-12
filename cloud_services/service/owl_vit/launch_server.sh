#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XLA_PYTHON_CLIENT_PREALLOCATE=false python "$SCRIPT_DIR/server.py" \
  --ip "${SERVICE_HOST:-0.0.0.0}" --port "${SERVICE_PORT:-4000}"
