#!/usr/bin/env bash
set -euo pipefail

exec uvicorn backend.app:app \
  --host "${NSYSSCOPE_HOST:-127.0.0.1}" \
  --port "${NSYSSCOPE_PORT:-8787}"
