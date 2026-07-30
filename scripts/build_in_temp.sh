#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nsysscope-node-build.XXXXXX")"

cleanup() {
  if [[ -n "${BUILD_DIR:-}" && -d "$BUILD_DIR" ]]; then
    find "$BUILD_DIR" -depth -delete
    BUILD_DIR=""
  fi
}
trap cleanup EXIT INT TERM

# Copy source without development dependencies or an old build.  npm ci and
# Vinext therefore never write node_modules into the checked-out application.
tar -C "$PROJECT_DIR" \
  --exclude='./node_modules' \
  --exclude='./dist' \
  --exclude='./.git' \
  -cf - . | tar -C "$BUILD_DIR" -xf -

cd "$BUILD_DIR"
npm ci --ignore-scripts
npm run build:site

# The local command is a validation/build check.  Sites builds from the
# pushed source, so do not copy dist back into the checkout either; the EXIT
# trap removes the generated artifact together with node_modules.
