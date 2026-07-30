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

# Sites expects the generated dist/ directory as the deployable build output.
# Copy only that small artifact back; all dependencies and temporary files are
# removed by the EXIT trap.
if [[ -d "$PROJECT_DIR/dist" ]]; then
  find "$PROJECT_DIR/dist" -depth -delete
fi
cp -a "$BUILD_DIR/dist" "$PROJECT_DIR/dist"
