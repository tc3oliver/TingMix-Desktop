#!/bin/sh
# Builds tingmix-onnx-worker (the crash-isolated probe/decode child of the ONNX Large backend) without downloading ONNX
# Runtime and stages it as a Tauri sidecar: src-tauri/binaries/tingmix-onnx-worker-<target triple>[.exe].
# The Windows NSIS bundle (src-tauri/tauri.ort-windows.conf.json) lists it as externalBin, so run this before `pnpm bundle:windows`.
# Usage: tools/stage-onnx-worker.sh [release|debug]   (run it on the target platform; Windows: from Git Bash)
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${1:-release}
FLAG=""
[ "$PROFILE" = release ] && FLAG="--release"
TRIPLE=$(rustc -vV | sed -n 's/^host: //p')
EXT=""
case "$TRIPLE" in *windows*) EXT=".exe" ;; esac
# shellcheck disable=SC2086
cargo build $FLAG -p tingmix-runtime-onnx --no-default-features --features load-dynamic,probe --bin tingmix-onnx-worker
TARGET_DIR=${CARGO_TARGET_DIR:-target}
mkdir -p src-tauri/binaries
cp "$TARGET_DIR/$PROFILE/tingmix-onnx-worker$EXT" "src-tauri/binaries/tingmix-onnx-worker-$TRIPLE$EXT"
echo "staged src-tauri/binaries/tingmix-onnx-worker-$TRIPLE$EXT"
