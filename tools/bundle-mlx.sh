#!/bin/sh
# Builds the app with the EXPERIMENTAL MLX Large slot and bundles what it needs (docs/architecture/app.md):
#   Contents/MacOS/tingmix-mlx-worker      crash-isolated probe/decode child (sidecar, rpath rewritten)
#   Contents/Frameworks/{libmlx.dylib,libjaccl.dylib,mlx.metallib}
# Needs: MLX_PREFIX (default: the experiments venv wheel layout), vendor/onnxruntime (tools/ort-fetch/fetch.sh).
# Usage: tools/bundle-mlx.sh [debug|release]
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PROFILE=${1:-debug}
MLX_PREFIX=${MLX_PREFIX:-$ROOT/experiments/macos/mlx/.venv/lib/python3.12/site-packages/mlx}
export MLX_PREFIX
for f in libmlx.dylib libjaccl.dylib mlx.metallib; do
  [ -f "$MLX_PREFIX/lib/$f" ] || { echo "missing $MLX_PREFIX/lib/$f" >&2; exit 1; }
done
FLAG=""; TFLAG="--debug"
[ "$PROFILE" = release ] && { FLAG="--release"; TFLAG=""; }

nice -n 19 cargo build $FLAG -p tingmix-runtime-mlx --features probe --bin tingmix-mlx-worker
mkdir -p src-tauri/binaries
W=src-tauri/binaries/tingmix-mlx-worker-aarch64-apple-darwin
cp "target/$PROFILE/tingmix-mlx-worker" "$W"
# the dev build embeds an absolute rpath to the wheel; the bundle must not
install_name_tool -delete_rpath "$MLX_PREFIX/lib" "$W"
install_name_tool -add_rpath @executable_path/../Frameworks "$W"
codesign -s - --force "$W"

L="$MLX_PREFIX/lib"
CONF=$(printf '{"bundle":{"externalBin":["binaries/tingmix-mlx-worker"],"macOS":{"signingIdentity":null,"files":{"Frameworks/libmlx.dylib":"%s/libmlx.dylib","Frameworks/libjaccl.dylib":"%s/libjaccl.dylib","Frameworks/mlx.metallib":"%s/mlx.metallib"}}}}' "$L" "$L" "$L")
# Tauri's own ad-hoc signing cannot seal a bundle whose Frameworks hold an unsigned data file (mlx.metallib; codesign keeps
# its signature in an xattr that the bundler's copy drops), so signing is switched off here and done below, inside the bundle.
pnpm tauri build $TFLAG --bundles app --features mlx \
  --config src-tauri/tauri.ort-macos.conf.json --config "$CONF"

APP="target/$PROFILE/bundle/macos/TingMix.app"
for f in Frameworks/mlx.metallib Frameworks/libmlx.dylib Frameworks/libjaccl.dylib MacOS/tingmix-mlx-worker; do
  codesign -s - --force "$APP/Contents/$f"
done
codesign -s - --force --entitlements src-tauri/Entitlements.plist "$APP"
codesign --verify --deep --strict "$APP"
codesign -dv "$APP" 2>&1 | grep -E "Identifier|Info.plist"
