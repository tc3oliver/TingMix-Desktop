#!/bin/sh
# Re-run the whole attribution from the cut clips (after a review export, run `uv run python -m ea.gt EXPORT.json` first).
# Needs: uv, ffmpeg, MLX_PREFIX (pip mlx wheel layout) and an MLX model package directory PKG (see README).
set -eu
cd "$(dirname "$0")"
: "${PKG:?set PKG to the MLX Large package directory}"
: "${MLX_PREFIX:?set MLX_PREFIX}"
NAMES="product finance"
for n in $NAMES; do
  for v in orig vocals slow shift; do
    [ "$v" = orig ] || [ -d "out/$n/$v/clips" ] || uv run python -m ea.counterfactual "$n"
    uv run python -m ea.decode "$n" --variant "$v"
    uv run python -m ea.decoders "$n" --variant "$v"
  done
done
(cd app-live && TINGMIX_MLX_DEV_RPATH=1 cargo build -q --release)
for n in $NAMES; do
  DYLD_LIBRARY_PATH="$MLX_PREFIX/lib" app-live/target/release/ea-app-live "$PKG" "out/$n/app_live.json" out/$n/regions/r*.wav
  for cap in 6 8 10 12 15 20; do
    EA_MAX_SEG_S=$cap DYLD_LIBRARY_PATH="$MLX_PREFIX/lib" app-live/target/release/ea-app-live "$PKG" "out/$n/app_live_cap$cap.json" out/$n/regions/r*.wav
  done
done
uv run python -m ea.score $NAMES > out/tables.md
uv run python -m ea.livecaps $NAMES >> out/tables.md
uv run python -m ea.winsweep $NAMES >> out/tables.md
uv run python -m ea.attribute $NAMES >> out/tables.md
uv run python -m ea.liveparity $NAMES >> out/tables.md
# sensitivity: the product video against its uploader subtitles (outputs out/*-subtitles.*)
EA_REF=subtitles uv run python -m ea.score product > out/tables-subtitles.md
EA_REF=subtitles uv run python -m ea.livecaps product >> out/tables-subtitles.md
EA_REF=subtitles uv run python -m ea.attribute product >> out/tables-subtitles.md
echo "done: out/tables.md, out/tables-subtitles.md, out/worst20.md, out/*.json"
