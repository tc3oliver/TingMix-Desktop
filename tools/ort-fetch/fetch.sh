#!/bin/bash
# Fetch a pinned, sha256-verified ONNX Runtime 1.31.0 per platform into vendor/onnxruntime/<platform>/ (git-ignored).
#   tools/ort-fetch/fetch.sh [macos-arm64|windows-x64|all]
# Needs network once; afterwards builds and runs are offline. Archives are cached in vendor/onnxruntime/.cache/.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="$root/vendor/onnxruntime"
want="${1:-all}"
mkdir -p "$out/.cache"
# shasum on macOS, sha256sum on Linux and in Git Bash on Windows
if command -v shasum >/dev/null 2>&1; then sha256() { shasum -a 256 "$@"; }; else sha256() { sha256sum "$@"; }; fi
sha() { sha256 "$1" | awk '{print $1}'; }

while IFS=$'\t' read -r platform url expect; do
  [[ -z "$platform" || "$platform" == \#* ]] && continue
  [[ "$want" != all && "$want" != "$platform" ]] && continue
  archive="$out/.cache/$(basename "$url")"
  if [[ ! -f "$archive" || "$(sha "$archive")" != "$expect" ]]; then
    rm -f "$archive"
    curl -fL --proto '=https' --tlsv1.2 -o "$archive.part" "$url"
    mv "$archive.part" "$archive"
  fi
  got="$(sha "$archive")"
  if [[ "$got" != "$expect" ]]; then
    echo "sha256 mismatch for $platform: expected $expect, got $got" >&2
    rm -f "$archive"
    exit 1
  fi
  dest="$out/$platform"
  rm -rf "$dest" "$dest.tmp"
  mkdir -p "$dest.tmp"
  case "$archive" in
    *.tgz) tar -xzf "$archive" -C "$dest.tmp" ;;
    *.zip) unzip -q "$archive" -d "$dest.tmp" ;;
  esac
  # archives wrap everything in one onnxruntime-<os>-<arch>-<ver>/ directory: hoist it
  if [[ "$(ls -A "$dest.tmp" | wc -l)" -eq 1 && -d "$dest.tmp/$(ls -A "$dest.tmp")" ]]; then
    inner="$dest.tmp/$(ls -A "$dest.tmp")"
    mv "$inner" "$dest.inner" && rm -rf "$dest.tmp" && mv "$dest.inner" "$dest.tmp"
  fi
  mv "$dest.tmp" "$dest"
  # the files we bundle must also match the recorded hashes (catches a tampered cache extraction)
  (cd "$out" && grep " $platform/" "$here/FILES.sha256" | sha256 -c --quiet -)
  echo "$platform: archive and bundled files verified ($got)"
done < "$here/MANIFEST.tsv"
echo "ONNX Runtime files:"; find "$out" -path "$out/.cache" -prune -o -path '*testdata*' -prune -o -path '*.dSYM*' -prune -o \( -name '*.dylib' -o -name '*.dll' -o -name 'LICENSE' -o -name 'ThirdPartyNotices.txt' \) -print
