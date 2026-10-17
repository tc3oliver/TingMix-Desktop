#!/bin/sh
# Standalone C++ parity driver (no Rust): sh build.sh && ../out/tingmix_mlx_test <artifact> <0|1|2> ../out/ref-bin <ids...>
set -e
here=$(cd "$(dirname "$0")" && pwd)
mlx="${MLX_PREFIX:-$here/../.venv/lib/python3.12/site-packages/mlx}"
crate="$here/../../../../crates/tingmix-runtime-mlx/cpp"
mkdir -p "$here/../out"
clang++ -std=c++20 -O2 -Wall -I"$mlx/include" -I"$crate" "$crate/tingmix_mlx.cpp" "$here/test_main.cpp" \
  -L"$mlx/lib" -lmlx -Wl,-rpath,"$mlx/lib" -o "$here/../out/tingmix_mlx_test"
