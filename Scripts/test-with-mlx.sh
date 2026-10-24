#!/bin/sh
# Runs the test suite with MLX enabled.
# `swift test` cannot locate MLX's Metal library next to the test bundle, so
# this script builds, colocates it as `mlx.metallib`, and runs without rebuilding.
set -e
cd "$(dirname "$0")/.."
BIN=.build/out/Products/Debug/BandalTests.xctest/Contents/MacOS
rm -f "$BIN/mlx.metallib"
# A leftover copy breaks code signing of the next plain `swift test`.
trap 'rm -f "$BIN/mlx.metallib"' EXIT
swift build --build-tests
LIB=$(find .build -name default.metallib | head -n 1)
cp "$LIB" "$BIN/mlx.metallib"
BANDAL_REQUIRE_MLX=1 swift test --skip-build "$@"
