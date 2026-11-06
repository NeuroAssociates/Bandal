#!/bin/sh
# Compares pandas and Bandal (Swift native / MLX CPU / MLX GPU) on identical workloads.
#
# 1. Creates a uv virtual environment (.benchmark-venv) with pandas and numpy.
# 2. Builds the Bandal benchmark in release mode (including MLX's Metal library).
# 3. Runs benchmark.py and benchmark.swift; only the computation is timed
#    (setup, compilation, warm-up and MLX kernel compilation are excluded).
# 4. Prints a comparison table.
#
# Usage: Scripts/benchmark.sh [--rows 100000,1000000,5000000] [--repeat 5] [--warmup 2]
set -e
cd "$(dirname "$0")/.."

VENV=.benchmark-venv
OUT=.benchmark-results
ARGS="$*"

command -v uv >/dev/null 2>&1 || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

echo "==> Preparing Python environment (uv)"
uv venv --quiet --allow-existing "$VENV"
uv pip install --quiet --python "$VENV/bin/python" pandas numpy

echo "==> Building Bandal benchmark (release)"
swift build -c release --product BandalPandasBenchmark
BIN="$(swift build -c release --show-bin-path)"

mkdir -p "$OUT"
echo "==> Running pandas"
# shellcheck disable=SC2086
"$VENV/bin/python" Scripts/benchmark.py $ARGS > "$OUT/pandas.json"
echo "==> Running Bandal"
# shellcheck disable=SC2086
"$BIN/BandalPandasBenchmark" $ARGS > "$OUT/bandal.json"

echo
"$VENV/bin/python" Scripts/benchmark_compare.py "$OUT/pandas.json" "$OUT/bandal.json"
