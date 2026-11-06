#!/usr/bin/env python3
"""pandas side of the pandas vs Bandal comparison (see benchmark.sh / benchmark.swift).

Every workload mirrors benchmark.swift. Only the computation is timed: data
generation, warm-up runs and checksum evaluation are excluded. Results are
printed as JSON.

Usage: benchmark.py [--rows 100000,1000000,5000000] [--repeat 5] [--warmup 2]
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd


def make_data(rows):
    # Deterministic integer-valued data, identical to benchmark.swift.
    i = np.arange(rows, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "group": pd.Series(("g" + (i % 100).astype(str)), dtype=object),
            "a": ((i * 7919) % 10007).astype(np.float64),
            "b": ((i * 104729) % 10009 + 1).astype(np.float64),
            "c": (i % 1000).astype(np.float64),
            "d": ((i * 31) % 9973).astype(np.float64),
        }
    )
    lookup = pd.DataFrame(
        {
            "group": [f"g{k}" for k in range(100)],
            "category": [float(k) for k in range(100)],
        }
    )
    return frame, lookup


def workloads(frame, lookup):
    numeric = frame[["a", "b", "c", "d"]]
    a, b, c = frame["a"], frame["b"], frame["c"]

    def stats_summary():
        return a.mean(), a.std(), a.median(), a.min(), a.max()

    def arithmetic_sqrt():
        return np.sqrt(a * b + c)

    return [
        ("stats_summary", stats_summary, lambda r: float(sum(r))),
        (
            "sort",
            lambda: frame.sort_values("a", kind="stable"),
            lambda r: float(r["a"].iloc[0] + r["a"].iloc[len(r) // 2] + r["a"].iloc[-1]),
        ),
        (
            "rolling_mean_20",
            lambda: numeric["a"].rolling(20).mean(),
            lambda r: float(r.sum()),
        ),
        ("expanding_sum", lambda: numeric["a"].expanding().sum(), lambda r: float(r.sum())),
        ("arithmetic_sqrt", arithmetic_sqrt, lambda r: float(r.sum())),
        ("correlation", lambda: numeric.corr(), lambda r: float(r.to_numpy().sum())),
        (
            "filter",
            lambda: frame[frame["a"] >= 5000],
            lambda r: float(len(r) + r["b"].sum()),
        ),
        (
            "groupby_sum",
            lambda: frame.groupby("group")["c"].sum(),
            lambda r: float(len(r) + r.sum()),
        ),
        (
            "left_join",
            lambda: frame.merge(lookup, on="group", how="left"),
            lambda r: float(len(r) + r["category"].sum()),
        ),
    ]


def measure(operation, checksum, repeat, warmup):
    for _ in range(warmup):
        operation()
    durations = []
    value = float("nan")
    for _ in range(repeat):
        start = time.perf_counter()
        result = operation()
        durations.append((time.perf_counter() - start) * 1000.0)
        value = checksum(result)
    durations.sort()
    return durations[len(durations) // 2], durations[0], value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="100000,1000000,5000000")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    results = []
    for rows in [int(r) for r in args.rows.split(",")]:
        print(f"pandas: {rows} rows", file=sys.stderr)
        frame, lookup = make_data(rows)
        for name, operation, checksum in workloads(frame, lookup):
            median, minimum, value = measure(operation, checksum, args.repeat, args.warmup)
            results.append(
                {
                    "rows": rows,
                    "workload": name,
                    "mode": "pandas",
                    "median_ms": median,
                    "min_ms": minimum,
                    "checksum": value,
                }
            )
    print(
        json.dumps(
            {
                "library": "pandas",
                "backend": f"pandas {pd.__version__} / numpy {np.__version__}",
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

