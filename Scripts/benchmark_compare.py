#!/usr/bin/env python3
"""Prints a pandas vs Bandal comparison table from the two benchmark JSON files.

Usage: benchmark_compare.py pandas.json bandal.json
Speedup columns are pandas_ms / bandal_ms (> 1 means Bandal is faster).
The last column reports whether every Bandal mode reproduced pandas' checksum.
"""

import json
import sys

MODES = ["native", "mlx_cpu", "mlx_gpu", "mlx_gpu_float32"]
LABELS = {
    "native": "Swift",
    "mlx_cpu": "MLX-CPU",
    "mlx_gpu": "MLX-GPU",
    "mlx_gpu_float32": "MLX-GPU32",
}
# float32 GPU reductions are not bit-exact, so they get a looser tolerance.
TOLERANCE = {"native": 1e-9, "mlx_cpu": 1e-9, "mlx_gpu": 1e-4, "mlx_gpu_float32": 1e-4}


def load(path):
    with open(path) as handle:
        return json.load(handle)


def close(a, b, tolerance):
    if a is None or b is None:
        return False
    return abs(a - b) <= tolerance * max(abs(a), abs(b), 1.0)


def main():
    pandas_doc, bandal_doc = load(sys.argv[1]), load(sys.argv[2])
    pandas = {(r["rows"], r["workload"]): r for r in pandas_doc["results"]}
    bandal = {(r["rows"], r["workload"], r["mode"]): r for r in bandal_doc["results"]}

    print(f"pandas : {pandas_doc['backend']}")
    print(f"Bandal : {bandal_doc['backend']}")
    print("Times are median milliseconds of the computation only (lower is better).")
    print("Speedup = pandas / Bandal (> 1.00x means Bandal is faster).\n")

    header = ["workload", "pandas"]
    header += [f"{LABELS[m]} ms" for m in MODES] + [f"{LABELS[m]} x" for m in MODES] + ["match"]
    mismatches = 0
    for rows in sorted({key[0] for key in pandas}):
        print(f"== {rows:,} rows ==")
        table = [header]
        for (r, workload), reference in pandas.items():
            if r != rows:
                continue
            row = [workload, f"{reference['median_ms']:.2f}"]
            times, speedups, ok = [], [], True
            for mode in MODES:
                entry = bandal.get((rows, workload, mode))
                if entry is None:
                    times.append("-")
                    speedups.append("-")
                    continue
                times.append(f"{entry['median_ms']:.2f}")
                speedups.append(f"{reference['median_ms'] / entry['median_ms']:.2f}x")
                ok &= close(reference["checksum"], entry["checksum"], TOLERANCE[mode])
            mismatches += 0 if ok else 1
            table.append(row + times + speedups + ["ok" if ok else "DIFF"])
        widths = [max(len(line[i]) for line in table) for i in range(len(header))]
        for index, line in enumerate(table):
            print("  ".join(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
                            for i, cell in enumerate(line)))
            if index == 0:
                print("  ".join("-" * w for w in widths))
        print()

    if mismatches:
        print(f"WARNING: {mismatches} workload(s) produced different checksums.")
        sys.exit(1)


if __name__ == "__main__":
    main()
