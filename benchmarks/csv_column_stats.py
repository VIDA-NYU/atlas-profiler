"""Benchmark the whole-file column statistics of a CSV path: one scan per column (current) against
one scan per group of columns.

    python benchmarks/csv_column_stats.py data.csv [--groups 1 16 64 0]

Group size 1 is the current strategy; 0 means all columns in one scan. Each run is a separate process,
so the reported peak memory (maximum resident set size) is that run's own.
"""

import argparse
import json
import resource
import subprocess
import sys
import time

RUN = r'''
import json, resource, sys, time
import polars as pl
from profiler.core import CSV_SAMPLE_ROWS, _collect_streaming, _csv_column_stats, _scan_csv

path, group = sys.argv[1], int(sys.argv[2])
sep = "\t" if path.endswith(".tsv") else ","
names = _scan_csv(pl, path, sep).collect_schema().names()
t = time.perf_counter()
if group == 1:
    for name in names:
        _csv_column_stats(pl, path, sep, name)
else:
    size = group or len(names)
    for i in range(0, len(names), size):
        exprs = []
        for j, name in enumerate(names[i:i + size]):
            value = pl.col(name).str.strip_chars()
            value = pl.when(value.is_null() | (value == "")).then(None).otherwise(value)
            numeric = value.cast(pl.Float64, strict=False)
            numeric = pl.when(numeric.is_finite() & (numeric.abs() < 3.4e38)).then(numeric)
            exprs += [
                value.filter(value.is_not_null()).approx_n_unique().alias(f"d{j}"),
                numeric.count().alias(f"c{j}"), numeric.min().alias(f"lo{j}"), numeric.max().alias(f"hi{j}"),
                numeric.mean().alias(f"m{j}"), numeric.std(ddof=0).alias(f"s{j}"),
                value.filter(value.is_not_null()).head(CSV_SAMPLE_ROWS).implode().alias(f"v{j}"),
            ]
        _collect_streaming(_scan_csv(pl, path, sep).select(exprs))
elapsed = time.perf_counter() - t
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mb = peak / 1e6 if sys.platform == "darwin" else peak / 1e3   # bytes on macOS, KiB on Linux
print(json.dumps({"columns": len(names), "group": group, "seconds": elapsed, "peak_mb": peak_mb}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 16, 64, 0])
    args = parser.parse_args()
    for group in args.groups:
        out = subprocess.run([sys.executable, "-c", RUN, args.csv, str(group)], capture_output=True, text=True, check=True)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        label = "one scan per column" if group == 1 else ("all columns, one scan" if group == 0 else f"{group} columns per scan")
        print(f"{label:24s} {r['columns']:4d} columns  {r['seconds']:7.1f} s  peak {r['peak_mb']:8.0f} MB")


if __name__ == "__main__":
    main()
