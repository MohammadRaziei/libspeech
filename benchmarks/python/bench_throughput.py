"""
bench_throughput.py: wall-clock time of load / resample / stft / mfcc for
every library that is installed, on every applicable corpus entry.

Each (operation, library, corpus entry) cell is repeated REPEATS times after
one untimed warm-up call; the best (min) time is kept -- the standard
micro-benchmark convention, closest a single run gets to "nothing else
interrupted this one" -- and the median is recorded next to it. GC is paused
inside the timed region. What each operation means is defined once, in
ops.py (also used by the memory and scaling benchmarks).

Usage: bench_throughput.py MANIFEST [MANIFEST ...] OUTPUT_JSON [--repeats N]
"""
from __future__ import annotations

# numpy is imported FIRST, deliberately (an isort-sorted file would put it after the stdlib
# block): its import cost, lazy sub-imports and thread-pool start-up then happen before anything
# is timed or memory-measured, and libspeech's NumPy-returning calls find it already loaded
# instead of importing it inside the measured call.
import numpy as np  # noqa: F401  (imported for its side effects: see above)

# isort: split

import argparse
import json
import sys

import ops

try:
    from tqdm import tqdm
except ImportError:  # tqdm is a convenience, not a requirement
    def tqdm(iterable, **kwargs):
        return iterable


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="corpus manifest(s), then the output JSON path")
    p.add_argument("--repeats", type=int, default=7)
    args = p.parse_args()
    manifests, output = args.paths[:-1], args.paths[-1]
    if not manifests:
        p.error("need at least one manifest before the output path")

    entries = []
    for m in manifests:
        with open(m, encoding="utf-8") as f:
            entries.extend(json.load(f)["entries"])

    skipped = {}
    cells = []
    work = []
    for op in ops.OPS:
        for e in entries:
            if not ops.op_applies(op, e["sample_rate"]):
                continue
            for lib in ops.LIBS_FOR_OP[op]:
                if not ops.library_available(lib):
                    skipped[lib] = "not installed"
                    continue
                work.append((op, lib, e))

    cache = {}
    for op, lib, e in tqdm(work, desc="throughput"):
        if e["path"] not in cache:
            cache.clear()  # keep at most one decoded signal alive
            cache[e["path"]] = ops.read_samples(e["path"])
        x, sr = cache[e["path"]]
        cell = {
            "op": op, "library": lib, "entry": e["name"],
            "duration_s": e["duration_s"], "sample_rate": e["sample_rate"],
        }
        try:
            run = ops.make_runner(op, lib, x, sr, e["path"])
            best, median, out = ops.time_runner(run, args.repeats)
            cell.update(seconds_min=best, seconds_median=median, output=ops.describe(out))
        except Exception as exc:  # a broken cell must not sink the whole sweep
            cell["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ! {op}/{lib}/{e['name']}: {cell['error']}", file=sys.stderr)
        cells.append(cell)

    versions = {lib: ops.library_version(lib) for lib in sorted({c["library"] for c in cells})}
    try:  # which kernel set libspeech ran: 'avx2+fma', 'neon' or 'generic'
        import libspeech

        simd_backend = libspeech.simd_backend()
    except (ImportError, AttributeError):
        simd_backend = None
    with open(output, "w", encoding="utf-8") as f:
        json.dump({
            "repeats": args.repeats,
            "parameters": {
                "n_fft": ops.N_FFT, "hop": ops.HOP, "n_mels": ops.N_MELS,
                "n_mfcc": ops.N_MFCC, "resample_target_hz": ops.RESAMPLE_TARGET,
            },
            "versions": versions,
            "libspeech_simd_backend": simd_backend,
            "skipped": skipped,
            "cells": cells,
        }, f, indent=2)
    print(f"throughput: {len(cells)} cells -> {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
