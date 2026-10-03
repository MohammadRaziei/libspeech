"""
bench_op_memory_driver.py: peak extra memory for load / resample / stft / mfcc,
every library, every applicable corpus entry.

This driver never decodes audio or imports any benchmarked library itself: it
only reads manifest metadata and spawns bench_op_memory_one.py as a fresh
child per data point, so no measurement inherits its parent's memory history
(see that script's docstring for why that matters).

Usage: bench_op_memory_driver.py MANIFEST [MANIFEST ...] OUTPUT_JSON
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# Duplicated from ops.py on purpose: importing ops would pull in numpy only,
# but keeping the driver free of any benchmark code is the point.
OPS = ("load", "resample", "stft", "mfcc")
LIBS_FOR_OP = {
    "load": ("libspeech", "libspeech_list", "librosa", "soundfile"),
    "resample": ("libspeech", "libspeech_list", "librosa", "audioflux", "scipy"),
    "stft": ("libspeech", "libspeech_list", "librosa", "audioflux"),
    "mfcc": ("libspeech", "libspeech_list", "librosa", "audioflux"),
}
TARGET_RATE = 16000


def applies(op, sample_rate):
    if op == "load":
        return True
    if op == "resample":
        return sample_rate != TARGET_RATE
    return sample_rate == TARGET_RATE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+")
    args = p.parse_args()
    manifests, output = args.paths[:-1], args.paths[-1]

    entries = []
    for m in manifests:
        with open(m, encoding="utf-8") as f:
            entries.extend(json.load(f)["entries"])

    points = []
    with tempfile.TemporaryDirectory() as tmp:
        for op in OPS:
            for e in entries:
                if not applies(op, e["sample_rate"]):
                    continue
                for lib in LIBS_FOR_OP[op]:
                    out_json = os.path.join(tmp, f"{op}_{lib}_{e['name']}.json")
                    proc = subprocess.run(
                        [sys.executable, os.path.join(HERE, "bench_op_memory_one.py"),
                         op, lib, e["path"], out_json],
                        capture_output=True, text=True, check=False,
                    )
                    point = {"op": op, "library": lib, "entry": e["name"],
                             "duration_s": e["duration_s"], "sample_rate": e["sample_rate"]}
                    if proc.returncode == 0 and os.path.exists(out_json):
                        with open(out_json, encoding="utf-8") as f:
                            point.update(json.load(f))
                    else:
                        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["failed"]
                        point["error"] = tail[0]
                        print(f"  ! {op}/{lib}/{e['name']}: {point['error']}", file=sys.stderr)
                    points.append(point)
                    print(f"memory: {op}/{lib}/{e['name']}", file=sys.stderr, flush=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump({"points": points}, f, indent=2)
    print(f"memory: {len(points)} points -> {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
