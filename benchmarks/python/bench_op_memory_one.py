"""
bench_op_memory_one.py: peak extra memory of ONE operation, in a fresh process.

Called by bench_op_memory_driver.py, once per (operation, library, entry).
Must be its own process: a peak-RSS number is a high-water mark, so anything
that ran earlier in the same interpreter (imports, other libraries' work)
would contaminate it.

Method (Linux): imports and input preparation happen first; then the
kernel's peak-RSS counter (VmHWM) is reset via /proc/self/clear_refs, the
operation runs once, and the reported figure is VmHWM afterwards minus the
RSS right before the call -- i.e. the extra memory the call itself needed on
top of an already-loaded interpreter, library and input.
Fallback where clear_refs is unavailable (macOS, restricted containers):
ru_maxrss after minus before, which reads 0 whenever the call peaks lower
than the import phase did; the `method` field says which one was used.

Usage: bench_op_memory_one.py OP LIBRARY WAV_PATH OUTPUT_JSON
"""
import argparse
import gc
import json
import resource
import sys

import ops


def _status_kib(field):
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise KeyError(field)


def _reset_peak():
    try:
        with open("/proc/self/clear_refs", "w", encoding="utf-8") as f:
            f.write("5")
        return True
    except OSError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("op")
    p.add_argument("library")
    p.add_argument("wav_path")
    p.add_argument("output_json")
    args = p.parse_args()

    x, sr = ops.read_samples(args.wav_path)
    run = ops.make_runner(args.op, args.library, x, sr, args.wav_path)
    gc.collect()

    if sys.platform.startswith("linux") and _reset_peak():
        method = "vmhwm_reset"
        before = _status_kib("VmRSS")
        out = run()
        after = _status_kib("VmHWM")
        delta_kib = max(after - before, 0)
    else:
        method = "ru_maxrss"
        scale = 1 if sys.platform.startswith("linux") else 1 / 1024  # macOS reports bytes
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
        out = run()
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
        delta_kib = max(after - before, 0)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({
            "op": args.op, "library": args.library,
            "peak_extra_mib": round(delta_kib / 1024.0, 3),
            "method": method, "output": ops.describe(out),
        }, f)


if __name__ == "__main__":
    main()
