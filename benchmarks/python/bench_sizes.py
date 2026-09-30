"""
bench_sizes.py: real download size of each package, with and without its
dependency tree, via the pip-size CLI (resolves against PyPI, downloads
nothing). Sizes are of the latest PyPI release of each package, not of the
local checkout.

Usage: bench_sizes.py OUTPUT_JSON
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

PACKAGES = ["libspeech", "librosa", "audioflux", "scipy", "soundfile"]


def pip_size(pkg, no_deps):
    exe = shutil.which("pip-size") or os.path.join(os.path.dirname(sys.executable), "pip-size")
    cmd = [exe, pkg, "--json", "--bytes", "--quiet"]
    if no_deps:
        cmd.append("--no-deps")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    text = proc.stdout
    start = text.find("{")
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    args = p.parse_args()

    rows = []
    for pkg in PACKAGES:
        own = pip_size(pkg, True)
        full = pip_size(pkg, False)
        rows.append({
            "package": pkg,
            "version": (own or full or {}).get("version"),
            "own_bytes": int(own["size"]) if own else None,
            "total_bytes": int(full["total_size"]) if full else None,
        })
        print(f"sizes: {pkg}: own={rows[-1]['own_bytes']} total={rows[-1]['total_bytes']}", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
