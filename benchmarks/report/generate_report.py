"""
generate_report.py: combine throughput.json, memory.json, sizes.json and
system_info.json into ONE standalone HTML file. Chart.js and all raw data are
embedded inline, so the output needs no server, network or sibling files --
which is why it is the one artifact meant to be committed under results/.

Methodology (corpus, best-of-N, per-process memory) lives in README.md, not
here. This report also never computes a single "N times faster" headline:
each row shows the measured times and the ratio to the fastest library in
that row, at that input size only.

Usage: generate_report.py RESULTS_DIR OUTPUT_HTML --chartjs-path PATH
"""
import argparse
import datetime
import json
import os

from jinja2 import Environment, FileSystemLoader

HERE = os.path.dirname(os.path.abspath(__file__))

LIB_LABELS = {
    "libspeech": "libspeech (numpy)",
    "libspeech_list": "libspeech (list API)",
    "librosa": "librosa",
    "audioflux": "audioflux",
    "scipy": "scipy",
    "soundfile": "soundfile",
}
LIB_COLORS = {
    "libspeech": "#e67225",
    "libspeech_list": "#f2b27d",
    "librosa": "#5b8cff",
    "audioflux": "#35d0ba",
    "scipy": "#b98bff",
    "soundfile": "#ffb454",
}
OP_LABELS = {
    "load": "Load WAV file",
    "resample": "Resample 44.1 kHz to 16 kHz",
    "stft": "STFT (n_fft 512, hop 128)",
    "mfcc": "MFCC (13 coeffs, 26 mels)",
}
OP_ORDER = ["load", "resample", "stft", "mfcc"]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_rows(cells, value_key):
    """Group cells into {op: [ {entry, duration_s, vals: {lib: number}} ]}."""
    by_op = {}
    for c in cells:
        if value_key not in c:
            continue
        rows = by_op.setdefault(c["op"], {})
        row = rows.setdefault(
            c["entry"], {"entry": c["entry"], "duration_s": c["duration_s"],
                         "sample_rate": c["sample_rate"], "vals": {}}
        )
        row["vals"][c["library"]] = c[value_key]
    out = {}
    for op, rows in by_op.items():
        ordered = sorted(rows.values(), key=lambda r: (r["sample_rate"], r["duration_s"]))
        for r in ordered:
            best = min(r["vals"].values()) if r["vals"] else None
            r["best"] = best
            r["ratios"] = {
                lib: (v / best if best and best > 0 else None) for lib, v in r["vals"].items()
            }
        out[op] = ordered
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_dir")
    p.add_argument("output")
    p.add_argument("--chartjs-path", required=True)
    args = p.parse_args()

    throughput = load(os.path.join(args.results_dir, "throughput.json")) or {"cells": []}
    memory = load(os.path.join(args.results_dir, "memory.json")) or {"points": []}
    sizes = load(os.path.join(args.results_dir, "sizes.json")) or {"rows": []}
    system = load(os.path.join(args.results_dir, "system_info.json")) or {}

    time_rows = build_rows(throughput["cells"], "seconds_min")
    mem_rows = build_rows(memory["points"], "peak_extra_mib")
    errors = [
        f"{c['op']}/{c['library']}/{c['entry']}: {c['error']}"
        for c in throughput["cells"] + memory["points"] if "error" in c
    ]

    libs = sorted({c["library"] for c in throughput["cells"]},
                  key=lambda l: list(LIB_LABELS).index(l) if l in LIB_LABELS else 99)

    with open(args.chartjs_path, encoding="utf-8") as f:
        chartjs = f.read()

    env = Environment(loader=FileSystemLoader(HERE), autoescape=True)
    html = env.get_template("template.html.jinja2").render(
        generated=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        chartjs=chartjs,
        libs=libs,
        lib_labels=LIB_LABELS,
        lib_colors=LIB_COLORS,
        op_labels=OP_LABELS,
        op_order=[o for o in OP_ORDER if o in time_rows or o in mem_rows],
        time_rows=time_rows,
        mem_rows=mem_rows,
        sizes=sizes["rows"],
        versions=throughput.get("versions", {}),
        parameters=throughput.get("parameters", {}),
        repeats=throughput.get("repeats"),
        skipped=throughput.get("skipped", {}),
        system=system,
        errors=errors,
        chart_json=json.dumps({
            "time": time_rows, "memory": mem_rows,
            "labels": LIB_LABELS, "colors": LIB_COLORS, "ops": OP_LABELS,
        }),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"report: wrote {args.output}")


if __name__ == "__main__":
    main()
