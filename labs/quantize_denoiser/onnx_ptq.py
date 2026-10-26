"""Post-training int8 quantization of the Facebook denoiser ONNX models (dns48, dns64, master64), with the tooling around it.

Sub-commands (all run on CPU, no PyTorch needed):

  sensitivity  quantize one Conv / ConvTranspose node at a time and measure how far the output moves from fp32,
               to see which layers should stay in fp32
  quantize     static QDQ int8 for Conv / ConvTranspose (per-channel weights, calibrated activations) and dynamic int8 for the
               LSTM, optionally leaving the most sensitive nodes in fp32; `--lstm-only` quantizes just the LSTM
  evaluate     compare a candidate model with the fp32 original: agreement in dB, speed (real-time factor), size

Quality is always measured against the fp32 model's own output (the teacher), on speech + noise mixtures; with real clean
speech the SI-SDR against the clean signal is reported too. Calibration and evaluation mixtures come from different seeds.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from audio_metrics import si_sdr, snr_db
from cli_output import say
from mixtures import MixtureSource, NoisyRecordings
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

QUANT_OPS = ("Conv", "ConvTranspose")
METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}
CALIBRATION_SEED = 0
EVALUATION_SEED = 1234


class MixtureReader(CalibrationDataReader):
    """Feeds mixtures (1-D float32 arrays) to the calibrator as (1, 1, length) inputs."""

    def __init__(self, mixtures, input_name="input"):
        self._batches = [{input_name: np.asarray(m, dtype=np.float32).reshape(1, 1, -1)} for m in mixtures]
        self.rewind()

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._batches)


def make_session(path, threads=1) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run_clip(session: ort.InferenceSession, mixture: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": mixture.reshape(1, 1, -1).astype(np.float32)})[0].reshape(-1)


def with_node_names(src, dst) -> list[str]:
    """Copy `src` to `dst` giving every unnamed node a unique name (ORT needs names to include or exclude nodes).

    Returns the names of the Conv / ConvTranspose nodes, in graph order.
    """
    model = onnx.load(str(src))
    used = {n.name for n in model.graph.node if n.name}
    for i, node in enumerate(model.graph.node):
        if not node.name:
            name = f"{node.op_type}_{i}"
            while name in used:
                name += "_"
            node.name = name
            used.add(name)
    onnx.save(model, str(dst))
    return [n.name for n in model.graph.node if n.op_type in QUANT_OPS]


def quantize(
    src,
    dst,
    calibration_mixtures,
    *,
    exclude=(),
    method="percentile",
    per_channel=True,
    lstm="dynamic",
    reduce_range=False,
    activation_symmetric=False,
) -> None:
    """Static QDQ int8 for the convolutions (weights int8 symmetric, activations uint8), dynamic int8 for the LSTM."""
    dst = Path(dst)
    with tempfile.TemporaryDirectory(dir=dst.parent) as tmp:
        named = Path(tmp) / "named.onnx"
        with_node_names(src, named)
        static_out = Path(tmp) / "static.onnx" if lstm == "dynamic" else dst
        quantize_static(
            str(named),
            str(static_out),
            MixtureReader(calibration_mixtures),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=list(QUANT_OPS),
            per_channel=per_channel,
            reduce_range=reduce_range,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=list(exclude),
            calibrate_method=METHODS[method],
            extra_options={"ActivationSymmetric": activation_symmetric, "WeightSymmetric": True},
        )
        if lstm == "dynamic":
            quantize_dynamic(str(static_out), str(dst), op_types_to_quantize=["LSTM"], weight_type=QuantType.QInt8)


def quantize_lstm_only(src, dst) -> None:
    """Dynamic int8 for the LSTM only; the convolutions stay fp32. Needs no calibration data."""
    quantize_dynamic(str(src), str(dst), op_types_to_quantize=["LSTM"], weight_type=QuantType.QInt8)


def reference_outputs(ref_session, mixtures) -> list[np.ndarray]:
    return [run_clip(ref_session, m) for m in mixtures]


def sensitivity(src, calibration_mixtures, eval_mixtures, *, method, per_channel, workdir, limit=None) -> list[dict]:
    """For every Conv / ConvTranspose node and for the LSTM: agreement with fp32 when only that part is quantized.

    Lower agreement = more sensitive. Sorted most sensitive first. `limit` tests only the first N convolution nodes (and
    the LSTM), for a quick dry run.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    named = workdir / "named.onnx"
    names = with_node_names(src, named)
    ref = make_session(named)
    refs = reference_outputs(ref, eval_mixtures)
    rows = []
    trials = [(n, set(names) - {n}, "fp32") for n in names[:limit]] + [("LSTM", set(names), "dynamic")]
    for i, (label, exclude, lstm) in enumerate(trials):
        out = workdir / f"trial_{i}.onnx"
        quantize(named, out, calibration_mixtures, exclude=exclude, method=method, per_channel=per_channel, lstm=lstm)
        session = make_session(out)
        agreement = [snr_db(run_clip(session, m), r) for m, r in zip(eval_mixtures, refs)]
        rows.append({"node": label, "mean_snr_vs_fp32_db": float(np.mean(agreement)), "min_snr_vs_fp32_db": float(np.min(agreement))})
        say(f"{label:60s} mean {rows[-1]['mean_snr_vs_fp32_db']:7.2f} dB   min {rows[-1]['min_snr_vs_fp32_db']:7.2f} dB")
        del session
        out.unlink()
    rows.sort(key=lambda r: r["mean_snr_vs_fp32_db"])
    return rows


def evaluate(ref_path, cand_path, mixtures, cleans=None, threads=1) -> dict:
    """Agreement with the fp32 model, quality against clean speech if available, speed and size."""
    ref, cand = make_session(ref_path, threads), make_session(cand_path, threads)
    run_clip(ref, mixtures[0])
    run_clip(cand, mixtures[0])  # warm-up, not timed
    t_ref = t_cand = seconds = 0.0
    agreement, sdr_ref, sdr_cand = [], [], []
    for i, mix in enumerate(mixtures):
        t0 = time.perf_counter()
        y_ref = run_clip(ref, mix)
        t1 = time.perf_counter()
        y_cand = run_clip(cand, mix)
        t2 = time.perf_counter()
        t_ref += t1 - t0
        t_cand += t2 - t1
        seconds += len(mix) / 16000
        agreement.append(snr_db(y_cand, y_ref))
        if cleans is not None:
            sdr_ref.append(si_sdr(y_ref, cleans[i]))
            sdr_cand.append(si_sdr(y_cand, cleans[i]))
    report = {
        "clips": len(mixtures),
        "mean_snr_vs_fp32_db": float(np.mean(agreement)),
        "min_snr_vs_fp32_db": float(np.min(agreement)),
        "rtf_fp32": t_ref / seconds,
        "rtf_candidate": t_cand / seconds,
        "size_fp32_mb": Path(ref_path).stat().st_size / 1e6,
        "size_candidate_mb": Path(cand_path).stat().st_size / 1e6,
        "threads": threads,
    }
    if cleans is not None:
        report["si_sdr_clean_fp32_db"] = float(np.mean(sdr_ref))
        report["si_sdr_clean_candidate_db"] = float(np.mean(sdr_cand))
    return report


def _source(args, seed):
    if args.noisy_dir:
        return NoisyRecordings(args.noisy_dir, seconds=args.seconds, seed=seed)
    return MixtureSource(args.speech_dir, args.noise_dir, seconds=args.seconds, seed=seed)


def _add_data_args(p):
    p.add_argument("--noisy-dir", help="folder of recordings that are noisy already; used as they are (overrides the two below)")
    p.add_argument("--speech-dir", help="folder of clean speech (.wav/.flac/.ogg/.mp3); synthetic speech if omitted")
    p.add_argument("--noise-dir", help="folder of noise recordings; synthetic noise if omitted")
    p.add_argument("--seconds", type=float, default=4.0, help="length of each mixture")
    p.add_argument("--calib-clips", type=int, default=16)
    p.add_argument("--method", choices=sorted(METHODS), default="percentile", help="activation calibration")
    p.add_argument("--per-tensor", action="store_true", help="per-tensor instead of per-channel conv weights")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sensitivity", help="per-node quantization sensitivity")
    s.add_argument("model")
    s.add_argument("--eval-clips", type=int, default=8)
    s.add_argument("--workdir", default="sensitivity_work")
    s.add_argument("--out", default="sensitivity.json")
    _add_data_args(s)

    q = sub.add_parser("quantize", help="static int8 quantization")
    q.add_argument("model")
    q.add_argument("output")
    q.add_argument("--exclude", nargs="*", default=[], help="node names to keep in fp32")
    q.add_argument("--sensitivity", help="sensitivity.json; the --keep-fp32 most sensitive nodes stay fp32")
    q.add_argument("--keep-fp32", type=int, default=0)
    q.add_argument("--lstm", choices=["dynamic", "fp32"], default="dynamic")
    q.add_argument("--lstm-only", action="store_true", help="int8 for the LSTM only, convolutions stay fp32 (no calibration)")
    q.add_argument("--reduce-range", action="store_true", help="7-bit activations (safer on x86 CPUs without VNNI)")
    _add_data_args(q)

    e = sub.add_parser("evaluate", help="compare a candidate with the fp32 model")
    e.add_argument("reference")
    e.add_argument("candidate")
    e.add_argument("--clips", type=int, default=16)
    e.add_argument("--threads", type=int, default=1)
    e.add_argument("--out", help="write the report as JSON here")
    _add_data_args(e)

    args = parser.parse_args(argv)

    if args.command == "sensitivity":
        calib, _ = _source(args, CALIBRATION_SEED).batch(args.calib_clips)
        evals, _ = _source(args, EVALUATION_SEED).batch(args.eval_clips)
        rows = sensitivity(args.model, calib, evals, method=args.method, per_channel=not args.per_tensor, workdir=args.workdir)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        say(f"wrote {args.out}; most sensitive first: {[r['node'] for r in rows[:5]]}")
    elif args.command == "quantize":
        if args.lstm_only:
            quantize_lstm_only(args.model, args.output)
            say(f"wrote {args.output}; LSTM only, convolutions in fp32")
            return
        exclude = list(args.exclude)
        if args.sensitivity and args.keep_fp32:
            rows = json.loads(Path(args.sensitivity).read_text())
            exclude += [r["node"] for r in rows if r["node"] != "LSTM"][: args.keep_fp32]
        calib, _ = _source(args, CALIBRATION_SEED).batch(args.calib_clips)
        quantize(
            args.model, args.output, calib, exclude=exclude, method=args.method, per_channel=not args.per_tensor,
            lstm=args.lstm, reduce_range=args.reduce_range,
        )
        say(f"wrote {args.output}; kept in fp32: {exclude or 'nothing'}")
    else:
        mixtures, cleans = _source(args, EVALUATION_SEED).batch(args.clips)
        report = evaluate(args.reference, args.candidate, mixtures, cleans if args.speech_dir and not args.noisy_dir else None, args.threads)
        say(json.dumps(report, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
