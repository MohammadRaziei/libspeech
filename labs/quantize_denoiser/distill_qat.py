"""Teacher-student quantization-aware fine-tuning of a Facebook denoiser (dns48, dns64, master64).

The teacher is the fp32 model. The student starts as a copy whose Conv / ConvTranspose / LSTM weights are fake-quantized
to `--bits` on every forward pass (see weight_qat.py), and is trained to reproduce the teacher's output on speech + noise
mixtures. No clean targets and none of the original training data are needed: the teacher's output is the label, so
any speech and noise recordings will do (synthetic ones if you have nothing, with a quality warning).

Result in `--out`:
  student_state.pt          Demucs state_dict with the weights already on the integer grid
  student_grid_fp32.onnx    fp32 graph holding those weights; feed it to `onnx_ptq.py quantize`, which then
                            reproduces the same int8 weights exactly and calibrates the activations
  report.json               agreement with the teacher before and after training

Usage:
  python distill_qat.py --model dns48 --speech-dir DATA/speech --noise-dir DATA/noise --device cuda --out runs/dns48
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from audio_metrics import snr_db
from cli_output import say
from denoiser_model import export_onnx, load_model, verify_export
from mixtures import MixtureSource, NoisyRecordings
from weight_qat import quantize_weights, snap_weights

STFT_RESOLUTIONS = ((512, 128), (1024, 256), (2048, 512))
EVALUATION_SEED = 1234


def stft_loss(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Multi-resolution spectral convergence + log-magnitude L1, averaged over the resolutions. Inputs: (batch, time)."""
    total = estimate.new_zeros(())
    for n_fft, hop in STFT_RESOLUTIONS:
        window = torch.hann_window(n_fft, device=estimate.device)
        est = torch.stft(estimate, n_fft, hop, window=window, return_complex=True).abs().clamp_min(1e-7)
        ref = torch.stft(reference, n_fft, hop, window=window, return_complex=True).abs().clamp_min(1e-7)
        convergence = torch.linalg.norm(ref - est) / torch.linalg.norm(ref)
        log_magnitude = torch.nn.functional.l1_loss(est.log(), ref.log())
        total = total + convergence + log_magnitude
    return total / len(STFT_RESOLUTIONS)


def distillation_loss(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    est, ref = estimate.squeeze(1), reference.squeeze(1)
    return torch.nn.functional.l1_loss(est, ref) + stft_loss(est, ref)


@torch.no_grad()
def agreement_db(student, teacher, mixtures: np.ndarray, device: str) -> float:
    """Mean dB of agreement between the student's and the teacher's output on `mixtures`."""
    values = []
    for mix in mixtures:
        x = torch.from_numpy(mix).reshape(1, 1, -1).to(device)
        values.append(snr_db(student(x).cpu().numpy(), teacher(x).cpu().numpy()))
    return float(np.mean(values))


def learning_rate(step: int, steps: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


def distill(
    teacher,
    source: MixtureSource,
    *,
    bits=8,
    steps=3000,
    batch=8,
    lr=2e-5,
    warmup=100,
    lstm_axis=None,
    keep_first_last=False,
    device="cpu",
    eval_mixtures=None,
    log_every=100,
):
    """Returns (student, report). The student is returned with its weights snapped onto the integer grid."""
    teacher = teacher.to(device).eval().requires_grad_(False)
    student = copy.deepcopy(teacher).requires_grad_(True)
    wrapped = quantize_weights(student, bits, lstm_axis, keep_first_last)
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    report = {"bits": bits, "wrapped_modules": wrapped, "steps": steps}
    if eval_mixtures is not None:
        report["agreement_db_before"] = agreement_db(student, teacher, eval_mixtures, device)
        say(f"before training (round-to-nearest, {bits}-bit weights): {report['agreement_db_before']:.2f} dB vs teacher")
    started = time.time()
    losses = []
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps, lr, warmup)
        mix, _ = source.batch(batch)
        x = torch.from_numpy(mix).unsqueeze(1).to(device)
        with torch.no_grad():
            target = teacher(x)
        loss = distillation_loss(student(x), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % log_every == 0 or step == 0:
            recent = float(np.mean(losses[-log_every:]))
            say(f"step {step + 1}/{steps}  loss {recent:.4f}  {time.time() - started:.0f} s")
    window = max(1, min(100, steps // 2))  # average over the first and last `window` steps (at most 100)
    report["loss_first"] = float(np.mean(losses[:window]))
    report["loss_last"] = float(np.mean(losses[-window:]))
    student.eval()
    if eval_mixtures is not None:
        report["agreement_db_after"] = agreement_db(student, teacher, eval_mixtures, device)
        say(f"after training: {report['agreement_db_after']:.2f} dB vs teacher")
    snap_weights(student)
    if eval_mixtures is not None:
        report["agreement_db_after_snap"] = agreement_db(student, teacher, eval_mixtures, device)
    return student, report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dns48", help="dns48 | dns64 | master64 | path/to/model.th | random:<hidden>")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--noisy-dir", help="folder of recordings that are noisy already; used as they are (overrides the two below)")
    p.add_argument("--speech-dir", help="folder of clean speech; synthetic speech if omitted")
    p.add_argument("--noise-dir", help="folder of noise recordings; synthetic noise if omitted")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bits", type=int, default=8, help="weight bits; onnxruntime's CPU kernels run 8")
    p.add_argument("--lstm-per-row", action="store_true", help="per-row LSTM weight scales instead of per tensor")
    p.add_argument("--keep-first-last", action="store_true", help="leave the first conv and the last transposed conv in fp32")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seconds", type=float, default=3.0, help="length of each training mixture")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--eval-clips", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if not (args.noisy_dir or (args.speech_dir and args.noise_dir)):
        say("WARNING: synthetic speech/noise in use; this only checks that the pipeline runs, not the quality it gives.")
    teacher = load_model(args.model, args.device)

    def source(seconds, seed):
        if args.noisy_dir:
            return NoisyRecordings(args.noisy_dir, seconds=seconds, seed=seed)
        return MixtureSource(args.speech_dir, args.noise_dir, seconds=seconds, seed=seed)

    train_source = source(args.seconds, args.seed)
    eval_mixtures, _ = source(4.0, EVALUATION_SEED).batch(args.eval_clips)
    student, report = distill(
        teacher,
        train_source,
        bits=args.bits,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        lstm_axis=0 if args.lstm_per_row else None,
        keep_first_last=args.keep_first_last,
        device=args.device,
        eval_mixtures=eval_mixtures,
    )
    student = student.cpu().eval()
    torch.save(student.state_dict(), out / "student_state.pt")
    export_onnx(student, out / "student_grid_fp32.onnx")
    report["export_max_relative_error"] = verify_export(student, out / "student_grid_fp32.onnx")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    say(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
