# Quantizing the Facebook denoisers (dns48, dns64, master64)

**Start with the notebook:** [`quantize_denoiser.ipynb`](quantize_denoiser.ipynb)
([open in Colab](https://colab.research.google.com/github/MohammadRaziei/libspeech/blob/master/labs/quantize_denoiser/quantize_denoiser.ipynb)).
It runs the two routes below and, optionally (`RUN_LIBSPEECH = True`), runs the results through libspeech itself and lets you listen. `lab_helpers.py` holds
its helper functions. Setting `SMOKE = True` in the notebook runs everything on a tiny random model in about a minute.

Tools to make int8 versions of the three `facebook-denoiser-*` ONNX models and to measure what the quantization costs.
They are development tools, not part of the wheel. Everything is measured against the fp32 model's own output (the
teacher), because that is the one thing always available; with real clean speech the SI-SDR against the clean signal is
reported too.

```
pip install numpy scipy soundfile onnx onnxruntime        # onnx_ptq.py: CPU only
pip install torch && pip install --no-deps denoiser       # distill_qat.py: PyTorch (the denoiser package's own
                                                          # dependencies are not needed; --no-deps avoids them)
```

## What the model is made of (dns48, counted from the exported graph)

| part | weights | share |
|---|---|---|
| LSTM (2 layers) | 9.44 M | 50.1 % |
| Conv, kernel 1 | 3.14 M | 16.7 % |
| Conv, kernel 8, stride 4 | 3.13 M | 16.6 % |
| ConvTranspose, kernel 8, stride 4 | 3.13 M | 16.6 % |

plus four fixed convolutions (kernel 112) that are the resampling filters. There are no `BatchNormalization` nodes.

## Two routes

**1. Post-training, CPU only (`onnx_ptq.py`).** Static QDQ int8 for the convolutions (per-channel int8 weights, uint8
activations calibrated on your audio) and dynamic int8 for the LSTM. `sensitivity` quantizes one node at a time and shows
which layers hurt most, so they can be kept in fp32 (`quantize --sensitivity sensitivity.json --keep-fp32 K`).

```
python onnx_ptq.py sensitivity dns48.onnx --noisy-dir recordings/ --out sensitivity.json
python onnx_ptq.py quantize    dns48.onnx dns48_int8.onnx --noisy-dir recordings/ --sensitivity sensitivity.json --keep-fp32 3
python onnx_ptq.py evaluate    dns48.onnx dns48_int8.onnx --noisy-dir recordings/ --clips 32
```

**2. Teacher-student fine-tuning (`distill_qat.py`, needs PyTorch, a GPU is much faster).** The student is a copy of the
fp32 model whose weights are fake-quantized on every forward pass; it is trained to reproduce the teacher's output on
mixtures. The teacher's output is the label, so no clean targets and none of the original training data are needed.
The result is an fp32 ONNX whose weights already lie on the int8 grid; `onnx_ptq.py quantize` then reproduces exactly
those int8 weights (the scale rule is the one onnxruntime uses, and a test checks it) and only calibrates activations.

```
python distill_qat.py --model dns48 --noisy-dir recordings/ --device cuda --steps 3000 --out runs/dns48
python onnx_ptq.py quantize runs/dns48/student_grid_fp32.onnx runs/dns48/dns48_int8.onnx --noisy-dir recordings/
python onnx_ptq.py evaluate dns48.onnx runs/dns48/dns48_int8.onnx --noisy-dir recordings/ --clips 32
```

`--model` is `dns48`, `dns64`, `master64` (the `denoiser` package downloads the PyTorch weights from Facebook), a `.th`
file, or `random:<hidden>` for a dry run. `student_grid_fp32.onnx` is exported with a length-independent graph
(input `input`, output `output`, opset 13), and `report.json` records the export check against PyTorch.

## Data

Any of these, best first: `--noisy-dir` (real recordings that are noisy already; used as they are), `--speech-dir` +
`--noise-dir` (clean speech mixed with noise at random SNRs between -5 and 20 dB), or nothing (synthetic speech and noise:
only good for checking that the pipeline runs). Files may be .wav, .flac, .ogg or .mp3 at any sample rate.
Calibration and evaluation clips are drawn with different seeds, but from the same files; hold some recordings out if you
want the evaluation to be independent of the calibration.

## What is not here, and why

* **int4 / 1.58-bit.** onnxruntime's int4 path (`MatMulNBits`) covers MatMul and Gemm only, not Conv, ConvTranspose or
  LSTM, and it has no ternary kernel. `distill_qat.py --bits 4` runs, but the exported model can only be stored as int8, so it
  gains no speed and no size over 8 bits; use it to study the accuracy cost, not to ship.
* **Activation quantization-aware training.** Only weights are fake-quantized during fine-tuning; activation ranges come from
  onnxruntime's calibration afterwards.
* **Listening tests.** The numbers are dB of agreement with fp32, not a perceptual score.

## Measured so far

`dns48`, two real noisy recordings (`alex-noisy.mp3`, `audio-fa-noisy.mp3` from libspeech's `resources` release), static
int8 from `onnx_ptq.py quantize` (percentile calibration on 8 clips of 4 s), 1 thread, one Linux sandbox CPU. Quality is
agreement with the fp32 model's output; the clips used to calibrate and to evaluate come from the same two recordings.

| | fp32 | int8 |
|---|---|---|
| `onnxruntime` 1.24.4, 8 clips of 4 s: agreement | | 25.8 dB mean, 24.5 dB min |
| same: real-time factor | 0.171 | 0.137 |
| same: size | 75.5 MB | 19.2 MB |
| libspeech's bundled ONNX Runtime, one 6 s clip: agreement | | 25.1 dB |
| same: real-time factor | 0.171 | 0.262 |

So the int8 model is smaller and, with a recent `onnxruntime`, faster, but with the ONNX Runtime bundled in libspeech it
ran slower than fp32 in this one measurement (a single 6 s clip, one run after a warm-up call). Nothing here was
listened to, and the fine-tuning route (`distill_qat.py`) has only been run on a tiny random model so far, not on a real
checkpoint.

`libspeech.Denoiser(backend, model)` should take a local path (ADR-0021 item 7 says so), but in the libspeech source this lab
was written against anything that is not an http(s) URL is taken as a file name of the models release, so a local path
fails with a 404 (see the checklist). `lab_helpers.register_in_libspeech` works around it by copying a model into
libspeech's cache with the `.sha256` and `.readme` files libspeech requires.
