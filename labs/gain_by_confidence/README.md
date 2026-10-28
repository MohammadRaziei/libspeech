# gain_by_confidence

**Status: idea only. Nothing is implemented and nothing is measured.**

Question: can the input gain of a neural speech model (SileroVad first, then the denoisers) be chosen by the model's own
output confidence, instead of by a signal statistic such as RMS or LUFS?

## Idea

Search one scalar, the input gain `g` (for example -30 dB to +30 dB), and keep the gain at which the model's output is the most decisive.

* VAD objective: mean binary entropy of `SileroVad.speech_probabilities()` (lower is more decisive). Alternative: the fraction
  of probabilities in the undecided band 0.2 to 0.8.
* Denoiser objective: undecided. No mask is exposed by `Denoiser` (nothing in `include/libspeech/models/` or `src/models/`
  mentions a mask), so the first experiment is VAD only. A waveform-level proxy needs a decision from the owner.
* Search: golden-section on log `g`, about 10 model passes on a representative excerpt of about 10 s. No labels, no weight change.

Known risks to test, not assume: entropy can fall at a gain that makes the model confidently wrong (noise amplified into
"speech"); the optimum may coincide with RMS normalization; a flat objective gives an unstable gain.

## Prior art (searched 2026-10-27 with web search and the Consensus connector, a few queries, not a systematic review)

None of these selects the input gain from the model's own confidence. Closest hits:

| work | what it does | difference |
|---|---|---|
| SUTA / SGEM (Kim et al., arXiv 2306.01981) | entropy minimization at test time for ASR | adapts model parameters, not the input gain |
| Raichle et al., arXiv 2601.14770 | mask polarization as the adaptation signal for speech enhancement | adapts model weights |
| Mitra and Franco, arXiv 1708.09516 | activation entropy as a confidence measure to select data for unsupervised adaptation | selects data, not gain |
| Fujita and Iso, Interspeech 2016 | phone-posterior entropy to reject background-speech frames after a DNN VAD | rejects frames, no gain search |
| NeMo CHiME-7 system, arXiv 2310.12378 | picks the best of several front-end outputs per utterance with an ASR confidence measure | picks among beamformer variants, not a gain |
| Higuchi et al., Interspeech 2016 | optimizes enhancement front-end parameters with an ASR-level criterion | front-end parameters, needs the recognizer's score |
| Takeda et al., APSIPA ASC 2024, "Scale-invariant Online Voice Activity Detection under Various Environments" | neural VAD made amplitude invariant by block-wise normalization inside the feature extraction and by training; compared with Silero | removes the gain dependence by design and needs a trained model; no gain search. Same problem, different solution |
| Wang et al., Sensors 2022, "ASR Performance Improvement for Mandarin Based on Optimizing Gain Control Strategy" (MOST) | derives a gain control strategy that minimizes WER degradation, from a noise-figure model | model-based theory for an optimal gain for ASR, not a search driven by the model's output confidence. Closest on the "optimal gain for a neural back end" question |
| US 9,401,685 | AGC response reduced or delayed when a confidence metric is low | confidence gates the AGC, no search over gain |
| US 10,867,617 | AGC time constant depends on confidence | time constants, not gain selection |

Not used: the alphaXiv search was not approved, SciSpace and Wiley were not queried.

Result: no identical method found. This does not prove novelty. Do not claim it in code, docs or ADRs.

## Protocol (draft, to be agreed with the owner before any number is produced)

Needs clean speech and noise clips from the owner (the sandbox cannot download corpora). Mixtures at -5, 0, 5, 10, 20 dB SNR with
a known speech mask, input offsets from -30 to +30 dB. Compare: no normalization, RMS -20 dBFS, speech-weighted RMS, LUFS -23, and
the confidence-selected gain. Metric for the VAD: F1 against the known mask, mean with confidence interval. Also report the
selected gain next to the RMS gain, how often the search ends on a range boundary, and the cost in model passes.
Pass/fail thresholds: to be set by the owner before the run.

## First run (alex-noisy.mp3, 16.9 s, RMS -15.4 dBFS, SileroVad, gtcrn), 2026-10-27

Script: `gain_by_confidence.py`. One clip, no reference, so this says nothing about quality in general.

* Entropy is lowest at -28.4 dB (0.359), against 0.487 at 0 dB. The fraction of windows called speech falls from 0.44 to 0.33
  at the same time. The curve is not smooth (local minimum near -6 dB, another near 24 dB).
* The RMS rule would choose -4.6 dB. The entropy objective chose a gain 24 dB lower, so on this clip it does not coincide with RMS.
* Reading: the lower entropy comes with fewer speech windows, so the model may simply be more certain that it hears no speech.
  This is the "confidently wrong" risk from above; without speech labels it cannot be confirmed or ruled out.
* gtcrn output with and without the selected gain (gain undone afterwards) agree to 13.0 dB SNR, so the outputs do differ.
  Which one is better was not measured (no clean reference); it needs listening.

## Second run (audio-fa-noisy.mp3, 499.5 s, RMS -16.5 dBFS, SileroVad, gtcrn), 2026-10-28

* Smooth curve with one minimum: entropy 0.171 at -1.8 dB, 0.176 at 0 dB, 0.301 at -30 dB. The speech fraction barely moves
  around the minimum (0.66 to 0.67).
* The selected gain (-1.8 dB) is close to the RMS rule (-3.5 dB), and gtcrn outputs with and without it agree to 36.4 dB SNR,
  i.e. the gain changes almost nothing here.
* On the first clip the same procedure chose -28.4 dB with a speech fraction that fell, so the behaviour is clip dependent.
  Two clips, no labels, no reference: still no evidence that the selected gain improves anything.

## Recovery test with known damping (alex-noisy.mp3), 2026-10-28

Script: `local_gain_recovery.py`. The clip is multiplied by m(t) = eps + sin^2(pi t / 6 s) with eps = 0.03 (dips to -30 dB at
t = 0, 6, 12 s), so the needed gain is known. A gain is chosen per 3 s window (hop 1 s, grid -12 to +36 dB in 4 dB steps) and
interpolated in dB. "Original" below means the undamped clip; VAD F1 uses SileroVad on the original as pseudo-labels
(no human labels exist), "damped parts" are the windows where m < 0.15 (20 % of windows).

| method | gain error (std, dB) | SNR vs original (dB) | VAD F1 all | VAD F1 damped parts |
|---|---|---|---|---|
| damped, no correction | 7.8 | 6.0 | 0.834 | 0.244 |
| local RMS (baseline) | 4.8 | 8.2 | 0.840 | 0.400 |
| local entropy (the idea) | 14.9 | 3.0 | 0.733 | 0.000 |

* The entropy rule did not recover the damping. It is worse than doing nothing on every metric. Several windows chose a gain
  far from the needed one (for example -12 dB where +20 dB was needed), which is the "confidently no speech" attractor seen in the
  first run.
* Local RMS helps only partly: a 3 s window averages a dip that is only about 1 s wide, so it asked for +5 dB where +20 dB was needed.
* One clip, one eps, one period, one window size, pseudo-labels from the same VAD. No tuning was done after seeing these numbers.

## Overlapped local RMS with libspeech.Resample (2026-10-28)

Reaction to the first recovery test (the owner found the local RMS weak): window 1 s, hop 10 ms (heavy overlap) instead of 3 s / 1 s,
and the dB gain curve (100 Hz) is brought to the audio rate with `libspeech.Resample` instead of `np.interp` (the two agree to
0.13 to 0.26 dB on the inner part). Same fixed target -20 dBFS, no entropy. The gain error is now measured per sample, so the
first two rows differ slightly from the table above.

| clip, damping | method | gain error std (dB) | SNR vs original (dB) | VAD F1 all | VAD F1 damped parts |
|---|---|---|---|---|---|
| alex, eps 0.03, period 6 s | no correction | 9.1 | 6.0 | 0.834 | 0.244 |
| | RMS 3 s window, hop 1 s | 6.9 | 8.2 | 0.840 | 0.400 |
| | RMS 1 s window, hop 10 ms, Resample | 3.1 | 12.7 | 0.904 | 0.853 |
| first 60 s of audio-fa-noisy, eps 0.05, period 4 s | no correction | 8.4 | 5.7 | 0.817 | 0.060 |
| | RMS 3 s window, hop 1 s | 8.3 | 5.7 | 0.845 | 0.139 |
| | RMS 1 s window, hop 10 ms, Resample | 4.8 | 8.7 | 0.956 | 0.935 |

The 1 s window was chosen from the width of the dip on alex (about 1 s) and not tuned afterwards; the Persian clip uses a
different eps and period but the same window. Two clips, VAD pseudo-labels from the same model, a deliberately smooth synthetic
damping; no check yet on real level changes or on a window size that does not match the dip.

## VAD-gated RMS with capped gain (`rms_vad_gain`), 2026-10-28

Level measured only over samples SileroVad calls speech (VAD run on the output of the previous pass, first pass = the overlapped
RMS above), target = median speech level of the clip, gain capped to [-20, +30] dB, windows with less than 0.15 s of speech filled by
interpolation, 0.5 s smoothing, two passes, then `libspeech.Resample`. Parameters were fixed before the run and not tuned afterwards.

| clip | method | gain error std (dB) | SNR vs original (dB) | VAD F1 all | VAD F1 damped parts |
|---|---|---|---|---|---|
| alex | RMS 1 s / 10 ms, Resample | 3.1 | 12.7 | 0.904 | 0.853 |
| | RMS gated by VAD, capped | 6.2 | 7.9 | 0.890 | 0.781 |
| fa, first 60 s | RMS 1 s / 10 ms, Resample | 4.8 | 8.7 | 0.956 | 0.935 |
| | RMS gated by VAD, capped | 5.6 | 9.0 | 0.944 | 0.882 |

Not better than the plain overlapped RMS: clearly worse on alex, about equal on the Persian clip (SNR 9.0 against 8.7, F1 a little
lower). The obvious guess, that the VAD misses speech inside the dips so those windows are interpolated, is not supported:
the share of windows with enough detected speech is 0.60 in the damped parts against 0.56 elsewhere (alex) and 0.81 against 0.91
(Persian). The cause is not identified.

## gain = f(p) * g(1/rms) (`fp_times_grms_*`), 2026-10-28

The form the owner asked for: a soft, multiplicative gain. g = target / rms (target -20 dBFS) capped to [-20, +30] dB, f(p) = 0.5 + 0.5 p
with p the VAD probability averaged over the same 1 s / 10 ms window, multiplied in the linear domain, then `libspeech.Resample`.
f_min = 0.5 and the linear f were fixed before the run and not tuned afterwards. "levelled": the VAD runs on the output of plain overlapped
RMS; "raw": the VAD runs on the damped signal itself. The gain error is also given on samples the original clip's VAD calls speech,
because f(p) intentionally lowers the gain elsewhere.

| clip | method | gain error all / speech samples (dB) | SNR vs original (dB) | VAD F1 all | VAD F1 damped parts |
|---|---|---|---|---|---|
| alex | RMS 1 s / 10 ms, Resample (no cap) | 3.1 / 2.4 | 12.7 | 0.904 | 0.853 |
| | f(p) g(1/rms), p levelled | 3.9 / 3.0 | 9.7 | 0.890 | 0.836 |
| | f(p) g(1/rms), p raw | 3.8 / 3.0 | 9.6 | 0.885 | 0.820 |
| fa, first 60 s | RMS 1 s / 10 ms, Resample (no cap) | 4.8 / 3.0 | 8.7 | 0.956 | 0.935 |
| | f(p) g(1/rms), p levelled | 4.3 / 3.1 | 9.6 | 0.951 | 0.905 |
| | f(p) g(1/rms), p raw | 4.8 / 4.0 | 8.6 | 0.939 | 0.873 |

Mixed: worse than the plain overlapped RMS on alex, slightly better SNR and overall gain error on the Persian clip, never better
on speech samples. The cap is a confound (the plain RMS row has none; alex needs up to +30.5 dB). Two clips, one setting each.

## Causal streaming h(f(p) * g(1/rms)) (`stream_fmin*`), 2026-10-28

Frame by frame (512 samples = 32 ms), Python simulation of what a streaming implementation would do (`SileroVad.feed` /
`pop_speech_probabilities`), not a C++ implementation. g = target (-20 dBFS) / EMA RMS (tau 0.25 s), capped to [-20, +30] dB.
p = VAD probability of the frame after the previous gain was applied (feedback), EMA tau 0.2 s. f = f_min + (1 - f_min) smoothstep(p, 0.2, 0.5).
h = smoothing of the dB gain, attack 0.05 s when the gain goes down, release 0.3 s when it goes up; gain ramped linearly in dB inside a frame; the
first gain is set at once. All parameters fixed before the run; only f_min was varied, as planned: {1, 0.5, 0.25}.

| clip | method | gain error all / speech samples (dB) | SNR vs original (dB) | VAD F1 all | VAD F1 damped parts |
|---|---|---|---|---|---|
| alex | no correction | 9.1 / 7.9 | 6.0 | 0.834 | 0.244 |
| | non-causal f(p) g(1/rms), levelled (best non-causal with f) | 3.9 / 3.0 | 9.7 | 0.890 | 0.836 |
| | streaming, f_min 1 (AGC only) | 5.9 / 6.5 | 6.8 | 0.863 | 0.700 |
| | streaming, f_min 0.5 | 6.7 / 6.6 | 5.1 | 0.841 | 0.678 |
| | streaming, f_min 0.25 | 8.2 / 7.8 | 2.8 | 0.795 | 0.642 |
| fa, first 60 s | no correction | 8.4 / 7.9 | 5.7 | 0.817 | 0.060 |
| | non-causal f(p) g(1/rms), levelled | 4.3 / 3.1 | 9.6 | 0.951 | 0.905 |
| | streaming, f_min 1 (AGC only) | 6.8 / 5.5 | 5.7 | 0.928 | 0.783 |
| | streaming, f_min 0.5 | 6.8 / 5.9 | 6.5 | 0.915 | 0.673 |
| | streaming, f_min 0.25 | 7.7 / 6.9 | 5.8 | 0.887 | 0.599 |

* This parameterization recovers clearly less than the non-causal windows (SNR on the Persian clip equal to no correction at f_min 1).
* Lowering f_min made it monotonically worse on both clips, so f(p) did not help in the streaming loop. A plausible cause is the feedback: where
  the signal is damped, p is low, f lowers the gain exactly where it is needed. Not tested separately.
* The delay of the causal averaging (RMS 0.25 s, release 0.3 s) against a dip about 1 s wide is another plausible cause. Not tested separately.
* No tuning of tau, attack, release or the smoothstep edges was done after seeing these numbers.

## Correction, and a test against a reference (`recovery_suite.py`), 2026-10-28

**Correction to the tables above.** The "SNR vs original" column compares the processed signal with the undamped *noisy* clip, so removing
noise counts as error. Any method that lowers the gain where there is no speech (f_min < 1, the VAD gating) was penalized by that column.
The statement "lowering f_min made the streaming version monotonically worse" is therefore not reliable for the SNR and gain-error columns;
only the VAD F1 columns are unaffected by this, and F1 also fell with f_min.

**New protocol.** A first denoiser (DNS64, `facebook`) is run on the noisy clip; its output is the reference and noisy clip minus reference is the
extracted noise. Test signals are built from the reference: speech damped by m(t) = 0.03 + sin^2(pi t / 6) (alex; 0.05 and 4 s for the Persian clip)
plus noise, in four scenarios: A everything damped (reference + alex noise); B damped reference + alex noise not damped; C damped reference +
pink noise at 10 dB SNR; D damped reference + 50 Hz hum and white noise at 10 dB SNR. The gain methods run on each signal, alone and followed by
a second denoiser (gtcrn). Score: SI-SDR against the reference (dB, higher is better). The reference is a denoiser output and not a studio
recording, so only comparisons inside a scenario mean something. No distortion other than the damping was added. Nothing was tuned.

SI-SDR against the reference, dB:

| alex | A | B | C | D |
|---|---|---|---|---|
| mix (no processing) | 2.6 | 0.8 | 2.1 | 2.1 |
| RMS 1 s / 10 ms (non-causal) | 5.8 | -0.4 | 0.8 | 0.6 |
| f(p) g(1/rms) levelled, f_min 0.5 (non-causal) | 4.7 | -0.9 | 1.7 | 1.6 |
| streaming f_min 1 | 2.7 | -2.2 | 0.6 | 0.5 |
| streaming f_min 0.5 | 1.1 | -2.4 | 1.7 | 1.7 |
| gtcrn only | 4.2 | 3.1 | 3.6 | 3.5 |
| RMS, then gtcrn | 11.6 | 4.4 | 6.3 | 6.1 |
| f(p) g(1/rms), then gtcrn | 9.2 | 2.8 | 5.6 | 5.1 |
| streaming f_min 1, then gtcrn | 7.2 | 3.7 | 5.3 | 4.9 |
| streaming f_min 0.5, then gtcrn | 5.2 | 2.0 | 4.5 | 4.3 |

| first 60 s of audio-fa-noisy | A | B | C | D |
|---|---|---|---|---|
| mix (no processing) | 3.0 | 1.8 | 1.7 | 1.7 |
| RMS 1 s / 10 ms (non-causal) | 4.8 | 0.1 | -0.3 | -0.5 |
| f(p) g(1/rms) levelled, f_min 0.5 (non-causal) | 5.9 | 1.0 | 1.1 | 0.9 |
| streaming f_min 1 | 2.4 | 0.1 | 0.0 | -0.2 |
| streaming f_min 0.5 | 3.6 | 0.9 | 1.3 | 1.0 |
| gtcrn only | 4.0 | 3.1 | 2.7 | 3.4 |
| RMS, then gtcrn | 9.5 | 4.6 | 4.7 | 5.8 |
| f(p) g(1/rms), then gtcrn | 9.3 | 4.6 | 4.3 | 5.4 |
| streaming f_min 1, then gtcrn | 5.1 | 3.8 | 3.9 | 4.6 |
| streaming f_min 0.5, then gtcrn | 5.5 | 3.6 | 3.7 | 4.4 |

* Gain followed by gtcrn beats gtcrn alone in every row except streaming f_min 0.5 on alex B (2.0 against 3.1). The largest gain is where
  everything is damped (A): 11.6 against 4.2 (alex), 9.5 against 4.0 (Persian).
* Gain alone is often below the unprocessed mix when only the speech is damped and the noise is not (B, C, D): the gain also boosts the noise
  that is left in the dips.
* With this metric f(p) helps the gain alone in B, C, D on both clips (e.g. Persian C: 1.1 against -0.3), but after gtcrn the plain RMS is equal or
  better, so f(p) is redundant when a denoiser follows.
* Streaming is below the non-causal versions in almost every cell.
* One alex clip and one Persian excerpt, one damping each, a pseudo-reference. Files for alex are numbered, see `INDEX.md` in the output.
* `make_listening_page.py out_dir` builds `index.html` next to the numbered files. Per scenario: the unprocessed input, then three columns,
  normalization only, denoise only (one file, shared by all rows, because it does not depend on the method) and denoise + normalization
  (normalization first, then gtcrn); with the SI-SDR, time and RTF of each file and the best one marked.
* `recovery_suite.py` also writes `results.json` next to the audio: per scenario and per output file the SI-SDR, the difference to the
  unprocessed mix and to gtcrn alone, RMS/peak/clipped fraction, seconds and RTF (gain stage, denoiser, total), whether the method is causal
  and its look-ahead or frame size, plus a median summary per method and the run's meta (clip, damping, machine, versions). Timing is the median of 3 runs
  after a warm-up, on one CPU core of the sandbox, so it only shows the order of magnitude.
* Median RTF on alex (16.9 s; this machine, one thread): gain stage RMS overlap 0.0004, f(p) g(1/rms) 0.0052, streaming f_min 1 / 0.5 about 0.009
  (Python per-frame simulation, with the VAD run every 32 ms); gtcrn alone 0.071; the reference denoiser (DNS64) 3.56 s, RTF 0.21.
  The gain stage is therefore small next to the denoiser. The non-causal versions need 0.5 s look-ahead; the streaming ones work on 32 ms frames.
