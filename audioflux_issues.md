# AudioFlux Issues — Found, Fixed, and Documented

libspeech vendors a subset of [libAudioFlux/audioflux](https://github.com/libAudioFlux/audioflux)
(MIT licensed) C sources directly into `src/dsp/vendor/audioflux/`, instead of
depending on the full submodule. While porting `speech::dsp::Resample`
(the first DSP operator ported) and writing TDD tests for it, two real
problems in AudioFlux's C code were found. Both are documented here in
detail, with repro steps and fixes, so they can eventually be turned into
pull requests against the upstream project.

**Source commit vendored from:** `824f76d5f19d0358779e513d708a987e4fb9224e`
**Files vendored (resample-only, so far):** see `src/dsp/vendor/audioflux/README.md`

Every time a new AudioFlux issue is found while porting another DSP
operator (FFT, DCT, window functions, ...), add a new numbered section below.

---

## Issue 1 — `resampleObj_setSamplate()` silently corrupts identity resamples

- **Severity:** Data-corrupting bug (silent, no error returned)
- **File:** `src/dsp/resample_algorithm.c`
- **Function:** `resampleObj_setSamplate`
- **Status:** ✅ Fixed in our vendored copy. Not yet submitted upstream.

### The bug

```c
// original upstream code
void resampleObj_setSamplate(ResampleObj resampleObj,int sourceRate,int targetRate){
    ...
    if(sourceRate==targetRate||
        sourceRate<=0||targetRate<=0){
        return;
    }
    ...
}
```

`resampleObj_newWithWindow()` initializes every new resample object with
hardcoded defaults `ratio=0.5, p=1, q=2` (apparently tuned for the library's
own internal default of 32000Hz -> 16000Hz). `resampleObj_setSamplate()` is
supposed to overwrite these with the ratio for whatever rates the caller
actually wants — but when `sourceRate == targetRate`, it returns immediately
*without* touching `ratio`/`p`/`q`. Any subsequent call to
`resampleObj_resample()` on that object then uses the leftover `ratio=0.5`
and silently discards half the input samples, instead of behaving as an
identity (no-op) resample as a caller would reasonably expect.

### How it was found

TDD test `Resample.SameRateIsExactIdentity` (`tests/dsp/test_resample.cpp`):
resampling 1000 samples at 16000Hz -> 16000Hz should return exactly 1000
samples. It returned 500.

### Repro (against AudioFlux's own C API directly)

```c
ResampleObj obj;
ResampleQualityType q = ResampleQuality_Best;
int isScale = 1, isContinue = 0;
resampleObj_new(&obj, &q, &isScale, &isContinue);
resampleObj_setSamplate(obj, 16000, 16000);  // sourceRate == targetRate

int outLen = resampleObj_calDataLength(obj, 1000);
// outLen == 500 -- expected 1000
```

### The fix

Remove the `sourceRate==targetRate` clause entirely. The general-case
computation a few lines further down already produces the mathematically
correct identity ratio when the rates are equal, because
`gcd(sourceRate, sourceRate) == sourceRate`:

```c
gcd = util_gcd(sourceRate, sourceRate);      // == sourceRate
p = targetRate / gcd;                        // == 1
q = sourceRate / gcd;                        // == 1
ratio = targetRate / (float)sourceRate;      // == 1.0
```

So only the validity guard needs to remain:

```c
// patched (src/dsp/vendor/audioflux/src/dsp/resample_algorithm.c)
if(sourceRate<=0||targetRate<=0){
    return;
}
```

### Defense in depth on our side

Even with the patch applied, `speech::dsp::Resample`'s C++ wrapper
short-circuits same-rate calls before ever reaching AudioFlux (see
`isIdentity` in `src/dsp/resample.cpp`), so the wrapper is correct even if
someone reverts the vendored patch by mistake. The patch is kept anyway so
the vendored C code is correct on its own terms and is PR-ready as-is.

---

## Issue 2 — `isScale=1` silently rescales waveform amplitude, not just energy

- **Severity:** Surprising/undocumented behavior (not a crash, but produces
  audibly/measurably wrong output for typical audio use if you don't know
  about it)
- **File:** `src/dsp/resample_algorithm.c`
- **Function:** `resampleObj_resample`
- **Status:** ⚠️ Not patched upstream (arguably intentional AudioFlux
  behavior for spectral/energy use cases) — worked around in our C++ wrapper
  instead by defaulting `isScale=0`.

### The behavior

```c
// step 5 of resampleObj_resample()
if(resampleObj->isScale){
    int _len = resampleObj->targetDataLength;
    float _value = sqrtf(resampleObj->ratio);
    for(int i=0;i<_len;i++){
        dataArr2[i] /= _value;
    }
}
```

When `isScale=1` (which is what every example/default in AudioFlux uses),
the output is divided by `sqrt(ratio)`, where
`ratio = targetRate / sourceRate`. This is an **energy-domain**
normalization (it preserves total signal *power* across a rate change,
which matters for some spectral/feature-extraction use cases) — but it is
**not** amplitude-preserving.

For a downsample from 44100Hz to 16000Hz:
`ratio ≈ 0.3628`, `sqrt(ratio) ≈ 0.6023`, so every sample gets divided by
~0.6, i.e. **amplified by ~1.66x**. A full-scale (±1.0) input sine wave comes
back out peaking at ~±1.66 — well past the normal audio range, which will
clip on playback or corrupt anything downstream (VAD, denoiser models, etc.)
that assumes `[-1, 1]`-normalized audio.

### How it was found

TDD test `Resample.PreservesSignalEnergyRoughly`: a resampled 440Hz sine
generated at amplitude 1.0 should still peak in roughly `[0.5, 1.5]` after
resampling. It measured a peak of `~1.66`. Instrumented with debug builds
that dumped per-sample values, all high-amplitude samples matched the
predicted `1/sqrt(ratio) ≈ 1.66` scaling factor exactly — confirming this
is the `isScale` code path, not e.g. filter ringing or a separate bug.

### The fix (in our wrapper, not upstream AudioFlux)

`speech::dsp::Resample`'s default constructor now passes `isScale=0` to
AudioFlux, so waveform amplitude is preserved by default:

```cpp
// src/dsp/resample.cpp
int isScale = 0;  // amplitude-preserving default; see audioflux_issues.md
```

The extended constructor (`Resample(sourceRate, targetRate, zeroNum, nbit,
winType, value, rollOff, isScale, isContinue)`) still exposes `isScale` as a
parameter for advanced users who specifically want AudioFlux's energy-domain
scaling.

### Why this isn't filed as an upstream bug

`isScale` does what its docstring-less implementation literally does — it's
not incorrect on its own terms, just a different normalization convention
than "preserve waveform amplitude," and it's easy to reach for by default
without realizing the effect. If we do open an upstream issue, it would be a
documentation request (clarify what `isScale` actually normalizes, and
warn about the amplitude implication) rather than a bug report.

---

## Issue 3 — `fftObj_idct()` mutates its input array in place

- **Severity:** API footgun (not a crash or wrong result, but silently
  modifies data the caller may still expect to read afterward)
- **File:** `src/dsp/fft_algorithm.c`
- **Function:** `fftObj_idct`
- **Status:** ⚠️ Not patched upstream (the in-place scaling is presumably an
  intentional micro-optimization) — worked around in our C++ wrapper by
  copying the input before calling into AudioFlux.

### The behavior

```c
void fftObj_idct(FFTObj fftObj,float *dataArr1,float *dataArr2,int isNorm){
    ...
    if(isNorm){
        dataArr1[0]/=fftObj->s0;
        for(int i=1;i<length;i++){
            dataArr1[i]/=fftObj->s1;
        }
    }

    dataArr1[0]/=2;
    for(int i=0;i<length;i++){
        dataArr1[i]/=fftObj->wLength;
        _realArr1[i]=dataArr1[i]*wCosArr1[i];
        _imageArr1[i]=dataArr1[i]*wSinArr1[i];
    }
    ...
}
```

`dataArr1` (the DCT coefficients being inverted) is divided in place by
`s0`/`s1`/`wLength` as part of the computation. Every other AudioFlux
function we vendor so far (`fftObj_fft`, `fftObj_ifft`, `resampleObj_resample`)
copies its input into an internal buffer first and leaves the caller's array
untouched — `fftObj_idct` is the one exception, and it's easy to miss because
nothing in the function name or signature suggests input mutation.

### How it was found

Not a failing test at first — noticed while writing
`FFT.DctIdctRoundTrip` and reasoning about what `idct()`'s contract should
be, then added a dedicated regression test (`FFT.IdctDoesNotMutateInput`)
before implementing the fix, to lock in the correct behavior going forward.

### The fix (in our wrapper, not upstream AudioFlux)

`speech::dsp::FFT::idct()` copies its input into a local `std::vector`
before calling `fftObj_idct()`, so `speech::dsp::FFT`'s public contract is
"never mutates arguments," full stop — callers don't need to know which
AudioFlux functions happen to mutate their C arrays internally.

```cpp
// src/dsp/fft.cpp
std::vector<float> inputCopy = data;
fftObj_idct(fftObj_, inputCopy.data(), output.data(), isNorm ? 1 : 0);
```

### Why this isn't filed as an upstream bug

The result is correct if you know to discard/not rely on `dataArr1` after
the call -- this is arguably fine for a low-level C API where the caller
owns and expects to manage buffer lifetimes explicitly. If filed upstream,
it would be a documentation request (note the in-place mutation in the
function's comment/header) rather than a bug report.

---

## How to add a new entry

When porting another DSP operator (FFT, DCT, filterbank, ...) and you find
another AudioFlux quirk or bug, add a new `## Issue N — <title>` section
above following the same structure: severity, file/function, status, the
bug/behavior with a minimal code snippet, how it was found (ideally via a
failing TDD test), a repro, and the fix (and whether it lives in the
vendored C code or in our C++ wrapper).
