# C++ API renames

libspeech's C++ API was renamed to one consistent, standard-library-style scheme (see
"Style notes" in [CONTRIBUTING.md](../CONTRIBUTING.md)): everything is `snake_case`, private data members
end in `_`, and nothing from AudioFlux leaks into the public headers. **The Python API is unchanged.**
This table lists every name that changed in the public headers since v0.3.x.

## Types

| old | new |
|---|---|
| `Audio` | `audio` |
| `AudioImpl` | `audio_impl` |
| `Backend` | `backend` |
| `BaseModel` | `base_model` |
| `DctII` | `dct_ii_plan` |
| `DefaultInitAllocator` | `default_init_allocator` |
| `Denoiser` | `denoiser` |
| `FFT` | `fft` |
| `FacebookDenoiser` | `facebook_denoiser` |
| `MFCC` | `mfcc` |
| `MfccMatrix` | `mfcc_matrix` |
| `ONNXModel` | `onnx_model` |
| `Polyphase` | `polyphase` |
| `RealFrameFFT` | `real_frame_fft` |
| `Resample` | `resample` |
| `STFT` | `stft` |
| `SileroVadModel` | `silero_vad_model` |
| `Spectrogram` | `complex_spectrogram` |
| `SpeechBrainDenoiser` | `speechbrain_denoiser` |
| `Traits` | `traits` |
| `UninitVector` | `uninit_vector` |
| `WindowType` | `window_type` |
| `timestamp_t` | `speech_timestamp` |

## Enumerators

| old | new |
|---|---|
| `Window_Hann, Window_Hamm, Window_Rect, ...  (AudioFlux)` | `window_type::hann, hamming, rect, ... (libspeech's own enum class)` |
| `Backend::Generic / X86Avx2Fma / Arm64Neon` | `backend::generic / x86_avx2_fma / arm64_neon` |

## Methods whose name changed for a reason beyond case

| old | new | why |
|---|---|---|
| `STFT::stft(...)` | `stft::compute(...)` | the method shared its name with the class |
| `Resample::resample(...)` | `resample::process(...)` | same reason |
| `Denoiser::Create(...)` | `denoiser::create(...)` |  |

## Functions, methods and fields (same words, snake_case)

| old | new | declared in |
|---|---|---|
| `activeBackend` | `active_backend` | `dsp/simd.h` |
| `backendName` | `backend_name` | `dsp/simd.h` |
| `buildMelFilterBank` | `build_mel_filter_bank` | `dsp/mfcc.h` |
| `buildPolyphaseExact` | `build_polyphase_exact` | `dsp/resample.h` |
| `buildPolyphaseFromTables` | `build_polyphase_from_tables` | `dsp/resample.h` |
| `calDataLength` | `cal_data_length` | `dsp/stft.h` |
| `calTimeLength` | `cal_time_length` | `dsp/stft.h` |
| `channelData` | `channel_data` | `audio.h` |
| `computeFlat` | `compute_flat` | `dsp/mfcc.h` |
| `customWindow` | `custom_window` | `dsp/resample.h` |
| `dataLength` | `data_length` | `dsp/stft.h` |
| `dctII` | `dct_ii` | `dsp/dct.h` |
| `downloadFile` | `download_file` | `utils/utils.h` |
| `enableContinuous` | `enable_continuous` | `dsp/resample.h` |
| `ensureAudioFlux` | `ensure_audio_flux` | `dsp/resample.h` |
| `fftLength` | `fft_length` | `dsp/mfcc.h` |
| `filePath` | `file_path` | `audio.h` |
| `firstFrame` | `first_frame` | `dsp/stft.h` |
| `getDefaultModelCacheDir` | `get_default_model_cache_dir` | `models/base_model.h` |
| `getTempDirectory` | `get_temp_directory` | `utils/utils.h` |
| `highFreqHz` | `high_freq_hz` | `dsp/mfcc.h` |
| `inputData` | `input_data` | `audio.h` |
| `inputSize` | `input_size` | `dsp/dct.h` |
| `isContinue` | `is_continue` | `dsp/resample.h` |
| `isEnabled` | `is_enabled` | `dsp/simd.h` |
| `isIdentity` | `is_identity` | `dsp/resample.h` |
| `isNorm` | `is_norm` | `dsp/fft.h` |
| `isScale` | `is_scale` | `dsp/resample.h` |
| `lowFreqHz` | `low_freq_hz` | `dsp/mfcc.h` |
| `methodType` | `method_type` | `dsp/stft.h` |
| `numBins` | `num_bins` | `dsp/stft.h` |
| `numCoefficients` | `num_coefficients` | `dsp/mfcc.h` |
| `numFrames` | `num_frames` | `dsp/mfcc.h` |
| `numMelFilters` | `num_mel_filters` | `dsp/mfcc.h` |
| `numOutputs` | `num_outputs` | `dsp/dct.h` |
| `outputPath` | `output_path` | `audio.h` |
| `outputSize` | `output_size` | `dsp/dct.h` |
| `pImpl` | `impl_` | `audio.h` |
| `powerSpectra` | `power_spectra` | `dsp/stft.h` |
| `processOnVector` | `process_on_vector` | `models/silero_vad.h` |
| `radix2Exp` | `radix2_exp` | `dsp/fft.h` |
| `resampleFlat` | `resample_flat` | `dsp/resample.h` |
| `resampleObj` | `resample_obj` | `dsp/resample.h` |
| `resamplePolyphase` | `resample_polyphase` | `dsp/resample.h` |
| `rollOff` | `roll_off` | `dsp/resample.h` |
| `sampleRate` | `sample_rate` | `audio.h` |
| `setEnabled` | `set_enabled` | `dsp/simd.h` |
| `setSampleRateRatio` | `set_sample_rate_ratio` | `dsp/resample.h` |
| `slideLength` | `slide_length` | `dsp/mfcc.h` |
| `sourceRate` | `source_rate` | `dsp/resample.h` |
| `stftInto` | `stft_into` | `dsp/stft.h` |
| `stftOnesidedInto` | `stft_onesided_into` | `dsp/stft.h` |
| `targetRate` | `target_rate` | `dsp/resample.h` |
| `targetSampleRate` | `target_sample_rate` | `audio.h` |
| `timeLength` | `time_length` | `dsp/stft.h` |
| `winType` | `win_type` | `dsp/resample.h` |
| `windowType` | `window_type_` | `dsp/stft.h` |
| `zeroNum` | `zero_num` | `dsp/resample.h` |

## Class getters without the `get_` prefix (Unreleased)

C++ and Python. Only member functions of classes; free functions such as `get_log_level` keep their names. The setter keeps
`set_`. No old names are kept as aliases.

| old | new | where |
|---|---|---|
| `silero_vad_model::get_speech_timestamps()` / `SileroVad.get_speech_timestamps()` | `speech_timestamps()` | `models/silero_vad.h` |
| `silero_vad_model::get_threshold()` / `SileroVad.get_threshold()` | `threshold()` (the data member is now `threshold_`) | `models/silero_vad.h` |
