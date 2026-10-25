# Labs

Experiments that are not part of the library: one folder per story, each with a notebook, the scripts it uses, and a README.

| lab | story |
|---|---|
| [`quantize_denoiser/`](quantize_denoiser/) | int8 versions of the Facebook denoisers (dns48, dns64, master64): post-training quantization, teacher-student fine-tuning, and what it costs |

Conventions:

* The notebook runs in Google Colab (the badge at its top opens it) and also locally from the lab folder. It downloads
  its audio and models with `wget` from the GitHub releases; libspeech itself is installed (from git) only by a part
  that is switched off by default and needs it.
* It has a `SMOKE = True` switch: a tiny random model, synthetic data and a few steps, so that the whole notebook can be
  checked in a minute without downloads or a GPU.
* Notebooks are committed without outputs and without execution counts (a test checks it); every function a notebook calls
  from the lab's modules has to exist (a test checks that too).
* The lab's scripts are plain modules next to the notebook, and its tests are in `<lab>/tests/`. Run them by hand with
  `pytest labs/<lab>/tests` (from the repository root); they are not part of CMake or CI, and a test module skips itself
  when an optional package (torch, soundfile, ...) is missing.
