libspeech Python API
=====================

**libspeech** is a lightweight speech processing library: audio I/O, DSP (STFT, FFT, MFCC,
resampling) and ONNX speech models (denoising, voice activity detection). The Python package is a
nanobind extension that mirrors the C++ API; NumPy arrays go in and come out without list
conversion.

.. code-block:: python

   import numpy as np
   import libspeech

   x = np.random.randn(16000).astype(np.float32)
   z = libspeech.stft(x, n_fft=512, hop_length=128)       # (bins, frames) complex64
   y = libspeech.istft(z, hop_length=128, length=len(x))

Installation
------------

.. code-block:: bash

   pip install libspeech

.. toctree::
   :maxdepth: 1

   Overview <self>
   dsp
   io
   models
   utils
