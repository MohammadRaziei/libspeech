DSP
===

Functions
---------

.. autofunction:: libspeech.stft
.. autofunction:: libspeech.istft
.. autofunction:: libspeech.dct
.. autofunction:: libspeech.window
.. autofunction:: libspeech.rms_normalize
.. autofunction:: libspeech.rms_dbfs
.. autofunction:: libspeech.speech_weights
.. autofunction:: libspeech.speech_rms_normalize
.. autofunction:: libspeech.rms_normalize_weighted
.. autofunction:: libspeech.weighted_rms
.. autofunction:: libspeech.limit_peaks
.. autofunction:: libspeech.weighted_cmvn
.. autofunction:: libspeech.frame_weights

Classes
-------

.. autoclass:: libspeech.STFT
   :members:
.. autoclass:: libspeech.FFT
   :members:
.. autoclass:: libspeech.MFCC
   :members:
.. autoclass:: libspeech.MFCCParams
   :members:
.. autoclass:: libspeech.Resample
   :members:
   :special-members: __init__

``Resample(source_rate, target_rate, quality)`` takes ``"best"`` (default), ``"high"``,
``"medium"`` or ``"fast"``; see the changelog for the measured speed and accuracy of each level.

Window types
------------

.. autoclass:: libspeech.WindowType
   :members:
   :undoc-members:
