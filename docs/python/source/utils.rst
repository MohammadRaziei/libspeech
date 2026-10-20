Utilities
=========

Logging
-------

The default level is ``warning``; ``LIBSPEECH_LOG`` sets it from the environment.

The native (C++) log arrives in Python's ``logging``, in the logger ``"libspeech"``, and that logger's level
decides what the native side writes (``setLevel``, ``basicConfig``, ``dictConfig`` and ``set_log_level`` all work).
``set_log_callback`` and ``set_log_file`` take the native log over instead (one output is active at a time).

.. autofunction:: libspeech.set_log_level
.. autofunction:: libspeech.get_log_level
.. autofunction:: libspeech.set_log_callback
.. autofunction:: libspeech.set_log_file

SIMD
----

.. autofunction:: libspeech.simd_backend
.. autofunction:: libspeech.simd_enabled
.. autofunction:: libspeech.set_simd_enabled
