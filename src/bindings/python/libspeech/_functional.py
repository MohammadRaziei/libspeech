"""The everyday, librosa-style API: ``libspeech.stft`` and ``libspeech.istft``.

These are thin wrappers over the :class:`libspeech.STFT` class (use that when you want to keep a
transform around, pick the layout yourself, or avoid the padding copy). They follow librosa's
conventions so that code moves over with a rename:

* keyword-only parameters after the signal, ``n_fft`` / ``hop_length`` / ``window`` / ``center``;
* a one-sided, complex64 result of shape ``(1 + n_fft // 2, n_frames)``;
* ``center=True`` pads the signal by ``n_fft // 2`` on both sides, so frame ``t`` is centred on
  sample ``t * hop_length`` and ``istft`` can trim it back.

The result is a transposed *view* of the ``(n_frames, bins)`` array the kernel writes, exactly like
librosa's Fortran-ordered output, so nothing is copied to get that shape.

NumPy is required to call these functions (it is not needed for the rest of the package).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Union

from .speech_dsp_py import STFT, WindowType

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

__all__ = ["istft", "stft"]

_WINDOWS = {
    "hann": WindowType.hann,
    "hanning": WindowType.hann,
    "hamming": WindowType.hamming,
    "hamm": WindowType.hamming,
    "blackman": WindowType.blackman,
    "rect": WindowType.rect,
    "rectangular": WindowType.rect,
    "boxcar": WindowType.rect,
    "bartlett": WindowType.bartlett,
    "triang": WindowType.triang,
    "triangular": WindowType.triang,
    "flattop": WindowType.flattop,
    "gauss": WindowType.gauss,
    "gaussian": WindowType.gauss,
}

WindowLike = Union[str, WindowType]


def _numpy():
    try:
        import numpy as np  # noqa: PLC0415 - NumPy stays an optional dependency of the package
    except ImportError as exc:  # pragma: no cover - exercised only without numpy installed
        msg = "libspeech.stft / libspeech.istft need NumPy: pip install numpy"
        raise ImportError(msg) from exc
    return np


def _window(window: WindowLike) -> WindowType:
    if isinstance(window, WindowType):
        return window
    try:
        return _WINDOWS[str(window).lower()]
    except KeyError:
        valid = ", ".join(sorted(set(_WINDOWS)))
        msg = f"unknown window {window!r}; use a WindowType or one of: {valid}"
        raise ValueError(msg) from None


def _check_n_fft(n_fft: int) -> int:
    n_fft = int(n_fft)
    if n_fft < 2 or n_fft & (n_fft - 1):
        msg = f"n_fft must be a power of two (e.g. 512, 1024, 2048), got {n_fft}"
        raise ValueError(msg)
    return n_fft


def _infer_n_fft(bins: int) -> int:
    """2**m + 1 bins is a one-sided spectrum (n_fft = 2**(m+1)); 2**m bins is a full one (n_fft = 2**m)."""
    if bins >= 2 and (bins - 1) & (bins - 2) == 0:  # bins - 1 is a power of two
        return 2 * (bins - 1)
    if bins >= 2 and bins & (bins - 1) == 0:
        return bins
    msg = f"cannot infer n_fft from {bins} bins (expected 2**m + 1 or 2**m); pass n_fft"
    raise ValueError(msg)


def _pad(y: NDArray[np.float32], amount: int, mode: str) -> NDArray[np.float32]:
    """Pad ``amount`` samples on both sides; zeros (the default) skip numpy.pad's Python overhead."""
    np = _numpy()
    if mode != "constant":
        return np.pad(y, amount, mode=mode)
    padded = np.empty(y.shape[0] + 2 * amount, dtype=np.float32)
    padded[:amount] = 0.0
    padded[amount : amount + y.shape[0]] = y
    padded[amount + y.shape[0] :] = 0.0
    return padded


def _hop(n_fft: int, hop_length: int | None) -> int:
    hop = n_fft // 4 if hop_length is None else int(hop_length)
    if hop < 1:
        msg = f"hop_length must be >= 1, got {hop}"
        raise ValueError(msg)
    return hop


# STFT objects hold the window and kernel tables, so building one per call would dominate short
# signals. Their fast path keeps no per-call state, which is what makes sharing them safe.
@functools.lru_cache(maxsize=32)
def _transform(n_fft: int, hop_length: int, window: WindowType) -> STFT:
    return STFT(n_fft=n_fft, window_type=window, hop_length=hop_length)


def stft(
    y: ArrayLike,
    *,
    n_fft: int = 2048,
    hop_length: int | None = None,
    window: WindowLike = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    onesided: bool = True,
) -> NDArray[np.complex64]:
    """Short-time Fourier transform of a mono signal.

    Parameters
    ----------
    y
        1-D signal (any float dtype; float32 is used as is, without a copy).
    n_fft
        Frame length in samples; a power of two. 16..8192 use the fast real-input path.
    hop_length
        Samples between frames; default ``n_fft // 4``.
    window
        ``"hann"`` (default), ``"hamming"``, ``"blackman"``, ``"rect"``, ``"bartlett"``,
        ``"triang"``, ``"flattop"``, ``"gauss"``, or a :class:`~libspeech.WindowType`.
    center
        Pad ``n_fft // 2`` samples on both sides so frames are centred (librosa's default).
        This makes one copy of the signal. ``False`` analyses ``y`` exactly as given.
    pad_mode
        How to pad when ``center`` is true: ``"constant"`` (zeros, librosa's default) or any
        :func:`numpy.pad` mode such as ``"reflect"``.
    onesided
        ``True`` (default): the ``n_fft // 2 + 1`` non-negative frequencies of a real signal.
        ``False``: all ``n_fft`` bins (the second half is the conjugate mirror).

    Returns
    -------
    numpy.ndarray
        complex64, shape ``(bins, n_frames)``, Fortran-ordered like librosa's result.
    """
    np = _numpy()
    n_fft = _check_n_fft(n_fft)
    hop = _hop(n_fft, hop_length)
    transform = _transform(n_fft, hop, _window(window))

    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1:
        msg = f"y must be a 1-D mono signal, got shape {y.shape}"
        raise ValueError(msg)
    if center:
        y = _pad(y, n_fft // 2, pad_mode)
    if y.shape[0] < n_fft:
        msg = f"the signal has {y.shape[0]} samples, fewer than n_fft={n_fft}; use center=True or a shorter n_fft"
        raise ValueError(msg)

    return transform.stft_complex(y, onesided=onesided).T


def istft(
    stft_matrix: ArrayLike,
    *,
    hop_length: int | None = None,
    n_fft: int | None = None,
    window: WindowLike = "hann",
    center: bool = True,
    length: int | None = None,
    method: str = "weighted",
) -> NDArray[np.float32]:
    """Inverse short-time Fourier transform (the counterpart of :func:`stft`).

    Parameters
    ----------
    stft_matrix
        Complex array of shape ``(bins, n_frames)`` with ``bins == n_fft // 2 + 1`` (one-sided,
        e.g. from :func:`stft`) or ``bins == n_fft``. A Fortran-ordered array -- what :func:`stft`
        returns -- is used without a copy; any other layout is copied once.
    hop_length, window
        As for :func:`stft`; they must match the values the spectrogram was made with.
    n_fft
        Frame length; inferred from the bin count if omitted (``2**m + 1`` bins: one-sided,
        ``2**m`` bins: the full spectrum).
    center
        Trim the ``n_fft // 2`` padding that ``stft(center=True)`` added.
    length
        Trim or zero-pad the result to exactly this many samples.
    method
        ``"weighted"`` (default): weighted overlap-add, normalised by the summed squared window
        (what librosa does). ``"plain"``: plain overlap-add, normalised by the summed window.

    Returns
    -------
    numpy.ndarray
        float32, 1-D.
    """
    np = _numpy()
    z = np.asarray(stft_matrix)
    if z.ndim != 2 or not np.iscomplexobj(z):
        msg = f"stft_matrix must be a 2-D complex array (bins, n_frames), got dtype {z.dtype} and shape {z.shape}"
        raise ValueError(msg)
    z = z.astype(np.complex64, copy=False)
    bins = z.shape[0]

    if n_fft is None:
        n_fft = _infer_n_fft(bins)
    n_fft = _check_n_fft(n_fft)
    if bins not in (n_fft // 2 + 1, n_fft):
        msg = f"a spectrogram with {bins} bins does not fit n_fft={n_fft} (expected {n_fft // 2 + 1} or {n_fft})"
        raise ValueError(msg)
    transform = _transform(n_fft, _hop(n_fft, hop_length), _window(window))

    y = transform.istft(z.T, method)  # z.T is (frames, bins): contiguous (no copy) for stft()'s output

    pad = n_fft // 2 if center else 0
    start = pad
    end = y.shape[0] - pad if length is None else start + int(length)
    y = y[start:end]
    if length is not None and y.shape[0] < length:
        y = np.pad(y, (0, length - y.shape[0]))
    return y
