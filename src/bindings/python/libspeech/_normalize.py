"""Speech-gated RMS normalization: ``libspeech.speech_rms_normalize``.

``libspeech.rms_normalize`` measures the level over every sample, so a recording with long pauses reads quieter than
its speech is and ends up too loud. This function measures the level only where the Silero VAD says there is speech:
the VAD's per-window speech probabilities are turned into weights (a hard or a soft threshold, see
:func:`libspeech.speech_weights`) and the weighted RMS is brought to the target. The gain is still one number for the
whole signal and the peak ceiling still wins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .speech_dsp_py import rms_normalize_weighted, speech_weights
from .speech_models_py import SileroVad

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

__all__ = ["frame_weights", "speech_rms_normalize"]


def speech_rms_normalize(
    signal: ArrayLike,
    *,
    sample_rate: int = 16000,
    mode: str = "hard",
    threshold: float | None = None,
    target_dbfs: float = -20.0,
    peak_ceiling: float = 0.99,
    clip_percentile: float = 100.0,
    limiter_ms: float | None = None,
    vad: SileroVad | None = None,
) -> NDArray[np.float32]:
    """Scales a mono signal so that its *speech* level is ``target_dbfs``.

    ``mode="hard"`` counts a window as speech (weight 1) when its probability reaches ``threshold``, otherwise it
    does not count (weight 0). ``mode="soft"`` gives a window the weight ``(p - threshold) / (1 - threshold)``
    clamped to [0, 1], so a window just above the threshold counts little and a certain one counts fully.
    ``threshold`` defaults to the threshold of the VAD (0.5 unless you changed it).

    ``clip_percentile`` (0 < p <= 100, default 100 = off) makes the level robust: the loudest windows (plosives,
    clicks) count at most as much as the weighted p-th percentile window, so a few of them do not pull the gain down.

    ``limiter_ms`` (default None) replaces the "peak ceiling lowers the gain of the whole signal" rule with a soft
    look-ahead limiter of that length in milliseconds (for example 5): the speech keeps the target level and only the
    samples around peaks above ``peak_ceiling`` are lowered.

    ``vad`` lets you reuse a loaded :class:`libspeech.SileroVad` (its model is loaded once); it must run at the
    sample rate of ``signal`` (8000 or 16000 Hz). Without it a VAD is created for ``sample_rate``. The signal is
    returned unchanged when no window counts as speech. 1-D only; the input array is not modified.
    """
    import numpy as np  # noqa: PLC0415 - NumPy stays an optional dependency of the package

    x = np.ascontiguousarray(signal, dtype=np.float32)
    if x.ndim != 1:
        msg = "speech_rms_normalize needs a 1-D (mono) signal"
        raise ValueError(msg)
    if mode not in ("hard", "soft"):
        msg = "mode must be 'hard' or 'soft'"
        raise ValueError(msg)
    if vad is None:
        vad = SileroVad(sample_rate=sample_rate)
    if threshold is None:
        threshold = vad.threshold()
    vad.reset()
    vad.process(x.tolist())
    probabilities = np.ascontiguousarray(vad.speech_probabilities(), dtype=np.float32)
    weights = speech_weights(probabilities, mode, threshold)
    lookahead = 0 if limiter_ms is None else max(1, round(limiter_ms * 1e-3 * sample_rate))
    return rms_normalize_weighted(
        x, weights, vad.window_samples, target_dbfs, peak_ceiling, clip_percentile, lookahead
    )


def frame_weights(weights: ArrayLike, window: int, hop_length: int, n_frames: int) -> NDArray[np.float32]:
    """Maps per-window VAD weights to feature frames, for :func:`libspeech.weighted_cmvn`.

    ``weights[k]`` belongs to samples ``[k * window, (k + 1) * window)``; feature frame ``t`` is centred on sample
    ``t * hop_length`` (``center=True``, as in :func:`libspeech.stft`) and takes the weight of the window that
    contains it (the last one for frames past the end).
    """
    import numpy as np  # noqa: PLC0415 - NumPy stays an optional dependency of the package

    w = np.ascontiguousarray(weights, dtype=np.float32)
    if w.ndim != 1 or w.size == 0:
        msg = "weights must be a non-empty 1-D array"
        raise ValueError(msg)
    if window <= 0 or hop_length <= 0 or n_frames < 0:
        msg = "window and hop_length must be > 0 and n_frames >= 0"
        raise ValueError(msg)
    index = np.minimum(np.arange(n_frames) * hop_length // window, w.size - 1)
    return w[index]
