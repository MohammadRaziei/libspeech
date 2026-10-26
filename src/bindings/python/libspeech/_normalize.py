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

__all__ = ["speech_rms_normalize"]


def speech_rms_normalize(
    signal: ArrayLike,
    *,
    sample_rate: int = 16000,
    mode: str = "hard",
    threshold: float | None = None,
    target_dbfs: float = -20.0,
    peak_ceiling: float = 0.99,
    vad: SileroVad | None = None,
) -> NDArray[np.float32]:
    """Scales a mono signal so that its *speech* level is ``target_dbfs``.

    ``mode="hard"`` counts a window as speech (weight 1) when its probability reaches ``threshold``, otherwise it
    does not count (weight 0). ``mode="soft"`` gives a window the weight ``(p - threshold) / (1 - threshold)``
    clamped to [0, 1], so a window just above the threshold counts little and a certain one counts fully.
    ``threshold`` defaults to the threshold of the VAD (0.5 unless you changed it).

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
    return rms_normalize_weighted(
        x, weights, vad.window_samples, target_dbfs, peak_ceiling
    )
