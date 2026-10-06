"""Shared low-level DSP helpers used across the mix/master tools.

Kept dependency-light (numpy + scipy only) so the analysis and health tools
can import it without pulling in heavy plotting/ML dependencies. The canonical
true-peak measurement lives here so it cannot drift between tools — previously
each of analyze / mix_health / master_health carried its own copy, and one of
them measured the true peak on the L+R monosum, which let a hard-panned
inter-sample peak slip past the master gate (~6 dB underestimate).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def true_peak_dbfs(signal: np.ndarray, oversample: int = 4) -> float:
    """Polyphase estimate of waveform true peak (dBTP) for one channel.

    Oversampling estimates inter-sample peaks in this waveform. It does not
    simulate codec encoding and has not been certified against BS.1770 tests.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
        raise ValueError("True peak requires nonempty finite mono audio")
    if not isinstance(oversample, int) or oversample < 1:
        raise ValueError("Oversampling factor must be a positive integer")
    up = resample_poly(signal, oversample, 1)
    peak = float(np.max(np.abs(up)))
    return float(20.0 * np.log10(max(peak, 1e-12)))


def worst_channel_true_peak_dbfs(data: np.ndarray, oversample: int = 4) -> float:
    """True peak (dBTP) of the loudest individual channel.

    Measuring per-channel — not on the L+R monosum — is essential: a hard-
    panned full-scale transient is ~6 dB quieter after summing to mono and
    would otherwise slip past a peak gate. Accepts mono (1-D) or any 2-D
    layout; channels are detected as the smaller axis (safe for audio, where
    the sample count always vastly exceeds the channel count).
    """
    arr = np.asarray(data)
    if arr.ndim not in (1, 2) or not arr.size:
        raise ValueError("Audio must be a nonempty mono or two-dimensional array")
    if arr.ndim == 1:
        return true_peak_dbfs(arr, oversample)
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T  # orient so axis 0 = channels
    return max(true_peak_dbfs(arr[ch], oversample) for ch in range(arr.shape[0]))
