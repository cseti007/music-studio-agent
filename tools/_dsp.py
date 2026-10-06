"""Shared low-level DSP helpers used across the mix/master tools.

Kept dependency-light (numpy + scipy only) so the analysis and health tools
can import it without pulling in heavy plotting/ML dependencies. The canonical
true-peak measurement lives here so it cannot drift between tools — previously
each of analyze / mix_health / master_health carried its own copy, and one of
them measured the true peak on the L+R monosum, which let a hard-panned
inter-sample peak slip past the master gate (~6 dB underestimate).
"""

from __future__ import annotations

import operator

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
    try:
        oversample = operator.index(oversample)
    except TypeError:
        raise ValueError("Oversampling factor must be a positive integer") from None
    if oversample < 1:
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


def linked_gain(buf: np.ndarray, processor, sr: int) -> np.ndarray:
    """Stereo-linked gain curve of a pedalboard dynamics processor.

    pedalboard's Compressor/Limiter detect each channel separately, which
    shifts the stereo image when only one side is loud. Running the
    processor on the rectified per-sample channel maximum and taking
    output/input yields one gain curve that follows the loudest channel.
    buf is (channels, samples); returns a (samples,) gain array.
    """
    detector = np.abs(np.asarray(buf, dtype=np.float32)).max(axis=0)
    out = processor(detector[np.newaxis, :], sr)[0]
    active = detector > 1e-9
    if not active.any():
        return np.ones(detector.shape, dtype=np.float64)
    idx = np.flatnonzero(active)
    gain = out[idx].astype(np.float64) / detector[idx]
    # Zero-valued detector samples carry no gain information: interpolate.
    return np.interp(np.arange(detector.size), idx, gain)


def brickwall_limit(buf: np.ndarray, sr: int, ceiling_db: float,
                    release_ms: float = 100.0, lookahead_ms: float = 5.0,
                    oversample: int = 8) -> tuple[np.ndarray, float]:
    """Stereo-linked lookahead true-peak limiter with a verified ceiling.

    Uses pedalboard.BrickwallLimiter (no makeup gain, 4x true-peak
    detection). Its documentation does not guarantee the reconstructed
    ceiling, so the result is measured at `oversample` and any residual
    overshoot is removed with a static gain trim. buf is (channels, samples).
    Returns (limited buffer, applied safety trim in dB, <= 0).
    """
    from pedalboard import BrickwallLimiter

    limiter = BrickwallLimiter(ceiling_db=float(ceiling_db), release_ms=float(release_ms),
                               lookahead_ms=float(lookahead_ms), true_peak=True)
    out = limiter(np.asarray(buf, dtype=np.float32), sr).astype(np.float64)
    target = float(ceiling_db) - 0.01
    trim_db = min(0.0, target - worst_channel_true_peak_dbfs(out, oversample))
    if trim_db < 0.0:
        out *= 10.0 ** (trim_db / 20.0)
    return out, trim_db
