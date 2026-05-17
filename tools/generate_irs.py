"""Synthesise a small impulse-response (IR) pack into tools/irs/.

Generates plate / room / hall / spring IRs from first principles (random
noise + decay envelope + spectral shaping). The IRs aren't replicas of
any commercial unit — they're physically-plausible synthetic
approximations that load cleanly into apply_reverb.py --ir-preset.

Why synthetic, not real samples:
  - No licensing friction (commercial IRs are usually proprietary)
  - The pack can ship with the repo
  - Predictable, reproducible spectral shape
  - Each IR is short (1-4 seconds) so convolution is fast

Run once at project setup:
  python tools/generate_irs.py
This writes WAV files into tools/irs/ that apply_reverb.py --ir-preset
can load by stem name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 48000


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _decay_envelope(length: int, decay_seconds: float, sr: int = SR) -> np.ndarray:
    """Exponential decay envelope. decay_seconds = time to drop -60 dB."""
    t = np.arange(length) / sr
    # -60 dB ≈ amplitude 1e-3; exp(-tau * decay_seconds) = 1e-3 → tau = 6.9 / decay_seconds
    tau = 6.9 / max(decay_seconds, 0.05)
    return np.exp(-tau * t)


def _shape(noise: np.ndarray, sr: int, lo_hz: float | None = None,
           hi_hz: float | None = None) -> np.ndarray:
    """Apply optional HP / LP shaping to the noise."""
    out = noise
    if lo_hz is not None:
        sos = butter(2, lo_hz / (sr / 2), btype="high", output="sos")
        out = sosfilt(sos, out, axis=0)
    if hi_hz is not None:
        sos = butter(2, hi_hz / (sr / 2), btype="low", output="sos")
        out = sosfilt(sos, out, axis=0)
    return out


def make_plate_short(sr: int = SR) -> np.ndarray:
    """Bright short plate — snare/vocal sit. 0.8s decay."""
    length = int(1.2 * sr)
    rng = _rng(11)
    L = rng.standard_normal(length) * 0.3
    R = rng.standard_normal(length) * 0.3
    env = _decay_envelope(length, 0.8, sr)
    # Plate tilt: HP at 250 Hz, LP at 7 kHz
    L = _shape(L, sr, 250, 7000) * env
    R = _shape(R, sr, 250, 7000) * env
    return np.stack([L, R], axis=1)


def make_plate_long(sr: int = SR) -> np.ndarray:
    """Big rock plate. 1.6s decay."""
    length = int(2.2 * sr)
    rng = _rng(12)
    L = rng.standard_normal(length) * 0.3
    R = rng.standard_normal(length) * 0.3
    env = _decay_envelope(length, 1.6, sr)
    L = _shape(L, sr, 200, 6000) * env
    R = _shape(R, sr, 200, 6000) * env
    return np.stack([L, R], axis=1)


def make_room_tight(sr: int = SR) -> np.ndarray:
    """Small dead-ish room. 0.35s decay, more LF, darker. Drum-bus glue."""
    length = int(0.6 * sr)
    rng = _rng(13)
    L = rng.standard_normal(length) * 0.3
    R = rng.standard_normal(length) * 0.3
    env = _decay_envelope(length, 0.35, sr)
    # Add an early reflection cluster — a few sparse spikes in the first 30 ms
    er_samples = [int(sr * t) for t in (0.005, 0.011, 0.018, 0.025)]
    er = np.zeros(length)
    for s, amp in zip(er_samples, (0.6, 0.4, 0.3, 0.2)):
        if s < length:
            er[s] = amp
    L = (L + er) * env
    R = (R + er * 0.9) * env  # slight decorrelation
    L = _shape(L, sr, 80, 6000)
    R = _shape(R, sr, 80, 6000)
    return np.stack([L, R], axis=1)


def make_room_live(sr: int = SR) -> np.ndarray:
    """Brighter, livelier medium room. 0.7s decay."""
    length = int(1.0 * sr)
    rng = _rng(14)
    L = rng.standard_normal(length) * 0.3
    R = rng.standard_normal(length) * 0.3
    env = _decay_envelope(length, 0.7, sr)
    er_samples = [int(sr * t) for t in (0.008, 0.017, 0.029, 0.041, 0.055)]
    er = np.zeros(length)
    for s, amp in zip(er_samples, (0.5, 0.4, 0.3, 0.22, 0.15)):
        if s < length: er[s] = amp
    L = (L + er) * env
    R = (R + er[::-1][:length] * 0.7) * env  # use reversed ER for R, decorrelation
    L = _shape(L, sr, 100, 9000)
    R = _shape(R, sr, 100, 9000)
    return np.stack([L, R], axis=1)


def make_hall_concert(sr: int = SR) -> np.ndarray:
    """Larger concert-hall IR. 2.5s decay, long pre-delay built in (30 ms gap)."""
    length = int(3.5 * sr)
    rng = _rng(15)
    L = rng.standard_normal(length) * 0.3
    R = rng.standard_normal(length) * 0.3
    env = _decay_envelope(length, 2.5, sr)
    # Insert a 30 ms gap before the diffuse tail (typical hall pre-delay)
    gap = int(0.030 * sr)
    L[:gap] = 0
    R[:gap] = 0
    # Rolloff highs more aggressively in a hall (air absorption)
    L = _shape(L, sr, 150, 5000) * env
    R = _shape(R, sr, 150, 5000) * env
    return np.stack([L, R], axis=1)


def make_spring(sr: int = SR) -> np.ndarray:
    """Guitar-amp spring reverb. Distinctive: very short, boingy, narrow,
    bright midrange. About 0.5s decay; HP@300, LP@5k. Mid emphasis at 1-2 kHz.
    """
    length = int(0.8 * sr)
    rng = _rng(16)
    mono = rng.standard_normal(length) * 0.3
    env = _decay_envelope(length, 0.5, sr)
    # Add discrete reflections to simulate the spring's metallic flutter
    flutter = np.zeros(length)
    for n in range(1, 15):
        s = int(0.012 * sr * n)
        if s < length:
            flutter[s] = (-1) ** n * 0.25 / n
    out = (mono + flutter) * env
    out = _shape(out, sr, 300, 5000)
    # Mid emphasis (peak around 1.5 kHz) — add a tilted copy
    sos_mid = butter(2, [800 / (sr / 2), 2500 / (sr / 2)], btype="band", output="sos")
    mid = sosfilt(sos_mid, out) * 0.5
    out = out + mid
    # Spring reverbs are usually mono → duplicate to stereo (full L=R correlation)
    return np.stack([out, out], axis=1)


IR_GENERATORS = {
    "plate_short":   make_plate_short,
    "plate_long":    make_plate_long,
    "room_tight":    make_room_tight,
    "room_live":     make_room_live,
    "hall_concert":  make_hall_concert,
    "spring_guitar": make_spring,
}


def main() -> None:
    out_dir = Path(__file__).parent / "irs"
    out_dir.mkdir(exist_ok=True)
    print(f"Writing IRs to {out_dir}")
    for name, fn in IR_GENERATORS.items():
        ir = fn()
        # Normalize to peak -3 dBFS so convolution doesn't blow up
        peak = float(np.max(np.abs(ir)))
        if peak > 0:
            ir = ir / peak * 10.0 ** (-3.0 / 20.0)
        out = out_dir / f"{name}.wav"
        sf.write(str(out), ir.astype(np.float64), SR, subtype="PCM_24")
        length_s = ir.shape[0] / SR
        print(f"  {name:<15} {length_s:.2f}s   {out}")


if __name__ == "__main__":
    main()
