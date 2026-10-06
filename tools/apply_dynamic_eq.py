"""Blend a bounded bell cut under a channel-linked RMS sidechain envelope.

Detector band: without --sidechain-path the input triggers itself through an
octave-wide band centred on --frequency-hz (so level elsewhere in the
spectrum does not trigger the cut). An external sidechain keeps the broad
80-5000 Hz detector. Explicit --detector-hp-hz / --detector-lp-hz override
either default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from apply_eq import filter_signal
from _recall import record_operation


def _control(data: np.ndarray, sr: int, threshold_db: float, attack_ms: float,
             release_ms: float, range_db: float) -> np.ndarray:
    """One millisecond linked RMS blocks, smoothed and interpolated to samples."""
    hop = max(1, round(sr / 1000))
    n = len(data)
    power = np.mean(np.square(data, dtype=np.float64), axis=1)
    power = np.pad(power, (0, (-n) % hop))
    rms = np.sqrt(power.reshape(-1, hop).mean(axis=1))
    attack = np.exp(-hop / (sr * attack_ms / 1000))
    release = np.exp(-hop / (sr * release_ms / 1000))
    levels = np.empty(len(rms))
    state = 0.0
    for index, level in enumerate(rms):
        coefficient = attack if level > state else release
        state = coefficient * state + (1 - coefficient) * level
        levels[index] = state
    weights = np.clip((20 * np.log10(np.maximum(levels, 1e-12)) - threshold_db) / range_db, 0, 1)
    # Interpolate block-end controls; retain zero before the first block.
    return np.interp(np.arange(n), (np.arange(len(weights)) + 1) * hop - 1,
                     weights, left=0).astype(np.float32)


@record_operation("apply_dynamic_eq")
def apply_dynamic_eq(input_path: Path, output_dir: Path, frequency_hz: float,
                     q: float = 1.0, max_cut_db: float = 2.0,
                     threshold_db: float = -30.0, attack_ms: float = 10.0,
                     release_ms: float = 120.0, range_db: float = 12.0,
                     sidechain_path: Path | None = None,
                     detector_hp_hz: float | None = None,
                     detector_lp_hz: float | None = None) -> dict:
    if not np.isfinite(frequency_hz):
        raise ValueError("Dynamic EQ parameters must be finite")
    data, sr = sf.read(input_path, always_2d=True, dtype="float32")
    if sidechain_path is None:
        default_hp, default_lp = frequency_hz / np.sqrt(2), min(frequency_hz * np.sqrt(2), 0.45 * sr)
    else:
        default_hp, default_lp = 80.0, 5000.0
    detector_hp_hz = float(default_hp if detector_hp_hz is None else detector_hp_hz)
    detector_lp_hz = float(default_lp if detector_lp_hz is None else detector_lp_hz)
    values = (q, max_cut_db, threshold_db, attack_ms, release_ms,
              range_db, detector_hp_hz, detector_lp_hz)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Dynamic EQ parameters must be finite")
    if not data.size or data.shape[1] not in (1, 2) or not np.isfinite(data).all():
        raise ValueError("Expected finite nonempty mono/stereo input")
    if not (0 < frequency_hz < sr / 2 and q > 0 and 0 <= max_cut_db <= 12
            and attack_ms > 0 and release_ms > 0 and range_db > 0
            and 0 < detector_hp_hz < detector_lp_hz < sr / 2):
        raise ValueError("Invalid band, cut depth, detector range, or envelope timing")
    if sidechain_path is None:
        detector = data.copy()
    else:
        detector, sc_sr = sf.read(sidechain_path, always_2d=True, dtype="float32")
        if sc_sr != sr or len(detector) != len(data):
            raise ValueError("Sidechain must have matching sample rate and frame count")
        if detector.shape[1] not in (1, 2) or not np.isfinite(detector).all():
            raise ValueError("Expected finite mono/stereo sidechain")
    detector = sosfilt(butter(2, [detector_hp_hz, detector_lp_hz], btype="bandpass", fs=sr,
                              output="sos"), detector, axis=0)
    weight = _control(detector, sr, threshold_db, attack_ms, release_ms, range_db)
    filtered = filter_signal(data, sr, {"type": "peak", "hz": frequency_hz,
                             "db": -max_cut_db, "q": q}, axis=0).astype(np.float32)
    result = data + weight[:, None] * (filtered - data)
    if not np.isfinite(result).all():
        raise ValueError("Dynamic EQ produced nonfinite samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{input_path.stem}_dynamic_eq.wav"
    sf.write(output, result, sr, subtype="FLOAT")
    cut = -20 * np.log10(np.maximum(1 - weight + weight * 10 ** (-max_cut_db / 20), 1e-12))
    report = {"input": str(input_path), "output": str(output),
              "sidechain": str(sidechain_path) if sidechain_path else None,
              "sample_rate": sr, "frames": len(data), "channels": data.shape[1],
              "frequency_hz": frequency_hz, "q": q, "max_cut_db": max_cut_db,
              "threshold_db": threshold_db, "attack_ms": attack_ms, "release_ms": release_ms,
              "range_db": range_db, "detector_hp_hz": detector_hp_hz, "detector_lp_hz": detector_lp_hz,
              "method": "Linear blend between dry and fixed minimum-phase bell; stereo-linked control",
              "max_center_cut_db": round(float(np.max(cut)), 3),
              "mean_center_cut_db": round(float(np.mean(cut)), 3),
              "active_fraction": round(float(np.mean(cut > 0.1)), 4),
              "listening_review": "pending"}
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frequency-hz", type=float, required=True)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--max-cut-db", type=float, default=2.0)
    parser.add_argument("--threshold-db", type=float, default=-30.0)
    parser.add_argument("--attack-ms", type=float, default=10.0)
    parser.add_argument("--release-ms", type=float, default=120.0)
    parser.add_argument("--range-db", type=float, default=12.0)
    parser.add_argument("--sidechain-path", type=Path)
    parser.add_argument("--detector-hp-hz", type=float, default=None,
                        help="Detector high-pass (default: frequency/sqrt(2) self-triggered, 80 with sidechain)")
    parser.add_argument("--detector-lp-hz", type=float, default=None,
                        help="Detector low-pass (default: frequency*sqrt(2) self-triggered, 5000 with sidechain)")
    print(json.dumps(apply_dynamic_eq(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
