"""Align phase between two assembled stems via cross-correlation.

Detects the sample delay and polarity relationship between a reference and
target stem, then applies the correction to produce an aligned output.

The correlation is computed on the first active segment of the recording
(configurable length) for efficiency. The correction is applied to the
full stem.

Typical use cases in a drum session:
  KICK OUT  → reference: KICK IN
  KICK SUB  → reference: KICK IN
  SN BOTTOM → reference: SN TOP  (almost always needs polarity flip)
  OH AEA R  → reference: OH AEA L
  ROOM mics → reference: KICK IN or SN TOP

Usage:
  python align_phase.py \\
      --reference "output/<session>/KICK IN.05/assembled.wav" \\
      --target    "output/<session>/KICK OUT.05/assembled.wav" \\
      --output-dir output/<session>

Reads [align] section from config.toml in the current working directory.
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate

_CONFIG_PATH = Path("config.toml")

DEFAULT_MAX_DELAY_MS    = 20.0
DEFAULT_SEGMENT_SEC     = 10.0
DEFAULT_ACTIVE_THRESHOLD_DB = -40.0


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    return {}


def _find_active_segment(signal: np.ndarray, sr: int, duration_sec: float, threshold_db: float) -> tuple[np.ndarray, int]:
    """Return the first `duration_sec` of audio where RMS exceeds threshold.

    Falls back to the beginning of the file if no active region is found.
    Returns (segment, start_sample).
    """
    threshold_linear = 10 ** (threshold_db / 20.0)
    frame_len = int(sr * 0.05)  # 50ms frames
    n_frames = (len(signal) - frame_len) // frame_len

    for i in range(n_frames):
        start = i * frame_len
        frame = signal[start : start + frame_len]
        if np.sqrt(np.mean(frame ** 2)) > threshold_linear:
            end = min(start + int(sr * duration_sec), len(signal))
            return signal[start:end], start

    end = min(int(sr * duration_sec), len(signal))
    return signal[:end], 0


def _parabolic_refine(corr_abs: np.ndarray, peak_idx: int) -> float:
    """Sub-sample peak offset from a 3-point parabolic fit around peak_idx.

    Given the discrete cross-correlation magnitudes y0, y1, y2 at three
    consecutive lags (peak_idx-1, peak_idx, peak_idx+1), fits a parabola and
    returns the offset (-0.5..+0.5) of the true continuous peak from peak_idx.

    Returns 0.0 if peak is at the edge or the fit is degenerate.
    """
    if peak_idx <= 0 or peak_idx >= len(corr_abs) - 1:
        return 0.0
    y0 = float(corr_abs[peak_idx - 1])
    y1 = float(corr_abs[peak_idx])
    y2 = float(corr_abs[peak_idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return 0.0
    offset = (y0 - y2) / (2.0 * denom)
    # Clamp — a value outside ±0.5 means we picked the wrong peak bin
    return float(np.clip(offset, -0.5, 0.5))


def _compute_alignment(
    ref: np.ndarray,
    tgt: np.ndarray,
    sr: int,
    max_delay_ms: float,
) -> tuple[float, bool, float]:
    """Compute sub-sample delay and polarity between ref and tgt.

    Returns (delay_samples, polarity_flip, correlation_score).

    delay_samples > 0: tgt arrived AFTER ref → shift tgt earlier
    delay_samples < 0: tgt arrived BEFORE ref → shift tgt later
    polarity_flip: True if signals are inverted relative to each other
    correlation_score: normalised peak correlation [-1, 1]

    Note: scipy.signal.correlate(ref, tgt) peaks at lag = -delay when tgt is
    late relative to ref, so we negate the lag to expose the human-friendly
    convention (positive = tgt later).
    """
    max_samples = int(max_delay_ms * sr / 1000.0)

    ref_n = ref / (np.std(ref) + 1e-10)
    tgt_n = tgt / (np.std(tgt) + 1e-10)

    corr = correlate(ref_n, tgt_n, mode="full", method="fft")
    n = len(ref_n)
    lags = np.arange(-(n - 1), len(tgt_n))

    # restrict search window
    mask = np.abs(lags) <= max_samples
    corr_abs = np.abs(corr)
    restricted = np.where(mask, corr_abs, -np.inf)
    peak_idx = int(np.argmax(restricted))

    # Sub-sample refinement via parabolic interpolation around the peak
    sub_offset = _parabolic_refine(corr_abs, peak_idx)

    # Negate so delay > 0 means tgt is later than ref (matches docstring).
    delay_samples = -(float(lags[peak_idx]) + sub_offset)
    polarity_flip = bool(corr[peak_idx] < 0)
    correlation_score = float(corr[peak_idx] / max(n, 1))

    return delay_samples, polarity_flip, correlation_score


def _integer_shift(signal: np.ndarray, samples: int) -> np.ndarray:
    """Shift by integer samples. Positive = delay (later), negative = advance (earlier)."""
    if samples == 0:
        return signal.copy()
    result = np.zeros_like(signal)
    if samples > 0:
        d = min(samples, len(signal))
        result[d:] = signal[: len(signal) - d]
    else:
        d = min(-samples, len(signal))
        result[: len(signal) - d] = signal[d:]
    return result


def _fractional_shift(signal: np.ndarray, samples: float, n_taps: int = 31) -> np.ndarray:
    """Shift `signal` by `samples` (can be non-integer) using a windowed-sinc FIR.

    Positive `samples` = delay (later); negative = advance (earlier).
    Integer part is handled by an integer shift; the fractional part is
    realised by a Hann-windowed sinc impulse of length n_taps.
    """
    int_part = int(np.floor(samples))
    frac = float(samples - int_part)
    if abs(frac) < 1e-6:
        return _integer_shift(signal, int_part)

    # Windowed-sinc fractional delay FIR. sinc(k - frac) shifts by `frac` samples.
    k = np.arange(-(n_taps // 2), n_taps // 2 + 1)
    h = np.sinc(k - frac) * np.hanning(n_taps)
    h /= h.sum()
    filtered = np.convolve(signal, h, mode="same")
    return _integer_shift(filtered, int_part)


def _apply_correction(signal: np.ndarray, delay_samples: float, polarity_flip: bool) -> np.ndarray:
    """Apply sub-sample delay correction and optional polarity flip.

    `delay_samples > 0` means tgt is late vs ref → we advance tgt by that much
    (shift earlier in time), i.e. apply a negative-direction shift.
    """
    out = -signal.copy() if polarity_flip else signal.copy()
    if abs(delay_samples) < 1e-9:
        return out
    # _fractional_shift convention: positive samples = delay; we want to advance.
    return _fractional_shift(out, -delay_samples)


def align_phase(
    reference_path: Path,
    target_path: Path,
    output_dir: Path,
    max_delay_ms: float = DEFAULT_MAX_DELAY_MS,
    segment_sec: float = DEFAULT_SEGMENT_SEC,
    threshold_db: float = DEFAULT_ACTIVE_THRESHOLD_DB,
) -> dict:
    ref_data, ref_sr = sf.read(str(reference_path), always_2d=True)
    tgt_data, tgt_sr = sf.read(str(target_path), always_2d=True)

    if ref_sr != tgt_sr:
        print(
            f"WARNING: sample rate mismatch — ref {ref_sr} Hz, tgt {tgt_sr} Hz",
            file=sys.stderr,
        )

    ref_mono = ref_data.mean(axis=1)
    tgt_mono = tgt_data.mean(axis=1)

    # pad to same length before correlation
    n = max(len(ref_mono), len(tgt_mono))
    ref_padded = np.pad(ref_mono, (0, n - len(ref_mono)))
    tgt_padded = np.pad(tgt_mono, (0, n - len(tgt_mono)))

    ref_seg, _ = _find_active_segment(ref_padded, ref_sr, segment_sec, threshold_db)
    tgt_seg, _ = _find_active_segment(tgt_padded, tgt_sr, segment_sec, threshold_db)

    # trim both segments to same length for correlation
    seg_len = min(len(ref_seg), len(tgt_seg))
    ref_seg, tgt_seg = ref_seg[:seg_len], tgt_seg[:seg_len]

    delay_samples, polarity_flip, correlation_score = _compute_alignment(
        ref_seg, tgt_seg, ref_sr, max_delay_ms
    )

    delay_ms = round(delay_samples / ref_sr * 1000.0, 3)

    # warn if delay hits the search boundary (may indicate a larger issue)
    max_samples = int(max_delay_ms * ref_sr / 1000.0)
    if abs(delay_samples) >= max_samples:
        print(
            f"WARNING: detected delay {delay_ms} ms hit the search boundary "
            f"(±{max_delay_ms} ms) — result may be unreliable",
            file=sys.stderr,
        )

    # apply correction to full target (all channels)
    corrected_channels = []
    for ch in range(tgt_data.shape[1]):
        ch_signal = np.pad(tgt_data[:, ch], (0, n - len(tgt_data[:, ch])))
        corrected = _apply_correction(ch_signal, delay_samples, polarity_flip)
        corrected_channels.append(corrected[:len(tgt_data)])

    corrected_data = np.stack(corrected_channels, axis=1)

    stem_dir = output_dir / target_path.parent.name
    stem_dir.mkdir(parents=True, exist_ok=True)
    out_path = stem_dir / "assembled_aligned.wav"
    sf.write(str(out_path), corrected_data, tgt_sr, subtype="PCM_24")

    report = {
        "reference": str(reference_path),
        "target": str(target_path),
        "output": str(out_path),
        "delay_samples": round(delay_samples, 3),
        "delay_ms": delay_ms,
        "subsample_alignment": True,
        "polarity_flip": polarity_flip,
        "correlation_score": round(correlation_score, 4),
        "segment_sec_used": round(seg_len / ref_sr, 2),
        "max_delay_ms": max_delay_ms,
    }

    report_path = stem_dir / "align_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    cfg = _load_config().get("align", {})

    parser = argparse.ArgumentParser(description="Phase-align a target stem to a reference stem.")
    parser.add_argument("--reference", type=Path, required=True, metavar="WAV", help="Reference assembled stem")
    parser.add_argument("--target",    type=Path, required=True, metavar="WAV", help="Target stem to align")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--max-delay-ms", type=float,
        default=cfg.get("max_delay_ms", DEFAULT_MAX_DELAY_MS),
        help=f"Max search range in ms (default: {DEFAULT_MAX_DELAY_MS})",
    )
    parser.add_argument(
        "--segment-sec", type=float,
        default=cfg.get("segment_duration_sec", DEFAULT_SEGMENT_SEC),
        help=f"Active segment length used for correlation (default: {DEFAULT_SEGMENT_SEC}s)",
    )
    args = parser.parse_args()

    for p in (args.reference, args.target):
        if not p.exists():
            print(json.dumps({"error": f"Not found: {p}"}), file=sys.stderr)
            sys.exit(1)

    align_phase(
        reference_path=args.reference,
        target_path=args.target,
        output_dir=args.output_dir,
        max_delay_ms=args.max_delay_ms,
        segment_sec=args.segment_sec,
    )


if __name__ == "__main__":
    main()
