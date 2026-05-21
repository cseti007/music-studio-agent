"""Frequency-specific sidechain compressor for vocal sibilance control.

A de-esser works by listening to a narrow band (here ~5-8 kHz where 's' / 'ch'
/ 't' sibilants live) and applying gain reduction to the full signal when the
sibilance band exceeds a threshold. The signal path stays untouched; only the
detection sidechain is band-filtered.

Chain placement (per the 2026 vocal-mixing consensus):
    EQ-subtractive → compression → DE-ESSER → EQ-additive → ...
The de-esser runs AFTER the compressor on purpose. Compression amplifies
sibilance peaks; the de-esser catches them at the comp's output where they
are at their worst.

Presets:
  deesser_smooth         — gentle, broadband sibilance control
  deesser_aggressive     — harder, for heavily-compressed pop vocals
  deesser_male_lead      — lower detection band (4-7 kHz)
  deesser_female_lead    — higher detection band (6-9 kHz)

Usage:
  python tools/apply_deesser.py vocal.wav --preset deesser_smooth --output-dir DIR
  python tools/apply_deesser.py vocal.wav --threshold -22 --ratio 4 \
      --detect-low 5500 --detect-high 8500 --output-dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


PRESETS_DIR = Path(__file__).parent / "presets"

PRESETS: dict[str, dict] = {
    "deesser_smooth": {
        "description": "Gentle broadband de-essing — preserves the natural vocal character",
        "notes": "Threshold -22 dBFS, 3:1 ratio, 1 ms attack, 60 ms release. Detection band 5.5-8.5 kHz covers most adult voices. Use as a default starting point.",
        "threshold_db": -22.0,
        "ratio": 3.0,
        "attack_ms": 1.0,
        "release_ms": 60.0,
        "detect_low_hz": 5500.0,
        "detect_high_hz": 8500.0,
    },
    "deesser_aggressive": {
        "description": "Hard de-essing for heavily-compressed pop vocals",
        "notes": "Threshold -26 dBFS, 6:1 ratio, 0.5 ms attack, 40 ms release. The lower threshold and faster ballistics catch every harsh sibilance the comp amplified. Use when the lead vocal sits on top of a dense modern-pop mix.",
        "threshold_db": -26.0,
        "ratio": 6.0,
        "attack_ms": 0.5,
        "release_ms": 40.0,
        "detect_low_hz": 5500.0,
        "detect_high_hz": 9000.0,
    },
    "deesser_male_lead": {
        "description": "Male lead vocal — lower detection band (4-7 kHz)",
        "notes": "Male voices typically have lower-frequency sibilance than female. Detection 4-7 kHz catches male 's' (often centred ~5 kHz). Threshold -22 dBFS, 3.5:1 ratio.",
        "threshold_db": -22.0,
        "ratio": 3.5,
        "attack_ms": 1.0,
        "release_ms": 60.0,
        "detect_low_hz": 4000.0,
        "detect_high_hz": 7000.0,
    },
    "deesser_female_lead": {
        "description": "Female lead vocal — higher detection band (6-9 kHz)",
        "notes": "Female voices typically have higher-frequency sibilance. Detection 6-9 kHz catches the brighter 's' (often centred 7-8 kHz). Threshold -22 dBFS, 3.5:1 ratio.",
        "threshold_db": -22.0,
        "ratio": 3.5,
        "attack_ms": 1.0,
        "release_ms": 60.0,
        "detect_low_hz": 6000.0,
        "detect_high_hz": 9000.0,
    },
}


def _band_pass(signal: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """4th-order Butterworth bandpass for the sidechain detection signal."""
    high_hz = min(high_hz, sr / 2 - 1)
    sos = butter(4, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    return sosfilt(sos, signal)


def _envelope_follower(
    sc_band: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    """Block-based peak detection + exponential ballistics. Returns per-sample
    gain multiplier [0..1] — exactly the shape compatible with sosfilt-style
    sample-by-sample application.

    Block size = 1 ms (sr/1000) for responsive but stable detection.
    """
    n = len(sc_band)
    block_n = max(1, sr // 1000)
    n_blocks = (n + block_n - 1) // block_n

    alpha_a = np.exp(-1.0 / max(1.0, attack_ms * 0.001 * sr / block_n))
    alpha_r = np.exp(-1.0 / max(1.0, release_ms * 0.001 * sr / block_n))

    threshold_lin = 10.0 ** (threshold_db / 20.0)
    inv_ratio = 1.0 / ratio

    gr_blocks = np.ones(n_blocks)
    env = 0.0

    for i in range(n_blocks):
        start = i * block_n
        end = min(start + block_n, n)
        level = float(np.max(np.abs(sc_band[start:end])))

        if level > env:
            env = alpha_a * env + (1.0 - alpha_a) * level
        else:
            env = alpha_r * env + (1.0 - alpha_r) * level

        if env > threshold_lin:
            env_db = 20.0 * np.log10(max(env, 1e-10))
            gr_db = (threshold_db + (env_db - threshold_db) * inv_ratio) - env_db
            gr_blocks[i] = 10.0 ** (gr_db / 20.0)

    block_centers = np.arange(n_blocks) * block_n + block_n // 2
    gain = np.interp(np.arange(n), block_centers, gr_blocks)
    return gain.astype(np.float64)


def _load_preset(name: str) -> dict:
    if name in PRESETS:
        return PRESETS[name]
    path = PRESETS_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unknown preset: {name}")


def _relevance_check(signal: np.ndarray, sr: int, detect_low_hz: float, detect_high_hz: float) -> dict:
    """Detect whether the input actually has sibilance worth removing.

    Returns:
      {recommend_skip: bool, reasons: [str], sibilance_peak_dbfs: float}
    """
    sc = _band_pass(signal, sr, detect_low_hz, detect_high_hz)
    peak = float(np.max(np.abs(sc)))
    peak_db = 20.0 * np.log10(max(peak, 1e-10))
    issues: list[str] = []

    if peak_db < -25.0:
        issues.append(f"sibilance band peak {peak_db:.1f} dBFS < -25 — nothing to de-ess")

    return {
        "recommend_skip": bool(issues),
        "reasons": issues,
        "sibilance_peak_dbfs": round(peak_db, 1),
    }


def apply_deesser(
    input_path: Path,
    output_dir: Path,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    detect_low_hz: float,
    detect_high_hz: float,
    preset_name: str | None = None,
    force: bool = False,
) -> dict:
    data, sr = sf.read(str(input_path), always_2d=True)
    channels = data.shape[1]
    mono = data.mean(axis=1).astype(np.float64) if channels > 1 else data[:, 0]

    rel = _relevance_check(mono, sr, detect_low_hz, detect_high_hz)
    if rel["recommend_skip"] and not force:
        report = {
            "input": str(input_path),
            "output": None,
            "preset": preset_name,
            "settings": {
                "threshold_db": threshold_db, "ratio": ratio,
                "attack_ms": attack_ms, "release_ms": release_ms,
                "detect_low_hz": detect_low_hz, "detect_high_hz": detect_high_hz,
            },
            "relevance_check": rel,
            "skipped": True,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "deesser_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return report

    # Compute gain envelope from the sidechain band, then apply to full signal
    sc_band = _band_pass(mono, sr, detect_low_hz, detect_high_hz)
    gain = _envelope_follower(sc_band, sr, threshold_db, ratio, attack_ms, release_ms)

    out = data.copy().astype(np.float64)
    for ch in range(channels):
        out[:, ch] *= gain

    in_peak = float(np.max(np.abs(data)))
    out_peak = float(np.max(np.abs(out)))
    in_peak_db = 20.0 * np.log10(max(in_peak, 1e-10))
    out_peak_db = 20.0 * np.log10(max(out_peak, 1e-10))

    # GR stats
    gr_db = 20.0 * np.log10(np.maximum(gain, 1e-10))
    mean_gr = float(np.mean(gr_db))
    peak_gr = float(np.min(gr_db))   # most negative = most reduction

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (input_path.stem + "_deessed.wav")
    sf.write(str(output_path), out, sr, subtype="PCM_24")

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "preset": preset_name,
        "settings": {
            "threshold_db": threshold_db, "ratio": ratio,
            "attack_ms": attack_ms, "release_ms": release_ms,
            "detect_low_hz": detect_low_hz, "detect_high_hz": detect_high_hz,
        },
        "relevance_check": rel,
        "mean_gain_reduction_db": round(mean_gr, 2),
        "peak_gain_reduction_db": round(peak_gr, 2),
        "input_peak_dbfs": round(in_peak_db, 1),
        "output_peak_dbfs": round(out_peak_db, 1),
        "sample_rate": sr,
        "skipped": False,
    }
    (output_dir / "deesser_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frequency-specific sidechain de-esser for vocal sibilance control."
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input vocal WAV")
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--preset", choices=list(PRESETS), default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--attack", type=float, default=None)
    parser.add_argument("--release", type=float, default=None)
    parser.add_argument("--detect-low", type=float, default=None)
    parser.add_argument("--detect-high", type=float, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Bypass the relevance check (process even when sibilance is below the threshold)")
    parser.add_argument("--list-presets", action="store_true")
    args = parser.parse_args()

    if args.list_presets:
        for name, p in PRESETS.items():
            print(f"  {name:<24} thr={p['threshold_db']:+.1f}  ratio={p['ratio']}:1  "
                  f"detect={p['detect_low_hz']}-{p['detect_high_hz']} Hz")
        return

    if not args.input or not args.output_dir:
        parser.error("input and --output-dir required")

    # Resolve preset / explicit args
    if args.preset:
        p = _load_preset(args.preset)
        threshold = p["threshold_db"]
        ratio = p["ratio"]
        attack = p["attack_ms"]
        release = p["release_ms"]
        detect_low = p["detect_low_hz"]
        detect_high = p["detect_high_hz"]
    else:
        threshold = args.threshold if args.threshold is not None else -22.0
        ratio = args.ratio if args.ratio is not None else 3.0
        attack = args.attack if args.attack is not None else 1.0
        release = args.release if args.release is not None else 60.0
        detect_low = args.detect_low if args.detect_low is not None else 5500.0
        detect_high = args.detect_high if args.detect_high is not None else 8500.0

    # Allow per-flag overrides on top of a preset
    if args.threshold is not None: threshold = args.threshold
    if args.ratio is not None: ratio = args.ratio
    if args.attack is not None: attack = args.attack
    if args.release is not None: release = args.release
    if args.detect_low is not None: detect_low = args.detect_low
    if args.detect_high is not None: detect_high = args.detect_high

    apply_deesser(
        args.input, args.output_dir,
        threshold_db=threshold, ratio=ratio,
        attack_ms=attack, release_ms=release,
        detect_low_hz=detect_low, detect_high_hz=detect_high,
        preset_name=args.preset, force=args.force,
    )


if __name__ == "__main__":
    main()
