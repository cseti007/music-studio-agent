"""Synthesize harmonics from a stem's sub-bass content so it survives on
small speakers (phones, laptops, BT speakers, car radios).

Psychoacoustic principle: the brain reconstructs a "missing fundamental"
from its harmonics. By generating the 2nd and 3rd harmonic of the 40-80 Hz
fundamental and mixing them back in lightly, the bass register feels deep
even on speakers that physically cannot reproduce sub frequencies.

DSP chain:
  1. Bandpass the input around the sub region (40-80 Hz default)
  2. Push it through tanh saturation to generate harmonics
  3. Bandpass the harmonics back into the 80-200 Hz region
  4. Mix the harmonics into the original signal at low level (10-25%)

Relevance check (the tool refuses to do anything useful otherwise):
  - The stem must actually have sub-bass content (sub_60hz band >= -35 dBFS).
    Adding harmonics from silence is just noise.
  - The crest of the sub band must be > 8 dB. If the sub is already
    massively compressed, harmonic generation will just add mud.

The relevance_check result is written into subharm_report.json. If it flags
recommend_skip:true, the tool prints a warning and exits without writing
audio unless --force is passed.

Usage:
  python apply_subharm.py bass.wav --output-dir output/session/BASS \\
      --preset subharm_subtle

  # explicit control
  python apply_subharm.py bass.wav --output-dir DIR \\
      --fundamental-hz-low 40 --fundamental-hz-high 80 \\
      --harmonic-mix 0.15
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


PRESETS: dict[str, dict] = {
    "subharm_subtle": {
        "description": "Subtle low-end weight for bass DI — phone/laptop translation",
        "fundamental_hz_low": 40.0,
        "fundamental_hz_high": 80.0,
        "harmonic_hz_low": 80.0,
        "harmonic_hz_high": 200.0,
        "drive": 1.8,
        "harmonic_mix": 0.15,
    },
    "subharm_kick": {
        "description": "Add 80-160 Hz weight to a kick that's all click and no body",
        "fundamental_hz_low": 50.0,
        "fundamental_hz_high": 100.0,
        "harmonic_hz_low": 100.0,
        "harmonic_hz_high": 220.0,
        "drive": 2.2,
        "harmonic_mix": 0.20,
    },
    "subharm_strong": {
        "description": "Strong sub-bass enhancement — use sparingly, easy to overdo",
        "fundamental_hz_low": 30.0,
        "fundamental_hz_high": 80.0,
        "harmonic_hz_low": 80.0,
        "harmonic_hz_high": 200.0,
        "drive": 2.5,
        "harmonic_mix": 0.25,
    },
}


# ---------------------------------------------------------------------------
# Relevance check — does sub-synth make sense for this stem?
# ---------------------------------------------------------------------------

_MIN_SUB_RMS_DB = -35.0
_MIN_SUB_CREST_DB = 8.0


def _band_filter(signal: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    nyq = sr / 2.0
    sos = butter(4, [lo / nyq, min(hi / nyq, 0.999)], btype="band", output="sos")
    return sosfilt(sos, signal)


def _relevance_check(signal: np.ndarray, sr: int) -> dict:
    sub = _band_filter(signal, sr, 30.0, 80.0)
    rms = float(np.sqrt(np.mean(sub ** 2)))
    peak = float(np.max(np.abs(sub)))
    rms_db = 20.0 * np.log10(max(rms, 1e-10))
    crest_db = 20.0 * np.log10(peak / max(rms, 1e-10)) if rms > 1e-10 else 0.0

    issues = []
    if rms_db < _MIN_SUB_RMS_DB:
        issues.append(
            f"sub band RMS {rms_db:.1f} dBFS < {_MIN_SUB_RMS_DB} — "
            f"nothing in the sub region to extract harmonics from"
        )
    if crest_db < _MIN_SUB_CREST_DB:
        issues.append(
            f"sub band crest {crest_db:.1f} dB < {_MIN_SUB_CREST_DB} — "
            f"sub is already squashed; harmonics will just add mud"
        )

    return {
        "tool": "subharm",
        "sub_band_rms_dbfs": round(rms_db, 1),
        "sub_band_crest_db": round(crest_db, 1),
        "recommend_skip": bool(issues),
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def apply_subharm(
    input_path: Path,
    output_dir: Path,
    fundamental_hz_low: float,
    fundamental_hz_high: float,
    harmonic_hz_low: float,
    harmonic_hz_high: float,
    drive: float,
    harmonic_mix: float,
    preset_name: str | None = None,
    force: bool = False,
) -> dict:
    data, sr = sf.read(str(input_path), always_2d=True)
    mono = data.mean(axis=1)

    rel = _relevance_check(mono, sr)
    if rel["recommend_skip"] and not force:
        print(
            f"WARNING: sub-synth relevance_check failed for {input_path.name}:",
            file=sys.stderr,
        )
        for issue in rel["issues"]:
            print(f"  - {issue}", file=sys.stderr)
        print("  Skipping (pass --force to apply anyway).", file=sys.stderr)
        report = {
            "input": str(input_path),
            "output": None,
            "preset": preset_name,
            "relevance_check": rel,
            "applied": False,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "subharm_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report

    # Process each channel independently
    out_channels = []
    for ch in range(data.shape[1]):
        signal = data[:, ch].astype(np.float64)
        # 1. Isolate the fundamental region
        fundamental = _band_filter(signal, sr, fundamental_hz_low, fundamental_hz_high)
        # 2. Generate harmonics via tanh
        saturated = np.tanh(fundamental * drive)
        # 3. Bandpass harmonics into the audible weight region
        harmonics = _band_filter(saturated, sr, harmonic_hz_low, harmonic_hz_high)
        # 4. Mix in
        out_channels.append(signal + harmonics * harmonic_mix)

    output_data = np.stack(out_channels, axis=1)

    # Clip guard
    peak = float(np.max(np.abs(output_data)))
    if peak > 1.0:
        print(
            f"WARNING: output peak {20*np.log10(peak):.1f} dBFS — scaling down",
            file=sys.stderr,
        )
        output_data = output_data / peak

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (input_path.stem + "_subharm.wav")
    sf.write(str(out_path), output_data, sr, subtype="PCM_24")

    report = {
        "input": str(input_path),
        "output": str(out_path),
        "preset": preset_name,
        "settings": {
            "fundamental_hz_low": fundamental_hz_low,
            "fundamental_hz_high": fundamental_hz_high,
            "harmonic_hz_low": harmonic_hz_low,
            "harmonic_hz_high": harmonic_hz_high,
            "drive": drive,
            "harmonic_mix": harmonic_mix,
        },
        "relevance_check": rel,
        "applied": True,
        "sample_rate": sr,
    }
    (output_dir / "subharm_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize sub-bass harmonics for small-speaker translation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Presets: " + ", ".join(PRESETS),
    )
    parser.add_argument("file", type=Path, nargs="?", help="Input WAV (typically bass or kick stem)")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--preset", choices=list(PRESETS), help="Sub-synth preset")
    parser.add_argument("--fundamental-hz-low", type=float, help="Fundamental band low edge (Hz)")
    parser.add_argument("--fundamental-hz-high", type=float, help="Fundamental band high edge (Hz)")
    parser.add_argument("--harmonic-hz-low", type=float, help="Harmonic band low edge (Hz)")
    parser.add_argument("--harmonic-hz-high", type=float, help="Harmonic band high edge (Hz)")
    parser.add_argument("--drive", type=float, help="Saturation drive (1.0-3.0 typical)")
    parser.add_argument("--harmonic-mix", type=float, help="Wet level of harmonics (0.05-0.30 typical)")
    parser.add_argument("--force", action="store_true",
                        help="Apply even if relevance_check recommends skip")
    parser.add_argument("--list-presets", action="store_true", help="List presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        for name, p in PRESETS.items():
            print(f"\n{name}: {p['description']}")
            print(f"  fundamental: {p['fundamental_hz_low']}-{p['fundamental_hz_high']} Hz  "
                  f"harmonic: {p['harmonic_hz_low']}-{p['harmonic_hz_high']} Hz")
            print(f"  drive={p['drive']}  mix={p['harmonic_mix']}")
        return

    if args.file is None:
        parser.error("file is required (unless --list-presets)")
    if args.output_dir is None:
        parser.error("--output-dir is required (unless --list-presets)")
    if not args.file.exists():
        print(json.dumps({"error": f"Not found: {args.file}"}), file=sys.stderr)
        sys.exit(1)

    params: dict = {}
    preset_name = None
    if args.preset:
        preset_name = args.preset
        p = PRESETS[args.preset]
        params = {
            "fundamental_hz_low": p["fundamental_hz_low"],
            "fundamental_hz_high": p["fundamental_hz_high"],
            "harmonic_hz_low": p["harmonic_hz_low"],
            "harmonic_hz_high": p["harmonic_hz_high"],
            "drive": p["drive"],
            "harmonic_mix": p["harmonic_mix"],
        }
    else:
        params = {
            "fundamental_hz_low": 40.0, "fundamental_hz_high": 80.0,
            "harmonic_hz_low": 80.0, "harmonic_hz_high": 200.0,
            "drive": 1.8, "harmonic_mix": 0.15,
        }

    overrides = {
        "fundamental_hz_low": args.fundamental_hz_low,
        "fundamental_hz_high": args.fundamental_hz_high,
        "harmonic_hz_low": args.harmonic_hz_low,
        "harmonic_hz_high": args.harmonic_hz_high,
        "drive": args.drive,
        "harmonic_mix": args.harmonic_mix,
    }
    for k, v in overrides.items():
        if v is not None:
            params[k] = v

    result = apply_subharm(
        input_path=args.file,
        output_dir=args.output_dir,
        preset_name=preset_name,
        force=args.force,
        **params,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
