"""Apply harmonic saturation to a stem.

Three saturation modes:
  tape     Symmetric tanh soft clipping — even + odd harmonics, tape-like warmth.
           Gentle on transients. Good for bus glue, drums, guitars.
  tube     Asymmetric tanh — positive half clips harder (triode model).
           Generates even-order harmonics (2nd, 4th) — warm without sounding harsh.
           --asymmetry controls how asymmetric the clipping is.
  clipper  Cubic soft clipper — hard ceiling at ±1.0 with smooth knee.
           Generates odd-order harmonics (3rd, 5th) — brightness, presence, aggression.
           Best used in parallel (--mix 0.2-0.4).

All modes RMS-normalize the saturated output to input level: harmonics are added, not gain.
Use --mix for parallel saturation (dry signal blended with RMS-matched saturated copy).

Usage:
  python apply_saturation.py assembled.wav --output-dir output/session/TRACK \\
      --mode tape --drive 0.3

  python apply_saturation.py assembled_eq_comp.wav --output-dir output/session/TRACK \\
      --preset sat_tube_bass

  python apply_saturation.py assembled.wav --output-dir output/session/TRACK \\
      --mode clipper --drive 0.5 --mix 0.25

  python apply_saturation.py x.wav --list-presets
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_PRESETS_DIR = Path(__file__).parent / "presets"


# ---------------------------------------------------------------------------
# Saturation curves
# ---------------------------------------------------------------------------

def _saturate_tape(data: np.ndarray, drive: float) -> np.ndarray:
    """Symmetric tanh — warm, tape-like, even + odd harmonics."""
    in_rms = np.sqrt(np.mean(data ** 2) + 1e-12)
    x = data * (1.0 + drive * 4.0)
    out = np.tanh(x)
    out_rms = np.sqrt(np.mean(out ** 2) + 1e-12)
    if out_rms > 1e-10:
        out = out * (in_rms / out_rms)
    return out


def _saturate_tube(data: np.ndarray, drive: float, asymmetry: float) -> np.ndarray:
    """Asymmetric tanh — positive half clips harder (triode model).

    Even-order harmonics (2nd, 4th) dominate — perceptually warm, not harsh.
    asymmetry=0: symmetric (same as tape). asymmetry=1: maximum asymmetry.
    """
    in_rms = np.sqrt(np.mean(data ** 2) + 1e-12)
    x = data * (1.0 + drive * 4.0)

    pos_factor = 1.0 + asymmetry
    pos_mask = x >= 0
    out = np.zeros_like(x)
    out[pos_mask] = np.tanh(x[pos_mask] * pos_factor) / pos_factor
    out[~pos_mask] = np.tanh(x[~pos_mask])

    out_rms = np.sqrt(np.mean(out ** 2) + 1e-12)
    if out_rms > 1e-10:
        out = out * (in_rms / out_rms)
    return out


def _saturate_clipper(data: np.ndarray, drive: float) -> np.ndarray:
    """Cubic soft clipper — odd harmonics, presence and aggression.

    Soft knee up to ±1.0 then hard ceiling. Best in parallel at low mix.
    """
    in_rms = np.sqrt(np.mean(data ** 2) + 1e-12)
    x = data * (1.0 + drive * 4.0)

    in_range = np.abs(x) <= 1.0
    out = np.zeros_like(x)
    # Cubic: x - x^3/3, scaled so peak output = 1.0 when input = 1.0
    out[in_range] = (x[in_range] - x[in_range] ** 3 / 3.0) * 1.5
    out[~in_range] = np.sign(x[~in_range])

    out_rms = np.sqrt(np.mean(out ** 2) + 1e-12)
    if out_rms > 1e-10:
        out = out * (in_rms / out_rms)
    return out


_SATURATORS = {
    "tape": _saturate_tape,
    "tube": _saturate_tube,
    "clipper": _saturate_clipper,
}


# ---------------------------------------------------------------------------
# Preset loader
# ---------------------------------------------------------------------------

def _load_preset(name: str) -> dict:
    path = _PRESETS_DIR / f"{name}.json"
    if not path.exists():
        available = [p.stem for p in sorted(_PRESETS_DIR.glob("sat_*.json"))]
        raise FileNotFoundError(
            f"Preset {name!r} not found. Available saturation presets: {', '.join(available)}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def list_presets() -> None:
    if not _PRESETS_DIR.exists():
        print("No presets directory found.")
        return
    for path in sorted(_PRESETS_DIR.glob("sat_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data.get("settings", {})
        desc = data.get("description", "")
        print(
            f"  {path.stem:<28}  {desc}"
            f"  [mode={s.get('mode','?')}  drive={s.get('drive','?')}  mix={s.get('mix','?')}]"
        )


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def apply_saturation(
    input_path: Path,
    output_dir: Path,
    mode: str = "tape",
    drive: float = 0.3,
    asymmetry: float = 0.5,
    mix: float = 1.0,
    preset_name: str | None = None,
) -> dict:
    if mode not in _SATURATORS:
        raise ValueError(f"Unknown mode {mode!r}. Choose from: {', '.join(_SATURATORS)}")

    data, sr = sf.read(str(input_path), always_2d=True)

    saturator = _SATURATORS[mode]
    if mode == "tube":
        saturated = saturator(data, drive, asymmetry)
    else:
        saturated = saturator(data, drive)

    # Parallel blend
    if mix < 0.999:
        output_data = data * (1.0 - mix) + saturated * mix
    else:
        output_data = saturated

    # Clip guard
    peak_linear = float(np.max(np.abs(output_data)))
    clipped = peak_linear > 1.0
    if clipped:
        print(
            f"WARNING: output peak {20 * np.log10(peak_linear):.1f} dBFS — "
            "scaling down to prevent clipping",
            file=sys.stderr,
        )
        output_data = output_data / peak_linear

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (input_path.stem + "_sat.wav")
    sf.write(str(out_path), output_data, sr, subtype="PCM_24")

    in_rms = float(np.sqrt(np.mean(data ** 2) + 1e-12))
    out_rms = float(np.sqrt(np.mean(output_data ** 2) + 1e-12))
    in_peak = float(20 * np.log10(max(float(np.max(np.abs(data))), 1e-10)))
    out_peak = float(20 * np.log10(max(float(np.max(np.abs(output_data))), 1e-10)))
    rms_change = float(20 * np.log10(max(out_rms / max(in_rms, 1e-10), 1e-10)))

    report = {
        "input": str(input_path),
        "output": str(out_path),
        "preset": preset_name,
        "settings": {
            "mode": mode,
            "drive": drive,
            "asymmetry": asymmetry if mode == "tube" else None,
            "mix": mix,
        },
        "input_peak_dbfs": round(in_peak, 1),
        "output_peak_dbfs": round(out_peak, 1),
        "rms_change_db": round(rms_change, 2),
        "clipping_prevented": clipped,
        "sample_rate": sr,
    }
    (output_dir / "sat_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply harmonic saturation to a stem.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", type=Path, nargs="?", help="Input WAV file")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument(
        "--mode", choices=["tape", "tube", "clipper"], default="tape",
        help="Saturation algorithm (default: tape)",
    )
    parser.add_argument(
        "--drive", type=float, default=0.3, metavar="0-1",
        help="Drive amount 0.0-1.0 (default 0.3)",
    )
    parser.add_argument(
        "--asymmetry", type=float, default=0.5, metavar="0-1",
        help="Tube mode: asymmetry amount 0.0-1.0 (default 0.5)",
    )
    parser.add_argument(
        "--mix", type=float, default=1.0, metavar="0-1",
        help="Wet/dry blend: 1.0=serial, <1.0=parallel (default 1.0)",
    )
    parser.add_argument("--preset", metavar="NAME", help="Saturation preset (see --list-presets)")
    parser.add_argument("--list-presets", action="store_true", help="List saturation presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        return

    if not args.file or not args.output_dir:
        parser.error("file and --output-dir are required (unless using --list-presets)")

    if not args.file.exists():
        print(json.dumps({"error": f"Not found: {args.file}"}), file=sys.stderr)
        sys.exit(1)

    mode = args.mode
    drive = args.drive
    asymmetry = args.asymmetry
    mix = args.mix
    preset_name = args.preset

    if args.preset:
        try:
            pdata = _load_preset(args.preset)
        except FileNotFoundError as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(1)
        s = pdata.get("settings", {})
        mode = s.get("mode", mode)
        drive = s.get("drive", drive)
        asymmetry = s.get("asymmetry", asymmetry)
        mix = s.get("mix", mix)
        print(
            f"Loaded preset {args.preset!r}: mode={mode} drive={drive} "
            f"asymmetry={asymmetry} mix={mix}",
            file=sys.stderr,
        )

    apply_saturation(
        args.file, args.output_dir,
        mode=mode, drive=drive, asymmetry=asymmetry, mix=mix,
        preset_name=preset_name,
    )


if __name__ == "__main__":
    main()
