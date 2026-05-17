"""Apply parametric EQ to an assembled stem.

Supported filter types (all minimum-phase, zero-phase via sosfiltfilt):
  notch     - narrow band removal (hum, resonances). Infinite attenuation at center.
              params: hz, q (default 30)
  highpass  - Butterworth high-pass
              params: hz, order (default 2)
  lowpass   - Butterworth low-pass
              params: hz, order (default 2)
  bandpass  - Butterworth band-pass (isolate a frequency range)
              params: hz_low, hz_high, order (default 2)
  peak      - RBJ peaking biquad, boost or cut (tonal shaping)
              params: hz, q, db (positive=boost, negative=cut)
  lowshelf  - RBJ low shelf: boost/cut below hz
              params: hz, db, slope (default 1.0 = max steepness)
  highshelf - RBJ high shelf: boost/cut above hz
              params: hz, db, slope (default 1.0)

Each --filter argument is a JSON object:
  {"type": "highpass",  "hz": 80}
  {"type": "lowshelf",  "hz": 120,  "db": 3.0}
  {"type": "notch",     "hz": 150,  "q": 30}
  {"type": "peak",      "hz": 400,  "q": 1.5, "db": -3.0}
  {"type": "highshelf", "hz": 8000, "db": 2.0}
  {"type": "bandpass",  "hz_low": 200, "hz_high": 4000}

--from-analysis  reads analysis.json and auto-generates notch filters for detected hum.
--preset NAME    loads a named instrument preset from tools/presets/<NAME>.json.
                 Preset filters are prepended; --filter arguments are appended after.
                 Run with --list-presets to see available presets.

Usage:
  python apply_eq.py assembled.wav --output-dir output/session/TRACK \\
      --preset kick_in \\
      --filter '{"type":"peak","hz":3500,"q":2,"db":2}'

  python apply_eq.py assembled.wav --output-dir output/session/TRACK \\
      --from-analysis output/session/TRACK/analysis.json \\
      --filter '{"type":"highpass","hz":80}'
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, tf2sos

_PRESETS_DIR = Path(__file__).parent / "presets"


# ---------------------------------------------------------------------------
# Filter SOS builders — all return (n_sections, 6) SOS arrays
# ---------------------------------------------------------------------------

def _notch_sos(hz: float, q: float, sr: int) -> np.ndarray:
    b, a = iirnotch(hz, q, sr)
    return tf2sos(b, a)


def _hp_sos(hz: float, order: int, sr: int) -> np.ndarray:
    return butter(order, hz / (sr / 2.0), btype="high", output="sos")


def _lp_sos(hz: float, order: int, sr: int) -> np.ndarray:
    return butter(order, hz / (sr / 2.0), btype="low", output="sos")


def _bp_sos(hz_low: float, hz_high: float, order: int, sr: int) -> np.ndarray:
    nyq = sr / 2.0
    return butter(order, [hz_low / nyq, hz_high / nyq], btype="band", output="sos")


def _peak_sos(hz: float, q: float, db_gain: float, sr: int) -> np.ndarray:
    """Peaking EQ biquad — Audio EQ Cookbook (R. Bristow-Johnson)."""
    A = 10.0 ** (db_gain / 40.0)
    w0 = 2.0 * np.pi * hz / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _lowshelf_sos(hz: float, db_gain: float, slope: float, sr: int) -> np.ndarray:
    """Low shelf biquad — Audio EQ Cookbook (R. Bristow-Johnson)."""
    A = 10.0 ** (db_gain / 40.0)
    w0 = 2.0 * np.pi * hz / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    sqA = np.sqrt(A)

    b0 =      A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqA * alpha)
    b1 =  2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 =      A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqA * alpha)
    a0 =          ((A + 1) + (A - 1) * cos_w0 + 2 * sqA * alpha)
    a1 =     -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 =          ((A + 1) + (A - 1) * cos_w0 - 2 * sqA * alpha)

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _highshelf_sos(hz: float, db_gain: float, slope: float, sr: int) -> np.ndarray:
    """High shelf biquad — Audio EQ Cookbook (R. Bristow-Johnson)."""
    A = 10.0 ** (db_gain / 40.0)
    w0 = 2.0 * np.pi * hz / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    sqA = np.sqrt(A)

    b0 =      A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqA * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 =      A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqA * alpha)
    a0 =          ((A + 1) - (A - 1) * cos_w0 + 2 * sqA * alpha)
    a1 =      2 * ((A - 1) - (A + 1) * cos_w0)
    a2 =          ((A + 1) - (A - 1) * cos_w0 - 2 * sqA * alpha)

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _build_sos(f: dict, sr: int) -> np.ndarray:
    ftype = f.get("type", "")
    if ftype == "notch":
        return _notch_sos(float(f["hz"]), float(f.get("q", 30.0)), sr)
    if ftype == "highpass":
        return _hp_sos(float(f["hz"]), int(f.get("order", 2)), sr)
    if ftype == "lowpass":
        return _lp_sos(float(f["hz"]), int(f.get("order", 2)), sr)
    if ftype == "bandpass":
        return _bp_sos(float(f["hz_low"]), float(f["hz_high"]), int(f.get("order", 2)), sr)
    if ftype == "peak":
        return _peak_sos(float(f["hz"]), float(f["q"]), float(f["db"]), sr)
    if ftype == "lowshelf":
        return _lowshelf_sos(float(f["hz"]), float(f["db"]), float(f.get("slope", 1.0)), sr)
    if ftype == "highshelf":
        return _highshelf_sos(float(f["hz"]), float(f["db"]), float(f.get("slope", 1.0)), sr)
    valid = "notch, highpass, lowpass, bandpass, peak, lowshelf, highshelf"
    raise ValueError(f"Unknown filter type: {ftype!r}. Valid: {valid}")


# ---------------------------------------------------------------------------
# Preset loader
# ---------------------------------------------------------------------------

def _load_preset(name: str) -> list[dict]:
    path = _PRESETS_DIR / f"{name}.json"
    if not path.exists():
        available = [p.stem for p in sorted(_PRESETS_DIR.glob("*.json"))]
        raise FileNotFoundError(
            f"Preset {name!r} not found. Available: {', '.join(available)}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    filters = data.get("filters", [])
    for f in filters:
        f["_preset"] = name
    return filters


def list_presets() -> None:
    if not _PRESETS_DIR.exists():
        print("No presets directory found.")
        return
    for path in sorted(_PRESETS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        desc = data.get("description", "")
        print(f"  {path.stem:<28}  {desc}")


# ---------------------------------------------------------------------------
# Hum auto-notch from analysis.json
# ---------------------------------------------------------------------------

def _filters_from_analysis(analysis_path: Path) -> list[dict]:
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    hum = data.get("hum_detection", {})
    if not hum.get("hum_detected"):
        return []
    filters = []
    for mains_key, harmonics in hum.get("harmonics", {}).items():
        for h in harmonics:
            filters.append({
                "type": "notch",
                "hz": float(h["frequency_hz"]),
                "q": 30.0,
                "_auto": f"hum {mains_key} prominence {h['prominence_db']} dB",
            })
    return filters


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def apply_eq(input_path: Path, output_dir: Path, filters: list[dict], phase: str = "minimum") -> dict:
    if not filters:
        print(json.dumps({"error": "No filters specified — nothing to apply"}), file=sys.stderr)
        sys.exit(1)

    if phase not in ("minimum", "zero"):
        raise ValueError(f"phase must be 'minimum' or 'zero', got {phase!r}")

    data, sr = sf.read(str(input_path), always_2d=True)

    # Validate all filter specs before touching audio
    for f in filters:
        _build_sos(f, sr)

    # minimum phase = causal sosfilt (preserves transients, adds group delay)
    # zero phase   = sosfiltfilt (pre-rings transients, no net delay)
    filt_fn = sosfilt if phase == "minimum" else sosfiltfilt

    result_channels = []
    for ch in range(data.shape[1]):
        signal = data[:, ch].astype(np.float64)
        for f in filters:
            signal = filt_fn(_build_sos(f, sr), signal)
        result_channels.append(signal)

    output_data = np.stack(result_channels, axis=1)

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
    out_path = output_dir / (input_path.stem + "_eq.wav")
    sf.write(str(out_path), output_data, sr, subtype="PCM_24")

    report = {
        "input": str(input_path),
        "output": str(out_path),
        "phase": phase,
        "filters_applied": [
            {k: v for k, v in f.items() if not k.startswith("_")} for f in filters
        ],
        "filter_notes": [f["_auto"] for f in filters if "_auto" in f] or None,
        "preset_used": next((f["_preset"] for f in filters if "_preset" in f), None),
        "sample_rate": sr,
        "clipping_prevented": clipped,
    }
    (output_dir / "eq_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply parametric EQ to a stem.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", type=Path, nargs="?", help="Input WAV file")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument(
        "--filter", dest="filters", action="append", metavar="JSON", default=[],
        help="Filter spec JSON. Repeatable.",
    )
    parser.add_argument(
        "--preset", metavar="NAME",
        help="Instrument preset name (see --list-presets)",
    )
    parser.add_argument(
        "--from-analysis", type=Path, metavar="JSON",
        help="analysis.json — auto-generates notch filters from detected hum",
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="List available instrument presets and exit",
    )
    parser.add_argument(
        "--phase", choices=["minimum", "zero"], default="minimum",
        help="Filter phase response. minimum=causal (preserves transients, "
             "default for insert EQ on tracks). zero=zero-phase (pre-rings "
             "transients, recommended only for mastering / linear-phase use).",
    )
    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        return

    if not args.file or not args.output_dir:
        parser.error("file and --output-dir are required (unless using --list-presets)")

    if not args.file.exists():
        print(json.dumps({"error": f"Not found: {args.file}"}), file=sys.stderr)
        sys.exit(1)

    filters: list[dict] = []

    if args.preset:
        try:
            preset_filters = _load_preset(args.preset)
            print(f"Loaded preset {args.preset!r} ({len(preset_filters)} filter(s)):", file=sys.stderr)
            for f in preset_filters:
                print(f"  {f}", file=sys.stderr)
            filters.extend(preset_filters)
        except FileNotFoundError as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(1)

    if args.from_analysis:
        if not args.from_analysis.exists():
            print(json.dumps({"error": f"Not found: {args.from_analysis}"}), file=sys.stderr)
            sys.exit(1)
        auto = _filters_from_analysis(args.from_analysis)
        if auto:
            print(f"Auto-generated {len(auto)} notch filter(s) from hum detection:", file=sys.stderr)
            for f in auto:
                print(f"  {f['hz']} Hz  Q={f['q']}  [{f.get('_auto', '')}]", file=sys.stderr)
        else:
            print("No hum detected in analysis — no auto-notch filters added.", file=sys.stderr)
        filters.extend(auto)

    for raw in args.filters:
        try:
            filters.append(json.loads(raw))
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid filter JSON: {e}"}), file=sys.stderr)
            sys.exit(1)

    apply_eq(args.file, args.output_dir, filters, phase=args.phase)


if __name__ == "__main__":
    main()
