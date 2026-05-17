"""3-band multiband compressor: split a stem into low/mid/high bands via
Linkwitz-Riley crossovers, compress each band independently, sum back.

Why this matters for "make it hit": a fullband comp on a mix or bus is forced
to react to whichever band is loudest. If the kick is hammering, the comp
ducks the cymbals too. Multiband solves this by giving each band its own
threshold, ratio, attack, and release.

Typical use cases:
  - Master bus: tight low-band comp (control sub punch), gentle mid comp
    (natural feel), slow high-band comp (preserve air).
  - Drum bus: tight comp on the kick/snare body, lighter on cymbals/transient.
  - Bass: tight low, less mid/high.

Relevance check:
  - At least 2 bands must show meaningful dynamic range (band crest > 6 dB).
    If everything is already squashed, multiband adds artifacts, not control.
  - Stem duration > 5 seconds (LRA/crest stats need enough material).

Crossover: 4th-order Linkwitz-Riley (cascaded Butterworth) — flat sum,
24 dB/oct slopes. Defaults: low/mid = 200 Hz, mid/high = 3000 Hz.

Usage:
  python apply_multiband_comp.py mix.wav --output-dir DIR \\
      --low-thr -16 --low-ratio 3 --low-attack 10 --low-release 200 \\
      --mid-thr -18 --mid-ratio 2 --mid-attack 20 --mid-release 250 \\
      --high-thr -22 --high-ratio 1.5 --high-attack 30 --high-release 350

  # Preset
  python apply_multiband_comp.py mix.wav --output-dir DIR --preset mb_master_glue
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Compressor, Gain, Pedalboard
from scipy.signal import butter, sosfilt


PRESETS: dict[str, dict] = {
    "mb_master_glue": {
        "description": "Gentle master-bus multiband — tighten low, breathe mid/high",
        "low_high_hz":  200.0,
        "mid_high_hz":  3000.0,
        "low":   {"threshold_db": -14, "ratio": 3.0, "attack_ms": 10, "release_ms": 200},
        "mid":   {"threshold_db": -16, "ratio": 2.0, "attack_ms": 25, "release_ms": 250},
        "high":  {"threshold_db": -20, "ratio": 1.5, "attack_ms": 30, "release_ms": 350},
    },
    "mb_drum_bus": {
        "description": "Drum bus multiband — punchy kick/snare, breathing cymbals",
        "low_high_hz":  150.0,
        "mid_high_hz":  4000.0,
        "low":   {"threshold_db": -12, "ratio": 4.0, "attack_ms": 5,  "release_ms": 120},
        "mid":   {"threshold_db": -14, "ratio": 2.5, "attack_ms": 15, "release_ms": 180},
        "high":  {"threshold_db": -20, "ratio": 1.5, "attack_ms": 25, "release_ms": 250},
    },
    "mb_bass": {
        "description": "Bass multiband — tight sub, controlled body, untouched top",
        "low_high_hz":  120.0,
        "mid_high_hz":  2000.0,
        "low":   {"threshold_db": -10, "ratio": 4.0, "attack_ms": 8,  "release_ms": 100},
        "mid":   {"threshold_db": -14, "ratio": 2.0, "attack_ms": 20, "release_ms": 200},
        "high":  {"threshold_db": -30, "ratio": 1.2, "attack_ms": 30, "release_ms": 300},
    },
}


# ---------------------------------------------------------------------------
# Linkwitz-Riley crossover (4th order = cascaded 2nd-order Butterworth)
# ---------------------------------------------------------------------------

def _lr4_split(signal: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split signal into (low, mid, high) using 4th-order Linkwitz-Riley.

    LR4 = two cascaded 2nd-order Butterworth filters. The crossover sum is
    flat (Linkwitz-Riley's defining property).
    """
    nyq = sr / 2.0
    # Low band: LR4 low-pass at lo_hz
    sos_lp_low = butter(2, lo_hz / nyq, btype="low", output="sos")
    low = sosfilt(sos_lp_low, signal)
    low = sosfilt(sos_lp_low, low)

    # High band: LR4 high-pass at hi_hz
    sos_hp_high = butter(2, hi_hz / nyq, btype="high", output="sos")
    high = sosfilt(sos_hp_high, signal)
    high = sosfilt(sos_hp_high, high)

    # Mid band: signal - low - high (so the crossover sum stays flat)
    mid = signal - low - high
    return low, mid, high


# ---------------------------------------------------------------------------
# Relevance check
# ---------------------------------------------------------------------------

_MIN_BAND_CREST_DB = 6.0
_MIN_DURATION_SEC = 5.0
_MIN_BANDS_WITH_DYNAMICS = 2


def _band_crest_db(band: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(band ** 2) + 1e-20))
    peak = float(np.max(np.abs(band)))
    if rms < 1e-10:
        return 0.0
    return 20.0 * np.log10(peak / rms)


def _relevance_check(signal: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> dict:
    duration = len(signal) / sr
    low, mid, high = _lr4_split(signal, sr, lo_hz, hi_hz)
    crests = {
        "low_crest_db": round(_band_crest_db(low), 1),
        "mid_crest_db": round(_band_crest_db(mid), 1),
        "high_crest_db": round(_band_crest_db(high), 1),
    }
    bands_with_dyn = sum(1 for v in crests.values() if v >= _MIN_BAND_CREST_DB)

    issues = []
    if duration < _MIN_DURATION_SEC:
        issues.append(f"duration {duration:.1f}s < {_MIN_DURATION_SEC}s — too short for stable multiband")
    if bands_with_dyn < _MIN_BANDS_WITH_DYNAMICS:
        issues.append(
            f"only {bands_with_dyn} band(s) above {_MIN_BAND_CREST_DB} dB crest — "
            f"material is already squashed, multiband adds artifacts not control"
        )

    return {
        "tool": "multiband_comp",
        "duration_sec": round(duration, 1),
        **crests,
        "bands_with_dynamics": bands_with_dyn,
        "recommend_skip": bool(issues),
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def _compress_band(band: np.ndarray, sr: int, params: dict) -> np.ndarray:
    board = Pedalboard([
        Compressor(
            threshold_db=float(params["threshold_db"]),
            ratio=float(params["ratio"]),
            attack_ms=float(params["attack_ms"]),
            release_ms=float(params["release_ms"]),
        ),
        Gain(gain_db=float(params.get("makeup_db", 0.0))),
    ])
    return board(band.astype(np.float32), sr).astype(np.float64)


def apply_multiband_comp(
    input_path: Path,
    output_dir: Path,
    low_high_hz: float,
    mid_high_hz: float,
    low_params: dict,
    mid_params: dict,
    high_params: dict,
    preset_name: str | None = None,
    force: bool = False,
) -> dict:
    data, sr = sf.read(str(input_path), always_2d=True)
    mono_for_check = data.mean(axis=1).astype(np.float64)
    rel = _relevance_check(mono_for_check, sr, low_high_hz, mid_high_hz)

    if rel["recommend_skip"] and not force:
        print(f"WARNING: multiband_comp relevance_check failed for {input_path.name}:", file=sys.stderr)
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
        (output_dir / "multiband_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    out_channels = []
    for ch in range(data.shape[1]):
        signal = data[:, ch].astype(np.float64)
        low, mid, high = _lr4_split(signal, sr, low_high_hz, mid_high_hz)
        low_c = _compress_band(low, sr, low_params)
        mid_c = _compress_band(mid, sr, mid_params)
        high_c = _compress_band(high, sr, high_params)
        out_channels.append(low_c + mid_c + high_c)

    output_data = np.stack(out_channels, axis=1)

    peak = float(np.max(np.abs(output_data)))
    if peak > 1.0:
        print(f"WARNING: output peak {20*np.log10(peak):.1f} dBFS — scaling down", file=sys.stderr)
        output_data = output_data / peak

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (input_path.stem + "_mbcomp.wav")
    sf.write(str(out_path), output_data, sr, subtype="PCM_24")

    report = {
        "input": str(input_path),
        "output": str(out_path),
        "preset": preset_name,
        "settings": {
            "crossover_low_high_hz": low_high_hz,
            "crossover_mid_high_hz": mid_high_hz,
            "low": low_params,
            "mid": mid_params,
            "high": high_params,
        },
        "relevance_check": rel,
        "applied": True,
        "sample_rate": sr,
    }
    (output_dir / "multiband_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-band multiband compressor (LR4 crossovers, pedalboard.Compressor per band).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Presets: " + ", ".join(PRESETS),
    )
    parser.add_argument("file", type=Path, nargs="?", help="Input WAV")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--preset", choices=list(PRESETS), help="Multiband preset")
    parser.add_argument("--low-high-hz", type=float, help="Low/mid crossover (Hz)")
    parser.add_argument("--mid-high-hz", type=float, help="Mid/high crossover (Hz)")
    for band in ("low", "mid", "high"):
        for p, suf in [("threshold_db", "thr"), ("ratio", "ratio"),
                       ("attack_ms", "attack"), ("release_ms", "release"),
                       ("makeup_db", "makeup")]:
            parser.add_argument(f"--{band}-{suf}", type=float,
                                help=f"{band.title()} band {p}")
    parser.add_argument("--force", action="store_true",
                        help="Apply even if relevance_check recommends skip")
    parser.add_argument("--list-presets", action="store_true", help="List presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        for name, p in PRESETS.items():
            print(f"\n{name}: {p['description']}")
            print(f"  crossovers: {p['low_high_hz']}/{p['mid_high_hz']} Hz")
            for b in ("low", "mid", "high"):
                pp = p[b]
                print(f"  {b}: thr={pp['threshold_db']} ratio={pp['ratio']} "
                      f"atk={pp['attack_ms']}ms rel={pp['release_ms']}ms")
        return

    if args.file is None:
        parser.error("file is required (unless --list-presets)")
    if args.output_dir is None:
        parser.error("--output-dir is required (unless --list-presets)")
    if not args.file.exists():
        print(json.dumps({"error": f"Not found: {args.file}"}), file=sys.stderr)
        sys.exit(1)

    if args.preset:
        p = PRESETS[args.preset]
        low_high_hz = p["low_high_hz"]
        mid_high_hz = p["mid_high_hz"]
        low_params = dict(p["low"])
        mid_params = dict(p["mid"])
        high_params = dict(p["high"])
    else:
        # Defaults if no preset
        low_high_hz = 200.0
        mid_high_hz = 3000.0
        low_params = {"threshold_db": -14, "ratio": 3.0, "attack_ms": 10, "release_ms": 200}
        mid_params = {"threshold_db": -16, "ratio": 2.0, "attack_ms": 25, "release_ms": 250}
        high_params = {"threshold_db": -20, "ratio": 1.5, "attack_ms": 30, "release_ms": 350}

    # CLI overrides
    if args.low_high_hz is not None: low_high_hz = args.low_high_hz
    if args.mid_high_hz is not None: mid_high_hz = args.mid_high_hz
    for band, target in (("low", low_params), ("mid", mid_params), ("high", high_params)):
        for p, suf in [("threshold_db", "thr"), ("ratio", "ratio"),
                       ("attack_ms", "attack"), ("release_ms", "release"),
                       ("makeup_db", "makeup")]:
            v = getattr(args, f"{band}_{suf}".replace("-", "_"))
            if v is not None:
                target[p] = v

    result = apply_multiband_comp(
        input_path=args.file,
        output_dir=args.output_dir,
        low_high_hz=low_high_hz,
        mid_high_hz=mid_high_hz,
        low_params=low_params,
        mid_params=mid_params,
        high_params=high_params,
        preset_name=args.preset,
        force=args.force,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
