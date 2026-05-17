"""Spectral exciter — generate high-frequency harmonics to add "air" and
"presence" without raising overall level.

Aphex Aural Exciter-style: high-pass the signal, run it through soft
saturation to generate harmonic content above the cutoff, mix the harmonics
back into the dry signal at low level. The dry signal is unchanged; only
the harmonic content is new.

Why this is different from just shelving up the highs:
  - Shelving boosts what's already there. If the source rolls off at 8 kHz,
    a high shelf gives you nothing.
  - Excitation *generates* new harmonics, so the result has top end even
    when the source is dull.

Relevance check:
  - Stem must lack air (8 kHz+ band < -40 dBFS, spectral centroid < 4 kHz).
    If the stem is already bright, the exciter just adds harshness.
  - Stem must not be a primarily HF source (cymbal, hi-hat) — excitation
    on top of HF content stacks harmonics and gets ugly.

Usage:
  python apply_exciter.py vocal.wav --output-dir DIR --preset exciter_vocal_air
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, welch


PRESETS: dict[str, dict] = {
    "exciter_vocal_air": {
        "description": "Subtle 'air' on lead vocal — top-end presence without harshness",
        "hp_hz": 6000,
        "drive": 1.5,
        "mix": 0.12,
    },
    "exciter_acoustic_guitar": {
        "description": "Sparkle on acoustic guitar — pick attack + body",
        "hp_hz": 5000,
        "drive": 1.8,
        "mix": 0.15,
    },
    "exciter_dull_mix": {
        "description": "Last-resort revival for a dull mix",
        "hp_hz": 7000,
        "drive": 2.0,
        "mix": 0.10,
    },
}


# ---------------------------------------------------------------------------
# Relevance check
# ---------------------------------------------------------------------------

_AIR_RMS_DB_THRESHOLD = -40.0
_CENTROID_HZ_THRESHOLD = 4000.0
# Low-frequency-dominant stems (kick, bass): even though their air band is
# technically empty, "exciting" them makes no musical sense — bass and kick
# don't want top-end harmonics. Detect via: low band 6+ dB louder than high
# band, AND spectral centroid below 800 Hz (typical kick: ~100-300 Hz, bass:
# ~200-600 Hz, low guitar tracks: still ~500-1500 Hz).
_LOW_DOMINANCE_DB = 6.0
_LOW_DOMINANT_CENTROID_HZ = 800.0


def _band_rms_db(signal: np.ndarray, sr: int, lo: float, hi: float) -> float:
    nyq = sr / 2.0
    sos = butter(4, [lo / nyq, min(hi / nyq, 0.999)], btype="band", output="sos")
    rms = float(np.sqrt(np.mean(sosfilt(sos, signal) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-10))


def _relevance_check(mono: np.ndarray, sr: int) -> dict:
    air_db = _band_rms_db(mono, sr, 8000.0, min(sr / 2.0 - 100.0, 20000.0))
    low_db = _band_rms_db(mono, sr, 30.0, 250.0)
    high_db = _band_rms_db(mono, sr, 2000.0, min(sr / 2.0 - 100.0, 8000.0))
    low_over_high_db = low_db - high_db

    nperseg = min(len(mono), 32768)
    freqs, psd = welch(mono, fs=sr, nperseg=nperseg)
    centroid = float(np.sum(freqs * psd) / max(np.sum(psd), 1e-12))

    issues = []
    if air_db > _AIR_RMS_DB_THRESHOLD:
        issues.append(
            f"air band already at {air_db:.1f} dBFS > {_AIR_RMS_DB_THRESHOLD} — exciter will add harshness"
        )
    if centroid > _CENTROID_HZ_THRESHOLD:
        issues.append(
            f"spectral centroid {centroid:.0f} Hz > {_CENTROID_HZ_THRESHOLD} — stem is already bright"
        )
    if (low_over_high_db > _LOW_DOMINANCE_DB and
            centroid < _LOW_DOMINANT_CENTROID_HZ):
        issues.append(
            f"low-dominant stem (low band {low_over_high_db:.1f} dB > high band, "
            f"centroid {centroid:.0f} Hz < {_LOW_DOMINANT_CENTROID_HZ:.0f}) — "
            f"looks like a kick / bass / sub source. Exciter on bass/kick adds "
            f"harshness, not 'air'; skip unless you want a slap-bass click."
        )

    return {
        "tool": "exciter",
        "air_band_rms_dbfs": round(air_db, 1),
        "low_band_rms_dbfs": round(low_db, 1),
        "high_band_rms_dbfs": round(high_db, 1),
        "low_over_high_db": round(low_over_high_db, 1),
        "spectral_centroid_hz": round(centroid, 0),
        "recommend_skip": bool(issues),
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def _excite(signal: np.ndarray, sr: int, hp_hz: float, drive: float) -> np.ndarray:
    """High-pass, then saturate to generate harmonics in the HF region."""
    nyq = sr / 2.0
    sos_hp = butter(4, hp_hz / nyq, btype="high", output="sos")
    hp = sosfilt(sos_hp, signal)
    # Asymmetric tanh — generates even+odd harmonics. Subtle drive so we don't
    # buzz the top end into noise.
    saturated = np.tanh(hp * drive)
    # Re-filter so we keep only the new HF content (saturation generates
    # subharmonics too, which we don't want stacking on the dry).
    return sosfilt(sos_hp, saturated)


def apply_exciter(
    input_path: Path,
    output_dir: Path,
    hp_hz: float,
    drive: float,
    mix: float,
    preset_name: str | None = None,
    force: bool = False,
) -> dict:
    data, sr = sf.read(str(input_path), always_2d=True)
    mono = data.mean(axis=1)

    rel = _relevance_check(mono, sr)
    if rel["recommend_skip"] and not force:
        print(f"WARNING: exciter relevance_check failed for {input_path.name}:", file=sys.stderr)
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
        (output_dir / "exciter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    out_channels = []
    for ch in range(data.shape[1]):
        sig = data[:, ch].astype(np.float64)
        harmonics = _excite(sig, sr, hp_hz, drive)
        out_channels.append(sig + harmonics * mix)

    output_data = np.stack(out_channels, axis=1)

    peak = float(np.max(np.abs(output_data)))
    if peak > 1.0:
        print(f"WARNING: output peak {20*np.log10(peak):.1f} dBFS — scaling down", file=sys.stderr)
        output_data = output_data / peak

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (input_path.stem + "_exciter.wav")
    sf.write(str(out_path), output_data, sr, subtype="PCM_24")

    report = {
        "input": str(input_path),
        "output": str(out_path),
        "preset": preset_name,
        "settings": {"hp_hz": hp_hz, "drive": drive, "mix": mix},
        "relevance_check": rel,
        "applied": True,
        "sample_rate": sr,
    }
    (output_dir / "exciter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spectral exciter — add HF harmonics for 'air' and presence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Presets: " + ", ".join(PRESETS),
    )
    parser.add_argument("file", type=Path, nargs="?", help="Input WAV")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--preset", choices=list(PRESETS), help="Exciter preset")
    parser.add_argument("--hp-hz", type=float, help="High-pass cutoff (Hz) — exciter only generates harmonics ABOVE this")
    parser.add_argument("--drive", type=float, help="Saturation drive (1.0-3.0 typical)")
    parser.add_argument("--mix", type=float, help="Harmonic mix level (0.05-0.20 typical)")
    parser.add_argument("--force", action="store_true",
                        help="Apply even if relevance_check recommends skip")
    parser.add_argument("--list-presets", action="store_true", help="List presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        for name, p in PRESETS.items():
            print(f"\n{name}: {p['description']}")
            print(f"  hp={p['hp_hz']}Hz  drive={p['drive']}  mix={p['mix']}")
        return

    if args.file is None:
        parser.error("file is required (unless --list-presets)")
    if args.output_dir is None:
        parser.error("--output-dir is required (unless --list-presets)")
    if not args.file.exists():
        print(json.dumps({"error": f"Not found: {args.file}"}), file=sys.stderr)
        sys.exit(1)

    params: dict = {"hp_hz": 6000.0, "drive": 1.5, "mix": 0.12}
    preset_name = None
    if args.preset:
        preset_name = args.preset
        p = PRESETS[args.preset]
        params = {"hp_hz": p["hp_hz"], "drive": p["drive"], "mix": p["mix"]}

    if args.hp_hz is not None: params["hp_hz"] = args.hp_hz
    if args.drive is not None: params["drive"] = args.drive
    if args.mix is not None:   params["mix"] = args.mix

    result = apply_exciter(
        input_path=args.file,
        output_dir=args.output_dir,
        preset_name=preset_name,
        force=args.force,
        **params,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
