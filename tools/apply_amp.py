"""Apply tube amp simulation and cabinet EQ to a bass DI stem.

Tube saturation: asymmetric soft clipping — positive half clips harder,
matching triode anode current saturation. Produces even-order harmonics
(2nd, 4th) that add warmth without sounding distorted at low drive levels.

Cabinet EQ: parametric chain modelling the Ampeg SVT 8x10 response:
  HP  @ 40 Hz        — removes inaudible sub-rumble, tightens low end
  Low shelf +Xdb @ 120 Hz  — cabinet body and punch (Ampeg hump 100-125 Hz)
  Peak boost @ mid_hz — "grind" presence (SVT midrange: 800 Hz or 1.6 kHz)
  LP  @ 5000 Hz      — speaker rolloff (-3dB at 5 kHz per SVT-810E spec)

Two modes:
  Insert (default): processed signal replaces the input. Use inline on bass DI.

Presets:
  ampeg_svt        — Ampeg SVT-CL character, moderate drive, warm and punchy
  ampeg_svt_driven — harder drive, more grind, cuts through dense guitars
  di_clean         — cabinet EQ only, no saturation

Usage:
  python apply_amp.py output/terido/BASS\\ DI\\ CLEAN/assembled_eq_comp.wav \\
      --preset ampeg_svt --output-dir output/terido/BASS\\ DI\\ CLEAN

  python apply_amp.py input.wav --drive 0.5 --asymmetry 0.3 \\
      --hp 40 --lp 5000 --low-shelf-hz 120 --low-shelf-db 3 \\
      --mid-hz 800 --mid-db 2 --mid-q 2 --output-dir DIR
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


PRESETS: dict[str, dict] = {
    "ampeg_svt": {
        "description": "Ampeg SVT-CL — warm tube character, 8x10 cabinet, moderate drive",
        "notes": "Drive=0.35 adds tube warmth without obvious distortion. HP@40Hz removes sub-rumble, low shelf +3dB@120Hz is the signature SVT cabinet body, 800Hz peak adds grind that cuts through guitars, LP@5kHz matches SVT-810E speaker rolloff (-3dB@5kHz per Ampeg spec). Asymmetry=0.3 tilts saturation toward even harmonics (2nd harmonic = octave below, adds warmth).",
        "drive": 0.35,
        "asymmetry": 0.30,
        "hp_hz": 40.0,
        "low_shelf_hz": 120.0,
        "low_shelf_db": 3.0,
        "mid_hz": 800.0,
        "mid_db": 2.0,
        "mid_q": 2.0,
        "lp_hz": 5000.0,
    },
    "ampeg_svt_driven": {
        "description": "Ampeg SVT pushed hard — more grind, more harmonics, cuts through dense guitars",
        "notes": "Drive=0.65 adds noticeable tube grit. 800Hz peak boost +4dB with Q=1.5 (broader) gives that aggressive midrange SVT character when the gain is cranked. Use when bass feels buried in a dense rock mix.",
        "drive": 0.65,
        "asymmetry": 0.35,
        "hp_hz": 40.0,
        "low_shelf_hz": 120.0,
        "low_shelf_db": 3.0,
        "mid_hz": 800.0,
        "mid_db": 4.0,
        "mid_q": 1.5,
        "lp_hz": 5000.0,
    },
    "ampeg_slap": {
        "description": "Ampeg tube character + slap bass EQ — warm tube harmonics, low thump, mid scoop, bright pop",
        "notes": "Ampeg drive (0.30) adds tube warmth and even harmonics. Low shelf +4dB@80Hz for thumb thump. Mid cut -3dB@700Hz Q=1.5 for the hollow slap scoop. LP@8kHz (vs SVT's 5kHz) lets the string pop and click through — critical for slap articulation.",
        "drive": 0.30,
        "asymmetry": 0.25,
        "hp_hz": 40.0,
        "low_shelf_hz": 80.0,
        "low_shelf_db": 4.0,
        "mid_hz": 700.0,
        "mid_db": -3.0,
        "mid_q": 1.5,
        "lp_hz": 8000.0,
    },
    "slap_bass": {
        "description": "Slap bass — low thump + high pop, mid scoop",
        "notes": "Low shelf +4dB@80Hz emphasises thumb thump. Mid cut -3dB@700Hz Q=1.5 creates the hollow slap scoop. LP@8kHz (vs SVT 5kHz) lets the string pop and click through. Light drive for clarity — slap needs attack definition, not warmth.",
        "drive": 0.20,
        "asymmetry": 0.15,
        "hp_hz": 40.0,
        "low_shelf_hz": 80.0,
        "low_shelf_db": 4.0,
        "mid_hz": 700.0,
        "mid_db": -3.0,
        "mid_q": 1.5,
        "lp_hz": 8000.0,
    },
    "di_clean": {
        "description": "Cabinet EQ only — no saturation, pure tonal shaping",
        "notes": "Applies the SVT cabinet frequency response shape without any tube saturation. Use when the DI already has good harmonics from a preamp or when saturation would conflict with parallel processing.",
        "drive": 0.0,
        "asymmetry": 0.0,
        "hp_hz": 40.0,
        "low_shelf_hz": 120.0,
        "low_shelf_db": 2.0,
        "mid_hz": None,
        "mid_db": 0.0,
        "mid_q": 2.0,
        "lp_hz": 5000.0,
    },
}


def _tube_saturate(data: np.ndarray, drive: float, asymmetry: float) -> np.ndarray:
    """Asymmetric soft clipping — positive half clips harder (triode model).

    Positive half: tanh(x * (1+asymmetry)) / (1+asymmetry)
    Negative half: tanh(x)

    The asymmetry generates even-order harmonics (2nd, 4th) — perceptually warm.
    Output is RMS-normalized to input level: harmonics are added, not volume.
    """
    if drive == 0.0:
        return data

    in_rms = np.sqrt(np.mean(data ** 2) + 1e-12)

    # Scale into saturation stage (drive=0 → no gain, drive=1 → 5x)
    x = data * (1.0 + drive * 4.0)

    pos_factor = 1.0 + asymmetry
    pos_mask = x >= 0
    neg_mask = ~pos_mask

    out = np.zeros_like(x)
    out[pos_mask] = np.tanh(x[pos_mask] * pos_factor) / pos_factor
    out[neg_mask] = np.tanh(x[neg_mask])

    # Normalize to input RMS — preserve level, only change spectrum
    out_rms = np.sqrt(np.mean(out ** 2) + 1e-12)
    if out_rms > 1e-10:
        out = out * (in_rms / out_rms)

    return out


def _hp(data: np.ndarray, sr: int, hz: float) -> np.ndarray:
    sos = butter(2, hz / (sr / 2.0), btype="high", output="sos")
    return sosfilt(sos, data, axis=0)


def _lp(data: np.ndarray, sr: int, hz: float) -> np.ndarray:
    sos = butter(2, hz / (sr / 2.0), btype="low", output="sos")
    return sosfilt(sos, data, axis=0)


def _low_shelf(data: np.ndarray, sr: int, hz: float, gain_db: float) -> np.ndarray:
    """Low shelf biquad — Audio EQ Cookbook (R. Bristow-Johnson), slope=1."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * hz / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    # slope=1.0: alpha = sin_w0/2 * sqrt(2)
    alpha = sin_w0 / 2.0 * np.sqrt(2.0)
    sqA = np.sqrt(A)

    b0 =     A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqA * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 =     A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqA * alpha)
    a0 =         ((A + 1) + (A - 1) * cos_w0 + 2 * sqA * alpha)
    a1 =    -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 =         ((A + 1) + (A - 1) * cos_w0 - 2 * sqA * alpha)

    sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
    return sosfilt(sos, data, axis=0)


def _peak(data: np.ndarray, sr: int, hz: float, gain_db: float, q: float) -> np.ndarray:
    """Peaking EQ biquad — Audio EQ Cookbook (R. Bristow-Johnson)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * hz / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
    return sosfilt(sos, data, axis=0)


def apply_amp(
    file_path: Path,
    output_dir: Path,
    drive: float = 0.35,
    asymmetry: float = 0.30,
    hp_hz: float | None = 40.0,
    lp_hz: float | None = 5000.0,
    low_shelf_hz: float | None = 120.0,
    low_shelf_db: float = 3.0,
    mid_hz: float | None = 800.0,
    mid_db: float = 2.0,
    mid_q: float = 2.0,
    preset_name: str | None = None,
) -> dict:
    data, sr = sf.read(str(file_path), always_2d=True)

    # Stage 1: tube saturation
    out = _tube_saturate(data, drive, asymmetry)

    # Stage 2: cabinet EQ
    if hp_hz is not None and hp_hz > 0:
        out = _hp(out, sr, hp_hz)
    if low_shelf_hz is not None and low_shelf_hz > 0 and abs(low_shelf_db) > 0.01:
        out = _low_shelf(out, sr, low_shelf_hz, low_shelf_db)
    if mid_hz is not None and mid_hz > 0 and abs(mid_db) > 0.01:
        out = _peak(out, sr, mid_hz, mid_db, mid_q)
    if lp_hz is not None and lp_hz > 0:
        out = _lp(out, sr, lp_hz)

    peak = float(np.max(np.abs(out)))
    if peak > 1.0:
        print(
            f"WARNING: clipping by {20 * np.log10(peak):.1f} dB — reducing gain",
            file=sys.stderr,
        )
        out = out / peak

    output_dir.mkdir(parents=True, exist_ok=True)
    preset_tag = f"_{preset_name}" if preset_name else ""
    out_path = output_dir / (file_path.stem + f"_amp{preset_tag}.wav")
    sf.write(str(out_path), out, sr, subtype="PCM_24")

    result = {
        "input": str(file_path),
        "output": str(out_path),
        "preset": preset_name,
        "drive": drive,
        "asymmetry": asymmetry,
        "hp_hz": hp_hz,
        "lp_hz": lp_hz,
        "low_shelf_hz": low_shelf_hz,
        "low_shelf_db": low_shelf_db,
        "mid_hz": mid_hz,
        "mid_db": mid_db,
        "mid_q": mid_q,
        "sample_rate": sr,
    }

    report_path = output_dir / "amp_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tube amp simulation and cabinet EQ for bass DI stems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available presets: " + ", ".join(PRESETS),
    )
    parser.add_argument("input", type=Path, nargs="?", help="Input WAV file")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--preset", choices=list(PRESETS), help="Amp preset")
    parser.add_argument("--drive", type=float, metavar="0-1", help="Saturation drive (0=off, 1=max)")
    parser.add_argument("--asymmetry", type=float, metavar="0-1", help="Asymmetry ratio for even harmonics (0=symmetric, 1=max)")
    parser.add_argument("--hp", type=float, metavar="HZ", help="High-pass filter (cabinet sub rolloff)")
    parser.add_argument("--lp", type=float, metavar="HZ", help="Low-pass filter (speaker rolloff)")
    parser.add_argument("--low-shelf-hz", type=float, metavar="HZ", help="Low shelf center frequency")
    parser.add_argument("--low-shelf-db", type=float, metavar="DB", help="Low shelf gain (positive=boost)")
    parser.add_argument("--mid-hz", type=float, metavar="HZ", help="Midrange peak frequency (grind)")
    parser.add_argument("--mid-db", type=float, metavar="DB", help="Midrange peak gain")
    parser.add_argument("--mid-q", type=float, metavar="Q", help="Midrange peak Q (bandwidth)")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")

    args = parser.parse_args()

    if args.list_presets:
        for name, p in PRESETS.items():
            print(f"\n{name}: {p['description']}")
            print(f"  drive={p['drive']}  asymmetry={p['asymmetry']}")
            print(f"  hp={p['hp_hz']}Hz  lp={p['lp_hz']}Hz")
            print(f"  low_shelf={p['low_shelf_hz']}Hz/{p['low_shelf_db']}dB  mid={p['mid_hz']}Hz/{p['mid_db']}dB  Q={p['mid_q']}")
            print(f"  {p['notes']}")
        return

    if args.input is None:
        parser.error("input file is required")
    if not args.input.exists():
        print(json.dumps({"error": f"Not found: {args.input}"}), file=sys.stderr)
        sys.exit(1)

    params: dict = {}
    preset_name = None
    if args.preset:
        preset_name = args.preset
        p = PRESETS[args.preset]
        params = {
            "drive": p["drive"],
            "asymmetry": p["asymmetry"],
            "hp_hz": p["hp_hz"],
            "lp_hz": p["lp_hz"],
            "low_shelf_hz": p["low_shelf_hz"],
            "low_shelf_db": p["low_shelf_db"],
            "mid_hz": p["mid_hz"],
            "mid_db": p["mid_db"],
            "mid_q": p["mid_q"],
        }
    else:
        params = {
            "drive": 0.35, "asymmetry": 0.30,
            "hp_hz": 40.0, "lp_hz": 5000.0,
            "low_shelf_hz": 120.0, "low_shelf_db": 3.0,
            "mid_hz": 800.0, "mid_db": 2.0, "mid_q": 2.0,
        }

    if args.drive is not None:         params["drive"] = args.drive
    if args.asymmetry is not None:     params["asymmetry"] = args.asymmetry
    if args.hp is not None:            params["hp_hz"] = args.hp
    if args.lp is not None:            params["lp_hz"] = args.lp
    if args.low_shelf_hz is not None:  params["low_shelf_hz"] = args.low_shelf_hz
    if args.low_shelf_db is not None:  params["low_shelf_db"] = args.low_shelf_db
    if args.mid_hz is not None:        params["mid_hz"] = args.mid_hz
    if args.mid_db is not None:        params["mid_db"] = args.mid_db
    if args.mid_q is not None:         params["mid_q"] = args.mid_q

    result = apply_amp(
        file_path=args.input,
        output_dir=args.output_dir,
        preset_name=preset_name,
        **params,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
