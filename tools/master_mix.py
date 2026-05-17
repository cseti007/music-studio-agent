"""Mastering: independent finishing pass on a stereo mix.wav.

Separate from `render_mix.py`'s master chain. The render_mix chain represents
the mix-engineer's polish — it's calibrated for stem-summed material and runs
inside the render. This tool represents the master-engineer's pass — runs on
a finished stereo file, may be more aggressive, and is calibrated for
format-specific delivery targets (Spotify -14, Apple -16, CD -9, etc).

Chain order (each step optional, controlled by the mastering preset):

  EQ -> glue comp -> harmonic exciter -> clipper -> LUFS norm -> true-peak limiter -> dither

Outputs are named after the format preset: `master_spotify.wav`, etc. The
`--all-formats` mode produces all delivery targets from one input in a
single batch run.

Usage:
  # Single format
  python master_mix.py mix.wav --output-dir output/<session>/masters \
      --format spotify --master-preset modern_rock

  # Batch — all major delivery formats
  python master_mix.py mix.wav --output-dir output/<session>/masters \
      --all-formats --master-preset modern_rock

  # Custom target (no format preset)
  python master_mix.py mix.wav --output-dir DIR \
      --target-lufs -12 --tp-ceiling -1.0 --master-preset gentle
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import Compressor, Gain, Limiter, Pedalboard
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt

# Reuse pieces already in the project
sys.path.insert(0, str(Path(__file__).parent))
from apply_eq import _build_sos  # noqa: E402
from render_mix import (  # noqa: E402
    _hard_clip,
    _measure_true_peak_dbfs,
    _soft_clip,
)


# ---------------------------------------------------------------------------
# Format delivery presets — LUFS / true peak / bit depth / dithering
# ---------------------------------------------------------------------------

FORMAT_PRESETS: dict[str, dict] = {
    "spotify": {
        "description": "Spotify streaming target (Ogg Vorbis encode)",
        "target_lufs": -14.0,
        "tp_ceiling_dbtp": -1.0,
        "bit_depth": 24,
        "dither": False,
    },
    "apple": {
        "description": "Apple Music streaming target (AAC encode)",
        "target_lufs": -16.0,
        "tp_ceiling_dbtp": -1.0,
        "bit_depth": 24,
        "dither": False,
    },
    "youtube": {
        "description": "YouTube streaming target",
        "target_lufs": -14.0,
        "tp_ceiling_dbtp": -1.0,
        "bit_depth": 24,
        "dither": False,
    },
    "tidal": {
        "description": "Tidal streaming target (HiFi tier)",
        "target_lufs": -14.0,
        "tp_ceiling_dbtp": -1.0,
        "bit_depth": 24,
        "dither": False,
    },
    "cd": {
        "description": "CD delivery — louder, 16-bit dithered",
        "target_lufs": -9.0,
        "tp_ceiling_dbtp": -1.0,
        "bit_depth": 16,
        "dither": True,
    },
    "vinyl_pre": {
        "description": "Vinyl pre-master — gentle, no clipper/limiter, preserves dynamics for cutting",
        "target_lufs": -12.0,
        "tp_ceiling_dbtp": -1.0,
        "bit_depth": 24,
        "dither": False,
        "skip_clipper": True,
        "skip_limiter": True,
    },
    "broadcast": {
        "description": "Broadcast EBU R128 target",
        "target_lufs": -23.0,
        "tp_ceiling_dbtp": -2.0,
        "bit_depth": 24,
        "dither": False,
    },
}


# ---------------------------------------------------------------------------
# Mastering chain presets — EQ / comp / exciter / clipper settings
# ---------------------------------------------------------------------------

MASTERING_PRESETS: dict[str, dict] = {
    "gentle": {
        "description": "Conservative master — preserves dynamics, no clipping. Good for jazz, acoustic, classical-leaning rock.",
        "eq": [],
        "comp": {"threshold_db": -12, "ratio": 1.5, "attack_ms": 30, "release_ms": 300, "makeup_db": 0.5},
        "exciter": None,
        "clipper": None,
    },
    "modern_rock": {
        "description": "Competitive rock master — moderate clipper, gentle exciter, audible LUFS lift.",
        "eq": [{"type": "highshelf", "hz": 12000, "db": 1.0, "slope": 1.0}],
        "comp": {"threshold_db": -10, "ratio": 2.0, "attack_ms": 20, "release_ms": 250, "makeup_db": 0.0},
        "exciter": {"hp_hz": 200.0, "drive": 0.4, "mix": 0.10},
        "clipper": {"threshold_db": -2.0, "knee_db": 1.5, "mode": "soft"},
    },
    "pop": {
        "description": "Pop master — bright EQ, present mids, soft clip for loudness.",
        "eq": [
            {"type": "highshelf", "hz": 10000, "db": 1.5, "slope": 1.0},
            {"type": "peak", "hz": 3500, "q": 1.0, "db": 1.0},
        ],
        "comp": {"threshold_db": -9, "ratio": 2.5, "attack_ms": 15, "release_ms": 200, "makeup_db": 0.0},
        "exciter": {"hp_hz": 6000.0, "drive": 0.5, "mix": 0.12},
        "clipper": {"threshold_db": -1.5, "knee_db": 1.0, "mode": "soft"},
    },
    "hip_hop": {
        "description": "Hip-hop master — sub weight, hard clip for impact, modest loudness.",
        "eq": [
            {"type": "lowshelf", "hz": 80, "db": 1.0, "slope": 1.0},
            {"type": "highshelf", "hz": 12000, "db": 0.5, "slope": 1.0},
        ],
        "comp": {"threshold_db": -8, "ratio": 2.0, "attack_ms": 25, "release_ms": 200, "makeup_db": 0.0},
        "exciter": {"hp_hz": 100.0, "drive": 0.6, "mix": 0.15},
        "clipper": {"threshold_db": -1.5, "knee_db": 1.0, "mode": "hard"},
    },
    "transparent": {
        "description": "No EQ, no exciter, only LUFS norm + true peak limit. For mixes that don't need master tone.",
        "eq": [],
        "comp": None,
        "exciter": None,
        "clipper": None,
    },
}


# ---------------------------------------------------------------------------
# Chain helpers
# ---------------------------------------------------------------------------

def _master_eq(stereo: np.ndarray, sr: int, filters: list) -> np.ndarray:
    """Apply EQ chain to a (2, N) master buffer. Zero-phase for mastering."""
    if not filters:
        return stereo
    out = stereo.copy()
    for f in filters:
        sos = _build_sos(f, sr)
        out = sosfiltfilt(sos, out, axis=1)
    return out


def _master_comp(stereo: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """Pedalboard-based glue compressor for the master bus."""
    board = Pedalboard([
        Compressor(
            threshold_db=float(params["threshold_db"]),
            ratio=float(params["ratio"]),
            attack_ms=float(params["attack_ms"]),
            release_ms=float(params["release_ms"]),
        ),
        Gain(gain_db=float(params.get("makeup_db", 0.0))),
    ])
    return board(stereo.astype(np.float32), sr).astype(np.float64)


def _harmonic_exciter(stereo: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """Add a saturated copy of the high-passed signal back in for spectral
    excitation. Different from apply_exciter.py's tool: this runs on the
    full stereo master, not a stem, and the mix is set by mastering preset.

    Generates 2nd-order harmonic content above hp_hz; mixes in at low level
    so the dry remains dominant.
    """
    hp_hz = float(params["hp_hz"])
    drive = float(params["drive"])
    mix = float(params["mix"])
    nyq = sr / 2.0
    sos = butter(4, hp_hz / nyq, btype="high", output="sos")

    out = stereo.copy()
    for ch in range(stereo.shape[0]):
        hp = sosfilt(sos, stereo[ch])
        saturated = np.tanh(hp * drive)
        # Re-filter so only HF content is added (saturation creates IM products)
        harmonics = sosfilt(sos, saturated)
        out[ch] = stereo[ch] + harmonics * mix
    return out


def _dither_to_16bit(stereo: np.ndarray) -> np.ndarray:
    """TPDF (triangular probability density) dither for 16-bit truncation.

    Adds noise at the LSB level so the quantization distortion becomes
    spectrally flat noise instead of audible artifacts on quiet passages.
    """
    # 16-bit LSB = 1/32768. TPDF: sum of two independent uniform [-0.5, 0.5] LSBs.
    lsb = 1.0 / 32768.0
    rng = np.random.default_rng(42)  # deterministic dither for reproducibility
    noise = (rng.random(stereo.shape) - 0.5) + (rng.random(stereo.shape) - 0.5)
    return stereo + noise * lsb


# ---------------------------------------------------------------------------
# Core mastering function
# ---------------------------------------------------------------------------

def master_mix(
    input_path: Path,
    output_dir: Path,
    format_name: str,
    mastering_preset: str,
    target_lufs: float | None = None,
    tp_ceiling: float | None = None,
) -> dict:
    """Master a stereo mix.wav to a delivery format.

    Returns a report dict; writes <output_dir>/master_<format>.wav and
    <output_dir>/master_<format>_report.json.
    """
    fmt = FORMAT_PRESETS[format_name]
    mp = MASTERING_PRESETS[mastering_preset]

    # Format overrides (user can specify --target-lufs / --tp-ceiling)
    actual_target_lufs = target_lufs if target_lufs is not None else fmt["target_lufs"]
    actual_tp = tp_ceiling if tp_ceiling is not None else fmt["tp_ceiling_dbtp"]
    bit_depth = fmt.get("bit_depth", 24)
    do_dither = fmt.get("dither", False)
    skip_clipper = fmt.get("skip_clipper", False)
    skip_limiter = fmt.get("skip_limiter", False)

    print(f"Mastering {input_path.name} → format={format_name} preset={mastering_preset}", flush=True)
    print(f"  Target LUFS {actual_target_lufs}, TP ceiling {actual_tp} dBTP, {bit_depth}-bit"
          + (" (dithered)" if do_dither else ""))

    data, sr = sf.read(str(input_path), always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    # Switch to (2, N) layout for the chain
    master = data.T.astype(np.float64)

    meter = pyln.Meter(sr)
    lufs_before = float(meter.integrated_loudness(master.T))
    peak_before = 20.0 * np.log10(max(np.max(np.abs(master)), 1e-12))
    print(f"  Input: LUFS {lufs_before:.2f}, sample peak {peak_before:.2f} dBFS")

    chain_applied: list[str] = []

    # 1. EQ
    if mp["eq"]:
        master = _master_eq(master, sr, mp["eq"])
        chain_applied.append(f"eq({len(mp['eq'])} filters)")

    # 2. Glue comp
    if mp["comp"]:
        lufs_pre = float(meter.integrated_loudness(master.T))
        master = _master_comp(master, sr, mp["comp"])
        lufs_post = float(meter.integrated_loudness(master.T))
        chain_applied.append(
            f"comp(thr={mp['comp']['threshold_db']}, ratio={mp['comp']['ratio']}, GR={lufs_post - lufs_pre:+.2f} LU)"
        )

    # 3. Exciter
    if mp["exciter"]:
        master = _harmonic_exciter(master, sr, mp["exciter"])
        chain_applied.append(f"exciter(hp={mp['exciter']['hp_hz']}, mix={mp['exciter']['mix']})")

    # 4. Clipper
    if mp["clipper"] and not skip_clipper:
        cp = mp["clipper"]
        if cp["mode"] == "hard":
            master = _hard_clip(master, float(cp["threshold_db"]))
        else:
            master = _soft_clip(master, float(cp["threshold_db"]), float(cp.get("knee_db", 1.5)))
        chain_applied.append(f"clipper({cp['mode']}, thr={cp['threshold_db']})")
    elif skip_clipper:
        chain_applied.append("clipper(SKIPPED for format)")

    # 5. LUFS normalization
    lufs_pre_norm = float(meter.integrated_loudness(master.T))
    if np.isfinite(lufs_pre_norm):
        gain_db = actual_target_lufs - lufs_pre_norm
        master = master * 10.0 ** (gain_db / 20.0)
        chain_applied.append(f"lufs_norm({gain_db:+.2f} dB → target {actual_target_lufs})")

    # 6. True peak limiter (oversampled — ISP-aware)
    if not skip_limiter:
        board = Pedalboard([Limiter(threshold_db=float(actual_tp), release_ms=100.0)])
        master = board(master.astype(np.float32), sr).astype(np.float64)
        # Second-pass ISP check + scale down if still over
        tp_measured = _measure_true_peak_dbfs(master)
        if tp_measured > actual_tp:
            scale_db = actual_tp - tp_measured
            master *= 10.0 ** (scale_db / 20.0)
            chain_applied.append(f"limiter(ISP corrected {scale_db:+.2f} dB)")
        else:
            chain_applied.append(f"limiter(TP {tp_measured:.2f} dBTP)")

        # Post-limiter LUFS correction: pedalboard.Limiter applies internal
        # makeup gain, so the integrated LUFS after limiting is usually
        # higher than the LUFS norm target we asked for. Re-measure and
        # scale DOWN if we overshoot (only downward — upward correction
        # would push the true peak back over the ceiling).
        lufs_after = float(meter.integrated_loudness(master.T))
        if np.isfinite(lufs_after):
            correction_db = actual_target_lufs - lufs_after
            if correction_db < -0.2:  # only correct if overshoot > 0.2 dB
                master *= 10.0 ** (correction_db / 20.0)
                chain_applied.append(f"lufs_post_correction({correction_db:+.2f} dB)")
    else:
        chain_applied.append("limiter(SKIPPED for format)")

    # 7. Dither (only for 16-bit output)
    if do_dither:
        master = _dither_to_16bit(master)
        chain_applied.append("dither(TPDF 16-bit)")

    # Output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"master_{format_name}.wav"
    subtype = "PCM_16" if bit_depth == 16 else "PCM_24"
    sf.write(str(out_path), master.T, sr, subtype=subtype)

    # Final measurements
    lufs_final = float(meter.integrated_loudness(master.T))
    lra_final = round(float(meter.loudness_range(master.T)), 2)
    tp_final = _measure_true_peak_dbfs(master)
    sample_peak_final = 20.0 * np.log10(max(np.max(np.abs(master)), 1e-12))

    report = {
        "input": str(input_path),
        "output": str(out_path),
        "format": format_name,
        "format_description": fmt["description"],
        "mastering_preset": mastering_preset,
        "mastering_preset_description": mp["description"],
        "chain": chain_applied,
        "input_lufs": round(lufs_before, 2),
        "output_lufs": round(lufs_final, 2),
        "target_lufs": actual_target_lufs,
        "lufs_delta_from_target": round(lufs_final - actual_target_lufs, 2),
        "output_lra_lu": lra_final,
        "output_true_peak_dbtp": round(tp_final, 2),
        "output_sample_peak_dbfs": round(sample_peak_final, 2),
        "tp_ceiling_dbtp": actual_tp,
        "bit_depth": bit_depth,
        "dithered": do_dither,
        "sample_rate": sr,
    }
    (output_dir / f"master_{format_name}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"  Output: {out_path}")
    print(f"  LUFS {lufs_final:.2f} (target {actual_target_lufs}, delta {lufs_final - actual_target_lufs:+.2f}), "
          f"LRA {lra_final} LU, TP {tp_final:.2f} dBTP")
    print(f"  Chain: {' -> '.join(chain_applied)}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_format_presets() -> None:
    print("Format presets (delivery targets):")
    for name, p in FORMAT_PRESETS.items():
        bit = f"{p['bit_depth']}-bit" + (" dithered" if p.get("dither") else "")
        skip = []
        if p.get("skip_clipper"):
            skip.append("no clipper")
        if p.get("skip_limiter"):
            skip.append("no limiter")
        skip_str = f"  [{', '.join(skip)}]" if skip else ""
        print(f"  {name:<14}  LUFS {p['target_lufs']:>5}, TP {p['tp_ceiling_dbtp']:>4} dBTP, "
              f"{bit}{skip_str}")
        print(f"  {'':<14}  {p['description']}")


def list_mastering_presets() -> None:
    print("Mastering chain presets:")
    for name, p in MASTERING_PRESETS.items():
        print(f"  {name:<14}  {p['description']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master a stereo mix.wav to one or all delivery formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, nargs="?", help="Input stereo mix WAV")
    parser.add_argument("--output-dir", type=Path, help="Output directory for master_<format>.wav files")
    parser.add_argument("--format", choices=list(FORMAT_PRESETS),
                        help="Single format preset (e.g. spotify, apple, cd)")
    parser.add_argument("--all-formats", action="store_true",
                        help="Render all major formats (spotify, apple, youtube, tidal, cd, vinyl_pre)")
    parser.add_argument("--master-preset", choices=list(MASTERING_PRESETS),
                        default="modern_rock",
                        help="Mastering chain preset (EQ + comp + exciter + clipper)")
    parser.add_argument("--target-lufs", type=float,
                        help="Override format target LUFS")
    parser.add_argument("--tp-ceiling", type=float,
                        help="Override format true peak ceiling (dBTP)")
    parser.add_argument("--list-formats", action="store_true",
                        help="List delivery format presets and exit")
    parser.add_argument("--list-master-presets", action="store_true",
                        help="List mastering chain presets and exit")
    args = parser.parse_args()

    if args.list_formats:
        list_format_presets()
        return
    if args.list_master_presets:
        list_mastering_presets()
        return

    if args.input is None or args.output_dir is None:
        parser.error("input and --output-dir are required (unless using --list-*)")
    if not args.input.exists():
        print(json.dumps({"error": f"Not found: {args.input}"}), file=sys.stderr)
        sys.exit(1)
    if not args.format and not args.all_formats:
        parser.error("either --format <name> or --all-formats is required")

    if args.all_formats:
        formats = ["spotify", "apple", "youtube", "tidal", "cd", "vinyl_pre"]
    else:
        formats = [args.format]

    results = []
    for fmt in formats:
        print()
        r = master_mix(
            input_path=args.input,
            output_dir=args.output_dir,
            format_name=fmt,
            mastering_preset=args.master_preset,
            target_lufs=args.target_lufs,
            tp_ceiling=args.tp_ceiling,
        )
        results.append(r)

    # Batch summary if multi-format
    if len(formats) > 1:
        print("\n" + "=" * 60)
        print("BATCH MASTERING SUMMARY")
        print("=" * 60)
        for r in results:
            print(f"  {r['format']:<14}  LUFS {r['output_lufs']:+.2f}  "
                  f"TP {r['output_true_peak_dbtp']:+.2f} dBTP  "
                  f"LRA {r['output_lra_lu']:.1f} LU  →  {Path(r['output']).name}")


if __name__ == "__main__":
    main()
