"""Apply dynamic range compression to an assembled stem.

Uses pedalboard (Spotify/JUCE) for sample-accurate compression — same engine
as professional DAW plugins.

Parameters:
  threshold  dB level where compression starts (e.g. -20)
  ratio      compression ratio (e.g. 4  means 4:1)
  attack     attack time in ms — how fast compression engages (e.g. 10)
  release    release time in ms — how fast compression lets go (e.g. 100)
  makeup     makeup gain in dB added after compression (default: auto)
  mix        wet/dry blend 0.0-1.0 (default 1.0 = serial; <1.0 = parallel)

Parallel compression (New York style): set mix < 1.0.
  Output = dry * (1 - mix) + compressed * mix
  Preserves original transients while adding body/sustain from the compressed copy.

Sidechain compression: use --sidechain to drive gain reduction from an external signal.
  --sidechain  path to reference WAV (e.g. kick driving bass compression)
  --sc-hp      high-pass the sidechain at this Hz (isolates kick sub, default: none)
  --sc-lp      low-pass the sidechain at this Hz (default: none)
  Example: compress bass, sidechain from kick, HP at 60 Hz to isolate beater click.

Usage:
  python apply_compression.py assembled.wav --output-dir output/session/TRACK \\
      --threshold -20 --ratio 4 --attack 10 --release 100

  python apply_compression.py assembled_eq.wav --output-dir output/session/TRACK \\
      --preset comp_kick_in

  python apply_compression.py assembled_eq.wav --output-dir output/session/TRACK \\
      --preset comp_snare_bottom --mix 0.3

  python apply_compression.py bass.wav --output-dir output/session/BASS \\
      --preset comp_bass_di --sidechain kick.wav --sc-hp 60

  python apply_compression.py *.wav --list-presets
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Compressor, Gain, Pedalboard
from scipy.signal import butter, sosfilt

_PRESETS_DIR = Path(__file__).parent / "presets"


# ---------------------------------------------------------------------------
# Auto makeup gain estimate
# ---------------------------------------------------------------------------

def _auto_makeup_db(threshold_db: float, ratio: float) -> float:
    """Estimate makeup gain from threshold and ratio (50% of max theoretical GR)."""
    return round(-threshold_db * (1.0 - 1.0 / ratio) * 0.5, 1)


# ---------------------------------------------------------------------------
# Preset loader
# ---------------------------------------------------------------------------

def _load_preset(name: str) -> dict:
    path = _PRESETS_DIR / f"{name}.json"
    if not path.exists():
        available = [p.stem for p in sorted(_PRESETS_DIR.glob("comp_*.json"))]
        raise FileNotFoundError(
            f"Preset {name!r} not found. Available compression presets: {', '.join(available)}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def list_presets() -> None:
    if not _PRESETS_DIR.exists():
        print("No presets directory found.")
        return
    for path in sorted(_PRESETS_DIR.glob("comp_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data.get("settings", {})
        desc = data.get("description", "")
        print(
            f"  {path.stem:<28}  {desc}"
            f"  [{s.get('ratio','?')}:1  atk {s.get('attack_ms','?')}ms  rel {s.get('release_ms','?')}ms]"
        )


# ---------------------------------------------------------------------------
# Sidechain helpers
# ---------------------------------------------------------------------------

def _sc_filter(audio: np.ndarray, sr: int, hp_hz: float | None, lp_hz: float | None) -> np.ndarray:
    """Apply Butterworth HP and/or LP to a sidechain signal (samples, channels)."""
    filtered = audio.copy()
    if hp_hz and hp_hz > 0:
        sos = butter(2, hp_hz, btype="highpass", fs=sr, output="sos")
        filtered = sosfilt(sos, filtered, axis=0)
    if lp_hz and lp_hz > 0:
        sos = butter(2, lp_hz, btype="lowpass", fs=sr, output="sos")
        filtered = sosfilt(sos, filtered, axis=0)
    return filtered


def _sidechain_gain_envelope(
    sc_mono: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    """Return a per-sample gain multiplier [0.0–1.0] driven by sc_mono.

    Uses 1ms block processing (peak detection) with exponential ballistics.
    """
    n = len(sc_mono)
    block_samples = max(1, sr // 1000)
    n_blocks = (n + block_samples - 1) // block_samples

    # Smoothing coefficients per block (alpha near 1 = slower response)
    alpha_a = np.exp(-1.0 / max(1.0, attack_ms * 0.001 * sr / block_samples))
    alpha_r = np.exp(-1.0 / max(1.0, release_ms * 0.001 * sr / block_samples))

    threshold_lin = 10.0 ** (threshold_db / 20.0)
    inv_ratio = 1.0 / ratio

    gr_blocks = np.ones(n_blocks)
    env = 0.0

    for i in range(n_blocks):
        start = i * block_samples
        end = min(start + block_samples, n)
        level = float(np.max(np.abs(sc_mono[start:end])))

        # Ballistic envelope follower
        if level > env:
            env = alpha_a * env + (1.0 - alpha_a) * level
        else:
            env = alpha_r * env + (1.0 - alpha_r) * level

        # Static compression characteristic
        if env > threshold_lin:
            env_db = 20.0 * np.log10(max(env, 1e-10))
            # Gain reduction in dB (always <= 0)
            gr_db = (threshold_db + (env_db - threshold_db) * inv_ratio) - env_db
            gr_blocks[i] = 10.0 ** (gr_db / 20.0)
        # else: gr_blocks[i] stays 1.0

    # Interpolate gain from block resolution to sample resolution
    block_centers = np.arange(n_blocks) * block_samples + block_samples // 2
    gain = np.interp(np.arange(n), block_centers, gr_blocks)
    return gain.astype(np.float64)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def apply_compression(
    input_path: Path,
    output_dir: Path,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float | None = None,
    mix: float = 1.0,
    preset_name: str | None = None,
    sidechain_path: Path | None = None,
    sc_hp_hz: float | None = None,
    sc_lp_hz: float | None = None,
) -> dict:
    data, sr = sf.read(str(input_path), always_2d=True)

    sidechain_info: dict | None = None

    if sidechain_path is not None:
        # In sidechain mode the compressor doesn't process the main signal directly,
        # so auto-makeup based on threshold/ratio is not meaningful — default to 0.
        if makeup_db is None:
            makeup_db = 0.0
        # --- Custom sidechain path ---
        sc_data, sc_sr = sf.read(str(sidechain_path), always_2d=True)

        # Resample sidechain if sample rates differ (rare but safe)
        if sc_sr != sr:
            raise ValueError(
                f"Sidechain sample rate {sc_sr} Hz differs from main {sr} Hz — "
                "resample before using as sidechain."
            )

        # Trim or zero-pad sidechain to match main signal length
        n_main = data.shape[0]
        if sc_data.shape[0] > n_main:
            sc_data = sc_data[:n_main]
        elif sc_data.shape[0] < n_main:
            pad = n_main - sc_data.shape[0]
            sc_data = np.pad(sc_data, ((0, pad), (0, 0)))

        # Sum to mono, apply SC filters
        sc_mono = sc_data.mean(axis=1)
        if sc_hp_hz or sc_lp_hz:
            sc_mono = _sc_filter(sc_mono[:, None], sr, sc_hp_hz, sc_lp_hz).squeeze()

        # Compute per-sample gain reduction envelope
        gain_env = _sidechain_gain_envelope(
            sc_mono, sr, threshold_db, ratio, attack_ms, release_ms
        )

        makeup_lin = 10.0 ** (makeup_db / 20.0)
        compressed = data * gain_env[:, None] * makeup_lin

        gr_db_mean = float(20.0 * np.log10(max(float(np.mean(gain_env)), 1e-10)))
        gr_db_max = float(20.0 * np.log10(max(float(np.min(gain_env)), 1e-10)))

        sidechain_info = {
            "file": str(sidechain_path),
            "sc_hp_hz": sc_hp_hz,
            "sc_lp_hz": sc_lp_hz,
            "mean_gain_reduction_db": round(gr_db_mean, 1),
            "peak_gain_reduction_db": round(gr_db_max, 1),
        }
    else:
        # --- Pedalboard path (no sidechain) ---
        if makeup_db is None:
            makeup_db = _auto_makeup_db(threshold_db, ratio)
        audio_f32 = data.T.astype(np.float32)
        board = Pedalboard([
            Compressor(
                threshold_db=float(threshold_db),
                ratio=float(ratio),
                attack_ms=float(attack_ms),
                release_ms=float(release_ms),
            ),
            Gain(gain_db=float(makeup_db)),
        ])
        compressed = board(audio_f32, sr).T.astype(np.float64)

    # Parallel blend: dry + compressed
    if mix < 0.999:
        output_data = data * (1.0 - mix) + compressed * mix
    else:
        output_data = compressed

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
    out_path = output_dir / (input_path.stem + "_comp.wav")
    sf.write(str(out_path), output_data, sr, subtype="PCM_24")

    in_peak = float(20 * np.log10(max(float(np.max(np.abs(data))), 1e-10)))
    out_peak = float(20 * np.log10(max(float(np.max(np.abs(output_data))), 1e-10)))

    report = {
        "input": str(input_path),
        "output": str(out_path),
        "preset": preset_name,
        "settings": {
            "threshold_db": threshold_db,
            "ratio": ratio,
            "attack_ms": attack_ms,
            "release_ms": release_ms,
            "makeup_db": makeup_db,
            "mix": mix,
        },
        "sidechain": sidechain_info,
        "input_peak_dbfs": round(in_peak, 1),
        "output_peak_dbfs": round(out_peak, 1),
        "net_level_change_db": round(out_peak - in_peak, 1),
        "clipping_prevented": clipped,
        "sample_rate": sr,
    }
    (output_dir / "comp_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply dynamic range compression to a stem.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", type=Path, nargs="?", help="Input WAV file")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--threshold", type=float, metavar="DB", help="Threshold in dBFS (e.g. -20)")
    parser.add_argument("--ratio",     type=float, help="Compression ratio (e.g. 4 for 4:1)")
    parser.add_argument("--attack",    type=float, metavar="MS", help="Attack time in ms")
    parser.add_argument("--release",   type=float, metavar="MS", help="Release time in ms")
    parser.add_argument("--makeup",    type=float, metavar="DB", help="Makeup gain dB (default: auto)")
    parser.add_argument(
        "--mix", type=float, default=1.0, metavar="0-1",
        help="Wet/dry blend: 1.0=serial, <1.0=parallel (default 1.0)",
    )
    parser.add_argument("--preset", metavar="NAME", help="Compression preset (see --list-presets)")
    parser.add_argument("--list-presets", action="store_true", help="List compression presets and exit")
    sc_group = parser.add_argument_group("sidechain")
    sc_group.add_argument(
        "--sidechain", type=Path, metavar="FILE",
        help="External sidechain reference WAV (e.g. kick driving bass compression)",
    )
    sc_group.add_argument(
        "--sc-hp", type=float, metavar="HZ",
        help="High-pass sidechain signal at this frequency before detection",
    )
    sc_group.add_argument(
        "--sc-lp", type=float, metavar="HZ",
        help="Low-pass sidechain signal at this frequency before detection",
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

    threshold = args.threshold
    ratio     = args.ratio
    attack    = args.attack
    release   = args.release
    makeup    = args.makeup
    mix       = args.mix
    preset_name = args.preset

    if args.preset:
        try:
            pdata = _load_preset(args.preset)
        except FileNotFoundError as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(1)
        s = pdata.get("settings", {})
        # CLI args override preset values
        if threshold is None: threshold = s["threshold_db"]
        if ratio     is None: ratio     = s["ratio"]
        if attack    is None: attack    = s["attack_ms"]
        if release   is None: release   = s["release_ms"]
        if makeup    is None: makeup    = s.get("makeup_db")
        if mix == 1.0 and "mix" in s: mix = s["mix"]
        print(f"Loaded preset {args.preset!r}: "
              f"thr={threshold} ratio={ratio}:1 atk={attack}ms rel={release}ms "
              f"makeup={makeup} mix={mix}", file=sys.stderr)

    for name, val in [("--threshold", threshold), ("--ratio", ratio),
                      ("--attack", attack), ("--release", release)]:
        if val is None:
            parser.error(f"{name} is required (or use --preset)")

    if args.sidechain and not args.sidechain.exists():
        print(json.dumps({"error": f"Sidechain file not found: {args.sidechain}"}), file=sys.stderr)
        sys.exit(1)

    apply_compression(
        args.file, args.output_dir,
        threshold_db=threshold, ratio=ratio,
        attack_ms=attack, release_ms=release,
        makeup_db=makeup, mix=mix,
        preset_name=preset_name,
        sidechain_path=args.sidechain,
        sc_hp_hz=args.sc_hp,
        sc_lp_hz=args.sc_lp,
    )


if __name__ == "__main__":
    main()
