"""Apply transient shaping to a percussive stem.

Shapes the attack and sustain of each hit independently using a fast/slow
envelope follower pair. Works offline — no latency constraints.

  attack_db  (+) sharper, punchier onset   (-) softer onset
  sustain_db (+) longer, roomier body      (-) tighter, drier decay

Typical use cases:
  Kick too boomy:        --sustain -6
  Snare lacks crack:     --attack +4
  Tom rings too long:    --sustain -8
  Slap bass uneven:      not recommended — use compression instead

The algorithm:
  fast_env  = 2ms RMS window  (tracks transient onset)
  slow_env  = 50ms RMS window (tracks sustained body)
  attack_portion = max(fast - slow, 0)  -- active during transient
  sustain_portion = slow_env            -- active throughout body
  gain_db = attack_db * attack_norm + sustain_db * sustain_norm
  output  = input * 10^(gain_db/20)

Usage:
  python apply_transient.py assembled_eq_comp.wav --preset transient_kick_punch \\
      --output-dir output/session/KICK

  python apply_transient.py assembled_eq_comp.wav \\
      --attack 4 --sustain -6 --output-dir output/session/KICK

  python apply_transient.py assembled.wav --list-presets
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d

_PRESETS_DIR = Path(__file__).parent / "presets"


# ---------------------------------------------------------------------------
# Core DSP
# ---------------------------------------------------------------------------

def _rms_envelope(signal: np.ndarray, sr: int, window_ms: float) -> np.ndarray:
    """Fast RMS envelope via uniform moving average on squared signal."""
    win = max(1, int(window_ms * sr / 1000))
    return np.sqrt(uniform_filter1d(signal.astype(np.float64) ** 2, size=win, mode="reflect") + 1e-12)


def apply_transient(
    signal: np.ndarray,
    sr: int,
    attack_db: float,
    sustain_db: float,
    fast_ms: float = 2.0,
    slow_ms: float = 50.0,
) -> np.ndarray:
    """Apply transient shaping to a mono or stereo (N, C) signal.

    For stereo: envelope is computed from the mono sum, gain applied to both channels.
    """
    is_stereo = signal.ndim == 2
    mono = signal.mean(axis=1).astype(np.float64) if is_stereo else signal.astype(np.float64)

    fast_env = _rms_envelope(mono, sr, fast_ms)
    slow_env = _rms_envelope(mono, sr, slow_ms)

    attack_env = np.maximum(fast_env - slow_env, 0.0)
    attack_max = float(np.max(attack_env)) + 1e-10
    attack_norm = attack_env / attack_max

    sustain_max = float(np.max(slow_env)) + 1e-10
    sustain_norm = slow_env / sustain_max

    gain_db = attack_db * attack_norm + sustain_db * sustain_norm
    gain_linear = 10.0 ** (gain_db / 20.0)

    if is_stereo:
        return (signal.T * gain_linear).T
    return mono * gain_linear


# ---------------------------------------------------------------------------
# Preset loader
# ---------------------------------------------------------------------------

def _load_preset(name: str) -> dict:
    path = _PRESETS_DIR / f"{name}.json"
    if not path.exists():
        available = [p.stem for p in sorted(_PRESETS_DIR.glob("transient_*.json"))]
        raise FileNotFoundError(
            f"Preset '{name}' not found. Available: {', '.join(available)}"
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def apply_transient_file(
    input_path: Path,
    output_dir: Path,
    attack_db: float = 0.0,
    sustain_db: float = 0.0,
    fast_ms: float = 2.0,
    slow_ms: float = 50.0,
    preset: str | None = None,
) -> dict:
    """File-level wrapper around `apply_transient`: read WAV → shape → write WAV + report.

    Used by replay_chain's in-process pipeline. CLI `main()` delegates here.
    Preset (if given) sets the defaults; explicit attack/sustain kwargs override.
    """
    if preset:
        p = _load_preset(preset)
        s = p.get("settings", {})
        if attack_db == 0.0:
            attack_db = float(s.get("attack_db", 0.0))
        if sustain_db == 0.0:
            sustain_db = float(s.get("sustain_db", 0.0))
        fast_ms = float(s.get("fast_ms", fast_ms))
        slow_ms = float(s.get("slow_ms", slow_ms))

    data, sr = sf.read(str(input_path), always_2d=True)
    shaped = apply_transient(data, sr, attack_db, sustain_db, fast_ms, slow_ms)

    peak_in = float(20 * np.log10(np.max(np.abs(data)) + 1e-10))
    peak_out = float(20 * np.log10(np.max(np.abs(shaped)) + 1e-10))

    out_name = f"{input_path.stem}_transient.wav"
    out_path = output_dir / out_name
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), shaped, sr, subtype="PCM_24")

    report = {
        "input": str(input_path),
        "output": str(out_path),
        "preset": preset,
        "attack_db": attack_db,
        "sustain_db": sustain_db,
        "fast_ms": fast_ms,
        "slow_ms": slow_ms,
        "peak_in_dbfs": round(peak_in, 2),
        "peak_out_dbfs": round(peak_out, 2),
    }
    (output_dir / "transient_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply transient shaping to a stem.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  apply_transient.py assembled_eq_comp.wav --preset transient_kick_punch --output-dir output/session/KICK
  apply_transient.py assembled_eq_comp.wav --attack 4 --sustain -6 --output-dir output/session/KICK
  apply_transient.py assembled.wav --list-presets
        """,
    )
    parser.add_argument("file", nargs="?", type=Path, help="Input WAV file")
    parser.add_argument("--output-dir", type=Path, help="Directory to write output files")
    parser.add_argument("--preset", type=str, default=None, help="Preset name (transient_*)")
    parser.add_argument("--attack", type=float, default=None, help="Attack gain in dB (+/-)")
    parser.add_argument("--sustain", type=float, default=None, help="Sustain gain in dB (+/-)")
    parser.add_argument("--fast-ms", type=float, default=2.0, help="Fast envelope window in ms (default: 2)")
    parser.add_argument("--slow-ms", type=float, default=50.0, help="Slow envelope window in ms (default: 50)")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        presets = sorted(_PRESETS_DIR.glob("transient_*.json"))
        for p in presets:
            d = json.loads(p.read_text())
            s = d.get("settings", {})
            print(f"  {p.stem:<30} attack={s.get('attack_db',0):+.1f} dB  sustain={s.get('sustain_db',0):+.1f} dB  -- {d.get('description','')}")
        return

    if not args.file:
        parser.error("File argument required unless --list-presets")
    if not args.output_dir:
        parser.error("--output-dir required")
    if not args.file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    attack_db = 0.0
    sustain_db = 0.0
    fast_ms = args.fast_ms
    slow_ms = args.slow_ms

    if args.preset:
        p = _load_preset(args.preset)
        s = p.get("settings", {})
        attack_db = float(s.get("attack_db", 0.0))
        sustain_db = float(s.get("sustain_db", 0.0))
        fast_ms = float(s.get("fast_ms", fast_ms))
        slow_ms = float(s.get("slow_ms", slow_ms))
        print(f"Preset: {args.preset} — attack {attack_db:+.1f} dB, sustain {sustain_db:+.1f} dB")

    if args.attack is not None:
        attack_db = args.attack
    if args.sustain is not None:
        sustain_db = args.sustain

    if attack_db == 0.0 and sustain_db == 0.0:
        print("Warning: both attack and sustain are 0 dB — no shaping applied", file=sys.stderr)

    data, sr = sf.read(str(args.file), always_2d=True)
    channels = data.shape[1]

    print(f"Input:   {args.file.name}  ({data.shape[0]/sr:.1f}s, {channels}ch, {sr}Hz)")
    print(f"Shaping: attack {attack_db:+.1f} dB  sustain {sustain_db:+.1f} dB  "
          f"(fast {fast_ms}ms / slow {slow_ms}ms)")

    shaped = apply_transient(data, sr, attack_db, sustain_db, fast_ms, slow_ms)

    peak_in = float(20 * np.log10(np.max(np.abs(data)) + 1e-10))
    peak_out = float(20 * np.log10(np.max(np.abs(shaped)) + 1e-10))
    print(f"Peak:    {peak_in:.1f} dBFS -> {peak_out:.1f} dBFS")

    stem = args.file.stem
    out_name = f"{stem}_transient.wav"
    out_path = args.output_dir / out_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), shaped, sr, subtype="PCM_24")

    report = {
        "input": str(args.file),
        "output": str(out_path),
        "preset": args.preset,
        "attack_db": attack_db,
        "sustain_db": sustain_db,
        "fast_ms": fast_ms,
        "slow_ms": slow_ms,
        "peak_in_dbfs": round(peak_in, 2),
        "peak_out_dbfs": round(peak_out, 2),
    }
    report_path = args.output_dir / "transient_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Output:  {out_path}")
    print(f"Report:  {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
