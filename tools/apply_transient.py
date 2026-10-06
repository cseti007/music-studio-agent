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

The algorithm (level-independent differential envelope):
  env      = fast_ms RMS envelope (tracks the hit)
  slow     = causal one-pole smoothing of env with slow_ms time constant
  diff_db  = 20*log10(env / slow)  -- positive on onsets, negative on decays
  gain_db  = attack_db  * clip(+diff_db / 6, 0, 1)
           + sustain_db * clip(-diff_db / 6, 0, 1)
  output   = input * 10^(gain_db/20)
  Only the ratio of the two envelopes is used, so the same hit shaped at
  any level receives the same gain curve. The gain never exceeds the
  requested attack/sustain amounts.

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
from scipy.signal import lfilter

from _recall import record_operation

_PRESETS_DIR = Path(__file__).parent / "presets"
# Envelope difference (dB) at which the full attack/sustain gain applies.
_FULL_SCALE_DIFF_DB = 6.0


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

    env = _rms_envelope(mono, sr, fast_ms)
    # Causal one-pole smoothing lags rising edges (onsets) and falling edges
    # (decays), so the env/slow ratio separates attack from sustain.
    a = np.exp(-1.0 / max(1.0, slow_ms * sr / 1000.0))
    slow, _ = lfilter([1.0 - a], [1.0, -a], env, zi=[a * env[0]])
    diff_db = 20.0 * np.log10(env / np.maximum(slow, 1e-12))

    attack_w = np.clip(diff_db / _FULL_SCALE_DIFF_DB, 0.0, 1.0)
    sustain_w = np.clip(-diff_db / _FULL_SCALE_DIFF_DB, 0.0, 1.0)
    gain_db = attack_db * attack_w + sustain_db * sustain_w
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

@record_operation("apply_transient")
def apply_transient_file(
    input_path: Path,
    output_dir: Path,
    attack_db: float | None = None,
    sustain_db: float | None = None,
    fast_ms: float | None = None,
    slow_ms: float | None = None,
    preset: str | None = None,
) -> dict:
    """File-level wrapper around `apply_transient`: read WAV → shape → write WAV + report.

    Used by replay_chain's in-process pipeline. CLI `main()` delegates here.
    Explicit (non-None) values win; otherwise the preset supplies them, then
    the defaults (0 dB, 0 dB, 2 ms, 50 ms).
    """
    s = _load_preset(preset).get("settings", {}) if preset else {}
    attack_db = float(attack_db if attack_db is not None else s.get("attack_db", 0.0))
    sustain_db = float(sustain_db if sustain_db is not None else s.get("sustain_db", 0.0))
    fast_ms = float(fast_ms if fast_ms is not None else s.get("fast_ms", 2.0))
    slow_ms = float(slow_ms if slow_ms is not None else s.get("slow_ms", 50.0))

    data, sr = sf.read(str(input_path), always_2d=True)
    shaped = apply_transient(data, sr, attack_db, sustain_db, fast_ms, slow_ms)

    peak_in = float(20 * np.log10(np.max(np.abs(data)) + 1e-10))
    peak_out = float(20 * np.log10(np.max(np.abs(shaped)) + 1e-10))

    out_name = f"{input_path.stem}_transient.wav"
    out_path = output_dir / out_name
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), shaped, sr, subtype="FLOAT")

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
    parser.add_argument("--fast-ms", type=float, default=None, help="Fast RMS envelope window in ms (default: preset or 2)")
    parser.add_argument("--slow-ms", type=float, default=None, help="Slow envelope time constant in ms (default: preset or 50)")
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

    if args.preset:
        try:
            _load_preset(args.preset)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    report = apply_transient_file(args.file, args.output_dir, args.attack,
                                  args.sustain, args.fast_ms, args.slow_ms,
                                  preset=args.preset)
    if report["attack_db"] == 0.0 and report["sustain_db"] == 0.0:
        print("Warning: both attack and sustain are 0 dB — no shaping applied", file=sys.stderr)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
