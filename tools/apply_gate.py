"""Apply a noise gate to an assembled stem.

Gate logic: envelope follower -> state machine (CLOSED/ATTACK/OPEN/HOLD/RELEASE) ->
gain envelope applied to audio. Processed in 1ms blocks for speed.

Parameters:
  threshold    dB level where gate opens (e.g. -30)
  range        attenuation in dB when closed (e.g. -80; use -40 for more natural sound)
  attack       time in ms for gate to fully open after trigger (e.g. 1-5)
  hold         time in ms gate stays open after signal drops below close threshold (e.g. 80)
  release      time in ms for gate to fully close after hold expires (e.g. 50)
  hysteresis   dB below threshold where gate starts its hold timer (prevents chattering).
               close_threshold = threshold - hysteresis. Default 6 dB.
  rms_window   RMS envelope follower window in ms (default 5)

State machine:
  CLOSED  -> (env >= open_thr)  -> ATTACK
  ATTACK  -> (ramp complete)    -> OPEN
  OPEN    -> (env < close_thr)  -> HOLD
  HOLD    -> (env >= open_thr)  -> OPEN
  HOLD    -> (env >= close_thr) -> reset hold timer (hysteresis: stay open)
  HOLD    -> (hold expires)     -> RELEASE
  RELEASE -> (env >= open_thr)  -> ATTACK
  RELEASE -> (ramp complete)    -> CLOSED

Usage:
  python apply_gate.py assembled.wav --output-dir output/session/TOM \\
      --threshold -30 --hold 150 --release 100

  python apply_gate.py assembled.wav --output-dir output/session/SN_TOP \\
      --preset gate_snare_top

  python apply_gate.py *.wav --list-presets
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d

_PRESETS_DIR = Path(__file__).parent / "presets"

# State constants
_CLOSED = 0
_ATTACK = 1
_OPEN = 2
_HOLD = 3
_RELEASE = 4


# ---------------------------------------------------------------------------
# Preset loader
# ---------------------------------------------------------------------------

def _load_preset(name: str) -> dict:
    path = _PRESETS_DIR / f"{name}.json"
    if not path.exists():
        available = [p.stem for p in sorted(_PRESETS_DIR.glob("gate_*.json"))]
        raise FileNotFoundError(
            f"Preset {name!r} not found. Available gate presets: {', '.join(available)}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def list_presets() -> None:
    if not _PRESETS_DIR.exists():
        print("No presets directory found.")
        return
    presets = sorted(_PRESETS_DIR.glob("gate_*.json"))
    if not presets:
        print("No gate presets found.")
        return
    for path in presets:
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data.get("settings", {})
        desc = data.get("description", "")
        print(
            f"  {path.stem:<28}  {desc}  "
            f"[thr={s.get('threshold_db','?')}dB  "
            f"hold={s.get('hold_ms','?')}ms  "
            f"rel={s.get('release_ms','?')}ms  "
            f"range={s.get('range_db','?')}dB]"
        )


# ---------------------------------------------------------------------------
# Core: gain envelope computation
# ---------------------------------------------------------------------------

def _gate_gain_envelope(
    mono: np.ndarray,
    sr: int,
    threshold_db: float,
    range_db: float,
    attack_ms: float,
    hold_ms: float,
    release_ms: float,
    hysteresis_db: float,
    rms_window_ms: float,
) -> np.ndarray:
    n = len(mono)
    block_ms = 1.0
    block_size = max(1, int(block_ms * sr / 1000))
    n_blocks = (n + block_size - 1) // block_size

    open_lin = 10.0 ** (threshold_db / 20.0)
    close_lin = 10.0 ** ((threshold_db - hysteresis_db) / 20.0)
    range_lin = 10.0 ** (range_db / 20.0)

    attack_blocks = max(1, int(attack_ms / block_ms))
    hold_blocks = max(1, int(hold_ms / block_ms))
    release_blocks = max(1, int(release_ms / block_ms))

    # Per-block RMS envelope
    rms_win_samples = max(1, int(rms_window_ms * sr / 1000))
    mono_sq = mono ** 2
    smooth_sq = uniform_filter1d(mono_sq, size=rms_win_samples, mode="constant")
    env_samples = np.sqrt(np.maximum(smooth_sq, 0.0))

    env_blocks = np.zeros(n_blocks)
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        env_blocks[b] = float(np.max(env_samples[start:end]))

    # State machine over 1ms blocks
    gain_blocks = np.zeros(n_blocks)
    state = _CLOSED
    counter = 0
    g = range_lin

    for b in range(n_blocks):
        e = env_blocks[b]

        if state == _CLOSED:
            g = range_lin
            if e >= open_lin:
                state = _ATTACK
                counter = 0

        if state == _ATTACK:
            counter += 1
            t = min(counter / attack_blocks, 1.0)
            g = range_lin + (1.0 - range_lin) * t
            if counter >= attack_blocks:
                g = 1.0
                state = _OPEN

        elif state == _OPEN:
            g = 1.0
            if e < close_lin:
                state = _HOLD
                counter = 0

        elif state == _HOLD:
            g = 1.0
            if e >= open_lin:
                state = _OPEN
                counter = 0
            elif e >= close_lin:
                counter = 0  # hysteresis: reset hold while above close threshold
            else:
                counter += 1
                if counter >= hold_blocks:
                    state = _RELEASE
                    counter = 0

        elif state == _RELEASE:
            counter += 1
            t = min(counter / release_blocks, 1.0)
            g = 1.0 - (1.0 - range_lin) * t
            if e >= open_lin:
                state = _ATTACK
                counter = 0
            elif counter >= release_blocks:
                g = range_lin
                state = _CLOSED

        gain_blocks[b] = g

    # Interpolate from block resolution to sample resolution
    block_centers = (np.arange(n_blocks) + 0.5) * block_size
    gain = np.interp(
        np.arange(n),
        block_centers,
        gain_blocks,
        left=float(gain_blocks[0]),
        right=float(gain_blocks[-1]),
    )
    return gain


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def apply_gate(
    input_path: Path,
    output_dir: Path,
    threshold_db: float,
    range_db: float,
    attack_ms: float,
    hold_ms: float,
    release_ms: float,
    hysteresis_db: float = 6.0,
    rms_window_ms: float = 5.0,
    preset_name: str | None = None,
) -> dict:
    data, sr = sf.read(str(input_path), always_2d=True)
    n_channels = data.shape[1]

    mono = data.mean(axis=1)
    gain = _gate_gain_envelope(
        mono, sr,
        threshold_db=threshold_db,
        range_db=range_db,
        attack_ms=attack_ms,
        hold_ms=hold_ms,
        release_ms=release_ms,
        hysteresis_db=hysteresis_db,
        rms_window_ms=rms_window_ms,
    )

    # Apply gain to all channels
    output_data = data * gain[:, np.newaxis]

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
    out_path = output_dir / (input_path.stem + "_gate.wav")
    sf.write(str(out_path), output_data, sr, subtype="PCM_24")

    # Report statistics
    open_fraction = float(np.mean(gain > (10.0 ** (range_db / 20.0) * 2)))
    n_openings = int(np.sum(np.diff((gain > 0.5).astype(int)) > 0))

    in_peak_db = float(20.0 * np.log10(max(float(np.max(np.abs(data))), 1e-10)))
    out_peak_db = float(20.0 * np.log10(max(float(np.max(np.abs(output_data))), 1e-10)))

    report = {
        "input": str(input_path),
        "output": str(out_path),
        "preset": preset_name,
        "settings": {
            "threshold_db": threshold_db,
            "range_db": range_db,
            "attack_ms": attack_ms,
            "hold_ms": hold_ms,
            "release_ms": release_ms,
            "hysteresis_db": hysteresis_db,
            "rms_window_ms": rms_window_ms,
        },
        "gate_open_pct": round(open_fraction * 100, 1),
        "gate_openings_count": n_openings,
        "input_peak_dbfs": round(in_peak_db, 1),
        "output_peak_dbfs": round(out_peak_db, 1),
        "sample_rate": sr,
    }
    (output_dir / "gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a noise gate to a stem.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", type=Path, nargs="?", help="Input WAV file")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument(
        "--threshold", type=float, metavar="DB",
        help="Level in dBFS where gate opens (e.g. -30)",
    )
    parser.add_argument(
        "--range", type=float, metavar="DB", dest="range_db",
        help="Attenuation in dB when closed (e.g. -80; -40 for more natural sound)",
    )
    parser.add_argument(
        "--attack", type=float, metavar="MS",
        help="Time in ms for gate to fully open (e.g. 1)",
    )
    parser.add_argument(
        "--hold", type=float, metavar="MS",
        help="Time in ms gate stays open after signal drops (e.g. 100)",
    )
    parser.add_argument(
        "--release", type=float, metavar="MS",
        help="Time in ms for gate to fully close (e.g. 60)",
    )
    parser.add_argument(
        "--hysteresis", type=float, metavar="DB", default=None,
        help="dB below threshold where close timer starts (default 6)",
    )
    parser.add_argument(
        "--rms-window", type=float, metavar="MS", default=None,
        help="Envelope follower RMS window in ms (default 5)",
    )
    parser.add_argument("--preset", metavar="NAME", help="Gate preset (see --list-presets)")
    parser.add_argument("--list-presets", action="store_true", help="List gate presets and exit")
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
    range_db  = args.range_db
    attack    = args.attack
    hold      = args.hold
    release   = args.release
    hysteresis = args.hysteresis
    rms_window = args.rms_window
    preset_name = args.preset

    if args.preset:
        try:
            pdata = _load_preset(args.preset)
        except FileNotFoundError as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(1)
        s = pdata.get("settings", {})
        # CLI args override preset values
        if threshold  is None: threshold  = s["threshold_db"]
        if range_db   is None: range_db   = s["range_db"]
        if attack     is None: attack     = s["attack_ms"]
        if hold       is None: hold       = s["hold_ms"]
        if release    is None: release    = s["release_ms"]
        if hysteresis is None: hysteresis = s.get("hysteresis_db", 6.0)
        if rms_window is None: rms_window = s.get("rms_window_ms", 5.0)
        print(
            f"Loaded preset {args.preset!r}: "
            f"thr={threshold}dB  range={range_db}dB  "
            f"atk={attack}ms  hold={hold}ms  rel={release}ms",
            file=sys.stderr,
        )

    for name, val in [("--threshold", threshold), ("--range", range_db),
                      ("--attack", attack), ("--hold", hold), ("--release", release)]:
        if val is None:
            parser.error(f"{name} is required (or use --preset)")

    if hysteresis is None:
        hysteresis = 6.0
    if rms_window is None:
        rms_window = 5.0

    apply_gate(
        args.file, args.output_dir,
        threshold_db=threshold,
        range_db=range_db,
        attack_ms=attack,
        hold_ms=hold,
        release_ms=release,
        hysteresis_db=hysteresis,
        rms_window_ms=rms_window,
        preset_name=preset_name,
    )


if __name__ == "__main__":
    main()
