"""Apply delay (echo) effect to a stem.

Three modes:
  normal     Single delay line — slapback, dotted-eighth echo, ambient pre-delay.
  pingpong   Stereo ping-pong: odd echoes on L, even echoes on R. Mono input
             is automatically expanded to stereo.

Echo series is computed as an FIR approximation:
  tap 1: gain = mix,                   at delay_ms
  tap 2: gain = mix * feedback,        at 2 * delay_ms
  tap n: gain = mix * feedback^(n-1),  at n * delay_ms
  (continued until tap gain < -80 dB below the first echo)

This is mathematically equivalent to a recursive delay line but runs fully vectorized.

BPM-synced delay: use --bpm and --division instead of --delay-ms.
  divisions: whole, half, quarter, dotted-quarter, eighth, dotted-eighth, sixteenth

Send mode (--send): outputs wet-only signal for use as a bus return. Dry signal unchanged.
  Use this when routing through a send/return bus rather than inserting on the channel.

Usage:
  python apply_delay.py snare.wav --output-dir DIR \\
      --delay-ms 80 --feedback 0 --mix 0.25 --lp 8000

  python apply_delay.py guitar.wav --output-dir DIR \\
      --preset delay_slapback_guitar

  python apply_delay.py snare.wav --output-dir DIR \\
      --bpm 120 --division dotted-eighth --feedback 0.4 --send

  python apply_delay.py snare.wav --output-dir DIR \\
      --mode pingpong --delay-ms 200 --feedback 0.5 --send

  python apply_delay.py x.wav --list-presets
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

_PRESETS_DIR = Path(__file__).parent / "presets"

_DIVISIONS = {
    "whole":          4.0,
    "half":           2.0,
    "quarter":        1.0,
    "dotted-quarter": 1.5,
    "eighth":         0.5,
    "dotted-eighth":  0.75,
    "sixteenth":      0.25,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bpm_delay_ms(bpm: float, division: str) -> float:
    if division not in _DIVISIONS:
        raise ValueError(f"Unknown division {division!r}. Choose from: {', '.join(_DIVISIONS)}")
    beat_ms = 60_000.0 / bpm
    return beat_ms * _DIVISIONS[division]


def _apply_filters(audio: np.ndarray, sr: int, hp_hz: float | None, lp_hz: float | None) -> np.ndarray:
    out = audio.copy()
    if hp_hz and hp_hz > 0:
        sos = butter(2, hp_hz, btype="highpass", fs=sr, output="sos")
        out = sosfilt(sos, out, axis=0)
    if lp_hz and lp_hz > 0:
        sos = butter(2, lp_hz, btype="lowpass", fs=sr, output="sos")
        out = sosfilt(sos, out, axis=0)
    return out


def _echo_series(
    data: np.ndarray,
    delay_samples: int,
    feedback: float,
    initial_gain: float,
    n_samples: int,
    max_taps: int = 200,
) -> np.ndarray:
    """Sum of delay taps using FIR approximation (vectorized).

    Returns wet signal array (same shape as data).
    """
    wet = np.zeros_like(data)
    tap_gain = initial_gain
    # -80 dB relative to initial_gain is the threshold for stopping
    gain_floor = initial_gain * 1e-4

    for tap in range(1, max_taps + 1):
        offset = delay_samples * tap
        if offset >= n_samples or tap_gain < gain_floor:
            break
        wet[offset:] += data[: n_samples - offset] * tap_gain
        tap_gain *= feedback

    return wet


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def apply_delay(
    input_path: Path,
    output_dir: Path,
    delay_ms: float,
    feedback: float = 0.0,
    mix: float = 0.25,
    mode: str = "normal",
    hp_hz: float | None = None,
    lp_hz: float | None = None,
    send: bool = False,
    preset_name: str | None = None,
    bpm: float | None = None,
    division: str | None = None,
) -> dict:
    if mode not in ("normal", "pingpong"):
        raise ValueError(f"Unknown mode {mode!r}. Choose 'normal' or 'pingpong'.")

    data, sr = sf.read(str(input_path), always_2d=True)
    n_samples, n_channels = data.shape

    delay_samples = max(1, int(round(delay_ms * sr / 1000.0)))

    if mode == "pingpong":
        # Expand mono to stereo for ping-pong
        if n_channels == 1:
            data = np.hstack([data, data])
            n_channels = 2

        # Odd taps → L channel, even taps → R channel
        wet_L = np.zeros(n_samples)
        wet_R = np.zeros(n_samples)
        tap_gain = mix
        gain_floor = mix * 1e-4

        for tap in range(1, 201):
            offset = delay_samples * tap
            if offset >= n_samples or tap_gain < gain_floor:
                break
            mono_in = data.mean(axis=1)
            if tap % 2 == 1:
                wet_L[offset:] += mono_in[: n_samples - offset] * tap_gain
            else:
                wet_R[offset:] += mono_in[: n_samples - offset] * tap_gain
            tap_gain *= max(feedback, 0.0)

        wet = np.stack([wet_L, wet_R], axis=1)

    else:
        # Normal: same delay on all channels
        wet = _echo_series(data, delay_samples, max(feedback, 0.0), mix, n_samples)

    # Filters on wet signal
    wet = _apply_filters(wet, sr, hp_hz, lp_hz)

    if send:
        output_data = wet
    else:
        output_data = data + wet

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
    suffix = "_delay_send" if send else "_delay"
    out_path = output_dir / (input_path.stem + suffix + ".wav")
    sf.write(str(out_path), output_data, sr, subtype="PCM_24")

    in_peak = float(20 * np.log10(max(float(np.max(np.abs(data))), 1e-10)))
    out_peak = float(20 * np.log10(max(float(np.max(np.abs(output_data))), 1e-10)))

    report = {
        "input": str(input_path),
        "output": str(out_path),
        "preset": preset_name,
        "settings": {
            "mode": mode,
            "delay_ms": round(delay_ms, 2),
            "feedback": feedback,
            "mix": mix,
            "hp_hz": hp_hz,
            "lp_hz": lp_hz,
            "send": send,
            "bpm": bpm,
            "division": division,
        },
        "input_peak_dbfs": round(in_peak, 1),
        "output_peak_dbfs": round(out_peak, 1),
        "clipping_prevented": clipped,
        "sample_rate": sr,
    }
    (output_dir / "delay_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# Preset loader
# ---------------------------------------------------------------------------

def _load_preset(name: str) -> dict:
    path = _PRESETS_DIR / f"{name}.json"
    if not path.exists():
        available = [p.stem for p in sorted(_PRESETS_DIR.glob("delay_*.json"))]
        raise FileNotFoundError(
            f"Preset {name!r} not found. Available delay presets: {', '.join(available)}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def list_presets() -> None:
    if not _PRESETS_DIR.exists():
        print("No presets directory found.")
        return
    for path in sorted(_PRESETS_DIR.glob("delay_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data.get("settings", {})
        desc = data.get("description", "")
        delay_info = (
            f"{s['bpm']}bpm/{s['division']}"
            if s.get("bpm") else f"{s.get('delay_ms', '?')}ms"
        )
        print(
            f"  {path.stem:<32}  {desc}"
            f"  [{delay_info}  fb={s.get('feedback','?')}  mix={s.get('mix','?')}]"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply delay/echo to a stem.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", type=Path, nargs="?", help="Input WAV file")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument(
        "--mode", choices=["normal", "pingpong"], default="normal",
        help="Delay mode (default: normal)",
    )
    parser.add_argument("--delay-ms", type=float, metavar="MS", help="Delay time in milliseconds")
    parser.add_argument(
        "--feedback", type=float, default=0.0, metavar="0-0.95",
        help="Feedback amount 0.0-0.95 (default 0.0 = single echo)",
    )
    parser.add_argument(
        "--mix", type=float, default=0.25, metavar="0-1",
        help="Wet level relative to dry (default 0.25)",
    )
    parser.add_argument("--hp", type=float, metavar="HZ", help="High-pass wet signal at this Hz")
    parser.add_argument("--lp", type=float, metavar="HZ", help="Low-pass wet signal at this Hz (darkens repeats)")
    parser.add_argument("--send", action="store_true", help="Send mode: output wet-only signal")
    parser.add_argument("--preset", metavar="NAME", help="Delay preset (see --list-presets)")
    parser.add_argument("--list-presets", action="store_true", help="List delay presets and exit")

    bpm_group = parser.add_argument_group("BPM sync")
    bpm_group.add_argument("--bpm", type=float, metavar="BPM", help="Song tempo (overrides --delay-ms when combined with --division)")
    bpm_group.add_argument(
        "--division",
        choices=list(_DIVISIONS),
        metavar="DIVISION",
        help=f"Note division: {', '.join(_DIVISIONS)} (requires --bpm)",
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

    delay_ms = args.delay_ms
    feedback = args.feedback
    mix = args.mix
    mode = args.mode
    hp_hz = args.hp
    lp_hz = args.lp
    send = args.send
    bpm = args.bpm
    division = args.division
    preset_name = args.preset

    if args.preset:
        try:
            pdata = _load_preset(args.preset)
        except FileNotFoundError as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(1)
        s = pdata.get("settings", {})
        delay_ms = s.get("delay_ms", delay_ms)
        feedback = s.get("feedback", feedback)
        mix = s.get("mix", mix)
        mode = s.get("mode", mode)
        hp_hz = s.get("hp_hz", hp_hz)
        lp_hz = s.get("lp_hz", lp_hz)
        send = s.get("send", send)
        bpm = s.get("bpm", bpm)
        division = s.get("division", division)

    # BPM sync overrides delay_ms
    if bpm and division:
        delay_ms = _bpm_delay_ms(bpm, division)
        print(f"BPM sync: {bpm} bpm / {division} = {delay_ms:.1f} ms", file=sys.stderr)
    elif bpm or division:
        parser.error("--bpm and --division must be used together")

    if delay_ms is None:
        parser.error("--delay-ms is required (or use --preset / --bpm + --division)")

    if not (0.0 <= feedback < 1.0):
        parser.error("--feedback must be in range 0.0-0.95")

    apply_delay(
        args.file, args.output_dir,
        delay_ms=delay_ms,
        feedback=feedback,
        mix=mix,
        mode=mode,
        hp_hz=hp_hz,
        lp_hz=lp_hz,
        send=send,
        preset_name=preset_name,
        bpm=bpm,
        division=division,
    )


if __name__ == "__main__":
    main()
