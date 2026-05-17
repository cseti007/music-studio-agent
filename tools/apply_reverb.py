"""Apply reverb to an audio stem.

Two engines:
  Algorithmic (Freeverb via pedalboard) — default, parameter-driven (room_size,
                                          damping, etc.). Fast, characterful
                                          for small/medium spaces and plates.
  Convolution (--ir IR.wav)              — convolves the stem with an impulse
                                          response. Authentic for halls,
                                          rooms, hardware emulations.

Both engines share the same pre-delay, HP/LP-on-return, gate, and wet/dry
plumbing. With --ir, the algorithmic parameters (room_size, damping, width)
are ignored; the IR defines the tail.

Two output modes:
  Insert (default): dry + wet signal mixed to output. Use for inline per-track reverb.
  Send (--send):    wet only output. Use to generate a reverb return bus that
                    render_mix.py can mix alongside the dry track.

Pre-delay keeps the direct transient clean before the reverb tail kicks in — essential
for punch on snare and drums.

Usage:
  # snare plate (algorithmic)
  python apply_reverb.py output/terido/SN\\ TOP/assembled_eq_comp.wav \\
      --preset snare_plate --output-dir output/terido/SN\\ TOP

  # large hall via impulse response (convolution)
  python apply_reverb.py input.wav --ir irs/concert_hall.wav \\
      --pre-delay 40 --hp 200 --lp 10000 --wet 0.15 --send --output-dir DIR
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pedalboard
import soundfile as sf
from scipy.signal import butter, sosfilt


PRESETS: dict[str, dict] = {
    "snare_plate": {
        "description": "Plate reverb for snare — pre-delay preserves punch, adds size",
        "notes": "15ms pre-delay keeps snare crack tight before reverb kicks in. HP at 500Hz removes low-mid mud from the reverb return — critical for punch. LP at 6kHz smooths harshness. Decay ~0.8s (room_size=0.5) works for modern rock; go to 0.65 for more size.",
        "room_size": 0.50,
        "damping": 0.30,
        "width": 1.0,
        "pre_delay_ms": 15.0,
        "wet": 0.22,
        "dry": 1.0,
        "hp_hz": 500,
        "lp_hz": 6000,
    },
    "snare_plate_big": {
        "description": "Bigger plate for rock snare — more size, classic sound",
        "notes": "Longer decay (room_size=0.65 ≈ 1.2s), 20ms pre-delay. Use when snare needs to fill the room more. Still tight enough for 120-140 BPM rock.",
        "room_size": 0.65,
        "damping": 0.25,
        "width": 1.0,
        "pre_delay_ms": 20.0,
        "wet": 0.25,
        "dry": 1.0,
        "hp_hz": 450,
        "lp_hz": 6000,
    },
    "room_drums": {
        "description": "Short room ambience for drums — glues kit in an acoustic space",
        "notes": "Small room, high damping. Adds cohesion between kick/snare/OH without audible reverb tail. HP at 300Hz critical — low-end in reverb kills punch.",
        "room_size": 0.28,
        "damping": 0.60,
        "width": 0.80,
        "pre_delay_ms": 8.0,
        "wet": 0.12,
        "dry": 1.0,
        "hp_hz": 300,
        "lp_hz": 6000,
    },
    "guitar_room": {
        "description": "Small room for electric guitar — depth without wash",
        "notes": "Medium room with pre-delay to keep the pick attack tight. Subtle wet level — guitars in a dense rock mix need space, not wash.",
        "room_size": 0.45,
        "damping": 0.50,
        "width": 0.85,
        "pre_delay_ms": 20.0,
        "wet": 0.18,
        "dry": 1.0,
        "hp_hz": 150,
        "lp_hz": 8000,
    },
    "snare_gated": {
        "description": "Gated plate reverb — classic 80s/rock snare, maximum punch",
        "notes": "Long decay (room_size=0.7 ≈ 1.8s) cut by a gate: hold=300ms, release=70ms. Result: snare has presence and size but reverb tail vanishes before the next hit. Use when snare sounds thin but straight plate washes too much.",
        "room_size": 0.70,
        "damping": 0.20,
        "width": 1.0,
        "pre_delay_ms": 15.0,
        "wet": 0.30,
        "dry": 1.0,
        "hp_hz": 400,
        "lp_hz": 6000,
        "gate_hold_ms": 300.0,
        "gate_release_ms": 70.0,
    },
    "hall_ambient": {
        "description": "Large hall — subtle depth, best as send on mix bus or vocals",
        "notes": "Long pre-delay (50ms) keeps sources clear. Low wet level (10%) — use sparingly in rock or it washes out the attack.",
        "room_size": 0.75,
        "damping": 0.40,
        "width": 1.0,
        "pre_delay_ms": 50.0,
        "wet": 0.10,
        "dry": 1.0,
        "hp_hz": 400,
        "lp_hz": 10000,
    },
}


def _gate(data: np.ndarray, sr: int, hold_ms: float, release_ms: float, threshold: float = 0.05) -> np.ndarray:
    """Simple noise gate for gated reverb effect.

    Opens when RMS exceeds threshold, stays open for hold_ms, then fades over release_ms.
    Applied per-channel on the reverb tail (wet signal only).
    """
    hold_samples = int(hold_ms * sr / 1000)
    release_samples = max(1, int(release_ms * sr / 1000))
    window = int(0.005 * sr)  # 5ms RMS window

    mono = np.abs(data).mean(axis=1) if data.ndim == 2 else np.abs(data)
    envelope = np.zeros(len(mono))
    for i in range(0, len(mono), window):
        envelope[i:i+window] = np.sqrt(np.mean(mono[i:i+window]**2) + 1e-10)

    gate = np.zeros(len(mono))
    open_until = 0
    for i in range(len(mono)):
        if envelope[i] > threshold:
            open_until = i + hold_samples
        if i < open_until:
            gate[i] = 1.0
        else:
            dist = i - open_until
            if dist < release_samples:
                gate[i] = 1.0 - dist / release_samples

    if data.ndim == 2:
        return data * gate[:, np.newaxis]
    return data * gate


def _highpass(data: np.ndarray, sr: int, hz: float) -> np.ndarray:
    sos = butter(2, hz / (sr / 2), btype="high", output="sos")
    return sosfilt(sos, data, axis=0)


def _lowpass(data: np.ndarray, sr: int, hz: float) -> np.ndarray:
    sos = butter(2, hz / (sr / 2), btype="low", output="sos")
    return sosfilt(sos, data, axis=0)


def apply_reverb(
    file_path: Path,
    output_dir: Path,
    room_size: float = 0.5,
    damping: float = 0.5,
    width: float = 1.0,
    pre_delay_ms: float = 0.0,
    wet: float = 0.2,
    dry: float = 1.0,
    hp_hz: float | None = None,
    lp_hz: float | None = None,
    gate_hold_ms: float | None = None,
    gate_release_ms: float | None = None,
    send_mode: bool = False,
    preset_name: str | None = None,
    ir_path: Path | None = None,
) -> dict:
    data, sr = sf.read(str(file_path), always_2d=True)

    # Convert mono to stereo for width processing
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
        was_mono = True
    else:
        was_mono = False

    pre_delay_samples = int(pre_delay_ms * sr / 1000.0)

    # Apply pre-delay: shift signal forward, pad with silence at start
    if pre_delay_samples > 0:
        delay_buf = np.zeros((pre_delay_samples, data.shape[1]), dtype=data.dtype)
        delayed = np.concatenate([delay_buf, data], axis=0)
    else:
        delayed = data.copy()

    # Engine: convolution if an IR is supplied, else algorithmic Freeverb.
    if ir_path is not None:
        board = pedalboard.Pedalboard([
            pedalboard.Convolution(impulse_response_filename=str(ir_path), mix=1.0)
        ])
    else:
        board = pedalboard.Pedalboard([
            pedalboard.Reverb(
                room_size=room_size,
                damping=damping,
                wet_level=1.0,
                dry_level=0.0,
                width=width,
            )
        ])
    reverb_out = board(delayed.T.astype(np.float32), sr).T.astype(np.float64)

    # Trim pre-delay padding from reverb output so it aligns with dry signal
    if pre_delay_samples > 0:
        reverb_out = reverb_out[pre_delay_samples:]
    # Match length to dry signal
    reverb_out = reverb_out[:len(data)]
    if len(reverb_out) < len(data):
        pad = np.zeros((len(data) - len(reverb_out), 2), dtype=np.float64)
        reverb_out = np.concatenate([reverb_out, pad], axis=0)

    # Apply HP/LP to reverb return (remove low mud and harshness)
    if hp_hz is not None and hp_hz > 0:
        reverb_out = _highpass(reverb_out, sr, hp_hz)
    if lp_hz is not None and lp_hz > 0:
        reverb_out = _lowpass(reverb_out, sr, lp_hz)

    # Apply gate to reverb tail (gated reverb effect)
    if gate_hold_ms is not None and gate_release_ms is not None:
        reverb_out = _gate(reverb_out, sr,
                           hold_ms=gate_hold_ms, release_ms=gate_release_ms)

    # Mix
    if send_mode:
        output = reverb_out * wet
    else:
        output = data * dry + reverb_out * wet

    # Back to mono if input was mono
    if was_mono:
        output = output.mean(axis=1, keepdims=True)

    # Prevent clipping
    peak = float(np.max(np.abs(output)))
    if peak > 1.0:
        print(
            f"WARNING: clipping by {20*np.log10(peak):.1f} dB — reducing gain",
            file=sys.stderr,
        )
        output = output / peak

    output_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = "_send" if send_mode else ""
    out_path = output_dir / (file_path.stem + f"_reverb{mode_tag}.wav")
    sf.write(str(out_path), output, sr, subtype="PCM_24")

    result = {
        "input": str(file_path),
        "output": str(out_path),
        "preset": preset_name,
        "engine": "convolution" if ir_path is not None else "freeverb",
        "ir": str(ir_path) if ir_path is not None else None,
        "mode": "send" if send_mode else "insert",
        "room_size": None if ir_path is not None else room_size,
        "damping": None if ir_path is not None else damping,
        "width": None if ir_path is not None else width,
        "pre_delay_ms": pre_delay_ms,
        "wet": wet,
        "dry": dry if not send_mode else 0.0,
        "hp_hz": hp_hz,
        "lp_hz": lp_hz,
        "gate_hold_ms": gate_hold_ms,
        "gate_release_ms": gate_release_ms,
        "sample_rate": sr,
    }

    report_path = output_dir / "reverb_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply reverb to a stem. Insert mode (dry+wet) or send mode (wet only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available presets: " + ", ".join(PRESETS),
    )
    parser.add_argument("input", type=Path, nargs="?", help="Input WAV file")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--preset", choices=list(PRESETS), help="Reverb preset")
    parser.add_argument("--send", action="store_true", help="Send mode: output wet only (no dry signal)")
    parser.add_argument("--pre-delay", type=float, metavar="MS", help="Pre-delay in milliseconds")
    parser.add_argument("--room-size", type=float, metavar="0-1", help="Room size (0=small, 1=large)")
    parser.add_argument("--damping", type=float, metavar="0-1", help="High-frequency damping (0=bright, 1=dark)")
    parser.add_argument("--width", type=float, metavar="0-1", help="Stereo width of reverb return")
    parser.add_argument("--wet", type=float, metavar="0-1", help="Reverb wet level")
    parser.add_argument("--dry", type=float, metavar="0-1", help="Dry signal level (insert mode only)")
    parser.add_argument("--hp", type=float, metavar="HZ", help="High-pass filter on reverb return")
    parser.add_argument("--lp", type=float, metavar="HZ", help="Low-pass filter on reverb return")
    parser.add_argument("--gate-hold", type=float, metavar="MS", help="Gate hold time in ms (enables gated reverb)")
    parser.add_argument("--gate-release", type=float, metavar="MS", help="Gate release time in ms (default 70ms)")
    parser.add_argument(
        "--ir", type=Path, metavar="WAV",
        help="Impulse response WAV — switches engine to convolution. "
             "Algorithmic params (room_size, damping, width) are ignored when set.",
    )
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")

    args = parser.parse_args()

    if args.list_presets:
        for name, p in PRESETS.items():
            print(f"\n{name}: {p['description']}")
            print(f"  room={p['room_size']}  damping={p['damping']}  width={p['width']}")
            print(f"  pre_delay={p['pre_delay_ms']}ms  wet={p['wet']}  hp={p['hp_hz']}Hz  lp={p['lp_hz']}Hz")
            print(f"  {p['notes']}")
        return

    if args.input is None:
        parser.error("input file is required")
    if not args.input.exists():
        print(json.dumps({"error": f"Not found: {args.input}"}), file=sys.stderr)
        sys.exit(1)
    if args.ir is not None and not args.ir.exists():
        print(json.dumps({"error": f"IR file not found: {args.ir}"}), file=sys.stderr)
        sys.exit(1)

    # Start from preset, then apply CLI overrides
    params: dict = {}
    preset_name = None
    if args.preset:
        preset_name = args.preset
        p = PRESETS[args.preset]
        params = {
            "room_size": p["room_size"],
            "damping": p["damping"],
            "width": p["width"],
            "pre_delay_ms": p["pre_delay_ms"],
            "wet": p["wet"],
            "dry": p["dry"],
            "hp_hz": p["hp_hz"],
            "lp_hz": p["lp_hz"],
            "gate_hold_ms": p.get("gate_hold_ms"),
            "gate_release_ms": p.get("gate_release_ms"),
        }
    else:
        params = {"room_size": 0.5, "damping": 0.5, "width": 1.0,
                  "pre_delay_ms": 0.0, "wet": 0.2, "dry": 1.0,
                  "hp_hz": None, "lp_hz": None,
                  "gate_hold_ms": None, "gate_release_ms": None}

    if args.pre_delay is not None:   params["pre_delay_ms"] = args.pre_delay
    if args.room_size is not None:   params["room_size"] = args.room_size
    if args.damping is not None:     params["damping"] = args.damping
    if args.width is not None:       params["width"] = args.width
    if args.wet is not None:         params["wet"] = args.wet
    if args.dry is not None:         params["dry"] = args.dry
    if args.hp is not None:          params["hp_hz"] = args.hp
    if args.lp is not None:          params["lp_hz"] = args.lp
    if args.gate_hold is not None:
        params["gate_hold_ms"] = args.gate_hold
        params["gate_release_ms"] = args.gate_release if args.gate_release else 70.0

    result = apply_reverb(
        file_path=args.input,
        output_dir=args.output_dir,
        preset_name=preset_name,
        send_mode=args.send,
        ir_path=args.ir,
        **params,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
