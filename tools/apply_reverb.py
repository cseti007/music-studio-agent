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
  python apply_reverb.py output/<session>/SN\\ TOP/assembled_eq_comp.wav \\
      --preset snare_plate --output-dir output/<session>/SN\\ TOP

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


_DIVISION_FACTORS: dict[str, float] = {
    "whole":         1.0,
    "half":          0.5,
    "dotted-quarter": 0.375,
    "quarter":       0.25,
    "dotted-eighth": 0.1875,
    "triplet-quarter": 1.0 / 6.0,
    "eighth":        0.125,
    "dotted-sixteenth": 0.09375,
    "triplet-eighth": 1.0 / 12.0,
    "sixteenth":     0.0625,
    "triplet-sixteenth": 1.0 / 24.0,
    "thirty-second": 0.03125,
}


def _bpm_to_pre_delay_ms(bpm: float, division: str) -> float:
    """Convert a tempo + note-division into a pre-delay in milliseconds.

    A whole note at BPM is `4 * 60000 / BPM` ms (because BPM counts
    quarters per minute). Each named division is a fraction of that.
    Example: 120 BPM eighth = 4 * 500 * 0.125 = 250 ms; 184 BPM
    sixteenth = 4 * 326.087 * 0.0625 = 81.5 ms.
    """
    if division not in _DIVISION_FACTORS:
        raise ValueError(
            f"Unknown division {division!r}. Known: {sorted(_DIVISION_FACTORS)}"
        )
    whole_note_ms = 4.0 * 60000.0 / float(bpm)
    return whole_note_ms * _DIVISION_FACTORS[division]


def _sidechain_envelope(sc_mono: np.ndarray, sr: int,
                        attack_ms: float = 5.0, release_ms: float = 150.0,
                        threshold_db: float = -30.0, depth_db: float = -12.0) -> np.ndarray:
    """Return a per-sample 0..1 multiplier driven by sc_mono.

    Used to duck the reverb tail every time the sidechain (typically the
    kick) hits — the "pumping reverb" pattern. depth_db is how far the
    reverb is pulled down at peak ducking (e.g. -12 dB → multiplier 0.25).
    """
    block_samples = max(1, sr // 1000)  # 1 ms blocks
    n_blocks = (len(sc_mono) + block_samples - 1) // block_samples
    alpha_a = np.exp(-1.0 / max(1.0, attack_ms * 0.001 * sr / block_samples))
    alpha_r = np.exp(-1.0 / max(1.0, release_ms * 0.001 * sr / block_samples))
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    depth_lin = 10.0 ** (depth_db / 20.0)

    env = 0.0
    gain_blocks = np.ones(n_blocks)
    for i in range(n_blocks):
        start = i * block_samples
        end = min(start + block_samples, len(sc_mono))
        level = float(np.max(np.abs(sc_mono[start:end])))
        # Ballistic envelope follower
        if level > env:
            env = alpha_a * env + (1.0 - alpha_a) * level
        else:
            env = alpha_r * env + (1.0 - alpha_r) * level
        if env > threshold_lin:
            # Map (env_db - threshold) onto a 0..1 ducking amount, capped
            over_db = 20.0 * np.log10(max(env, 1e-10)) - threshold_db
            # Light static: 1:4 ratio → 0.25 dB ducking per dB over threshold
            ducked = max(depth_lin, 10.0 ** ((-over_db * 0.25) / 20.0))
            gain_blocks[i] = ducked
        # else: leave at 1.0 (no ducking)

    # Interpolate to sample resolution
    block_centers = np.arange(n_blocks) * block_samples + block_samples // 2
    return np.interp(np.arange(len(sc_mono)), block_centers, gain_blocks).astype(np.float64)


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
    sidechain_path: Path | None = None,
    sc_depth_db: float = -12.0,
    sc_hp_hz: float | None = None,
    sc_lp_hz: float | None = None,
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

    # Sidechain ducking on the reverb tail (pumping-reverb pattern). The
    # tail dips every time the sidechain (typically the kick) hits, so the
    # reverb breathes with the song instead of washing over transients.
    sidechain_info: dict | None = None
    if sidechain_path is not None:
        sc_data, sc_sr = sf.read(str(sidechain_path), always_2d=True)
        if sc_sr != sr:
            raise ValueError(
                f"Sidechain sample rate {sc_sr} Hz differs from main {sr} Hz"
            )
        # Trim/pad to match
        n_main = reverb_out.shape[0]
        if sc_data.shape[0] > n_main:
            sc_data = sc_data[:n_main]
        elif sc_data.shape[0] < n_main:
            sc_data = np.pad(sc_data, ((0, n_main - sc_data.shape[0]), (0, 0)))
        sc_mono = sc_data.mean(axis=1)
        # Optional band-pass on the sidechain (isolate kick beater range)
        if sc_hp_hz and sc_hp_hz > 0:
            sos = butter(2, sc_hp_hz / (sr / 2), btype="high", output="sos")
            sc_mono = sosfilt(sos, sc_mono)
        if sc_lp_hz and sc_lp_hz > 0:
            sos = butter(2, sc_lp_hz / (sr / 2), btype="low", output="sos")
            sc_mono = sosfilt(sos, sc_mono)
        env = _sidechain_envelope(sc_mono, sr, depth_db=sc_depth_db)
        reverb_out = reverb_out * env[:, None]
        mean_gr_db = float(20.0 * np.log10(max(float(np.mean(env)), 1e-10)))
        peak_gr_db = float(20.0 * np.log10(max(float(np.min(env)), 1e-10)))
        sidechain_info = {
            "file": str(sidechain_path),
            "depth_db": sc_depth_db,
            "sc_hp_hz": sc_hp_hz,
            "sc_lp_hz": sc_lp_hz,
            "mean_gain_reduction_db": round(mean_gr_db, 2),
            "peak_gain_reduction_db": round(peak_gr_db, 2),
        }

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
        "sidechain": sidechain_info,
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
    parser.add_argument(
        "--ir-preset", metavar="NAME",
        help="Load an IR from tools/irs/<NAME>.wav. Alternative to --ir. "
             "Run --list-ir-presets to see the built-in IR pack.",
    )
    parser.add_argument(
        "--bpm", type=float, metavar="BPM",
        help="Tempo for BPM-synced pre-delay (used with --pre-delay-division)",
    )
    parser.add_argument(
        "--pre-delay-division", metavar="NAME",
        help="Note division for pre-delay (eighth, sixteenth, quarter, "
             "dotted-eighth, triplet-eighth, etc.). Requires --bpm.",
    )
    parser.add_argument(
        "--sidechain", type=Path, metavar="WAV",
        help="Sidechain WAV (typically kick) — ducks the reverb tail when "
             "the sidechain hits. Classic pumping-reverb pattern.",
    )
    parser.add_argument(
        "--sc-depth", type=float, default=-12.0, metavar="DB",
        help="Maximum gain reduction on the reverb tail at peak ducking "
             "(default -12 dB)",
    )
    parser.add_argument("--sc-hp", type=float, metavar="HZ", help="High-pass the sidechain before detection")
    parser.add_argument("--sc-lp", type=float, metavar="HZ", help="Low-pass the sidechain before detection")
    parser.add_argument("--list-presets", action="store_true", help="List available algorithmic presets and exit")
    parser.add_argument("--list-ir-presets", action="store_true", help="List built-in IR presets (tools/irs/) and exit")

    args = parser.parse_args()

    if args.list_presets:
        for name, p in PRESETS.items():
            print(f"\n{name}: {p['description']}")
            print(f"  room={p['room_size']}  damping={p['damping']}  width={p['width']}")
            print(f"  pre_delay={p['pre_delay_ms']}ms  wet={p['wet']}  hp={p['hp_hz']}Hz  lp={p['lp_hz']}Hz")
            print(f"  {p['notes']}")
        return

    irs_dir = Path(__file__).parent / "irs"
    if args.list_ir_presets:
        if not irs_dir.is_dir():
            print(f"No IR directory: {irs_dir}")
            return
        ir_files = sorted(irs_dir.glob("*.wav"))
        if not ir_files:
            print(f"No IR files in {irs_dir}")
            return
        print(f"IR presets in {irs_dir}:")
        for f in ir_files:
            print(f"  {f.stem}")
        return

    if args.input is None:
        parser.error("input file is required")
    if not args.input.exists():
        print(json.dumps({"error": f"Not found: {args.input}"}), file=sys.stderr)
        sys.exit(1)

    # Resolve --ir-preset to an --ir path
    if args.ir_preset is not None:
        ir_candidate = irs_dir / f"{args.ir_preset}.wav"
        if not ir_candidate.exists():
            print(json.dumps({"error": f"IR preset not found: {ir_candidate}. Run --list-ir-presets."}),
                  file=sys.stderr)
            sys.exit(1)
        args.ir = ir_candidate
    if args.ir is not None and not args.ir.exists():
        print(json.dumps({"error": f"IR file not found: {args.ir}"}), file=sys.stderr)
        sys.exit(1)
    if args.sidechain is not None and not args.sidechain.exists():
        print(json.dumps({"error": f"Sidechain file not found: {args.sidechain}"}), file=sys.stderr)
        sys.exit(1)
    if (args.bpm is None) != (args.pre_delay_division is None):
        parser.error("--bpm and --pre-delay-division must be used together")

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

    # BPM-synced pre-delay overrides explicit --pre-delay
    if args.bpm is not None and args.pre_delay_division is not None:
        params["pre_delay_ms"] = _bpm_to_pre_delay_ms(args.bpm, args.pre_delay_division)
    elif args.pre_delay is not None:
        params["pre_delay_ms"] = args.pre_delay
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
        sidechain_path=args.sidechain,
        sc_depth_db=args.sc_depth,
        sc_hp_hz=args.sc_hp,
        sc_lp_hz=args.sc_lp,
        **params,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
