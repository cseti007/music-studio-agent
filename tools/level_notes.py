"""Per-note volume leveling for percussive/uneven instrument takes.

Detects onsets in a target time range, measures each note's attack peak,
and applies a short gain envelope to lift quiet notes toward a target
peak level without disturbing the loud ones. Intended for fixing uneven
slap-bass or aggressive-finger-bass performances where the player's
dynamics swing wildly between accented and unaccented notes.

Envelope is intentionally short (~95 ms with 5 ms pre-fade, 30 ms hold,
60 ms fade-out) so it covers only the attack window and decays back to
unity before the next onset — avoids overlapping boosts that would push
the cumulative signal above 0 dBFS.

Safety scale only applies to the modified segment, never to material
outside the target range.

Usage:
    python tools/level_notes.py <input.wav> --output <output.wav> \\
        [--start SEC] [--end SEC] \\
        [--target-peak-db -4] [--quiet-threshold-db -6] [--max-boost-db 15] \\
        [--prominence-db 4]
"""

import argparse
import sys
import soundfile as sf
import numpy as np
import scipy.signal as sig


def level_notes(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    target_peak_db: float = -4.0,
    quiet_threshold_db: float = -6.0,
    max_boost_db: float = 15.0,
    prominence_db: float = 4.0,
) -> tuple[np.ndarray, dict]:
    """Returns (modified_audio, report).

    audio: shape (n, channels) or (n,)
    """
    is_mono = audio.ndim == 1
    if is_mono:
        audio_2d = audio[:, None]
    else:
        audio_2d = audio

    mono = audio_2d.mean(axis=1)
    fix_start = int(start_sec * sr)
    fix_end = min(int(end_sec * sr), len(mono))
    seg = mono[fix_start:fix_end]

    # 5 ms RMS envelope -> dB scale; pick onsets via local maxima
    win = int(sr * 0.005)
    env = np.sqrt(np.convolve(seg ** 2, np.ones(win) / win, mode="same"))
    env_db = 20 * np.log10(np.maximum(env, 1e-10))
    peaks, _ = sig.find_peaks(env_db, prominence=prominence_db, distance=sr // 8)

    attack_win = int(sr * 0.05)
    note_window = int(sr * 0.095)
    fade = int(sr * 0.005)
    hold_n = int(sr * 0.030)

    target_lin = 10 ** (target_peak_db / 20.0)
    quiet_lin = 10 ** (quiet_threshold_db / 20.0)

    gain_env = np.ones(len(seg))
    boosts: list[float] = []

    for p in peaks:
        attack_end = min(p + attack_win, len(seg))
        note_peak = float(np.max(np.abs(seg[p:attack_end])))
        if note_peak == 0 or note_peak >= quiet_lin:
            continue
        boost_db = min(20 * np.log10(target_lin / note_peak), max_boost_db)
        boost_lin = 10 ** (boost_db / 20.0)
        note_start = max(0, p - fade)
        note_end = min(p + note_window, len(seg))
        local = np.ones(note_end - note_start)
        fade_in_n = p - note_start
        if fade_in_n > 0:
            local[:fade_in_n] = np.linspace(1.0, boost_lin, fade_in_n)
        else:
            local[0] = boost_lin
        hold_end = min(fade_in_n + hold_n, len(local))
        local[fade_in_n:hold_end] = boost_lin
        if hold_end < len(local):
            local[hold_end:] = np.linspace(boost_lin, 1.0, len(local) - hold_end)
        gain_env[note_start:note_end] = np.maximum(gain_env[note_start:note_end], local)
        boosts.append(boost_db)

    seg_data = audio_2d[fix_start:fix_end].copy()
    for ch in range(seg_data.shape[1]):
        seg_data[:, ch] *= gain_env

    # Safety: scale only the modified segment if it overshoots
    seg_peak = float(np.max(np.abs(seg_data)))
    safety_scale_db = 0.0
    if seg_peak > 0.95:
        scale = 0.95 / seg_peak
        seg_data *= scale
        safety_scale_db = 20 * np.log10(scale)

    new_audio = audio_2d.copy()
    new_audio[fix_start:fix_end] = seg_data

    file_peak = float(np.max(np.abs(new_audio)))
    report = {
        "onsets_total": len(peaks),
        "notes_boosted": len(boosts),
        "boost_mean_db": float(np.mean(boosts)) if boosts else 0.0,
        "boost_max_db": float(np.max(boosts)) if boosts else 0.0,
        "boost_min_db": float(np.min(boosts)) if boosts else 0.0,
        "segment_safety_scale_db": safety_scale_db,
        "file_peak_dbfs": 20 * np.log10(max(file_peak, 1e-12)),
        "range_sec": (start_sec, end_sec),
        "params": {
            "target_peak_db": target_peak_db,
            "quiet_threshold_db": quiet_threshold_db,
            "max_boost_db": max_boost_db,
            "prominence_db": prominence_db,
        },
    }

    return (new_audio[:, 0] if is_mono else new_audio), report


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-note volume leveling on a target time range.")
    ap.add_argument("input", help="Input WAV file")
    ap.add_argument("--output", required=True, help="Output WAV file")
    ap.add_argument("--start", type=float, default=0.0, help="Range start in seconds (default 0)")
    ap.add_argument("--end", type=float, required=True, help="Range end in seconds")
    ap.add_argument("--target-peak-db", type=float, default=-4.0)
    ap.add_argument("--quiet-threshold-db", type=float, default=-6.0,
                    help="Only boost notes whose peak is below this (dBFS)")
    ap.add_argument("--max-boost-db", type=float, default=15.0)
    ap.add_argument("--prominence-db", type=float, default=4.0,
                    help="Onset detector prominence in dB")
    args = ap.parse_args()

    data, sr = sf.read(args.input, always_2d=True)
    out, report = level_notes(
        data, sr,
        start_sec=args.start,
        end_sec=args.end,
        target_peak_db=args.target_peak_db,
        quiet_threshold_db=args.quiet_threshold_db,
        max_boost_db=args.max_boost_db,
        prominence_db=args.prominence_db,
    )
    sf.write(args.output, out, sr, subtype="PCM_24")

    print(f"Onsets detected:        {report['onsets_total']}")
    print(f"Notes boosted:          {report['notes_boosted']}")
    if report["notes_boosted"]:
        print(f"Boost mean / max / min: {report['boost_mean_db']:+.1f} / {report['boost_max_db']:+.1f} / {report['boost_min_db']:+.1f} dB")
    print(f"Segment safety scale:   {report['segment_safety_scale_db']:+.2f} dB")
    print(f"File peak after:        {report['file_peak_dbfs']:+.2f} dBFS")
    print(f"Output:                 {args.output}")


if __name__ == "__main__":
    main()
