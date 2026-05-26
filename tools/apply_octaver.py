"""Sub-octave generator: pitch-shift a stem down by N octaves, low-pass it,
mix back into the original. Adds psychoacoustic weight to bass DI by
synthesizing a sub-octave layer below the fundamental.

Output is band-limited to the sub range so it doesn't muddy the upper bass.
Default: -1 octave, HP 30 Hz, LP 120 Hz, mix 0.2 (-14 dB).

Usage:
    python tools/apply_octaver.py bass.wav --output-dir out/ \\
        --octaves -1 --hp-hz 30 --lp-hz 120 --mix 0.2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt


def _design_bandpass(sr, hp_hz, lp_hz, order=2):
    nyq = sr / 2.0
    return butter(order, [hp_hz / nyq, lp_hz / nyq], btype="band", output="sos")


def apply_octaver(input_path: Path, output_path: Path,
                  octaves: float = -1.0,
                  hp_hz: float = 30.0,
                  lp_hz: float = 120.0,
                  mix: float = 0.2) -> dict:
    audio, sr = sf.read(str(input_path), always_2d=True)
    n_ch = audio.shape[1]

    # Pitch-shift each channel down by N octaves.
    # librosa expects (n_samples,) mono or (n_channels, n_samples) shape.
    shifted_channels = []
    for c in range(n_ch):
        x = audio[:, c].astype(np.float32)
        y = librosa.effects.pitch_shift(x, sr=sr, n_steps=octaves * 12.0)
        # librosa may return slightly different length; trim/pad to match
        if len(y) > len(x):
            y = y[:len(x)]
        elif len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        shifted_channels.append(y)
    shifted = np.stack(shifted_channels, axis=1).astype(np.float64)

    # Band-pass the shifted signal so only sub content survives
    sos = _design_bandpass(sr, hp_hz, lp_hz)
    for c in range(n_ch):
        shifted[:, c] = sosfilt(sos, shifted[:, c])

    # Mix back into the original
    output = audio + shifted * mix

    # Peak limit safety
    peak = float(np.max(np.abs(output)))
    safety_scale_db = 0.0
    if peak > 0.99:
        scale = 0.99 / peak
        output = output * scale
        safety_scale_db = 20 * np.log10(scale)

    sf.write(str(output_path), output, sr, subtype="PCM_24")

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "octaves": octaves,
        "hp_hz": hp_hz,
        "lp_hz": lp_hz,
        "mix": mix,
        "shifted_band_gain_db": round(20 * np.log10(mix + 1e-12), 2),
        "safety_scale_db": round(safety_scale_db, 2),
        "peak_dbfs": round(20 * np.log10(max(float(np.max(np.abs(output))), 1e-12)), 2),
        "sample_rate": sr,
        "duration_sec": round(len(output) / sr, 3),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="Input WAV (typically bass stem)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--octaves", type=float, default=-1.0,
                    help="Octaves to shift (negative = down). Default -1.")
    ap.add_argument("--hp-hz", type=float, default=30.0,
                    help="HP filter on the shifted signal (default 30 Hz).")
    ap.add_argument("--lp-hz", type=float, default=120.0,
                    help="LP filter on the shifted signal (default 120 Hz).")
    ap.add_argument("--mix", type=float, default=0.2,
                    help="Linear mix factor of the shifted signal (default 0.2 ~= -14 dB).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (args.input.stem + "_octaved.wav")
    report = apply_octaver(args.input, output_path,
                            octaves=args.octaves, hp_hz=args.hp_hz,
                            lp_hz=args.lp_hz, mix=args.mix)
    print(json.dumps(report, indent=2))
    report_path = args.output_dir / (args.input.stem + "_octaver_report.json")
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
