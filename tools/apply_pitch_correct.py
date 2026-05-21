"""Vocal pitch correction — librosa.pyin pitch detection + PSOLA shifting,
quantised to a named musical scale.

Chain placement: AFTER the additive EQ in the vocal chain, BEFORE saturation
and reverb sends. The corrected vocal goes back through the rest of the chain.

Algorithm:
  1. Detect frame-wise fundamental frequency with librosa.pyin (probabilistic
     YIN) — robust for monophonic vocal.
  2. Quantise each detected pitch to the nearest scale-note of the chosen
     key (e.g. A minor → {A, B, C, D, E, F, G} mapped onto the 12-TET grid).
  3. Blend between original and quantised pitch by `strength` (0=no
     correction, 1=full Auto-Tune-style snap). Cents-deviation gets scaled.
  4. PSOLA-shift each frame to land on the blended target pitch
     (psola.vocode). Formants are preserved.

Presets:
  pitch_correct_subtle      strength 0.3, gentle nudge
  pitch_correct_pop         strength 0.7, modern-pop snap
  pitch_correct_hard_tune   strength 1.0, full Auto-Tune effect

Usage:
  python tools/apply_pitch_correct.py vocal.wav --output-dir DIR \
      --scale-root F# --scale-mode minor --strength 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

try:
    import psola
    _HAS_PSOLA = True
except ImportError:
    _HAS_PSOLA = False


PRESETS_DIR = Path(__file__).parent / "presets"

PRESETS: dict[str, dict] = {
    "pitch_correct_subtle": {
        "description": "Gentle pitch nudge — preserves vocal character and pitch nuance",
        "notes": "strength 0.3. The detected pitch deviation from the nearest scale note is reduced by 30%. Use when the take has minor wobble but you don't want the corrected sound.",
        "strength": 0.3,
    },
    "pitch_correct_pop": {
        "description": "Modern pop snap — moderate correction, still natural",
        "notes": "strength 0.7. Strong corrective effect but not the full Auto-Tune sound — pitch deviation reduced by 70%. Standard for pop / rock leads.",
        "strength": 0.7,
    },
    "pitch_correct_hard_tune": {
        "description": "Full Auto-Tune effect — instant snap to scale grid",
        "notes": "strength 1.0. Pitch fully quantised to the scale grid. This is the audible 'Cher / T-Pain' effect when combined with fast pitch-tracking settings.",
        "strength": 1.0,
    },
}

# 12-TET semitone offsets from the scale root for major and minor modes
_SCALE_DEGREES = {
    "major":             [0, 2, 4, 5, 7, 9, 11],
    "minor":             [0, 2, 3, 5, 7, 8, 10],
    "natural_minor":     [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":    [0, 2, 3, 5, 7, 8, 11],
    "dorian":            [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":        [0, 2, 4, 5, 7, 9, 10],
    "chromatic":         list(range(12)),  # no quantisation effect
}

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_ENHARMONIC = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def _note_to_semitone(note: str) -> int:
    """Convert a note name (e.g. 'F#', 'Bb', 'A') to a 0-11 semitone index from C."""
    note = note.strip()
    note = _ENHARMONIC.get(note, note)
    if note not in _NOTE_NAMES:
        raise ValueError(f"Unknown note name: '{note}'. Use C, C#, D, ..., B (or flats: Bb, Eb, etc.)")
    return _NOTE_NAMES.index(note)


def _build_scale(root: str, mode: str) -> list[int]:
    """Return the 12-TET semitone indices of every scale degree (octave-wrapped)."""
    if mode not in _SCALE_DEGREES:
        raise ValueError(f"Unknown mode: '{mode}'. Available: {list(_SCALE_DEGREES)}")
    root_semi = _note_to_semitone(root)
    return [(root_semi + d) % 12 for d in _SCALE_DEGREES[mode]]


def _quantise_to_scale(f0_hz: np.ndarray, scale_semis: list[int]) -> np.ndarray:
    """For each detected pitch, snap to the nearest pitch on the scale grid.

    Works in cents-space: convert f0 -> midi note -> nearest scale-grid note ->
    back to Hz. Unvoiced frames (NaN in f0) pass through unchanged.
    """
    scale_set = set(scale_semis)
    out = np.copy(f0_hz)
    voiced_mask = ~np.isnan(f0_hz)
    if not np.any(voiced_mask):
        return out

    f0_voiced = f0_hz[voiced_mask]
    midi = 69 + 12 * np.log2(f0_voiced / 440.0)
    midi_int = np.round(midi).astype(int)
    midi_cents_offset = midi - midi_int  # fractional part in semitone units

    # For each midi note, find nearest semitone that's in the scale
    nearest_scale = np.zeros_like(midi_int)
    for i, m in enumerate(midi_int):
        m_in_octave = m % 12
        if m_in_octave in scale_set:
            nearest_scale[i] = m
        else:
            # Find closest scale degree in either direction
            best_diff = 999
            best_m = m
            for offset in range(-6, 7):
                candidate = m + offset
                if (candidate % 12) in scale_set and abs(offset) < best_diff:
                    best_diff = abs(offset)
                    best_m = candidate
            nearest_scale[i] = best_m

    # Convert back to Hz (preserve the fractional cents offset if strength=0)
    quantised_midi = nearest_scale + 0.0  # keep float
    quantised_hz = 440.0 * (2.0 ** ((quantised_midi - 69) / 12.0))
    out[voiced_mask] = quantised_hz
    return out


def _blend_pitch(f0_orig: np.ndarray, f0_quantised: np.ndarray, strength: float) -> np.ndarray:
    """Linear blend between original and quantised pitch.

    strength = 0   → original (no correction)
    strength = 1   → fully quantised
    strength = 0.5 → midway
    """
    strength = max(0.0, min(1.0, strength))
    voiced = ~np.isnan(f0_orig) & ~np.isnan(f0_quantised)
    out = f0_orig.copy()
    out[voiced] = (1.0 - strength) * f0_orig[voiced] + strength * f0_quantised[voiced]
    return out


def apply_pitch_correct(
    input_path: Path,
    output_dir: Path,
    scale_root: str,
    scale_mode: str,
    strength: float,
    fmin_hz: float = 80.0,
    fmax_hz: float = 600.0,
    preset_name: str | None = None,
) -> dict:
    if not _HAS_PSOLA:
        raise ImportError(
            "psola package not available — pip install psola (or run "
            "`conda run -n music-studio-agent pip install psola`)"
        )

    data, sr = sf.read(str(input_path), always_2d=True)
    if data.shape[1] > 1:
        # PSOLA is monophonic; if stereo, average to mono for processing
        # then duplicate to stereo on output
        mono_input = data.mean(axis=1).astype(np.float32)
        was_stereo = True
    else:
        mono_input = data[:, 0].astype(np.float32)
        was_stereo = False

    # Pitch detection
    f0_orig, voiced_flag, voiced_prob = librosa.pyin(
        mono_input, fmin=fmin_hz, fmax=fmax_hz, sr=sr,
    )

    # Scale quantisation
    scale_semis = _build_scale(scale_root, scale_mode)
    f0_quant = _quantise_to_scale(f0_orig, scale_semis)

    # Blend by strength
    f0_target = _blend_pitch(f0_orig, f0_quant, strength)

    # PSOLA-shift
    y_corrected = psola.vocode(
        mono_input, sample_rate=sr,
        target_pitch=f0_target,
        fmin=fmin_hz, fmax=fmax_hz,
    )

    # Restore stereo if input was stereo (duplicate corrected mono)
    if was_stereo:
        out_audio = np.stack([y_corrected, y_corrected], axis=1)
    else:
        out_audio = y_corrected[:, np.newaxis]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (input_path.stem + "_pitched.wav")
    sf.write(str(output_path), out_audio, sr, subtype="PCM_24")

    # Stats
    voiced_orig = f0_orig[~np.isnan(f0_orig)]
    voiced_target = f0_target[~np.isnan(f0_target)]
    if len(voiced_orig) > 0:
        cents_shift = 1200.0 * np.log2(np.maximum(voiced_target, 1) /
                                       np.maximum(voiced_orig, 1))
        mean_shift = float(np.mean(cents_shift))
        peak_shift = float(np.max(np.abs(cents_shift)))
    else:
        mean_shift = 0.0
        peak_shift = 0.0

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "preset": preset_name,
        "settings": {
            "scale_root": scale_root,
            "scale_mode": scale_mode,
            "strength": strength,
            "fmin_hz": fmin_hz,
            "fmax_hz": fmax_hz,
        },
        "stats": {
            "voiced_frames": int(np.sum(~np.isnan(f0_orig))),
            "total_frames": int(len(f0_orig)),
            "mean_cents_shift": round(mean_shift, 1),
            "peak_cents_shift": round(peak_shift, 1),
        },
        "sample_rate": sr,
    }
    (output_dir / "pitch_correct_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def _load_preset(name: str) -> dict:
    if name in PRESETS:
        return PRESETS[name]
    path = PRESETS_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unknown preset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vocal pitch correction (librosa.pyin + psola PSOLA shift + scale quantisation).",
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input vocal WAV")
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--scale-root", default="A", help="Scale root note (C, C#, D, ... B, or flat: Bb, Eb)")
    parser.add_argument("--scale-mode", default="minor",
                        choices=list(_SCALE_DEGREES),
                        help="Scale mode (default: minor)")
    parser.add_argument("--strength", type=float, default=None,
                        help="Pitch correction strength 0..1 (0=no correction, 1=full snap)")
    parser.add_argument("--preset", choices=list(PRESETS), default=None)
    parser.add_argument("--fmin", type=float, default=80.0, help="Lower vocal pitch bound (Hz)")
    parser.add_argument("--fmax", type=float, default=600.0, help="Upper vocal pitch bound (Hz)")
    parser.add_argument("--list-presets", action="store_true")
    args = parser.parse_args()

    if args.list_presets:
        for name, p in PRESETS.items():
            print(f"  {name:<28} strength={p['strength']}  — {p['description']}")
        return

    if not args.input or not args.output_dir:
        parser.error("input and --output-dir required")

    if args.preset:
        p = _load_preset(args.preset)
        strength = p["strength"]
    else:
        strength = args.strength if args.strength is not None else 0.7

    if args.strength is not None:
        strength = args.strength

    apply_pitch_correct(
        args.input, args.output_dir,
        scale_root=args.scale_root,
        scale_mode=args.scale_mode,
        strength=strength,
        fmin_hz=args.fmin, fmax_hz=args.fmax,
        preset_name=args.preset,
    )


if __name__ == "__main__":
    main()
