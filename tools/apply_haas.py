"""Haas (precedence) effect: widen a mono or near-mono stem by delaying one
channel by 5-25 ms relative to the other.

Psychoacoustic principle: the brain locates a sound by whichever ear hears it
first (precedence effect). Below ~30 ms delay, the two channels fuse into a
single image but the listener perceives extra width. Above 30 ms it splits
into a discrete echo.

Practical caveats — Haas is *the* classic mono-compatibility trap:
  - When summed to mono, the delayed and direct channels comb-filter, hollowing
    the sound. Use only when mono playback is acceptable (modern streaming
    rarely sums to mono, but check the target medium).
  - Don't use on the bass — phase issues at low frequencies are very audible.

Relevance check refuses to apply if:
  - The stem is already wide (ms_width_ratio > 0.3). Adding Haas to an already
    wide stem just smears the image.
  - The stem looks like a bass instrument (significant 30-120 Hz content with
    low spectral centroid). Haas on bass = mud + phase issues.

Usage:
  python apply_haas.py guitar.wav --output-dir DIR --delay-ms 12 --side right
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


PRESETS: dict[str, dict] = {
    "haas_guitar": {
        "description": "Subtle widening for rhythm guitar — 12ms, right side",
        "delay_ms": 12.0,
        "side": "right",
        "wet": 0.85,
    },
    "haas_vocal_doubler": {
        "description": "Vocal doubler effect — short delay, both sides",
        "delay_ms": 18.0,
        "side": "right",
        "wet": 0.6,
    },
    "haas_synth_pad": {
        "description": "Wide pad — longer delay, both sides, lower mix",
        "delay_ms": 22.0,
        "side": "left",
        "wet": 0.75,
    },
}


_MAX_SAFE_DELAY_MS = 30.0


# ---------------------------------------------------------------------------
# Relevance check
# ---------------------------------------------------------------------------

def _band_rms(signal: np.ndarray, sr: int, lo: float, hi: float) -> float:
    from scipy.signal import butter, sosfilt
    nyq = sr / 2.0
    sos = butter(4, [lo / nyq, min(hi / nyq, 0.999)], btype="band", output="sos")
    return float(np.sqrt(np.mean(sosfilt(sos, signal) ** 2)))


# Mono filenames whose pattern suggests they're one half of a stereo pair
# (overheads, room L/R, stereo mic pair). Applying Haas to such a stem makes
# no sense — the other half already carries the stereo information; widening
# one half breaks the relationship.
_STEREO_PAIR_PATTERNS = [
    " l.",  " l_",  " l-",  " l ",
    " r.",  " r_",  " r-",  " r ",
    ".l.", "_l.", "-l.",
    ".r.", "_r.", "-r.",
]


def _looks_like_stereo_pair_half(file_path: Path) -> str | None:
    """Returns a description string if the filename looks like one half of a
    stereo pair (OH L/R, room L/R, mic pair). Returns None if it does not."""
    # Look at the parent track-dir name + the filename, both lowercased.
    name = f"{file_path.parent.name} {file_path.name}".lower()
    # Surround with spaces so " l " etc. patterns match clean tokens
    name_padded = f" {name} "
    for pat in _STEREO_PAIR_PATTERNS:
        if pat in name_padded:
            # Try to extract which side
            side = "L" if " l" in name_padded or ".l." in name or "_l." in name or "-l." in name else "R"
            return f"filename contains '{pat.strip()}' — looks like the {side} channel of a stereo pair"
    return None


def _relevance_check(data: np.ndarray, sr: int, file_path: Path | None = None) -> dict:
    mono = data.mean(axis=1) if data.ndim == 2 and data.shape[1] > 1 else (
        data[:, 0] if data.ndim == 2 else data
    )
    n_channels = data.shape[1] if data.ndim == 2 else 1

    # Existing M/S width (only meaningful if input is stereo)
    if data.ndim == 2 and data.shape[1] == 2:
        L, R = data[:, 0], data[:, 1]
        rms_mid = float(np.sqrt(np.mean(((L + R) / 2) ** 2) + 1e-12))
        rms_side = float(np.sqrt(np.mean(((L - R) / 2) ** 2) + 1e-12))
        ms_width = rms_side / max(rms_mid, 1e-10)
    else:
        ms_width = 0.0

    # Bass-likeness: ratio of low-band RMS to overall RMS
    overall = float(np.sqrt(np.mean(mono ** 2) + 1e-12))
    low_rms = _band_rms(mono, sr, 30.0, 120.0)
    low_ratio = low_rms / max(overall, 1e-10)

    # Spectral centroid (rough — first PSD moment up to 8 kHz)
    from scipy.signal import welch
    nperseg = min(len(mono), 32768)
    freqs, psd = welch(mono, fs=sr, nperseg=nperseg)
    mask = freqs <= 8000
    centroid = float(np.sum(freqs[mask] * psd[mask]) / max(np.sum(psd[mask]), 1e-12))

    # Check if this mono file appears to be half of a stereo pair
    stereo_pair_warning = None
    if n_channels == 1 and file_path is not None:
        stereo_pair_warning = _looks_like_stereo_pair_half(file_path)

    issues = []
    if ms_width > 0.3:
        issues.append(f"already wide (M/S width {ms_width:.2f} > 0.3) — Haas will smear, not widen")
    if low_ratio > 0.4 and centroid < 500.0:
        issues.append(f"bass-heavy stem (low ratio {low_ratio:.2f}, centroid {centroid:.0f} Hz) — Haas causes phase mud")
    if stereo_pair_warning is not None:
        issues.append(
            f"{stereo_pair_warning}. Haas on one half breaks the pair's stereo "
            f"relationship — apply Haas to the summed pair, not one channel."
        )

    return {
        "tool": "haas",
        "ms_width_ratio": round(ms_width, 3),
        "low_band_ratio": round(low_ratio, 3),
        "spectral_centroid_hz": round(centroid, 0),
        "channels": n_channels,
        "looks_like_stereo_pair_half": stereo_pair_warning is not None,
        "recommend_skip": bool(issues),
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def apply_haas(
    input_path: Path,
    output_dir: Path,
    delay_ms: float,
    side: str = "right",
    wet: float = 1.0,
    preset_name: str | None = None,
    force: bool = False,
) -> dict:
    if delay_ms > _MAX_SAFE_DELAY_MS:
        print(
            f"WARNING: delay {delay_ms} ms > {_MAX_SAFE_DELAY_MS} ms — "
            f"image will split into a discrete echo, not widen",
            file=sys.stderr,
        )

    data, sr = sf.read(str(input_path), always_2d=True)

    rel = _relevance_check(data, sr, file_path=input_path)
    if rel["recommend_skip"] and not force:
        print(f"WARNING: Haas relevance_check failed for {input_path.name}:", file=sys.stderr)
        for issue in rel["issues"]:
            print(f"  - {issue}", file=sys.stderr)
        print("  Skipping (pass --force to apply anyway).", file=sys.stderr)
        report = {
            "input": str(input_path),
            "output": None,
            "preset": preset_name,
            "relevance_check": rel,
            "applied": False,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "haas_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    # Force stereo so the Haas effect has somewhere to go
    if data.shape[1] == 1:
        mono = data[:, 0]
        L = mono.copy()
        R = mono.copy()
    else:
        L = data[:, 0].copy()
        R = data[:, 1].copy()

    delay_samples = int(round(delay_ms * sr / 1000.0))
    if delay_samples > 0:
        if side == "left":
            # Delay the left channel
            delayed = np.concatenate([np.zeros(delay_samples), L[:-delay_samples]])
            L_out = L * (1.0 - wet) + delayed * wet
            R_out = R
        else:
            # Delay the right channel (default)
            delayed = np.concatenate([np.zeros(delay_samples), R[:-delay_samples]])
            R_out = R * (1.0 - wet) + delayed * wet
            L_out = L
    else:
        L_out, R_out = L, R

    output_data = np.stack([L_out, R_out], axis=1)

    peak = float(np.max(np.abs(output_data)))
    if peak > 1.0:
        print(f"WARNING: output peak {20*np.log10(peak):.1f} dBFS — scaling down", file=sys.stderr)
        output_data = output_data / peak

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (input_path.stem + "_haas.wav")
    sf.write(str(out_path), output_data, sr, subtype="PCM_24")

    report = {
        "input": str(input_path),
        "output": str(out_path),
        "preset": preset_name,
        "settings": {"delay_ms": delay_ms, "side": side, "wet": wet},
        "relevance_check": rel,
        "applied": True,
        "sample_rate": sr,
        "mono_compatibility_warning": (
            "Haas effect comb-filters when summed to mono — verify on a mono speaker"
        ),
    }
    (output_dir / "haas_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Widen a stem via the Haas precedence effect (short channel delay).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Presets: " + ", ".join(PRESETS),
    )
    parser.add_argument("file", type=Path, nargs="?", help="Input WAV")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--preset", choices=list(PRESETS), help="Haas preset")
    parser.add_argument("--delay-ms", type=float, help="Channel delay in ms (5-25 typical)")
    parser.add_argument("--side", choices=["left", "right"], help="Which side to delay")
    parser.add_argument("--wet", type=float, help="Wet mix on delayed side (0.0-1.0)")
    parser.add_argument("--force", action="store_true",
                        help="Apply even if relevance_check recommends skip")
    parser.add_argument("--list-presets", action="store_true", help="List presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        for name, p in PRESETS.items():
            print(f"\n{name}: {p['description']}")
            print(f"  delay={p['delay_ms']}ms  side={p['side']}  wet={p['wet']}")
        return

    if args.file is None:
        parser.error("file is required (unless --list-presets)")
    if args.output_dir is None:
        parser.error("--output-dir is required (unless --list-presets)")
    if not args.file.exists():
        print(json.dumps({"error": f"Not found: {args.file}"}), file=sys.stderr)
        sys.exit(1)

    params: dict = {"delay_ms": 12.0, "side": "right", "wet": 0.85}
    preset_name = None
    if args.preset:
        preset_name = args.preset
        params = dict(PRESETS[args.preset])
        params.pop("description", None)

    if args.delay_ms is not None: params["delay_ms"] = args.delay_ms
    if args.side is not None:     params["side"] = args.side
    if args.wet is not None:      params["wet"] = args.wet

    result = apply_haas(
        input_path=args.file,
        output_dir=args.output_dir,
        preset_name=preset_name,
        force=args.force,
        **params,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
