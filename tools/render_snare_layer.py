"""Render a sample-accurate snare layer triggered by a close snare track.

Snare hits in the input (e.g. the snare top microphone) are found with an
offline replica of the LSP Trigger detector (RMS sidechain with a 150 Hz
high-pass; see _snare.trigger_hits), refined to the true onset, and a copy of
the one-shot sample is placed at each onset (1 ms pre-roll), so there is no
trigger delay. Per-hit gain follows the hit level:

  gain_db = base + level_db + dynamics * (hit_db - median_hit_db)

where base makes the median layer hit peak equal the median input hit peak.
The result is a mono float WAV with the input's length and sample rate, to
be placed at the input's position (layered under the original, not replacing it).

Alignment: with polarity and offset "auto" the layer is summed with the kit
(kit_path; otherwise the input alone) at the hits for both polarities and
global offsets within +/-2 ms (0.25 ms steps), and the combination with the
most body (150-400 Hz) energy is kept; a crack-band (1-5 kHz) loss of more
than 1 dB against the best crack option is flagged for listening. Judge it on
the full kit: an alignment that suits the close microphone alone can cancel
against overheads and rooms. A positive offset moves the layer later.

A sample is release-safe only if its sidecar JSON (written by snare_match.py)
says release_ok; anything else (e.g. a hit cut from a commercial reference)
requires --private and must not leave this machine.

Usage:
  python tools/render_snare_layer.py snare_top.wav --sample candidates/01_x.wav \\
      --kit kit.wav --output-dir output/layer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

import _snare as S
from _recall import record_operation

PREROLL_MS = 1.0
GAIN_RANGE_DB = (-24.0, 6.0)
MAX_OFFSET_MS = 2.0   # alignment search range (+/-) for offset "auto"
OFFSET_STEP_MS = 0.25


def _license(sample_path: Path) -> dict:
    sidecar = sample_path.with_suffix(".json")
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        return {k: meta.get(k) for k in ("license", "release_ok", "attribution")}
    return {"license": "unknown", "release_ok": False, "attribution": None}


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return sosfilt(butter(2, [lo / (S.SR / 2), hi / (S.SR / 2)], "band", output="sos"), x)


@record_operation("render_snare_layer")
def render_snare_layer(
    input_path: Path,
    sample_path: Path,
    output_dir: Path,
    kit_path: Path | None = None,
    threshold_db: float = -26.0,
    react_ms: float = 2.0,
    detect_ms: float = 1.0,
    release_ms: float = 30.0,
    dynamics: float = 0.5,
    level_db: float = 0.0,
    polarity: str = "auto",
    offset_ms: float | str = "auto",
    private: bool = False,
) -> dict:
    if polarity not in ("auto", "normal", "inverted"):
        raise ValueError("polarity must be auto, normal or inverted")
    lic = _license(sample_path)
    if not lic["release_ok"] and not private:
        raise ValueError(f"{sample_path.name} has no release_ok license; use private=True for private listening only")

    info = sf.info(str(input_path))
    y = S.load_mono(input_path)
    hits = S.trigger_hits(y, threshold_db, react_ms, detect_ms, release_ms)
    if not hits:
        raise ValueError("no snare hits found in the input")

    sample = S.load_mono(sample_path)
    rect = np.abs(S._hp(sample, 150.0))
    s_on = S.attack_onset(rect, 0, S.ms(60))
    sample_db = 20.0 * np.log10(rect[s_on:s_on + S.ms(10)].max() + 1e-12)
    pre = s_on - max(s_on - S.ms(PREROLL_MS), 0)  # pre-roll actually available in the file
    sample = sample[s_on - pre:]

    levels = np.array([h["level_db"] for h in hits])
    base = float(np.median(levels) - sample_db + level_db)
    pad = S.ms(MAX_OFFSET_MS)
    layer = np.zeros(pad + len(y) + len(sample) + pad)
    for h, lvl in zip(hits, levels):
        rel = float(np.clip(dynamics * (lvl - np.median(levels)), *GAIN_RANGE_DB))
        start = pad + h["onset"] - pre
        layer[start:start + len(sample)] += sample * 10.0 ** ((base + rel) / 20.0)

    def shifted(offset_samples: int) -> np.ndarray:
        return layer[pad - offset_samples:pad - offset_samples + len(y)]

    starts = [h["onset"] for h in hits]
    ref = S.load_mono(kit_path)[:len(y)] if kit_path else y
    ref = np.pad(ref, (0, len(y) - len(ref)))
    offsets = (np.arange(-MAX_OFFSET_MS, MAX_OFFSET_MS + 1e-9, OFFSET_STEP_MS) if offset_ms == "auto"
               else [float(offset_ms)])
    signs = {"normal": 1.0, "inverted": -1.0} if polarity == "auto" else {polarity: 1.0 if polarity == "normal" else -1.0}
    # filtering is linear: filter kit and layer once per band, then shift and sum
    bands = {"body_db": (150, 400, 60), "crack_db": (1000, 5000, 20)}
    filtered = {k: (_bandpass(ref, lo, hi), _bandpass(layer, lo, hi), win) for k, (lo, hi, win) in bands.items()}
    trials = []
    for off in offsets:
        o = S.ms(off)
        trial = {"offset_ms": round(float(off), 2)}
        for name, sign in signs.items():
            trial = dict(trial, polarity=name)
            for key, (fr, fl, win) in filtered.items():
                w = S.ms(win)
                e = sum(np.sum((fr[st:st + w] + sign * fl[pad - o + st:pad - o + st + w]) ** 2) for st in starts)
                trial[key] = round(float(10.0 * np.log10(e + 1e-20)), 2)
            trials.append(trial)
    best = max(trials, key=lambda t: t["body_db"])
    zero = next((t for t in trials if t["offset_ms"] == 0.0 and t["polarity"] == "normal"), None)
    test = {"judged_on": "kit" if kit_path else "input (close microphone only)", "chosen": best,
            "zero_offset_normal": zero,
            "crack_disagrees": max(t["crack_db"] for t in trials) - best["crack_db"] > 1.0}
    polarity, offset = best["polarity"], best["offset_ms"]
    layer = shifted(S.ms(offset)) * (1.0 if polarity == "normal" else -1.0)

    if info.samplerate != S.SR:
        layer = resample_poly(layer, info.samplerate, S.SR)[:info.frames]
        layer = np.pad(layer, (0, info.frames - len(layer)))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{input_path.stem}_snare_layer.wav"
    sf.write(str(out_path), layer, info.samplerate, subtype="FLOAT")

    report = {"input": str(input_path), "output": str(out_path), "sample": str(sample_path),
              "kit": str(kit_path) if kit_path else None, "placed": len(hits),
              "median_hit_db": round(float(np.median(levels)), 2), "base_gain_db": round(base, 2),
              "polarity": polarity, "offset_ms": offset, "alignment_test": test, "private": private, **lic}
    (output_dir / "snare_layer_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a sample-accurate snare layer from a close snare track.")
    parser.add_argument("input", type=Path, help="Close snare track (e.g. snare top), WAV")
    parser.add_argument("--sample", type=Path, required=True, help="One-shot snare sample (snare_match candidate)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kit", type=Path, help="Kit mix without the layer, same start/length (polarity check)")
    parser.add_argument("--threshold-db", type=float, default=-26.0, help="Detection threshold (default -26)")
    parser.add_argument("--react-ms", type=float, default=2.0, help="Detection RMS window (default 2)")
    parser.add_argument("--detect-ms", type=float, default=1.0, help="Time above threshold to trigger (default 1)")
    parser.add_argument("--release-ms", type=float, default=30.0, help="Time below release level to re-arm (default 30)")
    parser.add_argument("--dynamics", type=float, default=0.5, help="Hit level tracking, 0..1 (default 0.5)")
    parser.add_argument("--level-db", type=float, default=0.0, help="Layer level re the input hits (default 0)")
    parser.add_argument("--polarity", choices=["auto", "normal", "inverted"], default="auto")
    parser.add_argument("--offset-ms", default="auto", help="Layer offset in ms, or auto (default: auto)")
    parser.add_argument("--private", action="store_true", help="Allow samples without a release_ok license")
    args = parser.parse_args()
    for path in [args.input, args.sample] + ([args.kit] if args.kit else []):
        if not path.exists():
            print(json.dumps({"error": f"Not found: {path}"}), file=sys.stderr)
            sys.exit(1)
    try:
        report = render_snare_layer(args.input, args.sample, args.output_dir, args.kit, args.threshold_db,
                                    args.react_ms, args.detect_ms, args.release_ms, args.dynamics,
                                    args.level_db, args.polarity,
                                    args.offset_ms if args.offset_ms == "auto" else float(args.offset_ms),
                                    args.private)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
