"""Encode an export with lossy codecs, decode it, and measure what playback gets.

Uses the ffmpeg command-line tool. Each available codec is encoded at a typical
streaming bitrate, decoded to 32-bit float WAV and measured: sample peak, 8x
true peak, samples above full scale and integrated loudness. The decoded files
are kept for listening. This is a technical measurement of one encoder build
and bitrate; platforms use their own encoders and settings (ffmpeg's native
AAC encoder is not Apple's), and the result does not replace a
codec_roundtrip listening review (see review_delivery.py). The time of the
worst sample peak is reported: an overshoot only at the very start or end of
the file is usually an encoder edge transient on an abrupt full-scale start.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from _dsp import worst_channel_true_peak_dbfs
from _recall import content_hash


# codec -> (ffmpeg encoder, container suffix, encoder arguments, note)
CODECS = {
    "aac": ("aac", ".m4a", ["-b:a", "256k"], "AAC 256 kb/s (Apple Music / YouTube class)"),
    "vorbis": ("libvorbis", ".ogg", ["-b:a", "320k"], "Ogg Vorbis 320 kb/s (Spotify very-high class)"),
    "opus": ("libopus", ".opus", ["-b:a", "160k"], "Opus 160 kb/s (YouTube class); decodes at 48 kHz"),
    "mp3": ("libmp3lame", ".mp3", ["-b:a", "320k"], "MP3 320 kb/s"),
}


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg not found on PATH; install it to run codec round-trips")
    return path


def available_encoders(ffmpeg: str) -> set[str]:
    proc = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
    return {line.split()[1] for line in proc.stdout.splitlines()
            if len(line.split()) > 1 and line.split()[0].startswith("A")}


def _measure(path: Path) -> dict:
    data, sr = sf.read(path, always_2d=True, dtype="float64")
    loudness = float(pyln.Meter(sr).integrated_loudness(data))
    return {"sample_rate": sr,
            "peak_time_sec": round(float(np.argmax(np.abs(data).max(axis=1)) / sr), 3),
            "sample_peak_dbfs": round(float(20 * np.log10(max(np.max(np.abs(data)), 1e-12))), 2),
            "true_peak_dbtp": round(worst_channel_true_peak_dbfs(data.T, oversample=8), 2),
            "samples_over_full_scale": int(np.sum(np.abs(data) > 1.0)),
            "integrated_lufs": round(loudness, 2) if np.isfinite(loudness) else None}


def codec_roundtrip(audio: Path, output_dir: Path, codecs: list[str] | None = None,
                    tp_ceiling: float = -1.0) -> dict:
    if not np.isfinite(tp_ceiling) or tp_ceiling > 0:
        raise ValueError("Ceiling must be a finite value <= 0 dBTP")
    unknown = sorted(set(codecs or []) - set(CODECS))
    if unknown:
        raise ValueError(f"Unknown codecs {unknown}; choose from {sorted(CODECS)}")
    ffmpeg = _ffmpeg()
    encoders = available_encoders(ffmpeg)
    source = _measure(audio)
    output_dir.mkdir(parents=True, exist_ok=True)
    results, skipped = {}, {}
    for name in codecs or list(CODECS):
        encoder, suffix, enc_args, note = CODECS[name]
        if encoder not in encoders:
            skipped[name] = f"ffmpeg encoder {encoder!r} not available in this build"
            continue
        decoded = output_dir / f"{audio.stem}.{name}_decoded.wav"
        with tempfile.TemporaryDirectory() as tmp:
            encoded = Path(tmp) / f"encoded{suffix}"
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio),
                            "-c:a", encoder, *enc_args, str(encoded)], check=True)
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(encoded),
                            "-c:a", "pcm_f32le", str(decoded)], check=True)
        measured = _measure(decoded)
        measured.update(codec=name, encoder=encoder, settings=" ".join(enc_args), note=note,
                        decoded_file=str(decoded.resolve()),
                        true_peak_rise_db=round(measured["true_peak_dbtp"] - source["true_peak_dbtp"], 2),
                        within_ceiling=measured["true_peak_dbtp"] <= tp_ceiling,
                        clips_on_fixed_point_decode=measured["samples_over_full_scale"] > 0)
        results[name] = measured
    report = {"audio": str(audio.resolve()), "artifact_hash": content_hash(audio),
              "tp_ceiling": tp_ceiling, "source": source, "codecs": results, "skipped": skipped,
              "all_within_ceiling": bool(results) and all(r["within_ceiling"] for r in results.values()),
              "listening_review": {"status": "pending", "performed_by_tool": False,
                                   "note": "Audition the decoded files; record approval as scope codec_roundtrip "
                                           "against artifact_hash"},
              "assessment_scope": "One local encoder build per codec; platform encoders may differ"}
    (output_dir / "codec_roundtrip.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="Exported master WAV")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for decoded WAVs and codec_roundtrip.json")
    parser.add_argument("--codecs", type=lambda s: [c.strip() for c in s.split(",") if c.strip()],
                        help=f"Comma-separated subset of {','.join(CODECS)} (default: all available)")
    parser.add_argument("--tp-ceiling", type=float, default=-1.0,
                        help="True-peak ceiling the decoded audio is checked against (default -1.0 dBTP)")
    args = parser.parse_args()
    report = codec_roundtrip(args.audio, args.output_dir, args.codecs, args.tp_ceiling)
    print(f"Source: TP {report['source']['true_peak_dbtp']:+.2f} dBTP, "
          f"{report['source']['integrated_lufs']} LUFS")
    for name, r in report["codecs"].items():
        flag = "[OK]  " if r["within_ceiling"] else "[OVER]"
        print(f"  {flag} {name:7s} TP {r['true_peak_dbtp']:+.2f} dBTP ({r['true_peak_rise_db']:+.2f} dB), "
              f"peak at {r['peak_time_sec']:.3f} s, overs {r['samples_over_full_scale']}, "
              f"{r['integrated_lufs']} LUFS -> {r['decoded_file']}")
    for name, reason in report["skipped"].items():
        print(f"  [skip] {name}: {reason}")
    print("Listening review pending: audition the decoded files before recording codec approval.")
    raise SystemExit(0 if report["all_within_ceiling"] else 1)


if __name__ == "__main__":
    main()
