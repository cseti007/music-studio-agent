"""Export level-matched excerpts using linear gain only; never approve their sound."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from _dsp import worst_channel_true_peak_dbfs
from _recall import content_hash


def prepare_audition(before: Path, after: Path, output_dir: Path,
                     start: float = 0.0, duration: float = 30.0,
                     after_start: float | None = None, target_lufs: float = -20.0,
                     tp_ceiling: float = -3.0) -> dict:
    after_start = start if after_start is None else after_start
    if not all(np.isfinite(v) for v in (start, after_start, duration, target_lufs, tp_ceiling)):
        raise ValueError("Audition settings must be finite")
    if min(start, after_start) < 0 or duration < 0.4 or tp_ceiling > 0:
        raise ValueError("Use nonnegative starts, duration >= 0.4 seconds and ceiling <= 0 dBTP")
    outputs = [output_dir / "before.wav", output_dir / "after.wav", output_dir / "audition.json"]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Use a new audition directory to preserve earlier comparisons")
    buffers, records = [], []
    for source, offset in ((before, start), (after, after_start)):
        info = sf.info(source)
        frames = round(duration * info.samplerate)
        first = round(offset * info.samplerate)
        if info.channels not in (1, 2) or first + frames > info.frames:
            raise ValueError("Excerpt must fit inside a mono/stereo source")
        data, sr = sf.read(source, start=first, frames=frames, always_2d=True)
        if not np.isfinite(data).all():
            raise ValueError("Excerpt contains nonfinite samples")
        loudness = float(pyln.Meter(sr).integrated_loudness(data))
        if not np.isfinite(loudness):
            raise ValueError("Excerpt is silent or below the loudness gate; choose another section")
        data *= 10 ** ((target_lufs - loudness) / 20)
        buffers.append(data)
        records.append({"source": str(source.resolve()), "source_hash": content_hash(source),
                        "start_frame": first, "frames": frames, "sample_rate": sr,
                        "channels": info.channels, "input_lufs": loudness,
                        "gain_db": target_lufs - loudness})
    if (records[0]["sample_rate"], records[0]["channels"]) != (records[1]["sample_rate"], records[1]["channels"]):
        raise ValueError("Prepare sources with matching sample rate and channel count first")
    highest_peak = max(worst_channel_true_peak_dbfs(data.T) for data in buffers)
    common_trim = min(0.0, tp_ceiling - 0.01 - highest_peak)
    output_dir.mkdir(parents=True, exist_ok=True)
    for data, record, path in zip(buffers, records, outputs):
        data *= 10 ** (common_trim / 20)
        sf.write(path, data, record["sample_rate"], subtype="PCM_24")
        delivered, sr = sf.read(path, always_2d=True)
        record.update(output=str(path.resolve()), output_hash=content_hash(path),
                      gain_db=record["gain_db"] + common_trim,
                      output_lufs=float(pyln.Meter(sr).integrated_loudness(delivered)),
                      output_true_peak_dbtp=worst_channel_true_peak_dbfs(delivered.T))
    report = {"assessment_scope": "Audition preparation only; no listening was performed",
              "processing": "Linear gain only; common attenuation preserves the loudness match",
              "alignment": "Explicit source offsets; no automatic time alignment",
              "common_trim_db": common_trim, "excerpts": records,
              "lufs_difference": records[1]["output_lufs"] - records[0]["output_lufs"]}
    outputs[2].write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--after-start", type=float)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--target-lufs", type=float, default=-20.0)
    parser.add_argument("--tp-ceiling", type=float, default=-3.0)
    args = parser.parse_args()
    print(json.dumps(prepare_audition(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
