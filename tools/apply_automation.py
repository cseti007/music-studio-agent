"""Apply explicit, channel-linked volume rides without changing edit timing."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from _recall import record_operation


@record_operation("apply_automation")
def apply_automation(input_path: Path, output_dir: Path, points: list[list[float]]) -> dict:
    """Interpolate [seconds, gain_db] points; hold endpoint gains outside them."""
    curve = np.asarray(points, dtype=float)
    if (curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2
            or not np.isfinite(curve).all() or np.any(np.diff(curve[:, 0]) <= 0)
            or curve[0, 0] < 0 or np.any(np.abs(curve[:, 1]) > 120)):
        raise ValueError("Supply at least two finite, strictly time-ordered gain points")
    data, sr = sf.read(input_path, always_2d=True, dtype="float32")
    if not data.size or data.shape[1] not in (1, 2) or not np.isfinite(data).all():
        raise ValueError("Expected nonempty finite mono or stereo audio")
    if curve[-1, 0] > len(data) / sr:
        raise ValueError("Automation extends beyond the input duration")
    for start in range(0, len(data), 65536):
        stop = min(start + 65536, len(data))
        db = np.interp(np.arange(start, stop) / sr, curve[:, 0], curve[:, 1])
        data[start:stop] *= (10 ** (db / 20))[:, None]
    if not np.isfinite(data).all():
        raise ValueError("Automation produced nonfinite audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{input_path.stem}_automated.wav"
    sf.write(output, data, sr, subtype="FLOAT")
    report = {"input": str(input_path), "output": str(output), "points": points,
              "sample_rate": sr, "channels": data.shape[1], "frames": len(data)}
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True, help="JSON list of [seconds, gain_db] pairs")
    args = parser.parse_args()
    print(json.dumps(apply_automation(args.input, args.output_dir,
                                     json.loads(args.points.read_text())), indent=2))
