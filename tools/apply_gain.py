"""Apply gain to an audio stem.

Gain source (in order of precedence):
  1. --gain-db  explicit dB value
  2. --from-analysis  path to an analysis.json — uses recommended_gain_db
  3. --target-lufs  compute gain to reach this LUFS (default: -23)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

DEFAULT_TARGET_LUFS = -23.0


def _measure_lufs(data: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    try:
        return float(meter.integrated_loudness(data))
    except Exception:
        return -120.0


def apply_gain(
    file_path: Path,
    output_dir: Path,
    gain_db: float | None = None,
    analysis_json: Path | None = None,
    target_lufs: float = DEFAULT_TARGET_LUFS,
) -> dict:
    data, sr = sf.read(str(file_path), always_2d=True)

    # determine gain
    if gain_db is not None:
        applied_gain_db = gain_db
        source = "explicit"
    elif analysis_json is not None:
        info = json.loads(analysis_json.read_text(encoding="utf-8"))
        applied_gain_db = float(info["recommended_gain_db"])
        source = f"analysis.json (recommended_gain_db)"
    else:
        lufs_before = _measure_lufs(data if data.shape[1] > 1 else data[:, 0], sr)
        applied_gain_db = round(target_lufs - lufs_before, 1) if lufs_before > -100 else 0.0
        source = f"auto (target {target_lufs} LUFS)"

    linear_gain = 10 ** (applied_gain_db / 20.0)
    processed = data * linear_gain

    # clip guard
    peak = float(np.max(np.abs(processed)))
    if peak > 1.0:
        clip_db = round(20 * np.log10(peak), 1)
        print(
            f"WARNING: clipping by {clip_db} dB after gain — consider reducing gain",
            file=sys.stderr,
        )

    stem_dir = output_dir / file_path.stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    out_path = stem_dir / f"{file_path.stem}_gained.wav"
    sf.write(str(out_path), processed, sr, subtype="PCM_24")

    lufs_after = _measure_lufs(processed if processed.shape[1] > 1 else processed[:, 0], sr)
    true_peak_after = round(20 * np.log10(max(float(np.max(np.abs(processed))), 1e-10)), 1)

    result = {
        "file": str(file_path),
        "output": str(out_path),
        "gain_applied_db": round(applied_gain_db, 1),
        "gain_source": source,
        "lufs_after": round(lufs_after, 1),
        "true_peak_after_dbfs": true_peak_after,
    }

    report_path = stem_dir / "gain_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply gain to an audio stem.")
    parser.add_argument("file", type=Path, help="Input audio file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: ./output)",
    )

    gain_group = parser.add_mutually_exclusive_group()
    gain_group.add_argument("--gain-db", type=float, help="Explicit gain in dB")
    gain_group.add_argument(
        "--from-analysis",
        type=Path,
        metavar="ANALYSIS_JSON",
        help="Use recommended_gain_db from an analysis.json",
    )
    gain_group.add_argument(
        "--target-lufs",
        type=float,
        default=DEFAULT_TARGET_LUFS,
        help=f"Auto-compute gain to reach this LUFS (default: {DEFAULT_TARGET_LUFS})",
    )

    args = parser.parse_args()

    if not args.file.exists():
        print(json.dumps({"error": f"File not found: {args.file}"}), file=sys.stderr)
        sys.exit(1)

    if args.from_analysis and not args.from_analysis.exists():
        print(json.dumps({"error": f"analysis.json not found: {args.from_analysis}"}), file=sys.stderr)
        sys.exit(1)

    result = apply_gain(
        file_path=args.file,
        output_dir=args.output_dir,
        gain_db=args.gain_db,
        analysis_json=args.from_analysis,
        target_lufs=args.target_lufs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
