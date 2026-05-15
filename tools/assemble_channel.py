"""Assemble a full-length channel WAV from clip layout in session.json.

Reads the canonical session.json produced by parse_session.py.
For each requested track, places every clip at the correct timeline
position (with silence in the gaps) and writes a single assembled.wav.

Usage:
  # list available tracks
  python assemble_channel.py output/terido/session.json

  # assemble one track
  python assemble_channel.py output/terido/session.json --track "BASS DI CLEAN"

  # assemble all tracks
  python assemble_channel.py output/terido/session.json --all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import librosa
    _LIBROSA = True
except ImportError:
    _LIBROSA = False


def _read_clip(source_file: str, offset_session: int, length_session: int, session_sr: int) -> tuple[np.ndarray, int]:
    """Read clip samples from source file, resampling if needed.

    Returns (samples ndarray shape (n, channels), actual_sr).
    """
    path = Path(source_file)
    if not path.exists():
        return np.zeros((length_session, 1), dtype=np.float64), session_sr

    info = sf.info(str(path))
    file_sr = info.samplerate

    if file_sr == session_sr:
        file_offset = offset_session
        file_frames = length_session
    else:
        # scale offset/length to file's native sample rate
        file_offset = int(offset_session * file_sr / session_sr)
        file_frames = int(length_session * file_sr / session_sr) + 1

    file_frames = min(file_frames, info.frames - file_offset)
    if file_frames <= 0:
        return np.zeros((length_session, 1), dtype=np.float64), session_sr

    data, _ = sf.read(str(path), start=file_offset, frames=file_frames, always_2d=True, dtype="float64")

    if file_sr != session_sr:
        if not _LIBROSA:
            raise RuntimeError("librosa required for resampling — pip install librosa")
        # librosa.resample expects (channels, samples)
        resampled = librosa.resample(data.T.astype(np.float32), orig_sr=file_sr, target_sr=session_sr)
        data = resampled.T.astype(np.float64)

    return data, session_sr


def _assemble_track(track: dict, session: dict, output_dir: Path) -> dict:
    sr = session["sample_rate"]
    duration = session["duration_samples"]
    clips = track["clips"]
    track_name = track["name"]

    if not clips:
        return {"track": track_name, "error": "no clips"}

    # determine channel count from first readable clip
    n_channels = 1
    for clip in clips:
        p = Path(clip["source_file"])
        if p.exists():
            n_channels = sf.info(str(p)).channels
            break

    output = np.zeros((duration, n_channels), dtype=np.float64)

    assembled = 0
    skipped = 0
    for clip in clips:
        data, _ = _read_clip(
            clip["source_file"],
            clip["source_offset_sample"],
            clip["length_samples"],
            sr,
        )

        if data.shape[1] != n_channels:
            if data.shape[1] == 1 and n_channels > 1:
                data = np.repeat(data, n_channels, axis=1)
            elif data.shape[1] > 1 and n_channels == 1:
                data = data.mean(axis=1, keepdims=True)

        tl_start = clip["timeline_start_sample"]
        tl_end = tl_start + len(data)

        if tl_start >= duration:
            skipped += 1
            continue

        if tl_end > duration:
            data = data[:duration - tl_start]
            tl_end = duration

        output[tl_start:tl_end] += data
        assembled += 1

    # warn on clip if overlap pushed levels over 0 dBFS
    peak = float(np.max(np.abs(output)))
    clipped = peak > 1.0
    if clipped:
        print(
            f"WARNING: {track_name}: peak {20*np.log10(peak):.1f} dBFS after assembly "
            f"(overlapping clips?) — not auto-normalizing",
            file=sys.stderr,
        )

    stem_dir = output_dir / track_name
    stem_dir.mkdir(parents=True, exist_ok=True)
    out_path = stem_dir / "assembled.wav"
    sf.write(str(out_path), output, sr, subtype="PCM_24")

    report = {
        "track": track_name,
        "output": str(out_path),
        "duration_sec": round(duration / sr, 2),
        "channels": n_channels,
        "clips_assembled": assembled,
        "clips_skipped": skipped,
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-10)), 1),
        "sample_rate": sr,
    }

    report_path = stem_dir / "assemble_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def assemble(
    session_json: Path,
    output_dir: Path,
    track_names: list[str] | None = None,
    all_tracks: bool = False,
) -> list[dict]:
    data = json.loads(session_json.read_text(encoding="utf-8"))
    tracks = data["tracks"]

    if not all_tracks and not track_names:
        # just list tracks
        print("Available tracks:")
        for t in tracks:
            print(f"  {len(t['clips']):4d} clips  {t['name']}")
        return []

    if track_names:
        selected = [t for t in tracks if t["name"] in track_names]
        missing = set(track_names) - {t["name"] for t in selected}
        if missing:
            print(f"WARNING: tracks not found: {missing}", file=sys.stderr)
    else:
        selected = tracks

    results = []
    for track in selected:
        print(f"Assembling: {track['name']} ({len(track['clips'])} clips)...", file=sys.stderr)
        result = _assemble_track(track, data, output_dir)
        results.append(result)
        print(json.dumps(result, indent=2))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble channel WAVs from session.json clip layout.")
    parser.add_argument("session_json", type=Path, help="session.json from parse_session.py")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--track", action="append", dest="tracks", metavar="TRACK_NAME",
        help="Track name to assemble (repeat for multiple)",
    )
    parser.add_argument(
        "--all", action="store_true", dest="all_tracks",
        help="Assemble all tracks",
    )
    args = parser.parse_args()

    if not args.session_json.exists():
        print(json.dumps({"error": f"Not found: {args.session_json}"}), file=sys.stderr)
        sys.exit(1)

    assemble(
        session_json=args.session_json,
        output_dir=args.output_dir,
        track_names=args.tracks,
        all_tracks=args.all_tracks,
    )


if __name__ == "__main__":
    main()
