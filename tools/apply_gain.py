"""Apply gain to audio stems.

Two modes:

  --per-clip session.json
      Reads the clip layout from session.json, applies clip gain to normalize
      each clip to a consistent LUFS level, then assembles the full-length stem.
      This is the primary gain-staging step — always run this first.
      Output: output_dir/<track_name>/assembled.wav

      Use --no-normalize to assemble clips at their original recording levels
      without any per-clip LUFS normalization. Recommended for drums, which are
      recorded in a single continuous take — use --no-normalize here, then
      apply --per-channel on the assembled result for uniform gain staging.

  --per-channel file.wav
      Applies a single gain to an already-assembled stem to reach a target
      LUFS or preset level. Use for delivery normalization or when receiving
      pre-assembled stems.
      Output: output_dir/<stem_name>_gained.wav

Reads [gain] section from config.toml in the current working directory.

Usage:
  # list available tracks
  python apply_gain.py --per-clip output/<session>/session.json

  # clip gain + assemble one track
  python apply_gain.py --per-clip output/<session>/session.json --track "BASS DI CLEAN"

  # clip gain + assemble all tracks
  python apply_gain.py --per-clip output/<session>/session.json --all

  # per-channel gain staging on assembled stem
  python apply_gain.py --per-channel output/<session>/BASS\\ DI\\ CLEAN/assembled.wav --preset stem
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

try:
    import librosa
    _LIBROSA = True
except ImportError:
    _LIBROSA = False

_CONFIG_PATH = Path("config.toml")

PRESETS: dict[str, dict] = {
    "stem":      {"target_lufs": -18.0, "peak_ceiling_db": -1.0},
    "premix":    {"target_lufs": -18.0, "peak_ceiling_db": -3.0},
    "spotify":   {"target_lufs": -14.0, "peak_ceiling_db": -2.0},
    "apple":     {"target_lufs": -16.0, "peak_ceiling_db": -2.0},
    "amazon":    {"target_lufs": -14.0, "peak_ceiling_db": -2.0},
    "broadcast": {"target_lufs": -23.0, "peak_ceiling_db": -2.0},
}

DEFAULT_PEAK_CEILING = -1.0


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    return {}


def _measure_lufs(data: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    try:
        mono = data[:, 0] if data.shape[1] == 1 else data
        return float(meter.integrated_loudness(mono))
    except Exception:
        return -120.0


def _read_clip(source_file: str, offset_session: int, length_session: int, session_sr: int) -> np.ndarray:
    """Read clip samples from source file, resampling if needed.

    Returns samples ndarray shape (n, channels).
    """
    path = Path(source_file)
    if not path.exists():
        return np.zeros((length_session, 1), dtype=np.float64)

    info = sf.info(str(path))
    file_sr = info.samplerate

    if file_sr == session_sr:
        file_offset = offset_session
        file_frames = length_session
    else:
        file_offset = int(offset_session * file_sr / session_sr)
        file_frames = int(length_session * file_sr / session_sr) + 1

    file_frames = min(file_frames, info.frames - file_offset)
    if file_frames <= 0:
        return np.zeros((length_session, 1), dtype=np.float64)

    data, _ = sf.read(str(path), start=file_offset, frames=file_frames, always_2d=True, dtype="float64")

    if file_sr != session_sr:
        if not _LIBROSA:
            raise RuntimeError("librosa required for resampling — pip install librosa")
        resampled = librosa.resample(data.T.astype(np.float32), orig_sr=file_sr, target_sr=session_sr)
        data = resampled.T.astype(np.float64)

    return data


def _apply_clip_gain(data: np.ndarray, sr: int, target_lufs: float, peak_ceiling_db: float) -> tuple[np.ndarray, float, float, bool]:
    """Normalize clip to target_lufs with peak ceiling protection.

    Returns (gained_data, requested_gain_db, applied_gain_db, was_limited).
    """
    lufs = _measure_lufs(data, sr)
    if lufs <= -100.0:
        return data, 0.0, 0.0, False

    requested_gain_db = target_lufs - lufs
    current_peak_dbfs = 20.0 * np.log10(max(float(np.max(np.abs(data))), 1e-10))
    max_safe_gain_db = peak_ceiling_db - current_peak_dbfs
    limited = requested_gain_db > max_safe_gain_db
    applied_gain_db = min(requested_gain_db, max_safe_gain_db)

    return data * (10 ** (applied_gain_db / 20.0)), requested_gain_db, applied_gain_db, limited


# ---------------------------------------------------------------------------
# Per-clip mode
# ---------------------------------------------------------------------------

_BOUNDARY_CHECK_WINDOW_SEC = 0.5
_BOUNDARY_CHECK_THRESHOLD_DB = 4.0
_BOUNDARY_CHECK_SILENCE_FLOOR_DB = -60.0


def _check_clip_boundaries(output: np.ndarray, sr: int, clips: list) -> list[dict]:
    """Check assembled output for abrupt RMS level changes at clip boundaries.

    Measures RMS in a window before and after each clip start. Skips boundaries
    where either side is essentially silence (e.g. a clip that starts with a fill
    or tom hit). Returns a list of warning dicts for jumps > threshold.
    """
    window = int(_BOUNDARY_CHECK_WINDOW_SEC * sr)
    mono = output.mean(axis=1) if output.ndim == 2 else output
    warnings = []

    for i, clip in enumerate(clips):
        if i == 0:
            continue
        boundary = clip["timeline_start_sample"]
        if boundary <= 0 or boundary >= len(mono):
            continue

        pre = mono[max(0, boundary - window):boundary]
        post = mono[boundary:min(len(mono), boundary + window)]
        if len(pre) == 0 or len(post) == 0:
            continue

        pre_rms = float(np.sqrt(np.mean(pre ** 2)))
        post_rms = float(np.sqrt(np.mean(post ** 2)))
        pre_db = 20.0 * np.log10(max(pre_rms, 1e-10))
        post_db = 20.0 * np.log10(max(post_rms, 1e-10))

        if pre_db < _BOUNDARY_CHECK_SILENCE_FLOOR_DB or post_db < _BOUNDARY_CHECK_SILENCE_FLOOR_DB:
            continue

        diff_db = abs(post_db - pre_db)
        if diff_db > _BOUNDARY_CHECK_THRESHOLD_DB:
            warnings.append({
                "clip_index": i,
                "boundary_sec": round(boundary / sr, 3),
                "rms_before_db": round(pre_db, 1),
                "rms_after_db": round(post_db, 1),
                "diff_db": round(diff_db, 1),
            })

    return warnings


def _assemble_track_with_clip_gain(
    track: dict,
    session: dict,
    output_dir: Path,
    target_lufs: float,
    peak_ceiling_db: float,
    normalize: bool = True,
    crossfade_ms: float = 5.0,
) -> dict:
    sr = session["sample_rate"]
    duration = session["duration_samples"]
    clips = track["clips"]
    track_name = track["name"]

    if not clips:
        return {"track": track_name, "error": "no clips"}

    n_channels = 1
    for clip in clips:
        p = Path(clip["source_file"])
        if p.exists():
            n_channels = sf.info(str(p)).channels
            break

    output = np.zeros((duration, n_channels), dtype=np.float64)

    assembled = 0
    skipped = 0
    clips_limited = 0

    # Constant-power crossfade envelopes for butt-up clip boundaries (no overlap
    # in the source, just adjacent timeline positions). The DAW (Pro Tools/Logic/
    # etc.) applies these by default — without them, slip-edit boundaries leave
    # a sample-level discontinuity that the ear hears as a click.
    xfade_samples = max(0, int(round(crossfade_ms * sr / 1000.0)))
    if xfade_samples > 0:
        _xf_t = np.linspace(0.0, 1.0, xfade_samples)
        _fade_in_env = np.sin(np.pi / 2.0 * _xf_t)
        _fade_out_env = np.cos(np.pi / 2.0 * _xf_t)
    else:
        _fade_in_env = None
        _fade_out_env = None

    # Sort clips by timeline position so we can detect butt-up boundaries
    # between adjacent clips (different from overlapping clips, which are
    # already handled by the additive write below).
    clips_sorted = sorted(clips, key=lambda c: c["timeline_start_sample"])
    crossfades_applied = 0

    # Tolerance ~ one crossfade length: clips that "almost" butt up (1-2 ms
    # of small overlap or gap from engineer slip-edits at the cut point) still
    # benefit from a crossfade. With a stricter tolerance, the very clicks
    # we're trying to smooth slip through.
    butt_up_tolerance = max(1, xfade_samples)

    def _butts_up(prev_clip, this_clip) -> bool:
        prev_end = prev_clip["timeline_start_sample"] + prev_clip["length_samples"]
        this_start = this_clip["timeline_start_sample"]
        return abs(prev_end - this_start) <= butt_up_tolerance

    for i, clip in enumerate(clips_sorted):
        has_prev_boundary = (xfade_samples > 0 and i > 0
                             and _butts_up(clips_sorted[i - 1], clip))
        has_next_boundary = (xfade_samples > 0 and i + 1 < len(clips_sorted)
                             and _butts_up(clip, clips_sorted[i + 1]))

        # Read extra samples past the clip end if there's a butt-up neighbor —
        # the extension supplies the fade-out tail. If the source doesn't have
        # enough samples past the original end, _read_clip returns fewer.
        extra_request = xfade_samples if has_next_boundary else 0
        data = _read_clip(
            clip["source_file"],
            clip["source_offset_sample"],
            clip["length_samples"] + extra_request,
            sr,
        )

        if data.shape[1] != n_channels:
            if data.shape[1] == 1 and n_channels > 1:
                data = np.repeat(data, n_channels, axis=1)
            elif data.shape[1] > 1 and n_channels == 1:
                data = data.mean(axis=1, keepdims=True)

        if normalize:
            data, _, _, limited = _apply_clip_gain(data, sr, target_lufs, peak_ceiling_db)
            if limited:
                clips_limited += 1

        # Crossfade: fade-in the first xfade_samples (if previous neighbour
        # exists) and fade-out the last xfade_samples (if next neighbour exists).
        # The two faded envelopes from adjacent clips sum to constant power in
        # the overlap window.
        if has_prev_boundary and data.shape[0] >= xfade_samples:
            data[:xfade_samples] = data[:xfade_samples] * _fade_in_env[:, None]
            crossfades_applied += 1
        if has_next_boundary and data.shape[0] >= xfade_samples:
            n_tail = min(xfade_samples, data.shape[0])
            data[-n_tail:] = data[-n_tail:] * _fade_out_env[-n_tail:][:, None]

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

    peak = float(np.max(np.abs(output)))
    if peak > 1.0:
        print(
            f"WARNING: {track_name}: peak {20*np.log10(peak):.1f} dBFS after assembly "
            f"(overlapping clips?) — not auto-normalizing",
            file=sys.stderr,
        )

    stem_dir = output_dir / track_name
    stem_dir.mkdir(parents=True, exist_ok=True)
    out_path = stem_dir / "assembled.wav"
    sf.write(str(out_path), output, sr, subtype="PCM_24")

    boundary_warnings = _check_clip_boundaries(output, sr, clips)
    for w in boundary_warnings:
        print(
            f"WARNING: {track_name}: level jump at clip boundary "
            f"{w['boundary_sec']:.2f}s — "
            f"before={w['rms_before_db']:+.1f} dB, after={w['rms_after_db']:+.1f} dB, "
            f"diff={w['diff_db']:.1f} dB. "
            f"Consider --no-normalize if this is a continuous recording.",
            file=sys.stderr,
        )

    report = {
        "track": track_name,
        "output": str(out_path),
        "mode": "per-clip" if normalize else "per-clip-no-normalize",
        "clip_gain_target_lufs": target_lufs if normalize else None,
        "peak_ceiling_db": peak_ceiling_db,
        "duration_sec": round(duration / sr, 2),
        "channels": n_channels,
        "clips_assembled": assembled,
        "clips_skipped": skipped,
        "clips_peak_limited": clips_limited,
        "peak_dbfs": round(20.0 * np.log10(max(peak, 1e-10)), 1),
        "sample_rate": sr,
        "boundary_warnings": boundary_warnings,
    }

    report_path = stem_dir / "gain_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def apply_gain_per_clip(
    session_json: Path,
    output_dir: Path,
    track_names: list[str] | None = None,
    all_tracks: bool = False,
    target_lufs: float | None = None,
    peak_ceiling_db: float = DEFAULT_PEAK_CEILING,
    normalize: bool = True,
    crossfade_ms: float = 5.0,
) -> list[dict]:
    cfg = _load_config().get("gain", {})
    if target_lufs is None:
        target_lufs = cfg.get("per_clip_target_lufs", -18.0)

    data = json.loads(session_json.read_text(encoding="utf-8"))
    tracks = data["tracks"]

    if not all_tracks and not track_names:
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
        mode_label = f"target {target_lufs} LUFS" if normalize else "no normalize (original levels)"
        xfade_label = f"crossfade {crossfade_ms} ms" if crossfade_ms > 0 else "no crossfade"
        print(
            f"Assembling: {track['name']} ({len(track['clips'])} clips, {mode_label}, {xfade_label})...",
            file=sys.stderr,
        )
        result = _assemble_track_with_clip_gain(
            track, data, output_dir, target_lufs, peak_ceiling_db,
            normalize=normalize, crossfade_ms=crossfade_ms,
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    return results


# ---------------------------------------------------------------------------
# Per-channel mode
# ---------------------------------------------------------------------------

def apply_gain_per_channel(
    file_path: Path,
    output_dir: Path,
    gain_db: float | None = None,
    analysis_json: Path | None = None,
    target_lufs: float = -18.0,
    peak_ceiling_db: float = DEFAULT_PEAK_CEILING,
) -> dict:
    data, sr = sf.read(str(file_path), always_2d=True)

    if gain_db is not None:
        requested_gain_db = gain_db
        source = "explicit"
    elif analysis_json is not None:
        info = json.loads(analysis_json.read_text(encoding="utf-8"))
        requested_gain_db = float(info["recommended_gain_db"])
        source = "analysis.json (recommended_gain_db)"
    else:
        lufs_before = _measure_lufs(data, sr)
        requested_gain_db = round(target_lufs - lufs_before, 1) if lufs_before > -100 else 0.0
        source = f"auto (target {target_lufs} LUFS)"

    current_peak_dbfs = 20.0 * np.log10(max(float(np.max(np.abs(data))), 1e-10))
    max_safe_gain_db = peak_ceiling_db - current_peak_dbfs
    peak_ceiling_limited = bool(requested_gain_db > max_safe_gain_db)
    applied_gain_db = min(requested_gain_db, max_safe_gain_db)

    if peak_ceiling_limited:
        print(
            f"INFO: peak ceiling: requested {requested_gain_db:+.1f} dB limited to "
            f"{applied_gain_db:+.1f} dB (ceiling {peak_ceiling_db:.1f} dBFS, "
            f"current peak {current_peak_dbfs:.1f} dBFS)",
            file=sys.stderr,
        )

    processed = data * (10 ** (applied_gain_db / 20.0))

    peak_after = float(np.max(np.abs(processed)))
    if peak_after > 1.0:
        print(
            f"WARNING: clipping by {20*np.log10(peak_after):.1f} dB — ceiling protection may have failed",
            file=sys.stderr,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{file_path.stem}_gained.wav"
    sf.write(str(out_path), processed, sr, subtype="PCM_24")

    lufs_after = _measure_lufs(processed, sr)

    result = {
        "file": str(file_path),
        "output": str(out_path),
        "mode": "per-channel",
        "gain_requested_db": round(requested_gain_db, 1),
        "gain_applied_db": round(applied_gain_db, 1),
        "gain_source": source,
        "peak_ceiling_db": peak_ceiling_db,
        "peak_ceiling_limited": peak_ceiling_limited,
        "lufs_after": round(lufs_after, 1),
        "true_peak_after_dbfs": round(20.0 * np.log10(max(peak_after, 1e-10)), 1),
    }

    report_path = output_dir / f"{file_path.stem}_gain_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply gain to audio stems. Two modes: --per-clip (clip gain + assembly) or --per-channel (stem-level gain).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--per-clip", metavar="SESSION_JSON", type=Path,
        help="session.json from parse_session.py — apply clip gain and assemble",
    )
    mode_group.add_argument(
        "--per-channel", metavar="AUDIO_FILE", type=Path,
        help="Assembled stem WAV — apply single gain to reach target LUFS",
    )

    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--peak-ceiling", type=float, default=DEFAULT_PEAK_CEILING, metavar="DBFS",
        help=f"Peak ceiling in dBFS (default: {DEFAULT_PEAK_CEILING})",
    )

    # per-clip options
    clip_group = parser.add_argument_group("per-clip options")
    clip_group.add_argument(
        "--track", action="append", dest="tracks", metavar="TRACK_NAME",
        help="Track name to assemble (repeat for multiple; omit to list tracks)",
    )
    clip_group.add_argument(
        "--all", action="store_true", dest="all_tracks",
        help="Assemble all tracks",
    )
    clip_group.add_argument(
        "--clip-target-lufs", type=float, default=None, metavar="LUFS",
        help="Per-clip LUFS target (default: from config.toml [gain] per_clip_target_lufs, fallback -18.0)",
    )
    clip_group.add_argument(
        "--no-normalize", action="store_true", dest="no_normalize",
        help="Assemble clips at original recording levels without per-clip LUFS normalization. "
             "Use for drums (single continuous take); follow up with --per-channel for uniform gain.",
    )
    clip_group.add_argument(
        "--crossfade-ms", type=float, default=5.0, metavar="MS",
        help="Crossfade length in ms at butt-up clip boundaries (default: 5.0 ms — matches "
             "Pro Tools / Logic default). Smooths source-discontinuity clicks at engineer "
             "slip-edits. Set to 0 to disable.",
    )

    # per-channel options
    chan_group = parser.add_argument_group("per-channel options")
    gain_source = chan_group.add_mutually_exclusive_group()
    gain_source.add_argument("--gain-db", type=float, help="Explicit gain in dB")
    gain_source.add_argument(
        "--from-analysis", type=Path, metavar="ANALYSIS_JSON",
        help="Use recommended_gain_db from an analysis.json",
    )
    preset_help = ", ".join(
        f"{k} ({v['target_lufs']} LUFS / {v['peak_ceiling_db']} dBFS)" for k, v in PRESETS.items()
    )
    gain_source.add_argument("--preset", choices=list(PRESETS), help=preset_help)
    gain_source.add_argument("--target-lufs", type=float, help="Auto-compute gain to reach this LUFS")

    args = parser.parse_args()

    if args.per_clip:
        if not args.per_clip.exists():
            print(json.dumps({"error": f"Not found: {args.per_clip}"}), file=sys.stderr)
            sys.exit(1)
        apply_gain_per_clip(
            session_json=args.per_clip,
            output_dir=args.output_dir,
            track_names=args.tracks,
            all_tracks=args.all_tracks,
            target_lufs=args.clip_target_lufs,
            peak_ceiling_db=args.peak_ceiling,
            normalize=not args.no_normalize,
            crossfade_ms=args.crossfade_ms,
        )

    elif args.per_channel:
        if not args.per_channel.exists():
            print(json.dumps({"error": f"Not found: {args.per_channel}"}), file=sys.stderr)
            sys.exit(1)
        if args.from_analysis and not args.from_analysis.exists():
            print(json.dumps({"error": f"Not found: {args.from_analysis}"}), file=sys.stderr)
            sys.exit(1)

        cfg = _load_config().get("gain", {})
        target_lufs = -18.0
        peak_ceiling = args.peak_ceiling

        if args.preset:
            p = PRESETS[args.preset]
            target_lufs = p["target_lufs"]
            peak_ceiling = p["peak_ceiling_db"]
        elif args.target_lufs:
            target_lufs = args.target_lufs
        else:
            default_preset = cfg.get("per_channel_preset", "stem")
            if default_preset in PRESETS:
                p = PRESETS[default_preset]
                target_lufs = p["target_lufs"]
                peak_ceiling = p["peak_ceiling_db"]

        result = apply_gain_per_channel(
            file_path=args.per_channel,
            output_dir=args.output_dir,
            gain_db=args.gain_db,
            analysis_json=args.from_analysis,
            target_lufs=target_lufs,
            peak_ceiling_db=peak_ceiling,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
