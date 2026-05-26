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


def _assemble_track_continuous(
    track: dict,
    session: dict,
    output_dir: Path,
    crossfade_ms: float = 50.0,
    interloper_head_ms: float | None = None,
    interloper_tail_ms: float | None = None,
    cluster_gap_sec: float = 1.0,
    normalize_per_source: bool = False,
    source_target_lufs: float = -18.0,
) -> dict:
    """Continuous-source assembly: bypass per-clip slip-edits within a take.

    For each unique source WAV referenced by the track, cluster the clips by
    timeline proximity. Each cluster becomes ONE placement that plays a
    continuous chunk of the source (covering what the editor used within the
    cluster). Placements are blended into the output with equal-power
    crossfades at edges; interloper placements (short clips landing inside a
    larger placement) get optionally wider head/tail crossfades.

    Eliminates the 100s of slip-edit boundaries within a source take that
    cause comb-filter warble / phase-discontinuity clicks when summed with
    short crossfades. Trade-off: bass timing tracks the player's natural
    playing, not the editor's slip-edit corrections.

    See test_continuous_bass.py for the original prototype + rationale.
    """
    from collections import defaultdict

    sr = session["sample_rate"]
    clips = track["clips"]
    track_name = track["name"]

    if not clips:
        return {"track": track_name, "error": "no clips"}

    cluster_gap_samples = int(cluster_gap_sec * sr)
    max_pad_samples = int(0.2 * sr)  # 200ms upper bound for adaptive pad

    # Group clips by source file, then cluster within source by timeline gap.
    by_source: dict[str, list[dict]] = defaultdict(list)
    for c in clips:
        by_source[c["source_file"]].append(c)

    placements: list[tuple[str, int, int, int]] = []
    for src_file, src_clips in by_source.items():
        cs_sorted = sorted(src_clips, key=lambda c: c["timeline_start_sample"])
        clusters: list[list[dict]] = [[cs_sorted[0]]]
        for c in cs_sorted[1:]:
            prev_end = (clusters[-1][-1]["timeline_start_sample"]
                        + clusters[-1][-1]["length_samples"])
            if c["timeline_start_sample"] - prev_end > cluster_gap_samples:
                clusters.append([])
            clusters[-1].append(c)

        try:
            n_frames = sf.info(src_file).frames
        except Exception as exc:
            print(f"  WARN: skip {src_file} ({exc})", file=sys.stderr)
            continue

        for cluster in clusters:
            anchors = [c["timeline_start_sample"] - c["source_offset_sample"]
                       for c in cluster]
            median_anchor = int(np.median(anchors))
            src_lo = min(c["source_offset_sample"] for c in cluster)
            src_hi = max(c["source_offset_sample"] + c["length_samples"]
                         for c in cluster)
            # Adaptive pad: scales with cluster duration / clip count, capped
            # at 200ms. Short single-clip clusters (interlopers) get minimal
            # pad to avoid playing source material the editor never used.
            cluster_dur = src_hi - src_lo
            cluster_pad = min(max_pad_samples, cluster_dur // 8,
                              len(cluster) * int(0.02 * sr))
            src_lo = max(0, src_lo - cluster_pad)
            src_hi = min(n_frames, src_hi + cluster_pad)
            placements.append((src_file, median_anchor, src_lo, src_hi))

    placements.sort(key=lambda p: p[1] + p[2])

    # Detect interlopers + assign per-placement crossfade widths.
    head_xfade_samples = int(round(
        (interloper_head_ms if interloper_head_ms is not None else crossfade_ms)
        * sr / 1000.0))
    tail_xfade_samples = int(round(
        (interloper_tail_ms if interloper_tail_ms is not None else crossfade_ms)
        * sr / 1000.0))
    default_xfade_samples = int(round(crossfade_ms * sr / 1000.0))

    placements_extended: list[tuple] = []
    for i, (src_file, anchor, src_lo, src_hi) in enumerate(placements):
        tl_start_i = anchor + src_lo
        tl_end_i = anchor + src_hi
        is_interloper = any(
            anchor_o + src_lo_o <= tl_start_i and tl_end_i <= anchor_o + src_hi_o
            for j, (_sf, anchor_o, src_lo_o, src_hi_o) in enumerate(placements)
            if j != i
        )
        if is_interloper and (head_xfade_samples > 0 or tail_xfade_samples > 0):
            try:
                n_frames = sf.info(src_file).frames
            except Exception:
                n_frames = src_hi
            new_src_lo = max(0, src_lo - head_xfade_samples)
            new_src_hi = min(n_frames, src_hi + tail_xfade_samples)
            placements_extended.append((src_file, anchor, new_src_lo, new_src_hi,
                                       True, head_xfade_samples, tail_xfade_samples))
        else:
            placements_extended.append((src_file, anchor, src_lo, src_hi,
                                       False, default_xfade_samples,
                                       default_xfade_samples))

    if not placements_extended:
        return {"track": track_name, "error": "no placements"}

    timeline_end = max(anchor + src_hi for _, anchor, _, src_hi, _, _, _
                       in placements_extended)
    output = np.zeros(timeline_end, dtype=np.float64)

    placement_log: list[dict] = []
    for src_file, anchor, src_lo, src_hi, is_interloper, head_xf, tail_xf in placements_extended:
        try:
            data, _sr = sf.read(src_file, start=src_lo, frames=src_hi - src_lo,
                                always_2d=True)
        except Exception as exc:
            print(f"  WARN: skip read {src_file} ({exc})", file=sys.stderr)
            continue
        if _sr != sr:
            print(f"  WARN: sr mismatch on {src_file} ({_sr} vs {sr})",
                  file=sys.stderr)
            continue
        # Force mono
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]

        tl_start = anchor + src_lo
        tl_end_this = tl_start + len(data)
        tl_start_c = max(0, tl_start)
        tl_end_c = min(timeline_end, tl_end_this)
        data_start = tl_start_c - tl_start
        data_end = data_start + (tl_end_c - tl_start_c)
        data_slice = data[data_start:data_end].copy()
        seg_len = len(data_slice)

        # Per-source normalize: bring each placement to source_target_lufs.
        # Evens out inter-source recording-gain variations (different sections
        # tracked at different levels). Applied BEFORE crossfade so the boundary
        # blends with already-normalized content.
        placement_lufs_before = None
        placement_gain_db = 0.0
        if normalize_per_source and seg_len > sr:  # need at least 1s for LUFS measurement
            try:
                # Make stereo for pyloudnorm (it requires shape (n, 2) or (n,))
                lufs_meas = float(pyln.Meter(sr).integrated_loudness(data_slice))
                if np.isfinite(lufs_meas):
                    placement_lufs_before = lufs_meas
                    placement_gain_db = source_target_lufs - lufs_meas
                    gain_linear = 10.0 ** (placement_gain_db / 20.0)
                    data_slice = data_slice * gain_linear
            except Exception:
                pass

        head_eff = min(head_xf, seg_len // 2)
        tail_eff = min(tail_xf, seg_len // 2)

        existing = output[tl_start_c:tl_end_c]
        head_has = head_eff > 0 and np.any(np.abs(existing[:head_eff]) > 1e-7)
        tail_has = tail_eff > 0 and np.any(np.abs(existing[-tail_eff:]) > 1e-7)

        new_segment = data_slice.copy()
        if head_has:
            fade_in = np.sqrt(np.linspace(0.0, 1.0, head_eff))
            fade_out = np.sqrt(np.linspace(1.0, 0.0, head_eff))
            new_segment[:head_eff] = (existing[:head_eff] * fade_out
                                     + new_segment[:head_eff] * fade_in)
        if tail_has:
            fade_in = np.sqrt(np.linspace(0.0, 1.0, tail_eff))
            fade_out = np.sqrt(np.linspace(1.0, 0.0, tail_eff))
            new_segment[-tail_eff:] = (new_segment[-tail_eff:] * fade_out
                                      + existing[-tail_eff:] * fade_in)

        output[tl_start_c:tl_end_c] = new_segment
        placement_log.append({
            "source": Path(src_file).name,
            "timeline_start_sec": round(tl_start / sr, 3),
            "timeline_end_sec": round(tl_end_this / sr, 3),
            "source_start_sec": round(src_lo / sr, 3),
            "source_end_sec": round(src_hi / sr, 3),
            "is_interloper": is_interloper,
            "head_xfade_ms": round(head_eff * 1000.0 / sr, 1),
            "tail_xfade_ms": round(tail_eff * 1000.0 / sr, 1),
            "lufs_before_norm": round(placement_lufs_before, 2) if placement_lufs_before is not None else None,
            "gain_applied_db": round(placement_gain_db, 2),
        })

    output_2d = output[:, None]
    peak = float(np.max(np.abs(output_2d)))
    if peak > 1.0:
        print(f"WARNING: {track_name}: peak {20*np.log10(peak):.1f} dBFS — "
              f"scaling down to -0.1 dBFS", file=sys.stderr)
        output_2d = output_2d * (0.99 / peak)
        peak = 0.99

    stem_dir = output_dir / track_name
    stem_dir.mkdir(parents=True, exist_ok=True)
    out_path = stem_dir / "assembled.wav"
    sf.write(str(out_path), output_2d, sr, subtype="PCM_24")

    report = {
        "track": track_name,
        "output": str(out_path),
        "mode": "continuous",
        "default_crossfade_ms": crossfade_ms,
        "interloper_head_ms": interloper_head_ms,
        "interloper_tail_ms": interloper_tail_ms,
        "cluster_gap_sec": cluster_gap_sec,
        "duration_sec": round(timeline_end / sr, 3),
        "channels": 1,
        "placements_total": len(placements_extended),
        "placements_interloper": sum(1 for p in placements_extended if p[4]),
        "peak_dbfs": round(20.0 * np.log10(max(peak, 1e-10)), 2),
        "sample_rate": sr,
        "placements": placement_log,
    }
    (stem_dir / "gain_report.json").write_text(json.dumps(report, indent=2),
                                              encoding="utf-8")
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
    source_mode: str = "per-clip",
    interloper_head_ms: float | None = None,
    interloper_tail_ms: float | None = None,
    normalize_per_source: bool = False,
    source_target_lufs: float = -18.0,
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
        if source_mode == "continuous":
            xfade_label = f"default xfade {crossfade_ms} ms"
            if interloper_head_ms is not None or interloper_tail_ms is not None:
                xfade_label += (f", interloper head={interloper_head_ms or crossfade_ms}ms "
                                f"tail={interloper_tail_ms or crossfade_ms}ms")
            norm_label = (f", normalize-per-source target {source_target_lufs} LUFS"
                          if normalize_per_source else "")
            print(
                f"Assembling (continuous mode): {track['name']} "
                f"({len(track['clips'])} clips, {xfade_label}{norm_label})...",
                file=sys.stderr,
            )
            result = _assemble_track_continuous(
                track, data, output_dir,
                crossfade_ms=crossfade_ms,
                interloper_head_ms=interloper_head_ms,
                interloper_tail_ms=interloper_tail_ms,
                normalize_per_source=normalize_per_source,
                source_target_lufs=source_target_lufs,
            )
        else:
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
    clip_group.add_argument(
        "--source-mode", choices=("per-clip", "continuous"), default="per-clip",
        help="Assembly strategy. 'per-clip' (default): assemble session-defined slip-edit "
             "clips with crossfade smoothing. 'continuous': bypass slip-edits — for each "
             "unique source WAV, cluster the clips by timeline proximity, play each "
             "cluster as one continuous chunk at its median timeline anchor. Eliminates "
             "warble/click artifacts on sustained material (bass DI especially) where "
             "100s of slip-edit boundaries fight phase coherence. Trade-off: timing tracks "
             "the player's natural playing, not the editor's slip-edit corrections.",
    )
    clip_group.add_argument(
        "--interloper-head-ms", type=float, default=None, metavar="MS",
        help="continuous mode: custom head crossfade (in ms) for interloper placements "
             "— short clips that land INSIDE a longer placement (chorus doublers, "
             "ornamentations). Defaults to --crossfade-ms. Useful with longer values "
             "(1000-3000ms) to mask the transition between two different takes.",
    )
    clip_group.add_argument(
        "--interloper-tail-ms", type=float, default=None, metavar="MS",
        help="continuous mode: custom tail crossfade (in ms) for interloper placements. "
             "Defaults to --crossfade-ms.",
    )
    clip_group.add_argument(
        "--normalize-per-source", action="store_true",
        help="continuous mode: normalize EACH placement (source-cluster) to "
             "--source-target-lufs before crossfade-blending. Evens out "
             "inter-source recording-gain variation (different sections tracked "
             "at different gain levels by the engineer) without re-introducing "
             "the per-clip-norm warble (only ~10-20 normalization points per "
             "track vs 100s in per-clip mode).",
    )
    clip_group.add_argument(
        "--source-target-lufs", type=float, default=-18.0, metavar="LUFS",
        help="continuous mode: target LUFS for --normalize-per-source (default -18.0).",
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
            source_mode=args.source_mode,
            interloper_head_ms=args.interloper_head_ms,
            interloper_tail_ms=args.interloper_tail_ms,
            normalize_per_source=args.normalize_per_source,
            source_target_lufs=args.source_target_lufs,
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
