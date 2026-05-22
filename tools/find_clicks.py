"""find_clicks.py — locate clicks / sharp transients in a session's audio chain.

Given a mix output (`output/<session>/mixes/mix.wav`), scan it for inter-sample
steps that are sharper than normal musical content (the click signature: a
single-sample jump of >= ~0.1 magnitude, which corresponds to a >12 kHz
spectral edge that the ear hears as a tick).

For each candidate click, optionally walk back through the chain to identify
WHERE the click first appears: source recording → assembled stem →
per-bus stem → mix. This lets you attribute the artifact to its actual
origin (engineer's slip-edit, polarity inversion, comp ceiling clipping, or
a real recording problem) instead of guessing.

Two modes:
  Sweep mode (default):
    find_clicks.py output/<session>
    -> Scans the whole mix.wav, lists all click candidates, ranked by step magnitude.

  Trace mode:
    find_clicks.py output/<session> --time 72.3
    -> Walks every chain stage at the given timestamp (±2 sec window by default)
       and reports which file/stage introduces the discontinuity first.

The "step magnitude" the tool measures is `abs(sample[n+1] - sample[n])`, i.e.
the first-order difference in the mono mix. Magnitudes ≥ 0.10 are uncommon for
normal music transients (which spread their attack across many samples) and
typically indicate either a digital glitch, a clip-boundary discontinuity, or
heavily-limited material.

Output: machine-readable JSON to stdout + a human-readable summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


# Detection thresholds. The big-step threshold matches what's typically audible
# as a click on bass / kick / snare content; tune via CLI if needed.
DEFAULT_STEP_THRESHOLD = 0.10           # |Δ sample| for a click candidate
DEFAULT_TRACE_WINDOW_S = 2.0            # ±this seconds around --time
DEFAULT_CLUSTER_GAP_MS = 50.0           # collapse multiple steps within X ms into one event


def _inter_sample_diffs(mono: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(mono))


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read WAV → mono float64 + sample rate. Stereo collapsed via L+R / 2."""
    data, sr = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return mono.astype(np.float64), sr


def find_click_candidates(
    mix_path: Path,
    threshold: float = DEFAULT_STEP_THRESHOLD,
    cluster_gap_ms: float = DEFAULT_CLUSTER_GAP_MS,
) -> list[dict]:
    """Return a list of click candidate events found in mix_path.

    Each event = {time_s, step_magnitude, step_dbfs, cluster_size}.
    Adjacent clicks within `cluster_gap_ms` are collapsed to a single event
    using their loudest step. This avoids reporting 20 entries for one
    drum hit with a bright attack.
    """
    mono, sr = _load_mono(mix_path)
    diffs = _inter_sample_diffs(mono)
    candidate_idx = np.where(diffs >= threshold)[0]
    if candidate_idx.size == 0:
        return []

    cluster_gap_samples = max(1, int(cluster_gap_ms * sr / 1000.0))

    events: list[dict] = []
    cur_cluster = [int(candidate_idx[0])]
    for idx in candidate_idx[1:]:
        if idx - cur_cluster[-1] <= cluster_gap_samples:
            cur_cluster.append(int(idx))
        else:
            best = max(cur_cluster, key=lambda i: diffs[i])
            events.append({
                "time_s": round(float(best) / sr, 4),
                "step_magnitude": round(float(diffs[best]), 5),
                "step_dbfs": round(20.0 * np.log10(float(diffs[best]) + 1e-12), 2),
                "cluster_size": len(cur_cluster),
            })
            cur_cluster = [int(idx)]
    # Final cluster
    best = max(cur_cluster, key=lambda i: diffs[i])
    events.append({
        "time_s": round(float(best) / sr, 4),
        "step_magnitude": round(float(diffs[best]), 5),
        "step_dbfs": round(20.0 * np.log10(float(diffs[best]) + 1e-12), 2),
        "cluster_size": len(cur_cluster),
    })

    events.sort(key=lambda e: -e["step_magnitude"])
    return events


def _scan_window(mono: np.ndarray, sr: int, t_lo: float, t_hi: float) -> dict:
    lo = max(0, int(t_lo * sr))
    hi = min(len(mono), int(t_hi * sr))
    if hi <= lo:
        return {"peak_db": None, "max_step": None, "max_step_db": None, "max_step_t": None}
    seg = mono[lo:hi]
    if seg.size < 2:
        return {"peak_db": None, "max_step": None, "max_step_db": None, "max_step_t": None}
    abs_seg = np.abs(seg)
    diffs = _inter_sample_diffs(seg)
    max_step_idx = int(np.argmax(diffs))
    return {
        "peak_db": round(20.0 * np.log10(float(np.max(abs_seg)) + 1e-12), 2),
        "max_step": round(float(np.max(diffs)), 5),
        "max_step_db": round(20.0 * np.log10(float(np.max(diffs)) + 1e-12), 2),
        "max_step_t": round((lo + max_step_idx) / sr, 4),
    }


def _find_chain_files_for_track(session_dir: Path, track_name: str) -> list[tuple[str, Path]]:
    """Locate the assembled / processed files for a track in chain order.

    Returns [(stage_label, path), ...] where stage_label is e.g. "assembled",
    "assembled_eq", "assembled_eq_comp", "assembled_eq_comp_aligned".
    Only returns paths that actually exist on disk.
    """
    track_dir = session_dir / "tracks" / track_name
    if not track_dir.is_dir():
        return []
    candidates = [
        "assembled.wav",
        "assembled_eq.wav",
        "assembled_eq_comp.wav",
        "assembled_eq_comp_aligned.wav",
        "assembled_aligned.wav",
        "assembled_eq_comp_deessed.wav",
        "assembled_eq_comp_deessed_eq.wav",
    ]
    out = []
    for fname in candidates:
        p = track_dir / fname
        if p.exists():
            out.append((p.stem, p))
    return out


def trace_at_time(
    session_dir: Path,
    target_time_s: float,
    window_s: float = DEFAULT_TRACE_WINDOW_S,
) -> dict:
    """Walk the chain at a specific time. Report max inter-sample step at
    each stage so the user can see where the click first appears.

    Returns a dict with per-track per-stage measurements and per-stem +
    per-mix measurements at the target time.
    """
    t_lo = target_time_s - window_s / 2.0
    t_hi = target_time_s + window_s / 2.0

    report: dict = {
        "session_dir": str(session_dir),
        "target_time_s": target_time_s,
        "window_s": window_s,
        "tracks": [],
        "stems": [],
        "mix": None,
    }

    # Per-track chain (only ACTIVE tracks per mix_config)
    mix_cfg_path = session_dir / "mix_config.json"
    active_tracks: list[str] = []
    if mix_cfg_path.exists():
        cfg = json.loads(mix_cfg_path.read_text(encoding="utf-8"))
        active_tracks = [t["name"] for t in cfg.get("tracks", []) if t.get("active", True)]

    for track_name in active_tracks:
        chain_files = _find_chain_files_for_track(session_dir, track_name)
        if not chain_files:
            continue
        track_entry: dict = {"name": track_name, "chain": []}
        any_hot = False
        for stage_label, path in chain_files:
            try:
                mono, sr = _load_mono(path)
            except Exception as e:
                track_entry["chain"].append({"stage": stage_label, "error": str(e)})
                continue
            m = _scan_window(mono, sr, t_lo, t_hi)
            track_entry["chain"].append({"stage": stage_label, **m})
            if m["max_step"] is not None and m["max_step"] >= DEFAULT_STEP_THRESHOLD:
                any_hot = True
        # Only include tracks with any hot chain stage (reduces noise)
        if any_hot:
            report["tracks"].append(track_entry)

    # Per-bus stems
    stems_dir = session_dir / "stems"
    if stems_dir.is_dir():
        for p in sorted(stems_dir.glob("stem_*.wav")):
            try:
                mono, sr = _load_mono(p)
            except Exception as e:
                report["stems"].append({"file": p.name, "error": str(e)})
                continue
            m = _scan_window(mono, sr, t_lo, t_hi)
            report["stems"].append({"file": p.name, **m})

    # Mix
    mix_p = session_dir / "mixes" / "mix.wav"
    if mix_p.exists():
        mono, sr = _load_mono(mix_p)
        m = _scan_window(mono, sr, t_lo, t_hi)
        report["mix"] = {"file": str(mix_p), **m}

    return report


def _render_trace_text(rep: dict) -> str:
    lines = []
    t = rep["target_time_s"]
    w = rep["window_s"]
    lines.append(f"=== Click trace at t={t}s (±{w/2}s window) ===\n")

    # Tracks where the chain shows a click
    if rep["tracks"]:
        lines.append("Per-track chain (only tracks with a step ≥ 0.10 shown):")
        lines.append(f"{'track':<32} {'stage':<32} {'peak dBFS':>10} {'max step':>10} {'step dBFS':>10}")
        for tr in rep["tracks"]:
            for c in tr["chain"]:
                if "error" in c:
                    lines.append(f"  {tr['name'][:32]:<32} {c['stage']:<32} ERROR: {c['error']}")
                    continue
                hot = " [HOT]" if c["max_step"] is not None and c["max_step"] >= DEFAULT_STEP_THRESHOLD else ""
                lines.append(
                    f"  {tr['name'][:32]:<32} {c['stage']:<32} "
                    f"{(c['peak_db'] if c['peak_db'] is not None else 0):>+10.2f} "
                    f"{(c['max_step'] if c['max_step'] is not None else 0):>10.5f} "
                    f"{(c['max_step_db'] if c['max_step_db'] is not None else 0):>+10.2f}{hot}"
                )
        lines.append("")
    else:
        lines.append("No per-track chain stage had a step ≥ 0.10 in the window.\n")

    if rep["stems"]:
        lines.append("Per-bus stems:")
        for st in rep["stems"]:
            if "error" in st:
                lines.append(f"  {st['file']:<25} ERROR: {st['error']}")
                continue
            hot = " [HOT]" if st["max_step"] is not None and st["max_step"] >= DEFAULT_STEP_THRESHOLD else ""
            lines.append(
                f"  {st['file']:<25} peak {(st['peak_db'] or 0):>+7.2f}  "
                f"max step {st['max_step'] or 0:.5f} ({(st['max_step_db'] or 0):+.2f} dBFS){hot}"
            )
        lines.append("")

    if rep["mix"]:
        m = rep["mix"]
        hot = " [HOT]" if m["max_step"] is not None and m["max_step"] >= DEFAULT_STEP_THRESHOLD else ""
        lines.append(f"Final mix:")
        lines.append(
            f"  peak {(m['peak_db'] or 0):>+7.2f} dBFS  "
            f"max step {(m['max_step'] or 0):.5f} ({(m['max_step_db'] or 0):+.2f} dBFS){hot}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find clicks / sharp inter-sample steps in a mix and trace them through the chain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Sweep the mix and list the top click candidates:
    find_clicks.py output/horgonyt

  Trace a known click time through every chain stage:
    find_clicks.py output/horgonyt --time 72.33

  Tighten the detection threshold for cleaner mixes:
    find_clicks.py output/horgonyt --threshold 0.05
        """,
    )
    parser.add_argument("session_dir", type=Path,
                        help="Session directory (must contain mixes/mix.wav)")
    parser.add_argument("--time", type=float, default=None,
                        help="Trace mode: walk the chain at this time (seconds) "
                             "and report the max inter-sample step at each stage. "
                             "Omit for sweep mode.")
    parser.add_argument("--window-s", type=float, default=DEFAULT_TRACE_WINDOW_S,
                        help=f"Trace-mode window size in seconds (default: {DEFAULT_TRACE_WINDOW_S})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_STEP_THRESHOLD,
                        help=f"Step magnitude threshold for a click candidate "
                             f"(default: {DEFAULT_STEP_THRESHOLD})")
    parser.add_argument("--cluster-gap-ms", type=float, default=DEFAULT_CLUSTER_GAP_MS,
                        help=f"Collapse clicks within this many ms into one event "
                             f"(default: {DEFAULT_CLUSTER_GAP_MS})")
    parser.add_argument("--top", type=int, default=20,
                        help="Sweep mode: show only the top N loudest candidates "
                             "(default: 20)")
    args = parser.parse_args()

    if not args.session_dir.is_dir():
        print(json.dumps({"error": f"Not a directory: {args.session_dir}"}), file=sys.stderr)
        sys.exit(1)

    if args.time is not None:
        rep = trace_at_time(args.session_dir, args.time, args.window_s)
        print(_render_trace_text(rep))
        print()
        print(json.dumps(rep, indent=2))
        return

    # Sweep mode
    mix_path = args.session_dir / "mixes" / "mix.wav"
    if not mix_path.exists():
        print(json.dumps({"error": f"No mix.wav at {mix_path}"}), file=sys.stderr)
        sys.exit(1)

    events = find_click_candidates(mix_path, args.threshold, args.cluster_gap_ms)
    print(f"=== Click candidates in {mix_path} ===")
    print(f"Threshold: |Δsample| ≥ {args.threshold}, cluster gap: {args.cluster_gap_ms} ms")
    print(f"Total candidates: {len(events)}")
    if events:
        top = events[:args.top]
        print(f"\nTop {len(top)} by step magnitude:")
        print(f"{'time':>10} {'step':>10} {'dBFS step':>10} {'cluster size':>13}")
        for e in top:
            mm = f"{e['time_s']//60:.0f}:{e['time_s']%60:05.2f}"
            print(f"{mm:>10} {e['step_magnitude']:>10.5f} {e['step_dbfs']:>+10.2f} {e['cluster_size']:>13}")
    print()
    print(json.dumps({"events": events[:args.top], "total": len(events)}, indent=2))


if __name__ == "__main__":
    main()
