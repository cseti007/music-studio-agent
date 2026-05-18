"""Aggregate per-stem *_report.json files into a single mix_chain.json recall sheet.

The recall sheet captures every processing step that touched each stem,
in the order it ran, with the exact parameters used. Feeding this back
through `tools/replay_chain.py` rebuilds the full mix from scratch.

This tool is **non-invasive**: it only reads existing reports. The
existing apply_*.py tools are not modified.

Usage:
  python tools/build_chain.py output/<session>
  # Writes: output/<session>/mix_chain.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Report -> chain-step mapping
# ---------------------------------------------------------------------------
#
# Each report type has a known structure. The mappers below pull only the
# fields needed to reproduce the step, plus the input/output file paths
# that drive topological ordering.

def _step_from_gain_per_clip(rpt: dict) -> dict:
    return {
        "step": "gain_per_clip",
        "input": f"session.json:{rpt.get('track')}",
        "output": rpt.get("output"),
        "args": {
            "target_lufs": rpt.get("clip_gain_target_lufs"),
            "peak_ceiling_db": rpt.get("peak_ceiling_db", -1.0),
            "normalize": rpt.get("mode") != "per-clip-no-normalize",
        },
    }


def _step_from_gain_per_channel(rpt: dict) -> dict:
    # Either gain_db was explicit, or it was auto-derived from a target LUFS.
    args: dict = {"peak_ceiling_db": rpt.get("peak_ceiling_db", -1.0)}
    src = rpt.get("gain_source", "")
    if "auto (target" in src:
        # "auto (target -18.0 LUFS)" → derive target lufs
        try:
            args["target_lufs"] = float(src.split("target")[-1].split("LUFS")[0].strip())
        except (ValueError, IndexError):
            args["target_lufs"] = -18.0
    else:
        args["gain_db"] = rpt.get("gain_applied_db", rpt.get("gain_requested_db"))
    return {
        "step": "gain_per_channel",
        "input": rpt.get("file"),
        "output": rpt.get("output"),
        "args": args,
    }


def _step_from_align(rpt: dict) -> dict:
    return {
        "step": "align_phase",
        "input": rpt.get("target"),
        "output": rpt.get("output"),
        "args": {
            "reference": rpt.get("reference"),
            "max_delay_ms": rpt.get("max_delay_ms", 20.0),
            "segment_sec": rpt.get("segment_sec_used", 10.0),
        },
    }


def _step_from_eq(rpt: dict) -> dict:
    args: dict = {
        "phase": rpt.get("phase", "minimum"),
    }
    if rpt.get("preset_used"):
        args["preset"] = rpt["preset_used"]
    else:
        args["filters"] = rpt.get("filters_applied", [])
    return {
        "step": "eq",
        "input": rpt.get("input"),
        "output": rpt.get("output"),
        "args": args,
    }


def _step_from_comp(rpt: dict) -> dict:
    args: dict = dict(rpt.get("settings", {}))
    if rpt.get("preset"):
        args["preset"] = rpt["preset"]
    sc = rpt.get("sidechain")
    if sc:
        args["sidechain"] = sc
    return {
        "step": "comp",
        "input": rpt.get("input"),
        "output": rpt.get("output"),
        "args": args,
    }


def _step_from_gate(rpt: dict) -> dict:
    args: dict = dict(rpt.get("settings", {}))
    if rpt.get("preset"):
        args["preset"] = rpt["preset"]
    return {
        "step": "gate",
        "input": rpt.get("input"),
        "output": rpt.get("output"),
        "args": args,
    }


def _step_from_amp(rpt: dict) -> dict:
    args = {k: rpt[k] for k in (
        "drive", "asymmetry", "hp_hz", "lp_hz",
        "low_shelf_hz", "low_shelf_db", "mid_hz", "mid_db", "mid_q",
    ) if k in rpt}
    if rpt.get("preset"):
        args["preset"] = rpt["preset"]
    return {
        "step": "amp",
        "input": rpt.get("input"),
        "output": rpt.get("output"),
        "args": args,
    }


def _step_from_reverb(rpt: dict) -> dict:
    args = {k: rpt[k] for k in (
        "room_size", "damping", "width", "pre_delay_ms", "wet", "dry",
        "hp_hz", "lp_hz", "gate_hold_ms", "gate_release_ms",
    ) if k in rpt}
    if rpt.get("preset"):
        args["preset"] = rpt["preset"]
    if rpt.get("ir"):
        args["ir"] = rpt["ir"]
    if rpt.get("mode") == "send":
        args["send"] = True
    sc = rpt.get("sidechain")
    if sc:
        args["sidechain"] = sc
    return {
        "step": "reverb",
        "input": rpt.get("input"),
        "output": rpt.get("output"),
        "args": args,
    }


def _step_from_transient(rpt: dict) -> dict:
    args = {k: rpt[k] for k in ("attack_db", "sustain_db", "fast_ms", "slow_ms") if k in rpt}
    if rpt.get("preset"):
        args["preset"] = rpt["preset"]
    return {
        "step": "transient",
        "input": rpt.get("input"),
        "output": rpt.get("output"),
        "args": args,
    }


def _step_from_saturation(rpt: dict) -> dict:
    args: dict = dict(rpt.get("settings", {}))
    if rpt.get("preset"):
        args["preset"] = rpt["preset"]
    return {
        "step": "saturation",
        "input": rpt.get("input"),
        "output": rpt.get("output"),
        "args": args,
    }


def _step_from_delay(rpt: dict) -> dict:
    args: dict = dict(rpt.get("settings", {}))
    if rpt.get("preset"):
        args["preset"] = rpt["preset"]
    return {
        "step": "delay",
        "input": rpt.get("input"),
        "output": rpt.get("output"),
        "args": args,
    }


def _step_from_make_it_hit(step_name: str, rpt: dict) -> dict:
    """subharm / haas / exciter / multiband — same shape."""
    args = {k: v for k, v in rpt.items() if k not in ("input", "output", "relevance_check")}
    return {
        "step": step_name,
        "input": rpt.get("input"),
        "output": rpt.get("output"),
        "args": args,
    }


_REPORT_TO_MAPPER = {
    "gain_report.json":             _step_from_gain_per_clip,
    "assembled_gain_report.json":   _step_from_gain_per_channel,
    "align_report.json":            _step_from_align,
    "eq_report.json":               _step_from_eq,
    "comp_report.json":             _step_from_comp,
    "gate_report.json":             _step_from_gate,
    "amp_report.json":              _step_from_amp,
    "reverb_report.json":           _step_from_reverb,
    "transient_report.json":        _step_from_transient,
    "sat_report.json":              _step_from_saturation,
    "delay_report.json":            _step_from_delay,
    "subharm_report.json":          lambda r: _step_from_make_it_hit("subharm",   r),
    "haas_report.json":             lambda r: _step_from_make_it_hit("haas",      r),
    "exciter_report.json":          lambda r: _step_from_make_it_hit("exciter",   r),
    "multiband_report.json":        lambda r: _step_from_make_it_hit("multiband", r),
}


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------

def _topo_sort_chain(steps: list[dict]) -> list[dict]:
    """Order chain steps by input→output dependency.

    Each step has an `input` path and an `output` path. A step S2 depends on
    S1 if S2.input's filename == S1.output's filename. Matching by filename
    rather than full path is intentional: a few historical runs wrote output
    files into accidentally-nested paths (e.g. "<stem>/<stem>/assembled_aligned.wav"
    instead of "<stem>/assembled_aligned.wav"). Since every chain step within
    a single stem directory produces a uniquely-named file (stage suffixes:
    assembled.wav, assembled_gained.wav, assembled_eq.wav, ...) the basename
    is a reliable join key without requiring path correctness.

    Steps without a predecessor (e.g. the first gain_per_clip whose input is
    "session.json:<name>") come first.
    """
    if not steps:
        return []

    def _basename(p: str | None) -> str | None:
        if not p:
            return None
        return Path(p).name if "/" in p or "\\" in p else None

    by_output_name = {
        _basename(s["output"]): s
        for s in steps
        if s.get("output") and _basename(s["output"])
    }

    ordered: list[dict] = []
    seen: set[int] = set()

    def visit(step: dict) -> None:
        sid = id(step)
        if sid in seen:
            return
        seen.add(sid)
        inp_name = _basename(step.get("input"))
        if inp_name and inp_name in by_output_name:
            predecessor = by_output_name[inp_name]
            if predecessor is not step:
                visit(predecessor)
        ordered.append(step)

    for step in steps:
        visit(step)
    return ordered


def build_stem_chain(track_dir: Path) -> list[dict]:
    """Collect every *_report.json in a stem directory, map to chain steps,
    return them topologically ordered."""
    steps: list[dict] = []
    for report_path in track_dir.glob("*_report.json"):
        mapper = _REPORT_TO_MAPPER.get(report_path.name)
        if mapper is None:
            continue
        try:
            rpt = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  WARNING: could not read {report_path}: {exc}", file=sys.stderr)
            continue
        step = mapper(rpt)
        if step.get("output"):
            steps.append(step)
    return _topo_sort_chain(steps)


def build_chain(session_dir: Path) -> dict:
    """Walk every <session_dir>/tracks/<stem>/ and produce a session-wide
    mix_chain.json structure."""
    tracks_root = session_dir / "tracks"
    if not tracks_root.is_dir():
        raise FileNotFoundError(f"No tracks/ directory in {session_dir}")

    # active set from mix_config.json, if present
    active: dict[str, bool] = {}
    mix_config = session_dir / "mix_config.json"
    if mix_config.exists():
        try:
            cfg = json.loads(mix_config.read_text(encoding="utf-8"))
            for track in cfg.get("tracks", []):
                active[track.get("name", "")] = bool(track.get("active", True))
        except (OSError, json.JSONDecodeError):
            pass  # config absent or broken — default everything to active

    stems: list[dict] = []
    for stem_dir in sorted(tracks_root.iterdir()):
        if not stem_dir.is_dir():
            continue
        chain = build_stem_chain(stem_dir)
        if not chain:
            continue
        stems.append({
            "name": stem_dir.name,
            "active": active.get(stem_dir.name, True),
            "chain": chain,
        })

    return {
        "session_dir": str(session_dir),
        "session_json": str(session_dir / "session.json"),
        "mix_config": str(session_dir / "mix_config.json"),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "stems": stems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-stem *_report.json files into mix_chain.json",
    )
    parser.add_argument("session_dir", type=Path,
                        help="Session output directory (e.g. output/terido)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write mix_chain.json (default: <session_dir>/mix_chain.json)")
    args = parser.parse_args()

    chain = build_chain(args.session_dir)
    out_path = args.output or (args.session_dir / "mix_chain.json")
    out_path.write_text(json.dumps(chain, indent=2, ensure_ascii=False), encoding="utf-8")

    n_stems = len(chain["stems"])
    n_steps = sum(len(s["chain"]) for s in chain["stems"])
    print(f"Wrote {out_path} — {n_stems} stems, {n_steps} chain steps total")


if __name__ == "__main__":
    main()
