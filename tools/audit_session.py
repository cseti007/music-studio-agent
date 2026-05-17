"""Audit a parsed session.json for tracks sharing identical source files.

DAW sessions sometimes contain duplicate tracks — same mic, same audio,
two track entries. When both go into render_mix, they sum phase-coherently
on the bus and add +6 dB to that mic's frequency band. The user hears it
as "the drummer played that part twice" — but it's not the drummer, it's
the session editor accidentally cloning a track.

This tool groups tracks by the **set of source files their clips
reference**. Any group with more than one track is a duplicate candidate:
identical audio routed to two (or more) mix_config entries.

Output:
  audit_report.json — full groupings + recommendations
  audit_report.txt  — human-readable summary

Usage:
  python audit_session.py output/<session>/session.json \\
      --output-dir output/<session>/analysis

Run this at session start, BEFORE generating mix_config. Any
duplicate group flagged here is a candidate for `active: false` on
all-but-one of its tracks in mix_config.json.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def find_duplicates(session_path: Path) -> dict:
    """Return groups of tracks that share the exact same set of source files."""
    data = json.loads(session_path.read_text(encoding="utf-8"))
    tracks = data.get("tracks", [])

    track_sources: dict[str, set] = {}
    for t in tracks:
        sources = set()
        for c in t.get("clips", []):
            src = c.get("source_file", "")
            if src:
                # Use basename only — full paths may differ but the actual
                # audio file is what matters
                sources.add(Path(src).name)
        track_sources[t["name"]] = sources

    # Group tracks by their source-file set
    sources_to_tracks: dict = defaultdict(list)
    for name, srcs in track_sources.items():
        if not srcs:
            continue
        key = tuple(sorted(srcs))
        sources_to_tracks[key].append(name)

    groups = []
    for srcs, names in sorted(sources_to_tracks.items()):
        if len(names) > 1:
            # Suggest a primary (shortest name — usually the original,
            # before .dup1.XX suffix was added) and recommend deactivating
            # the rest in mix_config.json.
            primary = min(names, key=lambda n: (len(n), n))
            deactivate = [n for n in names if n != primary]
            groups.append({
                "n_tracks": len(names),
                "tracks": sorted(names),
                "source_files": list(srcs),
                "recommend_primary": primary,
                "recommend_deactivate": deactivate,
            })

    return {
        "session_file": str(session_path),
        "total_tracks": len(tracks),
        "duplicate_groups": groups,
        "summary": {
            "n_groups": len(groups),
            "n_tracks_affected": sum(g["n_tracks"] for g in groups),
            "n_tracks_to_deactivate": sum(len(g["recommend_deactivate"]) for g in groups),
        },
    }


def render_text(report: dict) -> str:
    lines = ["SESSION AUDIT REPORT", "=" * 60,
             f"  Session : {report['session_file']}",
             f"  Tracks  : {report['total_tracks']}",
             ""]
    s = report["summary"]
    if s["n_groups"] == 0:
        lines.append("[OK] No duplicate source-file groups found.")
        lines.append("     Every track references a unique set of audio files.")
        return "\n".join(lines)

    lines.append(f"[!] {s['n_groups']} duplicate group(s) — "
                 f"{s['n_tracks_affected']} tracks share audio; "
                 f"{s['n_tracks_to_deactivate']} suggested for deactivation")
    lines.append("")
    lines.append("Setting `active: false` on the suggested tracks in mix_config.json")
    lines.append("will prevent phase-coherent +6 dB doubling on those mics.")
    lines.append("")
    lines.append("DUPLICATE GROUPS")
    lines.append("-" * 60)
    for i, g in enumerate(report["duplicate_groups"], 1):
        lines.append(f"\nGroup {i} ({g['n_tracks']} tracks):")
        lines.append(f"  [+] keep   : {g['recommend_primary']}")
        for name in g["recommend_deactivate"]:
            lines.append(f"  [-] disable: {name}")
        srcs = g["source_files"]
        srcs_str = srcs[0] if len(srcs) == 1 else f"{srcs[0]} (and {len(srcs) - 1} more)"
        lines.append(f"      source : {srcs_str}")
    lines.append("")
    lines.append("RECOMMENDED ACTION")
    lines.append("-" * 60)
    lines.append("In mix_config.json, set `active: false` on each [-] line above,")
    lines.append("then re-render and re-run mix_health.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find tracks sharing identical source files in a session.json.",
    )
    parser.add_argument("session", type=Path, help="Path to session.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write audit_report.{json,txt}")
    args = parser.parse_args()

    if not args.session.exists():
        print(json.dumps({"error": f"Not found: {args.session}"}), file=sys.stderr)
        sys.exit(1)

    report = find_duplicates(args.session)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    text = render_text(report)
    (args.output_dir / "audit_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
