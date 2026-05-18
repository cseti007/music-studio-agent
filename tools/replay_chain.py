"""Replay a mix_chain.json recall sheet — rebuild the entire mix from scratch.

Reads a mix_chain.json (produced by `tools/build_chain.py`) and re-runs every
processing step on every active stem in the recorded order, by calling the
existing apply_*.py CLI tools via subprocess. After all stems are rebuilt,
runs `render_mix --render --stems` against the session's mix_config.json.

Usage:
  python tools/replay_chain.py output/<session>/mix_chain.json
  # Or just the session dir — mix_chain.json is found relative to it
  python tools/replay_chain.py output/<session>

  python tools/replay_chain.py output/<session>/mix_chain.json --dry-run
  # Prints every command it would execute without actually running anything

  python tools/replay_chain.py output/<session>/mix_chain.json --stem "KICK IN.05"
  # Replay a single stem only (for debugging)

The default behaviour is overwrite-in-place — the original output/<session>/
directory is the target. Back up first if you want to keep the previous run.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Per-step → CLI argv builder
# ---------------------------------------------------------------------------

def _argv_gain_per_clip(step: dict, session_json: Path) -> list[str]:
    args = step["args"]
    output_dir = str(Path(step["output"]).parent.parent)  # .../tracks  (per-clip writes into tracks/<name>/)
    # The original gain_per_clip expects --track <NAME>, infer from "session.json:<name>" syntax
    track_name = step["input"].split(":", 1)[1] if ":" in step["input"] else step["input"]
    argv = [
        sys.executable, str(TOOLS_DIR / "apply_gain.py"),
        "--per-clip", str(session_json),
        "--track", track_name,
        "--output-dir", output_dir,
        "--peak-ceiling", str(args.get("peak_ceiling_db", -1.0)),
    ]
    if not args.get("normalize", True):
        argv.append("--no-normalize")
    elif args.get("target_lufs") is not None:
        argv += ["--clip-target-lufs", str(args["target_lufs"])]
    return argv


def _argv_gain_per_channel(step: dict) -> list[str]:
    args = step["args"]
    output_dir = str(Path(step["output"]).parent)
    argv = [
        sys.executable, str(TOOLS_DIR / "apply_gain.py"),
        "--per-channel", step["input"],
        "--output-dir", output_dir,
        "--peak-ceiling", str(args.get("peak_ceiling_db", -1.0)),
    ]
    if "gain_db" in args:
        argv += ["--gain-db", str(args["gain_db"])]
    elif "target_lufs" in args:
        argv += ["--target-lufs", str(args["target_lufs"])]
    return argv


def _argv_align(step: dict) -> list[str]:
    args = step["args"]
    output_dir = str(Path(step["output"]).parent)
    return [
        sys.executable, str(TOOLS_DIR / "align_phase.py"),
        "--reference", args["reference"],
        "--target", step["input"],
        "--output-dir", output_dir,
        "--max-delay-ms", str(args.get("max_delay_ms", 20.0)),
        "--segment-sec", str(args.get("segment_sec", 10.0)),
    ]


def _argv_eq(step: dict) -> list[str]:
    args = step["args"]
    output_dir = str(Path(step["output"]).parent)
    argv = [
        sys.executable, str(TOOLS_DIR / "apply_eq.py"),
        step["input"],
        "--output-dir", output_dir,
        "--phase", args.get("phase", "minimum"),
    ]
    if "preset" in args:
        argv += ["--preset", args["preset"]]
    else:
        for f in args.get("filters", []):
            argv += ["--filter", json.dumps(f)]
    return argv


def _argv_comp(step: dict) -> list[str]:
    args = step["args"]
    output_dir = str(Path(step["output"]).parent)
    argv = [
        sys.executable, str(TOOLS_DIR / "apply_compression.py"),
        step["input"],
        "--output-dir", output_dir,
    ]
    if "preset" in args:
        argv += ["--preset", args["preset"]]
    for k_cli, k_json in (
        ("--threshold", "threshold_db"),
        ("--ratio",     "ratio"),
        ("--attack",    "attack_ms"),
        ("--release",   "release_ms"),
        ("--makeup",    "makeup_db"),
        ("--mix",       "mix"),
    ):
        if k_json in args and args[k_json] is not None:
            argv += [k_cli, str(args[k_json])]
    if "sidechain" in args and isinstance(args["sidechain"], dict):
        sc = args["sidechain"]
        argv += ["--sidechain", sc.get("file", "")]
        if sc.get("sc_hp_hz") is not None:
            argv += ["--sc-hp", str(sc["sc_hp_hz"])]
        if sc.get("sc_lp_hz") is not None:
            argv += ["--sc-lp", str(sc["sc_lp_hz"])]
    return argv


def _argv_gate(step: dict) -> list[str]:
    args = step["args"]
    output_dir = str(Path(step["output"]).parent)
    argv = [
        sys.executable, str(TOOLS_DIR / "apply_gate.py"),
        step["input"],
        "--output-dir", output_dir,
    ]
    if "preset" in args:
        argv += ["--preset", args["preset"]]
    for k_cli, k_json in (
        ("--threshold",   "threshold_db"),
        ("--range",       "range_db"),
        ("--attack",      "attack_ms"),
        ("--hold",        "hold_ms"),
        ("--release",     "release_ms"),
        ("--hysteresis",  "hysteresis_db"),
    ):
        if k_json in args and args[k_json] is not None:
            argv += [k_cli, str(args[k_json])]
    return argv


def _argv_amp(step: dict) -> list[str]:
    args = step["args"]
    output_dir = str(Path(step["output"]).parent)
    argv = [
        sys.executable, str(TOOLS_DIR / "apply_amp.py"),
        step["input"],
        "--output-dir", output_dir,
    ]
    if "preset" in args:
        argv += ["--preset", args["preset"]]
    for k_cli, k_json in (
        ("--drive",         "drive"),
        ("--asymmetry",     "asymmetry"),
        ("--hp",            "hp_hz"),
        ("--lp",            "lp_hz"),
        ("--low-shelf-hz",  "low_shelf_hz"),
        ("--low-shelf-db",  "low_shelf_db"),
        ("--mid-hz",        "mid_hz"),
        ("--mid-db",        "mid_db"),
        ("--mid-q",         "mid_q"),
    ):
        if k_json in args and args[k_json] is not None:
            argv += [k_cli, str(args[k_json])]
    return argv


def _argv_reverb(step: dict) -> list[str]:
    args = step["args"]
    output_dir = str(Path(step["output"]).parent)
    argv = [
        sys.executable, str(TOOLS_DIR / "apply_reverb.py"),
        step["input"],
        "--output-dir", output_dir,
    ]
    if "preset" in args:
        argv += ["--preset", args["preset"]]
    if "ir" in args and args["ir"]:
        argv += ["--ir", args["ir"]]
    if args.get("send"):
        argv.append("--send")
    for k_cli, k_json in (
        ("--pre-delay", "pre_delay_ms"),
        ("--wet",       "wet"),
        ("--hp",        "hp_hz"),
        ("--lp",        "lp_hz"),
    ):
        if k_json in args and args[k_json] is not None:
            argv += [k_cli, str(args[k_json])]
    if "sidechain" in args and isinstance(args["sidechain"], dict):
        sc = args["sidechain"]
        argv += ["--sidechain", sc.get("file", "")]
        if sc.get("depth_db") is not None:
            argv += ["--sc-depth", str(sc["depth_db"])]
    return argv


def _argv_transient(step: dict) -> list[str]:
    args = step["args"]
    output_dir = str(Path(step["output"]).parent)
    argv = [
        sys.executable, str(TOOLS_DIR / "apply_transient.py"),
        step["input"],
        "--output-dir", output_dir,
    ]
    if "preset" in args:
        argv += ["--preset", args["preset"]]
    for k_cli, k_json in (
        ("--attack",  "attack_db"),
        ("--sustain", "sustain_db"),
        ("--fast-ms", "fast_ms"),
        ("--slow-ms", "slow_ms"),
    ):
        if k_json in args and args[k_json] is not None:
            argv += [k_cli, str(args[k_json])]
    return argv


def _argv_saturation(step: dict) -> list[str]:
    args = step["args"]
    output_dir = str(Path(step["output"]).parent)
    argv = [
        sys.executable, str(TOOLS_DIR / "apply_saturation.py"),
        step["input"],
        "--output-dir", output_dir,
    ]
    if "preset" in args:
        argv += ["--preset", args["preset"]]
    for k_cli, k_json in (
        ("--mode",       "mode"),
        ("--drive",      "drive"),
        ("--asymmetry",  "asymmetry"),
        ("--mix",        "mix"),
    ):
        if k_json in args and args[k_json] is not None:
            argv += [k_cli, str(args[k_json])]
    return argv


def _argv_delay(step: dict) -> list[str]:
    args = step["args"]
    output_dir = str(Path(step["output"]).parent)
    argv = [
        sys.executable, str(TOOLS_DIR / "apply_delay.py"),
        step["input"],
        "--output-dir", output_dir,
    ]
    if "preset" in args:
        argv += ["--preset", args["preset"]]
    for k_cli, k_json in (
        ("--mode",     "mode"),
        ("--delay-ms", "delay_ms"),
        ("--feedback", "feedback"),
        ("--mix",      "mix"),
        ("--bpm",      "bpm"),
        ("--division", "division"),
        ("--hp",       "hp_hz"),
        ("--lp",       "lp_hz"),
    ):
        if k_json in args and args[k_json] is not None:
            argv += [k_cli, str(args[k_json])]
    if args.get("send"):
        argv.append("--send")
    return argv


_STEP_TO_BUILDER = {
    "gain_per_clip":     _argv_gain_per_clip,
    "gain_per_channel":  _argv_gain_per_channel,
    "align_phase":       _argv_align,
    "eq":                _argv_eq,
    "comp":              _argv_comp,
    "gate":              _argv_gate,
    "amp":               _argv_amp,
    "reverb":            _argv_reverb,
    "transient":         _argv_transient,
    "saturation":        _argv_saturation,
    "delay":             _argv_delay,
}


def _build_argv(step: dict, session_json: Path) -> list[str] | None:
    """Return argv to run for this chain step, or None if unsupported."""
    builder = _STEP_TO_BUILDER.get(step["step"])
    if builder is None:
        return None
    if step["step"] == "gain_per_clip":
        return builder(step, session_json)
    return builder(step)


# ---------------------------------------------------------------------------
# Top-level replay
# ---------------------------------------------------------------------------

def _resolve_chain_path(arg: Path) -> Path:
    """Accept either a mix_chain.json path or a session directory."""
    if arg.is_dir():
        candidate = arg / "mix_chain.json"
        if candidate.exists():
            return candidate
    return arg


def replay(chain_path: Path, dry_run: bool, stem_filter: str | None) -> int:
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    session_json = Path(chain.get("session_json", ""))
    mix_config = Path(chain.get("mix_config", ""))

    if not dry_run:
        if not session_json.exists():
            print(f"FATAL: session.json not found at {session_json}", file=sys.stderr)
            return 2

    n_stems = 0
    n_steps_total = 0
    n_steps_failed = 0
    n_steps_skipped = 0
    t0 = time.time()

    for stem in chain["stems"]:
        if stem_filter and stem["name"] != stem_filter:
            continue
        if not stem.get("active", True):
            print(f"[skip inactive] {stem['name']}")
            continue
        n_stems += 1
        print(f"\n=== {stem['name']} ({len(stem['chain'])} steps) ===")
        for i, step in enumerate(stem["chain"], 1):
            argv = _build_argv(step, session_json)
            if argv is None:
                print(f"  [{i}] {step['step']}: UNSUPPORTED — skipping")
                n_steps_skipped += 1
                continue
            n_steps_total += 1
            quoted = " ".join(shlex.quote(a) for a in argv)
            print(f"  [{i}] {step['step']}  → {Path(step['output']).name}")
            if dry_run:
                print(f"      $ {quoted}")
                continue
            proc = subprocess.run(argv, capture_output=True, text=True)
            if proc.returncode != 0:
                n_steps_failed += 1
                print(f"      FAIL ({proc.returncode}): {proc.stderr.strip()[:200]}", file=sys.stderr)

    if not dry_run and mix_config.exists():
        print(f"\n=== Rendering mix ({mix_config}) ===")
        argv = [
            sys.executable, str(TOOLS_DIR / "render_mix.py"),
            str(mix_config), "--render", "--stems",
        ]
        proc = subprocess.run(argv)
        if proc.returncode != 0:
            print("FAIL: render_mix returned non-zero", file=sys.stderr)
            n_steps_failed += 1

    elapsed = time.time() - t0
    print(f"\nReplay done in {elapsed:.1f}s — {n_stems} stems, {n_steps_total} steps run, "
          f"{n_steps_failed} failed, {n_steps_skipped} unsupported")
    return 1 if n_steps_failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a mix_chain.json recall sheet — rebuild the mix from scratch.",
    )
    parser.add_argument("chain", type=Path,
                        help="Path to mix_chain.json (or the session directory containing it)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print every command without executing")
    parser.add_argument("--stem", type=str, default=None,
                        help="Replay only this single stem (for debugging)")
    args = parser.parse_args()

    chain_path = _resolve_chain_path(args.chain)
    if not chain_path.exists():
        print(f"FATAL: chain file not found: {chain_path}", file=sys.stderr)
        sys.exit(2)

    rc = replay(chain_path, dry_run=args.dry_run, stem_filter=args.stem)
    sys.exit(rc)


if __name__ == "__main__":
    main()
