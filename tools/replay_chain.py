"""Replay a mix_chain.json recall sheet — rebuild the entire mix from scratch.

Reads a mix_chain.json (produced by `tools/build_chain.py`) and re-runs every
processing step on every active stem in the recorded order. After all stems
are rebuilt, runs `render_mix --render --stems` against the session's
mix_config.json.

**In-process mode (default):** the apply_*.py modules are imported once and
each step is dispatched to its callable function directly. Avoids per-step
Python cold-start (~3-5s × N steps) — a 25-step chain replay runs in seconds
of DSP time instead of minutes of subprocess overhead.

**Subprocess mode (`--subprocess`):** falls back to spawning `python tool.py`
for each step. Useful for debugging when in-process state might be misleading.

Usage:
  python tools/replay_chain.py output/<session>/mix_chain.json
  # Or just the session dir — mix_chain.json is found relative to it
  python tools/replay_chain.py output/<session>

  python tools/replay_chain.py output/<session>/mix_chain.json --dry-run
  # Prints every command it would execute without actually running anything

  python tools/replay_chain.py output/<session>/mix_chain.json --stem "KICK IN.05"
  # Replay a single stem only (for debugging)

  python tools/replay_chain.py output/<session>/mix_chain.json --subprocess
  # Force the old subprocess path (each step in its own Python process)

The default behaviour is overwrite-in-place — the original output/<session>/
directory is the target. Back up first if you want to keep the previous run.
"""

from __future__ import annotations

import argparse
import json
import inspect
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent

# Make sibling tools importable when this file is run directly (not as a module)
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


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
    argv += ["--source-mode", args.get("source_mode", "per-clip"),
             "--crossfade-ms", str(args.get("crossfade_ms", 5.0)),
             "--source-target-lufs", str(args.get("source_target_lufs", -18.0))]
    for key in ("interloper_head_ms", "interloper_tail_ms"):
        if args.get(key) is not None:
            argv += ["--" + key.replace("_", "-"), str(args[key])]
    if args.get("normalize_per_source"):
        argv.append("--normalize-per-source")
    if not args.get("normalize", True):
        argv.append("--no-normalize")
    else:
        argv.append("--normalize")
    if args.get("target_lufs") is not None:
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
    output_dir = str(Path(step["output"]).parent.parent)
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
        ("--room-size", "room_size"),
        ("--damping", "damping"),
        ("--width", "width"),
        ("--dry", "dry"),
        ("--gate-hold", "gate_hold_ms"),
        ("--gate-release", "gate_release_ms"),
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
        for key, flag in (("hp_hz", "--sc-hp"), ("lp_hz", "--sc-lp")):
            if sc.get(key) is not None:
                argv += [flag, str(sc[key])]
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


# ---------------------------------------------------------------------------
# In-process dispatch — (callable, kwargs) instead of argv
# ---------------------------------------------------------------------------
#
# Imports are lazy so --dry-run / --subprocess modes pay no import cost.

def _lazy_import(modname: str):
    import importlib
    return importlib.import_module(modname)


def _call_gain_per_clip(step: dict, session_json: Path):
    args = step["args"]
    output_dir = Path(step["output"]).parent.parent  # .../tracks/
    track_name = step["input"].split(":", 1)[1] if ":" in step["input"] else step["input"]
    mod = _lazy_import("apply_gain")
    return mod.apply_gain_per_clip, {
        "session_json": session_json,
        "output_dir": output_dir,
        "track_names": [track_name],
        "all_tracks": False,
        "target_lufs": args.get("target_lufs"),
        "peak_ceiling_db": float(args.get("peak_ceiling_db", -1.0)),
        "normalize": bool(args.get("normalize", True)),
        "source_mode": args.get("source_mode", "per-clip"),
        "crossfade_ms": args.get("crossfade_ms", 5.0),
        "interloper_head_ms": args.get("interloper_head_ms"),
        "interloper_tail_ms": args.get("interloper_tail_ms"),
        "normalize_per_source": args.get("normalize_per_source", False),
        "source_target_lufs": args.get("source_target_lufs", -18.0),
    }


def _call_gain_per_channel(step: dict):
    args = step["args"]
    output_dir = Path(step["output"]).parent
    mod = _lazy_import("apply_gain")
    kwargs = {
        "file_path": Path(step["input"]),
        "output_dir": output_dir,
        "peak_ceiling_db": float(args.get("peak_ceiling_db", -1.0)),
    }
    if "gain_db" in args:
        kwargs["gain_db"] = float(args["gain_db"])
    elif "target_lufs" in args:
        kwargs["target_lufs"] = float(args["target_lufs"])
    return mod.apply_gain_per_channel, kwargs


def _call_align(step: dict):
    args = step["args"]
    output_dir = Path(step["output"]).parent.parent
    mod = _lazy_import("align_phase")
    return mod.align_phase, {
        "reference_path": Path(args["reference"]),
        "target_path": Path(step["input"]),
        "output_dir": output_dir,
        "max_delay_ms": float(args.get("max_delay_ms", 20.0)),
        "segment_sec": float(args.get("segment_sec", 10.0)),
    }


def _call_eq(step: dict):
    args = step["args"]
    output_dir = Path(step["output"]).parent
    mod = _lazy_import("apply_eq")
    # Preset → expand to filters list (apply_eq's signature only takes filters list)
    filters = args.get("filters", [])
    if "preset" in args and not filters:
        preset = args["preset"]
        preset_path = TOOLS_DIR / "presets" / f"{preset}.json"
        if preset_path.exists():
            filters = json.loads(preset_path.read_text(encoding="utf-8")).get("filters", [])
    return mod.apply_eq, {
        "input_path": Path(step["input"]),
        "output_dir": output_dir,
        "filters": filters,
        "phase": args.get("phase", "minimum"),
    }


def _call_comp(step: dict):
    args = step["args"]
    output_dir = Path(step["output"]).parent
    mod = _lazy_import("apply_compression")
    # apply_compression signature requires threshold/ratio/attack/release; if a
    # preset is provided, expand to those values.
    if "preset" in args:
        preset_path = TOOLS_DIR / "presets" / f"{args['preset']}.json"
        if preset_path.exists():
            preset_data = json.loads(preset_path.read_text(encoding="utf-8"))
            p = preset_data.get("settings", preset_data)
            kwargs = {
                "input_path": Path(step["input"]),
                "output_dir": output_dir,
                "threshold_db": float(p.get("threshold_db", -10.0)),
                "ratio": float(p.get("ratio", 2.0)),
                "attack_ms": float(p.get("attack_ms", 10.0)),
                "release_ms": float(p.get("release_ms", 100.0)),
                "makeup_db": p.get("makeup_db"),
                "mix": float(p.get("mix", 1.0)),
            }
        else:
            raise FileNotFoundError(f"comp preset not found: {preset_path}")
    else:
        kwargs = {
            "input_path": Path(step["input"]),
            "output_dir": output_dir,
            "threshold_db": float(args["threshold_db"]),
            "ratio": float(args["ratio"]),
            "attack_ms": float(args["attack_ms"]),
            "release_ms": float(args["release_ms"]),
            "makeup_db": args.get("makeup_db"),
            "mix": float(args.get("mix", 1.0)),
        }
    for key in ("threshold_db", "ratio", "attack_ms", "release_ms", "makeup_db", "mix"):
        if args.get(key) is not None:
            kwargs[key] = args[key]
    if isinstance(args.get("sidechain"), dict):
        sc = args["sidechain"]
        kwargs.update(sidechain_path=Path(sc["file"]), sc_hp_hz=sc.get("sc_hp_hz"), sc_lp_hz=sc.get("sc_lp_hz"))
    return mod.apply_compression, kwargs


def _call_gate(step: dict):
    args = step["args"]
    output_dir = Path(step["output"]).parent
    mod = _lazy_import("apply_gate")
    if "preset" in args:
        preset_path = TOOLS_DIR / "presets" / f"{args['preset']}.json"
        if preset_path.exists():
            p = json.loads(preset_path.read_text(encoding="utf-8")).get("settings", {})
            kwargs = {
                "input_path": Path(step["input"]),
                "output_dir": output_dir,
                "threshold_db": float(p.get("threshold_db", -40.0)),
                "range_db": float(p.get("range_db", 30.0)),
                "attack_ms": float(p.get("attack_ms", 1.0)),
                "hold_ms": float(p.get("hold_ms", 50.0)),
                "release_ms": float(p.get("release_ms", 100.0)),
                "hysteresis_db": float(p.get("hysteresis_db", 6.0)),
            }
        else:
            raise FileNotFoundError(f"gate preset not found: {preset_path}")
    else:
        kwargs = {
            "input_path": Path(step["input"]),
            "output_dir": output_dir,
            "threshold_db": float(args["threshold_db"]),
            "range_db": float(args["range_db"]),
            "attack_ms": float(args["attack_ms"]),
            "hold_ms": float(args["hold_ms"]),
            "release_ms": float(args["release_ms"]),
            "hysteresis_db": float(args.get("hysteresis_db", 6.0)),
        }
    for key in ("threshold_db", "range_db", "attack_ms", "hold_ms", "release_ms", "hysteresis_db"):
        if args.get(key) is not None:
            kwargs[key] = args[key]
    return mod.apply_gate, kwargs


def _call_amp(step: dict):
    args = step["args"]
    output_dir = Path(step["output"]).parent
    mod = _lazy_import("apply_amp")
    kwargs = {
        "file_path": Path(step["input"]),
        "output_dir": output_dir,
    }
    if "preset" in args:
        preset_path = TOOLS_DIR / "presets" / f"{args['preset']}.json"
        if preset_path.exists():
            p = json.loads(preset_path.read_text(encoding="utf-8")).get("settings", {})
            for k in ("drive", "asymmetry", "hp_hz", "lp_hz", "low_shelf_hz",
                      "low_shelf_db", "mid_hz", "mid_db", "mid_q"):
                if k in p:
                    kwargs[k] = p[k]
    for k in ("drive", "asymmetry", "hp_hz", "lp_hz", "low_shelf_hz",
              "low_shelf_db", "mid_hz", "mid_db", "mid_q"):
        if k in args and args[k] is not None:
            kwargs[k] = args[k]
    return mod.apply_amp, kwargs


def _call_reverb(step: dict):
    args = step["args"]
    output_dir = Path(step["output"]).parent
    mod = _lazy_import("apply_reverb")
    kwargs = {
        "file_path": Path(step["input"]),
        "output_dir": output_dir,
    }
    for key in ("room_size", "damping", "width", "pre_delay_ms", "wet", "dry",
                "hp_hz", "lp_hz", "gate_hold_ms", "gate_release_ms"):
        if args.get(key) is not None:
            kwargs[key] = args[key]
    if args.get("ir"):
        kwargs["ir_path"] = Path(args["ir"])
    kwargs["send_mode"] = bool(args.get("send"))
    if isinstance(args.get("sidechain"), dict):
        sc = args["sidechain"]
        kwargs.update(sidechain_path=Path(sc["file"]), sc_depth_db=sc.get("depth_db", -12.0),
                      sc_hp_hz=sc.get("hp_hz"), sc_lp_hz=sc.get("lp_hz"))
    return mod.apply_reverb, kwargs


def _call_transient(step: dict):
    args = step["args"]
    output_dir = Path(step["output"]).parent
    mod = _lazy_import("apply_transient")
    kwargs = {
        "input_path": Path(step["input"]),
        "output_dir": output_dir,
    }
    for k in ("attack_db", "sustain_db", "fast_ms", "slow_ms"):
        if k in args and args[k] is not None:
            kwargs[k] = args[k]
    return mod.apply_transient_file, kwargs


def _call_saturation(step: dict):
    args = step["args"]
    output_dir = Path(step["output"]).parent
    mod = _lazy_import("apply_saturation")
    kwargs = {
        "input_path": Path(step["input"]),
        "output_dir": output_dir,
    }
    if "preset" in args:
        kwargs["preset_name"] = args["preset"]
    for k in ("mode", "drive", "asymmetry", "mix"):
        if k in args and args[k] is not None:
            kwargs[k] = args[k]
    return mod.apply_saturation, kwargs


def _call_delay(step: dict):
    args = step["args"]
    output_dir = Path(step["output"]).parent
    mod = _lazy_import("apply_delay")
    kwargs = {
        "input_path": Path(step["input"]),
        "output_dir": output_dir,
    }
    if "preset" in args:
        kwargs["preset_name"] = args["preset"]
    for k_json, k_kwarg in (
        ("mode", "mode"), ("delay_ms", "delay_ms"), ("feedback", "feedback"),
        ("mix", "mix"), ("bpm", "bpm"), ("division", "division"),
        ("hp_hz", "hp_hz"), ("lp_hz", "lp_hz"), ("send", "send"),
    ):
        if k_json in args and args[k_json] is not None:
            kwargs[k_kwarg] = args[k_json]
    return mod.apply_delay, kwargs


_STEP_TO_CALLABLE = {
    "gain_per_clip":     _call_gain_per_clip,
    "gain_per_channel":  _call_gain_per_channel,
    "align_phase":       _call_align,
    "eq":                _call_eq,
    "comp":              _call_comp,
    "gate":              _call_gate,
    "amp":               _call_amp,
    "reverb":            _call_reverb,
    "transient":         _call_transient,
    "saturation":        _call_saturation,
    "delay":             _call_delay,
}


def _build_inproc(step: dict, session_json: Path):
    """Return (callable, kwargs) for an in-process call, or (None, None) if unsupported."""
    builder = _STEP_TO_CALLABLE.get(step["step"])
    if builder is None:
        return None, None
    if step["step"] == "gain_per_clip":
        return builder(step, session_json)
    return builder(step)


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


def _recorded_callable(step):
    from _recall import CALLS
    args = step["args"]
    module, name = args["module"], args["function"]
    if name not in CALLS.get(module, ()) or step.get("schema_version") != 1:
        raise ValueError("Unsupported recorded operation")
    function = getattr(_lazy_import(module), name)
    kwargs = dict(args["kwargs"])
    for key in args["path_keys"]:
        kwargs[key] = Path(kwargs[key])
    inspect.signature(function).bind(**kwargs)
    return function, kwargs


def _run_recorded(step):
    from _recall import content_hash
    for path, digest in step["dependencies"].items():
        if not Path(path).is_file() or content_hash(path) != digest:
            raise ValueError(f"Recall dependency changed or missing: {path}")
    function, kwargs = _recorded_callable(step)
    function(**kwargs)
    if not Path(step["output"]).is_file() or content_hash(step["output"]) != step["output_hash"]:
        raise ValueError(f"Replayed output differs from recorded audio: {step['output']}")


def replay(chain_path: Path, dry_run: bool, stem_filter: str | None,
           use_subprocess: bool = False, allow_legacy: bool = False) -> int:
    """Preflight the entire selected chain, then stop on the first failure."""
    from _recall import content_hash
    try:
        chain = json.loads(chain_path.read_text(encoding="utf-8"))
        stems = [stem for stem in chain["stems"] if stem.get("active", True)
                 and (stem_filter is None or stem["name"] == stem_filter)]
        if not stems:
            raise ValueError("No matching active stems")
        steps = [step for stem in stems for step in stem["chain"]]
        if not steps:
            raise ValueError("No processing operations in selected chain")
        outputs = {str(Path(step["output"]).resolve()): step for step in steps}
        if len(outputs) != len(steps):
            raise ValueError("Ambiguous recall: multiple operations write the same output")
        session_json = Path(chain.get("session_json") or "session.json")
        mix_config = Path(chain["mix_config"]) if chain.get("mix_config") else None
        if stem_filter is None and (mix_config is None or not mix_config.is_file()):
            raise ValueError("Full replay requires an existing mix_config")
        if stem_filter is None:
            _lazy_import("render_mix").validate_mix_config(
                json.loads(mix_config.read_text(encoding="utf-8")), pending_files=outputs)
        prepared = {}
        for step in steps:
            if step["step"] == "recorded_call":
                function, kwargs = _recorded_callable(step)
                dependencies = step["dependencies"]
                for path, digest in dependencies.items():
                    if str(Path(path).resolve()) not in outputs:
                        if not Path(path).is_file() or content_hash(path) != digest:
                            raise ValueError(f"Recall dependency changed or missing: {path}")
            else:
                if step["step"] not in _STEP_TO_CALLABLE:
                    raise ValueError(f"Unsupported recall step: {step['step']}")
                if not allow_legacy:
                    raise ValueError("Legacy reports lack verified recall metadata; regenerate operations or explicitly use --allow-legacy for best-effort replay")
                function, kwargs = _build_inproc(step, session_json)
                inspect.signature(function).bind(**kwargs)
                dependencies = {str(value): None for key, value in kwargs.items()
                                if isinstance(value, Path) and key != "output_dir"}
                for path in dependencies:
                    if str(Path(path).resolve()) not in outputs and not Path(path).is_file():
                        raise ValueError(f"Missing recall input: {path}")
            prepared[id(step)] = (function, kwargs, dependencies)

        # Include sidechains and references in the graph, including across stems.
        ordered, seen, visiting = [], set(), set()
        def visit(step):
            key = id(step)
            if key in visiting:
                raise ValueError("Recall dependency cycle")
            if key in seen:
                return
            visiting.add(key)
            for path in prepared[key][2]:
                producer = outputs.get(str(Path(path).resolve()))
                if producer is not None:
                    visit(producer)
            visiting.remove(key)
            seen.add(key)
            ordered.append(step)
        for step in steps:
            visit(step)
        for step in ordered:
            print(f"{step['step']} -> {step['output']}")
            if dry_run:
                continue
            if step["step"] == "recorded_call":
                if use_subprocess:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                    "--operation", json.dumps(step)], check=True)
                else:
                    _run_recorded(step)
            elif use_subprocess:
                subprocess.run(_build_argv(step, session_json), check=True)
            else:
                function, kwargs, _ = prepared[id(step)]
                function(**kwargs)
            if not dry_run and not Path(step["output"]).is_file():
                raise ValueError(f"Processing did not create {step['output']}")
        if not dry_run and stem_filter is None:
            _lazy_import("render_mix").render_mix(mix_config, render_stems=True)
        print(f"Replay {'validated' if dry_run else 'completed'}: {len(ordered)} operations")
        return 0
    except (Exception, SystemExit) as exc:
        print(f"Replay stopped: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a mix_chain.json recall sheet — rebuild the mix from scratch.",
    )
    parser.add_argument("chain", type=Path, nargs="?",
                        help="Path to mix_chain.json (or the session directory containing it)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print every command without executing")
    parser.add_argument("--stem", type=str, default=None,
                        help="Replay only this single stem (for debugging)")
    parser.add_argument("--subprocess", action="store_true",
                        help="Force per-step subprocess dispatch (legacy / debugging). "
                             "Default is in-process: tool modules are imported once and "
                             "called directly, skipping the ~3-5s Python cold-start per step.")
    parser.add_argument("--allow-legacy", action="store_true", help="Allow best-effort legacy replay without verified parameters or hashes")
    parser.add_argument("--operation", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.operation:
        _run_recorded(json.loads(args.operation))
        return
    if args.chain is None:
        parser.error("chain is required")

    chain_path = _resolve_chain_path(args.chain)
    if not chain_path.exists():
        print(f"FATAL: chain file not found: {chain_path}", file=sys.stderr)
        sys.exit(2)

    rc = replay(
        chain_path,
        dry_run=args.dry_run,
        stem_filter=args.stem,
        use_subprocess=args.subprocess,
        allow_legacy=args.allow_legacy,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
