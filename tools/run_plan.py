"""Run a declarative processing plan: an ordered list of tool invocations.

A plan replaces ad-hoc driver scripts. Each step names one public tool in
tools/ and its command-line arguments; the tools record their own
.operation.json files as usual, so a plan adds orchestration and a run log,
not a second recall format. Only tools/*.py can be run (no shell commands).

Plan file (JSON):

  {
    "schema_version": 1,
    "vars": {"session": "output/song"},
    "steps": [
      {"id": "analyze", "tool": "batch_analyze", "args": ["${session}", "--workers", "8"]},
      {"id": "kick-eq", "tool": "apply_eq",
       "foreach": "${session}/tracks/KICK*/assembled.wav", "parallel": 4,
       "args": ["${file}", "--preset", "kick_in", "--output-dir", "${dir}"]},
      {"id": "render", "tool": "render_mix",
       "args": ["${session}/mix_config.json", "--render", "--stems"]},
      {"id": "codec", "tool": "codec_roundtrip", "allow_exit_codes": [0, 1],
       "args": ["${session}/masters/master_spotify.wav", "--output-dir", "${session}/codec"]}
    ]
  }

${name} expands plan vars (overridable with --var name=value). In a foreach
step, ${file}, ${dir}, ${name} (parent folder, e.g. the track) and ${stem}
(file name without suffix) describe each matched file; a foreach that
matches nothing is an error unless "allow_empty" is true. Paths are relative
to the working directory. Steps run in order and the run stops at the first
step whose exit code is not allowed (default: only 0). Every run writes
<plan>.run.json (commands, exit codes, durations, plan hash, git commit) and
<plan>.run.log (tool output) next to the plan.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import string
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
STEP_KEYS = {"id", "tool", "args", "foreach", "parallel", "allow_empty", "allow_exit_codes", "note"}


def _expand(text: str, mapping: dict[str, str], where: str) -> str:
    try:
        return string.Template(text).substitute(mapping)
    except KeyError as exc:
        raise ValueError(f"{where}: unknown variable ${{{exc.args[0]}}}") from None
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None


def load_plan(path: Path, overrides: dict[str, str] | None = None) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1:
        raise ValueError("Plan schema_version must be 1")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Plan needs a nonempty 'steps' list")
    variables = {k: str(v) for k, v in plan.get("vars", {}).items()}
    variables.update(overrides or {})
    seen = set()
    for i, step in enumerate(steps):
        where = f"step {i + 1}"
        if not isinstance(step, dict) or not isinstance(step.get("tool"), str):
            raise ValueError(f"{where}: each step needs a 'tool' name")
        unknown = set(step) - STEP_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        step.setdefault("id", f"{i + 1}-{step['tool']}")
        if step["id"] in seen:
            raise ValueError(f"{where}: duplicate step id {step['id']!r}")
        seen.add(step["id"])
        tool = TOOLS_DIR / f"{step['tool']}.py"
        if step["tool"].startswith("_") or "/" in step["tool"] or not tool.is_file():
            raise ValueError(f"{where}: {step['tool']!r} is not a public tool in tools/")
        if not all(isinstance(a, (str, int, float)) for a in step.get("args", [])):
            raise ValueError(f"{where}: args must be strings or numbers")
        if int(step.get("parallel", 1)) < 1:
            raise ValueError(f"{where}: parallel must be >= 1")
    return {"vars": variables, "steps": steps}


_FILE_VARS = ("file", "dir", "name", "stem")


def check_step(step: dict, variables: dict[str, str]) -> None:
    """Fail on unknown variables before anything runs (foreach files may not exist yet)."""
    where = f"step {step['id']!r}"
    local = dict(variables)
    if "foreach" in step:
        _expand(step["foreach"], variables, where)
        local.update({k: k for k in _FILE_VARS})
    for a in step.get("args", []):
        _expand(str(a), local, where)


def expand_step(step: dict, variables: dict[str, str]) -> list[list[str]]:
    """Command lines for one step (several for a foreach step)."""
    where = f"step {step['id']!r}"
    args = [str(a) for a in step.get("args", [])]
    tool = [sys.executable, str(TOOLS_DIR / f"{step['tool']}.py")]
    if "foreach" not in step:
        return [tool + [_expand(a, variables, where) for a in args]]
    pattern = _expand(step["foreach"], variables, where)
    files = sorted(glob.glob(pattern))
    if not files and not step.get("allow_empty", False):
        raise ValueError(f"{where}: foreach pattern {pattern!r} matched no files")
    commands = []
    for f in files:
        p = Path(f)
        local = {**variables, "file": str(p), "dir": str(p.parent), "name": p.parent.name, "stem": p.stem}
        commands.append(tool + [_expand(a, local, where) for a in args])
    return commands


def _git_commit() -> str | None:
    proc = subprocess.run(["git", "-C", str(TOOLS_DIR), "rev-parse", "HEAD"], capture_output=True, text=True)
    return proc.stdout.strip() or None


def run_plan(plan_path: Path, dry_run: bool = False, only: list[str] | None = None,
             start_from: str | None = None, overrides: dict[str, str] | None = None) -> dict:
    plan = load_plan(plan_path, overrides)
    ids = [s["id"] for s in plan["steps"]]
    for name in (only or []) + ([start_from] if start_from else []):
        if name not in ids:
            raise ValueError(f"Unknown step id {name!r}; plan has {ids}")
    selected = [s for s in plan["steps"]
                if (not only or s["id"] in only)
                and (start_from is None or ids.index(s["id"]) >= ids.index(start_from))]
    for step in selected:
        check_step(step, plan["vars"])
    report = {"plan": str(plan_path.resolve()),
              "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              "git_commit": _git_commit(), "vars": plan["vars"], "dry_run": dry_run,
              "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "steps": [], "status": "running"}
    log_path = plan_path.with_suffix(".run.log")
    log = None if dry_run else log_path.open("a", encoding="utf-8")
    try:
        for step in selected:
            if dry_run:
                # foreach files may be produced by earlier steps: show the template then.
                pattern = _expand(step["foreach"], plan["vars"], step["id"]) if "foreach" in step else None
                if pattern is not None and not glob.glob(pattern):
                    commands = []
                    print(f"[{step['id']}] for each {pattern}: {step['tool']} {' '.join(map(str, step.get('args', [])))}")
                else:
                    commands = expand_step(step, plan["vars"])
                    for c in commands:
                        print(f"[{step['id']}] " + " ".join(c[1:]))
                report["steps"].append({"id": step["id"], "tool": step["tool"],
                                        "commands": [c[1:] for c in commands], "status": "planned"})
                continue
            commands = expand_step(step, plan["vars"])
            entry = {"id": step["id"], "tool": step["tool"], "commands": [c[1:] for c in commands]}
            report["steps"].append(entry)
            allowed = set(step.get("allow_exit_codes", [0]))
            print(f"[{step['id']}] {step['tool']} ({len(commands)} command(s))", flush=True)
            started = time.monotonic()

            def run(cmd):
                proc = subprocess.run(cmd, capture_output=True, text=True)
                return cmd, proc
            with ThreadPoolExecutor(max_workers=int(step.get("parallel", 1))) as pool:
                results = list(pool.map(run, commands))
            entry["exit_codes"] = [proc.returncode for _, proc in results]
            entry["duration_sec"] = round(time.monotonic() - started, 2)
            for cmd, proc in results:
                log.write(f"\n=== [{step['id']}] {' '.join(cmd[1:])} -> exit {proc.returncode}\n"
                          f"{proc.stdout}{proc.stderr}")
            log.flush()
            failed = [code for code in entry["exit_codes"] if code not in allowed]
            entry["status"] = "failed" if failed else "ok"
            if failed:
                report["status"] = "failed"
                report["failed_step"] = step["id"]
                tail = (results[entry["exit_codes"].index(failed[0])][1].stderr or "").strip()[-800:]
                print(f"[{step['id']}] FAILED (exit {failed[0]}); see {log_path}\n{tail}", file=sys.stderr)
                break
        else:
            report["status"] = "planned" if dry_run else "ok"
    finally:
        if log is not None:
            log.close()
    report["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not dry_run:
        plan_path.with_suffix(".run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path, help="Plan JSON file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the plan and print the expanded commands; run nothing")
    parser.add_argument("--only", action="append", metavar="ID", help="Run only this step (repeatable)")
    parser.add_argument("--from", dest="start_from", metavar="ID", help="Start at this step")
    parser.add_argument("--var", action="append", default=[], metavar="NAME=VALUE",
                        help="Override a plan variable (repeatable)")
    args = parser.parse_args()
    overrides = {}
    for item in args.var:
        name, sep, value = item.partition("=")
        if not sep or not name:
            parser.error(f"--var expects NAME=VALUE, got {item!r}")
        overrides[name] = value
    try:
        report = run_plan(args.plan, args.dry_run, args.only, args.start_from, overrides)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"run_plan: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if report["status"] == "planned":
        print(f"Plan validated (dry run): {len(report['steps'])} step(s)")
    else:
        print(f"Plan {report['status']}: {sum(s.get('status') == 'ok' for s in report['steps'])} step(s) ran")
    raise SystemExit(0 if report["status"] in ("ok", "planned") else 1)


if __name__ == "__main__":
    main()
