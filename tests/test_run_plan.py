"""run_plan.py: declarative tool orchestration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from run_plan import load_plan, run_plan  # noqa: E402

SR = 48000


def _session(tmp_path: Path) -> Path:
    rng = np.random.default_rng(5)
    for name in ("KICK IN.01", "KICK OUT.01", "SN TOP.01"):
        d = tmp_path / "session" / "tracks" / name
        d.mkdir(parents=True)
        sf.write(d / "assembled.wav", rng.standard_normal((SR, 2)) * 0.1, SR, subtype="FLOAT")
    return tmp_path / "session"


def _write_plan(tmp_path: Path, steps: list, vars_: dict | None = None) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema_version": 1, "vars": vars_ or {}, "steps": steps}))
    return path


def test_foreach_parallel_runs_tool_per_file_and_writes_run_log(tmp_path):
    session = _session(tmp_path)
    plan = _write_plan(tmp_path, [{
        "id": "kick-eq", "tool": "apply_eq", "foreach": "${session}/tracks/KICK*/assembled.wav",
        "parallel": 2, "args": ["${file}", "--filter", '{"type": "highpass", "hz": 30}',
                                "--output-dir", "${dir}"]}], {"session": str(session)})
    report = run_plan(plan)
    assert report["status"] == "ok"
    assert report["steps"][0]["exit_codes"] == [0, 0]
    for name in ("KICK IN.01", "KICK OUT.01"):
        assert (session / "tracks" / name / "assembled_eq.wav").is_file()
        assert (session / "tracks" / name / "assembled_eq.operation.json").is_file()
    assert not (session / "tracks" / "SN TOP.01" / "assembled_eq.wav").exists()
    run = json.loads(plan.with_suffix(".run.json").read_text())
    assert run["plan_sha256"] and run["steps"][0]["status"] == "ok"
    assert "kick-eq" in plan.with_suffix(".run.log").read_text()


def test_dry_run_executes_nothing(tmp_path, capsys):
    session = _session(tmp_path)
    plan = _write_plan(tmp_path, [{"tool": "apply_eq", "foreach": "${s}/tracks/*/assembled.wav",
                                   "args": ["${file}", "--preset", "kick_in", "--output-dir", "${dir}"]}],
                       {"s": str(session)})
    report = run_plan(plan, dry_run=True)
    assert report["status"] == "planned" and len(report["steps"][0]["commands"]) == 3
    assert not list(session.glob("tracks/*/assembled_eq.wav"))
    assert not plan.with_suffix(".run.json").exists()
    assert "apply_eq.py" in capsys.readouterr().out


def test_stops_at_first_failing_step_and_honours_allowed_exit_codes(tmp_path):
    session = _session(tmp_path)
    missing = str(session / "nope.wav")
    plan = _write_plan(tmp_path, [
        {"id": "tolerated", "tool": "apply_eq", "args": [missing, "--output-dir", str(tmp_path)],
         "allow_exit_codes": [0, 1, 2]},
        {"id": "fails", "tool": "apply_eq", "args": [missing, "--output-dir", str(tmp_path)]},
        {"id": "never", "tool": "analyze", "args": [missing]},
    ])
    report = run_plan(plan)
    assert report["status"] == "failed" and report["failed_step"] == "fails"
    assert [s["id"] for s in report["steps"]] == ["tolerated", "fails"]


@pytest.mark.parametrize("steps, message", [
    ([{"tool": "rm"}], "not a public tool"),
    ([{"tool": "_recall"}], "not a public tool"),
    ([{"tool": "apply_eq", "args": ["${undefined}"]}], "unknown variable"),
    ([{"tool": "apply_eq", "shell": "ls"}], "unknown keys"),
    ([{"id": "a", "tool": "analyze"}, {"id": "a", "tool": "analyze"}], "duplicate step id"),
])
def test_invalid_plans_fail_before_running(tmp_path, steps, message):
    plan = _write_plan(tmp_path, steps)
    with pytest.raises(ValueError, match=message):
        run_plan(plan)


def test_var_override_and_only_selection(tmp_path):
    plan = _write_plan(tmp_path, [{"id": "a", "tool": "analyze", "args": ["${x}"]},
                                  {"id": "b", "tool": "analyze", "args": ["${x}"]}], {"x": "one"})
    loaded = load_plan(plan, {"x": "two"})
    assert loaded["vars"]["x"] == "two"
    report = run_plan(plan, dry_run=True, only=["b"], overrides={"x": "two"})
    assert [s["id"] for s in report["steps"]] == ["b"]
    assert report["steps"][0]["commands"][0][-1] == "two"
