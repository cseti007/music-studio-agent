"""Guard against documentation drift between the docs and the tools/ dir.

CLAUDE.md carries a large, hand-maintained tool table, and README.md /
docs/knowledge.md carry command examples. It is easy for them to reference a
tool, flag or preset that no longer exists (which misdirects the agent) or to
forget to document a newly added tool. These checks keep the docs and the
code in sync cheaply: flags and preset names are looked up in the tool source
and preset directories instead of running every tool's --help.
"""

from __future__ import annotations

import functools
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
DOCS = [CLAUDE_MD, REPO_ROOT / "README.md", REPO_ROOT / "docs" / "knowledge.md"]

# Private helper modules (leading underscore) are internal, not user-facing
# tools, so they are exempt from the "must be documented" reverse check.
_PRIVATE_PREFIX = "_"

_TOOL_REF = re.compile(r"tools/([A-Za-z0-9_]+)\.py")
_FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
# Options whose literal value must name an existing preset / profile / format.
_NAMED_VALUE = re.compile(r"--(preset|ir-preset|master-preset|style|format) ([a-z][a-z0-9_]+)\b")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tool_source(name: str) -> str:
    return (TOOLS_DIR / f"{name}.py").read_text(encoding="utf-8")


@functools.lru_cache(maxsize=None)
def _tool_help(name: str) -> str:
    """--help output, for flags generated at runtime (e.g. per-band options)."""
    proc = subprocess.run([sys.executable, str(TOOLS_DIR / f"{name}.py"), "--help"],
                          capture_output=True, text=True, timeout=120)
    return proc.stdout


def _command_lines(text: str) -> list[str]:
    """Commands that invoke a tool: fenced code lines (backslash continuations
    joined) and the Key-args cell of tool-table rows (tool name + last cell).
    Prose cells are skipped: they often cite flags of other tools."""
    lines, buf, in_code = [], "", False
    for raw in text.splitlines():
        if raw.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            if raw.rstrip().endswith("\\"):
                buf += raw.rstrip()[:-1] + " "
                continue
            line, buf = buf + raw, ""
            if _TOOL_REF.search(line):
                lines.append(line)
        elif raw.startswith("| `tools/"):
            cells = [c.strip() for c in raw.strip().strip("|").split("|")]
            lines.append(f"{cells[0]} {cells[-1]}")
    return lines


def _known_names() -> str:
    """All names a --preset/--style/--format value may legitimately refer to."""
    names = [p.stem for p in (TOOLS_DIR / "presets").glob("*.json")]
    names += [p.stem for p in (TOOLS_DIR / "style_profiles").glob("*.json")]
    names += [p.stem for p in (TOOLS_DIR / "irs").glob("*.wav")]
    sources = "\n".join(p.read_text(encoding="utf-8") for p in TOOLS_DIR.glob("*.py"))
    return "\n".join(names) + "\n" + sources


def test_every_referenced_tool_exists():
    """Every `tools/<name>.py` mentioned in the docs must exist on disk."""
    missing = sorted(
        f"{doc.name}: {name}"
        for doc in DOCS
        for name in set(_TOOL_REF.findall(_text(doc)))
        if not (TOOLS_DIR / f"{name}.py").exists()
    )
    assert not missing, f"docs reference non-existent tools: {missing}"


def test_every_tool_is_documented():
    """Every public tools/*.py must be referenced as tools/<name>.py in CLAUDE.md."""
    referenced = set(_TOOL_REF.findall(_text(CLAUDE_MD)))
    undocumented = sorted(
        p.name
        for p in TOOLS_DIR.glob("*.py")
        if not p.name.startswith(_PRIVATE_PREFIX) and p.stem not in referenced
    )
    assert not undocumented, f"tools/ files missing from CLAUDE.md: {undocumented}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_documented_flags_exist(doc):
    """Flags in a command that invokes exactly one tool must exist in that tool."""
    unknown = []
    for line in _command_lines(_text(doc)):
        tools = set(_TOOL_REF.findall(line))
        if len(tools) != 1:
            continue  # prose mentioning several tools: owner of a flag is ambiguous
        name = tools.pop()
        if not (TOOLS_DIR / f"{name}.py").exists():
            continue  # reported by test_every_referenced_tool_exists
        source = _tool_source(name)
        for flag in set(_FLAG.findall(line)):
            if f'"{flag}"' in source or f"'{flag}'" in source:
                continue
            if not re.search(rf"{re.escape(flag)}(?![\w-])", _tool_help(name)):
                unknown.append(f"{name}.py {flag}")
    assert not unknown, f"{doc.name} documents flags the tools do not define: {sorted(set(unknown))}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_documented_preset_names_exist(doc):
    """Literal --preset/--style/--format values in the docs must exist."""
    known = _known_names()
    unknown = sorted({
        f"--{opt} {value}"
        for opt, value in _NAMED_VALUE.findall(_text(doc))
        if value not in {"name"} and value not in known
    })
    assert not unknown, f"{doc.name} names presets/profiles/formats that do not exist: {unknown}"
