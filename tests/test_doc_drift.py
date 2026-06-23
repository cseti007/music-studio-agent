"""Guard against documentation drift between CLAUDE.md and the tools/ dir.

CLAUDE.md carries a large, hand-maintained tool table. It is easy for it to
reference a tool that no longer exists (a phantom tool misdirects the agent)
or to forget to document a newly added tool. These two checks keep the table
and the filesystem in sync cheaply.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"

# Private helper modules (leading underscore) are internal, not user-facing
# tools, so they are exempt from the "must be documented" reverse check.
_PRIVATE_PREFIX = "_"


def _claude_md_text() -> str:
    return CLAUDE_MD.read_text(encoding="utf-8")


def test_every_referenced_tool_exists():
    """Every `tools/<name>.py` mentioned in CLAUDE.md must exist on disk."""
    text = _claude_md_text()
    referenced = set(re.findall(r"tools/([A-Za-z0-9_]+)\.py", text))
    missing = sorted(
        name for name in referenced if not (TOOLS_DIR / f"{name}.py").exists()
    )
    assert not missing, f"CLAUDE.md references non-existent tools: {missing}"


def test_every_tool_is_documented():
    """Every public tools/*.py must be mentioned somewhere in CLAUDE.md."""
    text = _claude_md_text()
    undocumented = sorted(
        p.name
        for p in TOOLS_DIR.glob("*.py")
        if not p.name.startswith(_PRIVATE_PREFIX) and p.name not in text
    )
    assert not undocumented, f"tools/ files missing from CLAUDE.md: {undocumented}"
