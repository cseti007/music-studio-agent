"""Regression tests that lock in specific bug fixes so they cannot silently
revert. Each test names the bug it guards.

These cover the fixes made in the true-peak / verdict / recall-sheet cleanup:
- worst-channel true peak (was measured on the L+R monosum)
- style_check range-based borderline severity (was hard-coded 0.0)
- build_chain basename join key (returned None for bare filenames)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

SR = 48000


# ---------------------------------------------------------------------------
# Bug: true peak measured on the L+R monosum under-reported hard-panned peaks
# by ~6 dB, letting an inter-sample-clipping mix pass the master gate as green.
# Fix: _dsp.worst_channel_true_peak_dbfs measures the loudest channel.
# ---------------------------------------------------------------------------

class TestWorstChannelTruePeak:
    def _hard_panned(self):
        """Full-scale signal in L, silence in R, as (N, 2)."""
        t = np.arange(SR) / SR
        left = 0.99 * np.sin(2 * np.pi * 1000 * t)
        right = np.zeros(SR)
        return np.stack([left, right], axis=1)

    def test_worst_channel_beats_monosum_on_hard_pan(self):
        from _dsp import true_peak_dbfs, worst_channel_true_peak_dbfs

        data = self._hard_panned()
        mono_tp = true_peak_dbfs(data.mean(axis=1))
        worst_tp = worst_channel_true_peak_dbfs(data)
        # The monosum loses ~6 dB; the worst-channel measure must recover it.
        assert worst_tp > mono_tp + 5.0

    def test_layout_agnostic(self):
        """Accepts (N, ch), (ch, N), and 1-D mono with the same result."""
        from _dsp import worst_channel_true_peak_dbfs

        data = self._hard_panned()             # (N, 2)
        tp_nch = worst_channel_true_peak_dbfs(data)
        tp_chn = worst_channel_true_peak_dbfs(data.T)   # (2, N)
        tp_mono = worst_channel_true_peak_dbfs(data[:, 0])
        assert tp_nch == pytest.approx(tp_chn)
        assert tp_nch == pytest.approx(tp_mono)  # the loud channel dominates

    def test_mix_health_uses_worst_channel(self):
        """mix_health's loudness section must report the hard-panned peak,
        not the 6-dB-quieter monosum value."""
        from mix_health import _loudness_section

        data = self._hard_panned()
        mono = data.mean(axis=1)
        section = _loudness_section(mono, data, SR, lufs_target=-14.0, tp_ceiling=-1.0)
        # Hard-panned full-scale sine -> true peak near 0 dBTP, well above the
        # ~-6 dBTP the monosum would have reported.
        assert section["true_peak_dbtp"] > -2.0


# ---------------------------------------------------------------------------
# Bug: _grade_range returned severity 0.0 for every in-range value, so the
# borderline flag (GREEN and severity >= 0.7) never fired for LRA / crest.
# Fix: severity rises from 0 at the centre to 1 at either edge.
# ---------------------------------------------------------------------------

class TestRangeBorderline:
    def test_center_is_low_severity(self):
        from style_check import _grade_range

        verdict, sev = _grade_range(9.5, 9.0, 7.0, 12.0)  # centre of [7,12]
        assert verdict == "GREEN"
        assert sev < 0.2

    def test_near_edge_is_borderline(self):
        from style_check import _grade_range

        verdict, sev = _grade_range(11.5, 9.0, 7.0, 12.0)  # near upper edge
        assert verdict == "GREEN"
        assert sev >= 0.7   # borderline flag would now fire

    def test_outside_range_is_yellow_or_red(self):
        from style_check import _grade_range

        assert _grade_range(6.5, 9.0, 7.0, 12.0)[0] in ("YELLOW", "RED")


# ---------------------------------------------------------------------------
# Bug: build_chain._basename returned None for a bare filename (no separator),
# dropping that step from the topo-sort join key and risking wrong ordering.
# Fix: Path(p).name handles bare names too.
# ---------------------------------------------------------------------------

class TestBuildChainBasename:
    def test_bare_filename_joins_in_topo_sort(self):
        import build_chain

        # _basename is a closure inside _topo_sort_chain; exercise it via a
        # two-step chain where the predecessor's output is a BARE filename
        # (no path separator) — the case that previously returned None and
        # dropped the join. Pass the steps reversed so correct ordering can
        # only come from the dependency edge, not input order.
        steps = [
            {"tool": "eq", "input": "assembled.wav", "output": "assembled_eq.wav"},
            {"tool": "gain", "input": "session.json:KICK", "output": "assembled.wav"},
        ]
        ordered = build_chain._topo_sort_chain(steps)
        tools = [s["tool"] for s in ordered]
        assert tools.index("gain") < tools.index("eq")
