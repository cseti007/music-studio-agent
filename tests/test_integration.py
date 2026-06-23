"""End-to-end smoke test on tiny synthetic stems.

Builds a minimal two-stem session, runs the real generate-config -> render ->
mix_health flow, and asserts the artifacts land where users expect them. This
exercises the integration-level fixes that the unit tests can't reach:

- stems are written to <session>/stems even for a --stage render (not into
  <session>/mixes/stems)
- the premaster verdict reflects the peak-normalized handoff level

It is deliberately small (2 s stems) so it runs in a few seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

SR = 48000
DUR = 2  # seconds


def _kick(n: int) -> np.ndarray:
    """Repeated decaying low-frequency bursts — drum-like."""
    sig = np.zeros(n)
    period = SR // 2  # 2 hits/sec
    t = np.arange(period) / SR
    burst = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 25.0)
    for onset in range(0, n - period, period):
        sig[onset:onset + period] += burst
    return 0.5 * sig


def _bass(n: int) -> np.ndarray:
    """Sustained 80 Hz tone — bass-like."""
    t = np.arange(n) / SR
    return 0.3 * np.sin(2 * np.pi * 80 * t)


@pytest.fixture
def session(tmp_path):
    """A minimal session: tracks/<name>/assembled.wav for two stems."""
    n = SR * DUR
    tracks = {"KICK": _kick(n), "BASS DI": _bass(n)}
    for name, sig in tracks.items():
        d = tmp_path / "tracks" / name
        d.mkdir(parents=True)
        sf.write(str(d / "assembled.wav"), sig.astype(np.float32), SR, subtype="PCM_24")
    return tmp_path


class TestRenderPipeline:
    def test_generate_config_and_render(self, session):
        import render_mix

        config_path = session / "mix_config.json"
        render_mix.generate_config(session, config_path)
        assert config_path.exists()

        cfg = json.loads(config_path.read_text())
        # Two stems detected onto real buses (not the fallback 'master')
        assert len(cfg["tracks"]) == 2
        buses = {t["bus"] for t in cfg["tracks"]}
        assert "drums" in buses and "bass" in buses

        render_mix.render_mix(config_path, render_stems=True)

        mix_wav = session / "mixes" / "mix.wav"
        report_path = session / "mixes" / "mix_report.json"
        assert mix_wav.exists()
        assert report_path.exists()

        report = json.loads(report_path.read_text())
        # Premaster handoff: peak-normalized to ~-3 dBFS, no LUFS target baked in
        assert report["mix_stage"] == "premaster"
        assert isinstance(report["integrated_lufs"], (int, float))
        assert report["sample_peak_dbfs"] == pytest.approx(-3.0, abs=0.5)
        assert report["master_peaks"]["verdict"] in ("[OK]", "[WARN]", "[CLIP]")

        # The #2 fix: stems live under <session>/stems, NOT <session>/mixes/stems
        stems_dir = session / "stems"
        assert stems_dir.is_dir()
        assert list(stems_dir.glob("stem_*.wav"))
        assert not (session / "mixes" / "stems").exists()

    def test_stage_render_writes_stems_to_session_root(self, session):
        """Regression for the --stage stems path: the stage WAV goes to
        mixes/stages/, but the stems must still land in <session>/stems."""
        import render_mix

        config_path = session / "mix_config.json"
        render_mix.generate_config(session, config_path)
        render_mix.render_mix(config_path, render_stems=True, stage="raw")

        assert (session / "mixes" / "stages" / "mix_stage_raw.wav").exists()
        assert list((session / "stems").glob("stem_*.wav"))
        # The bug put stems here; the fix must not:
        assert not (session / "mixes" / "stems").exists()

    def test_mix_health_runs_and_scores(self, session):
        import render_mix
        import mix_health

        config_path = session / "mix_config.json"
        render_mix.generate_config(session, config_path)
        render_mix.render_mix(config_path, render_stems=True)

        out_dir = session / "analysis"
        report = mix_health.mix_health(session, out_dir)

        assert (out_dir / "mix_health.json").exists()
        assert (out_dir / "mix_health.txt").exists()
        # Loudness section present with the worst-channel true peak
        assert "true_peak_dbtp" in report["loudness"]
        assert report["loudness"]["true_peak_verdict"] in ("[OK]", "[!] ", "[X] ")
        # The scorecard text carries an overall verdict line
        txt = (out_dir / "mix_health.txt").read_text()
        assert "OVERALL:" in txt
