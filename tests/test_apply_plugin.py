"""apply_plugin.py: external plugin hosting with recall.

No VST3 plugin ships with the repo, so most tests use a stand-in object with
pedalboard's plugin interface. Set MUSIC_MIX_TEST_VST3=/path/to/plugin.vst3
to also run a smoke test against a real plugin.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import apply_plugin as ap  # noqa: E402
from replay_chain import _run_recorded  # noqa: E402

SR = 48000


class FakeParameter:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def __str__(self):
        return f"<range {self.lo}..{self.hi}>"


class FakePlugin:
    """Gain plugin with pedalboard's ExternalPlugin surface."""

    name, descriptive_name, manufacturer_name = "FakeGain", "Fake gain", "Test"
    version, identifier, category, reported_latency_samples = "1.2.3", "fake.gain", "Fx", 0

    def __init__(self):
        self.parameters = {"gain_db": FakeParameter(-24, 24), "bypass": FakeParameter(0, 1)}
        self.gain_db, self.bypass = 0.0, False

    @property
    def raw_state(self):
        return json.dumps({"gain_db": self.gain_db, "bypass": self.bypass}).encode()

    @raw_state.setter
    def raw_state(self, blob):
        state = json.loads(blob)
        self.gain_db, self.bypass = state["gain_db"], state["bypass"]

    def process(self, audio, sr, reset=True):
        return audio if self.bypass else audio * 10 ** (self.gain_db / 20)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    path = tmp_path / "FakeGain.vst3"
    (path / "Contents").mkdir(parents=True)
    (path / "Contents" / "binary").write_bytes(b"v1")
    monkeypatch.setattr(ap, "_load", lambda plugin_path, plugin_name: FakePlugin())
    return path


def _stem(tmp_path, channels=2):
    path = tmp_path / "in.wav"
    sf.write(path, np.random.default_rng(1).standard_normal((SR, channels)) * 0.1, SR, subtype="FLOAT")
    return path


def test_processes_records_identity_snapshot_and_state(tmp_path, bundle):
    src = _stem(tmp_path)
    report = ap.apply_plugin(src, tmp_path / "out", bundle, params={"gain_db": -6.0}, mix=1.0)
    out, _ = sf.read(report["output"], always_2d=True)
    ref, _ = sf.read(src, always_2d=True)
    assert np.allclose(out, ref * 10 ** (-6 / 20), atol=1e-6)
    assert report["plugin"]["version"] == "1.2.3"
    assert report["plugin"]["hash"].startswith("bundle-sha256:")
    assert report["parameters_after"]["gain_db"] == -6.0
    assert json.loads(Path(report["state_file"]).read_bytes())["gain_db"] == -6.0
    op = json.loads(Path(report["output"]).with_suffix(".operation.json").read_text())
    assert op["dependencies"][str(bundle.resolve())].startswith("bundle-sha256:")


def test_state_restore_then_params_override_and_mix(tmp_path, bundle):
    state = tmp_path / "state.bin"
    state.write_bytes(json.dumps({"gain_db": 6.0, "bypass": False}).encode())
    src = _stem(tmp_path)
    report = ap.apply_plugin(src, tmp_path / "out", bundle, state_path=state, mix=0.5)
    assert report["parameters_after"]["gain_db"] == 6.0
    out, _ = sf.read(report["output"], always_2d=True)
    ref, _ = sf.read(src, always_2d=True)
    assert np.allclose(out, ref * (0.5 + 0.5 * 10 ** (6 / 20)), atol=1e-6)
    report = ap.apply_plugin(src, tmp_path / "out2", bundle, params={"gain_db": 0.0}, state_path=state)
    assert report["parameters_after"]["gain_db"] == 0.0


def test_tail_padding_and_unknown_parameter(tmp_path, bundle):
    src = _stem(tmp_path)
    report = ap.apply_plugin(src, tmp_path / "out", bundle, tail_sec=0.5)
    assert sf.info(report["output"]).frames == SR + SR // 2
    with pytest.raises(ValueError, match="Unknown plugin parameters"):
        ap.apply_plugin(src, tmp_path / "out", bundle, params={"drive": 1.0})


def test_replay_verifies_and_detects_changed_plugin_bundle(tmp_path, bundle):
    src = _stem(tmp_path)
    report = ap.apply_plugin(src, tmp_path / "out", bundle, params={"gain_db": -3.0})
    op = json.loads(Path(report["output"]).with_suffix(".operation.json").read_text())
    _run_recorded(op)  # same plugin bundle: hash matches, output reinstalled
    (bundle / "Contents" / "binary").write_bytes(b"v2")  # plugin updated
    with pytest.raises(ValueError, match="dependency changed"):
        _run_recorded(op)


@pytest.mark.skipif(not os.environ.get("MUSIC_MIX_TEST_VST3"), reason="set MUSIC_MIX_TEST_VST3 to a plugin")
def test_real_plugin_smoke(tmp_path):
    plugin = Path(os.environ["MUSIC_MIX_TEST_VST3"])
    report = ap.apply_plugin(_stem(tmp_path), tmp_path / "out", plugin)
    assert Path(report["output"]).is_file() and Path(report["state_file"]).stat().st_size > 0
