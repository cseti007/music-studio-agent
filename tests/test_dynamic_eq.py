"""Verify dynamic EQ timing, selective action, linked stereo, and recall inputs."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from apply_dynamic_eq import apply_dynamic_eq

SR = 24000


@pytest.fixture
def signals(tmp_path):
    t = np.arange(SR * 3) / SR
    x = 0.1 * np.sin(2 * np.pi * 1000 * t) + 0.05 * np.sin(2 * np.pi * 7000 * t)
    sidechain = np.zeros_like(t)
    sidechain[SR:2 * SR] = 0.3 * np.sin(2 * np.pi * 500 * t[SR:2 * SR])
    main, trigger = tmp_path / "main.wav", tmp_path / "trigger.wav"
    sf.write(main, np.column_stack((x, -0.5 * x)), SR, subtype="FLOAT")
    sf.write(trigger, sidechain, SR, subtype="FLOAT")
    return main, trigger


def amplitude(data, start, frequency):
    x = data[start:start + SR // 4, 0]
    t = np.arange(len(x)) / SR
    return abs(np.sum(x * np.exp(-2j * np.pi * frequency * t))) / len(x)


def test_cut_is_triggered_selective_and_stereo_linked(signals, tmp_path):
    main, trigger = signals
    report = apply_dynamic_eq(main, tmp_path / "out", 1000, q=2, max_cut_db=3,
                              threshold_db=-40, sidechain_path=trigger)
    x, _ = sf.read(main); y, sr = sf.read(report["output"])
    assert y.shape == x.shape and sr == SR
    np.testing.assert_array_equal(x[:SR], y[:SR])
    np.testing.assert_allclose(y[:, 1], -0.5 * y[:, 0], atol=1e-7)
    low = 20 * np.log10(amplitude(y, SR + SR // 2, 1000) / amplitude(x, SR + SR // 2, 1000))
    high = 20 * np.log10(amplitude(y, SR + SR // 2, 7000) / amplitude(x, SR + SR // 2, 7000))
    assert -3.1 < low < -2.9
    assert abs(high) < 0.1
    operation = json.loads(Path(report["output"]).with_suffix(".operation.json").read_text())
    assert str(trigger.resolve()) in operation["dependencies"]


def test_silent_trigger_is_exact_bypass(signals, tmp_path):
    main, trigger = signals
    sf.write(trigger, np.zeros(SR * 3), SR, subtype="FLOAT")
    report = apply_dynamic_eq(main, tmp_path / "out", 1000, sidechain_path=trigger)
    np.testing.assert_array_equal(sf.read(main)[0], sf.read(report["output"])[0])
    assert report["active_fraction"] == 0


def test_mismatched_trigger_timeline_is_rejected(signals, tmp_path):
    main, trigger = signals
    sf.write(trigger, np.zeros(SR), SR)
    with pytest.raises(ValueError, match="frame count"):
        apply_dynamic_eq(main, tmp_path / "out", 1000, sidechain_path=trigger)


@pytest.mark.parametrize("settings", [{"max_cut_db": -1}, {"attack_ms": 0}, {"q": 0}, {"detector_lp_hz": SR}])
def test_invalid_settings_fail(signals, tmp_path, settings):
    with pytest.raises(ValueError):
        apply_dynamic_eq(signals[0], tmp_path / "out", 1000, **settings)
