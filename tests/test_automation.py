"""Verify volume rides preserve channels and timing and reject invalid plans."""
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from apply_automation import apply_automation


def test_linked_stereo_ride_preserves_timing(tmp_path):
    source = tmp_path / "source.wav"
    signal = np.column_stack([np.full(48000, .1), np.full(48000, -.2)])
    sf.write(source, signal, 48000, subtype="FLOAT")
    report = apply_automation(source, tmp_path, [[0, 0], [1, -6]])
    result, sr = sf.read(report["output"])
    assert result.shape == signal.shape and sr == 48000
    assert result[24000, 0] == pytest.approx(.1 * 10 ** (-3 / 20), abs=1e-8)
    assert np.allclose(result[:, 1], -2 * result[:, 0])


@pytest.mark.parametrize("points", [[[0, 0], [0, -6]], [[0, 0], [2, -6]], [[0, 0], [1, float('nan')]]])
def test_invalid_ride_writes_no_audio(tmp_path, points):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(48000), 48000)
    with pytest.raises(ValueError):
        apply_automation(source, tmp_path / "out", points)
    assert not (tmp_path / "out").exists()
