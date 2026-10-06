"""Check alias reduction, block continuity, and saturation parameter safety."""

from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from apply_saturation import apply_saturation, _oversampled


def test_oversampling_reduces_folded_harmonic(tmp_path):
    sr = 48000
    t = np.arange(sr * 2) / sr
    audio = .2 * np.sin(2 * np.pi * 11000 * t)
    source = tmp_path / "tone.wav"
    sf.write(source, audio, sr, subtype="FLOAT")
    aliases = []
    for factor in (1, 4):
        report = apply_saturation(source, tmp_path / str(factor), drive=1,
                                  input_gain_db=12, oversample=factor)
        output, rate = sf.read(report["output"])
        assert len(output) == len(audio) and rate == sr and np.isfinite(output).all()
        spectrum = np.abs(np.fft.rfft(output[sr // 2:sr // 2 + sr]))
        aliases.append(spectrum[15000] / spectrum[11000])
    assert aliases[1] < aliases[0] * .1


def test_blocks_match_continuous_processing():
    rng = np.random.default_rng(42)
    mono = rng.normal(0, .07, 140003)
    audio = np.column_stack([mono, -mono])
    result = _oversampled(audio, "tape", .7, .5, 4, 8)
    up = resample_poly(audio, 4, 1, axis=0)
    expected = resample_poly(np.tanh(up * (1 + .7 * 4) * 10 ** (8 / 20)), 1, 4, axis=0)
    expected *= np.sqrt(np.mean(audio ** 2) / np.mean(expected ** 2))
    np.testing.assert_allclose(result, expected, atol=1e-10)
    np.testing.assert_allclose(result[:, 0], -result[:, 1], atol=1e-12)


def test_dry_bypass_preserves_float_headroom(tmp_path):
    source = tmp_path / "float.wav"
    audio = np.array([0, 1.5, -.5, 0], dtype=np.float32)
    sf.write(source, audio, 48000, subtype="FLOAT")
    report = apply_saturation(source, tmp_path / "out", mix=0, oversample=4)
    result, _ = sf.read(report["output"], dtype="float32")
    np.testing.assert_array_equal(result, audio)


@pytest.mark.parametrize("params", [{"mix": -1}, {"drive": 2}, {"oversample": 3},
                                     {"input_gain_db": float("nan")}])
def test_invalid_parameters_rejected(tmp_path, params):
    with pytest.raises(ValueError):
        apply_saturation(tmp_path / "unused.wav", tmp_path / "out", **params)
