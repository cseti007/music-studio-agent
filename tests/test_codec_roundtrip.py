"""codec_roundtrip.py: encode/decode with the local ffmpeg and measure peaks."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from codec_roundtrip import codec_roundtrip  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
SR = 48000


def _hot_master(path: Path, peak: float) -> Path:
    t = np.arange(SR * 3) / SR
    tone = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((110, 220, 440, 1760), 1))
    sig = np.stack([np.tanh(2 * tone), np.tanh(1.8 * tone)], axis=1)
    # 50 ms fades: avoid encoder edge transients on an abrupt full-scale start/end.
    sig *= np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.05)[:, None]
    sig *= peak / np.max(np.abs(sig))
    sf.write(path, sig, SR, subtype="FLOAT")
    return path


def test_roundtrip_measures_decoded_audio_and_keeps_it(tmp_path):
    src = _hot_master(tmp_path / "master.wav", 10 ** (-0.3 / 20))
    report = codec_roundtrip(src, tmp_path / "rt", codecs=["aac"], tp_ceiling=-1.0)
    aac = report["codecs"]["aac"]
    assert Path(aac["decoded_file"]).is_file()
    # A -0.3 dBFS hot master cannot sit within a -1 dBTP ceiling after decoding.
    assert aac["within_ceiling"] is False
    assert report["all_within_ceiling"] is False
    assert report["listening_review"]["status"] == "pending"
    saved = json.loads((tmp_path / "rt" / "codec_roundtrip.json").read_text())
    assert saved["artifact_hash"] == report["artifact_hash"]


def test_quiet_master_passes_ceiling(tmp_path):
    src = _hot_master(tmp_path / "master.wav", 10 ** (-6 / 20))
    report = codec_roundtrip(src, tmp_path / "rt", codecs=["aac"], tp_ceiling=-1.0)
    assert report["codecs"]["aac"]["within_ceiling"] is True
    assert report["codecs"]["aac"]["samples_over_full_scale"] == 0


def test_rejects_unknown_codec_and_positive_ceiling(tmp_path):
    src = _hot_master(tmp_path / "master.wav", 0.5)
    with pytest.raises(ValueError):
        codec_roundtrip(src, tmp_path / "rt", codecs=["flacx"])
    with pytest.raises(ValueError):
        codec_roundtrip(src, tmp_path / "rt", tp_ceiling=0.5)
