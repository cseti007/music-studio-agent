"""Coverage tests for previously-untested tools whose failures silently
produce a bad mix: parse_session, apply_compression, detect_masking,
compare_reference.

Same conventions as test_smoke.py — synthetic signals, imports inside tests,
tools/ on sys.path. The parse_session .als case is an integration test because
the gzip-XML parse is the riskiest input surface in the project.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the tools/ package importable from the project root
TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

SR = 48000


# ---------------------------------------------------------------------------
# parse_session
# ---------------------------------------------------------------------------

_ABLETON_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Ableton>
  <LiveSet>
    <MasterTrack>
      <DeviceChain>
        <Mixer>
          <Tempo>
            <Manual Value="120" />
          </Tempo>
        </Mixer>
      </DeviceChain>
    </MasterTrack>
    <Tracks>
      <AudioTrack>
        <Name>
          <EffectiveName Value="BASS DI" />
        </Name>
        <DeviceChain>
          <MainSequencer>
            <ClipTimeable>
              <ArrangerAutomation>
                <Events>
                  <AudioClip Time="4">
                    <StartRelative Value="0" />
                    <OutMarker Value="8" />
                    <SampleRef>
                      <FileRef>
                        <Path Value="/audio/bass.wav" />
                      </FileRef>
                      <SampleRate Value="48000" />
                    </SampleRef>
                  </AudioClip>
                </Events>
              </ArrangerAutomation>
            </ClipTimeable>
          </MainSequencer>
        </DeviceChain>
      </AudioTrack>
    </Tracks>
  </LiveSet>
</Ableton>
"""


def _write_als(path: Path, xml: str = _ABLETON_XML) -> Path:
    """Ableton .als is gzip-compressed XML."""
    with gzip.open(str(path), "wb") as f:
        f.write(xml.encode("utf-8"))
    return path


class TestParseSession:
    def test_detect_format(self):
        from parse_session import _detect_format

        assert _detect_format(Path("x.als")) == "ableton"
        assert _detect_format(Path("x.ptx")) == "protools"
        assert _detect_format(Path("x.pts")) == "protools"
        assert _detect_format(Path("x.ptf")) == "protools"
        with pytest.raises(ValueError):
            _detect_format(Path("x.wav"))

    def test_beats_to_samples(self):
        from parse_session import _beats_to_samples

        # 4 beats at 120 BPM = 2.0 s = 96000 samples @ 48 kHz
        assert _beats_to_samples(4, 120.0, 48000) == 96000
        assert _beats_to_samples(0, 120.0, 48000) == 0

    def test_resolve_audio_no_dir_returns_unchanged(self):
        from parse_session import _resolve_audio

        assert _resolve_audio("kick.wav", None) == "kick.wav"

    def test_resolve_audio_glob_match(self, tmp_path):
        from parse_session import _resolve_audio

        (tmp_path / "kick.wav").write_bytes(b"\x00")
        resolved = _resolve_audio("kick.wav", tmp_path)
        assert resolved == str(tmp_path / "kick.wav")

    def test_resolve_audio_unresolved_returns_filename(self, tmp_path):
        from parse_session import _resolve_audio

        assert _resolve_audio("missing.wav", tmp_path) == "missing.wav"

    def test_parse_ableton_minimal(self, tmp_path):
        from parse_session import _parse_ableton

        als = _write_als(tmp_path / "test.als")
        result = _parse_ableton(als)

        assert result["daw"] == "ableton"
        assert result["sample_rate"] == 48000
        assert result["bpm"] == 120.0
        assert len(result["tracks"]) == 1

        track = result["tracks"][0]
        assert track["name"] == "BASS DI"
        assert len(track["clips"]) == 1

        clip = track["clips"][0]
        assert clip["timeline_start_sample"] == 96000   # 4 beats
        assert clip["source_offset_sample"] == 0
        assert clip["length_samples"] == 192000          # 8 beats
        assert clip["source_file"] == "/audio/bass.wav"
        assert result["duration_samples"] == 96000 + 192000

    def test_parse_ableton_drops_zero_length_clips(self, tmp_path):
        from parse_session import _parse_ableton

        # OutMarker == StartRelative -> length 0 -> clip dropped -> track dropped
        xml = _ABLETON_XML.replace('<OutMarker Value="8" />', '<OutMarker Value="0" />')
        als = _write_als(tmp_path / "empty.als", xml)
        result = _parse_ableton(als)
        assert result["tracks"] == []

    def test_parse_session_writes_json(self, tmp_path):
        from parse_session import parse_session

        als = _write_als(tmp_path / "test.als")
        out_dir = tmp_path / "out"
        result = parse_session(als, out_dir)
        assert (out_dir / "session.json").exists()
        assert result["daw"] == "ableton"


# ---------------------------------------------------------------------------
# apply_compression
# ---------------------------------------------------------------------------

class TestApplyCompression:
    def test_auto_makeup_db(self):
        from apply_compression import _auto_makeup_db

        # -(-20) * (1 - 1/4) * 0.5 = 20 * 0.75 * 0.5 = 7.5
        assert _auto_makeup_db(-20.0, 4.0) == 7.5
        # ratio 1:1 -> no compression -> no makeup
        assert _auto_makeup_db(-20.0, 1.0) == 0.0

    def test_sidechain_envelope_ducks_on_loud_sc(self):
        from apply_compression import _sidechain_gain_envelope

        # Loud sidechain well above threshold -> some ducking (gain < 1)
        loud = np.ones(SR) * 0.9
        env = _sidechain_gain_envelope(loud, SR, threshold_db=-20.0, ratio=4.0,
                                       attack_ms=5.0, release_ms=100.0)
        assert env.min() < 0.95
        # Quiet sidechain below threshold -> essentially no ducking
        quiet = np.ones(SR) * 0.001
        env_q = _sidechain_gain_envelope(quiet, SR, threshold_db=-20.0, ratio=4.0,
                                         attack_ms=5.0, release_ms=100.0)
        assert env_q.min() > 0.99

    def test_sidechain_envelope_range(self):
        from apply_compression import _sidechain_gain_envelope

        rng = np.random.default_rng(0)
        sc = rng.standard_normal(SR) * 0.5
        env = _sidechain_gain_envelope(sc, SR, threshold_db=-30.0, ratio=8.0,
                                       attack_ms=1.0, release_ms=50.0)
        assert env.min() >= 0.0
        assert env.max() <= 1.0 + 1e-9
        assert len(env) == SR

    def test_sc_filter_highpass_removes_dc(self):
        from apply_compression import _sc_filter

        sig = (np.ones(SR) + 0.1 * np.sin(2 * np.pi * 5000 * np.arange(SR) / SR))
        out = _sc_filter(sig[:, None], SR, hp_hz=200.0, lp_hz=None).squeeze()
        # HP removes the DC/constant offset -> output mean near zero
        assert abs(out[SR // 2:].mean()) < abs(sig.mean()) * 0.1

    def test_compression_reduces_section_level_gap(self, tmp_path):
        import soundfile as sf
        from apply_compression import apply_compression

        # Sustained loud section (~-9 dBFS RMS) followed by a quiet one
        # (~-34 dBFS RMS). With threshold between them, only the loud part is
        # pulled down, so the loud-vs-quiet level gap must shrink.
        t = np.arange(SR) / SR
        loud = 0.5 * np.sin(2 * np.pi * 200 * t)
        quiet = 0.02 * np.sin(2 * np.pi * 200 * t)
        sig = np.concatenate([loud, quiet])
        in_path = tmp_path / "sections.wav"
        sf.write(str(in_path), sig, SR, subtype="PCM_24")

        def _rms_db(x):
            return 20 * np.log10(max(np.sqrt(np.mean(x ** 2)), 1e-12))

        gap_in = _rms_db(sig[:SR]) - _rms_db(sig[SR:])

        report = apply_compression(
            in_path, tmp_path, threshold_db=-20.0, ratio=6.0,
            attack_ms=5.0, release_ms=80.0, makeup_db=0.0,
        )
        out_path = Path(report["output"])
        assert out_path.exists()
        out, _ = sf.read(str(out_path))
        # Ignore the attack-settling edge of each section
        gap_out = _rms_db(out[SR // 4:SR]) - _rms_db(out[SR + SR // 4:])
        assert gap_out < gap_in - 2.0   # dynamic range meaningfully reduced


# ---------------------------------------------------------------------------
# detect_masking
# ---------------------------------------------------------------------------

class TestDetectMasking:
    def test_severity_boundaries(self):
        from detect_masking import _severity

        assert _severity(0.0) == "CRITICAL"
        assert _severity(2.9) == "CRITICAL"
        assert _severity(3.0) == "HIGH"
        assert _severity(5.9) == "HIGH"
        assert _severity(6.0) == "MODERATE"
        assert _severity(9.9) == "MODERATE"
        assert _severity(10.0) == ""   # above MODERATE -> not flagged
        assert _severity(50.0) == ""

    def test_coactivity_identical(self):
        from detect_masking import _coactivity_ratio

        env = np.array([True, True, False, True])
        assert _coactivity_ratio(env, env) == pytest.approx(1.0)

    def test_coactivity_disjoint(self):
        from detect_masking import _coactivity_ratio

        a = np.array([True, True, False, False])
        b = np.array([False, False, True, True])
        assert _coactivity_ratio(a, b) == 0.0

    def test_coactivity_empty(self):
        from detect_masking import _coactivity_ratio

        assert _coactivity_ratio(np.zeros(0, bool), np.zeros(0, bool)) == 0.0

    def test_lufs_normalize_hits_target(self):
        import pyloudnorm as pyln
        from detect_masking import _lufs_normalize

        rng = np.random.default_rng(7)
        noise = rng.standard_normal(SR * 3) * 0.05
        normed = _lufs_normalize(noise, SR, target_lufs=-18.0)
        measured = pyln.Meter(SR).integrated_loudness(normed)
        assert measured == pytest.approx(-18.0, abs=1.0)

    def test_activity_envelope_gates_silence(self):
        from detect_masking import _activity_envelope

        rng = np.random.default_rng(9)
        sig = np.zeros(SR * 4)
        sig[: SR * 2] = rng.standard_normal(SR * 2) * 0.3  # active first half only
        env = _activity_envelope(sig, SR)
        n = len(env)
        assert env[: n // 2].mean() > 0.9   # mostly active
        assert env[n // 2:].mean() < 0.1    # mostly silent

    def test_third_octave_psd_peaks_at_tone(self):
        from detect_masking import _third_octave_psd_db, _band_region_db

        t = np.arange(SR * 2) / SR
        tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
        bands = _third_octave_psd_db(tone, SR)
        near_1k = _band_region_db(bands, 900, 1100)
        near_100 = _band_region_db(bands, 90, 110)
        assert near_1k > near_100 + 20  # energy concentrated at 1 kHz


# ---------------------------------------------------------------------------
# compare_reference
# ---------------------------------------------------------------------------

class TestCompareReference:
    def test_crest_factor_sine(self):
        from compare_reference import _crest_factor_db

        t = np.arange(SR) / SR
        sine = np.sin(2 * np.pi * 1000 * t)
        # peak/rms for a sine = sqrt(2) -> 3.01 dB
        assert _crest_factor_db(sine) == pytest.approx(3.01, abs=0.1)

    def test_crest_factor_silence(self):
        from compare_reference import _crest_factor_db

        assert _crest_factor_db(np.zeros(SR)) == 0.0

    def test_filters_from_delta_inverse_sign(self):
        from compare_reference import _filters_from_delta

        delta = [{"hz": 2000.0, "delta_db": 4.0}]
        filters = _filters_from_delta(delta, threshold_db=1.0)
        assert len(filters) == 1
        assert filters[0]["type"] == "peak"
        assert filters[0]["hz"] == 2000.0
        assert filters[0]["db"] == -4.0   # inverse of the +4 dB excess

    def test_filters_from_delta_caps_gain(self):
        from compare_reference import _filters_from_delta

        delta = [{"hz": 500.0, "delta_db": 20.0}]
        filters = _filters_from_delta(delta, threshold_db=1.0)
        assert filters[0]["db"] == -6.0   # clamped to ±6 dB

    def test_filters_from_delta_respects_threshold(self):
        from compare_reference import _filters_from_delta

        delta = [{"hz": 500.0, "delta_db": 0.5}]
        assert _filters_from_delta(delta, threshold_db=2.0) == []

    def test_filters_from_delta_max_count_and_order(self):
        from compare_reference import _filters_from_delta

        delta = [{"hz": 100.0 * i, "delta_db": float(i)} for i in range(2, 12)]
        filters = _filters_from_delta(delta, threshold_db=1.0)
        assert len(filters) == 6  # capped at _APPLY_MAX_FILTERS
        # sorted by |delta| descending -> largest delta (11) first
        assert filters[0]["hz"] == 1100.0
