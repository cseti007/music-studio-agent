"""Numerical regressions for rendering, delivery, and session preservation."""

import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
SR = 48000


def test_true_peak_shortcut_cannot_hide_intersample_overload():
    from render_mix import _measure_true_peak_dbfs
    signal = np.tile([1., 1., -1., -1.], SR // 4) * 10 ** (-3.1 / 20)
    assert _measure_true_peak_dbfs(np.vstack([signal, signal])) > 0


@pytest.mark.parametrize("gain", [-6., 3.])
def test_master_eq_obeys_requested_gain(gain):
    from master_mix import _master_eq
    t = np.arange(SR * 2) / SR
    signal = .01 * np.sin(2 * np.pi * 1000 * t)
    output = _master_eq(np.vstack([signal, signal]), SR,
                        [{"type": "peak", "hz": 1000, "q": 1, "db": gain}])
    measured = 20 * np.log10(np.std(output[:, SR:]) / np.std(signal[SR:]))
    assert measured == pytest.approx(gain, abs=.02)


@pytest.mark.parametrize("frequency", [40, 100, 200, 1000, 3000, 10000])
def test_crossover_has_flat_sum_and_bounded_bands(frequency):
    from apply_multiband_comp import _lr4_split
    t = np.arange(SR * 2) / SR
    signal = np.sin(2 * np.pi * frequency * t)
    bands = _lr4_split(signal, SR, 200, 3000)
    rms = np.std(signal[SR:])
    assert np.std(sum(bands)[SR:]) / rms == pytest.approx(1, abs=.002)
    assert all(np.std(band[SR:]) / rms <= 1.001 for band in bands)
    if frequency == 100:
        assert np.std(bands[1][SR:]) / rms < .1


def test_vinyl_export_measures_saved_audio_and_preserves_headroom(tmp_path):
    from master_mix import master_mix
    from _dsp import worst_channel_true_peak_dbfs
    signal = np.random.default_rng(12).normal(0, .02, (SR * 4, 2))
    signal[SR] += .8
    source = tmp_path / "mix.wav"
    sf.write(source, signal, SR, subtype="FLOAT")
    report = master_mix(source, tmp_path, "vinyl_pre", "modern_rock_spatial_v10")
    saved, sr = sf.read(report["output"], always_2d=True)
    measured = pyln.Meter(sr).integrated_loudness(saved)
    assert report["output_lufs"] == pytest.approx(measured, abs=.011)
    assert worst_channel_true_peak_dbfs(saved, 8) <= report["tp_ceiling_dbtp"] + .01
    assert np.max(np.abs(saved)) < 1
    assert report["loudness_target_met"] is None  # vinyl_pre has no LUFS target


def test_cd_resamples_before_quantization(tmp_path):
    from master_mix import master_mix
    source = tmp_path / "mix.wav"
    sf.write(source, np.random.default_rng(1).normal(0, .02, (SR, 2)), SR)
    report = master_mix(source, tmp_path, "cd", "transparent")
    info = sf.info(report["output"])
    assert info.samplerate == 44100
    assert info.subtype == "PCM_16"


def test_alignment_keeps_common_timeline(tmp_path):
    from align_phase import align_phase
    ref = np.zeros(SR * 3)
    ref[SR + 2399:SR * 2] = np.random.default_rng(0).normal(0, .3, SR - 2399)
    ref[SR + 2399] = .5
    target = np.zeros_like(ref)
    target[240:] = ref[:-240]
    ref_path, target_path = tmp_path / "ref.wav", tmp_path / "target.wav"
    sf.write(ref_path, ref, SR, subtype="FLOAT")
    sf.write(target_path, target, SR, subtype="FLOAT")
    report = align_phase(ref_path, target_path, tmp_path / "out", segment_sec=1, threshold_db=-60)
    assert report["delay_samples"] == pytest.approx(240, abs=.1)


def test_unrelated_audio_is_not_automatically_aligned(tmp_path):
    from align_phase import align_phase
    rng = np.random.default_rng(3)
    paths = [tmp_path / "ref.wav", tmp_path / "target.wav"]
    for path in paths:
        sf.write(path, rng.normal(0, .1, SR), SR, subtype="FLOAT")
    with pytest.raises(ValueError, match="confidence"):
        align_phase(*paths, tmp_path / "out", segment_sec=1)
    assert not (tmp_path / "out").exists()


def test_audit_does_not_equate_basenames_or_different_edits(tmp_path):
    from audit_session import find_duplicates
    base = {"timeline_start_sample": 0, "source_offset_sample": 0, "length_samples": 100}
    tracks = [
        {"name": "A", "clips": [{**base, "source_file": "/take1/audio.wav"}]},
        {"name": "B", "clips": [{**base, "source_file": "/take2/audio.wav"}]},
        {"name": "C", "clips": [{**base, "source_file": "/take1/audio.wav", "source_offset_sample": 200}]},
    ]
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"tracks": tracks}))
    assert find_duplicates(session)["duplicate_groups"] == []


def test_overlapping_clips_keep_float_headroom(tmp_path):
    from apply_gain import apply_gain_per_clip
    source = tmp_path / "source.wav"
    sf.write(source, np.full(SR, .75), SR, subtype="FLOAT")
    clip = {"source_file": str(source), "timeline_start_sample": 0,
            "source_offset_sample": 0, "length_samples": SR}
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"sample_rate": SR, "duration_samples": SR,
                                  "tracks": [{"name": "Track", "clips": [clip, clip]}]}))
    reports = apply_gain_per_clip(session, tmp_path / "tracks", all_tracks=True,
                                  normalize=False, crossfade_ms=0)
    saved, _ = sf.read(reports[0]["output"])
    assert np.max(saved) == pytest.approx(1.5)


def test_low_lra_alone_does_not_imply_previous_mastering():
    from master_health import _compression_history
    signal = np.random.default_rng(5).normal(0, .02, (2, SR * 4))
    assert not _compression_history(signal, SR, pyln.Meter(SR))["likely_already_mastered"]


def test_master_report_agrees_with_independent_ffmpeg_meter(tmp_path):
    import re
    import shutil
    import subprocess
    from master_mix import master_mix
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("Independent metering requires ffmpeg")
    source = tmp_path / "mix.wav"
    sf.write(source, np.random.default_rng(4).normal(0, .02, (SR * 4, 2)), SR)
    report = master_mix(source, tmp_path, "spotify", "transparent")
    proc = subprocess.run([ffmpeg, "-hide_banner", "-i", report["output"],
                           "-af", "ebur128=peak=true", "-f", "null", "-"],
                          capture_output=True, text=True, check=True)
    summary = proc.stderr.rsplit("Summary:", 1)[-1]
    loudness = float(re.search(r"I:\s*([-\d.]+) LUFS", summary).group(1))
    peak = float(re.search(r"Peak:\s*([-\d.]+) dBFS", summary).group(1))
    assert report["output_lufs"] == pytest.approx(loudness, abs=.15)
    assert abs(report["output_lufs"] + 14) < .5
    assert peak <= -1


@pytest.mark.parametrize("audio", [np.zeros((SR, 2)), np.zeros((10, 2)), np.full((SR, 2), np.nan), np.zeros((SR, 3))])
def test_invalid_master_input_does_not_create_delivery(tmp_path, audio):
    from master_mix import master_mix
    source = tmp_path / "invalid.wav"
    sf.write(source, audio, SR, subtype="FLOAT")
    with pytest.raises(ValueError):
        master_mix(source, tmp_path / "delivery", "spotify", "transparent")
    assert not (tmp_path / "delivery").exists()


def test_default_assembly_preserves_performance_levels(tmp_path):
    from apply_gain import apply_gain_per_clip
    source = tmp_path / "source.wav"
    audio = np.random.default_rng(7).normal(0, .01, SR * 2)
    audio[SR:] *= 4
    sf.write(source, audio, SR, subtype="FLOAT")
    clips = [{"source_file": str(source), "timeline_start_sample": i * SR,
              "source_offset_sample": i * SR, "length_samples": SR} for i in range(2)]
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"sample_rate": SR, "duration_samples": SR * 2,
                                  "tracks": [{"name": "Track", "clips": clips}]}))
    reports = apply_gain_per_clip(session, tmp_path / "tracks", all_tracks=True)
    saved, _ = sf.read(reports[0]["output"])
    assert saved == pytest.approx(audio, abs=1e-8)


def test_missing_source_does_not_turn_into_silence(tmp_path):
    from apply_gain import _read_clip
    with pytest.raises(FileNotFoundError):
        _read_clip(str(tmp_path / "missing.wav"), 0, SR, SR)


def test_continuous_assembly_preserves_stereo_and_session_duration(tmp_path):
    from apply_gain import apply_gain_per_clip
    source = tmp_path / "source.wav"
    audio = np.random.default_rng(1).normal(0, .02, (SR, 2))
    sf.write(source, audio, SR, subtype="FLOAT")
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"sample_rate": SR, "duration_samples": SR * 2,
        "tracks": [{"name": "Track", "clips": [{"source_file": str(source),
        "timeline_start_sample": 0, "source_offset_sample": 0, "length_samples": SR}]}]}))
    report = apply_gain_per_clip(session, tmp_path / "tracks", all_tracks=True,
                                 source_mode="continuous")[0]
    saved, _ = sf.read(report["output"])
    assert saved.shape == (SR * 2, 2)
    assert saved[:SR] == pytest.approx(audio, abs=1e-8)
    assert not np.any(saved[SR:])


def test_out_of_range_source_offset_fails(tmp_path):
    from apply_gain import _read_clip
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(SR), SR)
    with pytest.raises(ValueError, match="offset"):
        _read_clip(str(source), SR + 1, SR, SR)


@pytest.mark.parametrize("track_ids,expected_clips", [([1, 2], 1), ([1, 1], 2)])
def test_protools_stereo_channels_are_not_summed_twice(tmp_path, monkeypatch, track_ids, expected_clips):
    import parse_session
    from types import SimpleNamespace
    sf.write(tmp_path / "stereo.wav", np.zeros((SR, 2)), SR)
    parser = tmp_path / "ptftool"
    parser.touch()
    monkeypatch.setattr(parse_session, "PTFTOOL", parser)
    lines = [f"`Vocal` t({i}) (stereo.wav) @ 0 + 0, {SR}" for i in track_ids]
    monkeypatch.setattr(parse_session.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=0, stdout="\n".join(lines), stderr=""))
    session = parse_session._parse_protools(tmp_path / "song.ptx", tmp_path)
    assert len(session["tracks"][0]["clips"]) == expected_clips


@pytest.mark.parametrize("mismatch", [None, "offset", "multiplicity"])
def test_protools_stereo_layout_comparison_ignores_listing_order(tmp_path, monkeypatch, mismatch):
    import parse_session
    from types import SimpleNamespace
    sf.write(tmp_path / "stereo.wav", np.zeros((SR * 3, 2)), SR)
    parser = tmp_path / "ptftool"
    parser.touch()
    monkeypatch.setattr(parse_session, "PTFTOOL", parser)
    lines = [f"`Vocal` t(1) (stereo.wav) @ 0 + 0, {SR}",
             f"`Vocal` t(1) (stereo.wav) @ {SR} + {SR}, {SR}",
             f"`Vocal` t(2) (stereo.wav) @ {SR} + {SR + (mismatch == 'offset')}, {SR}",
             f"`Vocal` t(2) (stereo.wav) @ 0 + 0, {SR}"]
    if mismatch == "multiplicity":
        lines.append(lines[-1])
    monkeypatch.setattr(parse_session.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=0, stdout="\n".join(lines), stderr=""))
    if mismatch:
        with pytest.raises(ValueError, match="Ambiguous"):
            parse_session._parse_protools(tmp_path / "song.ptx", tmp_path)
    else:
        session = parse_session._parse_protools(tmp_path / "song.ptx", tmp_path)
        assert len(session["tracks"][0]["clips"]) == 2


def test_reverb_predelay_is_not_removed(tmp_path):
    from apply_reverb import apply_reverb
    source = tmp_path / "impulse.wav"
    signal = np.zeros((SR, 2)); signal[0] = .25
    sf.write(source, signal, SR, subtype="FLOAT")
    dry_delay = apply_reverb(source, tmp_path / "zero", pre_delay_ms=0, send_mode=True, wet=1)
    delayed = apply_reverb(source, tmp_path / "delayed", pre_delay_ms=50, send_mode=True, wet=1)
    a, _ = sf.read(dry_delay["output"]); b, _ = sf.read(delayed["output"])
    offset = int(.05 * SR)
    assert np.any(a)
    assert not np.any(b[:offset])
    assert np.allclose(b[offset:], a[:-offset], atol=1e-8)
