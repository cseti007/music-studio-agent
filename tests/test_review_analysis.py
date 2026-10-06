"""Regression tests for review findings in the analysis tools.

Covers analyze.py, detect_masking.py, compare_reference.py and style_check.py.
All signals are synthetic; no session data is required.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

SR = 48000


def _band_noise(lo, hi, seconds, seed, sr=SR):
    """Brick-wall band-limited white noise, peak 0.5."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    spec[(f < lo) | (f > hi)] = 0.0
    x = np.fft.irfft(spec, n)
    return 0.5 * x / np.max(np.abs(x))


def _pink_noise(seconds, seed, sr=SR):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0.0
    x = np.fft.irfft(spec, n)
    return 0.5 * x / np.max(np.abs(x))


def _strict_json(path):
    def reject(constant):
        raise ValueError(f"non-standard JSON constant {constant}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


# ---------------------------------------------------------------------------
# detect_masking
# ---------------------------------------------------------------------------

def _masking_bands(sig):
    from detect_masking import (
        _activity_envelope, _gated_audio, _lufs_normalize, _third_octave_psd_db,
    )
    x = _lufs_normalize(sig, SR, -18.0)
    active = _activity_envelope(x, SR)
    return _third_octave_psd_db(_gated_audio(x, SR, active), SR), active


def test_masking_pink_noise_clears_floor_in_most_bands():
    from detect_masking import _DEFAULT_FLOOR_DB
    bands, _ = _masking_bands(_pink_noise(6, 1))
    above = sum(b["db"] >= _DEFAULT_FLOOR_DB for b in bands)
    assert above >= len(bands) - 2


def test_masking_overlapping_noise_flags_shared_bands():
    from detect_masking import find_masking_pairs
    a, act_a = _masking_bands(_band_noise(200, 5000, 6, 1))
    b, act_b = _masking_bands(_band_noise(200, 5000, 6, 2))
    events = find_masking_pairs({"a": a, "b": b}, {"a": act_a, "b": act_b}, threshold_db=6.0)
    hz = {e["hz"] for e in events}
    assert len(hz) >= 8
    edge = 2.0 ** (1.0 / 6.0)
    assert all(h * edge >= 200 and h / edge <= 5000 for h in hz)


def test_masking_disjoint_noise_has_no_events():
    from detect_masking import find_masking_pairs
    a, act_a = _masking_bands(_band_noise(80, 400, 6, 1))
    b, act_b = _masking_bands(_band_noise(2000, 8000, 6, 2))
    events = find_masking_pairs({"a": a, "b": b}, {"a": act_a, "b": act_b}, threshold_db=10.0)
    assert events == []


def test_coactivity_short_part_fully_inside_long_part():
    from detect_masking import _coactivity_ratio
    bass = np.ones(100, dtype=bool)
    solo = np.zeros(100, dtype=bool)
    solo[40:50] = True
    assert _coactivity_ratio(bass, solo) == pytest.approx(1.0)


def test_report_wording_is_hypothesis_level(tmp_path):
    from detect_masking import detect_masking
    paths = {}
    for i, name in enumerate(("a", "b")):
        p = tmp_path / f"{name}.wav"
        sf.write(p, _band_noise(200, 5000, 4, i), SR)
        paths[name] = p
    report = detect_masking(paths, tmp_path / "out", threshold_db=10.0, stage="manual")
    text = (tmp_path / "out" / "masking_report.txt").read_text(encoding="utf-8")
    assert "almost certainly audible" not in text
    assert "Cut the non-hero" not in text
    assert "dBFS" not in text
    assert "normaliz" in report["normalization_note"].lower()


def _touch_wav(path, seconds=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(SR * seconds)), SR)


def test_fx_stage_picks_final_file_and_never_a_send(tmp_path):
    from detect_masking import _find_stem_file
    d = tmp_path / "GTR"
    for name in ("assembled.wav", "assembled_eq.wav", "assembled_eq_comp.wav",
                 "assembled_eq_comp_reverb_send.wav", "assembled_eq_comp_delay_send.wav"):
        _touch_wav(d / name)
    assert _find_stem_file(d, "fx").name == "assembled_eq_comp.wav"

    _touch_wav(d / "assembled_eq_comp_amp.wav")
    _touch_wav(d / "assembled_eq_comp_amp_reverb.wav")
    assert _find_stem_file(d, "fx").name == "assembled_eq_comp_amp_reverb.wav"


def test_fx_stage_aligned_comp_is_not_treated_as_fx(tmp_path):
    from detect_masking import _find_stem_file
    d = tmp_path / "KICK"
    for name in ("assembled.wav", "assembled_aligned.wav", "assembled_aligned_eq.wav",
                 "assembled_aligned_eq_comp.wav", "assembled_aligned_eq_comp_reverb_send.wav"):
        _touch_wav(d / name)
    assert _find_stem_file(d, "fx").name == "assembled_aligned_eq_comp.wav"


def test_discover_stems_honours_mix_config(tmp_path):
    from detect_masking import discover_stems
    session = tmp_path / "session"
    for track in ("A", "B"):
        _touch_wav(session / "tracks" / track / "assembled.wav")
        _touch_wav(session / "tracks" / track / "assembled_eq.wav")
        _touch_wav(session / "tracks" / track / "assembled_eq_reverb.wav")
    config = {"tracks": [
        {"name": "A", "file": str(session / "tracks/A/assembled_eq.wav"), "active": True},
        {"name": "B", "file": str(session / "tracks/B/assembled_eq.wav"), "active": False},
    ]}
    (session / "mix_config.json").write_text(json.dumps(config))
    comp = discover_stems(session, "comp")
    assert set(comp) == {"A"}
    fx = discover_stems(session, "fx")
    assert set(fx) == {"A"}
    assert fx["A"].name == "assembled_eq.wav"


# ---------------------------------------------------------------------------
# analyze.py - hum detection
# ---------------------------------------------------------------------------

def _bass_line(notes_hz, seconds=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    out = np.zeros_like(t)
    for i in range(int(seconds)):
        f0 = notes_hz[i % len(notes_hz)]
        seg = slice(i * SR, (i + 1) * SR)
        tt = t[seg] - i
        tone = sum((0.5 / k) * np.sin(2 * np.pi * k * f0 * tt) for k in range(1, 6))
        out[seg] = 0.4 * tone * np.exp(-2.0 * tt)
    return out + 1e-4 * rng.standard_normal(len(t))


@pytest.mark.parametrize("notes", [(49.0, 98.0), (61.74,), (49.0,)])
def test_hum_not_flagged_on_bass_notes(notes):
    from analyze import _detect_hum
    result = _detect_hum(_bass_line(notes), SR)
    assert result["hum_detected"] is False, result


def test_hum_detected_under_quiet_noise():
    from analyze import _detect_hum
    rng = np.random.default_rng(3)
    seconds = 16
    t = np.arange(SR * seconds) / SR
    hum = 3e-3 * np.sin(2 * np.pi * 50 * t) + 1.5e-3 * np.sin(2 * np.pi * 100 * t) \
        + 1e-3 * np.sin(2 * np.pi * 150 * t)
    sig = hum + 1e-3 * rng.standard_normal(len(t))
    # Loud musical passage in the first half; the quiet half still carries hum.
    sig[: SR * 8] += _bass_line((55.0, 82.4), seconds=8, seed=4)
    result = _detect_hum(sig, SR)
    assert result["hum_detected"] is True
    assert result["dominant_mains_hz"] == 50
    freqs = {h["frequency_hz"] for h in result["harmonics"]["50hz"]}
    assert {50, 100}.issubset(freqs)
    assert "high-pass" not in result["recommendation"].lower()
    assert "notch" in result["recommendation"].lower()


# ---------------------------------------------------------------------------
# analyze.py - vocal metrics
# ---------------------------------------------------------------------------

FRAME_RATE = SR / 512.0


def _melody_f0(cents_offset, seed=0, jitter=2.0):
    """In-tune melody (whole semitones) with a global tuning offset."""
    rng = np.random.default_rng(seed)
    notes = [0, 2, 4, 5, 7, 5, 4, 2]
    frames = []
    for n in notes:
        cents = n * 100 + cents_offset + jitter * rng.standard_normal(int(0.4 * FRAME_RATE))
        frames.append(220.0 * 2.0 ** (cents / 1200.0))
        frames.append(np.full(5, np.nan))  # unvoiced gap between notes
    return np.concatenate(frames)


def test_pitch_stats_removes_global_tuning_offset():
    from analyze import _pitch_stats
    stats = _pitch_stats(_melody_f0(cents_offset=49.0), FRAME_RATE)
    assert stats["cents_std"] < 8.0
    assert stats["fraction_over_25_cents"] < 0.05
    assert abs(abs(stats["tuning_offset_cents"]) - 49.0) < 5.0 or \
        abs(abs(stats["tuning_offset_cents"]) - 51.0) < 5.0


def test_pitch_stats_unbounded_by_semitone_wrap():
    from analyze import _pitch_stats
    # A sustained note that drifts over 160 cents: the old wrapped metric
    # folded the far end back into +-50 cents.
    cents = np.linspace(-40, 120, int(2 * FRAME_RATE))
    f0 = 220.0 * 2.0 ** (cents / 1200.0)
    stats = _pitch_stats(f0, FRAME_RATE)
    assert stats["cents_std"] > 45.0
    assert stats["fraction_over_25_cents"] > 0.4


def test_vibrato_detected_on_short_phrase():
    from analyze import _vibrato_stats
    t = np.arange(int(3.0 * FRAME_RATE)) / FRAME_RATE
    cents = 50.0 * np.sin(2 * np.pi * 5.5 * t)
    f0 = 220.0 * 2.0 ** (cents / 1200.0)
    vib = _vibrato_stats(f0, FRAME_RATE)
    assert vib["rate_hz"] == pytest.approx(5.5, abs=0.3)
    assert vib["extent_cents"] == pytest.approx(50.0, abs=8.0)


def _voice_like(seconds, seed):
    return 0.2 * _band_noise(300, 3000, seconds, seed) / 0.5


def test_plosive_burst_counts_once_and_reports_rate():
    from analyze import _vocal_metrics
    sig = _voice_like(60, 1)
    t = np.arange(int(0.1 * SR)) / SR
    burst = 0.5 * np.sin(2 * np.pi * 50 * t) * np.hanning(len(t))
    for start in (5.0, 20.0, 40.0):
        i = int(start * SR)
        sig[i:i + len(burst)] += burst
    plo = _vocal_metrics(sig, SR, run_pitch=False)["plosive"]
    assert plo["events_count"] == 3
    assert plo["events_per_minute"] == pytest.approx(3.0, abs=0.1)


def test_plosive_clean_voice_has_no_events():
    from analyze import _vocal_metrics
    plo = _vocal_metrics(_voice_like(30, 2), SR, run_pitch=False)["plosive"]
    assert plo["events_count"] == 0


# ---------------------------------------------------------------------------
# analyze.py - peaks, stereo, JSON validity, key
# ---------------------------------------------------------------------------

def _analyze(tmp_path, data, name="GTR test.wav", subtype="PCM_24"):
    from analyze import analyze
    wav = tmp_path / name
    sf.write(wav, data, SR, subtype=subtype)
    analyze(wav, output_dir=tmp_path / "out", use_cache=False)
    return _strict_json(tmp_path / "out" / "analysis.json")


def test_hard_panned_sine_peak_crest_and_stereo(tmp_path):
    t = np.arange(SR * 3) / SR
    left = 0.9 * np.sin(2 * np.pi * 440 * t)
    result = _analyze(tmp_path, np.column_stack((left, np.zeros_like(left))))
    loud = result["loudness"]
    assert loud["sample_peak_dbfs"] == pytest.approx(-0.9, abs=0.1)
    assert loud["crest_factor_db"] == pytest.approx(3.0, abs=0.2)
    assert result["stereo"]["balance_db"] is None
    assert result["stereo"]["lr_correlation"] is None


def test_silent_stem_writes_strict_json(tmp_path):
    result = _analyze(tmp_path, np.zeros((SR * 3, 2)))
    assert result["loudness"]["integrated_lufs"] is None
    assert result["loudness"]["loudness_range_lu"] is None
    assert result["recommended_gain_db"] is None


def test_non_finite_samples_raise_clear_error(tmp_path):
    data = np.zeros((SR, 2), dtype=np.float32)
    data[100, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _analyze(tmp_path, data, subtype="FLOAT")


def test_key_confidence_low_for_white_noise_higher_for_scale():
    from analyze import _estimated_key
    rng = np.random.default_rng(0)
    noise_conf = _estimated_key(0.3 * rng.standard_normal(SR * 6), SR)["confidence"]
    assert noise_conf < 0.5  # cosine similarity gave ~0.96
    t = np.arange(SR * 6) / SR
    weights = {0: 3.0, 2: 1.0, 4: 2.0, 5: 1.0, 7: 2.5, 9: 1.0, 11: 1.0}  # C major scale
    scale = sum(a * np.sin(2 * np.pi * 261.63 * 2 ** (k / 12) * t) for k, a in weights.items())
    key = _estimated_key(0.05 * scale, SR)
    assert (key["key"], key["mode"]) == ("C", "major")
    assert key["confidence"] > noise_conf + 0.2


# ---------------------------------------------------------------------------
# compare_reference
# ---------------------------------------------------------------------------

def _band_deltas(ref, tgt):
    from compare_reference import _matched_delta_bands, _third_octave_psd_db
    return _matched_delta_bands(_third_octave_psd_db(ref, SR), _third_octave_psd_db(tgt, SR))[0]


def test_apply_reduces_bump_without_worsening_other_bands(tmp_path):
    from apply_eq import filter_signal
    from compare_reference import compare_reference
    ref = _pink_noise(10, 5)
    tgt = filter_signal(ref, SR, {"type": "peak", "hz": 2500, "q": 1.0, "db": 8.0})
    ref_path, tgt_path = tmp_path / "ref.wav", tmp_path / "tgt.wav"
    sf.write(ref_path, np.column_stack((ref, ref)) * 0.5, SR, subtype="FLOAT")
    sf.write(tgt_path, np.column_stack((tgt, tgt)) * 0.5, SR, subtype="FLOAT")
    out_path = tmp_path / "applied.wav"
    report = compare_reference(ref_path, tgt_path, tmp_path / "cmp", threshold_db=2.0,
                               apply_output=out_path)
    assert all(abs(f["db"]) <= 6.0 for f in report["auto_eq_filters"])
    out, _ = sf.read(out_path, always_2d=True)
    before = {b["hz"]: b["delta_db"] for b in _band_deltas(ref, tgt)}
    after = {b["hz"]: b["delta_db"] for b in _band_deltas(ref, out[:, 0] / 0.5)}
    for hz in before:
        assert abs(after[hz]) <= abs(before[hz]) + 0.5, (hz, before[hz], after[hz])
    bump = min(before, key=lambda h: abs(h - 2500))
    assert abs(after[bump]) < abs(before[bump]) - 3.0


def test_apply_output_is_not_peak_normalized(tmp_path):
    from compare_reference import _apply_eq_to_target
    t = np.arange(SR) / SR
    x = 0.9 * np.sin(2 * np.pi * 1000 * t)
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src, np.column_stack((x, x)), SR, subtype="FLOAT")
    result = _apply_eq_to_target(src, dst, [{"type": "peak", "hz": 1000.0, "q": 1.0, "db": 6.0}])
    out, _ = sf.read(dst, always_2d=True)
    assert np.max(np.abs(out)) > 1.5
    assert result["output_peak_dbfs"] > 3.0


def test_compare_reports_true_peak_of_worst_channel(tmp_path):
    from compare_reference import compare_reference
    t = np.arange(SR * 4) / SR
    left = 0.9 * np.sin(2 * np.pi * 440 * t)
    stereo = np.column_stack((left, np.zeros_like(left)))
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(a, stereo, SR, subtype="FLOAT")
    sf.write(b, stereo, SR, subtype="FLOAT")
    report = compare_reference(a, b, tmp_path / "cmp")
    assert report["loudness"]["reference_peak_dbfs"] == pytest.approx(-0.9, abs=0.1)


def test_compare_silent_target_writes_strict_json(tmp_path):
    from compare_reference import compare_reference
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(a, 0.1 * _pink_noise(4, 1), SR)
    sf.write(b, np.zeros(SR * 4), SR)
    compare_reference(a, b, tmp_path / "cmp")
    report = _strict_json(tmp_path / "cmp" / "comparison.json")
    assert report["loudness"]["target_lufs"] is None
    assert report["loudness"]["delta_lufs"] is None


# ---------------------------------------------------------------------------
# style_check
# ---------------------------------------------------------------------------

def _premaster_noise(peak_dbfs=-3.0, seconds=10, seed=1):
    x = _pink_noise(seconds, seed)
    x = x / np.max(np.abs(x)) * 10 ** (peak_dbfs / 20)
    return np.column_stack((x, x))


def test_style_check_marks_loudness_not_applicable_on_premaster(tmp_path):
    from style_check import check_style, load_profile
    path = tmp_path / "mix.wav"
    sf.write(path, _premaster_noise(), SR, subtype="FLOAT")
    result = check_style(path, load_profile("modern_rock"), tmp_path / "out")
    checks = {c["name"]: c for c in result["checks"]}
    assert result["input_kind"] == "premaster"
    for name in ("integrated_lufs", "lra_lu", "crest_factor_db"):
        assert checks[name]["verdict"] == "N/A"
    text = (tmp_path / "out" / "style_check.txt").read_text(encoding="utf-8")
    for phrase in ("increase loudness", "more compression", "to reach style target"):
        assert phrase not in text
    _strict_json(tmp_path / "out" / "style_check.json")


def test_style_check_master_flag_grades_loudness(tmp_path):
    from style_check import check_style, load_profile
    path = tmp_path / "mix.wav"
    sf.write(path, _premaster_noise(), SR, subtype="FLOAT")
    result = check_style(path, load_profile("modern_rock"), tmp_path / "out", input_kind="master")
    checks = {c["name"]: c for c in result["checks"]}
    assert checks["integrated_lufs"]["verdict"] in ("GREEN", "YELLOW", "RED")


def test_style_check_silent_mix_grades_na_and_writes_strict_json(tmp_path):
    from style_check import check_style, load_profile
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros((SR * 4, 2)), SR)
    result = check_style(path, load_profile("modern_rock"), tmp_path / "out", input_kind="master")
    checks = {c["name"]: c for c in result["checks"]}
    assert checks["integrated_lufs"]["verdict"] == "N/A"
    assert checks["lra_lu"]["verdict"] == "N/A"
    _strict_json(tmp_path / "out" / "style_check.json")


def test_style_help_lists_every_profile(capsys, monkeypatch):
    import style_check
    monkeypatch.setattr(sys, "argv", ["style_check.py", "--help"])
    with pytest.raises(SystemExit):
        style_check.main()
    out = capsys.readouterr().out
    for name in style_check.list_profiles():
        assert name in out


def test_analyze_config_is_resolved_from_project_root_not_cwd(tmp_path, monkeypatch):
    import tomllib
    import analyze
    project_config = Path(__file__).resolve().parents[1] / "config.toml"
    assert analyze._CONFIG_PATH == project_config
    monkeypatch.chdir(tmp_path)
    expected = analyze._FALLBACK_TARGET_LUFS
    if project_config.exists():
        cfg = tomllib.loads(project_config.read_text(encoding="utf-8"))
        expected = float(cfg.get("analyze", {}).get("default_target_lufs", expected))
    assert analyze._config_target_lufs() == expected
