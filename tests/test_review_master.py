"""Regressions for the render/master review findings (limiter, pan law,
linked dynamics, clipper knee, vinyl pre-master, delivery ceilings, health
verdicts, stem locations and config validation)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

SR = 48000


def _tp8(data) -> float:
    from _dsp import worst_channel_true_peak_dbfs
    return worst_channel_true_peak_dbfs(np.asarray(data), 8)


def _noise_mix(path: Path, seconds: float = 5.0, scale: float = 0.1, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal((int(SR * seconds), 2)) * scale
    sf.write(str(path), sig, SR, subtype="FLOAT")
    return path


def _render_session(tmp_path: Path, tracks: list[dict], buses: dict, master: dict | None = None,
                    stems: bool = False, stage: str | None = None, extra: dict | None = None):
    """Write audio + config for a tiny session and render it."""
    from render_mix import render_mix
    tracks_cfg = []
    for t in tracks:
        tdir = tmp_path / "tracks" / t["name"]
        tdir.mkdir(parents=True, exist_ok=True)
        path = tdir / "assembled.wav"
        sf.write(str(path), t["audio"], SR, subtype="FLOAT")
        tracks_cfg.append({"name": t["name"], "file": str(path), "active": True,
                           "bus": t["bus"], "volume_db": 0.0, "pan": t.get("pan", 0.0)})
    config = {"session_dir": str(tmp_path), "output_dir": str(tmp_path / "mixes"),
              "sample_rate": SR, "master": master or {"premaster_mode": True},
              "buses": buses, "tracks": tracks_cfg, **(extra or {})}
    cfg_path = tmp_path / "mix_config.json"
    cfg_path.write_text(json.dumps(config))
    render_mix(cfg_path, render_stems=stems, stage=stage)
    return cfg_path


# ---------------------------------------------------------------------------
# 1. Limiter: a real ceiling limiter with an 8x guarantee
# ---------------------------------------------------------------------------

def test_master_limiter_is_transparent_below_ceiling(tmp_path):
    """pedalboard.Limiter compressed 4:1 above -10 dBFS and added makeup even
    when the signal was far below the ceiling. A brickwall limiter must leave
    it untouched apart from the loudness gain."""
    from master_mix import master_mix
    t = np.arange(SR * 5) / SR
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    source = tmp_path / "mix.wav"
    sf.write(str(source), np.stack([tone, tone], axis=1), SR, subtype="FLOAT")
    r = master_mix(source, tmp_path / "out", "spotify", "transparent")
    out, _ = sf.read(r["output"], always_2d=True)
    gain = np.std(out[:, 0]) / np.std(tone)
    assert np.max(np.abs(out[:, 0] - tone * gain)) < 1e-3
    assert "BrickwallLimiter" in r["limiter"]


def test_master_true_peak_guaranteed_at_8x_and_health_agrees(tmp_path):
    from master_health import master_health
    from master_mix import master_mix
    rng = np.random.default_rng(3)
    sig = rng.standard_normal((SR * 5, 2)) * 0.05
    sig[::4800] = 0.9  # sparse transients force limiting
    source = tmp_path / "mix.wav"
    sf.write(str(source), sig, SR, subtype="FLOAT")
    r = master_mix(source, tmp_path / "out", "spotify", "modern_rock")
    out, _ = sf.read(r["output"], always_2d=True)
    assert _tp8(out) <= r["tp_ceiling_dbtp"]
    assert r["true_peak_target_met"] is True
    health = master_health(Path(r["output"]), tmp_path / "health", "spotify", use_cache=False)
    assert health["conformance"]["true_peak_8x_verdict"] == "[OK]"
    assert not any("lufs_post_correction" in step for step in r["chain"])


def test_legacy_render_chain_uses_brickwall_limiter(tmp_path):
    rng = np.random.default_rng(4)
    audio = rng.standard_normal((SR * 4, 2)) * 0.2
    _render_session(tmp_path, [{"name": "BASS", "bus": "bass", "audio": audio}],
                    {"bass": {"volume_db": 0.0, "parent_bus": None}},
                    master={"premaster_mode": False, "lufs_target": -10.0, "true_peak_dbfs": -1.0})
    report = json.loads((tmp_path / "mixes" / "mix_report.json").read_text())
    assert "BrickwallLimiter" in report["limiter"]
    out, _ = sf.read(tmp_path / "mixes" / "mix.wav", always_2d=True)
    assert _tp8(out) <= -1.0 + 0.001


# ---------------------------------------------------------------------------
# 2. Pan law: unity at centre for stereo buffers, constant power for mono
# ---------------------------------------------------------------------------

def test_stereo_pan_is_continuous_at_centre():
    from render_mix import _pan
    buf = np.ones((2, 10))
    assert np.allclose(_pan(buf, 0.0), buf)
    near = _pan(buf, 0.01)
    assert np.allclose(near[1], 1.0) and np.allclose(near[0], 0.99)
    assert np.allclose(_pan(buf, -1.0), [[1.0] * 10, [0.0] * 10])


def test_mono_source_pan_keeps_constant_power():
    from render_mix import _pan
    buf = np.ones((2, 10))
    assert np.allclose(_pan(buf, 0.0, mono_source=True), np.sqrt(0.5))
    left = _pan(buf, -1.0, mono_source=True)
    assert np.allclose(left[0], 1.0) and np.allclose(left[1], 0.0, atol=1e-12)


def test_stereo_track_and_bus_pan_preserve_level(tmp_path):
    t = np.arange(SR * 3) / SR
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)
    _render_session(
        tmp_path,
        [{"name": "BASS ST", "bus": "bass", "audio": np.stack([tone, tone], axis=1)},
         {"name": "BASS MONO", "bus": "bass2", "audio": tone}],
        {"bass": {"volume_db": 0.0, "pan": 0.01, "parent_bus": None},
         "bass2": {"volume_db": 0.0, "pan": 0.0, "parent_bus": None}},
        stems=True)
    stereo_stem, _ = sf.read(tmp_path / "stems" / "stem_bass.wav", always_2d=True)
    mono_stem, _ = sf.read(tmp_path / "stems" / "stem_bass2.wav", always_2d=True)
    assert np.max(np.abs(stereo_stem[:, 1])) == pytest.approx(0.5, abs=1e-3)
    assert np.max(np.abs(stereo_stem[:, 0])) == pytest.approx(0.495, abs=1e-3)
    # Mono source at centre is unchanged: constant-power -3 dB per channel.
    assert np.max(np.abs(mono_stem)) == pytest.approx(0.5 * np.sqrt(0.5), abs=1e-3)


def test_auto_trim_has_no_pan_discontinuity(tmp_path):
    from render_mix import _compute_bus_auto_trims
    path = _noise_mix(tmp_path / "stem.wav", seconds=4)
    def trims(pan):
        cfg = {"sample_rate": SR,
               "buses": {"bass": {"volume_db": 0.0, "pan": pan, "parent_bus": None}},
               "tracks": [{"name": "B", "file": str(path), "bus": "bass", "pan": 0.0}]}
        return _compute_bus_auto_trims(cfg, verbose=False)["bass"]
    assert abs(trims(0.0) - trims(0.01)) <= 0.2


# ---------------------------------------------------------------------------
# 3. Stereo-linked compression
# ---------------------------------------------------------------------------

def _scaled_pair(seconds=3.0):
    rng = np.random.default_rng(9)
    left = rng.standard_normal(int(SR * seconds)) * 0.5
    return np.vstack([left, 0.1 * left]).astype(np.float32)


def test_bus_comp_preset_is_linked():
    from render_mix import _apply_comp_preset
    buf = _scaled_pair()
    out = _apply_comp_preset(buf, "comp_drum_bus", SR)
    assert np.allclose(out[1], 0.1 * out[0], atol=1e-6)
    assert np.std(out[0]) < np.std(buf[0]) * 0.95  # it did compress


def test_master_comp_and_multiband_are_linked():
    from master_mix import MASTERING_PRESETS, _master_comp, _master_multiband
    buf = _scaled_pair()
    out = _master_comp(buf, SR, {"threshold_db": -20, "ratio": 4, "attack_ms": 5,
                                 "release_ms": 100, "makeup_db": 0})
    assert np.allclose(out[1], 0.1 * out[0], atol=1e-6)
    mb = _master_multiband(buf.astype(np.float64), SR, MASTERING_PRESETS["modern_rock_mb"]["multiband"])
    assert np.allclose(mb[1], 0.1 * mb[0], atol=1e-6)


def test_render_master_glue_comp_is_linked(tmp_path):
    buf = _scaled_pair(4.0).T
    _render_session(tmp_path, [{"name": "BASS", "bus": "bass", "audio": buf}],
                    {"bass": {"volume_db": 0.0, "parent_bus": None}},
                    master={"premaster_mode": True,
                            "comp": {"threshold_db": -20, "ratio": 4, "attack_ms": 5, "release_ms": 100}})
    out, _ = sf.read(tmp_path / "mixes" / "mix.wav", always_2d=True)
    assert np.allclose(out[:, 1], 0.1 * out[:, 0], atol=2e-5)


# ---------------------------------------------------------------------------
# 4. Soft clipper is monotone and never exceeds its threshold
# ---------------------------------------------------------------------------

def test_soft_clip_monotone_and_bounded():
    from render_mix import _soft_clip
    x = np.linspace(0, 3, 30001)
    y = _soft_clip(x, 0.0, 1.5)
    assert np.all(np.diff(y) >= -1e-12)
    assert np.max(y) <= 1.0 + 1e-12
    knee_start = 10 ** (-1.5 / 20)
    assert np.allclose(y[x < knee_start], x[x < knee_start])
    assert np.allclose(_soft_clip(-x, 0.0, 1.5), -y)


# ---------------------------------------------------------------------------
# 5. Vinyl pre-master
# ---------------------------------------------------------------------------

def test_vinyl_elliptical_does_not_smear_image():
    from master_mix import _vinyl_elliptical_eq
    t = np.arange(SR * 2) / SR
    left = 0.5 * np.sin(2 * np.pi * 300 * t)
    out = _vinyl_elliptical_eq(np.vstack([left, np.zeros_like(left)]), SR, 150.0)
    core = slice(SR // 2, -SR // 2)
    leak_db = 20 * np.log10(np.std(out[1, core]) / np.std(out[0, core]))
    assert leak_db < -25


def test_vinyl_pre_peak_normalizes_without_loudness_target(tmp_path):
    from master_mix import FORMAT_PRESETS, master_mix
    assert FORMAT_PRESETS["vinyl_pre"]["target_lufs"] is None
    source = _noise_mix(tmp_path / "mix.wav", scale=0.02)
    r = master_mix(source, tmp_path / "out", "vinyl_pre", "gentle")
    out, _ = sf.read(r["output"], always_2d=True)
    assert _tp8(out) == pytest.approx(r["tp_ceiling_dbtp"], abs=0.05)
    assert r["tp_ceiling_dbtp"] == -3.0
    assert r["target_lufs"] is None and r["loudness_target_met"] is None
    assert not any(step.startswith("lufs_norm") for step in r["chain"])
    assert not any("vinyl_elliptical" in step for step in r["chain"])
    r2 = master_mix(source, tmp_path / "out2", "vinyl_pre", "gentle", vinyl_elliptical_hz=150.0)
    assert any("vinyl_elliptical" in step for step in r2["chain"])


def test_master_health_handles_format_without_loudness_target(tmp_path):
    from master_health import master_health
    from master_mix import master_mix
    source = _noise_mix(tmp_path / "mix.wav", scale=0.02)
    r = master_mix(source, tmp_path / "out", "vinyl_pre", "transparent")
    health = master_health(Path(r["output"]), tmp_path / "health", "vinyl_pre", use_cache=False)
    assert health["conformance"]["verdict"] == "[OK]"
    assert health["conformance"]["lufs_delta"] is None


# ---------------------------------------------------------------------------
# 6. Delivery ceilings
# ---------------------------------------------------------------------------

def test_loud_targets_use_minus_two_dbtp_unless_overridden(tmp_path):
    from master_mix import FORMAT_PRESETS, master_mix
    source = _noise_mix(tmp_path / "mix.wav")
    assert master_mix(source, tmp_path / "a", "spotify", "transparent")["tp_ceiling_dbtp"] == -1.0
    loud = master_mix(source, tmp_path / "b", "spotify", "transparent", target_lufs=-9.0)
    assert loud["tp_ceiling_dbtp"] == -2.0
    explicit = master_mix(source, tmp_path / "c", "spotify", "transparent", target_lufs=-9.0, tp_ceiling=-1.0)
    assert explicit["tp_ceiling_dbtp"] == -1.0
    assert "artistic" in FORMAT_PRESETS["cd"]["description"]
    assert "house" in FORMAT_PRESETS["broadcast"]["description"]


# ---------------------------------------------------------------------------
# 7. mix_health premaster peak gate; LRA advisory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("peak_db, expected", [(1.9, "[X] "), (-0.5, "[!] "), (-1.5, "[OK]")])
def test_premaster_true_peak_gate(peak_db, expected):
    from mix_health import _STAGE_TARGETS, _loudness_section
    sg = _STAGE_TARGETS["premaster"]
    t = np.arange(SR * 4) / SR
    tone = 10 ** (peak_db / 20) * np.sin(2 * np.pi * 997 * t)
    data = np.stack([tone, tone], axis=1)
    section = _loudness_section(tone, data, SR, sg["lufs_target"], sg["tp_ceiling_dbtp"], stage_targets=sg)
    assert section["true_peak_verdict"] == expected
    assert section["lra_verdict"] != "[X] "  # a steady tone has ~0 LU LRA


# ---------------------------------------------------------------------------
# 8. Stems: measured levels, stage stems kept apart, bus_balance CLI
# ---------------------------------------------------------------------------

def test_stage_stems_do_not_overwrite_main_stems(tmp_path):
    rng = np.random.default_rng(5)
    audio = rng.standard_normal((SR * 3, 2)) * 0.1
    cfg = _render_session(tmp_path, [{"name": "BASS", "bus": "bass", "audio": audio}],
                          {"bass": {"volume_db": 0.0, "parent_bus": None}}, stems=True)
    main_stem = tmp_path / "stems" / "stem_bass.wav"
    before = main_stem.read_bytes()
    from render_mix import render_mix
    render_mix(cfg, render_stems=True, stage="raw")
    assert (tmp_path / "stems" / "raw" / "stem_bass.wav").exists()
    assert main_stem.read_bytes() == before


def test_bus_balance_reports_measured_stem_loudness(tmp_path):
    import bus_balance
    rng = np.random.default_rng(6)
    audio = rng.standard_normal((SR * 4, 2)) * 0.1
    cfg = _render_session(tmp_path, [{"name": "BASS", "bus": "bass", "audio": audio}],
                          {"bass": {"volume_db": -6.0, "parent_bus": None}}, stems=True)
    (tmp_path / "mixes" / "mix_report.json").unlink()  # force the former fallback path
    stem, _ = sf.read(tmp_path / "stems" / "stem_bass.wav", always_2d=True)
    rows = bus_balance.measure_buses(cfg)
    assert rows[0]["lufs"] == pytest.approx(pyln.Meter(SR).integrated_loudness(stem), abs=0.01)


def test_bus_balance_help_exits_cleanly():
    proc = subprocess.run([sys.executable, str(TOOLS_DIR / "bus_balance.py"), "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0 and "mix_config" in proc.stdout


# ---------------------------------------------------------------------------
# 9. _find_final_file never aborts config generation
# ---------------------------------------------------------------------------

def _op(path: Path, inp: Path | None, out: Path):
    sf.write(str(out), np.zeros(10), SR)
    out.with_suffix(".operation.json").write_text(json.dumps(
        {"input": str(inp.resolve()) if inp else None, "output": str(out.resolve())}))


def test_find_final_file_branch_warns_instead_of_raising(tmp_path, capsys):
    from render_mix import _find_final_file
    a = tmp_path / "assembled.wav"
    _op(tmp_path, None, a)
    _op(tmp_path, a, tmp_path / "assembled_eq.wav")
    _op(tmp_path, a, tmp_path / "assembled_alt.wav")
    result = _find_final_file(tmp_path)
    assert result is not None
    assert "WARNING" in capsys.readouterr().out


def test_find_final_file_partial_records_use_filename_ladder(tmp_path, capsys):
    from render_mix import _find_final_file
    _op(tmp_path, None, tmp_path / "assembled.wav")
    sf.write(str(tmp_path / "assembled_eq_comp.wav"), np.zeros(10), SR)
    assert Path(_find_final_file(tmp_path)).name == "assembled_eq_comp.wav"
    assert "WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 10. --all-formats on short input
# ---------------------------------------------------------------------------

def test_all_formats_summary_handles_short_input(tmp_path):
    source = _noise_mix(tmp_path / "short.wav", seconds=1.0)
    proc = subprocess.run([sys.executable, str(TOOLS_DIR / "master_mix.py"), str(source),
                           "--output-dir", str(tmp_path / "out"), "--all-formats",
                           "--master-preset", "transparent"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "BATCH MASTERING SUMMARY" in proc.stdout


# ---------------------------------------------------------------------------
# 11. Config validation and reverb tails
# ---------------------------------------------------------------------------

def _base_config(tmp_path):
    path = _noise_mix(tmp_path / "a.wav", seconds=1.0)
    return {"sample_rate": SR, "master": {}, "buses": {"bass": {"parent_bus": None}},
            "tracks": [{"name": "A", "file": str(path), "bus": "bass"}]}


def test_validate_rejects_unknown_presets(tmp_path):
    from render_mix import validate_mix_config
    cfg = _base_config(tmp_path)
    cfg["buses"]["bass"]["comp_preset"] = "no_such_comp"
    with pytest.raises(ValueError, match="no_such_comp"):
        validate_mix_config(cfg)
    cfg = _base_config(tmp_path)
    cfg["buses"]["bass"]["reverb_send"] = {"preset": "no_such_reverb"}
    with pytest.raises(ValueError, match="no_such_reverb"):
        validate_mix_config(cfg)
    cfg = _base_config(tmp_path)
    cfg["reverb_buses"] = {"plate": {"preset": "no_such_plate"}}
    with pytest.raises(ValueError, match="no_such_plate"):
        validate_mix_config(cfg)


def test_reverb_tail_is_not_truncated(tmp_path):
    rng = np.random.default_rng(8)
    audio = rng.standard_normal((SR * 2, 2)) * 0.2  # loud right up to the end
    _render_session(tmp_path, [{"name": "SN", "bus": "drums", "audio": audio}],
                    {"drums": {"volume_db": 0.0, "parent_bus": None,
                               "reverb_send": {"preset": "hall_ambient", "wet": 0.5}}})
    out, _ = sf.read(tmp_path / "mixes" / "mix.wav", always_2d=True)
    assert len(out) > SR * 3
    assert np.max(np.abs(out[SR * 2 + SR // 2:])) > 1e-3


# ---------------------------------------------------------------------------
# 12-14. M/S relevance, premaster verdict, vocal room takes
# ---------------------------------------------------------------------------

def test_ms_relevance_counts_side_gain_as_boost():
    from render_mix import _ms_relevance_check
    rng = np.random.default_rng(10)
    wide = rng.standard_normal((2, SR))
    rel = _ms_relevance_check(wide, {"side_gain_db": 3.0, "side_eq": []}, SR)
    assert rel["recommend_skip"] is True


def test_premaster_verdict_accepts_default_handoff_peak(tmp_path):
    # Premaster bands intentionally differ from _peak_verdict: the default
    # -3 dBFS handoff must read [OK], not a permanent [WARN].
    rng = np.random.default_rng(11)
    _render_session(tmp_path, [{"name": "BASS", "bus": "bass", "audio": rng.standard_normal((SR * 3, 2)) * 0.1}],
                    {"bass": {"volume_db": 0.0, "parent_bus": None}})
    mp = json.loads((tmp_path / "mixes" / "mix_report.json").read_text())["master_peaks"]
    worst = max(mp["final_sample_peak"], mp["final_true_peak"])
    assert worst <= -2.0
    assert mp["verdict"] == "[OK]"


def test_vocal_room_takes_are_not_drums():
    from render_mix import _detect_bus
    assert _detect_bus("LEAD VOX room") == "vocal_lead"
    assert _detect_bus("BG VOX ROOM L") == "vocal_bg"
    assert _detect_bus("ROOM CLOSE.01 L.05") == "drums"
    assert _detect_bus("KICK IN.05") == "drums"


# ---------------------------------------------------------------------------
# 15. Compression history can flag a limited master
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("peak_db, flagged", [(-1.0, True), (-2.2, True), (-3.0, False)])
def test_compression_history_flags_limited_masters_not_premasters(peak_db, flagged):
    from master_health import _compression_history
    rng = np.random.default_rng(12)
    sig = np.tanh(rng.standard_normal((2, SR * 4)) * 5)
    sig *= 10 ** (peak_db / 20) / np.max(np.abs(sig))
    assert _compression_history(sig, SR, pyln.Meter(SR))["likely_already_mastered"] is flagged
