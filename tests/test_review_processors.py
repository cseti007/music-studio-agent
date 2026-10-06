"""Regression tests for verified processor review findings (DSP behavior,
preset/CLI precedence, float headroom, sidechain validation, doc truth)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

SR = 44100


def _write(path, data, sr=SR):
    sf.write(str(path), data, sr, subtype="FLOAT")
    return path


def _sine(freq, amp, seconds=2.0, sr=SR):
    return amp * np.sin(2 * np.pi * freq * np.arange(int(seconds * sr)) / sr)


def _amp(x, freq, sr=SR):
    """Hann-windowed single-bin amplitude estimate."""
    w = np.hanning(len(x))
    t = np.arange(len(x)) / sr
    return 2 * abs(np.sum(x * w * np.exp(-2j * np.pi * freq * t))) / w.sum()


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def _cli(script, *args):
    return subprocess.run([sys.executable, str(TOOLS / script), *map(str, args)],
                          capture_output=True, text=True, check=True)


# 1. Transient shaping ---------------------------------------------------------

def _kick(amp):
    t = np.arange(int(0.4 * SR)) / SR
    x = np.zeros(SR)
    x[SR // 4:SR // 4 + len(t)] = amp * np.sin(2 * np.pi * 90 * t) * np.exp(-t / 0.06)
    return x


def test_transient_gain_is_level_independent_and_bounded():
    from apply_transient import apply_transient
    on = SR // 4
    attack = slice(on, on + int(0.004 * SR))
    sustain = slice(on + int(0.12 * SR), on + int(0.3 * SR))
    measured = []
    for amp in (0.8, 0.08):
        x = _kick(amp)
        y = apply_transient(x, SR, 6.0, -8.0)
        measured.append((20 * np.log10(_rms(y[attack]) / _rms(x[attack])),
                         20 * np.log10(_rms(y[sustain]) / _rms(x[sustain]))))
        mask = np.abs(x) > 1e-4
        gain_db = 20 * np.log10(np.abs(y[mask] / x[mask]))
        assert gain_db.max() <= 6.0 + 1e-6 and gain_db.min() >= -8.0 - 1e-6
    assert measured[0] == pytest.approx(measured[1], abs=0.1)
    assert measured[0][0] > 4.0 and measured[0][1] < -6.0


def test_transient_cli_records_preset_and_explicit_zero_wins(tmp_path):
    src = _write(tmp_path / "kick.wav", _kick(0.5))
    _cli("apply_transient.py", src, "--preset", "transient_kick_punch",
         "--attack", "0", "--output-dir", tmp_path / "out")
    report = json.loads((tmp_path / "out" / "transient_report.json").read_text())
    assert report["preset"] == "transient_kick_punch"
    assert report["attack_db"] == 0.0
    assert report["sustain_db"] == -2.0


# 2. Sub-harmonic synthesis ---------------------------------------------------

def test_subharm_harmonic_path_is_mostly_second_and_third_harmonic():
    from apply_subharm import _harmonics
    x = _sine(50, 0.5, 3.0)
    h = _harmonics(x, SR, 40.0, 80.0, 80.0, 200.0, 1.8)[SR:2 * SR]
    leak = _amp(h, 50)
    assert _amp(h, 100) > 10 * leak
    assert _amp(h, 150) > 10 * leak


# 3. Gate retrigger -----------------------------------------------------------

def test_gate_retrigger_during_release_does_not_dip():
    from apply_gate import _gate_gain_envelope
    x = np.zeros(SR // 2)
    x[:int(0.05 * SR)] = 0.5
    retrigger = int(0.10 * SR)
    x[retrigger:retrigger + int(0.05 * SR)] = 0.5
    gain = _gate_gain_envelope(x, SR, threshold_db=-30, range_db=-40, attack_ms=10,
                               hold_ms=20, release_ms=100, hysteresis_db=6, rms_window_ms=5)
    window = gain[retrigger - int(0.005 * SR):retrigger + int(0.02 * SR)]
    assert gain[retrigger - int(0.005 * SR)] < 0.95  # retriggered from the release ramp
    assert np.diff(window).min() > -1e-3


# 4. Stereo-linked compression -----------------------------------------------

def _unbalanced_stereo():
    return np.column_stack([_sine(1000, 0.8), _sine(1000, 0.05)])


def _channel_gain_db(out, x):
    seg = slice(SR // 2, SR)
    return [20 * np.log10(_rms(out[seg, c]) / _rms(x[seg, c])) for c in range(2)]


def test_compression_is_stereo_linked(tmp_path):
    from apply_compression import apply_compression
    x = _unbalanced_stereo()
    src = _write(tmp_path / "in.wav", x)
    rep = apply_compression(src, tmp_path / "out", -20, 4, 1, 50, makeup_db=0.0)
    left, right = _channel_gain_db(sf.read(rep["output"])[0], x)
    assert left < -6
    assert right == pytest.approx(left, abs=0.2)


def test_multiband_compression_is_stereo_linked(tmp_path):
    from apply_multiband_comp import apply_multiband_comp
    x = _unbalanced_stereo()
    src = _write(tmp_path / "in.wav", x)
    band = {"threshold_db": -20, "ratio": 4, "attack_ms": 1, "release_ms": 50}
    rep = apply_multiband_comp(src, tmp_path / "out", 200, 3000, band, band, band, force=True)
    left, right = _channel_gain_db(sf.read(rep["output"])[0], x)
    assert left < -6
    assert right == pytest.approx(left, abs=0.2)


# 5. Float headroom is preserved; peaks are reported ---------------------------

def _hot(tmp_path):
    t = np.arange(3 * SR) / SR
    x = 1.5 * (0.5 * np.sin(2 * np.pi * 60 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)
               + 0.2 * np.sin(2 * np.pi * 7000 * t))
    return _write(tmp_path / "hot.wav", np.column_stack([x, 0.9 * x])), float(np.max(np.abs(x)))


def _run_eq(src, out):
    from apply_eq import apply_eq
    return apply_eq(src, out, [{"type": "peak", "hz": 1000, "q": 1, "db": 3}])


def _run_comp(src, out):
    from apply_compression import apply_compression
    return apply_compression(src, out, -20, 2, 10, 100, makeup_db=6)


def _run_gate(src, out):
    from apply_gate import apply_gate
    return apply_gate(src, out, -60, -40, 1, 50, 50)


def _run_sat(src, out):
    from apply_saturation import apply_saturation
    return apply_saturation(src, out, mode="tape", drive=0.3, mix=0.5)


def _run_delay(src, out):
    from apply_delay import apply_delay
    return apply_delay(src, out, 100, mix=0.5)


def _run_reverb(src, out):
    from apply_reverb import apply_reverb
    return apply_reverb(src, out, wet=0.5)


def _run_amp(src, out):
    from apply_amp import apply_amp
    return apply_amp(src, out, hp_hz=None, lp_hz=None, low_shelf_hz=None, mid_hz=None)


def _run_exciter(src, out):
    from apply_exciter import apply_exciter
    return apply_exciter(src, out, 5000, 1.5, 0.3, force=True)


def _run_haas(src, out):
    from apply_haas import apply_haas
    return apply_haas(src, out, 10.0, force=True)


def _run_subharm(src, out):
    from apply_subharm import apply_subharm
    return apply_subharm(src, out, 40, 80, 80, 200, 1.8, 0.2, force=True)


def _run_mb(src, out):
    from apply_multiband_comp import apply_multiband_comp
    band = {"threshold_db": -10, "ratio": 2, "attack_ms": 10, "release_ms": 100, "makeup_db": 6}
    return apply_multiband_comp(src, out, 200, 3000, band, band, band, force=True)


def _run_octaver(src, out):
    from apply_octaver import apply_octaver
    out.mkdir(parents=True, exist_ok=True)
    return apply_octaver(src, out / "oct.wav")


@pytest.mark.parametrize("run", [_run_eq, _run_comp, _run_gate, _run_sat, _run_delay,
                                 _run_reverb, _run_amp, _run_exciter, _run_haas,
                                 _run_subharm, _run_mb, _run_octaver])
def test_processors_keep_float_overs_and_report_them(tmp_path, run):
    src, in_peak = _hot(tmp_path)
    report = run(src, tmp_path / "out")
    out, _ = sf.read(report["output"])
    peak = float(np.max(np.abs(out)))
    assert peak > 1.0 and peak > 0.5 * in_peak  # not rescaled to 0 dBFS
    assert report["output_exceeds_0dbfs"] is True
    assert report["output_peak_dbfs"] == pytest.approx(20 * np.log10(peak), abs=0.06)


def test_level_notes_has_no_gain_step_at_range_boundaries(tmp_path):
    from level_notes import level_notes
    x = np.zeros(4 * SR)
    for onset in np.arange(1.2, 2.8, 0.25):
        i = int(onset * SR)
        n = np.arange(int(0.1 * SR))
        x[i:i + len(n)] = (0.3 if int(onset * 4) % 2 else 1.4) * np.sin(2 * np.pi * 110 * n / SR)
    x += 0.01 * np.sin(2 * np.pi * 55 * np.arange(len(x)) / SR)
    y, report = level_notes(x, SR, 1.0, 3.0, target_peak_db=3.0, quiet_threshold_db=0.0)
    assert report["notes_boosted"] > 0
    outside = np.ones(len(x), bool)
    outside[SR:3 * SR] = False
    np.testing.assert_array_equal(y[outside], x[outside])
    # Notes are only ever lifted: no segment-wide rescale (gain step) inside.
    assert np.all(np.abs(y) >= np.abs(x) - 1e-12)
    assert report["output_exceeds_0dbfs"] is True


# 6. Saturation DC / amp oversampled tube --------------------------------------

@pytest.mark.parametrize("oversample", [1, 4])
def test_tube_saturation_has_no_dc(tmp_path, oversample):
    from apply_saturation import apply_saturation
    src = _write(tmp_path / "in.wav", _sine(100, 0.5))
    rep = apply_saturation(src, tmp_path / "out", mode="tube", drive=0.8, asymmetry=1.0,
                           oversample=oversample)
    y = sf.read(rep["output"])[0][SR // 2:]
    assert abs(np.mean(y)) < 0.002 * _rms(y)


def test_amp_tube_stage_has_no_dc(tmp_path):
    from apply_amp import apply_amp
    src = _write(tmp_path / "in.wav", _sine(100, 0.5))
    rep = apply_amp(src, tmp_path / "out", drive=0.8, asymmetry=1.0, hp_hz=None, lp_hz=None,
                    low_shelf_hz=None, mid_hz=None)
    y = sf.read(rep["output"])[0][SR // 2:]
    assert abs(np.mean(y)) < 0.002 * _rms(y)


# 7. Exciter aliasing ------------------------------------------------------------

def test_exciter_nonlinearity_is_oversampled():
    from apply_exciter import _excite
    y = _excite(_sine(9000, 0.5, 1.0), SR, 6000, 2.0)
    # The 3rd harmonic (27 kHz) folds back to 17.1 kHz without oversampling.
    assert 20 * np.log10(_amp(y, 17100) / _amp(y, 9000)) < -60


# 8. Reverb engine calibration ----------------------------------------------------

def _pink(seconds=4.0):
    rng = np.random.default_rng(0)
    w = rng.standard_normal((int(seconds * SR), 2))
    spec = np.fft.rfft(w, axis=0)
    f = np.fft.rfftfreq(len(w), 1 / SR)
    spec[1:] /= np.sqrt(f[1:, None])
    spec[0] = 0
    pink = np.fft.irfft(spec, n=len(w), axis=0)
    return 0.1 * pink / np.std(pink)


@pytest.mark.parametrize("preset,ir", [("snare_plate", "plate_short"), ("room_drums", "room_tight"),
                                       ("vocal_hall_wide", "hall_concert")])
def test_reverb_engines_have_comparable_wet_level(tmp_path, preset, ir):
    from apply_reverb import PRESETS, apply_reverb
    src = _write(tmp_path / "pink.wav", _pink())
    p = {k: PRESETS[preset][k] for k in ("room_size", "damping", "width", "wet", "hp_hz", "lp_hz")}
    algo = apply_reverb(src, tmp_path / "algo", send_mode=True, **p)
    conv = apply_reverb(src, tmp_path / "conv", send_mode=True, ir_path=TOOLS / "irs" / f"{ir}.wav", **p)
    level = [_rms(sf.read(r["output"])[0][SR:]) for r in (algo, conv)]
    assert abs(20 * np.log10(level[1] / level[0])) < 3.0


# 9. Delay preset truth -------------------------------------------------------------

def test_pre_delay_preset_outputs_only_the_delayed_signal(tmp_path):
    x = _sine(440, 0.5, 1.0)
    src = _write(tmp_path / "in.wav", x)
    _cli("apply_delay.py", src, "--output-dir", tmp_path, "--preset", "delay_pre_delay")
    y = sf.read(next(tmp_path.glob("in_delay*.wav")))[0]
    d = int(round(0.025 * SR))
    np.testing.assert_allclose(y[d:], x[:-d], atol=1e-6)
    assert np.all(y[:d] == 0)


# 10. Explicit CLI flags beat presets ----------------------------------------------

def test_saturation_cli_flag_beats_preset(tmp_path):
    src = _write(tmp_path / "in.wav", _sine(100, 0.3))
    _cli("apply_saturation.py", src, "--output-dir", tmp_path, "--preset", "sat_tape_subtle",
         "--drive", "0.5")
    settings = json.loads((tmp_path / "sat_report.json").read_text())["settings"]
    assert settings["drive"] == 0.5 and settings["mix"] == 0.5


def test_delay_cli_flag_beats_preset(tmp_path):
    src = _write(tmp_path / "in.wav", _sine(100, 0.3))
    _cli("apply_delay.py", src, "--output-dir", tmp_path, "--preset", "delay_slapback_snare",
         "--mix", "0.5", "--feedback", "0.2")
    settings = json.loads((tmp_path / "delay_report.json").read_text())["settings"]
    assert settings["mix"] == 0.5 and settings["feedback"] == 0.2
    assert settings["delay_ms"] == 80


# 11. Sidechain timelines must match --------------------------------------------------

def test_compression_rejects_mismatched_sidechain(tmp_path):
    from apply_compression import apply_compression
    src = _write(tmp_path / "in.wav", _sine(100, 0.3))
    sc = _write(tmp_path / "sc.wav", _sine(100, 0.3, 1.0))
    with pytest.raises(ValueError, match="frame count"):
        apply_compression(src, tmp_path / "out", -20, 4, 5, 50, sidechain_path=sc)


def test_reverb_rejects_mismatched_sidechain(tmp_path):
    from apply_reverb import apply_reverb
    src = _write(tmp_path / "in.wav", _sine(100, 0.3))
    sc = _write(tmp_path / "sc.wav", _sine(100, 0.3, 1.0))
    with pytest.raises(ValueError, match="frame count"):
        apply_reverb(src, tmp_path / "out", sidechain_path=sc)


# 12. Dynamic EQ self-detector centred on the band ------------------------------------

def test_dynamic_eq_self_detector_ignores_out_of_band_level(tmp_path):
    from apply_dynamic_eq import apply_dynamic_eq
    x = _sine(200, 0.5) + _sine(3000, 0.005)
    src = _write(tmp_path / "in.wav", x)
    report = apply_dynamic_eq(src, tmp_path / "out", 3000, threshold_db=-30)
    assert report["max_center_cut_db"] < 0.1
    loud = _write(tmp_path / "loud.wav", _sine(200, 0.01) + _sine(3000, 0.5))
    assert apply_dynamic_eq(loud, tmp_path / "out2", 3000, threshold_db=-30)["max_center_cut_db"] > 1.0


# 13. IR pack -----------------------------------------------------------------------

def test_ir_pack_matches_generator():
    import generate_irs
    for name, fn in generate_irs.IR_GENERATORS.items():
        ir = fn()
        ir = ir / np.max(np.abs(ir)) * 10 ** (-3 / 20)
        committed, sr = sf.read(TOOLS / "irs" / f"{name}.wav")
        assert sr == generate_irs.SR
        np.testing.assert_allclose(committed, ir, atol=2 ** -22, err_msg=name)


@pytest.mark.parametrize("name", ["room_tight", "room_live"])
def test_ir_early_reflections_reach_both_channels_early(monkeypatch, name):
    import generate_irs

    class Silent:
        def standard_normal(self, n):
            return np.zeros(n)

    monkeypatch.setattr(generate_irs, "_rng", lambda seed: Silent())
    er = generate_irs.IR_GENERATORS[name]()  # early reflections only
    sr = generate_irs.SR
    early = np.sum(er[:int(0.08 * sr)] ** 2, axis=0)
    assert np.sum(er[int(0.1 * sr):] ** 2) < 0.01 * early.sum()
    assert abs(10 * np.log10(early[1] / early[0])) < 3.0


# 14. EQ presets, pitch-correct message --------------------------------------------

def test_eq_rejects_presets_without_filters_and_lists_only_eq(capsys):
    import apply_eq
    with pytest.raises(ValueError, match="no EQ filters"):
        apply_eq._load_preset("comp_kick_in")
    apply_eq.list_presets()
    listed = capsys.readouterr().out
    assert "kick_in" in listed
    assert "comp_" not in listed and "gate_" not in listed and "sat_" not in listed


def test_pitch_correct_names_project_env():
    source = (TOOLS / "apply_pitch_correct.py").read_text()
    assert "music-studio-agent" not in source


# 16. Shared envelope follower --------------------------------------------------------

def test_deesser_reuses_compression_envelope():
    import apply_compression
    import apply_deesser
    assert apply_deesser._envelope_follower is apply_compression._sidechain_gain_envelope
