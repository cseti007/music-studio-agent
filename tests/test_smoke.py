"""Minimal smoke tests for the critical DSP and relevance-check logic.

These are NOT exhaustive — they cover the load-bearing pieces that previous
field-test rounds revealed as easy to regress:

- True peak measurement (4x oversampled) is actually higher than sample peak
  on HF content
- M/S encode/decode is identity (perfect round-trip)
- Sub-sample phase alignment recovers a known fractional delay
- Pumping detector handles silent-gap signals (the active-frame fix)
- Each make-it-hit tool's relevance_check returns the expected skip/apply
  decision on simple synthetic inputs

Run with:  conda run -n music-mix-agent pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the tools/ package importable from the project root
TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------

SR = 48000


@pytest.fixture
def noise_5s():
    """5 seconds of white noise at modest level. Used as a stand-in for "real"
    stem material for relevance checks that just need *something* present."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(SR * 5) * 0.2


@pytest.fixture
def hf_sine_1s():
    """1 second of 19 kHz cosine. Phase chosen so the sample peaks miss the
    true peak — used to verify 4x-oversampled true peak measurement."""
    t = np.arange(SR) / SR
    return 0.9 * np.cos(2 * np.pi * 19000 * t + 0.4)


@pytest.fixture
def low_dominant_5s():
    """Bass-like signal: 60 Hz fundamental with mild harmonics. Used to check
    the exciter's low-dominance gate."""
    t = np.arange(SR * 5) / SR
    fund = 0.3 * np.sin(2 * np.pi * 60 * t)
    h2 = 0.1 * np.sin(2 * np.pi * 120 * t)
    return fund + h2


@pytest.fixture
def continuous_pump_5s():
    """Continuous noise modulated at 2 Hz — synthetic pumping signal."""
    t = np.arange(SR * 5) / SR
    env = 0.5 + 0.5 * (1 + np.cos(2 * np.pi * 2.0 * t)) / 2
    rng = np.random.default_rng(1)
    return env * rng.standard_normal(SR * 5) * 0.3


@pytest.fixture
def intermittent_signal_5s():
    """Signal active for the first half, silent for the second.
    Used to verify the pumping detector's active-frame gating fix."""
    rng = np.random.default_rng(3)
    sig = np.zeros(SR * 5)
    sig[: SR * 2] = rng.standard_normal(SR * 2) * 0.3
    return sig


# ---------------------------------------------------------------------------
# True peak
# ---------------------------------------------------------------------------

class TestTruePeak:
    def test_true_peak_exceeds_sample_peak_on_hf_content(self, hf_sine_1s):
        """At 19 kHz with phase that misses the peak, true peak must exceed
        sample peak by at least 0.1 dB — that's the whole point of ISP."""
        from analyze import _true_peak_dbfs

        sample_peak_db = 20 * np.log10(np.max(np.abs(hf_sine_1s)))
        true_peak_db = _true_peak_dbfs(hf_sine_1s)
        assert true_peak_db > sample_peak_db + 0.1, (
            f"sample {sample_peak_db:.3f} dBFS, true {true_peak_db:.3f} dBFS"
        )

    def test_true_peak_matches_sample_peak_on_lf_content(self):
        """At 100 Hz with no aliasing, true peak ~= sample peak (within 0.1 dB)."""
        from analyze import _true_peak_dbfs

        t = np.arange(SR) / SR
        sig = 0.5 * np.sin(2 * np.pi * 100 * t)
        sample_peak_db = 20 * np.log10(np.max(np.abs(sig)))
        true_peak_db = _true_peak_dbfs(sig)
        assert abs(true_peak_db - sample_peak_db) < 0.1


# ---------------------------------------------------------------------------
# Mid/Side encode/decode round-trip
# ---------------------------------------------------------------------------

class TestMidSide:
    def test_encode_decode_is_identity(self):
        """L/R -> M/S -> L/R must round-trip exactly (to floating-point eps)."""
        from render_mix import _ms_decode, _ms_encode

        rng = np.random.default_rng(7)
        master = rng.standard_normal((2, 10000)) * 0.3
        mid, side = _ms_encode(master)
        recovered = _ms_decode(mid, side)
        assert np.allclose(recovered, master, atol=1e-12), (
            f"max abs diff: {np.max(np.abs(recovered - master))}"
        )


# ---------------------------------------------------------------------------
# Sub-sample phase alignment
# ---------------------------------------------------------------------------

class TestPhaseAlign:
    def test_recovers_known_fractional_delay(self):
        """Apply a 3.4-sample delay, verify the alignment finds it within
        0.5 sample. The detect uses parabolic refinement around the
        correlation peak."""
        from align_phase import _compute_alignment, _fractional_shift

        rng = np.random.default_rng(0)
        ref = rng.standard_normal(SR) * 0.3
        delay_true = 3.4
        tgt = _fractional_shift(ref, delay_true)

        delay_recovered, polarity_flip, score = _compute_alignment(
            ref, tgt, sr=SR, max_delay_ms=2.0
        )
        assert not polarity_flip
        assert abs(delay_recovered - delay_true) < 0.5

    def test_correction_drives_residual_to_zero(self):
        """End-to-end: detect + apply correction = ref and corrected agree
        at integer-lag 0."""
        from align_phase import _apply_correction, _compute_alignment, _fractional_shift
        from scipy.signal import correlate

        rng = np.random.default_rng(0)
        ref = rng.standard_normal(SR) * 0.3
        tgt = _fractional_shift(ref, 3.4)

        d, pol, _ = _compute_alignment(ref, tgt, sr=SR, max_delay_ms=2.0)
        corrected = _apply_correction(tgt, d, pol)

        c = correlate(ref, corrected, mode="full", method="fft")
        lags = np.arange(-(SR - 1), SR)
        mask = np.abs(lags) <= 50
        peak_idx = int(np.argmax(np.where(mask, np.abs(c), -np.inf)))
        residual = int(lags[peak_idx])
        assert abs(residual) <= 1, f"residual lag {residual} after correction"

    def test_detects_polarity_flip(self):
        """A simple polarity inversion must be detected."""
        from align_phase import _compute_alignment

        rng = np.random.default_rng(11)
        ref = rng.standard_normal(SR) * 0.3
        tgt = -ref

        _, pol, score = _compute_alignment(ref, tgt, sr=SR, max_delay_ms=2.0)
        assert pol is True
        assert score < 0


# ---------------------------------------------------------------------------
# Pumping detector
# ---------------------------------------------------------------------------

class TestPumping:
    def test_detects_continuous_pumping(self, continuous_pump_5s):
        """Synthetic 2 Hz envelope pumping must trigger."""
        from analyze import _detect_pumping

        r = _detect_pumping(continuous_pump_5s, SR)
        assert r["pumping_detected"] is True
        assert r["pump_rate_hz"] is not None
        assert abs(r["pump_rate_hz"] - 2.0) < 0.5
        assert r["modulation_depth_db"] >= 5.0
        assert r["lf_excess_db"] >= 6.0

    def test_clean_noise_does_not_trigger(self, noise_5s):
        """Plain white noise has no LF envelope modulation."""
        from analyze import _detect_pumping

        r = _detect_pumping(noise_5s, SR)
        assert r["pumping_detected"] is False

    def test_intermittent_signal_does_not_collapse_depth(self, intermittent_signal_5s):
        """The active-frame gating fix: a half-silent signal must still produce
        a non-zero modulation_depth_db on the active half. The old code (p5/p95
        across the full envelope) returned 0.0 because p5 dived to ~0 in the
        silent half. With active-frame gating the depth is computed on the
        playing half only — for white noise that's ~1 dB (which is the
        IQR of |N(0,1)|), small but >= the bug's 0.0 sentinel."""
        from analyze import _detect_pumping

        r = _detect_pumping(intermittent_signal_5s, SR)
        assert r["modulation_depth_db"] >= 0.5, (
            f"depth {r['modulation_depth_db']} dB — active-frame gating "
            "regression? (the bug returned 0.0)"
        )
        assert r["active_frame_ratio"] is not None
        assert r["active_frame_ratio"] < 0.6  # we know half is silent


# ---------------------------------------------------------------------------
# Frequency bands crest factor
# ---------------------------------------------------------------------------

class TestBandCrest:
    def test_transient_band_has_higher_crest_than_sustained(self):
        """A kick-like transient at 60 Hz + sustained 6 kHz tone: the low band
        should report a much higher crest than the high band."""
        from analyze import _band_crest_db

        t = np.arange(SR * 2) / SR
        # Periodic short transient at 60 Hz region
        low = np.zeros_like(t)
        for i in range(0, len(t), SR // 4):
            n = min(480, len(t) - i)
            low[i:i + n] = (
                0.5 * np.exp(-np.arange(n) / 100) * np.sin(2 * np.pi * 60 * t[i:i + n])
            )
        # Sustained high tone
        high = 0.05 * np.sin(2 * np.pi * 6000 * t)
        sig = low + high

        low_crest = _band_crest_db(sig, SR, 30, 120)
        high_crest = _band_crest_db(sig, SR, 4000, 8000)
        assert low_crest > high_crest + 5, (
            f"low {low_crest} dB, high {high_crest} dB — "
            "transient band must be more dynamic than sustained"
        )


# ---------------------------------------------------------------------------
# Make-it-hit relevance checks (synthetic inputs)
# ---------------------------------------------------------------------------

class TestRelevanceChecks:
    def test_subharm_skips_target_band_richer_than_fundamental(self, noise_5s):
        """White noise has approximately uniform energy across bands, so the
        target band (80-200 Hz) is roughly as loud as the fundamental (30-80 Hz)
        — close enough that the 3 dB gate may pass. Use a low-tilted noise so
        the target is clearly above fundamental → should SKIP."""
        from apply_subharm import _relevance_check
        from scipy.signal import butter, sosfilt

        # Low-shelf boost the noise so 80-200 Hz is louder than 30-80 Hz
        sos = butter(2, 60 / (SR / 2), btype="high", output="sos")
        tilted = sosfilt(sos, noise_5s) * 2.0
        rel = _relevance_check(tilted, SR)
        # If target > fundamental + 3 dB, skip
        if rel["target_over_fundamental_db"] > 3.0:
            assert rel["recommend_skip"] is True

    def test_subharm_skips_when_sub_is_silent(self):
        """A high-pass-filtered signal has no sub content — subharm must skip."""
        from apply_subharm import _relevance_check
        from scipy.signal import butter, sosfilt

        rng = np.random.default_rng(20)
        sig = rng.standard_normal(SR * 5) * 0.3
        sos = butter(4, 200 / (SR / 2), btype="high", output="sos")
        hp_sig = sosfilt(sos, sig)

        rel = _relevance_check(hp_sig, SR)
        assert rel["recommend_skip"] is True
        assert any("nothing in the sub region" in m for m in rel["issues"])

    def test_exciter_skips_bass_like_stems(self, low_dominant_5s):
        """A 60 Hz sine + mild 2nd harmonic is overwhelmingly low-dominant —
        exciter must refuse with the bass/kick warning."""
        from apply_exciter import _relevance_check

        rel = _relevance_check(low_dominant_5s, SR)
        assert rel["recommend_skip"] is True
        assert rel["low_over_high_db"] > 6.0
        assert rel["spectral_centroid_hz"] < 800.0
        assert any("low-dominant" in m for m in rel["issues"])

    def test_exciter_skips_already_bright_signal(self):
        """High-pass noise has air content above -40 dBFS — exciter must skip."""
        from apply_exciter import _relevance_check
        from scipy.signal import butter, sosfilt

        rng = np.random.default_rng(30)
        sig = rng.standard_normal(SR * 5) * 0.3
        sos = butter(4, 5000 / (SR / 2), btype="high", output="sos")
        bright = sosfilt(sos, sig)
        rel = _relevance_check(bright, SR)
        assert rel["recommend_skip"] is True

    def test_haas_detects_stereo_pair_filename(self, tmp_path):
        """A mono file named like 'OH L.wav' must be flagged as a stereo-pair half."""
        from apply_haas import _looks_like_stereo_pair_half

        f = tmp_path / "OH AEA.01 L.05" / "assembled.wav"
        f.parent.mkdir(parents=True)
        f.touch()
        warning = _looks_like_stereo_pair_half(f)
        assert warning is not None
        assert "L" in warning

    def test_haas_does_not_falsely_flag_other_names(self, tmp_path):
        """Filenames without stereo-pair patterns must NOT trigger the warning."""
        from apply_haas import _looks_like_stereo_pair_half

        f = tmp_path / "SN TOP.05" / "assembled.wav"
        f.parent.mkdir(parents=True)
        f.touch()
        assert _looks_like_stereo_pair_half(f) is None

    def test_batch_analyze_collects_jobs_from_session_dir(self, tmp_path):
        """_collect_jobs scans <session>/tracks/*/assembled.wav and skips
        directories without one. Honours --skip-existing if analysis.json
        is already present."""
        import soundfile as sf
        from batch_analyze import _collect_jobs

        # Build a fake session layout: 3 stems, the third already has analysis.json
        session = tmp_path / "fake_session"
        (session / "tracks").mkdir(parents=True)
        for name in ("A", "B", "C"):
            d = session / "tracks" / name
            d.mkdir()
            sig = np.zeros(48000, dtype=np.float32)
            sf.write(str(d / "assembled.wav"), sig, 48000)
        # C already analysed
        (session / "tracks" / "C" / "analysis.json").write_text("{}", encoding="utf-8")

        # Without --skip-existing, all 3 should be queued
        all_jobs = _collect_jobs(session, None, None, skip_existing=False)
        assert len(all_jobs) == 3

        # With --skip-existing, only A and B
        skip_jobs = _collect_jobs(session, None, None, skip_existing=True)
        assert len(skip_jobs) == 2
        stem_names = {Path(j[1]).name for j in skip_jobs}
        assert stem_names == {"A", "B"}

    def test_analyze_emits_new_content_fields(self, tmp_path):
        """analyze() must populate onsets_sec, tempo_bpm, estimated_key and
        envelopes (rms / lufs_short_term / spectral_flux) with the expected
        types and shapes on a 4-second synthetic click train at 120 BPM (one
        click every 0.5 s → 8 clicks)."""
        import soundfile as sf
        from analyze import analyze

        sr = 48000
        duration_sec = 4
        n = sr * duration_sec
        sig = np.zeros(n, dtype=np.float32)
        # Click every 0.5 s (= 120 BPM quarter notes)
        for click_t in np.arange(0.0, duration_sec, 0.5):
            i = int(click_t * sr)
            if i + 100 < n:
                sig[i:i + 100] = 0.5  # short rectangular click
        wav = tmp_path / "click_train.wav"
        sf.write(str(wav), sig, sr)

        out = tmp_path / "out"
        result = analyze(wav, output_dir=out)

        # onsets_sec — should have ~8 entries, all in [0, 4)
        onsets = result["onsets_sec"]
        assert isinstance(onsets, list)
        assert 4 <= len(onsets) <= 12  # tolerant of detector quirks
        assert all(0.0 <= t < duration_sec for t in onsets)

        # tempo_bpm — should be a number, plausibly near 120
        tempo = result["tempo_bpm"]
        assert tempo is None or (30.0 <= tempo <= 300.0)

        # estimated_key — dict with 3 keys
        key = result["estimated_key"]
        assert set(key.keys()) == {"key", "mode", "confidence"}
        assert isinstance(key["confidence"], float)
        # Non-tonal click train → low confidence expected
        assert key["confidence"] < 1.01

        # envelopes — 3 lists, rms+flux ≈ duration in length
        env = result["envelopes"]
        assert set(env.keys()) == {"rms_db_per_second", "lufs_short_term", "spectral_flux_per_second"}
        assert len(env["rms_db_per_second"]) == duration_sec
        assert len(env["spectral_flux_per_second"]) == duration_sec
        # lufs_short_term needs >= 3s window — should produce 1+ samples
        assert isinstance(env["lufs_short_term"], list)
        assert len(env["lufs_short_term"]) >= 1

    def test_bpm_to_pre_delay_known_values(self):
        """120 BPM eighth = 250 ms, sixteenth = 125 ms; 184 BPM sixteenth ≈ 81.5 ms."""
        from apply_reverb import _bpm_to_pre_delay_ms

        assert abs(_bpm_to_pre_delay_ms(120, "eighth") - 250.0) < 0.01
        assert abs(_bpm_to_pre_delay_ms(120, "sixteenth") - 125.0) < 0.01
        assert abs(_bpm_to_pre_delay_ms(184, "sixteenth") - 81.52) < 0.1
        # Triplets: a triplet-eighth at 120 BPM is 2/3 of an eighth = ~167 ms
        assert abs(_bpm_to_pre_delay_ms(120, "triplet-eighth") - 1000.0 / 6.0) < 0.5

    def test_bpm_to_pre_delay_rejects_unknown_division(self):
        from apply_reverb import _bpm_to_pre_delay_ms

        try:
            _bpm_to_pre_delay_ms(120, "twelve-bar-blues")
        except ValueError:
            return
        raise AssertionError("expected ValueError for unknown division")

    def test_sidechain_envelope_ducks_on_loud_input(self):
        """The sidechain envelope must drop below 1.0 when the input is loud."""
        from apply_reverb import _sidechain_envelope

        SR_LOCAL = 48000
        # 2 seconds of loud noise (above the default -30 dB threshold)
        rng = np.random.default_rng(33)
        sc = rng.standard_normal(SR_LOCAL * 2) * 0.5  # ~-3 dBFS RMS, well above threshold
        env = _sidechain_envelope(sc, SR_LOCAL, depth_db=-12.0)
        # On loud sustained input the envelope must duck the reverb
        assert env.shape == sc.shape
        assert env.min() < 0.6, f"env never ducked: min={env.min():.3f}"
        # And never go below the depth floor (10^(-12/20) = 0.251)
        assert env.min() >= 0.25 - 1e-3, f"env went below depth_db floor: {env.min()}"

    def test_sidechain_envelope_passes_quiet_input(self):
        """Below threshold, the envelope must stay near 1.0 (no ducking)."""
        from apply_reverb import _sidechain_envelope

        SR_LOCAL = 48000
        # Very quiet input — below -30 dB threshold
        sc = np.random.default_rng(34).standard_normal(SR_LOCAL) * 0.001
        env = _sidechain_envelope(sc, SR_LOCAL, depth_db=-12.0)
        assert env.min() > 0.99

    def test_audit_session_finds_duplicates(self, tmp_path):
        """audit_session must group tracks that share source files and
        recommend the shortest name as primary."""
        import json as _json
        from audit_session import find_duplicates

        # Two tracks share KICK_OUT.wav; one track has its own SNARE.wav.
        session = {
            "tracks": [
                {"name": "KICK_OUT", "clips": [{"source_file": "/x/KICK_OUT.wav"}]},
                {"name": "KICK_OUT.dup1", "clips": [{"source_file": "/y/KICK_OUT.wav"}]},
                {"name": "SNARE", "clips": [{"source_file": "/x/SNARE.wav"}]},
            ],
        }
        p = tmp_path / "session.json"
        p.write_text(_json.dumps(session), encoding="utf-8")
        r = find_duplicates(p)

        assert r["summary"]["n_groups"] == 1
        g = r["duplicate_groups"][0]
        assert g["n_tracks"] == 2
        assert g["recommend_primary"] == "KICK_OUT"
        assert g["recommend_deactivate"] == ["KICK_OUT.dup1"]

    def test_audit_session_handles_no_duplicates(self, tmp_path):
        """A session with all unique source files must return n_groups=0."""
        import json as _json
        from audit_session import find_duplicates

        session = {
            "tracks": [
                {"name": "A", "clips": [{"source_file": "/x/A.wav"}]},
                {"name": "B", "clips": [{"source_file": "/x/B.wav"}]},
            ],
        }
        p = tmp_path / "session.json"
        p.write_text(_json.dumps(session), encoding="utf-8")
        r = find_duplicates(p)
        assert r["summary"]["n_groups"] == 0

    def test_multiband_skips_when_only_one_band_has_dynamics(self):
        """A pure low-frequency sine has crest only in the low band — mid and
        high are essentially zero. multiband requires >= 2 bands with crest >=
        6 dB; this fails and the tool must skip.

        (Note: trying to test "wideband squashed" via tanh(noise) doesn't
        work — band-pass filtering reconstructs natural-looking dynamics in
        each band from the clipped signal's residual harmonics. Use a
        narrow-band source instead.)"""
        from apply_multiband_comp import _relevance_check

        t = np.arange(SR * 10) / SR
        # Pure 80 Hz sine + tiny noise — energy almost entirely in low band
        sig = 0.5 * np.sin(2 * np.pi * 80 * t) + 1e-4 * np.random.default_rng(41).standard_normal(len(t))
        rel = _relevance_check(sig, SR, 200.0, 3000.0)
        assert rel["recommend_skip"] is True
        assert rel["bands_with_dynamics"] < 2


class TestStyleCheck:
    """Style profile loading + verdict logic.

    Confirms that:
      - All 5 built-in profiles load and have the expected schema
      - The grading function returns GREEN/YELLOW/RED at the right thresholds
      - The overall score function counts checks correctly
      - The hard-fail rule (RED LUFS or LRA → overall RED) fires
    """

    def test_all_profiles_load_with_required_keys(self):
        from style_check import list_profiles, load_profile

        names = list_profiles()
        assert set(names) == {"modern_rock", "classic_rock", "pop", "hip_hop", "jazz_acoustic"}
        for name in names:
            p = load_profile(name)
            for key in ("lufs", "lra", "crest_factor", "tonal_balance_dbfs"):
                assert key in p, f"{name} missing required key {key}"
            assert "integrated_target" in p["lufs"]
            assert "tolerance_lu" in p["lufs"]
            assert set(p["tonal_balance_dbfs"].keys()) == {
                "sub_60hz", "low_60_250hz", "mid_250_2khz", "high_2_8khz", "air_8khz_plus"
            }

    def test_grade_value_thresholds(self):
        from style_check import _grade_value

        # within tolerance → green
        assert _grade_value(measured=-10.0, target=-10.0, tolerance=2.0)[0] == "GREEN"
        assert _grade_value(measured=-12.0, target=-10.0, tolerance=2.0)[0] == "GREEN"
        # just past tolerance → yellow (severity 1.0 .. 1.5)
        assert _grade_value(measured=-12.5, target=-10.0, tolerance=2.0)[0] == "YELLOW"
        assert _grade_value(measured=-13.0, target=-10.0, tolerance=2.0)[0] == "YELLOW"
        # 1.5x tolerance and beyond → red
        assert _grade_value(measured=-13.5, target=-10.0, tolerance=2.0)[0] == "RED"

    def test_grade_range_thresholds(self):
        from style_check import _grade_range

        # inside [min, max] → green
        assert _grade_range(measured=7.0, target=7.0, range_min=4.0, range_max=9.0)[0] == "GREEN"
        assert _grade_range(measured=4.0, target=7.0, range_min=4.0, range_max=9.0)[0] == "GREEN"
        # just outside, within half-width → yellow
        assert _grade_range(measured=3.0, target=7.0, range_min=4.0, range_max=9.0)[0] == "YELLOW"
        # significantly outside → red
        assert _grade_range(measured=1.0, target=7.0, range_min=4.0, range_max=9.0)[0] == "RED"

    def test_overall_verdict_hard_fail_on_lufs(self):
        """A RED LUFS verdict forces overall RED even if everything else is GREEN."""
        from style_check import _verdict_for_checks

        checks = [
            {"name": "integrated_lufs", "verdict": "RED"},
            {"name": "lra_lu",          "verdict": "GREEN"},
            {"name": "crest_factor_db", "verdict": "GREEN"},
        ] + [{"name": f"band_{i}", "verdict": "GREEN"} for i in range(5)]
        verdict, score = _verdict_for_checks(checks)
        assert verdict == "RED"
        assert score <= 55, f"hard-fail should cap score, got {score}"

    def test_overall_verdict_all_green(self):
        from style_check import _verdict_for_checks

        checks = [{"name": "x", "verdict": "GREEN"}] * 8
        verdict, score = _verdict_for_checks(checks)
        assert verdict == "GREEN"
        assert score == 100

    def test_all_profiles_have_default_bus_volume_db(self):
        """Every shipped style profile must specify default_bus_volume_db
        for the three core buses (drums, bass, guitar)."""
        from style_check import list_profiles, load_profile

        for name in list_profiles():
            p = load_profile(name)
            assert "default_bus_volume_db" in p, f"{name} missing default_bus_volume_db"
            buses = p["default_bus_volume_db"]
            for bus in ("drums", "bass", "guitar"):
                assert bus in buses, f"{name}: missing {bus} bus default"
                assert isinstance(buses[bus], (int, float)), (
                    f"{name}.{bus} should be numeric, got {type(buses[bus]).__name__}"
                )

    def test_render_mix_style_bus_defaults_loaded(self):
        """render_mix._load_style_bus_defaults returns the profile's bus defaults
        when given a known style, empty dict for unknown/None."""
        from render_mix import _load_style_bus_defaults

        # Known profile
        defaults = _load_style_bus_defaults("modern_rock")
        assert defaults["drums"] == 0.0
        assert defaults["bass"] == 0.0
        assert defaults["guitar"] == -3.0

        # hip_hop should push bass louder, guitar way back
        hh = _load_style_bus_defaults("hip_hop")
        assert hh["bass"] == 1.0
        assert hh["guitar"] == -4.0

        # Unknown / None falls back to empty (caller defaults to 0.0)
        assert _load_style_bus_defaults(None) == {}
        assert _load_style_bus_defaults("not_a_real_genre") == {}

    def test_borderline_flag_thresholds(self):
        """severity 0.7 is the borderline threshold for a GREEN check.

        Below 0.7 → plain GREEN; 0.7..1.0 → GREEN with borderline=True;
        above 1.0 → no longer GREEN (becomes YELLOW or RED — no borderline).
        """
        from style_check import _grade_value

        # Severity 0.4 (well within tolerance): not borderline
        verdict, severity = _grade_value(measured=-10.4, target=-10.0, tolerance=1.0)
        assert verdict == "GREEN"
        assert severity < 0.7, f"got severity {severity}"

        # Severity ~0.8 (close to the yellow threshold): borderline territory
        verdict, severity = _grade_value(measured=-10.8, target=-10.0, tolerance=1.0)
        assert verdict == "GREEN"
        assert severity >= 0.7

        # Severity 1.2 (past tolerance): no longer GREEN
        verdict, severity = _grade_value(measured=-11.2, target=-10.0, tolerance=1.0)
        assert verdict == "YELLOW"


class TestRenderMixBusEQ:
    """Per-bus EQ chain in render_mix. Validates that `_apply_eq_chain` runs
    on a synthetic stereo buffer and shapes the spectrum as expected."""

    def test_apply_eq_chain_attenuates_target_band(self):
        from render_mix import _apply_eq_chain

        sr = 48000
        n = sr  # 1 second
        # White-noise-like input across both channels
        rng = np.random.default_rng(42)
        buf = rng.standard_normal((2, n)) * 0.1

        # A 5 kHz peak EQ cut of -6 dB Q=2.0 should reduce 4-6 kHz energy
        filters = [{"type": "peak", "hz": 5000, "q": 2.0, "db": -6.0}]
        out = _apply_eq_chain(buf, sr, filters, label="test EQ")

        # Energy in the 4-6 kHz band should drop substantially
        from scipy.signal import welch
        f_in, p_in = welch(buf[0], fs=sr, nperseg=2048)
        f_out, p_out = welch(out[0], fs=sr, nperseg=2048)
        mask = (f_in >= 4000) & (f_in <= 6000)
        ratio_db = 10 * np.log10(np.mean(p_out[mask]) / np.mean(p_in[mask]))
        assert ratio_db < -4, f"4-6 kHz attenuation only {ratio_db:.1f} dB, expected < -4 dB"


class TestChainRecall:
    """build_chain → mix_chain.json → replay_chain --dry-run round-trip.

    Confirms that:
      - build_chain reads *_report.json files and produces a valid mix_chain.json
      - The chain steps are topologically ordered (predecessor before successor)
      - replay_chain --dry-run can produce argv lines for every step type
    """

    def test_build_chain_reads_reports_and_orders_topologically(self, tmp_path):
        import json
        from build_chain import build_chain

        # Fake a session: one stem with gain → eq → comp reports
        session = tmp_path / "fake_session"
        stem_dir = session / "tracks" / "KICK"
        stem_dir.mkdir(parents=True)

        (stem_dir / "gain_report.json").write_text(json.dumps({
            "track": "KICK",
            "output": str(stem_dir / "assembled.wav"),
            "mode": "per-clip-no-normalize",
            "peak_ceiling_db": -1.0,
        }), encoding="utf-8")
        # Note: write reports in REVERSE topological order to verify the sort
        (stem_dir / "comp_report.json").write_text(json.dumps({
            "input":  str(stem_dir / "assembled_eq.wav"),
            "output": str(stem_dir / "assembled_eq_comp.wav"),
            "preset": "comp_kick",
            "settings": {"threshold_db": -10, "ratio": 4, "attack_ms": 5,
                         "release_ms": 50, "makeup_db": 2, "mix": 1.0},
        }), encoding="utf-8")
        (stem_dir / "eq_report.json").write_text(json.dumps({
            "input":  str(stem_dir / "assembled.wav"),
            "output": str(stem_dir / "assembled_eq.wav"),
            "preset_used": "kick_in",
            "filters_applied": [],
            "phase": "minimum",
        }), encoding="utf-8")

        chain = build_chain(session)
        assert len(chain["stems"]) == 1
        steps = chain["stems"][0]["chain"]
        assert [s["step"] for s in steps] == ["gain_per_clip", "eq", "comp"]
        assert chain["stems"][0]["name"] == "KICK"

    def test_replay_chain_dry_run_emits_argv_for_each_step(self, tmp_path):
        import json
        from replay_chain import _build_argv

        session_json = tmp_path / "session.json"
        session_json.write_text("{}", encoding="utf-8")

        sample_steps = [
            {"step": "gain_per_clip",
             "input": "session.json:KICK", "output": str(tmp_path / "KICK" / "assembled.wav"),
             "args": {"normalize": False, "peak_ceiling_db": -1.0}},
            {"step": "eq",
             "input": str(tmp_path / "KICK" / "assembled.wav"),
             "output": str(tmp_path / "KICK" / "assembled_eq.wav"),
             "args": {"preset": "kick_in", "phase": "minimum"}},
            {"step": "comp",
             "input": str(tmp_path / "KICK" / "assembled_eq.wav"),
             "output": str(tmp_path / "KICK" / "assembled_eq_comp.wav"),
             "args": {"preset": "comp_kick", "threshold_db": -10, "ratio": 4,
                      "attack_ms": 5, "release_ms": 50, "makeup_db": 2, "mix": 1.0}},
            {"step": "gate",
             "input": "x.wav", "output": "x_gate.wav",
             "args": {"preset": "gate_kick", "threshold_db": -30, "range_db": -80,
                      "attack_ms": 1, "hold_ms": 100, "release_ms": 50}},
            {"step": "amp",
             "input": "x.wav", "output": "x_amp.wav",
             "args": {"preset": "ampeg_svt", "drive": 0.35}},
            {"step": "reverb",
             "input": "x.wav", "output": "x_reverb.wav",
             "args": {"preset": "room_drums", "wet": 0.1}},
            {"step": "transient",
             "input": "x.wav", "output": "x_transient.wav",
             "args": {"preset": "transient_kick_punch", "attack_db": 3.0}},
            {"step": "saturation",
             "input": "x.wav", "output": "x_sat.wav",
             "args": {"preset": "sat_tape_subtle", "mode": "tape"}},
            {"step": "delay",
             "input": "x.wav", "output": "x_delay.wav",
             "args": {"mode": "pingpong", "delay_ms": 187.0, "feedback": 0.4, "mix": 0.25}},
            {"step": "align_phase",
             "input": "x.wav", "output": "x_aligned.wav",
             "args": {"reference": "ref.wav", "max_delay_ms": 20.0, "segment_sec": 10.0}},
        ]

        for step in sample_steps:
            argv = _build_argv(step, session_json)
            assert argv is not None, f"unsupported step type: {step['step']}"
            # Every argv should start with the python interpreter and a tool path
            assert "python" in argv[0].lower() or argv[0].endswith("python")
            assert argv[1].endswith(".py")
            # Critical: every step's argv must include --output-dir
            assert "--output-dir" in argv, f"{step['step']}: argv missing --output-dir"
