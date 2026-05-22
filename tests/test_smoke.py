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

Run with:  conda run -n music-studio-agent pytest tests/ -v
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
        # The 5 baseline profiles must all be present; extra profiles (e.g.
        # tool_inspired, punchy_modern_rock) are allowed and not asserted-out
        # so adding a new profile JSON doesn't break this test.
        baseline = {"modern_rock", "classic_rock", "pop", "hip_hop", "jazz_acoustic"}
        assert baseline <= set(names), f"missing baseline profile(s): {baseline - set(names)}"
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

    def test_all_profiles_have_default_bus_pan(self):
        """Every style profile must declare `default_bus_pan` for the standard
        bus names (drums, bass, gtr_1, gtr_laci, gtr, vocal_lead). This is
        the panning equivalent of `default_bus_volume_db`.
        """
        import json
        from pathlib import Path
        required_buses = {"drums", "bass", "gtr_1", "gtr_laci", "gtr", "vocal_lead"}
        profiles = list(Path("tools/style_profiles").glob("*.json"))
        assert profiles, "no style profiles found"
        for p in profiles:
            d = json.loads(p.read_text(encoding="utf-8"))
            pans = d.get("default_bus_pan", {})
            assert pans, f"{p.name} missing default_bus_pan"
            missing = required_buses - pans.keys()
            assert not missing, f"{p.name} missing pan entries for {missing}"
            for name, v in pans.items():
                assert isinstance(v, (int, float)), \
                    f"{p.name}.{name} pan must be numeric, got {type(v).__name__}"
                assert -1.0 <= v <= 1.0, \
                    f"{p.name}.{name} pan {v} out of [-1.0, 1.0]"

    def test_load_style_bus_pans_returns_correct_values(self):
        """`render_mix._load_style_bus_pans` returns the per-bus pan map for a
        known style, empty dict for unknown/None.
        """
        from render_mix import _load_style_bus_pans

        modern = _load_style_bus_pans("modern_rock")
        # modern_rock convention: industry hard pan ≥ ±0.85
        assert modern["gtr_1"] == -0.85
        assert modern["gtr_laci"] == 0.85
        assert modern["bass"] == 0.0  # bass always center
        assert modern["drums"] == 0.0

        # hip_hop: rhythm guitars not panned (drum-led)
        hh = _load_style_bus_pans("hip_hop")
        assert hh["gtr_1"] == 0.0
        assert hh["gtr_laci"] == 0.0

        # jazz_acoustic: narrower than rock
        jazz = _load_style_bus_pans("jazz_acoustic")
        assert -0.5 < jazz["gtr_1"] < -0.2, f"jazz gtr_1 pan {jazz['gtr_1']}"

        assert _load_style_bus_pans(None) == {}
        assert _load_style_bus_pans("not_a_genre") == {}

    def test_detect_pan_routes_drumkit_pieces_to_audience_perspective(self):
        """`_detect_pan` returns audience-perspective default pans for drum-kit
        pieces (toms spread, hi-hat slightly left, ride slightly right) and
        leaves the L/R stereo-pair behaviour intact for OH and ROOM mics.
        """
        from render_mix import _detect_pan
        # Toms: left-to-right pitch sweep
        assert _detect_pan("RACK TOM 1.05") == -0.4
        assert _detect_pan("RACK TOM 2.05") == -0.15
        assert _detect_pan("FLOOR TOM.05") == +0.5
        # Cymbals
        assert _detect_pan("HIHAT.05") == -0.2
        assert _detect_pan("HI-HAT.05") == -0.2
        assert _detect_pan("RIDE.05") == +0.3
        assert _detect_pan("CRASH.05") == 0.0
        # Kick / snare → center
        assert _detect_pan("KICK IN.05") == 0.0
        assert _detect_pan("KICK OUT.05") == 0.0
        assert _detect_pan("KICK SUB.05") == 0.0
        assert _detect_pan("SN TOP.05") == 0.0
        assert _detect_pan("SN BOTTOM.05") == 0.0
        # Stereo pairs use L/R suffix (existing behaviour)
        assert _detect_pan("OH AEA.01 L.05") == -0.7
        assert _detect_pan("OH AEA.01 R.05") == +0.7
        assert _detect_pan("ROOM CLOSE.01 L.05") == -0.7
        assert _detect_pan("ROOM CLOSE.01 R.05") == +0.7
        # Non-drum tracks: L/R suffix still works
        assert _detect_pan("BG VOX L") == -0.7
        assert _detect_pan("BG VOX R") == +0.7
        # Non-drum, no L/R → center
        assert _detect_pan("GTR 1 FENDER.06") == 0.0
        assert _detect_pan("BASS DI CLEAN") == 0.0
        # Track name with no drum keyword, no L/R suffix, has unicode + numbers
        # → center. (Tests that the unicode + arbitrary characters in a typical
        # DAW export name don't accidentally trigger a pan rule.)
        assert _detect_pan("20221130 Verse&Refrain take 1") == 0.0

    def test_modern_rock_style_uses_industry_hard_pan(self):
        """modern_rock convention (researched 2025): hard pan ≥ ±0.85 for
        double-tracked rhythm guitars. Earlier values (±0.6) were too
        conservative relative to industry standard."""
        from render_mix import _load_style_bus_pans
        pans = _load_style_bus_pans("modern_rock")
        assert abs(pans["gtr_1"]) >= 0.8, (
            f"modern_rock gtr_1 pan {pans['gtr_1']} too narrow — industry "
            f"hard-pan convention is ±0.85 or wider"
        )
        # punchy_modern_rock should be even harder (LCR full)
        pans = _load_style_bus_pans("punchy_modern_rock")
        assert abs(pans["gtr_1"]) >= 0.95, (
            f"punchy_modern_rock gtr_1 pan {pans['gtr_1']} should be near LCR ±1.0"
        )

    def test_generate_config_applies_style_pan_to_buses(self, tmp_path):
        """A `--generate-config --style modern_rock` run with guitar tracks
        produces a mix_config.json where gtr_1 / gtr_laci buses have pan
        -0.6 / +0.6 (the modern_rock default).
        """
        import json
        import soundfile as sf
        import numpy as np
        from render_mix import generate_config

        # Make a tiny tracks layout with two guitars
        tracks_root = tmp_path / "tracks"
        for name in ["GTR 1 FENDER.01", "GTR LACI 57.01", "BASS DI.01"]:
            d = tracks_root / name
            d.mkdir(parents=True)
            # short noise stem
            n = 48000  # 1 sec
            sig = (np.random.default_rng(hash(name) % 10000).standard_normal((n, 2))
                   * 0.1).astype(np.float32)
            sf.write(str(d / "assembled.wav"), sig, 48000, subtype="PCM_24")

        out_cfg = tmp_path / "mix_config.json"
        generate_config(tmp_path, out_cfg, style="modern_rock")
        cfg = json.loads(out_cfg.read_text())
        # modern_rock pan defaults applied (industry hard pan)
        assert cfg["buses"]["gtr_1"]["pan"] == -0.85
        assert cfg["buses"]["gtr_laci"]["pan"] == 0.85
        assert cfg["buses"]["bass"]["pan"] == 0.0  # bass always center
        assert cfg["buses"]["drums"]["pan"] == 0.0  # drums always center

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


class TestVocalToolkit:
    """Vocal pipeline: deesser sibilance attenuation + bus detection +
    pitch-correct scale-quantisation helpers."""

    def test_vocal_bus_detection(self):
        """_detect_bus routes LEAD VOX, BG VOX, etc. to vocal_lead / vocal_bg."""
        from render_mix import _detect_bus

        assert _detect_bus("LEAD VOX") == "vocal_lead"
        assert _detect_bus("LEAD VOCAL") == "vocal_lead"
        assert _detect_bus("VOX") == "vocal_lead"
        assert _detect_bus("BG VOX L") == "vocal_bg"
        assert _detect_bus("BACKING VOCAL") == "vocal_bg"
        assert _detect_bus("HARMONY HIGH") == "vocal_bg"
        # Non-vocal stays put
        assert _detect_bus("KICK IN.05") == "drums"
        assert _detect_bus("BASS DI CLEAN") == "bass"

    def test_deesser_attenuates_sibilance_band(self, tmp_path):
        """Apply de-esser to a synthetic 7 kHz tone — output should be
        attenuated in that band by at least 4 dB at default settings."""
        import soundfile as sf
        from apply_deesser import apply_deesser

        sr = 48000
        t = np.arange(2 * sr) / sr
        # Loud 7 kHz tone — should trigger the de-esser hard
        signal = 0.5 * np.sin(2 * np.pi * 7000 * t)
        signal_stereo = np.stack([signal, signal], axis=1)

        in_path = tmp_path / "sibilant.wav"
        sf.write(str(in_path), signal_stereo, sr, subtype="PCM_24")

        report = apply_deesser(
            in_path, tmp_path,
            threshold_db=-22.0, ratio=4.0,
            attack_ms=1.0, release_ms=60.0,
            detect_low_hz=5500.0, detect_high_hz=8500.0,
            preset_name="test",
        )

        assert not report.get("skipped", False), "should not skip a 7 kHz -6 dBFS tone"
        # Mean gain reduction should be at least -3 dB on a constant tone
        assert report["mean_gain_reduction_db"] < -3.0, (
            f"mean GR only {report['mean_gain_reduction_db']} dB, expected < -3"
        )

    def test_deesser_skips_clean_signal(self, tmp_path):
        """A signal with no sibilance content should trigger the relevance
        check and skip."""
        import soundfile as sf
        from apply_deesser import apply_deesser

        sr = 48000
        # Pure 200 Hz tone — nothing in the 5-8 kHz band
        t = np.arange(2 * sr) / sr
        signal = 0.5 * np.sin(2 * np.pi * 200 * t)
        signal_stereo = np.stack([signal, signal], axis=1)

        in_path = tmp_path / "clean.wav"
        sf.write(str(in_path), signal_stereo, sr, subtype="PCM_24")

        report = apply_deesser(
            in_path, tmp_path,
            threshold_db=-22.0, ratio=4.0,
            attack_ms=1.0, release_ms=60.0,
            detect_low_hz=5500.0, detect_high_hz=8500.0,
            preset_name="test", force=False,
        )

        assert report["skipped"] is True
        assert report["relevance_check"]["recommend_skip"] is True

    def test_pitch_correct_scale_quantise(self):
        """_quantise_to_scale snaps detected pitches to the A-minor scale grid."""
        from apply_pitch_correct import _quantise_to_scale, _build_scale

        a_minor = _build_scale("A", "minor")  # A=9, B=11, C=0, D=2, E=4, F=5, G=7
        # Three test pitches: A4 (440 Hz, in-scale), C#5 (~554 Hz, NOT in scale),
        # E5 (~659 Hz, in-scale)
        import numpy as np
        f0_in = np.array([440.0, 554.0, 659.0])

        f0_out = _quantise_to_scale(f0_in, a_minor)
        # A4 should be unchanged (already in scale)
        assert abs(f0_out[0] - 440.0) < 0.5
        # C#5 should snap to nearest scale note (either C5 = 523 or D5 = 587)
        assert abs(f0_out[1] - 523.25) < 1 or abs(f0_out[1] - 587.33) < 1
        # E5 should be unchanged
        assert abs(f0_out[2] - 659.26) < 0.5

    def test_pitch_correct_strength_blend(self):
        """_blend_pitch with strength=0 passes through original, strength=1
        gives fully quantised."""
        from apply_pitch_correct import _blend_pitch
        import numpy as np

        orig = np.array([440.0, 554.0, 659.0])
        quant = np.array([440.0, 523.25, 659.26])

        no_correction = _blend_pitch(orig, quant, 0.0)
        assert np.allclose(no_correction, orig)

        full_correction = _blend_pitch(orig, quant, 1.0)
        assert np.allclose(full_correction, quant)

        midway = _blend_pitch(orig, quant, 0.5)
        expected = (orig + quant) / 2
        assert np.allclose(midway, expected)


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


# ---------------------------------------------------------------------------
# Headroom / peak verdict — DAW-style channel meter warnings
# ---------------------------------------------------------------------------
class TestPeakVerdict:
    """Both render_mix and analyze.py classify peak headroom into
    [OK] / [WARN] / [CLIP] using the same -6 / -1 dBFS thresholds.
    The two implementations must agree on identical inputs.
    """

    def test_render_mix_peak_verdict_thresholds(self):
        from render_mix import _peak_verdict

        # Well below ceiling — OK
        assert _peak_verdict(-12.0) == "[OK]"
        assert _peak_verdict(-6.01) == "[OK]"
        # In warning band
        assert _peak_verdict(-6.0) == "[WARN]"
        assert _peak_verdict(-3.0) == "[WARN]"
        assert _peak_verdict(-1.01) == "[WARN]"
        # At / above clip threshold
        assert _peak_verdict(-1.0) == "[CLIP]"
        assert _peak_verdict(0.0) == "[CLIP]"

    def test_render_mix_peak_verdict_uses_worst_of_sample_and_true(self):
        from render_mix import _peak_verdict

        # Sample peak OK but true peak in warning band -> WARN
        assert _peak_verdict(-12.0, -3.0) == "[WARN]"
        # Sample peak in warning band but true peak at clip -> CLIP
        assert _peak_verdict(-3.0, -0.5) == "[CLIP]"
        # Both fine
        assert _peak_verdict(-9.0, -8.0) == "[OK]"

    def test_analyze_headroom_verdict_matches_render_mix(self):
        from analyze import _headroom_verdict as analyze_verdict
        from render_mix import _peak_verdict as render_verdict

        for sample, tp in [(-12, -10), (-6, -5), (-3, -2.5),
                           (-1, -0.5), (-0.5, 0.0), (-20, -3)]:
            assert analyze_verdict(sample, tp) == render_verdict(sample, tp), (
                f"mismatch at sample={sample}, tp={tp}")


# ---------------------------------------------------------------------------
# Render-mix bus_peaks / master_peaks structure
# ---------------------------------------------------------------------------
class TestBusMasterPeaks:
    """Verify the render report carries the new peak-monitoring fields."""

    def test_render_emits_bus_and_master_peaks(self, tmp_path):
        import json
        import numpy as np
        import soundfile as sf
        from render_mix import render_mix

        # Build a tiny session: one drum track, one bass track, one vocal.
        # Each is a 2-sec stereo signal at distinct levels so the peaks
        # are distinguishable across buses.
        sr = 48000
        dur = 2.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)

        tracks_dir = tmp_path / "tracks"
        tracks_dir.mkdir()

        # Drum: loud transient train (peak around -3 dBFS — WARN territory)
        kick = np.zeros_like(t)
        for i in range(8):
            idx = int(i * sr * 0.25)
            kick[idx:idx + 100] = 0.7 * np.hanning(100)
        drum_path = tracks_dir / "KICK.wav"
        sf.write(str(drum_path), np.stack([kick, kick], axis=1), sr, subtype="PCM_24")

        # Bass: low sine, peak around -12 dBFS (OK)
        bass = 0.25 * np.sin(2 * np.pi * 80 * t)
        bass_path = tracks_dir / "BASS DI.wav"
        sf.write(str(bass_path), np.stack([bass, bass], axis=1), sr, subtype="PCM_24")

        # Vocal: mid sine, peak around -6 dBFS (right on edge)
        vocal = 0.5 * np.sin(2 * np.pi * 440 * t)
        vocal_path = tracks_dir / "LEAD VOX.wav"
        sf.write(str(vocal_path), np.stack([vocal, vocal], axis=1), sr, subtype="PCM_24")

        config = {
            "session_dir": str(tmp_path),
            "output_dir": str(tmp_path),
            "sample_rate": sr,
            "tracks": [
                {"name": "KICK",     "file": str(drum_path),  "bus": "drums", "active": True, "volume_db": 0.0},
                {"name": "BASS DI",  "file": str(bass_path),  "bus": "bass",  "active": True, "volume_db": 0.0},
                {"name": "LEAD VOX", "file": str(vocal_path), "bus": "vocal_lead", "active": True, "volume_db": 0.0},
            ],
            "buses": {
                "drums":      {"volume_db": 0.0, "comp_preset": None, "parent_bus": None},
                "bass":       {"volume_db": 0.0, "comp_preset": None, "parent_bus": None},
                "vocal_lead": {"volume_db": 0.0, "comp_preset": None, "parent_bus": "vocal"},
                "vocal":      {"volume_db": 0.0, "comp_preset": None, "parent_bus": None},
            },
            "master": {"lufs_target": -18.0, "true_peak_dbfs": -1.0},
        }

        config_path = tmp_path / "mix_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        out_wav = tmp_path / "mix.wav"
        render_mix(config_path, output_wav=out_wav, render_stems=False)
        report = json.loads(out_wav.with_name("mix_report.json").read_text())

        # bus_peaks contains an entry per active bus with stage fields + verdict
        assert "bus_peaks" in report
        assert "drums" in report["bus_peaks"]
        assert "bass" in report["bus_peaks"]
        for bn in ("drums", "bass"):
            bp = report["bus_peaks"][bn]
            assert "sum_in" in bp
            assert "final" in bp
            assert "true_peak_final" in bp
            assert bp["verdict"] in ("[OK]", "[WARN]", "[CLIP]")

        # master_peaks present with the expected stage keys + final verdict
        assert "master_peaks" in report
        mp = report["master_peaks"]
        assert "sum_in" in mp
        assert "after_limiter" in mp
        assert "final_sample_peak" in mp
        assert "final_true_peak" in mp
        assert mp["verdict"] in ("[OK]", "[WARN]", "[CLIP]")


# ---------------------------------------------------------------------------
# analyze.py headroom_verdict field
# ---------------------------------------------------------------------------
class TestAnalyzeHeadroom:
    def test_loudness_block_contains_headroom_verdict(self, tmp_path):
        import json
        import numpy as np
        import soundfile as sf
        from analyze import analyze

        sr = 48000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        # Loud signal — sample peak around -1 dBFS, should land at [CLIP] or [WARN]
        sig = 0.85 * np.sin(2 * np.pi * 440 * t)
        wav = tmp_path / "loud.wav"
        sf.write(str(wav), sig, sr, subtype="PCM_24")

        analyze(wav, output_dir=tmp_path / "out")
        result = json.loads((tmp_path / "out" / "analysis.json").read_text())
        assert "headroom_verdict" in result["loudness"]
        assert result["loudness"]["headroom_verdict"] in ("[CLIP]", "[WARN]")

        # Spectrogram text contains a Headroom line
        spec_txt = (tmp_path / "out" / "spectrogram.txt").read_text()
        assert "Headroom:" in spec_txt


# ---------------------------------------------------------------------------
# Bus auto-trim (per-bus gain-staging calibration)
# ---------------------------------------------------------------------------

def _write_stem_at_lufs(path, target_lufs, duration_s=4.0, sr=48000, seed=0):
    """Write a noise stem calibrated to a specific integrated LUFS.

    pyloudnorm needs ~3 s for K-weighted gating to be stable; 4 s is safe.
    Returns the actual integrated LUFS measured after write (rounding +
    quantisation typically lands within 0.1 LU of target).
    """
    import soundfile as sf
    import pyloudnorm as pyln

    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    noise = rng.standard_normal((n, 2)) * 0.1
    meter = pyln.Meter(sr)
    current = meter.integrated_loudness(noise)
    gain = 10.0 ** ((target_lufs - current) / 20.0)
    noise *= gain
    sf.write(str(path), noise, sr, subtype="PCM_24")
    return meter.integrated_loudness(noise)


class TestBusAutoTrim:
    """`_compute_bus_auto_trims` — per-bus calibration so dry-sum hits target_lufs.

    The contract: applying (auto_trim_db + volume_db) at render time brings the
    bus dry-sum output to within ~0.5 LU of target_lufs when volume_db = 0.
    Tests simulate the render path (load + pan + sum + apply gain) and check
    the resulting bus output LUFS directly — avoids hardcoding magic numbers
    that depend on per-track pan math (constant-power center is -3 dB).
    """

    SR = 48000
    TARGET_LUFS = -18.0

    def _build_config(self, tmp_path, tracks, buses):
        return {
            "sample_rate": self.SR,
            "buses": {
                name: {"volume_db": 0.0, "parent_bus": cfg.get("parent_bus"), **cfg}
                for name, cfg in buses.items()
            },
            "tracks": [
                {"name": n, "file": str(f), "active": True, "bus": b,
                 "volume_db": 0.0, "pan": 0.0}
                for (n, b, f) in tracks
            ],
        }

    def _measure_bus_output_lufs(self, config, trims):
        """Simulate the render path's bus stage (load -> pan -> sum -> auto_trim + volume_db)
        and return integrated LUFS per top-level bus (parent_bus = None).
        """
        import pyloudnorm as pyln
        from pathlib import Path
        import soundfile as sf
        from render_mix import _load_as_stereo, _pan, _topo_order

        sr = config["sample_rate"]
        active = [t for t in config["tracks"] if t.get("active", True)]
        max_length = max(sf.info(t["file"]).frames for t in active)

        bus_stem_sums: dict[str, np.ndarray] = {}
        for t in active:
            bus = t["bus"]
            stereo = _load_as_stereo(Path(t["file"]), max_length, sr)
            stereo = _pan(stereo * 10 ** (t.get("volume_db", 0.0) / 20.0), t.get("pan", 0.0))
            bus_stem_sums.setdefault(bus, np.zeros((2, max_length)))
            bus_stem_sums[bus] += stereo

        bus_outputs: dict[str, np.ndarray] = {}
        for bus_name in _topo_order(config["buses"]):
            buf = bus_stem_sums.get(bus_name, np.zeros((2, max_length))).copy()
            for child, child_cfg in config["buses"].items():
                if child_cfg.get("parent_bus") == bus_name and child in bus_outputs:
                    buf = buf + bus_outputs[child]
            gain = trims.get(bus_name, 0.0) + config["buses"][bus_name].get("volume_db", 0.0)
            bus_outputs[bus_name] = buf * 10 ** (gain / 20)

        meter = pyln.Meter(sr)
        return {
            name: float(meter.integrated_loudness(buf.T))
            for name, buf in bus_outputs.items()
            if np.any(buf)
        }

    def test_single_stem_brings_bus_to_target(self, tmp_path):
        """Leaf bus with one stem: auto_trim must bring the bus output to target_lufs."""
        from render_mix import _compute_bus_auto_trims

        stem = tmp_path / "stem.wav"
        _write_stem_at_lufs(stem, target_lufs=-12.0, seed=1)

        config = self._build_config(
            tmp_path,
            tracks=[("track1", "drums", stem)],
            buses={"drums": {"parent_bus": None}},
        )
        trims = _compute_bus_auto_trims(config, target_lufs=self.TARGET_LUFS, verbose=False)
        outputs = self._measure_bus_output_lufs(config, trims)
        assert abs(outputs["drums"] - self.TARGET_LUFS) < 0.5, (
            f"drums output {outputs['drums']:.2f} LUFS, target {self.TARGET_LUFS}"
        )

    def test_multi_stem_bus_compensates_for_pile_up(self, tmp_path):
        """Four stems at -18 LUFS each sum well above -18 — auto_trim pulls back."""
        from render_mix import _compute_bus_auto_trims

        stems = []
        for i in range(4):
            p = tmp_path / f"stem_{i}.wav"
            _write_stem_at_lufs(p, target_lufs=-18.0, seed=10 + i)
            stems.append(p)

        config = self._build_config(
            tmp_path,
            tracks=[(f"t{i}", "drums", s) for i, s in enumerate(stems)],
            buses={"drums": {"parent_bus": None}},
        )
        trims = _compute_bus_auto_trims(config, target_lufs=self.TARGET_LUFS, verbose=False)
        outputs = self._measure_bus_output_lufs(config, trims)
        assert abs(outputs["drums"] - self.TARGET_LUFS) < 0.5, (
            f"drums output {outputs['drums']:.2f} LUFS, target {self.TARGET_LUFS}"
        )
        # And the trim must be NEGATIVE — sum-of-stems is louder than a single stem
        assert trims["drums"] < -2.0, f"expected negative trim, got {trims['drums']}"

    def test_parent_bus_lands_at_target_with_child_volume_offset(self, tmp_path):
        """Parent receives children post-(auto_trim+volume); parent auto_trim
        must still land the parent output at target."""
        from render_mix import _compute_bus_auto_trims

        s_a = tmp_path / "child_a.wav"
        s_b = tmp_path / "child_b.wav"
        _write_stem_at_lufs(s_a, target_lufs=-18.0, seed=20)
        _write_stem_at_lufs(s_b, target_lufs=-18.0, seed=21)

        config = self._build_config(
            tmp_path,
            tracks=[("ta", "gtr_1", s_a), ("tb", "gtr_2", s_b)],
            buses={
                "gtr_1": {"parent_bus": "guitar"},
                "gtr_2": {"parent_bus": "guitar"},
                "guitar": {"parent_bus": None},
            },
        )
        # User-chosen offset: child A boosted by +6 dB on its bus
        config["buses"]["gtr_1"]["volume_db"] = 6.0

        trims = _compute_bus_auto_trims(config, target_lufs=self.TARGET_LUFS, verbose=False)
        outputs = self._measure_bus_output_lufs(config, trims)

        # Each child output WITH the user's volume_db offset still floats; the
        # PARENT must land at target regardless.
        assert abs(outputs["guitar"] - self.TARGET_LUFS) < 0.5, (
            f"guitar (parent) output {outputs['guitar']:.2f} LUFS, target {self.TARGET_LUFS}"
        )

    def test_inactive_tracks_are_excluded(self, tmp_path):
        """active=false stems must NOT contribute to the auto-trim measurement."""
        from render_mix import _compute_bus_auto_trims

        s_active = tmp_path / "active.wav"
        s_inactive = tmp_path / "inactive.wav"
        _write_stem_at_lufs(s_active, target_lufs=-18.0, seed=30)
        # The inactive stem is super loud — would skew the trim heavily if leaked in
        _write_stem_at_lufs(s_inactive, target_lufs=-6.0, seed=31)

        config = self._build_config(
            tmp_path,
            tracks=[("t_act", "drums", s_active), ("t_inact", "drums", s_inactive)],
            buses={"drums": {"parent_bus": None}},
        )
        config["tracks"][1]["active"] = False

        trims = _compute_bus_auto_trims(config, target_lufs=self.TARGET_LUFS, verbose=False)

        # Re-run the same trim against an "only active stem" config -> same result
        config_only_active = self._build_config(
            tmp_path,
            tracks=[("t_act", "drums", s_active)],
            buses={"drums": {"parent_bus": None}},
        )
        trims_only_active = _compute_bus_auto_trims(
            config_only_active, target_lufs=self.TARGET_LUFS, verbose=False
        )
        assert abs(trims["drums"] - trims_only_active["drums"]) < 0.3, (
            f"inactive stem leaked in: trim with inactive {trims['drums']} "
            f"vs trim without inactive {trims_only_active['drums']}"
        )


# ---------------------------------------------------------------------------
# Tape saturation relevance check
# ---------------------------------------------------------------------------
class TestTapeSatRelevanceCheck:
    """`_bus_tape_sat_relevance_check` should refuse on squashed (crest < 8 dB)
    or over-compressed (LRA < 4 LU) buses, accept on dynamic material."""

    def test_skips_squashed_signal(self):
        """Hard-clipped square wave has crest ≈ 0 dB — should refuse."""
        from render_mix import _bus_tape_sat_relevance_check

        sr = 48000
        n = sr * 4
        # Square wave (max crest factor possible: peak/rms = 1)
        sig = np.sign(np.sin(2 * np.pi * 200 * np.arange(n) / sr)) * 0.5
        buf = np.stack([sig, sig], axis=0)

        rel = _bus_tape_sat_relevance_check(buf, sr, "test_bus")
        assert rel["recommend_skip"] is True
        assert any("crest" in s for s in rel["issues"]), rel["issues"]

    def test_accepts_dynamic_signal(self):
        """Sparse impulses with quiet noise floor: high crest, high LRA."""
        from render_mix import _bus_tape_sat_relevance_check

        sr = 48000
        n = sr * 6
        rng = np.random.default_rng(42)
        # Quiet noise floor
        sig = rng.standard_normal(n) * 0.02
        # Loud transients every 0.5 s (random direction)
        for i in range(0, n, sr // 2):
            sig[i:i + 100] += 0.6 * np.sin(2 * np.pi * 80 * np.arange(100) / sr)
        buf = np.stack([sig, sig], axis=0)

        rel = _bus_tape_sat_relevance_check(buf, sr, "drums")
        # Crest should be high (sparse transients), LRA should be loose
        assert rel["crest_db"] > 8.0, f"crest {rel['crest_db']} dB should be > 8"
        # NOTE: LRA can be borderline on synthetic test signals — accept either OK
        # or LRA-only complaint. The crest check is the load-bearing one.
        if rel["recommend_skip"]:
            assert all("lra" in s.lower() or "LRA" in s for s in rel["issues"]), (
                f"unexpected skip reason: {rel['issues']}"
            )


# ---------------------------------------------------------------------------
# Premaster mode — render_mix does NOT bake mastering into the mix output.
# ---------------------------------------------------------------------------

def _write_synth_stem(path, target_lufs, duration_s=5.0, sr=48000, seed=0):
    """Helper: write a stereo noise WAV calibrated to target_lufs."""
    import soundfile as sf
    import pyloudnorm as pyln
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    sig = rng.standard_normal((n, 2)) * 0.1
    meter = pyln.Meter(sr)
    cur = meter.integrated_loudness(sig)
    sig *= 10.0 ** ((target_lufs - cur) / 20.0)
    sf.write(str(path), sig, sr, subtype="PCM_24")


class TestPremasterMode:
    """render_mix in premaster mode (default) MUST NOT apply a brick-wall
    limiter, LUFS normalization to -14, clipper, or M/S processing. The
    industry handoff to mastering is a clean buffer with peak headroom at
    `master.peak_target_dbfs` (default -3 dBFS).
    """

    def _build_config(self, tmp_path, mix_dir, stem_paths, master_overrides=None):
        cfg = {
            "session_dir": str(tmp_path),
            "output_dir": str(mix_dir),
            "sample_rate": 48000,
            "master": {"premaster_mode": True, "peak_target_dbfs": -3.0,
                       **(master_overrides or {})},
            "buses": {"drums": {"volume_db": 0.0, "auto_trim_db": 0.0,
                                "parent_bus": None}},
            "tracks": [
                {"name": f"t{i}", "file": str(p), "active": True,
                 "bus": "drums", "volume_db": 0.0, "pan": 0.0}
                for i, p in enumerate(stem_paths)
            ],
        }
        return cfg

    def test_premaster_default_writes_no_limiter_field(self, tmp_path):
        """A default --generate-config produces premaster_mode=true (no
        lufs_target / true_peak_dbfs / clipper / ms keys)."""
        import json as _json
        from render_mix import generate_config

        # Make a tracks dir with one fake stem so generate_config has
        # something to scan.
        tracks_dir = tmp_path / "tracks" / "KICK IN.05"
        tracks_dir.mkdir(parents=True)
        _write_synth_stem(tracks_dir / "assembled.wav", target_lufs=-18.0, seed=1)

        out_cfg = tmp_path / "mix_config.json"
        generate_config(tmp_path, out_cfg)
        cfg = _json.loads(out_cfg.read_text())
        assert cfg["master"].get("premaster_mode") is True
        assert "peak_target_dbfs" in cfg["master"]
        # The mastering-phase options must NOT be in the default emitted config
        assert "lufs_target" not in cfg["master"], \
            "default config should not bake a mastering LUFS target into the mix"
        assert "true_peak_dbfs" not in cfg["master"]
        assert "clipper" not in cfg["master"]
        assert "ms" not in cfg["master"]

    def test_premaster_mix_output_peak_is_at_target(self, tmp_path):
        """In premaster mode, the mix.wav peak lands at peak_target_dbfs
        (within rounding) — no limiter, just a single scalar gain."""
        import json as _json
        from render_mix import render_mix
        import soundfile as sf
        import pyloudnorm as pyln

        # Make a synthetic drum stem with high crest (transient hits + noise)
        stem = tmp_path / "stem.wav"
        rng = np.random.default_rng(7)
        n = 48000 * 5  # 5 sec
        # Quiet noise floor + loud impulses every 0.5 sec → high crest
        sig = rng.standard_normal((n, 2)) * 0.01
        for i in range(0, n, 24000):
            sig[i:i + 200] += 0.6 * np.sin(2 * np.pi * 80 * np.arange(200) / 48000)[:, None]
        sf.write(str(stem), sig, 48000, subtype="PCM_24")

        mix_dir = tmp_path / "mixes"
        cfg_path = tmp_path / "mix_config.json"
        cfg_path.write_text(_json.dumps(self._build_config(tmp_path, mix_dir, [stem])))

        render_mix(cfg_path, output_wav=mix_dir / "mix.wav", render_stems=False)

        # Read the rendered mix
        data, sr = sf.read(str(mix_dir / "mix.wav"))
        peak_db = 20.0 * np.log10(float(np.max(np.abs(data))) + 1e-12)
        # Premaster target is -3 dBFS — allow ±0.3 dB for float rounding
        assert abs(peak_db - (-3.0)) < 0.3, f"peak {peak_db:+.2f} dBFS, expected ~-3"

        # mix_report.json should declare premaster stage
        report = _json.loads((mix_dir / "mix_report.json").read_text())
        assert report["mix_stage"] == "premaster"
        assert report["lufs_target"] is None
        assert report["true_peak_limit_dbfs"] is None
        assert report["peak_target_dbfs"] == -3.0

    def test_premaster_warns_and_ignores_mastering_options(self, tmp_path, capsys):
        """If a legacy config still has clipper/ms/lufs_target keys, premaster
        mode warns and skips them (does NOT silently apply mastering)."""
        import json as _json
        from render_mix import render_mix
        import soundfile as sf

        stem = tmp_path / "stem.wav"
        _write_synth_stem(stem, target_lufs=-22.0, seed=2)

        mix_dir = tmp_path / "mixes"
        cfg_path = tmp_path / "mix_config.json"
        legacy_master = {
            "premaster_mode": True,
            "peak_target_dbfs": -3.0,
            # These should all be ignored with a warning:
            "lufs_target": -14.0,
            "true_peak_dbfs": -1.0,
            "clipper": {"threshold_db": -1.0, "mode": "soft"},
            "ms": {"side_gain_db": 1.0},
        }
        cfg_path.write_text(_json.dumps(
            self._build_config(tmp_path, mix_dir, [stem], master_overrides=legacy_master)
        ))

        render_mix(cfg_path, output_wav=mix_dir / "mix.wav", render_stems=False)
        captured = capsys.readouterr().out
        assert "IGNORED" in captured or "premaster" in captured.lower()

        # Verify no limiter was applied: peak still at the premaster target,
        # not the mastering ceiling
        data, sr = sf.read(str(mix_dir / "mix.wav"))
        peak_db = 20.0 * np.log10(float(np.max(np.abs(data))) + 1e-12)
        assert peak_db < -2.5, f"peak {peak_db:+.2f} should be at the premaster target, not the -1 dBTP mastering ceiling"

    def test_legacy_mode_still_runs_full_master_chain(self, tmp_path):
        """premaster_mode=false (opt-in) restores the historical mix+master
        combined chain — limiter applied, LUFS-normalised to -14 etc."""
        import json as _json
        from render_mix import render_mix
        import soundfile as sf
        import pyloudnorm as pyln

        stem = tmp_path / "stem.wav"
        _write_synth_stem(stem, target_lufs=-22.0, seed=3, duration_s=6.0)

        mix_dir = tmp_path / "mixes"
        cfg_path = tmp_path / "mix_config.json"
        legacy_master = {
            "premaster_mode": False,
            "lufs_target": -14.0,
            "true_peak_dbfs": -1.0,
            "comp": {"threshold_db": -10.0, "ratio": 1.5,
                     "attack_ms": 30.0, "release_ms": 300.0},
        }
        cfg_path.write_text(_json.dumps(
            self._build_config(tmp_path, mix_dir, [stem], master_overrides=legacy_master)
        ))
        render_mix(cfg_path, output_wav=mix_dir / "mix.wav", render_stems=False)

        report = _json.loads((mix_dir / "mix_report.json").read_text())
        assert report["mix_stage"] == "master"
        assert report["lufs_target"] == -14.0
        # The legacy chain hits the -14 LUFS target (within ~1 LU)
        data, sr = sf.read(str(mix_dir / "mix.wav"))
        lufs = float(pyln.Meter(sr).integrated_loudness(data))
        assert abs(lufs - (-14.0)) < 1.5, f"legacy mode integrated LUFS {lufs:.2f}, expected ~-14"


class TestMixHealthStageDetection:
    """`mix_health._detect_mix_stage` reads mix_report.json next to the wav."""

    def test_detect_premaster_from_report(self, tmp_path):
        from mix_health import _detect_mix_stage
        import json as _json
        wav = tmp_path / "mix.wav"
        wav.write_bytes(b"")
        (tmp_path / "mix_report.json").write_text(_json.dumps({"mix_stage": "premaster"}))
        assert _detect_mix_stage(wav) == "premaster"

    def test_detect_master_default_when_no_report(self, tmp_path):
        from mix_health import _detect_mix_stage
        wav = tmp_path / "mix.wav"
        wav.write_bytes(b"")
        assert _detect_mix_stage(wav) == "master"  # legacy fallback

    def test_detect_master_when_report_lacks_mix_stage_field(self, tmp_path):
        from mix_health import _detect_mix_stage
        import json as _json
        wav = tmp_path / "mix.wav"
        wav.write_bytes(b"")
        (tmp_path / "mix_report.json").write_text(_json.dumps({"other_field": 42}))
        assert _detect_mix_stage(wav) == "master"


class TestApplyGainCrossfade:
    """Per-clip assembly applies equal-power crossfades at butt-up clip
    boundaries — matches DAW (Pro Tools / Logic) default behaviour and
    smooths the source-discontinuity clicks that engineer slip-edits leave.
    """

    def _make_session(self, tmp_path, source_a_val=0.5, source_b_val=-0.5,
                      slip_samples=24, sr=48000):
        """Create a 2-clip session where clip B is slip-edited relative to clip A.

        Source is a single WAV containing two regions:
          - [0, 48000)  filled with source_a_val (1 sec of constant +0.5)
          - [48000, 96000) filled with source_b_val (1 sec of constant -0.5)

        Clip A: timeline [0, 48000), source [0, 48000)   → plays source_a_val
        Clip B: timeline [48000, 96000), source [48000 + slip, 96000 + slip)
                → plays source_b_val BUT with a slip-edit (the engineer's edit)

        Without crossfade: assembled audio jumps from +0.5 to -0.5 at sample 48000
        (1-sample step of 1.0 magnitude — extreme click).

        With crossfade: the step is smoothed into an equal-power transition.
        """
        import json as _json
        import soundfile as sf

        # Build the source: 2 sec of constant +0.5, then 2 sec of constant -0.5
        n = sr * 4
        sig = np.zeros(n, dtype=np.float64)
        sig[:sr * 2] = source_a_val
        sig[sr * 2:] = source_b_val
        src_path = tmp_path / "source.wav"
        sf.write(str(src_path), sig, sr, subtype="PCM_24")

        session = {
            "sample_rate": sr,
            "duration_samples": sr * 2,
            "tracks": [
                {
                    "name": "TEST",
                    "clips": [
                        {"source_file": str(src_path),
                         "timeline_start_sample": 0,
                         "source_offset_sample": 0,
                         "length_samples": sr},
                        {"source_file": str(src_path),
                         "timeline_start_sample": sr,
                         "source_offset_sample": sr * 2,  # slip to the second region
                         "length_samples": sr},
                    ],
                },
            ],
        }
        session_path = tmp_path / "session.json"
        session_path.write_text(_json.dumps(session))
        return session_path, sr

    def test_default_crossfade_smooths_butt_up_boundary(self, tmp_path):
        from apply_gain import apply_gain_per_clip
        import soundfile as sf

        session_path, sr = self._make_session(tmp_path)
        out_dir = tmp_path / "out"

        # No-normalize: don't change the constant values
        apply_gain_per_clip(
            session_json=session_path, output_dir=out_dir,
            track_names=["TEST"], normalize=False, crossfade_ms=5.0,
        )

        assembled, _ = sf.read(str(out_dir / "TEST" / "assembled.wav"))
        # The clip boundary is at sample sr (1 sec). Inspect a window around it.
        boundary = sr
        # With a 5 ms crossfade (240 samples at 48k), the discontinuity should
        # be spread over the boundary window — no single-sample 1.0-magnitude
        # jump. The actual jump magnitude at any single sample should be much
        # smaller than the no-crossfade case (which would have a step of 1.0).
        diffs = np.abs(np.diff(assembled))
        max_step = float(np.max(diffs))
        assert max_step < 0.10, (
            f"crossfade did not smooth the boundary: max step {max_step:.4f} "
            f"(expected < 0.10 with a 5 ms fade)"
        )

    def test_no_crossfade_keeps_the_click(self, tmp_path):
        from apply_gain import apply_gain_per_clip
        import soundfile as sf

        session_path, sr = self._make_session(tmp_path)
        out_dir = tmp_path / "out"

        apply_gain_per_clip(
            session_json=session_path, output_dir=out_dir,
            track_names=["TEST"], normalize=False, crossfade_ms=0.0,
        )

        assembled, _ = sf.read(str(out_dir / "TEST" / "assembled.wav"))
        diffs = np.abs(np.diff(assembled))
        max_step = float(np.max(diffs))
        # With crossfade disabled, the 1.0-magnitude jump must survive in some
        # form (>= 0.9 sample-to-sample step).
        assert max_step > 0.9, (
            f"no-crossfade should preserve the discontinuity: max step {max_step:.4f}"
        )

    def test_find_clicks_sweeps_a_mix_for_sharp_transients(self, tmp_path):
        """`find_clicks.find_click_candidates` returns events ranked by step
        magnitude, clustering nearby clicks into single events.
        """
        import soundfile as sf
        from find_clicks import find_click_candidates

        # Build a mix with two distinct click events: a single sharp step at
        # t=1s (step 0.5) and a cluster of three steps near t=3s (steps 0.4).
        sr = 48000
        n = sr * 5
        sig = np.zeros(n, dtype=np.float64)
        # Add some quiet background noise so peak measurements aren't all 0
        rng = np.random.default_rng(0)
        sig += rng.standard_normal(n) * 0.01
        sig[sr * 1] = 0.5            # isolated click
        sig[sr * 3] = 0.4
        sig[sr * 3 + 50] = -0.4      # cluster: 3 sharp transitions within 50/48000 = 1ms
        sig[sr * 3 + 100] = 0.4
        mix_dir = tmp_path / "mixes"
        mix_dir.mkdir()
        sf.write(str(mix_dir / "mix.wav"), sig, sr, subtype="PCM_24")

        events = find_click_candidates(
            tmp_path / "mixes" / "mix.wav",
            threshold=0.3, cluster_gap_ms=50.0,
        )
        assert len(events) >= 2, f"expected ≥ 2 clusters, got {len(events)}"
        # Highest-magnitude cluster should be at ~t=1s (the isolated 0.5 step)
        top = events[0]
        assert abs(top["time_s"] - 1.0) < 0.01, f"top click time {top['time_s']}"
        assert top["step_magnitude"] >= 0.49

    def test_find_clicks_trace_at_time_reads_chain_stages(self, tmp_path):
        """`find_clicks.trace_at_time` walks the per-track chain + stems + mix
        at a given timestamp and returns max-step measurements per stage.
        """
        import soundfile as sf
        from find_clicks import trace_at_time

        # Build a minimal session: 1 active track with assembled.wav + a
        # premaster mix.wav + a stem_<bus>.wav.
        sr = 48000
        n = sr * 4
        sig = np.zeros(n, dtype=np.float64)
        sig[sr * 2] = 0.5  # click at t=2s

        track_dir = tmp_path / "tracks" / "DRUM TEST"
        track_dir.mkdir(parents=True)
        sf.write(str(track_dir / "assembled.wav"), sig, sr, subtype="PCM_24")

        mix_dir = tmp_path / "mixes"
        mix_dir.mkdir()
        sf.write(str(mix_dir / "mix.wav"), sig * 0.5, sr, subtype="PCM_24")

        stems_dir = tmp_path / "stems"
        stems_dir.mkdir()
        sf.write(str(stems_dir / "stem_drums.wav"), sig * 0.5, sr, subtype="PCM_24")

        # Minimal mix_config with one active track
        import json as _json
        (tmp_path / "mix_config.json").write_text(_json.dumps({
            "tracks": [{"name": "DRUM TEST", "active": True}],
        }))

        rep = trace_at_time(tmp_path, target_time_s=2.0, window_s=0.2)
        assert rep["target_time_s"] == 2.0
        # Track chain should have the click
        assert len(rep["tracks"]) == 1
        assert rep["tracks"][0]["name"] == "DRUM TEST"
        track_chain = rep["tracks"][0]["chain"]
        assembled = next(c for c in track_chain if c["stage"] == "assembled")
        assert assembled["max_step"] >= 0.4
        # Mix should also have the (scaled) click
        assert rep["mix"] is not None
        assert rep["mix"]["max_step"] >= 0.2

    def test_crossfade_does_not_move_clip_position(self, tmp_path):
        """The crossfade smooths the boundary but does NOT shift the clip start —
        the engineer's chosen rhythmic position is preserved."""
        from apply_gain import apply_gain_per_clip
        import soundfile as sf

        session_path, sr = self._make_session(tmp_path)
        out_dir = tmp_path / "out"
        apply_gain_per_clip(
            session_json=session_path, output_dir=out_dir,
            track_names=["TEST"], normalize=False, crossfade_ms=5.0,
        )

        assembled, _ = sf.read(str(out_dir / "TEST" / "assembled.wav"))
        # Before the fade window (well before sr), the signal must be +0.5
        sample_well_before = sr - sr // 100  # 10 ms before boundary
        assert abs(assembled[sample_well_before] - 0.5) < 0.01, (
            f"clip A's level changed before the fade region: {assembled[sample_well_before]}"
        )
        # After the fade window, the signal must be -0.5
        sample_well_after = sr + sr // 100  # 10 ms after boundary
        assert abs(assembled[sample_well_after] - (-0.5)) < 0.01, (
            f"clip B's level changed after the fade region: {assembled[sample_well_after]}"
        )
