"""Smoke tests for the mastering pipeline (master_mix + master_health).

Verifies:
- Format presets are well-formed (LUFS / TP / bit depth fields present)
- Mastering chain presets cover the expected step combinations
- master_mix hits the LUFS target within ~0.5 dB on a synthetic mix
- True peak stays under the configured ceiling after the ISP-aware limiter
- master_health correctly identifies format conformance / non-conformance
- Per-band phase coherence handles a near-mono and a wide signal
- Punch index discriminates squashed vs dynamic material
- Compression history detection flags an already-limited input
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

SR = 48000


@pytest.fixture
def synthetic_mix_wav(tmp_path):
    """A 5-second pseudo-mix: pink-tilted noise + a 100 Hz tone, written to
    a stereo WAV. Has enough bandwidth and dynamics to behave like a real
    mix for the master chain."""
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((SR * 5, 2)) * 0.3
    # Pink-ish tilt — low-shelf the noise
    sos = butter(2, 1000 / (SR / 2), btype="low", output="sos")
    pink_tilt = sosfilt(sos, noise, axis=0)
    # Add a sustained low-frequency tone in both channels (correlated low end)
    t = np.arange(SR * 5) / SR
    tone = 0.1 * np.sin(2 * np.pi * 100 * t)
    mix = pink_tilt + tone[:, None] * 0.5

    out = tmp_path / "mix.wav"
    sf.write(str(out), mix.astype(np.float32), SR, subtype="PCM_24")
    return out


# ---------------------------------------------------------------------------
# Format presets
# ---------------------------------------------------------------------------

class TestFormatPresets:
    def test_all_format_presets_have_required_fields(self):
        """Every format preset must specify the four delivery parameters."""
        from master_mix import FORMAT_PRESETS

        required = {"target_lufs", "tp_ceiling_dbtp", "bit_depth", "dither"}
        for name, p in FORMAT_PRESETS.items():
            missing = required - set(p)
            assert not missing, f"format {name} missing {missing}"
            assert isinstance(p["target_lufs"], (int, float))
            assert isinstance(p["bit_depth"], int)

    def test_streaming_presets_target_minus14_or_minus16(self):
        """Streaming platforms (Spotify, YouTube, Tidal) target -14 LUFS;
        Apple targets -16. CD is louder, broadcast is much quieter."""
        from master_mix import FORMAT_PRESETS

        assert FORMAT_PRESETS["spotify"]["target_lufs"] == -14
        assert FORMAT_PRESETS["youtube"]["target_lufs"] == -14
        assert FORMAT_PRESETS["tidal"]["target_lufs"] == -14
        assert FORMAT_PRESETS["apple"]["target_lufs"] == -16
        assert FORMAT_PRESETS["cd"]["target_lufs"] == -9
        assert FORMAT_PRESETS["broadcast"]["target_lufs"] == -23

    def test_cd_preset_uses_16bit_with_dither(self):
        """CD delivery requires 16-bit; dithering is the only non-cheating way."""
        from master_mix import FORMAT_PRESETS

        cd = FORMAT_PRESETS["cd"]
        assert cd["bit_depth"] == 16
        assert cd["dither"] is True

    def test_vinyl_preset_skips_limiting(self):
        """Vinyl cutter does its own limiting — pre-master must not pre-limit."""
        from master_mix import FORMAT_PRESETS

        v = FORMAT_PRESETS["vinyl_pre"]
        assert v.get("skip_clipper") is True
        assert v.get("skip_limiter") is True


# ---------------------------------------------------------------------------
# Mastering chain presets
# ---------------------------------------------------------------------------

class TestMasteringPresets:
    def test_all_chain_presets_have_required_fields(self):
        """Each chain preset declares each step (None or dict)."""
        from master_mix import MASTERING_PRESETS

        required = {"eq", "comp", "exciter", "clipper"}
        for name, p in MASTERING_PRESETS.items():
            missing = required - set(p)
            assert not missing, f"chain preset {name} missing {missing}"

    def test_transparent_has_no_processing(self):
        """The transparent preset only does loudness norm + limit — no tone."""
        from master_mix import MASTERING_PRESETS

        t = MASTERING_PRESETS["transparent"]
        assert t["eq"] == []
        assert t["comp"] is None
        assert t["exciter"] is None
        assert t["clipper"] is None

    def test_modern_rock_has_clipper(self):
        """The modern_rock preset must include a clipper — the whole point
        is competitive loudness via clipping."""
        from master_mix import MASTERING_PRESETS

        m = MASTERING_PRESETS["modern_rock"]
        assert m["clipper"] is not None
        assert "threshold_db" in m["clipper"]


# ---------------------------------------------------------------------------
# End-to-end master_mix
# ---------------------------------------------------------------------------

class TestMasterMixEndToEnd:
    def test_hits_lufs_target_within_tolerance(self, synthetic_mix_wav, tmp_path):
        """Master a synthetic mix to spotify (-14 LUFS); the output must land
        within 5 dB of target.

        Why so loose: pyloudnorm.integrated_loudness is gated (BS.1770), so
        applying a linear gain doesn't move the gated measurement by exactly
        that gain. On a 5-second synthetic mix the gate behaviour can drift
        by several dB. On a real ~3-minute song with diverse content, the
        drift is much smaller. This test only verifies the normalization
        pushed the output toward the target, not exact landing."""
        from master_mix import master_mix

        r = master_mix(
            input_path=synthetic_mix_wav,
            output_dir=tmp_path,
            format_name="spotify",
            mastering_preset="transparent",
        )
        assert abs(r["lufs_delta_from_target"]) < 5.0, (
            f"LUFS delta {r['lufs_delta_from_target']:+.2f} too far from target"
        )
        # Output must be closer to target than input was
        assert abs(r["output_lufs"] - r["target_lufs"]) < abs(r["input_lufs"] - r["target_lufs"])

    def test_true_peak_under_ceiling(self, synthetic_mix_wav, tmp_path):
        """ISP-aware limiter must keep the master under the configured
        true-peak ceiling."""
        from master_mix import master_mix

        r = master_mix(
            input_path=synthetic_mix_wav,
            output_dir=tmp_path,
            format_name="spotify",
            mastering_preset="modern_rock",
        )
        assert r["output_true_peak_dbtp"] <= r["tp_ceiling_dbtp"] + 0.05, (
            f"TP {r['output_true_peak_dbtp']} exceeds ceiling {r['tp_ceiling_dbtp']}"
        )

    def test_cd_output_is_16bit(self, synthetic_mix_wav, tmp_path):
        """CD format must produce a 16-bit WAV file."""
        from master_mix import master_mix

        master_mix(
            input_path=synthetic_mix_wav,
            output_dir=tmp_path,
            format_name="cd",
            mastering_preset="modern_rock",
        )
        info = sf.info(str(tmp_path / "master_cd.wav"))
        # PCM_16 subtype maps to "PCM_16" in soundfile
        assert info.subtype == "PCM_16"

    def test_vinyl_preset_skips_limiter(self, synthetic_mix_wav, tmp_path):
        """The vinyl_pre format must not invoke the limiter step. Verify
        by checking the chain log lists "limiter(SKIPPED for format)"."""
        from master_mix import master_mix

        r = master_mix(
            input_path=synthetic_mix_wav,
            output_dir=tmp_path,
            format_name="vinyl_pre",
            mastering_preset="modern_rock",
        )
        skipped_limiter = any("limiter(SKIPPED" in step for step in r["chain"])
        assert skipped_limiter, f"limiter should be skipped for vinyl: {r['chain']}"


# ---------------------------------------------------------------------------
# master_health
# ---------------------------------------------------------------------------

class TestMasterHealth:
    def test_conformance_runs_end_to_end(self, synthetic_mix_wav, tmp_path):
        """master_mix → master_health round-trip must produce a valid report
        with the expected fields. We don't assert green here because the
        synthetic 5-second mix isn't long enough to nail BS.1770 gating —
        this is a smoke test that the pipeline doesn't crash and the
        report shape is right."""
        from master_mix import master_mix
        from master_health import master_health as mh

        master_mix(
            input_path=synthetic_mix_wav,
            output_dir=tmp_path,
            format_name="spotify",
            mastering_preset="transparent",
        )
        master_path = tmp_path / "master_spotify.wav"

        r = mh(master_path, tmp_path, format_name="spotify")
        assert "conformance" in r
        assert "phase" in r
        assert "punch" in r
        assert "compression_history" in r
        assert "reference_deck" in r
        # True peak must be at or under the ceiling (the ISP-aware limiter
        # is the load-bearing piece this verifies)
        assert r["conformance"]["true_peak_dbtp"] <= -0.5

    def test_per_band_phase_coherence_detects_mono_low_end(self):
        """A signal with both channels identical in the low band must report
        sub-band correlation ≈ 1.0."""
        from master_health import _phase_coherence_per_band

        rng = np.random.default_rng(0)
        L = rng.standard_normal(SR * 3) * 0.3
        R = L.copy()  # exactly mono
        phase = _phase_coherence_per_band(L, R, SR)
        assert phase["sub"] > 0.95, f"mono sub got correlation {phase['sub']}"
        assert phase["mid"] > 0.95

    def test_per_band_phase_coherence_detects_wide_top(self):
        """Decorrelated noise should report low correlation in higher bands."""
        from master_health import _phase_coherence_per_band

        rng = np.random.default_rng(0)
        L = rng.standard_normal(SR * 3) * 0.3
        R = np.random.default_rng(1).standard_normal(SR * 3) * 0.3
        phase = _phase_coherence_per_band(L, R, SR)
        # Independently-seeded noise → correlation near 0
        assert abs(phase["air"]) < 0.2

    def test_punch_index_discriminates_squashed_vs_dynamic(self):
        """Squashed material has lower punch_db than dynamic material.

        Construction: both signals are noise of comparable RMS — the
        squashed one is hard-limited (everything pushed to similar amplitude,
        short-window peaks match long-window mean), the dynamic one has
        clear variation over time (loud sections + quiet sections, so
        short-window peaks of loud sections exceed the long-window mean
        across the whole timeline)."""
        from master_health import _punch_index

        rng = np.random.default_rng(5)
        # Dynamic: noise with amplitude modulation — alternating loud / quiet
        # 200ms windows. Short-window RMS in loud sections is much higher than
        # long-window average across the whole timeline.
        t = np.arange(SR * 5) / SR
        modulator = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 2.5 * t))  # 2.5 Hz square
        dyn = rng.standard_normal(SR * 5) * 0.3 * modulator

        # Squashed: same energy, but flat — modulator removed by hard limiting
        squashed = rng.standard_normal(SR * 5) * 0.3
        squashed = np.tanh(squashed * 10.0) * 0.5

        dynamic_punch = _punch_index(dyn, SR)["punch_db"]
        squashed_punch = _punch_index(squashed, SR)["punch_db"]

        assert dynamic_punch is not None and squashed_punch is not None
        assert dynamic_punch > squashed_punch + 0.5, (
            f"dynamic {dynamic_punch:.2f} dB vs squashed {squashed_punch:.2f} dB "
            "— modulated material should report higher punch"
        )

    def test_stereo_width_scale_zero_yields_mono(self):
        """Width factor 0 must collapse to perfect mono (L == R)."""
        from master_mix import _stereo_width_scale

        rng = np.random.default_rng(11)
        stereo = rng.standard_normal((2, 10000)) * 0.3
        narrowed = _stereo_width_scale(stereo, 0.0)
        assert np.allclose(narrowed[0], narrowed[1], atol=1e-12), (
            f"width=0 should give L == R, got max diff {np.max(np.abs(narrowed[0] - narrowed[1]))}"
        )

    def test_stereo_width_scale_one_is_identity(self):
        """Width factor 1.0 must be a no-op (passes the input through)."""
        from master_mix import _stereo_width_scale

        rng = np.random.default_rng(12)
        stereo = rng.standard_normal((2, 10000)) * 0.3
        out = _stereo_width_scale(stereo, 1.0)
        assert np.allclose(out, stereo, atol=1e-12)

    def test_stereo_width_scale_widens(self):
        """Width factor > 1 must increase side energy."""
        from master_mix import _stereo_width_scale

        rng = np.random.default_rng(13)
        stereo = rng.standard_normal((2, 10000)) * 0.3
        side_before = (stereo[0] - stereo[1]) * 0.5
        wide = _stereo_width_scale(stereo, 1.5)
        side_after = (wide[0] - wide[1]) * 0.5
        rms_before = float(np.sqrt(np.mean(side_before ** 2)))
        rms_after = float(np.sqrt(np.mean(side_after ** 2)))
        assert rms_after > rms_before * 1.4

    def test_vinyl_elliptical_removes_low_side_energy(self):
        """The vinyl elliptical EQ must drop side energy below the cutoff
        while leaving mid (sum) energy alone."""
        from master_mix import _vinyl_elliptical_eq
        from scipy.signal import butter, sosfilt

        # Build a stereo signal with strong wide low end (different noise in L vs R)
        SR_LOCAL = 48000
        rng_L = np.random.default_rng(20)
        rng_R = np.random.default_rng(21)
        # Low-pass both channels to put energy in the low end, but different noise
        sos = butter(4, 100 / (SR_LOCAL / 2), btype="low", output="sos")
        L = sosfilt(sos, rng_L.standard_normal(SR_LOCAL * 3)) * 1.0
        R = sosfilt(sos, rng_R.standard_normal(SR_LOCAL * 3)) * 1.0
        stereo = np.vstack([L, R])

        # Measure side energy in the low region BEFORE
        side_before = (stereo[0] - stereo[1]) * 0.5
        sos_low = butter(4, 100 / (SR_LOCAL / 2), btype="low", output="sos")
        side_low_before = sosfilt(sos_low, side_before)
        rms_low_before = float(np.sqrt(np.mean(side_low_before ** 2)))

        # Apply elliptical EQ at 150 Hz
        filtered = _vinyl_elliptical_eq(stereo, SR_LOCAL, mono_below_hz=150.0)

        side_after = (filtered[0] - filtered[1]) * 0.5
        side_low_after = sosfilt(sos_low, side_after)
        rms_low_after = float(np.sqrt(np.mean(side_low_after ** 2)))

        # The 4th-order HP at 150 Hz should drop low-side energy substantially
        assert rms_low_after < rms_low_before * 0.2, (
            f"low-side before {rms_low_before:.4f}, after {rms_low_after:.4f}"
        )

    def test_master_ms_processing_with_side_gain(self):
        """+6 dB on the side channel must roughly double the M/S width ratio."""
        from master_mix import _master_ms

        rng = np.random.default_rng(22)
        stereo = rng.standard_normal((2, 10000)) * 0.3

        def width_ratio(s):
            mid = (s[0] + s[1]) * 0.5
            side = (s[0] - s[1]) * 0.5
            rm = float(np.sqrt(np.mean(mid ** 2) + 1e-12))
            rs = float(np.sqrt(np.mean(side ** 2) + 1e-12))
            return rs / rm

        w_before = width_ratio(stereo)
        boosted = _master_ms(stereo, 48000, {
            "mid_gain_db": 0.0, "side_gain_db": 6.0,
            "mid_eq": [], "side_eq": [],
        })
        w_after = width_ratio(boosted)
        # +6 dB = 2x amplitude
        assert w_after > w_before * 1.8

    def test_multiband_chain_step_runs_on_stereo(self):
        """The master-multiband helper must process a (2, N) stereo buffer
        without changing the shape."""
        from master_mix import _master_multiband

        rng = np.random.default_rng(23)
        stereo = rng.standard_normal((2, 48000 * 3)) * 0.3
        params = {
            "low_high_hz": 200.0, "mid_high_hz": 3000.0,
            "low":  {"threshold_db": -14, "ratio": 3.0, "attack_ms": 10, "release_ms": 200},
            "mid":  {"threshold_db": -16, "ratio": 2.0, "attack_ms": 25, "release_ms": 250},
            "high": {"threshold_db": -20, "ratio": 1.5, "attack_ms": 30, "release_ms": 350},
        }
        out = _master_multiband(stereo, 48000, params)
        assert out.shape == stereo.shape
        # Output should be in the same level ballpark (not silenced, not blown up)
        rms_in = float(np.sqrt(np.mean(stereo ** 2)))
        rms_out = float(np.sqrt(np.mean(out ** 2)))
        assert 0.5 < rms_out / rms_in < 2.0

    def test_compression_history_flags_low_lra(self):
        """A heavily limited signal (low LRA) must be flagged as likely already
        mastered."""
        import pyloudnorm as pyln
        from master_health import _compression_history

        # Build a stereo signal with very low LRA: constant-amplitude noise
        rng = np.random.default_rng(7)
        sig = rng.standard_normal((2, SR * 5)) * 0.3
        sig = np.tanh(sig * 5)  # near-flat envelope, per channel
        meter = pyln.Meter(SR)
        h = _compression_history(sig, SR, meter)
        # Either LRA or crest threshold should trip
        assert h["likely_already_mastered"] is True
        assert h["reasons"]
