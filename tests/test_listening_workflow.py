"""Prevent technical success and narrow feedback from becoming song approval."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from _recall import content_hash
from prepare_audition import prepare_audition
from review_delivery import listening_status, review_delivery

SR = 48000


@pytest.fixture
def audio(tmp_path):
    t = np.arange(SR * 2) / SR
    x = 0.15 * np.sin(2 * np.pi * 330 * t)
    path = tmp_path / "mix.wav"
    sf.write(path, np.column_stack((x, x * 0.6)), SR, subtype="PCM_24")
    return path


def feedback(path, scope="full_song", decision="approved"):
    return {"artifact_hash": content_hash(path), "reviewer": "Test listener",
            "source": "test-message-1", "quote": "Fixture feedback for the specified scope",
            "scope": scope, "decision": decision}


def test_technical_success_cannot_approve_listening(audio, tmp_path):
    result = review_delivery(audio, tmp_path / "review", tp_ceiling=-1)
    assert result["technical"]["status"] == "passed"
    assert result["status"] == "listening_pending"
    assert not result["delivery_ready"]


@pytest.mark.parametrize("scope", ["vocal_balance", "section", "codec_roundtrip"])
def test_narrow_feedback_does_not_approve_full_song(audio, scope):
    result = listening_status(content_hash(audio), [feedback(audio, scope)])
    assert result["status"] == "pending"
    assert "full_song" in result["missing_scopes"]


def test_current_full_song_approval_with_checks_allows_delivery(audio, tmp_path):
    result = review_delivery(audio, tmp_path / "review", [feedback(audio)],
                             tp_ceiling=-1, sample_rate=SR, bit_depth=24)
    assert result["delivery_ready"]
    assert result["status"] == "ready_for_delivery"


def test_changed_audio_invalidates_prior_approval(audio, tmp_path):
    approval = feedback(audio)
    data, sr = sf.read(audio)
    sf.write(audio, data * 0.8, sr, subtype="PCM_24")
    result = review_delivery(audio, tmp_path / "review", [approval], tp_ceiling=-1)
    assert not result["delivery_ready"]
    assert result["listening"]["stale_records_ignored"] == 1


def test_revision_request_overrides_approval_and_narrow_followup(audio):
    records = [feedback(audio), feedback(audio, decision="revision_requested"),
               feedback(audio, scope="vocal_balance")]
    assert listening_status(content_hash(audio), records)["status"] == "revision_requested"


def test_missing_specification_cannot_be_called_passed(audio, tmp_path):
    result = review_delivery(audio, tmp_path / "review", [feedback(audio)])
    assert result["status"] == "technical_incomplete"
    assert not result["delivery_ready"]


def test_approval_does_not_override_failed_export_requirement(audio, tmp_path):
    result = review_delivery(audio, tmp_path / "review", [feedback(audio)],
                             tp_ceiling=-1, sample_rate=44100)
    assert result["status"] == "technical_failed"
    assert not result["delivery_ready"]


def test_codec_review_is_separate_from_wav_approval(audio):
    records = [feedback(audio)]
    result = listening_status(content_hash(audio), records, require_codec_review=True)
    assert result["missing_scopes"] == ["codec_roundtrip"]
    records.append(feedback(audio, "codec_roundtrip"))
    assert listening_status(content_hash(audio), records, True)["status"] == "approved"


def test_missing_feedback_provenance_is_rejected(audio):
    record = feedback(audio)
    del record["source"]
    with pytest.raises(ValueError, match="source"):
        listening_status(content_hash(audio), [record])


def test_audition_matches_loudness_preserves_shape_and_source(audio, tmp_path):
    original_hash = content_hash(audio)
    data, sr = sf.read(audio)
    after = tmp_path / "quieter.wav"
    sf.write(after, data * 0.05, sr, subtype="FLOAT")
    report = prepare_audition(audio, after, tmp_path / "ab", start=0.5,
                              duration=1, target_lufs=-3, tp_ceiling=-3)
    assert abs(report["lufs_difference"]) < 0.01
    assert report["common_trim_db"] < 0
    for entry in report["excerpts"]:
        assert entry["output_true_peak_dbtp"] <= -3
        actual, _ = sf.read(entry["output"], always_2d=True)
        source, _ = sf.read(entry["source"], start=entry["start_frame"], frames=entry["frames"], always_2d=True)
        np.testing.assert_allclose(actual, source * 10 ** (entry["gain_db"] / 20), atol=2e-7)
    assert content_hash(audio) == original_hash
    with pytest.raises(FileExistsError):
        prepare_audition(audio, after, tmp_path / "ab", duration=1)


@pytest.mark.parametrize("settings", [{"start": -1}, {"duration": 3}, {"after_start": float("nan")}])
def test_invalid_audition_windows_fail_before_export(audio, tmp_path, settings):
    with pytest.raises(ValueError):
        prepare_audition(audio, audio, tmp_path / "ab", **settings)
    assert not (tmp_path / "ab").exists()


def test_silent_audition_cannot_be_loudness_matched(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros((SR, 2)), SR)
    with pytest.raises(ValueError, match="silent"):
        prepare_audition(path, path, tmp_path / "ab", duration=1)


def test_master_dynamics_warning_does_not_claim_audible_damage():
    from master_mix import _dynamics_review
    result = _dynamics_review(18, 10, 10.7, -5, 3)
    assert len(result["warnings"]) == 4
    assert "not quality requirements" in result["threshold_policy"]
    assert _dynamics_review(12, 12, 0, 0, 0)["warnings"] == []


def test_master_report_keeps_listening_pending(audio, tmp_path):
    from master_mix import master_mix
    result = master_mix(audio, tmp_path / "master", "spotify", mastering_preset="transparent")
    assert not result["delivery_ready"]
    assert result["listening_review"]["status"] == "pending"
    assert "dynamics_review" in result
    assert not any("GR=" in stage for stage in result["chain"])


def test_missing_format_in_health_report_is_not_a_pass(audio, tmp_path):
    from master_health import master_health
    result = master_health(audio, tmp_path / "health")
    text = (tmp_path / "health/master_health_generic.txt").read_text()
    assert not result["delivery_ready"]
    assert "TECHNICAL CONFORMANCE: [!]" in text
    assert "Measured technical checks passed" not in text


def test_red_style_similarity_is_not_a_failed_command(audio, tmp_path, monkeypatch):
    import style_check
    monkeypatch.setattr(sys, "argv", ["style_check.py", str(audio), "--style", "modern_rock", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as result:
        style_check.main()
    assert result.value.code == 0
    import json
    report = json.loads((tmp_path / "style_check.json").read_text())
    assert report["verdict"] == "RED"
    assert not report["delivery_ready"]
