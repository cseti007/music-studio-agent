"""Regression tests for the recall, assembly and session-parsing review findings."""

import gzip
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

SR = 48000


def _noise(path, seed, frames=SR, channels=2, scale=.02):
    shape = (frames, channels) if channels > 1 else frames
    sf.write(path, np.random.default_rng(seed).normal(0, scale, shape), SR, subtype="FLOAT")
    return path


def _eq_chain(tmp_path):
    from apply_eq import apply_eq
    from build_chain import build_stem_chain
    source = _noise(tmp_path / "source.wav", 1)
    report = apply_eq(source, tmp_path, [{"type": "highpass", "hz": 60}])
    steps = build_stem_chain(tmp_path)
    chain = tmp_path / "chain.json"
    chain.write_text(json.dumps({"stems": [{"name": "Track", "chain": steps}]}))
    return Path(report["output"]), chain, steps


# --- Finding 1: a differing replay must not destroy the recorded evidence ---

def test_mismatching_replay_keeps_original_output_and_operation(tmp_path):
    from _recall import content_hash
    from replay_chain import replay
    output, chain, steps = _eq_chain(tmp_path)
    operation = output.with_suffix(".operation.json")
    # Simulate audio recorded by an older engine: different content, matching hash.
    _noise(output, 99)
    steps[0]["output_hash"] = content_hash(output)
    chain.write_text(json.dumps({"stems": [{"name": "Track", "chain": steps}]}))
    original_audio, original_operation = output.read_bytes(), operation.read_bytes()
    assert replay(chain, False, "Track") != 0
    assert output.read_bytes() == original_audio
    assert operation.read_bytes() == original_operation
    candidates = list(tmp_path.glob(".replay-*/*.wav"))
    assert len(candidates) == 1 and content_hash(candidates[0]) != steps[0]["output_hash"]


def test_matching_replay_installs_output_without_rewriting_operation(tmp_path):
    from replay_chain import replay
    output, chain, _ = _eq_chain(tmp_path)
    operation = output.with_suffix(".operation.json")
    expected, recorded = sf.read(output)[0], operation.read_bytes()
    output.unlink()
    assert replay(chain, False, "Track") == 0
    assert np.array_equal(sf.read(output)[0], expected)
    assert operation.read_bytes() == recorded
    assert not list(tmp_path.glob(".replay-*"))


# --- Finding 2: engine hashes are separate from data dependencies ---

def test_operation_separates_engine_from_data_dependencies(tmp_path):
    output, _, _ = _eq_chain(tmp_path)
    operation = json.loads(output.with_suffix(".operation.json").read_text())
    assert operation["schema_version"] == 2
    assert str(TOOLS_DIR / "apply_eq.py") in operation["engine"]
    assert not any(path.endswith(".py") for path in operation["dependencies"])
    assert str((tmp_path / "source.wav").resolve()) in operation["dependencies"]


@pytest.mark.parametrize("schema", [1, 2])
def test_engine_change_warns_but_data_change_fails(tmp_path, capsys, schema):
    from replay_chain import replay
    _, chain, steps = _eq_chain(tmp_path)
    engine = str(TOOLS_DIR / "render_mix.py")
    if schema == 1:
        steps[0]["schema_version"] = 1
        steps[0]["dependencies"].update(steps[0].pop("engine"))
        steps[0]["dependencies"][engine] = "stale"
    else:
        steps[0]["engine"][engine] = "stale"
    chain.write_text(json.dumps({"stems": [{"name": "Track", "chain": steps}]}))
    assert replay(chain, True, "Track") == 0
    assert "render_mix.py" in capsys.readouterr().err
    _noise(tmp_path / "source.wav", 2)
    assert replay(chain, True, "Track") != 0


# --- Finding 3: legacy reports do not crash build_chain ---

def test_unrecorded_report_marks_chain_unverified(tmp_path):
    from build_chain import build_chain
    track = tmp_path / "tracks" / "Vocal"
    track.mkdir(parents=True)
    (track / "deesser_report.json").write_text(json.dumps(
        {"input": str(track / "a.wav"), "output": str(track / "a_deessed.wav")}))
    chain = build_chain(tmp_path)
    assert chain["verified_operations"] is False
    assert any("deesser_report.json" in warning for warning in chain["warnings"])


def test_build_chain_cli_reports_invalid_json_cleanly(tmp_path):
    track = tmp_path / "tracks" / "Vocal"
    track.mkdir(parents=True)
    (track / "deesser_report.json").write_text("{broken")
    proc = subprocess.run([sys.executable, str(TOOLS_DIR / "build_chain.py"), str(tmp_path),
                           "--output", str(tmp_path / "chain.json")], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "Traceback" not in proc.stderr and "deesser_report.json" in proc.stderr


# --- Findings 4, 5, 9: config, per-track dependencies and assembly defaults ---

def _session(tmp_path, tracks):
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"sample_rate": SR, "duration_samples": 2 * SR, "tracks": tracks}))
    return session


def _clip(source, start=0, offset=0, length=SR):
    return {"source_file": str(source), "timeline_start_sample": start,
            "source_offset_sample": offset, "length_samples": length}


def test_clip_target_comes_from_project_config_and_is_recorded(tmp_path, monkeypatch):
    import tomllib
    from apply_gain import apply_gain_per_clip
    source = _noise(tmp_path / "a.wav", 3, 2 * SR)
    session = _session(tmp_path, [{"name": "A", "clips": [_clip(source)]}])
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    (cwd / "config.toml").write_text("[gain]\nper_clip_target_lufs = -30.0\n")
    monkeypatch.chdir(cwd)
    report = apply_gain_per_clip(session, tmp_path / "tracks", track_names=["A"])[0]
    kwargs = json.loads(Path(report["output"]).with_suffix(".operation.json").read_text())["args"]["kwargs"]
    project = tomllib.loads((TOOLS_DIR.parent / "config.toml").read_text())
    assert kwargs["target_lufs"] == project["gain"]["per_clip_target_lufs"]
    assert kwargs["crossfade_ms"] == 0.0


def test_listing_tracks_hashes_nothing_and_dependencies_are_per_track(tmp_path, monkeypatch):
    import _recall
    from apply_gain import apply_gain_per_clip
    a, b = _noise(tmp_path / "a.wav", 4, 2 * SR), _noise(tmp_path / "b.wav", 5, 2 * SR)
    session = _session(tmp_path, [{"name": "A", "clips": [_clip(a)]}, {"name": "B", "clips": [_clip(b)]}])
    hashed = []
    original = _recall.content_hash
    monkeypatch.setattr(_recall, "content_hash", lambda path: hashed.append(str(path)) or original(path))
    assert apply_gain_per_clip(session, tmp_path / "tracks") == []
    assert not any(path.endswith(".wav") for path in hashed)
    reports = apply_gain_per_clip(session, tmp_path / "tracks", all_tracks=True)
    for report, own, other in zip(reports, (a, b), (b, a)):
        deps = json.loads(Path(report["output"]).with_suffix(".operation.json").read_text())["dependencies"]
        assert str(own.resolve()) in deps and str(other.resolve()) not in deps


def test_continuous_mode_defaults_to_50ms_crossfade(tmp_path):
    from apply_gain import apply_gain_per_clip
    source = _noise(tmp_path / "a.wav", 6, 2 * SR)
    session = _session(tmp_path, [{"name": "A", "clips": [_clip(source)]}])
    report = apply_gain_per_clip(session, tmp_path / "tracks", track_names=["A"], source_mode="continuous")[0]
    assert report["default_crossfade_ms"] == 50.0


def test_no_normalize_flag_is_documented_as_deprecated():
    proc = subprocess.run([sys.executable, str(TOOLS_DIR / "apply_gain.py"), "--help"],
                          capture_output=True, text=True, check=True)
    assert "deprecated" in " ".join(proc.stdout.split()).lower()


# --- Finding 10: same-take crossfades must not boost correlated audio ---

def test_contiguous_same_take_crossfade_preserves_level(tmp_path):
    from apply_gain import apply_gain_per_clip
    source = tmp_path / "take.wav"
    sf.write(source, np.full(3 * SR, .5), SR, subtype="FLOAT")
    session = _session(tmp_path, [{"name": "A", "clips": [_clip(source), _clip(source, SR, SR)]}])
    report = apply_gain_per_clip(session, tmp_path / "tracks", track_names=["A"], crossfade_ms=5)[0]
    audio = sf.read(report["output"])[0]
    assert np.max(np.abs(audio[:2 * SR] - .5)) < 1e-6


# --- Finding 6: legacy steps are rejected before mix validation ---

def test_legacy_chain_is_rejected_before_mix_config_is_opened(tmp_path, capsys):
    from replay_chain import replay
    config = tmp_path / "mix_config.json"
    config.write_text(json.dumps({"sample_rate": SR, "master": {}, "buses": {},
        "tracks": [{"name": "Track", "file": str(tmp_path / "missing.wav")}]}))
    chain = tmp_path / "chain.json"
    chain.write_text(json.dumps({"mix_config": str(config), "stems": [{"name": "Track", "chain": [
        {"step": "eq", "input": str(tmp_path / "a.wav"), "output": str(tmp_path / "a_eq.wav"),
         "args": {"filters": []}}]}]}))
    assert replay(chain, True, None) != 0
    assert "Legacy reports lack" in capsys.readouterr().err
    assert replay(chain, True, None, allow_legacy=True) != 0
    assert "missing.wav" in capsys.readouterr().err


# --- Finding 7: alignment threshold is configurable and reported cleanly ---

def test_alignment_threshold_is_configurable(tmp_path, monkeypatch, capsys):
    import align_phase
    ref = _noise(tmp_path / "ref.wav", 7)
    (tmp_path / "Target").mkdir()
    tgt = _noise(tmp_path / "Target" / "assembled.wav", 8)
    with pytest.raises(ValueError, match="min-correlation"):
        align_phase.align_phase(ref, tgt, tmp_path / "out")
    assert align_phase.align_phase(ref, tgt, tmp_path / "out", min_correlation=0.0)["output"]
    monkeypatch.setattr(sys, "argv", ["align_phase.py", "--reference", str(ref), "--target", str(tgt),
                                      "--output-dir", str(tmp_path / "cli")])
    with pytest.raises(SystemExit) as exc:
        align_phase.main()
    assert exc.value.code == 1 and "min-correlation" in capsys.readouterr().err


# --- Finding 8: Pro Tools IDs and Ableton tempo/warp handling ---

def _protools(tmp_path, monkeypatch, lines):
    import parse_session
    parser = tmp_path / "ptftool"
    parser.touch()
    monkeypatch.setattr(parse_session, "PTFTOOL", parser)
    monkeypatch.setattr(parse_session.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=0, stdout="\n".join(lines), stderr=""))
    return parse_session._parse_protools(tmp_path / "song.ptx", tmp_path)


def test_protools_track_id_is_not_read_from_track_name(tmp_path, monkeypatch):
    sf.write(tmp_path / "stereo.wav", np.zeros((SR, 2)), SR)
    session = _protools(tmp_path, monkeypatch,
                        [f"`Kick Out(2)` t({i}) (stereo.wav) @ 0 + 0, {SR}" for i in (5, 6)])
    assert len(session["tracks"][0]["clips"]) == 1


def test_protools_unresolved_stereo_source_is_reported_as_unresolved(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="unresolved"):
        _protools(tmp_path, monkeypatch, [f"`Vox` t({i}) (gone.wav) @ 0 + 0, {SR}" for i in (1, 2)])


_ALS = """<?xml version="1.0" encoding="UTF-8"?>
<Ableton><LiveSet>
  <MasterTrack>
    <AutomationEnvelopes><Envelopes>{envelope}</Envelopes></AutomationEnvelopes>
    <DeviceChain><Mixer><Tempo><Manual Value="120" /><AutomationTarget Id="8" /></Tempo></Mixer></DeviceChain>
  </MasterTrack>
  <Tracks><AudioTrack><Name><EffectiveName Value="BASS" /></Name><DeviceChain><MainSequencer>
    <ClipTimeable><ArrangerAutomation><Events>
      <AudioClip Time="4"><StartRelative Value="0" /><OutMarker Value="8" /><IsWarped Value="{warped}" />
        <WarpMarkers><WarpMarker Id="0" SecTime="0" BeatTime="0" /><WarpMarker Id="1" SecTime="{sec}" BeatTime="8" /></WarpMarkers>
        <SampleRef><FileRef><Path Value="/audio/bass.wav" /></FileRef><SampleRate Value="48000" /></SampleRef>
      </AudioClip>
    </Events></ArrangerAutomation></ClipTimeable>
  </MainSequencer></DeviceChain></AudioTrack></Tracks>
</LiveSet></Ableton>"""

_TEMPO_ENVELOPE = """<AutomationEnvelope><EnvelopeTarget><PointeeId Value="8" /></EnvelopeTarget>
  <Automation><Events><FloatEvent Time="-63072000" Value="120" /><FloatEvent Time="16" Value="140" /></Events></Automation>
</AutomationEnvelope>"""


@pytest.mark.parametrize("envelope,warped,sec,error", [
    ("", "false", 4, None),
    ("", "true", 4, None),            # warped at the project tempo: timing unchanged
    ("", "true", 5, "warp"),          # warping stretches the audio
    (_TEMPO_ENVELOPE, "false", 4, "Tempo automation"),
])
def test_ableton_tempo_and_warp_checks(tmp_path, envelope, warped, sec, error):
    from parse_session import _parse_ableton
    path = tmp_path / "set.als"
    with gzip.open(path, "wb") as stream:
        stream.write(_ALS.format(envelope=envelope, warped=warped, sec=sec).encode())
    if error:
        with pytest.raises(ValueError, match=error):
            _parse_ableton(path)
    else:
        assert _parse_ableton(path)["tracks"][0]["clips"][0]["length_samples"] == 4 * SR


# --- Finding 11: 32-bit float deliveries ---

def test_bit_depth_32_accepts_float_delivery(tmp_path):
    from review_delivery import review_delivery
    path = tmp_path / "mix.wav"
    sf.write(path, np.random.default_rng(9).normal(0, .05, (SR, 2)), SR, subtype="FLOAT")
    report = review_delivery(path, tmp_path / "review", tp_ceiling=-1, bit_depth=32)
    assert report["technical"]["checks"]["bit_depth"] is True


# --- Finding 12: str paths and NaN arguments ---

def test_string_paths_are_hashed_and_nan_is_named(tmp_path):
    from apply_eq import apply_eq
    source = _noise(tmp_path / "source.wav", 10)
    report = apply_eq(str(source), str(tmp_path), [{"type": "highpass", "hz": 60}])
    operation = json.loads(Path(report["output"]).with_suffix(".operation.json").read_text())
    assert str(source.resolve()) in operation["dependencies"]
    assert "input_path" in operation["args"]["path_keys"]
    with pytest.raises(ValueError, match="filters"):
        apply_eq(source, tmp_path, [{"type": "highpass", "hz": float("nan")}])


# --- Finding 13: batch_analyze collisions and find_clicks stage discovery ---

def test_batch_files_with_same_name_get_distinct_outputs(tmp_path):
    from batch_analyze import _collect_jobs
    files = []
    for name in ("A", "B"):
        (tmp_path / name).mkdir()
        files.append(_noise(tmp_path / name / "assembled.wav", 11))
    jobs = _collect_jobs(None, files, tmp_path / "out", False)
    assert len({target for _, target in jobs}) == 2


def test_find_clicks_discovers_recorded_chain_and_source(tmp_path):
    from find_clicks import trace_at_time
    track = tmp_path / "tracks" / "Bass"
    track.mkdir(parents=True)
    click = np.zeros(4 * SR)
    click[2 * SR] = .5
    sf.write(tmp_path / "source.wav", click, SR, subtype="FLOAT")
    for name in ("assembled", "assembled_eq_amp_comp"):
        sf.write(track / f"{name}.wav", click, SR, subtype="FLOAT")
    (tmp_path / "session.json").write_text(json.dumps({"sample_rate": SR, "tracks": [
        {"name": "Bass", "clips": [_clip(tmp_path / "source.wav", 0, 0, 4 * SR)]}]}))
    (tmp_path / "mix_config.json").write_text(json.dumps({"tracks": [{"name": "Bass"}]}))
    stages = [entry["stage"] for entry in trace_at_time(tmp_path, 2.0, .2)["tracks"][0]["chain"]]
    assert stages[0].startswith("source") and stages[1:] == ["assembled", "assembled_eq_amp_comp"]
