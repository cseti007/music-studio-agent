"""Recall must preserve operations and fail before partial processing."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_preset_plus_custom_eq_keeps_all_filters():
    from build_chain import _step_from_eq
    filters = [{"type": "highpass", "hz": 30}, {"type": "peak", "hz": 500, "q": 2, "db": -4}]
    step = _step_from_eq({"preset_used": "bass_di", "filters_applied": filters})
    assert step["args"]["filters"] == filters
    assert "preset" not in step["args"]


def test_repeated_eq_roundtrip(tmp_path):
    from apply_eq import apply_eq
    from build_chain import build_stem_chain
    from replay_chain import replay
    source = tmp_path / "assembled.wav"
    sf.write(source, np.random.default_rng(8).normal(0, .02, (48000, 2)), 48000, subtype="FLOAT")
    first = apply_eq(source, tmp_path, [{"type": "highpass", "hz": 60}])
    second = apply_eq(Path(first["output"]), tmp_path, [{"type": "peak", "hz": 1000, "q": 1, "db": -3}])
    expected, _ = sf.read(second["output"])
    steps = build_stem_chain(tmp_path)
    assert len(steps) == 2
    chain = tmp_path / "chain.json"
    chain.write_text(json.dumps({"stems": [{"name": "Track", "chain": steps}]}))
    Path(first["output"]).unlink()
    Path(second["output"]).unlink()
    assert replay(chain, dry_run=False, stem_filter="Track") == 0
    assert np.array_equal(sf.read(second["output"])[0], expected)


def test_unsupported_step_fails_before_running_anything(tmp_path, monkeypatch):
    import replay_chain
    calls = []
    monkeypatch.setattr(replay_chain, "_build_argv", lambda *a: calls.append(a))
    chain = tmp_path / "chain.json"
    chain.write_text(json.dumps({"stems": [{"name": "Track", "chain": [
        {"step": "unknown", "input": "a.wav", "output": "b.wav", "args": {}}
    ]}]}))
    assert replay_chain.replay(chain, False, "Track") != 0
    assert calls == []


def test_changed_source_is_rejected_before_replay(tmp_path):
    from apply_eq import apply_eq
    from build_chain import build_stem_chain
    from replay_chain import replay
    source = tmp_path / "source.wav"
    sf.write(source, np.random.default_rng(9).normal(0, .02, 48000), 48000)
    report = apply_eq(source, tmp_path, [{"type": "highpass", "hz": 60}])
    original = Path(report["output"]).read_bytes()
    chain = tmp_path / "chain.json"
    chain.write_text(json.dumps({"stems": [{"name": "Track", "chain": build_stem_chain(tmp_path)}]}))
    sf.write(source, np.zeros(48000), 48000)
    assert replay(chain, False, "Track") != 0
    assert Path(report["output"]).read_bytes() == original


def test_bus_cycles_are_rejected():
    from render_mix import _topo_order
    with pytest.raises(ValueError, match="cycle"):
        _topo_order({"a": {"parent_bus": "b"}, "b": {"parent_bus": "a"}})


def test_invalid_mix_routing_fails_before_rebuilding_stems(tmp_path):
    from apply_eq import apply_eq
    from build_chain import build_stem_chain
    from replay_chain import replay
    source = tmp_path / "source.wav"
    sf.write(source, np.random.default_rng(9).normal(0, .02, 48000), 48000)
    report = apply_eq(source, tmp_path, [{"type": "highpass", "hz": 60}])
    config = tmp_path / "mix_config.json"
    config.write_text(json.dumps({"sample_rate": 48000, "master": {}, "buses": {},
        "tracks": [{"name": "Track", "file": report["output"], "bus": "missing"}]}))
    chain = tmp_path / "chain.json"
    chain.write_text(json.dumps({"mix_config": str(config),
        "stems": [{"name": "Track", "chain": build_stem_chain(tmp_path)}]}))
    Path(report["output"]).unlink()
    assert replay(chain, False, None) != 0
    assert not Path(report["output"]).exists()


@pytest.mark.parametrize("use_subprocess", [False, True])
def test_recall_preserves_sidechain_parameters(tmp_path, use_subprocess):
    from apply_compression import apply_compression
    from build_chain import build_stem_chain
    from replay_chain import replay
    rng = np.random.default_rng(3)
    source, sidechain = tmp_path / "source.wav", tmp_path / "sidechain.wav"
    sf.write(source, rng.normal(0, .02, (48000, 2)), 48000, subtype="FLOAT")
    sf.write(sidechain, rng.normal(0, .1, (48000, 2)), 48000, subtype="FLOAT")
    report = apply_compression(source, tmp_path, -30, 4, 5, 80,
                               makeup_db=0, mix=.5, sidechain_path=sidechain,
                               sc_hp_hz=80, sc_lp_hz=2000)
    expected, _ = sf.read(report["output"])
    chain = tmp_path / "chain.json"
    chain.write_text(json.dumps({"stems": [{"name": "Track", "chain": build_stem_chain(tmp_path)}]}))
    Path(report["output"]).unlink()
    assert replay(chain, False, "Track", use_subprocess=use_subprocess) == 0
    assert np.array_equal(sf.read(report["output"])[0], expected)


def test_continuous_assembly_recall_preserves_mode_and_crossfades(tmp_path):
    from apply_gain import apply_gain_per_clip
    from build_chain import build_stem_chain
    from replay_chain import replay
    source = tmp_path / "source.wav"
    sf.write(source, np.random.default_rng(8).normal(0, .02, 96000), 48000, subtype="FLOAT")
    session = tmp_path / "session.json"
    clips = [{"source_file": str(source), "timeline_start_sample": i * 48000,
              "source_offset_sample": i * 48000, "length_samples": 48000} for i in range(2)]
    session.write_text(json.dumps({"sample_rate": 48000, "duration_samples": 96000,
                                  "tracks": [{"name": "Track", "clips": clips}]}))
    report = apply_gain_per_clip(session, tmp_path / "tracks", all_tracks=True,
                                 source_mode="continuous", crossfade_ms=20,
                                 interloper_head_ms=100, interloper_tail_ms=40)[0]
    expected, _ = sf.read(report["output"])
    steps = build_stem_chain(Path(report["output"]).parent)
    assert steps[0]["args"]["kwargs"]["source_mode"] == "continuous"
    chain = tmp_path / "chain.json"
    chain.write_text(json.dumps({"stems": [{"name": "Track", "chain": steps}]}))
    Path(report["output"]).unlink()
    assert replay(chain, False, "Track") == 0
    assert np.array_equal(sf.read(report["output"])[0], expected)


def test_recall_hash_ignores_wav_peak_timestamp(tmp_path):
    import struct
    from _recall import content_hash, file_hash
    source = tmp_path / "source.wav"
    sf.write(source, np.ones(48000) * .1, 48000, subtype="FLOAT")
    expected_audio, original_file = content_hash(source), file_hash(source)
    data = bytearray(source.read_bytes())
    offset = 12
    while data[offset:offset + 4] != b"PEAK":
        size = struct.unpack_from("<I", data, offset + 4)[0]
        offset += 8 + size + size % 2
        assert offset < len(data)
    timestamp = struct.unpack_from("<I", data, offset + 12)[0]
    struct.pack_into("<I", data, offset + 12, timestamp + 1)
    source.write_bytes(data)
    assert file_hash(source) != original_file
    assert content_hash(source) == expected_audio


def test_full_replay_restores_audio_and_direct_master_routing(tmp_path):
    from apply_eq import apply_eq
    from build_chain import build_chain
    from render_mix import render_mix
    from replay_chain import replay
    source = tmp_path / "source.wav"
    sf.write(source, np.random.default_rng(8).normal(0, .02, (48000, 2)), 48000, subtype="FLOAT")
    report = apply_eq(source, tmp_path / "tracks" / "Track", [{"type": "highpass", "hz": 60}])
    config = tmp_path / "mix_config.json"
    config.write_text(json.dumps({"sample_rate": 48000, "output_dir": str(tmp_path / "mixes"),
        "master": {"premaster_mode": True, "peak_target_dbfs": -3}, "buses": {},
        "tracks": [{"name": "Track", "file": report["output"], "bus": "master"}]}))
    render_mix(config)
    mix = tmp_path / "mixes" / "mix.wav"
    expected, _ = sf.read(mix)
    assert np.max(np.abs(expected)) == pytest.approx(10 ** (-3 / 20), abs=1e-6)
    chain = tmp_path / "mix_chain.json"
    chain.write_text(json.dumps(build_chain(tmp_path)))
    Path(report["output"]).unlink()
    mix.unlink()
    assert replay(chain, False, None) == 0
    assert np.array_equal(sf.read(mix)[0], expected)
