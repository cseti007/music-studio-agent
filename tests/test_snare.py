"""Snare matching tools: descriptors, trigger, profile, match and layer render.

Synthetic snares (damped tone + decaying noise) at 48 kHz stand in for real
recordings, so every expected value is known.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import _snare as S  # noqa: E402
import render_snare_layer as rsl  # noqa: E402
import snare_match as sm  # noqa: E402
import snare_profile as sp  # noqa: E402
from replay_chain import _run_recorded  # noqa: E402

SR = S.SR


def snare(f0=230.0, decay_ms=120.0, wires=0.5, seed=0, n=S.ms(450)):
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    tone = np.sin(2 * np.pi * f0 * t) * np.exp(-t / (decay_ms / 1000 / 2.3))
    noise = rng.standard_normal(n) * np.exp(-t / 0.05) * wires
    return (tone + noise) / np.max(np.abs(tone + noise))  # peak 0 dBFS


def kick(n=S.ms(300)):
    t = np.arange(n) / SR
    return 0.8 * np.sin(2 * np.pi * 55 * t) * np.exp(-t / 0.08)


def place(length_s, events):
    y = np.zeros(int(length_s * SR))
    for start, sig, gain in events:
        y[start:start + len(sig)] += gain * sig[:len(y) - start]
    return y


# ---------------------------------------------------------------------------
# descriptors and helpers
# ---------------------------------------------------------------------------

def test_descriptors_measure_tuning_and_decay():
    y = place(1.0, [(S.ms(100), snare(f0=230.0, decay_ms=120.0), 1.0)])
    d = S.hit_descriptors(y, S.ms(100))
    assert abs(d["f0_hz"] - 230.0) < 2.0
    assert abs(d["decay_ms"] - 120.0) / 120.0 < 0.1
    brighter = S.hit_descriptors(place(1.0, [(S.ms(100), snare(wires=2.0), 1.0)]), S.ms(100))
    assert brighter["wb_db"] > d["wb_db"] + 3.0


@pytest.mark.parametrize("semitones", [-2.0, 1.5])
def test_varispeed_retunes(semitones):
    y = place(1.0, [(S.ms(100), S.varispeed(snare(), semitones), 1.0)])
    f0 = S.hit_descriptors(y, S.ms(100))["f0_hz"]
    assert abs(12 * np.log2(f0 / 230.0) - semitones) < 0.1


def test_distance_is_zero_for_identical_and_grows_with_detune():
    descs = [S.hit_descriptors(place(1.0, [(S.ms(100), snare(seed=i), 1.0)]), S.ms(100)) for i in range(6)]
    prof = S.aggregate(descs)
    d_same, _ = S.distance(S.as_descriptor(prof), prof)
    detuned = S.hit_descriptors(place(1.0, [(S.ms(100), snare(f0=290.0), 1.0)]), S.ms(100))
    d_far, z = S.distance(detuned, prof)
    assert d_same < 1e-6
    assert d_far > 1.0 and z["f0"] > 3.0


def test_trigger_finds_hits_ignores_bleed_and_tracks_level():
    rng = np.random.default_rng(1)
    events, onsets = [], []
    for i in range(12):
        start = S.ms(200 + 500 * i)
        gain = 10 ** (rng.uniform(-14, -4) / 20)
        events.append((start, snare(seed=i), gain))
        onsets.append(start)
        events.append((start + S.ms(250), kick(), 10 ** (-40 / 20)))  # quiet bleed between hits
    y = place(6.5, events)
    hits = S.trigger_hits(y)
    assert len(hits) == 12
    assert max(abs(h["onset"] - o) for h, o in zip(hits, onsets)) <= S.ms(0.5)
    gains = [20 * np.log10(g) for _, _, g in events[::2]]
    assert np.corrcoef(gains, [h["level_db"] for h in hits])[0, 1] > 0.95


def test_reference_hits_skip_kicks_and_dense_hits():
    events = [(S.ms(300 + 600 * i), snare(seed=i), 0.3) for i in range(12)]
    events += [(S.ms(600 + 600 * i), kick(), 0.9) for i in range(12)]
    events += [(S.ms(300 + 600 * 5) + S.ms(80), snare(seed=99), 0.3)]  # flam: not isolated
    drums = place(8.0, events)
    noise = np.random.default_rng(2).standard_normal(len(drums)) * 1e-4
    hits = S.find_reference_hits(drums, noise)
    times = sorted(h["onset"] for h in hits)
    assert all(any(abs(t - e[0]) <= S.ms(1) for e in events[:12]) for t in times)  # snares only
    assert not any(abs(t - (S.ms(300 + 600 * 5))) <= S.ms(1) for t in times)  # flam rejected
    assert len(hits) >= 3


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

def _write(path, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, SR, subtype="FLOAT")
    return path


def _reference(tmp_path, f0=240.0):
    events = [(S.ms(300 + 600 * i), snare(f0=f0, seed=i, wires=0.8), 0.3) for i in range(30)]
    drums = place(19.0, events)
    rest = np.random.default_rng(3).standard_normal(len(drums)) * 1e-4
    return (_write(tmp_path / "ref" / "drums.wav", drums), _write(tmp_path / "ref" / "no_drums.wav", rest))


def _library(tmp_path, name="lib", licensed=True):
    lib = tmp_path / name
    for i, (f0, wires) in enumerate([(200.0, 0.3), (226.0, 0.8), (260.0, 0.2), (180.0, 1.5)]):
        for v in range(3):  # three velocities per instrument
            _write(lib / f"snare{i}" / f"v{v}.wav", (0.2 + 0.15 * v) * snare(f0=f0, wires=wires, seed=10 + i))
    if licensed:
        (lib / "license.json").write_text(json.dumps({"license": "CC0", "release_ok": True, "attribution": None}))
    return lib


def test_profile_and_match_pick_the_closest_and_correct_it(tmp_path):
    drums, rest = _reference(tmp_path)
    prof = sp.snare_profile(drums, tmp_path / "prof", drums_stem=drums, no_drums_stem=rest)
    assert prof["status"] == "ok" and abs(prof["profile"]["f0_hz"]["median"] - 240.0) < 3.0
    assert all(Path(f).is_file() for f in prof["private_ref_hits"])
    report = sm.snare_match([tmp_path / "prof" / "snare_profile.json"], [_library(tmp_path)], tmp_path / "match")
    best = report["candidates"][0]
    assert best["id"].startswith("lib/snare1/")  # f0 226 Hz, wires 0.8: the closest
    assert best["D_final"] <= best["D_raw"]
    assert all(s["D_after"] < s["D_before"] for s in best["steps"] if s["applied"])
    assert Path(best["file"]).is_file() and best["release_ok"]


def test_match_skips_unlicensed_library_unless_private(tmp_path):
    drums, rest = _reference(tmp_path)
    sp.snare_profile(drums, tmp_path / "prof", drums_stem=drums, no_drums_stem=rest)
    lib = _library(tmp_path, "nolicense", licensed=False)
    profile = [tmp_path / "prof" / "snare_profile.json"]
    with pytest.raises(ValueError):
        sm.snare_match(profile, [lib], tmp_path / "m1")
    report = sm.snare_match(profile, [lib], tmp_path / "m2", private=True)
    assert report["candidates"] and not report["candidates"][0]["release_ok"]


def test_render_places_hits_sample_accurately_and_replays(tmp_path):
    onsets = [S.ms(200 + 500 * i) for i in range(10)]
    track = _write(tmp_path / "snare_top.wav", place(5.5, [(o, snare(seed=i), 0.5) for i, o in enumerate(onsets)]))
    sample = _write(tmp_path / "cand" / "s.wav", snare(f0=210.0, seed=5))
    sample.with_suffix(".json").write_text(json.dumps({"license": "CC0", "release_ok": True}))
    report = rsl.render_snare_layer(track, sample, tmp_path / "out", polarity="normal", offset_ms=0.0)
    layer, sr = sf.read(report["output"])
    assert sr == SR and len(layer) == sf.info(str(track)).frames and report["placed"] == 10
    rect = np.abs(S._hp(layer, 150.0))
    for o in onsets:
        assert abs(S.attack_onset(rect, o - S.ms(5), o + S.ms(5)) - o) <= S.ms(0.25)
    op = json.loads(Path(report["output"]).with_suffix(".operation.json").read_text())
    _run_recorded(op)  # replays to the same audio hash


def test_render_refuses_unlicensed_sample_unless_private(tmp_path):
    track = _write(tmp_path / "snare_top.wav", place(2.0, [(S.ms(200), snare(), 0.3)]))
    sample = _write(tmp_path / "private_ref" / "hit.wav", snare(seed=7))
    with pytest.raises(ValueError):
        rsl.render_snare_layer(track, sample, tmp_path / "out")
    assert rsl.render_snare_layer(track, sample, tmp_path / "out2", private=True)["private"]


@pytest.mark.parametrize("sign,expected", [(1.0, "normal"), (-1.0, "inverted")])
def test_auto_alignment_finds_coherent_polarity_and_zero_offset(tmp_path, sign, expected):
    hit = snare(seed=5)
    onsets = [S.ms(200 + 500 * i) for i in range(8)]
    track = _write(tmp_path / "snare_top.wav", place(4.5, [(o, sign * hit, 0.5) for o in onsets]))
    sample = _write(tmp_path / "cand" / "s.wav", hit)
    sample.with_suffix(".json").write_text(json.dumps({"license": "CC0", "release_ok": True}))
    report = rsl.render_snare_layer(track, sample, tmp_path / "out")
    assert report["polarity"] == expected and report["offset_ms"] == 0.0
