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


# ---------------------------------------------------------------------------
# regressions from review (2026-10-22)
# ---------------------------------------------------------------------------

def _bleed(n, db, hp_hz, seed=0):
    from scipy.signal import butter, sosfilt
    noise = np.random.default_rng(seed).standard_normal(n)
    noise = sosfilt(butter(4, hp_hz / (SR / 2), "high", output="sos"), noise)
    return noise / np.sqrt(np.mean(noise ** 2)) * 10 ** (db / 20)


def test_decay_ignores_next_beat_kick():
    """A kick on the next beat (+462 ms at 130 bpm) must not replace the snare decay."""
    beat = S.ms(60000 / 130)
    events = []
    for i in range(30):
        start = S.ms(300) + 2 * i * beat
        events += [(start, snare(seed=i), 0.5), (start + beat, kick(), 0.6)]
    drums = place(30 * 2 * 60 / 130 + 1.0, events)
    noise = np.random.default_rng(2).standard_normal(len(drums)) * 1e-4
    descs = [h["desc"] for h in S.find_reference_hits(drums, noise)]
    isolated = S.hit_descriptors(place(1.0, [(S.ms(100), snare(seed=0), 0.5)]), S.ms(100))["decay_ms"]
    decays = [d["decay_ms"] for d in descs if np.isfinite(d["decay_ms"])]
    assert descs and all(abs(d - isolated) / isolated < 0.15 for d in decays)


@pytest.mark.parametrize("bleed_db", [-45.0, -40.0])
def test_trigger_onsets_stay_accurate_with_bleed(bleed_db):
    rng = np.random.default_rng(1)
    onsets = [S.ms(200 + 500 * i) for i in range(12)]
    y = place(6.5, [(o, snare(seed=i), 10 ** (rng.uniform(-14, -4) / 20)) for i, o in enumerate(onsets)])
    y += _bleed(len(y), bleed_db, 2000.0)
    hits = S.trigger_hits(y)
    assert len(hits) == 12
    assert max(abs(h["onset"] - o) for h, o in zip(hits, onsets)) <= S.ms(0.25)


def test_reference_onsets_and_crack_robust_to_cymbal_wash():
    events = [(S.ms(300 + 600 * i), snare(seed=i, wires=0.8), 0.3) for i in range(30)]
    clean = place(19.0, events)
    washed = clean + _bleed(len(clean), 20 * np.log10(0.3) - 26.0, 3000.0, seed=4)
    rest = np.random.default_rng(3).standard_normal(len(clean)) * 1e-4
    true = {e[0] for e in events}
    hits = S.find_reference_hits(washed, rest)
    assert hits and all(min(abs(h["onset"] - t) for t in true) <= S.ms(1) for h in hits)
    crack = lambda hs: np.median([h["desc"]["crack_db"] for h in hs])  # noqa: E731
    assert abs(crack(hits) - crack(S.find_reference_hits(clean, rest))) < 1.0


def test_f0_ignores_sustained_bass_under_the_snare():
    events = [(S.ms(300 + 600 * i), snare(f0=240.0, seed=i, wires=0.8), 0.3) for i in range(30)]
    drums = place(19.0, events)
    t = np.arange(len(drums)) / SR
    drums += 0.3 * 10 ** (-26 / 20) * np.sin(2 * np.pi * 147.0 * t)
    rest = np.random.default_rng(3).standard_normal(len(drums)) * 1e-4
    prof = S.aggregate([h["desc"] for h in S.find_reference_hits(drums, rest)])
    assert abs(prof["f0_hz"]["median"] - 240.0) < 4.0


def test_one_shot_libraries_are_not_dropped_by_velocity(tmp_path):
    drums, rest = _reference(tmp_path)
    sp.snare_profile(drums, tmp_path / "prof", drums_stem=drums, no_drums_stem=rest)
    lib = tmp_path / "oneshots"
    for i, (f0, gain) in enumerate([(200.0, 0.9), (240.0, 0.2), (300.0, 0.6)]):  # closest is the quietest
        _write(lib / f"snare_{i}.wav", gain * snare(f0=f0, wires=0.8, seed=20 + i))
    (lib / "license.json").write_text(json.dumps({"license": "CC0", "release_ok": True}))
    report = sm.snare_match([tmp_path / "prof" / "snare_profile.json"], [lib], tmp_path / "m")
    assert report["candidates"][0]["id"] == "oneshots/snare_1.wav"


def test_drumgizmo_kit_bottom_polarity_and_default_filechannel(tmp_path):
    lib = tmp_path / "Kit"
    inst = lib / "Snare"
    (inst / "samples").mkdir(parents=True)
    top = snare(f0=230.0, seed=1)
    sf.write(str(inst / "samples" / "1.wav"), np.stack([top, -0.6 * top, 0.2 * top], axis=1), SR, subtype="FLOAT")
    (inst / "Snare.xml").write_text(
        '<instrument version="2.0" name="Snare"><samples><sample name="Snare-1" power="1.0">'
        '<audiofile channel="SnareTop" file="samples/1.wav"/>'
        '<audiofile channel="SnareBottom" file="samples/1.wav" filechannel="2"/>'
        '<audiofile channel="OHLeft" file="samples/1.wav" filechannel="3"/></sample></samples></instrument>')
    (hit,) = sm._library_hits(lib)
    kit, close = sm._load_hit(hit)
    assert np.allclose(close, 1.6 * top) and np.allclose(kit, 1.8 * top)
    assert hit["close_channels"] == ["SnareTop", "SnareBottom"]


def _render_case(tmp_path, last_hit_s=None):
    onsets = [S.ms(200 + 500 * i) for i in range(6)]
    if last_hit_s is not None:
        onsets.append(S.ms(last_hit_s * 1000))
    length = (max(onsets) / SR) + (0.03 if last_hit_s is not None else 0.6)
    track = _write(tmp_path / "snare_top.wav", place(length, [(o, snare(seed=i), 0.5) for i, o in enumerate(onsets)]))
    sample = _write(tmp_path / "cand" / "s.wav", snare(f0=210.0, seed=5))
    sample.with_suffix(".json").write_text(json.dumps({"license": "CC0", "release_ok": True}))
    return track, sample, onsets


def test_render_offset_bounds_and_shift(tmp_path):
    track, sample, _ = _render_case(tmp_path)
    with pytest.raises(ValueError):
        rsl.render_snare_layer(track, sample, tmp_path / "bad", offset_ms=3.0)
    zero = sf.read(rsl.render_snare_layer(track, sample, tmp_path / "z", polarity="normal", offset_ms=0.0)["output"])[0]
    late = sf.read(rsl.render_snare_layer(track, sample, tmp_path / "p", polarity="normal", offset_ms=2.0)["output"])[0]
    assert len(late) == len(zero) and np.allclose(late[S.ms(2):], zero[:-S.ms(2)])


def test_render_hit_near_the_end(tmp_path):
    track, sample, _ = _render_case(tmp_path, last_hit_s=3.4)
    report = rsl.render_snare_layer(track, sample, tmp_path / "out")
    assert len(sf.read(report["output"])[0]) == sf.info(str(track)).frames and report["placed"] == 7


def test_render_refuses_a_shifted_kit(tmp_path):
    track, sample, _ = _render_case(tmp_path)
    y = sf.read(str(track))[0]
    kit = _write(tmp_path / "kit.wav", np.concatenate([np.zeros(S.ms(100)), y]))  # exported from another start
    with pytest.raises(ValueError, match="shifted"):
        rsl.render_snare_layer(track, sample, tmp_path / "out", kit_path=kit)
    aligned = _write(tmp_path / "kit_ok.wav", y)
    assert abs(rsl.render_snare_layer(track, sample, tmp_path / "ok", kit_path=aligned)["alignment_test"]["kit_lag_ms"]) < 1


def test_render_reports_hits_below_the_threshold(tmp_path):
    onsets = [S.ms(200 + 500 * i) for i in range(10)]
    gains = [0.25] * 5 + [0.08] * 5  # second half ~10 dB softer, below the -26 dB RMS threshold
    track = _write(tmp_path / "t.wav", place(5.5, [(o, snare(seed=i), g) for i, (o, g) in enumerate(zip(onsets, gains))]))
    sample = _write(tmp_path / "cand" / "s.wav", snare(seed=5))
    sample.with_suffix(".json").write_text(json.dumps({"license": "CC0", "release_ok": True}))
    report = rsl.render_snare_layer(track, sample, tmp_path / "out")
    assert report["placed"] + report["possibly_missed"]["count"] >= 9
