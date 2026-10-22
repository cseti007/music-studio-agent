"""Pick and correct licensed snare samples to match reference snare profiles.

Reads one or more snare_profile.json files (see snare_profile.py), merges
them (median of the per-song medians; a spread report warns when songs
disagree more than hits within a song), scans sample libraries, ranks their
snare hits by the profile distance D, then corrects the best candidates:

  1. tune  - varispeed by n = 12*log2(f0_ref / f0_candidate), |n| <= 3
  2. EQ    - up to 6 bounded peak filters from the 1/3-octave shape difference

Each step is kept only if D drops and no descriptor gets worse by more than
1 z. Candidates are measured on the library kit mix (all microphones, like a
reference drum stem); the written one-shot is the close snare microphones
(top + polarity-aligned bottom) with the same corrections, ready to layer
under our own kit. With several snare microphones each one is also written
(<candidate>_top.wav, <candidate>_bottom.wav; same gain, same time base, so
they keep the library's balance) to layer next to our own top and bottom
tracks. D ranks hypotheses; listening decides.

Libraries: a DrumGizmo kit folder (instrument XML + multichannel WAVs) or a
folder of one-shot WAVs. A license.json in the library folder states
{"license", "release_ok", "attribution"}; without it release_ok is false and
the library is used only with --private.

Usage:
  python tools/snare_match.py --profile ref/a/snare_profile.json --profile ref/b/snare_profile.json \\
      --library ~/audio-libs/CrocellKit --output-dir output/snare_match
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

import _snare as S
from _recall import file_hash
from apply_eq import filter_signal
from compare_reference import _filters_from_delta, _fit_filter_gains

SCHEMA = "snare-match/1"
INDEX_SCHEMA = "snare-index/1"
EXCLUDE = ("rest", "whisk", "brush")  # not snare hits
MIN_VELOCITY_PCT = 0.67               # backbeat-strength hits
MAX_TUNE_ST = 3.0


def _license(lib: Path) -> dict:
    path = lib / "license.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"id": lib.name, "license": "unknown", "release_ok": False, "attribution": None}


def _to_sr(x: np.ndarray, sr: int) -> np.ndarray:
    return x if sr == S.SR else resample_poly(x, S.SR, sr, axis=0)


def _hit_onset(close: np.ndarray) -> int:
    """Hit onset measured on the close microphones (the kit mix peaks later, in the room)."""
    return S.attack_onset(np.abs(S._hp(close, 150.0)), 0, S.ms(60))


def _describe(kit: np.ndarray, close: np.ndarray) -> dict:
    desc = S.hit_descriptors(kit, _hit_onset(close))
    desc.pop("levels")
    return desc


def _mic_role(channel: str, top: str) -> str:
    low = channel.lower()
    if channel == top:
        return "top"
    return "bottom" if any(x in low for x in ("bottom", "bot", "under")) else low


def _load_hit(hit: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(kit mix, {role: close snare microphone}) of a library hit, mono at SR.

    Roles are "top", "bottom" (or the channel name); the other snare
    microphones are polarity-aligned to the top. A one-shot file is {"mono": ...}.
    """
    if "channels" not in hit:
        data, sr = sf.read(hit["file"], always_2d=True)
        mono = _to_sr(data.mean(axis=1), sr)
        return mono, {"mono": mono}
    cache, ch = {}, {}
    for name, (file, idx) in hit["channels"].items():
        if file not in cache:
            data, sr = sf.read(file, always_2d=True)
            cache[file] = _to_sr(data, sr)
        ch[name] = cache[file][:, idx]
    snare = [c for c in ch if "snare" in c.lower()]
    if not snare:
        raise ValueError(f"{hit['id']}: no snare microphone channel among {sorted(ch)}")
    top = next((c for c in snare if "top" in c.lower()), snare[0])
    body = butter(2, [150 / (S.SR / 2), 400 / (S.SR / 2)], "band", output="sos")
    for c in snare:  # align the other snare microphones (e.g. bottom) to the top's polarity
        if c != top and np.dot(sosfilt(body, ch[top]), sosfilt(body, ch[c])) < 0:
            ch[c] = -ch[c]
    hit["close_channels"] = snare
    return np.sum(list(ch.values()), axis=0), {_mic_role(c, top): ch[c] for c in snare}


def _close(mics: dict[str, np.ndarray]) -> np.ndarray:
    return np.sum(list(mics.values()), axis=0)


def _library_hits(lib: Path) -> list[dict]:
    hits = []
    for xml in sorted(lib.glob("*/*.xml")):
        root = ET.parse(xml).getroot()
        name = root.get("name", "")
        if root.tag != "instrument" or "snare" not in name.lower() or any(x in name.lower() for x in EXCLUDE):
            continue
        for sample in root.iter("sample"):
            channels = {af.get("channel"): (str(xml.parent / af.get("file")), int(af.get("filechannel", "1")) - 1)
                        for af in sample.iter("audiofile")}
            hits.append({"id": f"{lib.name}/{name}/{sample.get('name')}", "instrument": name,
                         "power": float(sample.get("power", 0.0)), "channels": channels})
    if not hits:  # plain one-shot folder
        for wav in sorted(lib.rglob("*.wav")):
            if sf.info(str(wav)).channels > 2:
                raise ValueError(f"{wav}: {sf.info(str(wav)).channels} channels without a DrumGizmo "
                                 "instrument XML; pass the kit folder that holds the instrument folders")
            hits.append({"id": f"{lib.name}/{wav.relative_to(lib)}", "instrument": wav.parent.name,
                         "file": str(wav)})
    return hits


def build_index(lib: Path, cache_dir: Path) -> list[dict]:
    """Library snare hits with kit-mix descriptors.

    Cached; the cache key covers the audio and instrument XML files (size,
    mtime) and the code that computes the descriptors.
    """
    stamp = {"code": [file_hash(Path(__file__).with_name(f)) for f in ("_snare.py", "snare_match.py")]}
    hits = _library_hits(lib)
    files = {str(x) for x in lib.glob("*/*.xml")}
    for hit in hits:
        files |= {f for f, _ in hit["channels"].values()} if "channels" in hit else {hit["file"]}
    for file in sorted(files):
        st = Path(file).stat()
        stamp[file] = [st.st_size, st.st_mtime_ns]
    cache = cache_dir / f"{lib.name}.json"
    if cache.is_file():
        index = json.loads(cache.read_text(encoding="utf-8"))
        if (index.get("schema") == INDEX_SCHEMA and index.get("extractor") == S.EXTRACTOR_VERSION
                and index.get("stamp") == stamp):
            return index["hits"]
    # velocity_pct ranks DrumGizmo velocity layers (sample power) within an
    # instrument; one-shot files are separate sounds, not velocity layers
    for instrument in {h["instrument"] for h in hits}:
        group = [h for h in hits if h["instrument"] == instrument and "power" in h]
        ranks = np.argsort(np.argsort([h["power"] for h in group]))
        for h, r in zip(group, ranks):
            h["velocity_pct"] = round(float(r / (len(group) - 1)), 3) if len(group) >= 3 else 1.0
    for h in hits:
        h.setdefault("velocity_pct", 1.0)
    for hit in hits:
        kit, mics = _load_hit(hit)
        hit["desc"] = _describe(kit, _close(mics))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"schema": INDEX_SCHEMA, "extractor": S.EXTRACTOR_VERSION, "library": str(lib),
                                 "stamp": stamp, "hits": hits}),
                     encoding="utf-8")
    return hits


def merge_profiles(profiles: list[dict]) -> tuple[dict, dict]:
    """Median of per-song medians; spread report per descriptor."""
    if len(profiles) == 1:
        return profiles[0], {}

    def med(values):
        v = [x for x in values if x is not None]
        return float(np.median(v)) if v else None

    merged = {"n": sum(p["n"] for p in profiles), "shape_hz": S.SHAPE_HZ, "songs": len(profiles),
              "shape": {"median": [med(col) for col in zip(*(p["shape"]["median"] for p in profiles))],
                        "mad": [med(col) for col in zip(*(p["shape"]["mad"] for p in profiles))]},
              "f0_hz": {"median": med(p["f0_hz"]["median"] for p in profiles),
                        "mad_st": med(p["f0_hz"]["mad_st"] for p in profiles)},
              "decay_ms": {"median": med(p["decay_ms"]["median"] for p in profiles),
                           "mad_log": med(p["decay_ms"]["mad_log"] for p in profiles)}}
    for key in ("wb_db", "crack_db", "tail_db"):
        merged[key] = {"median": med(p[key]["median"] for p in profiles), "mad": med(p[key]["mad"] for p in profiles)}
    spread = {}
    checks = {"f0_st": ("f0_hz", "mad_st", lambda v: 12.0 * np.log2(v)),
              "decay_log": ("decay_ms", "mad_log", np.log),
              "wb_db": ("wb_db", "mad", float), "crack_db": ("crack_db", "mad", float)}
    for name, (key, mad_key, fn) in checks.items():
        vals = [fn(p[key]["median"]) for p in profiles if p[key]["median"]]
        within = merged[key][mad_key]
        if len(vals) > 1 and within is not None:
            between = float(S.MAD_SCALE * np.median(np.abs(np.array(vals) - np.median(vals))))
            spread[name] = {"between_songs": round(between, 3), "within_song": round(within, 3),
                            "warn": between > within}
    return merged, spread


def _guarded(d_new: float, z_new: dict, d_old: float, z_old: dict) -> bool:
    worse = any(abs(z_new.get(k, 0.0)) - abs(z_old.get(k, 0.0)) > 1.0 for k in z_new)
    return d_new < d_old and not worse


def correct(kit: np.ndarray, close: np.ndarray, target: dict) -> dict:
    """Tune, then EQ, keeping each step only if it lowers D (see module docstring)."""
    desc = _describe(kit, close)
    d, z = S.distance(desc, target)
    out = {"D_raw": round(d, 3), "tune_semitones": 0.0, "eq_filters": [], "steps": []}
    cur, cur_close = kit, close
    f_ref, f_cand = target["f0_hz"]["median"], desc["f0_hz"]
    if f_ref and np.isfinite(f_cand):
        n = float(np.clip(12.0 * np.log2(f_ref / f_cand), -MAX_TUNE_ST, MAX_TUNE_ST))
        if abs(n) >= 0.1:
            y, y_close = S.varispeed(cur, n), S.varispeed(cur_close, n)
            dn, zn = S.distance(_describe(y, y_close), target)
            keep = _guarded(dn, zn, d, z)
            out["steps"].append({"name": "tune", "semitones": round(n, 2), "applied": keep,
                                 "D_before": round(d, 3), "D_after": round(dn, 3)})
            if keep:
                cur, cur_close, d, z, out["tune_semitones"] = y, y_close, dn, zn, round(n, 2)
    cur_desc = _describe(cur, cur_close)
    shape = np.array(cur_desc["shape"], dtype=float)
    ref = np.array([np.nan if x is None else x for x in target["shape"]["median"]], dtype=float)
    ok = np.isfinite(shape) & np.isfinite(ref)
    delta = np.zeros(len(shape))
    delta[ok] = (shape[ok] - shape[ok].mean()) - (ref[ok] - ref[ok].mean())
    # masked bands stay in the list at 0 dB so they split runs instead of
    # joining deviations on both sides into one filter
    runs = [{"hz": hz, "delta_db": round(float(v), 1)} for hz, v in zip(S.SHAPE_HZ, delta)]
    bands = [b for b, valid in zip(runs, ok) if valid]
    filters, _ = _fit_filter_gains(_filters_from_delta(runs, threshold_db=1.0), bands, S.SR)
    filters = [{k: v for k, v in f.items() if k != "_auto"} for f in filters]
    if filters:
        y = cur
        for f in filters:
            y = filter_signal(y, S.SR, f)
        dn, zn = S.distance(S.hit_descriptors(y, _hit_onset(cur_close)), target)
        keep = _guarded(dn, zn, d, z)
        out["steps"].append({"name": "eq", "filters": filters, "applied": keep,
                             "D_before": round(d, 3), "D_after": round(dn, 3)})
        if keep:
            d, z, out["eq_filters"] = dn, zn, filters
    out.update(D_final=round(d, 3), z_final=z)
    return out


def render_candidate(mics: dict[str, np.ndarray], corr: dict) -> dict[str, np.ndarray]:
    """Tuned and EQ'd one-shots: "close" (sum of the snare microphones) and,
    with several microphones, each one. All share one gain (loudest at -1 dBFS),
    so the microphones keep their balance and add up to "close"."""
    out = {}
    for role, y in mics.items():
        y = S.varispeed(y, corr["tune_semitones"])
        for f in corr["eq_filters"]:
            y = filter_signal(y, S.SR, f)
        out[role] = y
    result = {"close": _close(out), **(out if len(out) > 1 else {})}
    gain = 10.0 ** (-1.0 / 20.0) / (max(np.abs(y).max() for y in result.values()) + 1e-12)
    return {role: y * gain for role, y in result.items()}


def snare_match(profiles: list[Path], libraries: list[Path], output_dir: Path, top: int = 5,
                private: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in profiles]
    usable = [r for r in reports if r.get("profile")]
    if not usable:
        raise ValueError("no profile with measured hits")
    target, spread = merge_profiles([r["profile"] for r in usable])
    loo = [r["loo_d_p75"] for r in usable if r.get("loo_d_p75") is not None]
    accept_d = float(np.median(loo)) if loo else None

    pool, libs, skipped = [], [], []
    for lib in libraries:
        lic = _license(lib)
        if not lic.get("release_ok") and not private:
            skipped.append({"library": str(lib), "reason": "no release_ok license (use --private to include)"})
            continue
        hits = build_index(lib, output_dir / "index")
        libs.append({"library": str(lib), "license": lic, "n_hits": len(hits)})
        pool += [dict(h, lic=lic, library=lib) for h in hits]
    if not libs:
        raise ValueError(f"no usable library: {skipped}")

    # only tuning is a hard limit (varispeed range); brightness gaps are left to D and the EQ,
    # because a mastered reference is typically much brighter than raw library recordings
    f_ref = target["f0_hz"]["median"]
    pool = [h for h in pool if h["velocity_pct"] >= MIN_VELOCITY_PCT
            and not (f_ref and np.isfinite(h["desc"]["f0_hz"])
                     and abs(12.0 * np.log2(f_ref / h["desc"]["f0_hz"])) > MAX_TUNE_ST)]
    pool.sort(key=lambda h: S.distance(h["desc"], target)[0])

    candidates = []
    for h in pool[:max(2 * top, 10)]:
        kit, mics = _load_hit(h)
        corr = correct(kit, _close(mics), target)
        candidates.append((h, kit, mics, corr))
    candidates.sort(key=lambda c: c[3]["D_final"])

    out_dir = output_dir / "candidates"
    out_dir.mkdir(exist_ok=True)
    for old in list(out_dir.glob("*.wav")) + list(out_dir.glob("*.json")):  # no stale candidates
        old.unlink()
    listed = []
    for rank, (h, _, mics, corr) in enumerate(candidates[:top], 1):
        lic = h["lic"]
        stem = f"{rank:02d}_" + h["id"].replace("/", "_").replace(" ", "_")
        audio = render_candidate(mics, corr)
        files = {role: out_dir / (f"{stem}.wav" if role == "close" else f"{stem}_{role}.wav") for role in audio}
        meta = {"rank": rank, "id": h["id"], "instrument": h["instrument"], "velocity_pct": h["velocity_pct"],
                "close_channels": h.get("close_channels"), **corr,
                "accepted": accept_d is None or corr["D_final"] <= accept_d,
                "license": lic.get("license"), "release_ok": bool(lic.get("release_ok")),
                "attribution": lic.get("attribution"), "file": str(files["close"]),
                "mic_files": {role: str(f) for role, f in files.items() if role != "close"}}
        for role, wav in files.items():
            sf.write(str(wav), audio[role], S.SR, subtype="PCM_24")
            wav.with_suffix(".json").write_text(json.dumps(dict(meta, mic=role), indent=2), encoding="utf-8")
        listed.append(meta)

    report = {"schema": SCHEMA, "profiles": [str(p) for p in profiles], "merged_profile": target,
              "song_spread": spread, "accept_d": accept_d, "libraries": libs, "skipped_libraries": skipped,
              "pool_size": len(pool), "candidates": listed,
              "status": "ok" if any(c["accepted"] for c in listed) else "no_acceptable_match"}
    (output_dir / "snare_match.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Match licensed snare samples to reference snare profiles.")
    parser.add_argument("--profile", type=Path, action="append", required=True,
                        help="snare_profile.json (repeat for several songs)")
    parser.add_argument("--library", type=Path, action="append", required=True,
                        help="Sample library folder (repeat for several)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=5, help="Candidates to write (default 5)")
    parser.add_argument("--private", action="store_true",
                        help="Also use libraries without a release_ok license (private listening only)")
    args = parser.parse_args()
    for path in args.profile + args.library:
        if not path.exists():
            print(json.dumps({"error": f"Not found: {path}"}), file=sys.stderr)
            sys.exit(1)
    try:
        report = snare_match(args.profile, args.library, args.output_dir, args.top, args.private)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)
    summary = {k: report[k] for k in ("status", "accept_d", "song_spread", "pool_size", "skipped_libraries")}
    summary["candidates"] = [{k: c[k] for k in ("rank", "id", "D_raw", "D_final", "tune_semitones", "accepted",
                                                "release_ok", "file", "mic_files")} for c in report["candidates"]]
    summary["report"] = str(args.output_dir / "snare_match.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
