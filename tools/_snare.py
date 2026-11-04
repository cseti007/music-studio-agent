"""Snare hit detection, descriptors and matching shared by the snare tools.

All analysis runs on mono audio at SR (48 kHz). One hit is described by:
  shape     1/3-octave levels 160 Hz-10 kHz of the first 150 ms, mean removed (dB);
            lower bands are left out: kick and bass residue dominate them
  f0_hz     drum tuning: lowest spectral peak within 6 dB of the strongest,
            160-450 Hz, in the 15-120 ms window
  wb_db     wires vs body: energy 2-8 kHz over 150-400 Hz in the first 100 ms
  crack_db  mean power 1-5 kHz in the first 5 ms over 5-50 ms
  decay_ms  5 ms RMS envelope fall from its peak to -20 dB
  tail_db   energy 120-500 ms over 0-120 ms (room/ring; reported, not matched)

A profile is the median and MAD of these over many hits. The distance D of a
hit (or profile) to a profile is a weighted RMS of robust z-scores; it ranks
candidates and measures whether a correction helped. It is a hypothesis
generator, not a listening verdict.
"""

from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, resample_poly, sosfilt

SR = 48000
SHAPE_HZ = [round(100.0 * 2.0 ** (k / 3.0), 1) for k in range(2, 21)]  # 159 Hz .. 10.2 kHz
# Bump when descriptor definitions change, so cached library indexes are rebuilt.
EXTRACTOR_VERSION = 5
WEIGHTS = {"shape": 2.0, "wb_db": 2.0, "f0": 1.5, "decay": 1.5, "crack_db": 1.0}
# Minimum spread used in z-scores, so a very consistent reference cannot make
# tiny differences look huge.
FLOORS = {"f0_st": 0.25, "wb_db": 1.0, "crack_db": 1.0, "shape_db": 0.5, "decay_log": 0.10}
MAD_SCALE = 1.4826
MIN_LONG_HITS = 5  # hits with a clean 500 ms window needed for decay and tail


def ms(value: float) -> int:
    return int(round(value * SR / 1000.0))


def load_mono(path) -> np.ndarray:
    """Read any soundfile-supported file as mono float64 at SR (channels averaged)."""
    data, sr = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        ratio = Fraction(SR, sr)
        mono = resample_poly(mono, ratio.numerator, ratio.denominator)
    return mono


def _hp(y: np.ndarray, hz: float, order: int = 2) -> np.ndarray:
    b, a = butter(order, hz / (SR / 2), "high")
    return lfilter(b, a, y)


def attack_onset(rect: np.ndarray, start: int, end: int) -> int:
    """Onset of the first attack in rect[start:end] (rect: rectified signal).

    Envelope: 1 ms centred RMS. Noise: 90th percentile of the envelope in the
    40 ms before `start`; zero when there is no room before it (a file that
    starts at the hit), so pre-hit noise above 10% of the peak in such a file
    is not handled. The attack peak is the envelope maximum within 3 ms
    of the first sample above max(10% of the window maximum, 2x noise), so a
    later, larger resonance cannot pull the onset late. The onset is the last
    crossing of max(10% of that peak, 2x noise) before the attack peak (the
    noise term capped at half the level it is compared with), refined
    to the first raw sample that also clears 1.5x the largest raw value of the
    millisecond before it (bleed spikes are not taken as the onset).
    """
    n = ms(1)
    lo_env = max(start - ms(40), 0)
    seg = rect[lo_env:min(end + ms(10), len(rect))]
    env = np.sqrt(np.convolve(seg ** 2, np.ones(n) / n, mode="same"))
    w0, w1 = start - lo_env, min(end - lo_env, len(env))
    window = env[w0:w1]
    pre = env[:w0]
    noise = float(np.percentile(pre, 90)) if len(pre) > n else 0.0
    # noise may raise a threshold, but never above half the level it is applied to
    first = w0 + int(np.flatnonzero(window >= max(0.1, min(2.0 * noise / window.max(), 0.5)) * window.max())[0])
    peak = first + int(np.argmax(env[first:first + ms(3) + 1]))
    thr = max(0.1 * env[peak], min(2.0 * noise, 0.5 * env[peak]))
    lo = max(peak - ms(10), 0)
    below = np.flatnonzero(env[lo:peak] < thr)
    cross = lo + (int(below[-1]) + 1 if len(below) else 0)
    # the centred envelope rises ~0.5 ms early: refine on the raw signal
    s0 = max(cross - ms(0.5), 0)
    raw_floor = 1.5 * float(seg[max(s0 - n, 0):s0].max()) if s0 > 0 else 0.0
    hits = np.flatnonzero(seg[s0:peak + 1] >= max(thr, raw_floor))
    return lo_env + (s0 + int(hits[0]) if len(hits) else cross)


def detect_onsets(y: np.ndarray, rel_db: float = -30.0, rise_db: float = 6.0,
                  min_gap_ms: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Onsets of new events in the 150 Hz high-passed signal.

    An event starts where a 2 ms RMS envelope rises rise_db above a 30 ms
    one-pole average of itself (a decaying drum's own oscillation does not
    qualify). Events quieter than rel_db below the loudest are ignored.
    Returns (onset sample indices, high-passed peak levels, full-band peak
    levels), levels in dBFS.
    """
    rect = np.abs(_hp(y, 150.0))
    full = np.abs(y)
    n = ms(2)
    fast = np.sqrt(np.convolve(rect ** 2, np.ones(n) / n, mode="same"))
    tau = 1.0 - np.exp(-1.0 / ms(30))
    slow = lfilter([tau], [1.0, tau - 1.0], fast)
    rising = 20.0 * np.log10((fast + 1e-9) / (np.concatenate([[0.0], slow[:-1]]) + 1e-9)) >= rise_db
    starts = np.flatnonzero(np.diff(rising.astype(np.int8)) == 1) + 1
    onsets, levels, full_levels, last = [], [], [], -ms(min_gap_ms)
    for st in starts:
        if st - last < ms(min_gap_ms) or st + ms(10) >= len(rect):
            continue
        on = attack_onset(rect, max(st - ms(5), 0), st + ms(5))
        onsets.append(on)
        levels.append(20.0 * np.log10(rect[on:on + ms(10)].max() + 1e-12))
        full_levels.append(20.0 * np.log10(full[on:on + ms(10)].max() + 1e-12))
        last = st
    onsets, levels, full_levels = np.array(onsets, dtype=int), np.array(levels), np.array(full_levels)
    keep = levels >= levels.max() + rel_db if len(levels) else np.array([], dtype=bool)
    return onsets[keep], levels[keep], full_levels[keep]


def _power_spectrum(seg: np.ndarray, n_fft: int, window: bool = False):
    if window:
        seg = seg * np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg, n_fft)) ** 2
    return np.fft.rfftfreq(n_fft, 1.0 / SR), spec


def _band(freqs, spec, lo, hi) -> float:
    return float(spec[(freqs >= lo) & (freqs < hi)].sum()) + 1e-20


def band_levels(seg: np.ndarray) -> np.ndarray:
    """Absolute 1/3-octave levels (dB) at SHAPE_HZ of a segment (rectangular window)."""
    freqs, spec = _power_spectrum(seg, 1 << 14)
    k = 2.0 ** (1.0 / 6.0)
    return np.array([10.0 * np.log10(spec[(freqs >= fc / k) & (freqs < fc * k)].mean() + 1e-20)
                     for fc in SHAPE_HZ])


def hit_descriptors(y: np.ndarray, onset: int) -> dict:
    """Descriptors of one hit starting at `onset` (see module docstring)."""
    seg = y[onset:onset + ms(500)]
    if len(seg) < ms(500):
        seg = np.pad(seg, (0, ms(500) - len(seg)))
    levels = band_levels(seg[:ms(150)])

    # A snare has several strong modes (e.g. ~200 and ~330 Hz); the strongest
    # flips between them from hit to hit. The lowest peak within 6 dB of the
    # strongest is a stable tuning reference.
    freqs, spec = _power_spectrum(seg[ms(15):ms(120)], 1 << 14, window=True)
    mag_db = 10.0 * np.log10(spec + 1e-20)
    idx = np.flatnonzero((freqs >= 160) & (freqs <= 450))
    local = idx[1:-1][(mag_db[idx[1:-1]] > mag_db[idx[:-2]]) & (mag_db[idx[1:-1]] >= mag_db[idx[2:]])]
    if len(local):
        i = int(local[mag_db[local] >= mag_db[local].max() - 6.0].min())
        a, b, c = mag_db[i - 1:i + 2]
        shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        f0 = float(freqs[i] + shift * (freqs[1] - freqs[0]))
    else:
        f0 = float("nan")

    freqs, spec = _power_spectrum(seg[:ms(100)], 1 << 13)
    wb = 10.0 * np.log10(_band(freqs, spec, 2000, 8000) / _band(freqs, spec, 150, 400))

    crack_band = sosfilt(butter(2, [1000 / (SR / 2), 5000 / (SR / 2)], "band", output="sos"), seg)
    crack = 10.0 * np.log10((np.mean(crack_band[:ms(5)] ** 2) + 1e-20)
                            / (np.mean(crack_band[ms(5):ms(50)] ** 2) + 1e-20))

    win = ms(5)
    env = np.sqrt(np.convolve(seg ** 2, np.ones(win) / win, mode="full")[:len(seg)])
    p = int(np.argmax(env[:ms(30)]))  # the hit's own peak, not a later event's
    below = np.flatnonzero(env[p:] < env[p] * 0.1)
    decay = (below[0] if len(below) else len(env) - p) * 1000.0 / SR

    tail = 10.0 * np.log10((np.sum(seg[ms(120):] ** 2) + 1e-20) / (np.sum(seg[:ms(120)] ** 2) + 1e-20))
    return {"levels": levels.tolist(), "shape": (levels - levels.mean()).tolist(), "f0_hz": f0,
            "wb_db": float(wb), "crack_db": float(crack), "decay_ms": float(decay), "tail_db": float(tail)}


def _num(x) -> float | None:
    """JSON-safe rounded number (None for NaN)."""
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def _robust(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).any():
        return float("nan"), float("nan")
    med = float(np.nanmedian(v))
    return med, float(MAD_SCALE * np.nanmedian(np.abs(v - med)))


def aggregate(descs: list[dict]) -> dict:
    """Median and scaled MAD of descriptors over hits. NaN values are ignored;
    decay and tail stay None unless MIN_LONG_HITS hits measured them."""
    shapes = np.array([d["shape"] for d in descs], dtype=float)
    with np.errstate(all="ignore"):
        valid = np.isfinite(shapes).any(axis=0)
        shape_med = np.full(shapes.shape[1], np.nan)
        shape_mad = np.full(shapes.shape[1], np.nan)
        shape_med[valid] = np.nanmedian(shapes[:, valid], axis=0)
        shape_mad[valid] = MAD_SCALE * np.nanmedian(np.abs(shapes[:, valid] - shape_med[valid]), axis=0)
        f0_med, _ = _robust([d["f0_hz"] for d in descs])
        _, f0_mad_st = _robust([12.0 * np.log2(d["f0_hz"] / f0_med) for d in descs])
        decays = [d["decay_ms"] for d in descs]
        if np.isfinite(decays).sum() < MIN_LONG_HITS:  # too few clean tails to trust
            decays = [float("nan")] * len(descs)
        decay_med, _ = _robust(decays)
        _, decay_mad_log = _robust([np.log(d / decay_med) for d in decays])
    prof = {"n": len(descs), "shape_hz": SHAPE_HZ,
            "shape": {"median": [_num(x) for x in shape_med], "mad": [_num(x) for x in shape_mad]},
            "f0_hz": {"median": _num(f0_med), "mad_st": _num(f0_mad_st)},
            "decay_ms": {"median": _num(decay_med), "mad_log": _num(decay_mad_log),
                         "n": int(np.isfinite([d["decay_ms"] for d in descs]).sum())}}
    for key in ("wb_db", "crack_db", "tail_db"):
        values = [d[key] for d in descs]
        if key == "tail_db" and np.isfinite(values).sum() < MIN_LONG_HITS:
            values = [float("nan")] * len(descs)
        med, mad = _robust(values)
        prof[key] = {"median": _num(med), "mad": _num(mad)}
    return prof


def as_descriptor(profile: dict) -> dict:
    """Profile medians in the per-hit descriptor format (for profile-to-profile D)."""
    nan = float("nan")
    med = lambda key: nan if profile[key]["median"] is None else profile[key]["median"]  # noqa: E731
    return {"shape": [nan if x is None else x for x in profile["shape"]["median"]],
            "f0_hz": med("f0_hz"), "wb_db": med("wb_db"), "crack_db": med("crack_db"),
            "decay_ms": med("decay_ms")}


def distance(desc: dict, profile: dict) -> tuple[float, dict]:
    """Weighted RMS of robust z-scores of `desc` against `profile`. Returns (D, z).

    Components missing on either side (NaN / None) are left out and the
    remaining weights renormalized.
    """
    arr = lambda xs: np.array([np.nan if x is None else x for x in xs], dtype=float)  # noqa: E731
    spread = lambda value, floor: floor if value is None else max(value, floor)  # noqa: E731
    shape, ref, mad = arr(desc["shape"]), arr(profile["shape"]["median"]), arr(profile["shape"]["mad"])
    ok = np.isfinite(shape) & np.isfinite(ref)
    z = {}
    if ok.sum() >= 3:
        # compare shapes over the bands both have, each re-centred on those bands
        d_shape = (shape[ok] - shape[ok].mean()) - (ref[ok] - ref[ok].mean())
        z["shape"] = float(np.sqrt(np.mean((d_shape / np.maximum(np.nan_to_num(mad[ok]), FLOORS["shape_db"])) ** 2)))
    p = profile
    with np.errstate(all="ignore"):
        if p["f0_hz"]["median"]:
            z["f0"] = 12.0 * np.log2(desc["f0_hz"] / p["f0_hz"]["median"]) / spread(p["f0_hz"]["mad_st"], FLOORS["f0_st"])
        for key in ("wb_db", "crack_db"):
            if p[key]["median"] is not None:
                z[key] = (desc[key] - p[key]["median"]) / spread(p[key]["mad"], FLOORS[key])
        if p["decay_ms"]["median"]:
            z["decay"] = np.log(desc["decay_ms"] / p["decay_ms"]["median"]) / spread(p["decay_ms"]["mad_log"],
                                                                                     FLOORS["decay_log"])
    z = {k: float(v) for k, v in z.items() if np.isfinite(v)}
    if not z:
        return float("nan"), {}
    d = float(np.sqrt(sum(WEIGHTS[k] * v ** 2 for k, v in z.items()) / sum(WEIGHTS[k] for k in z)))
    return d, {k: round(v, 2) for k, v in z.items()}


def find_reference_hits(drums: np.ndarray, no_drums: np.ndarray) -> list[dict]:
    """Clean, snare-like hits in a separated drum stem.

    Snare-like: body (150-400 Hz) and wire (1.5-8 kHz) energy present and
    little kick (<100 Hz) in the first 60 ms. Of those, the loudest third
    (backbeats) with no other event within 12 dB of the hit in the next 150 ms
    are kept; hits with no event within 20 dB in the whole 500 ms analysis
    window provide decay and tail (otherwise those are NaN). Levels are
    compared both high-passed and full band, so a kick counts as an event. Hits whose shape is an outlier (log-spectral
    distance > 2x the median) are dropped. Shape bands less than 6 dB above
    the no_drums residue are set to NaN.
    """
    onsets, levels, full_levels = detect_onsets(drums)
    snare_like = []
    for i, (on, level) in enumerate(zip(onsets, levels)):
        if on < ms(100) or on + ms(500) > len(drums):
            continue
        freqs, spec = _power_spectrum(drums[on:on + ms(60)], 1 << 12)
        total = _band(freqs, spec, 20, 20000)
        kick, body, wires = (_band(freqs, spec, 20, 100) / total, _band(freqs, spec, 150, 400) / total,
                             _band(freqs, spec, 1500, 8000) / total)
        if kick <= 0.25 and body >= 0.08 and wires >= 0.08:
            snare_like.append((i, on, level, kick, body, wires))
    if not snare_like:
        return []
    loud_floor = np.percentile([s[2] for s in snare_like], 66.7)
    hits = []
    for i, on, level, kick, body, wires in snare_like:
        after = onsets > on + ms(30)

        def events_within(db: float, until_ms: float) -> bool:
            near = (levels > level - db) | (full_levels > full_levels[i] - db)
            return bool(np.any(after & near & (onsets < on + ms(until_ms))))

        if level < loud_floor or events_within(12.0, 150):
            continue
        long_clean = not events_within(20.0, 500)
        desc = hit_descriptors(drums, on)
        if not long_clean:
            desc["decay_ms"] = desc["tail_db"] = float("nan")
        residue_levels = band_levels(no_drums[on:on + ms(150)])
        shape = np.array(desc["shape"])
        shape[np.array(desc["levels"]) - residue_levels < 6.0] = np.nan
        if np.isfinite(shape).sum() < 6:  # too few bands clear of the residue
            continue
        desc["shape"] = shape.tolist()
        pre = 10.0 * np.log10(np.mean(drums[on - ms(80):on - ms(5)] ** 2) + 1e-20)
        residue = 10.0 * np.log10(np.mean(no_drums[on - ms(20):on + ms(150)] ** 2) + 1e-20) - level
        hits.append({"onset": int(on), "time_s": round(on / SR, 3), "level_db": round(float(level), 1),
                     "isolation_db": round(float(level - pre), 1), "residue_db": round(float(residue), 1),
                     "long_clean": bool(long_clean), "body": round(float(body), 3),
                     "wires": round(float(wires), 3), "kick": round(float(kick), 3), "desc": desc})
    if not hits:
        return []
    shapes = np.array([h["desc"]["shape"] for h in hits])
    with warnings.catch_warnings():  # bands that are NaN in every hit stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(shapes, axis=0)
        lsd = np.array([np.sqrt(np.nanmean((shape - med) ** 2)) for shape in shapes])
    for h, value in zip(hits, lsd):
        h["lsd_db"] = round(float(value), 2)
        # quality order for listening: clean long tail, low residue, typical shape
        h["score"] = round(float(h["isolation_db"] - max(h["residue_db"] + 30.0, 0.0) - 3.0 * value
                                 + (5.0 if h["long_clean"] else 0.0)), 2)
    kept = [h for h, value in zip(hits, lsd) if value <= 2.0 * np.nanmedian(lsd)]
    return sorted(kept, key=lambda h: -h["score"])


def trigger_hits(y: np.ndarray, threshold_db: float = -26.0, react_ms: float = 2.0, detect_ms: float = 1.0,
                 release_ms: float = 30.0, release_ratio: float = 0.7079, hpf_hz: float = 150.0) -> list[dict]:
    """Offline replica of LSP Trigger (RMS sidechain + state machine), refined offline.

    The sidechain is the RMS (react_ms window) of the hpf_hz high-passed input;
    a hit opens when it stays >= threshold for detect_ms and re-arms after it
    stays <= threshold * release_ratio for release_ms. Each trigger is then
    refined without lookahead limits: its onset comes from attack_onset() and
    its level is the largest high-passed peak within 10 ms (dBFS).
    """
    h = _hp(y, hpf_hz)
    rect = np.abs(h)
    n = max(ms(react_ms), 1)
    c = np.concatenate([[0.0], np.cumsum(h ** 2)])
    idx = np.arange(1, len(h) + 1)
    env = np.sqrt(np.maximum((c[idx] - c[np.maximum(idx - n, 0)]) / n, 0.0))
    det = 10.0 ** (threshold_db / 20.0)
    nd, nr = ms(detect_ms), ms(release_ms)

    def runs(mask):
        d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        return starts, ends - starts

    a_s, a_l = runs(env >= det)
    r_s, r_l = runs(env <= det * release_ratio)
    keep = a_l >= nd + 1
    a_s, a_l = a_s[keep], a_l[keep]
    keep = r_l >= nr + 1
    r_s, r_l = r_s[keep], r_l[keep]
    hits, pos = [], 0
    while True:
        i = np.searchsorted(a_s + a_l, pos + 1)
        if i >= len(a_s):
            break
        start = max(a_s[i], pos)
        if a_s[i] + a_l[i] - start < nd + 1:
            pos = a_s[i] + a_l[i]
            continue
        t = start + nd
        lo, hi = max(t - ms(5), 0), min(t + ms(10), len(rect) - 1)
        peak = lo + int(np.argmax(rect[lo:hi + 1]))  # hit level: largest peak
        hits.append({"trigger": int(t), "onset": attack_onset(rect, lo, t + ms(3)),
                     "level_db": round(float(20.0 * np.log10(rect[peak] + 1e-12)), 2)})
        j = np.searchsorted(r_s + r_l, t + nr + 1)
        while j < len(r_s) and r_s[j] + r_l[j] - max(r_s[j], t) < nr + 1:
            j += 1
        if j >= len(r_s):
            break
        pos = max(r_s[j], t) + nr + 1
    return hits


def varispeed(y: np.ndarray, semitones: float) -> np.ndarray:
    """Retune by resampling (like a tape speed change): pitch and duration scale together.

    Drums keep their transient shape this way; a phase-vocoder pitch shift
    would smear the attack.
    """
    if abs(semitones) < 1e-6:
        return y.copy()
    ratio = Fraction(2.0 ** (-semitones / 12.0)).limit_denominator(2000)
    return resample_poly(y, ratio.numerator, ratio.denominator)
