"""Master-level health scorecard.

Distinct from mix_health.py:
  - mix_health scores a finished STEM mix (LUFS, M/S width avg, masking pairs,
    stem pumping)
  - master_health scores a finished STEREO MASTER (format conformance,
    per-band phase coherence, M/S width profile, punch index, oversampled true peak
    simulation, compression-history detection, optional reference-deck
    comparison)

The master engineer cares about different things than the mix engineer.
Mix health asks "is the mix done?". Master health asks "will this master
survive Spotify's encoder, sound right on a phone speaker, hold up next
to commercial releases?"

Usage:
  python master_health.py master.wav --format spotify --output-dir DIR

  # With a reference deck (multiple commercial tracks averaged)
  python master_health.py master.wav --format spotify --output-dir DIR \
      --reference ref1.wav ref2.wav ref3.wav

  # Just analyse — no format conformance check
  python master_health.py mix.wav --output-dir DIR
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt, welch

sys.path.insert(0, str(Path(__file__).parent))
from _dsp import worst_channel_true_peak_dbfs  # noqa: E402
from master_mix import FORMAT_PRESETS  # noqa: E402

GREEN = "[OK]"
YELLOW = "[!] "
RED = "[X] "


def _verdict(green: bool, yellow: bool) -> str:
    if green:
        return GREEN
    if yellow:
        return YELLOW
    return RED


# ---------------------------------------------------------------------------
# Per-band phase coherence
# ---------------------------------------------------------------------------

_PHASE_BANDS = [
    ("sub",  20,    100),
    ("low",  100,   300),
    ("mid",  300,   2000),
    ("high", 2000,  8000),
    ("air",  8000,  20000),
]


def _band_filter(channel: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    nyq = sr / 2.0
    if lo <= 1.0:
        sos = butter(4, hi / nyq, btype="low", output="sos")
    else:
        sos = butter(4, [lo / nyq, min(hi / nyq, 0.999)], btype="band", output="sos")
    return sosfilt(sos, channel)


def _filter_bands_LR(L: np.ndarray, R: np.ndarray, sr: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Filter L + R into each named band ONCE. Returns {name: (L_band, R_band)}.

    Both `_phase_coherence_per_band` and `_ms_width_per_band` used to call
    `_band_filter` independently for the same bands — 20 sosfilt calls on
    multi-MB stereo signals (~5 s wasted). Sharing the filtered arrays cuts
    that in half.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, lo, hi in _PHASE_BANDS:
        out[name] = (_band_filter(L, sr, lo, hi), _band_filter(R, sr, lo, hi))
    return out


def _phase_coherence_per_band(bands_LR: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """L/R correlation per frequency band.

    Sub band correlation should be near 1.0 (sub mono); air band correlation
    can be lower (wide stereo image up top). This is the master-level
    diagnostic for "does the bass collapse on mono?" / "is the top wide enough?".
    """
    bands: dict[str, float] = {}
    for name, (L_band, R_band) in bands_LR.items():
        if np.std(L_band) > 1e-10 and np.std(R_band) > 1e-10:
            corr = float(np.corrcoef(L_band, R_band)[0, 1])
        else:
            corr = 1.0
        bands[name] = round(corr, 3)
    return bands


def _ms_width_per_band(bands_LR: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Side/mid energy ratio per frequency band.

    Mastering target: sub band side energy should be near zero (mono sub),
    mid and air can be wider. This is finer-grained than the single
    ms_width_ratio that mix_health reports.
    """
    bands: dict[str, float] = {}
    for name, (L_band, R_band) in bands_LR.items():
        mid = (L_band + R_band) * 0.5
        side = (L_band - R_band) * 0.5
        rms_m = float(np.sqrt(np.mean(mid ** 2) + 1e-12))
        rms_s = float(np.sqrt(np.mean(side ** 2) + 1e-12))
        bands[name] = round(rms_s / max(rms_m, 1e-12), 3)
    return bands


# ---------------------------------------------------------------------------
# Punch index
# ---------------------------------------------------------------------------

def _punch_index(mono: np.ndarray, sr: int) -> dict:
    """A rough perceptual punch score.

    Method: compare the loud-percentile of the short-window envelope (the
    transient peaks) to the average of the long-window envelope (the
    sustained bed). The ratio in dB indicates how far the peaks pop above
    the bed.

      - short_env[i] = RMS of 10 ms hop  -> resolves transients
      - long_env[i]  = RMS of 200 ms hop -> resolves "song-level" loudness
      - punch_db = 20*log10( percentile_90(short_env) / mean(long_env) )

    A squashed master has short_env values clustered near long_env mean →
    punch ≈ 0 dB. A transient-rich master has occasional high short_env
    values that push the 90th percentile well above the long-window mean →
    punch is meaningfully positive.

    Typical values:
      - Limited / squashed pop master: 1-3 dB
      - Modern rock master with healthy transients: 4-7 dB
      - Live recording / dynamic jazz: 8+ dB
    """
    short_hop = max(1, int(sr * 0.01))
    long_hop = max(1, int(sr * 0.2))
    n_short = len(mono) // short_hop
    n_long = len(mono) // long_hop
    if n_short < 50 or n_long < 5:
        return {"punch_db": None, "note": "signal too short"}
    short_env = np.array([
        np.sqrt(np.mean(mono[i * short_hop:(i + 1) * short_hop] ** 2) + 1e-20)
        for i in range(n_short)
    ])
    long_env = np.array([
        np.sqrt(np.mean(mono[i * long_hop:(i + 1) * long_hop] ** 2) + 1e-20)
        for i in range(n_long)
    ])
    peak_value = float(np.percentile(short_env, 90))
    bed_value = float(np.mean(long_env))
    if bed_value < 1e-12:
        return {"punch_db": None, "note": "signal too quiet"}
    return {"punch_db": round(20.0 * np.log10(peak_value / bed_value), 2)}


# ---------------------------------------------------------------------------
# Oversampled waveform true peak (no codec simulation)
# ---------------------------------------------------------------------------

def _oversampled_true_peak(stereo: np.ndarray, sr: int) -> float:
    """Measure 8x waveform true peak. This does not simulate a codec."""
    return worst_channel_true_peak_dbfs(stereo, oversample=8)


# ---------------------------------------------------------------------------
# Compression history detection
# ---------------------------------------------------------------------------

def _compression_history(stereo: np.ndarray, sr: int, meter: pyln.Meter) -> dict:
    """Try to tell whether the input has already been compressed/limited.

    BS.1770 measurements use the stereo signal (channel-weighted); crest is
    computed on the loudest channel since clipping is per-channel.
    """
    try:
        lra = float(meter.loudness_range(stereo.T))
    except Exception:
        lra = 0.0
    # Per-channel crest, take the worst (most-limited) channel
    crests = []
    peak_db_per_ch = []
    for ch in range(stereo.shape[0]):
        rms = float(np.sqrt(np.mean(stereo[ch] ** 2) + 1e-20))
        peak = float(np.max(np.abs(stereo[ch])))
        if rms > 1e-12:
            crests.append(20.0 * np.log10(peak / rms))
        peak_db_per_ch.append(20.0 * np.log10(max(peak, 1e-12)))
    crest = min(crests) if crests else 0.0
    peak_dbfs = max(peak_db_per_ch) if peak_db_per_ch else -120.0

    likely_mastered = bool(crest < 10.0 and peak_dbfs > -0.5)
    reasons = []

    if crest < 10.0:
        reasons.append(f"crest {crest:.1f} dB < 10")
    if peak_dbfs > -0.5:
        reasons.append(f"sample peak {peak_dbfs:.1f} dBFS > -0.5")

    return {
        "lra_lu": round(lra, 1),
        "crest_db": round(crest, 1),
        "sample_peak_dbfs": round(peak_dbfs, 1),
        "likely_already_mastered": likely_mastered,
        "interpretation": "Heuristic only; waveform metrics cannot establish processing history",
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Reference deck — multi-reference spectral balance
# ---------------------------------------------------------------------------

def _third_octave_psd_db(mono: np.ndarray, sr: int) -> dict:
    nperseg = min(len(mono), 32768)
    freqs, psd = welch(mono, fs=sr, nperseg=nperseg, average="mean")
    psd_db = 10.0 * np.log10(psd + 1e-20)
    out: dict[float, float] = {}
    f = 20.0
    while f <= min(sr / 2.0, 20000.0):
        lo = f / 2.0 ** (1.0 / 6.0)
        hi = f * 2.0 ** (1.0 / 6.0)
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            out[round(f, 1)] = float(np.mean(psd_db[mask]))
        f *= 2.0 ** (1.0 / 3.0)
    return out


def _reference_deck_delta(mono: np.ndarray, sr: int, refs: list[Path]) -> dict:
    """Average the references' 1/3-octave PSDs, loudness-match, and compute
    per-band delta. Returns max delta, region averages, and verdict."""
    tgt_bands = _third_octave_psd_db(mono, sr)

    ref_bands_list: list[dict] = []
    for ref_path in refs:
        ref_data, ref_sr = sf.read(str(ref_path), always_2d=True)
        ref_mono = ref_data.mean(axis=1).astype(np.float64)
        bands = _third_octave_psd_db(ref_mono, ref_sr)
        # Reference bands are stored raw; the absolute level difference is
        # cancelled further down by subtracting the mean delta, so we compare
        # spectral shape rather than loudness here.
        ref_bands_list.append({hz: db for hz, db in bands.items()})

    # Average the references' PSDs at common bands
    all_hz = set(tgt_bands)
    for r in ref_bands_list:
        all_hz &= set(r)
    common = sorted(all_hz)
    avg_ref = {hz: float(np.mean([r[hz] for r in ref_bands_list])) for hz in common}

    # Use the average reference's LUFS as the loudness-match target
    # (approximate — last ref's lufs is good enough for spectral comparison)
    deltas = []
    for hz in common:
        d = (tgt_bands[hz]) - avg_ref[hz]
        deltas.append({"hz": hz, "delta_db": round(d, 1)})

    # Normalise the overall offset (subtract mean delta) so we compare
    # spectral shape, not absolute level
    if deltas:
        mean_d = float(np.mean([d["delta_db"] for d in deltas]))
        for d in deltas:
            d["delta_db"] = round(d["delta_db"] - mean_d, 1)

    abs_max = max((abs(d["delta_db"]) for d in deltas), default=0.0)

    def region_avg(lo, hi):
        vals = [d["delta_db"] for d in deltas if lo <= d["hz"] < hi]
        return round(float(np.mean(vals)), 1) if vals else 0.0

    return {
        "n_references": len(refs),
        "references": [str(p) for p in refs],
        "region_delta_db": {
            "bottom_20_250": region_avg(20, 250),
            "mids_250_4k":   region_avg(250, 4000),
            "top_4k_20k":    region_avg(4000, 20000),
        },
        "max_band_delta_db": round(abs_max, 1),
    }


# ---------------------------------------------------------------------------
# Section verdicts
# ---------------------------------------------------------------------------

def _conformance_section(mono: np.ndarray, stereo: np.ndarray, sr: int,
                         format_preset: dict | None) -> dict:
    meter = pyln.Meter(sr)
    # BS.1770: stereo LUFS measured on the (N, 2) signal — channel-weighted
    try:
        lufs = float(meter.integrated_loudness(stereo.T))
    except Exception:
        lufs = -120.0
    try:
        lra = float(meter.loudness_range(stereo.T))
    except Exception:
        lra = 0.0
    tp = worst_channel_true_peak_dbfs(stereo)
    tp_8x = _oversampled_true_peak(stereo, sr)
    sample_peak = 20.0 * np.log10(max(np.max(np.abs(stereo)), 1e-12))

    if format_preset is None:
        return {
            "format": None,
            "codec_check": {"available": False, "note": "No codec encode/decode was performed"},
            "integrated_lufs": round(lufs, 2),
            "lra_lu": round(lra, 2),
            "true_peak_dbtp": round(tp, 2),
            "true_peak_8x_dbtp": round(tp_8x, 2),
            "sample_peak_dbfs": round(sample_peak, 2),
            "verdict": YELLOW,
            "note": "no format specified — measurements only, no conformance check",
        }

    target_lufs = format_preset["target_lufs"]
    tp_ceiling = format_preset["tp_ceiling_dbtp"]
    lufs_err = abs(lufs - target_lufs)
    lufs_v = _verdict(green=lufs_err <= 0.5, yellow=lufs_err <= 1.5)

    skip_limiter = format_preset.get("skip_limiter", False)
    tp_v = _verdict(green=tp <= tp_ceiling, yellow=tp <= tp_ceiling + 0.1)
    tp_8x_v = _verdict(green=tp_8x <= tp_ceiling, yellow=tp_8x <= tp_ceiling + 0.1)
    tp_note = "Peak safety applies even when limiting is disabled"
    # Musical loudness is advisory unless a delivery contract requires it.
    required_loudness = format_preset.get("loudness_policy") == "requirement"
    overall = lufs_v if required_loudness else GREEN
    for verdict in (tp_v, tp_8x_v):
        if verdict == RED:
            overall = RED
        elif verdict == YELLOW and overall == GREEN:
            overall = YELLOW

    return {
        "format_target_lufs": target_lufs,
        "loudness_policy": format_preset.get("loudness_policy", "preference"),
        "codec_check": {"available": False, "note": "No codec encode/decode was performed"},
        "format_tp_ceiling_dbtp": tp_ceiling,
        "format_skip_limiter": skip_limiter,
        "integrated_lufs": round(lufs, 2),
        "lufs_delta": round(lufs - target_lufs, 2),
        "lufs_verdict": lufs_v,
        "lra_lu": round(lra, 2),
        "true_peak_dbtp": round(tp, 2),
        "true_peak_verdict": tp_v,
        "true_peak_note": tp_note,
        "true_peak_8x_dbtp": round(tp_8x, 2),
        "true_peak_8x_verdict": tp_8x_v,
        "sample_peak_dbfs": round(sample_peak, 2),
        "verdict": overall,
    }


def _phase_section(L: np.ndarray, R: np.ndarray, sr: int) -> dict:
    # Filter into all bands once; both stats read from the shared cache.
    bands_LR = _filter_bands_LR(L, R, sr)
    phase = _phase_coherence_per_band(bands_LR)
    ms = _ms_width_per_band(bands_LR)

    issues = []
    # Mastering best practice: sub correlation > 0.85 (near mono)
    if phase["sub"] < 0.85:
        issues.append(f"sub band L/R correlation {phase['sub']} < 0.85 — sub may collapse on mono")
    # Air should be widish but not anti-correlated
    if phase["air"] < 0.2:
        issues.append(f"air band L/R correlation {phase['air']} < 0.2 — out-of-phase highs")
    # Sub side energy should be very low
    if ms["sub"] > 0.15:
        issues.append(f"sub band M/S width {ms['sub']} > 0.15 — significant stereo energy in the sub region")

    verdict = GREEN if not issues else (YELLOW if len(issues) == 1 else RED)
    return {
        "phase_coherence_per_band": phase,
        "ms_width_per_band": ms,
        "issues": issues,
        "verdict": verdict,
    }


def _punch_section(mono: np.ndarray, sr: int) -> dict:
    p = _punch_index(mono, sr)
    db = p.get("punch_db")
    verdict = GREEN
    note = None
    if db is None:
        return {**p, "verdict": YELLOW}
    if db < 2.0:
        verdict = YELLOW
        note = "Low transient contrast; may be intentional. Compare matched listening excerpts."
    elif db < 4.0:
        verdict = YELLOW
        note = "Low envelope contrast under a project heuristic; audition before changing dynamics"
    return {**p, "note": note, "verdict": verdict}


def _compression_history_section(stereo: np.ndarray, sr: int) -> dict:
    meter = pyln.Meter(sr)
    h = _compression_history(stereo, sr, meter)
    verdict = YELLOW if h["likely_already_mastered"] else GREEN
    note = ("input shows signs of prior compression/limiting — applying more master "
            "may flatten further") if h["likely_already_mastered"] else None
    return {**h, "note": note, "verdict": verdict}


def _reference_deck_section(mono: np.ndarray, sr: int, refs: list[Path]) -> dict:
    if not refs:
        return {"available": False, "verdict": GREEN, "note": "no reference deck supplied"}
    deck = _reference_deck_delta(mono, sr, refs)
    abs_max = deck["max_band_delta_db"]
    verdict = _verdict(green=abs_max < 2.0, yellow=abs_max < 4.0)
    return {"available": True, **deck, "verdict": verdict}


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def _render_text(report: dict) -> str:
    lines = ["MASTER HEALTH REPORT", "=" * 60, f"  Master: {report['master_file']}", ""]

    C = report["conformance"]
    lines.append("FORMAT CONFORMANCE")
    lines.append("-" * 60)
    if C.get("format_target_lufs") is None:
        lines.append("  (no format target supplied — measurements only)")
        lines.append(f"      LUFS                : {C['integrated_lufs']:+.2f}")
        lines.append(f"      LRA                 : {C['lra_lu']:.2f} LU")
        lines.append(f"      True peak (4x)      : {C['true_peak_dbtp']:+.2f} dBTP")
        lines.append(f"      True peak (8x)  : {C['true_peak_8x_dbtp']:+.2f} dBTP")
    else:
        lines.append(f"  Target: {report['format']}  ({C['format_target_lufs']} LUFS, "
                     f"{C['format_tp_ceiling_dbtp']} dBTP ceiling)")
        lines.append(f"  {C['lufs_verdict']} Integrated LUFS    : {C['integrated_lufs']:+.2f}  "
                     f"(delta {C['lufs_delta']:+.2f})")
        lines.append(f"  {C['true_peak_verdict']} True peak          : {C['true_peak_dbtp']:+.2f} dBTP")
        lines.append(f"  {C['true_peak_8x_verdict']} True peak (8x) : {C['true_peak_8x_dbtp']:+.2f} dBTP  (8x oversampled)")
        if C.get("true_peak_note"):
            lines.append(f"      note: {C['true_peak_note']}")
        lines.append(f"      LRA                : {C['lra_lu']:.2f} LU")

    P = report["phase"]
    lines.append("")
    lines.append("STEREO PHASE / WIDTH PROFILE")
    lines.append("-" * 60)
    lines.append(f"  {P['verdict']} Per-band coherence and M/S width:")
    lines.append(f"      {'band':<8}{'L/R corr':>10}{'M/S width':>12}")
    for name in ("sub", "low", "mid", "high", "air"):
        pc = P["phase_coherence_per_band"][name]
        mw = P["ms_width_per_band"][name]
        lines.append(f"      {name:<8}{pc:>10.3f}{mw:>12.3f}")
    for i in P["issues"]:
        lines.append(f"      ! {i}")

    PUNCH = report["punch"]
    lines.append("")
    lines.append("PUNCH INDEX")
    lines.append("-" * 60)
    if PUNCH.get("punch_db") is not None:
        lines.append(f"  {PUNCH['verdict']} Punch index: {PUNCH['punch_db']:+.2f} dB  "
                     f"(short-window peak vs long-window bed)")
        if PUNCH.get("note"):
            lines.append(f"      {PUNCH['note']}")
        lines.append("      Project heuristic only; compare the same passage before/after at matched loudness")
    else:
        lines.append(f"  (skipped — {PUNCH.get('note', 'unknown')})")

    H = report["compression_history"]
    lines.append("")
    lines.append("COMPRESSION HISTORY")
    lines.append("-" * 60)
    lines.append(f"  {H['verdict']} Likely already mastered: {H['likely_already_mastered']}")
    lines.append(f"      LRA {H['lra_lu']} LU, crest {H['crest_db']} dB, sample peak {H['sample_peak_dbfs']} dBFS")
    if H.get("note"):
        lines.append(f"      {H['note']}")
    if H["reasons"]:
        lines.append(f"      signals: {', '.join(H['reasons'])}")

    R = report["reference_deck"]
    lines.append("")
    lines.append("REFERENCE DECK COMPARISON")
    lines.append("-" * 60)
    if not R["available"]:
        lines.append(f"  (skipped — {R.get('note', 'no references')})")
    else:
        lines.append(f"  {R['verdict']} Averaged over {R['n_references']} reference(s)")
        rd = R["region_delta_db"]
        lines.append(f"      bottom (20-250 Hz)  : {rd['bottom_20_250']:+.1f} dB")
        lines.append(f"      mids (250-4 kHz)    : {rd['mids_250_4k']:+.1f} dB")
        lines.append(f"      top (4-20 kHz)      : {rd['top_4k_20k']:+.1f} dB")
        lines.append(f"      max single-band     : {R['max_band_delta_db']:.1f} dB")

    # Overall
    verdicts = [C["verdict"]]
    n_green = verdicts.count(GREEN)
    n_yellow = verdicts.count(YELLOW)
    n_red = verdicts.count(RED)
    overall = C["verdict"]

    lines += ["", "=" * 60,
              f"TECHNICAL CONFORMANCE: {overall}  {n_green} green, {n_yellow} yellow, {n_red} red",
              "LISTENING REVIEW: PENDING (this tool does not audition audio)"]
    if overall == GREEN:
        lines.append("  Measured technical checks passed. Codec audition and listening approval remain required.")
    elif overall == YELLOW:
        lines.append("  Review technical warnings; musical diagnostics are advisory.")
    else:
        lines.append("  Technical checks failed; correct the export before delivery.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_HEALTH_CACHE_VERSION = 3


def _health_cache_signature(master_path: Path, format_name: str | None,
                            reference_paths: list[Path] | None) -> dict:
    """Stable signature for caching a master_health run.

    Cache key = master WAV (mtime + size) + format name + reference set
    (each ref's mtime + size). If unchanged, we can replay the JSON without
    redoing band filtering, LUFS measurement, true-peak resample, etc.
    """
    stat = master_path.stat()
    refs = []
    for r in reference_paths or []:
        if r.exists():
            rs = r.stat()
            refs.append({"path": str(r), "mtime_ns": int(rs.st_mtime_ns), "size": int(rs.st_size)})
    return {
        "version": _HEALTH_CACHE_VERSION,
        "master_path": str(master_path.resolve()),
        "master_mtime_ns": int(stat.st_mtime_ns),
        "master_size": int(stat.st_size),
        "format": format_name,
        "references": refs,
    }


def master_health(master_path: Path, output_dir: Path,
                  format_name: str | None = None,
                  reference_paths: list[Path] | None = None,
                  use_cache: bool = True) -> dict:
    json_path = output_dir / f"master_health_{format_name or 'generic'}.json"
    txt_path = output_dir / f"master_health_{format_name or 'generic'}.txt"

    if use_cache and master_path.exists() and json_path.exists() and txt_path.exists():
        try:
            cached = json.loads(json_path.read_text(encoding="utf-8"))
            if cached.get("_cache") == _health_cache_signature(master_path, format_name, reference_paths):
                print(f"Reading {master_path}...  -> CACHE HIT ({json_path})", flush=True)
                # Replay the text body so the user still sees the verdict
                print(txt_path.read_text(encoding="utf-8"))
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    print(f"Reading {master_path}...", flush=True)
    data, sr = sf.read(str(master_path), always_2d=True, dtype="float32")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    if data.shape[1] != 2 or len(data) < int(sr * .4) or not np.isfinite(data).all():
        raise ValueError("Health analysis requires at least 400 ms of finite mono/stereo audio")
    stereo = data.T.astype(np.float32)
    L, R = stereo[0], stereo[1]
    mono = (L + R) * 0.5

    fmt = FORMAT_PRESETS.get(format_name) if format_name else None

    print("  [1/5] Format conformance...", flush=True)
    conformance = _conformance_section(mono, stereo, sr, fmt)
    info = sf.info(str(master_path))
    format_ok = fmt is None or (sr == fmt.get("sample_rate", sr)
                               and info.subtype == f"PCM_{fmt['bit_depth']}")
    conformance.update(sample_rate=sr, subtype=info.subtype, file_format_matches=bool(format_ok))
    if not format_ok:
        conformance["verdict"] = RED
    print("  [2/5] Per-band phase coherence and M/S width...", flush=True)
    phase = _phase_section(L, R, sr)
    print("  [3/5] Punch index...", flush=True)
    punch = _punch_section(mono, sr)
    print("  [4/5] Compression history...", flush=True)
    history = _compression_history_section(stereo, sr)
    print("  [5/5] Reference deck...", flush=True)
    refdeck = _reference_deck_section(mono, sr, reference_paths or [])

    report = {
        "master_file": str(master_path),
        "delivery_ready": False,
        "listening_review": {"status": "pending", "performed_by_tool": False},
        "assessment_scope": "Technical conformance; musical diagnostics require listening",
        "format": format_name,
        "conformance": conformance,
        "phase": phase,
        "punch": punch,
        "compression_history": history,
        "reference_deck": refdeck,
        "_cache": _health_cache_signature(master_path, format_name, reference_paths),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"master_health_{format_name or 'generic'}.json"
    txt_path = output_dir / f"master_health_{format_name or 'generic'}.txt"
    # numpy bools / scalars sneak in from the scipy/np-based checks; coerce them
    def _json_default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    text = _render_text(report)
    txt_path.write_text(text, encoding="utf-8")
    print()
    print(text)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master-level health scorecard. Run after master_mix.py.",
    )
    parser.add_argument("master", type=Path, nargs="?",
                        help="Mastered stereo WAV (omit when using --all-formats)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write master_health_<format>.{json,txt}")
    parser.add_argument("--format", choices=list(FORMAT_PRESETS), default=None,
                        help="Format target for conformance check (spotify/apple/...)")
    parser.add_argument("--all-formats", action="store_true",
                        help="Batch mode: scan --output-dir for master_<format>.wav files "
                             "and run a per-format health check on each. Useful after "
                             "master_mix --all-formats.")
    parser.add_argument("--reference", type=Path, nargs="*", default=None,
                        help="One or more reference WAVs (mastered tracks) for the deck comparison")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip the mtime-based cache and force re-analysis even if "
                             "master_health_<format>.{json,txt} exist with a matching cache key.")
    args = parser.parse_args()

    refs = [p for p in (args.reference or []) if p.exists()]
    if args.reference:
        missing = [p for p in args.reference if not p.exists()]
        for p in missing:
            print(f"WARNING: reference not found: {p}", file=sys.stderr)

    if args.all_formats:
        # Scan the output dir for master_<format>.wav files
        candidates = []
        for fmt in FORMAT_PRESETS:
            wav = args.output_dir / f"master_{fmt}.wav"
            if wav.exists():
                candidates.append((fmt, wav))
        if not candidates:
            print(json.dumps({"error": f"No master_<format>.wav files in {args.output_dir}"}),
                  file=sys.stderr)
            sys.exit(1)

        print(f"Batch master_health for {len(candidates)} format(s):", flush=True)
        for fmt, _ in candidates:
            print(f"  - master_{fmt}.wav", flush=True)

        results = []
        for fmt, wav in candidates:
            print(f"\n{'=' * 60}\n{fmt.upper()}\n{'=' * 60}", flush=True)
            r = master_health(
                master_path=wav,
                output_dir=args.output_dir,
                format_name=fmt,
                reference_paths=refs,
                use_cache=not args.no_cache,
            )
            results.append((fmt, r))

        # Cross-format overall summary
        print("\n" + "=" * 60)
        print("BATCH SUMMARY")
        print("=" * 60)
        for fmt, r in results:
            C = r["conformance"]
            lufs = C.get("integrated_lufs", "?")
            tp = C.get("true_peak_dbtp", "?")
            verdict = C.get("verdict", "?")
            target_delta = C.get("lufs_delta", "n/a")
            delta_str = f"{target_delta:+.2f}" if isinstance(target_delta, (int, float)) else "n/a"
            print(f"  {verdict} {fmt:<14}  LUFS {lufs:+.2f}  (delta {delta_str})  "
                  f"TP {tp:+.2f} dBTP")
        return

    # Single-file mode
    if args.master is None:
        parser.error("master path is required (or use --all-formats)")
    if not args.master.exists():
        print(json.dumps({"error": f"Not found: {args.master}"}), file=sys.stderr)
        sys.exit(1)

    master_health(
        master_path=args.master,
        output_dir=args.output_dir,
        format_name=args.format,
        reference_paths=refs,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
