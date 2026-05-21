"""Master-level health scorecard.

Distinct from mix_health.py:
  - mix_health scores a finished STEM mix (LUFS, M/S width avg, masking pairs,
    stem pumping)
  - master_health scores a finished STEREO MASTER (format conformance,
    per-band phase coherence, M/S width profile, punch index, codec-ISP
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
from scipy.signal import butter, resample_poly, sosfilt, welch

sys.path.insert(0, str(Path(__file__).parent))
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


def _phase_coherence_per_band(L: np.ndarray, R: np.ndarray, sr: int) -> dict:
    """L/R correlation per frequency band.

    Sub band correlation should be near 1.0 (sub mono); air band correlation
    can be lower (wide stereo image up top). This is the master-level
    diagnostic for "does the bass collapse on mono?" / "is the top wide enough?".
    """
    bands: dict[str, float] = {}
    for name, lo, hi in _PHASE_BANDS:
        L_band = _band_filter(L, sr, lo, hi)
        R_band = _band_filter(R, sr, lo, hi)
        if np.std(L_band) > 1e-10 and np.std(R_band) > 1e-10:
            corr = float(np.corrcoef(L_band, R_band)[0, 1])
        else:
            corr = 1.0
        bands[name] = round(corr, 3)
    return bands


def _ms_width_per_band(L: np.ndarray, R: np.ndarray, sr: int) -> dict:
    """Side/mid energy ratio per frequency band.

    Mastering target: sub band side energy should be near zero (mono sub),
    mid and air can be wider. This is finer-grained than the single
    ms_width_ratio that mix_health reports.
    """
    bands: dict[str, float] = {}
    for name, lo, hi in _PHASE_BANDS:
        L_band = _band_filter(L, sr, lo, hi)
        R_band = _band_filter(R, sr, lo, hi)
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
# Codec-ISP simulator (proxy via heavy oversampling)
# ---------------------------------------------------------------------------

def _codec_isp_estimate(stereo: np.ndarray, sr: int) -> float:
    """Estimate the post-codec inter-sample peak.

    A true codec-roundtrip simulator (Ogg Vorbis encode + decode + measure)
    needs ffmpeg or similar; here we use a conservative proxy: 8x oversampled
    true peak (vs. the 4x used elsewhere). Lossy encoding tends to overshoot
    the original by 0.5-1.5 dB on transient-rich material, and 8x sampling
    catches most of that.
    """
    up_l = resample_poly(stereo[0], 8, 1)
    up_r = resample_poly(stereo[1], 8, 1)
    tp = max(float(np.max(np.abs(up_l))), float(np.max(np.abs(up_r))))
    return 20.0 * np.log10(max(tp, 1e-12))


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

    likely_mastered = lra < 4.0 or crest < 10.0 or peak_dbfs > -0.5
    reasons = []
    if lra < 4.0:
        reasons.append(f"LRA {lra:.1f} LU < 4")
    if crest < 10.0:
        reasons.append(f"crest {crest:.1f} dB < 10")
    if peak_dbfs > -0.5:
        reasons.append(f"sample peak {peak_dbfs:.1f} dBFS > -0.5")

    return {
        "lra_lu": round(lra, 1),
        "crest_db": round(crest, 1),
        "sample_peak_dbfs": round(peak_dbfs, 1),
        "likely_already_mastered": likely_mastered,
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
    meter = pyln.Meter(sr)
    try:
        tgt_lufs = float(meter.integrated_loudness(mono))
    except Exception:
        tgt_lufs = -120.0
    tgt_bands = _third_octave_psd_db(mono, sr)

    ref_bands_list: list[dict] = []
    for ref_path in refs:
        ref_data, ref_sr = sf.read(str(ref_path), always_2d=True)
        ref_mono = ref_data.mean(axis=1).astype(np.float64)
        meter_r = pyln.Meter(ref_sr)
        try:
            ref_lufs = float(meter_r.integrated_loudness(ref_mono))
        except Exception:
            ref_lufs = -120.0
        offset = ref_lufs - tgt_lufs if ref_lufs > -100 and tgt_lufs > -100 else 0.0
        bands = _third_octave_psd_db(ref_mono, ref_sr)
        # Apply LUFS-match offset to target so reference bands are absolute
        # (we'll compare matched target to raw reference)
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
    from analyze import _true_peak_dbfs
    # True peak: worst of L/R, sample-summed peak after oversampling
    tp = max(_true_peak_dbfs(stereo[0]), _true_peak_dbfs(stereo[1]))
    tp_codec = _codec_isp_estimate(stereo, sr)
    sample_peak = 20.0 * np.log10(max(np.max(np.abs(stereo)), 1e-12))

    if format_preset is None:
        return {
            "format": None,
            "integrated_lufs": round(lufs, 2),
            "lra_lu": round(lra, 2),
            "true_peak_dbtp": round(tp, 2),
            "codec_isp_estimate_dbtp": round(tp_codec, 2),
            "sample_peak_dbfs": round(sample_peak, 2),
            "verdict": YELLOW,
            "note": "no format specified — measurements only, no conformance check",
        }

    target_lufs = format_preset["target_lufs"]
    tp_ceiling = format_preset["tp_ceiling_dbtp"]
    lufs_err = abs(lufs - target_lufs)
    lufs_v = _verdict(green=lufs_err <= 0.5, yellow=lufs_err <= 1.5)

    # Vinyl / no-limiter formats deliberately skip the limiter (cutter does
    # its own limiting). A TP > ceiling is expected on the WAV — yellow at
    # worst, never red, with a note that this is by design.
    skip_limiter = format_preset.get("skip_limiter", False)
    if skip_limiter:
        tp_v = YELLOW if tp > tp_ceiling + 3.0 else GREEN
        tp_codec_v = YELLOW if tp_codec > tp_ceiling + 3.0 else GREEN
        tp_note = "no-limiter format (vinyl/etc) — TP ceiling enforced downstream by the cutter / mastering plant"
    else:
        tp_v = _verdict(green=tp <= tp_ceiling, yellow=tp <= tp_ceiling + 1.0)
        tp_codec_v = _verdict(green=tp_codec <= tp_ceiling + 0.5,
                              yellow=tp_codec <= tp_ceiling + 1.5)
        tp_note = None

    overall = lufs_v
    for v in (tp_v, tp_codec_v):
        if v == RED:
            overall = RED
        elif v == YELLOW and overall == GREEN:
            overall = YELLOW

    return {
        "format_target_lufs": target_lufs,
        "format_tp_ceiling_dbtp": tp_ceiling,
        "format_skip_limiter": skip_limiter,
        "integrated_lufs": round(lufs, 2),
        "lufs_delta": round(lufs - target_lufs, 2),
        "lufs_verdict": lufs_v,
        "lra_lu": round(lra, 2),
        "true_peak_dbtp": round(tp, 2),
        "true_peak_verdict": tp_v,
        "true_peak_note": tp_note,
        "codec_isp_estimate_dbtp": round(tp_codec, 2),
        "codec_isp_verdict": tp_codec_v,
        "sample_peak_dbfs": round(sample_peak, 2),
        "verdict": overall,
    }


def _phase_section(L: np.ndarray, R: np.ndarray, sr: int) -> dict:
    phase = _phase_coherence_per_band(L, R, sr)
    ms = _ms_width_per_band(L, R, sr)

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
        verdict = RED
        note = "squashed — transients are buried in the bed (typical of over-limited masters)"
    elif db < 4.0:
        verdict = YELLOW
        note = "low punch — borderline; may sound fatiguing"
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
        lines.append(f"      Codec-ISP est (8x)  : {C['codec_isp_estimate_dbtp']:+.2f} dBTP")
    else:
        lines.append(f"  Target: {report['format']}  ({C['format_target_lufs']} LUFS, "
                     f"{C['format_tp_ceiling_dbtp']} dBTP ceiling)")
        lines.append(f"  {C['lufs_verdict']} Integrated LUFS    : {C['integrated_lufs']:+.2f}  "
                     f"(delta {C['lufs_delta']:+.2f})")
        lines.append(f"  {C['true_peak_verdict']} True peak          : {C['true_peak_dbtp']:+.2f} dBTP")
        lines.append(f"  {C['codec_isp_verdict']} Codec-ISP estimate : {C['codec_isp_estimate_dbtp']:+.2f} dBTP  (8x oversampled)")
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
        lines.append("      reference values: limited pop ~3-5 dB, healthy rock ~6-9 dB, dynamic >10 dB")
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
    verdicts = [C["verdict"], P["verdict"], PUNCH["verdict"], H["verdict"]]
    if R["available"]:
        verdicts.append(R["verdict"])
    n_green = verdicts.count(GREEN)
    n_yellow = verdicts.count(YELLOW)
    n_red = verdicts.count(RED)
    overall = GREEN if n_red == 0 and n_yellow <= 1 else (YELLOW if n_red == 0 else RED)

    lines += ["", "=" * 60,
              f"OVERALL: {overall}  {n_green} green, {n_yellow} yellow, {n_red} red"]
    if overall == GREEN:
        lines.append("  Master is delivery-ready.")
    elif overall == YELLOW:
        lines.append("  Master is close — address yellow items.")
    else:
        lines.append("  Master needs work — red items must be fixed before delivery.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def master_health(master_path: Path, output_dir: Path,
                  format_name: str | None = None,
                  reference_paths: list[Path] | None = None) -> dict:
    print(f"Reading {master_path}...", flush=True)
    data, sr = sf.read(str(master_path), always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    stereo = data.T.astype(np.float64)
    L, R = stereo[0], stereo[1]
    mono = (L + R) * 0.5

    fmt = FORMAT_PRESETS.get(format_name) if format_name else None

    print("  [1/5] Format conformance...", flush=True)
    conformance = _conformance_section(mono, stereo, sr, fmt)
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
        "format": format_name,
        "conformance": conformance,
        "phase": phase,
        "punch": punch,
        "compression_history": history,
        "reference_deck": refdeck,
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
    )


if __name__ == "__main__":
    main()
