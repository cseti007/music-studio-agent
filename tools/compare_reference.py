"""Compare a target mix against a reference mix — spectral delta and loudness report.

Outputs:
  comparison.json  — per-band dB delta, loudness delta, spectral balance, EQ recommendations
  comparison.txt   — ASCII two-sided bar chart (negative = target is thin, positive = target is bright)

Spectral comparison is level-matched on the median 1/3-octave band delta: the
target spectrum is shifted so that the median band difference is 0 dB. Unlike a
single LUFS offset, a large local difference (e.g. a presence bump) does not
shift every other band. The LUFS delta is reported separately.

EQ recommendations are generated for bands where |delta| >= --threshold (default 2.0 dB).
They are hypotheses: arrangement, tuning and production intent also change spectra.

--apply merges adjacent same-sign bands into one peak filter (centre and Q from
the run), then scales the whole set down if the combined response would exceed
+-6 dB or move any band further from the reference. The output is 32-bit float
and is not peak-normalized; its peak is reported.

Peaks are worst-channel 4x-oversampled true-peak estimates (dBTP).

Usage:
  python compare_reference.py reference.wav target_mix.wav --output-dir output/session

  python compare_reference.py ref.wav mix.wav --output-dir output/session --threshold 1.5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import sosfreqz, welch

sys.path.insert(0, str(Path(__file__).parent))
from _dsp import worst_channel_true_peak_dbfs  # noqa: E402
from _recall import record_operation  # noqa: E402
from apply_eq import _build_sos, filter_signal  # noqa: E402

# 1/3-octave center frequencies (ISO 266), 20 Hz – 20 kHz
_THIRD_OCT_BASE = 20.0
_THIRD_OCT_STEP = 2.0 ** (1.0 / 3.0)

# Spectral balance regions (for summary)
_REGIONS = [
    ("bottom",  20,   250,  "Low end (sub + bass)"),
    ("mids",   250,  4000,  "Midrange"),
    ("top",   4000, 20000,  "High end (presence + air)"),
]

# EQ filter type hints per frequency range
def _eq_filter_hint(hz: float) -> str:
    if hz <= 40:
        return "lowshelf or highpass adjustment"
    if hz <= 120:
        return "lowshelf or peak EQ"
    if hz <= 500:
        return "peak EQ"
    if hz <= 8000:
        return "peak EQ"
    return "highshelf or peak EQ"


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _finite(value) -> float | None:
    """float(value), or None when it is not a finite number."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _lufs(data: np.ndarray, sr: int) -> float | None:
    """Integrated LUFS; None for silent or too-short audio."""
    meter = pyln.Meter(sr)
    try:
        return _finite(meter.integrated_loudness(data))
    except Exception:
        return None


def _lra(data: np.ndarray, sr: int) -> float | None:
    """Loudness range in LU; None when not measurable."""
    meter = pyln.Meter(sr)
    try:
        value = _finite(meter.loudness_range(data))
    except Exception:
        return None
    return None if value is None else round(value, 1)


def _round(value: float | None, ndigits: int = 1) -> float | None:
    return None if value is None else round(value, ndigits)


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else round(a - b, 1)


def _fmt(value: float | None, spec: str = ".1f") -> str:
    return "n/a" if value is None else format(value, spec)


def _crest_factor_db(mono: np.ndarray) -> float:
    rms = np.sqrt(np.mean(mono ** 2))
    peak = np.max(np.abs(mono))
    if rms < 1e-10:
        return 0.0
    return float(20.0 * np.log10(peak / rms))


def _third_octave_psd_db(mono: np.ndarray, sr: int) -> list[dict]:
    """Compute per-band mean PSD in dB for 1/3-octave bands (Welch method)."""
    nperseg = min(len(mono), 32768)
    freqs, psd = welch(mono.astype(np.float64), fs=sr, nperseg=nperseg, average="mean")
    psd_db = 10.0 * np.log10(psd + 1e-20)

    centers: list[float] = []
    f = _THIRD_OCT_BASE
    while f <= min(sr / 2.0, 20000.0):
        centers.append(f)
        f *= _THIRD_OCT_STEP

    bands = []
    for fc in centers:
        lo = fc / 2.0 ** (1.0 / 6.0)
        hi = fc * 2.0 ** (1.0 / 6.0)
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            bands.append({"hz": round(fc, 1), "db": round(float(np.mean(psd_db[mask])), 2)})

    return bands


def _region_mean_db(bands: list[dict], lo_hz: float, hi_hz: float) -> float:
    vals = [b["db"] for b in bands if lo_hz <= b["hz"] < hi_hz]
    return round(float(np.mean(vals)), 1) if vals else 0.0


def _matched_delta_bands(ref_bands: list[dict], tgt_bands: list[dict]) -> tuple[list[dict], float]:
    """Per-band target-minus-reference deltas after median level matching.

    Returns (delta_bands, offset_db): offset_db is added to the target levels
    so that the median band delta is 0 dB.
    """
    ref_by_hz = {b["hz"]: b["db"] for b in ref_bands}
    tgt_by_hz = {b["hz"]: b["db"] for b in tgt_bands}
    common_hz = sorted(set(ref_by_hz) & set(tgt_by_hz))
    if not common_hz:
        return [], 0.0
    offset = -float(np.median([tgt_by_hz[hz] - ref_by_hz[hz] for hz in common_hz]))
    delta_bands = []
    for hz in common_hz:
        ref_db = ref_by_hz[hz]
        tgt_db = tgt_by_hz[hz] + offset
        delta_bands.append({
            "hz": hz,
            "reference_db": round(ref_db, 1),
            "target_db": round(tgt_db, 1),
            "delta_db": round(tgt_db - ref_db, 1),
        })
    return delta_bands, offset


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _recommendations(
    delta_bands: list[dict],
    threshold_db: float,
) -> list[str]:
    recs = []
    for b in delta_bands:
        d = b["delta_db"]
        hz = b["hz"]
        if abs(d) < threshold_db:
            continue
        hz_label = f"{hz:.0f} Hz" if hz < 1000 else f"{hz / 1000:.2f} kHz"
        hint = _eq_filter_hint(hz)
        if d > 0:
            recs.append(
                f"{hz_label}: target +{d:.1f} dB above reference — "
                f"cut {abs(d):.0f} dB ({hint})"
            )
        else:
            recs.append(
                f"{hz_label}: target {d:.1f} dB below reference — "
                f"boost {abs(d):.0f} dB ({hint})"
            )
    return recs


# Maximum number of peak EQ filters --apply will inject. The strongest deltas
# win; more than ~6 filters chained tend to phase-smear rather than help.
_APPLY_MAX_FILTERS = 6
# Cap individual filter gain and the combined response of the whole set.
_APPLY_MAX_FILTER_DB = 6.0
_APPLY_MAX_TOTAL_DB = 6.0
# A band may end up at most this much further from the reference than before
# (prediction vs. measurement tolerance); otherwise the set is scaled down.
_APPLY_WORSEN_TOLERANCE_DB = 0.25
_APPLY_Q_RANGE = (0.3, 4.32)  # 4.32 = one 1/3-octave band


def _filters_from_delta(
    delta_bands: list[dict],
    threshold_db: float,
    max_filters: int = _APPLY_MAX_FILTERS,
) -> list[dict]:
    """Convert per-band deltas into a list of peak-EQ filter specs.

    Adjacent bands whose |delta| >= threshold_db with the same sign form one
    run and get one filter, the inverse of the run's largest delta (a target
    +3 dB above reference -> a -3 dB cut), clamped to +-_APPLY_MAX_FILTER_DB.
    The centre is that band; the bandwidth spans the contiguous bands around
    it whose |delta| is at least half the peak (the RBJ peaking filter's
    bandwidth is defined at half its dB gain), one 1/3 octave per band.

    Filters are sorted by |delta| descending; only the top `max_filters`
    are returned.
    """
    def sign(i: int) -> float:
        return float(np.sign(delta_bands[i]["delta_db"]))

    runs: list[list[int]] = []
    for i, b in enumerate(delta_bands):
        if abs(b["delta_db"]) < threshold_db:
            continue
        if runs and runs[-1][-1] == i - 1 and sign(i) == sign(i - 1):
            runs[-1].append(i)
        else:
            runs.append([i])

    ranked = []
    for run in runs:
        peak = max(run, key=lambda i: abs(delta_bands[i]["delta_db"]))
        half = abs(delta_bands[peak]["delta_db"]) / 2.0
        lo = hi = peak
        while lo > 0 and sign(lo - 1) == sign(peak) and abs(delta_bands[lo - 1]["delta_db"]) >= half:
            lo -= 1
        while (hi < len(delta_bands) - 1 and sign(hi + 1) == sign(peak)
               and abs(delta_bands[hi + 1]["delta_db"]) >= half):
            hi += 1
        bw_oct = (hi - lo + 1) / 3.0
        q = float(np.clip(1.0 / (2.0 * np.sinh(np.log(2.0) / 2.0 * bw_oct)), *_APPLY_Q_RANGE))
        b = delta_bands[peak]
        db = max(-_APPLY_MAX_FILTER_DB, min(_APPLY_MAX_FILTER_DB, -b["delta_db"]))
        ranked.append((abs(b["delta_db"]), {
            "type": "peak",
            "hz": float(b["hz"]),
            "q": round(q, 2),
            "db": round(db, 1),
            "_auto": (f"compare_reference: target was {b['delta_db']:+.1f} dB vs ref at {b['hz']} Hz "
                      f"({hi - lo + 1} band(s) above half gain)"),
        }))
    ranked.sort(key=lambda item: -item[0])
    return [f for _, f in ranked[:max_filters]]


def _response_db(filters: list[dict], hz: np.ndarray, sr: int, phase: str = "minimum") -> np.ndarray:
    """Combined magnitude response (dB) of the filter chain at the given frequencies."""
    total = np.zeros(len(hz))
    passes = 2 if phase == "zero" else 1  # zero phase: half gain, applied twice
    for f in filters:
        spec = {**f, "db": float(f["db"]) / passes}
        _, h = sosfreqz(_build_sos(spec, sr), worN=hz, fs=sr)
        total += passes * 20.0 * np.log10(np.abs(h) + 1e-12)
    return total


def _fit_filter_gains(
    filters: list[dict],
    delta_bands: list[dict],
    sr: int,
    phase: str = "minimum",
) -> tuple[list[dict], float]:
    """Scale the whole filter set down until it is safe to apply.

    Safe = combined response within +-_APPLY_MAX_TOTAL_DB and no band moved
    further from the reference than before (within _APPLY_WORSEN_TOLERANCE_DB).
    Returns (scaled filters, scale); ([], 0.0) when no tried scale is safe.
    """
    if not filters:
        return [], 1.0
    hz = np.array([b["hz"] for b in delta_bands], dtype=float)
    delta = np.array([b["delta_db"] for b in delta_bands], dtype=float)
    valid = hz < sr / 2.0
    hz, delta = hz[valid], delta[valid]
    for scale in (1.0, 0.75, 0.5, 0.25):
        scaled = [{**f, "db": round(f["db"] * scale, 1)} for f in filters]
        resp = _response_db(scaled, hz, sr, phase)
        if np.max(np.abs(resp)) > _APPLY_MAX_TOTAL_DB + 0.05:
            continue
        if np.all(np.abs(delta + resp) <= np.abs(delta) + _APPLY_WORSEN_TOLERANCE_DB):
            return scaled, scale
    return [], 0.0


@record_operation("compare_reference")
def _apply_eq_to_target(
    target_path: Path,
    output_path: Path,
    filters: list[dict],
    phase: str = "minimum",
) -> dict:
    """Run the generated EQ filter chain on the target file via apply_eq.

    Reuses apply_eq's biquad implementations. Output goes to `output_path`
    directly (we bypass apply_eq's stem-based naming). The output is 32-bit
    float and keeps its level; a peak above 0 dBFS is reported, not hidden.
    """
    data, sr = sf.read(str(target_path), always_2d=True)

    out_channels = []
    for ch in range(data.shape[1]):
        signal = data[:, ch].astype(np.float64)
        for f in filters:
            signal = filter_signal(signal, sr, f, phase)
        out_channels.append(signal)
    output_data = np.stack(out_channels, axis=1)

    peak = float(np.max(np.abs(output_data)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), output_data, sr, subtype="FLOAT")
    return {
        "input": str(target_path),
        "output": str(output_path),
        "output_peak_dbfs": round(20.0 * np.log10(max(peak, 1e-10)), 2),
    }


def _ascii_chart(delta_bands: list[dict], threshold_db: float) -> str:
    if not delta_bands:
        return ""

    max_delta = max(abs(b["delta_db"]) for b in delta_bands)
    bar_width = 24  # chars per side
    scale = bar_width / max(max_delta, threshold_db)

    lines = [
        "",
        "SPECTRAL DELTA  (target vs. reference, level-matched on median band delta)",
        f"  {'─' * bar_width}╫{'─' * bar_width}",
        f"  {'below reference':>{bar_width}}  {'above reference'}",
        f"  {'─' * bar_width}╫{'─' * bar_width}",
    ]

    for b in delta_bands:
        hz = b["hz"]
        d = b["delta_db"]
        hz_label = f"{hz:6.0f} Hz" if hz < 1000 else f"{hz / 1000:5.2f} kHz"
        bar_len = int(min(abs(d) * scale, bar_width))
        flag = " [!]" if abs(d) >= threshold_db else ""

        if d < 0:
            left  = "═" * bar_len
            right = ""
            row = f"  {left:>{bar_width}}╫{right:<{bar_width}}  {hz_label}  {d:+.1f} dB{flag}"
        elif d > 0:
            left  = ""
            right = "═" * bar_len
            row = f"  {left:>{bar_width}}╫{right:<{bar_width}}  {hz_label}  {d:+.1f} dB{flag}"
        else:
            row = f"  {'':>{bar_width}}╫{'':>{bar_width}}  {hz_label}   0.0 dB"

        lines.append(row)

    lines.append(f"  {'─' * bar_width}╫{'─' * bar_width}")
    lines.append(f"  [!] = delta >= {threshold_db:.1f} dB threshold — flagged for EQ correction")
    return "\n".join(lines)


def _summary_text(report: dict, chart: str) -> str:
    loud = report["loudness"]
    bal = report["spectral_balance"]
    recs = report["eq_recommendations"]

    lines = [
        "COMPARISON REPORT",
        "=" * 60,
        f"  Reference : {report['reference']}",
        f"  Target    : {report['target']}",
        "",
        "LOUDNESS",
        "-" * 40,
        f"  Integrated LUFS : ref {_fmt(loud['reference_lufs'])}  |  target {_fmt(loud['target_lufs'])}  |  delta {_fmt(loud['delta_lufs'], '+.1f')} dB",
        f"  LRA             : ref {_fmt(loud['reference_lra'])} LU  |  target {_fmt(loud['target_lra'])} LU  |  delta {_fmt(loud['delta_lra'], '+.1f')} LU",
        f"  True peak       : ref {loud['reference_peak_dbfs']:.1f} dBTP  |  target {loud['target_peak_dbfs']:.1f} dBTP  (worst channel)",
        f"  Crest factor    : ref {loud['reference_crest_db']:.1f} dB  |  target {loud['target_crest_db']:.1f} dB  |  delta {loud['delta_crest_db']:+.1f} dB",
        "",
        f"SPECTRAL BALANCE (level-matched on median band delta, offset {report['loudness_match_offset_db']:+.1f} dB)",
        "-" * 40,
    ]

    for key, label in [("bottom", "Low  (20-250 Hz)"), ("mids", "Mid  (250-4 kHz)"), ("top", "High (4-20 kHz)")]:
        r = bal[key]
        lines.append(f"  {label:<18}  ref {r['reference_db']:+.1f} dB  |  target {r['target_db']:+.1f} dB  |  delta {r['delta_db']:+.1f} dB")

    lines.append(chart)

    if recs:
        lines += ["", "EQ RECOMMENDATIONS", "-" * 40]
        for rec in recs:
            lines.append(f"  [+] {rec}")
    else:
        lines += ["", "EQ RECOMMENDATIONS", "-" * 40, "  No significant tonal differences above threshold."]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def compare_reference(
    reference_path: Path,
    target_path: Path,
    output_dir: Path,
    threshold_db: float = 2.0,
    apply_output: Path | None = None,
    apply_phase: str = "minimum",
) -> dict:
    ref_data, ref_sr = sf.read(str(reference_path), always_2d=True)
    tgt_data, tgt_sr = sf.read(str(target_path), always_2d=True)

    ref_mono = ref_data.mean(axis=1).astype(np.float64)
    tgt_mono = tgt_data.mean(axis=1).astype(np.float64)

    print("Computing loudness metrics...", flush=True)
    ref_lufs = _lufs(ref_data, ref_sr)
    tgt_lufs = _lufs(tgt_data, tgt_sr)
    ref_lra  = _lra(ref_data, ref_sr)
    tgt_lra  = _lra(tgt_data, tgt_sr)
    ref_peak = worst_channel_true_peak_dbfs(ref_data)
    tgt_peak = worst_channel_true_peak_dbfs(tgt_data)
    ref_crest = _crest_factor_db(ref_mono)
    tgt_crest = _crest_factor_db(tgt_mono)

    print("Computing 1/3-octave frequency response...", flush=True)
    ref_bands = _third_octave_psd_db(ref_mono, ref_sr)
    tgt_bands = _third_octave_psd_db(tgt_mono, tgt_sr)

    # Level-match on the median band delta (robust to a few large local deltas)
    delta_bands, match_offset = _matched_delta_bands(ref_bands, tgt_bands)

    # Spectral balance per region (level-matched)
    tgt_bands_matched = [{"hz": b["hz"], "db": b["db"] + match_offset} for b in tgt_bands]
    spectral_balance = {}
    for key, lo, hi, _ in _REGIONS:
        ref_region_db = _region_mean_db(ref_bands, lo, hi)
        tgt_region_db = _region_mean_db(tgt_bands_matched, lo, hi)
        spectral_balance[key] = {
            "hz_range": [lo, hi],
            "reference_db": ref_region_db,
            "target_db": tgt_region_db,
            "delta_db": round(tgt_region_db - ref_region_db, 1),
        }

    recs = _recommendations(delta_bands, threshold_db)
    auto_filters, auto_scale = _fit_filter_gains(
        _filters_from_delta(delta_bands, threshold_db), delta_bands, tgt_sr, apply_phase,
    )

    report = {
        "reference": str(reference_path),
        "target": str(target_path),
        "threshold_db": threshold_db,
        "level_match_method": "median_band_delta",
        "loudness_match_offset_db": round(match_offset, 2),
        "auto_eq_filters": auto_filters,
        "auto_eq_scale": auto_scale,
        "loudness": {
            "reference_lufs": _round(ref_lufs),
            "target_lufs": _round(tgt_lufs),
            "delta_lufs": _diff(tgt_lufs, ref_lufs),
            "reference_lra": ref_lra,
            "target_lra": tgt_lra,
            "delta_lra": _diff(tgt_lra, ref_lra),
            "reference_peak_dbfs": round(ref_peak, 1),
            "target_peak_dbfs": round(tgt_peak, 1),
            "delta_peak_db": round(tgt_peak - ref_peak, 1),
            "reference_crest_db": round(ref_crest, 1),
            "target_crest_db": round(tgt_crest, 1),
            "delta_crest_db": round(tgt_crest - ref_crest, 1),
        },
        "spectral_balance": spectral_balance,
        "third_octave_delta": delta_bands,
        "eq_recommendations": recs,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    txt_path  = output_dir / "comparison.txt"

    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    chart = _ascii_chart(delta_bands, threshold_db)
    summary = _summary_text(report, chart)
    txt_path.write_text(summary, encoding="utf-8")

    print(summary)

    # Auto-apply mode: bake the inverse-delta EQ chain into a new WAV.
    if apply_output is not None:
        if not auto_filters:
            reason = ("no filter set passed the combined-response safety check"
                      if auto_scale == 0.0 else f"no bands exceed threshold {threshold_db} dB")
            print(f"\n--apply requested but {reason} — nothing to do.")
        else:
            print(f"\n--apply: writing EQ-matched output to {apply_output}")
            print(f"  filters ({len(auto_filters)}, phase={apply_phase}):")
            for f in auto_filters:
                hz_lbl = f"{f['hz']:6.0f} Hz" if f['hz'] < 1000 else f"{f['hz']/1000:5.2f} kHz"
                print(f"    {hz_lbl}  Q={f['q']:.1f}  {f['db']:+.1f} dB")
            applied = _apply_eq_to_target(target_path, apply_output, auto_filters, phase=apply_phase)
            report["apply_output"] = str(apply_output)
            report["apply_phase"] = apply_phase
            report["apply_output_peak_dbfs"] = applied["output_peak_dbfs"]
            if applied["output_peak_dbfs"] > 0.0:
                print(f"  NOTE: output sample peak {applied['output_peak_dbfs']:+.2f} dBFS (float file, not normalized)")
            # Update the json report with the apply info
            json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare target mix spectral balance against a reference mix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("reference", type=Path, help="Reference mix WAV file")
    parser.add_argument("target",    type=Path, help="Target mix WAV file to evaluate")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--threshold", type=float, default=2.0, metavar="DB",
        help="dB threshold for flagging and generating EQ recommendations (default: 2.0)",
    )
    parser.add_argument(
        "--apply", type=Path, metavar="WAV",
        help="Auto-generate inverse-delta peak EQ filters from the spectral "
             "comparison and write a corrected 32-bit float WAV to this path "
             "(not peak-normalized). Adjacent same-sign bands share one filter; "
             "up to 6 filters, each and their combined response capped at ±6 dB, "
             "scaled down if any band would move further from the reference.",
    )
    parser.add_argument(
        "--apply-phase", choices=["minimum", "zero"], default="minimum",
        help="Phase response for --apply (default: minimum; use zero for mastering chains).",
    )
    args = parser.parse_args()

    for p in (args.reference, args.target):
        if not p.exists():
            print(json.dumps({"error": f"Not found: {p}"}), file=sys.stderr)
            sys.exit(1)

    compare_reference(
        args.reference, args.target, args.output_dir,
        threshold_db=args.threshold,
        apply_output=args.apply,
        apply_phase=args.apply_phase,
    )


if __name__ == "__main__":
    main()
