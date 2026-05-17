"""Compare a target mix against a reference mix — spectral delta and loudness report.

Outputs:
  comparison.json  — per-band dB delta, loudness delta, spectral balance, EQ recommendations
  comparison.txt   — ASCII two-sided bar chart (negative = target is thin, positive = target is bright)

Spectral comparison is loudness-matched: the LUFS delta between reference and target is
applied as a uniform offset to the target PSD before computing per-band differences.
This isolates tonal balance from overall level — the LUFS delta is reported separately.

EQ recommendations are generated for bands where |delta| >= --threshold (default 2.0 dB).

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
from scipy.signal import welch

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

def _lufs(data: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    try:
        return float(meter.integrated_loudness(data))
    except Exception:
        return -120.0


def _lra(data: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    try:
        return round(float(meter.loudness_range(data)), 1)
    except Exception:
        return 0.0


def _true_peak_dbfs(mono: np.ndarray) -> float:
    return float(20.0 * np.log10(max(float(np.max(np.abs(mono))), 1e-10)))


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


def _ascii_chart(delta_bands: list[dict], threshold_db: float) -> str:
    if not delta_bands:
        return ""

    max_delta = max(abs(b["delta_db"]) for b in delta_bands)
    bar_width = 24  # chars per side
    scale = bar_width / max(max_delta, threshold_db)

    lines = [
        "",
        "SPECTRAL DELTA  (target vs. reference, loudness-matched)",
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
        f"  Integrated LUFS : ref {loud['reference_lufs']:.1f}  |  target {loud['target_lufs']:.1f}  |  delta {loud['delta_lufs']:+.1f} dB",
        f"  LRA             : ref {loud['reference_lra']:.1f} LU  |  target {loud['target_lra']:.1f} LU  |  delta {loud['delta_lra']:+.1f} LU",
        f"  True peak       : ref {loud['reference_peak_dbfs']:.1f} dBFS  |  target {loud['target_peak_dbfs']:.1f} dBFS",
        f"  Crest factor    : ref {loud['reference_crest_db']:.1f} dB  |  target {loud['target_crest_db']:.1f} dB  |  delta {loud['delta_crest_db']:+.1f} dB",
        "",
        "SPECTRAL BALANCE (loudness-matched)",
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
    ref_peak = _true_peak_dbfs(ref_mono)
    tgt_peak = _true_peak_dbfs(tgt_mono)
    ref_crest = _crest_factor_db(ref_mono)
    tgt_crest = _crest_factor_db(tgt_mono)

    # Offset to add to target PSD to level-match to reference LUFS
    lufs_offset = ref_lufs - tgt_lufs if ref_lufs > -100 and tgt_lufs > -100 else 0.0

    print("Computing 1/3-octave frequency response...", flush=True)
    ref_bands = _third_octave_psd_db(ref_mono, ref_sr)
    tgt_bands = _third_octave_psd_db(tgt_mono, tgt_sr)

    # Align bands by center frequency (both should match since same 1/3-oct grid)
    ref_by_hz = {b["hz"]: b["db"] for b in ref_bands}
    tgt_by_hz = {b["hz"]: b["db"] for b in tgt_bands}
    common_hz = sorted(set(ref_by_hz) & set(tgt_by_hz))

    delta_bands = []
    for hz in common_hz:
        ref_db = ref_by_hz[hz]
        tgt_db = tgt_by_hz[hz] + lufs_offset  # level-matched
        delta_bands.append({
            "hz": hz,
            "reference_db": round(ref_db, 1),
            "target_db": round(tgt_db, 1),
            "delta_db": round(tgt_db - ref_db, 1),
        })

    # Spectral balance per region (level-matched)
    tgt_bands_matched = [{"hz": b["hz"], "db": b["db"] + lufs_offset} for b in tgt_bands]
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

    report = {
        "reference": str(reference_path),
        "target": str(target_path),
        "threshold_db": threshold_db,
        "loudness_match_offset_db": round(lufs_offset, 2),
        "loudness": {
            "reference_lufs": round(ref_lufs, 1),
            "target_lufs": round(tgt_lufs, 1),
            "delta_lufs": round(tgt_lufs - ref_lufs, 1),
            "reference_lra": ref_lra,
            "target_lra": tgt_lra,
            "delta_lra": round(tgt_lra - ref_lra, 1),
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

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    chart = _ascii_chart(delta_bands, threshold_db)
    summary = _summary_text(report, chart)
    txt_path.write_text(summary, encoding="utf-8")

    print(summary)
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
    args = parser.parse_args()

    for p in (args.reference, args.target):
        if not p.exists():
            print(json.dumps({"error": f"Not found: {p}"}), file=sys.stderr)
            sys.exit(1)

    compare_reference(args.reference, args.target, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()
