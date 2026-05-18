"""Grade a mix against a named style profile — quantitative answer to
"is this a modern_rock mix" without needing a reference track.

Reads a stereo mix.wav, measures integrated LUFS, LRA, crest factor, and
5-band spectral RMS, then compares each value to the targets in a style
profile JSON. Emits an overall score (0-100), a traffic-light verdict
(GREEN / YELLOW / RED), and per-check deltas.

Tonal balance is measured on a LUFS-normalized copy of the mix (normalised
to the profile's integrated LUFS target). This way the spectral check is
loudness-independent — a quiet mix at -23 LUFS and a loud one at -8 LUFS
get the same band-RMS reading if their tonal balance is actually identical.

Usage:
  python tools/style_check.py mix.wav --style modern_rock --output-dir output/<session>/analysis

Outputs:
  style_check.json  — full structured report
  style_check.txt   — ASCII bar chart + per-check deltas + verdict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt


_STYLE_PROFILES_DIR = Path(__file__).resolve().parent / "style_profiles"

# Band edges must match analyze.py / mix_health.py
_BANDS = {
    "sub_60hz":       (0,    60),
    "low_60_250hz":   (60,   250),
    "mid_250_2khz":   (250,  2000),
    "high_2_8khz":    (2000, 8000),
    "air_8khz_plus":  (8000, 20000),
}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _band_rms_db(signal: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    high_hz = min(high_hz, sr / 2 - 1)
    if low_hz <= 0:
        sos = butter(4, high_hz, btype="low", fs=sr, output="sos")
    else:
        sos = butter(4, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    filt = sosfilt(sos, signal)
    rms = float(np.sqrt(np.mean(filt ** 2) + 1e-12))
    return 20.0 * np.log10(max(rms, 1e-10))


def _crest_factor_db(signal: np.ndarray) -> float:
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(signal ** 2) + 1e-12))
    if rms < 1e-9 or peak < 1e-9:
        return 0.0
    return round(20.0 * np.log10(peak / rms), 1)


def _normalize_to_lufs(data: np.ndarray, sr: int, current_lufs: float, target_lufs: float) -> np.ndarray:
    if current_lufs < -100.0:
        return data
    gain_db = target_lufs - current_lufs
    return data * (10.0 ** (gain_db / 20.0))


# ---------------------------------------------------------------------------
# Profile / verdict logic
# ---------------------------------------------------------------------------

def list_profiles() -> list[str]:
    return sorted(p.stem for p in _STYLE_PROFILES_DIR.glob("*.json"))


def load_profile(name: str) -> dict:
    path = _STYLE_PROFILES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Style profile not found: {name}.\n"
            f"Available profiles: {', '.join(list_profiles())}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _grade_value(measured: float, target: float, tolerance: float) -> tuple[str, float]:
    """Return (verdict_color, severity) where severity is 0..1+ (>1 = red).

    GREEN  if |delta| <= tolerance
    YELLOW if tolerance < |delta| <= 1.5 * tolerance
    RED    if |delta| > 1.5 * tolerance
    """
    delta = abs(measured - target)
    severity = delta / max(tolerance, 0.1)
    if severity <= 1.0:
        return ("GREEN", severity)
    if severity <= 1.5:
        return ("YELLOW", severity)
    return ("RED", severity)


def _grade_range(measured: float, target: float, range_min: float, range_max: float) -> tuple[str, float]:
    """Inside [min, max] = green; just outside (±10% of range width) = yellow; further = red."""
    if range_min <= measured <= range_max:
        return ("GREEN", 0.0)
    width = max(range_max - range_min, 0.1)
    if measured < range_min:
        delta = range_min - measured
    else:
        delta = measured - range_max
    severity = delta / (width * 0.5)
    if severity <= 1.0:
        return ("YELLOW", severity)
    return ("RED", severity)


def _verdict_for_checks(checks: list[dict]) -> tuple[str, int]:
    """Overall verdict + 0..100 score based on per-check colours.

    Score: each GREEN check = full points, YELLOW = half, RED = 0.
    Total normalised to 100. Verdict: GREEN if score >= 85, YELLOW 60-84,
    RED below 60. Hard-fail: a RED on LUFS or LRA forces overall RED.
    """
    n = len(checks)
    if n == 0:
        return ("RED", 0)
    pts = 0
    hard_red = False
    for c in checks:
        if c["verdict"] == "GREEN":
            pts += 100
        elif c["verdict"] == "YELLOW":
            pts += 50
        else:
            if c["name"] in ("integrated_lufs", "lra_lu"):
                hard_red = True
    score = int(round(pts / n))
    if hard_red:
        return ("RED", min(score, 55))
    if score >= 85:
        return ("GREEN", score)
    if score >= 60:
        return ("YELLOW", score)
    return ("RED", score)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def check_style(mix_path: Path, profile: dict, output_dir: Path) -> dict:
    data, sr = sf.read(str(mix_path), always_2d=True)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    mono = data.mean(axis=1).astype(np.float64)

    # Measurements on the original (LUFS measured on stereo, others on mono mix)
    meter = pyln.Meter(sr)
    lufs_input = data if data.shape[1] > 1 else mono
    try:
        integrated_lufs = float(meter.integrated_loudness(lufs_input))
    except Exception:
        integrated_lufs = -120.0
    try:
        lra = float(meter.loudness_range(lufs_input))
    except Exception:
        lra = 0.0
    crest = _crest_factor_db(mono)

    # Spectral balance measured on a LUFS-normalised copy
    target_lufs = profile["lufs"]["integrated_target"]
    mono_norm = _normalize_to_lufs(mono, sr, integrated_lufs, target_lufs)
    bands = {
        name: round(_band_rms_db(mono_norm, sr, lo, hi), 1)
        for name, (lo, hi) in _BANDS.items()
    }

    # Grade every check
    checks: list[dict] = []

    # LUFS — symmetric tolerance
    v, s = _grade_value(integrated_lufs, target_lufs, profile["lufs"]["tolerance_lu"])
    checks.append({
        "name": "integrated_lufs", "measured": round(integrated_lufs, 1),
        "target": target_lufs, "delta": round(integrated_lufs - target_lufs, 1),
        "tolerance": profile["lufs"]["tolerance_lu"], "verdict": v, "severity": round(s, 2),
        "borderline": bool(v == "GREEN" and s >= 0.7),
    })

    # LRA — range based
    lra_min, lra_max = profile["lra"]["range_lu"]
    v, s = _grade_range(lra, profile["lra"]["target_lu"], lra_min, lra_max)
    checks.append({
        "name": "lra_lu", "measured": round(lra, 1),
        "target": profile["lra"]["target_lu"], "range": [lra_min, lra_max],
        "verdict": v, "severity": round(s, 2),
        "borderline": bool(v == "GREEN" and s >= 0.7),
    })

    # Crest — range based
    cr_min, cr_max = profile["crest_factor"]["range_db"]
    v, s = _grade_range(crest, profile["crest_factor"]["target_db"], cr_min, cr_max)
    checks.append({
        "name": "crest_factor_db", "measured": crest,
        "target": profile["crest_factor"]["target_db"], "range": [cr_min, cr_max],
        "verdict": v, "severity": round(s, 2),
        "borderline": bool(v == "GREEN" and s >= 0.7),
    })

    # 5 band checks
    for band_name, band_spec in profile["tonal_balance_dbfs"].items():
        meas = bands[band_name]
        v, s = _grade_value(meas, band_spec["target"], band_spec["tolerance"])
        checks.append({
            "name": f"band_{band_name}", "measured": meas,
            "target": band_spec["target"], "delta": round(meas - band_spec["target"], 1),
            "tolerance": band_spec["tolerance"], "verdict": v, "severity": round(s, 2),
            "borderline": bool(v == "GREEN" and s >= 0.7),
        })

    verdict, score = _verdict_for_checks(checks)
    counts = {
        "green":      sum(1 for c in checks if c["verdict"] == "GREEN"),
        "borderline": sum(1 for c in checks if c.get("borderline")),
        "yellow":     sum(1 for c in checks if c["verdict"] == "YELLOW"),
        "red":        sum(1 for c in checks if c["verdict"] == "RED"),
    }

    result = {
        "mix_file": str(mix_path),
        "style_profile": profile["name"],
        "profile_version": profile.get("version", "?"),
        "score": score,
        "verdict": verdict,
        "counts": counts,
        "measurements": {
            "integrated_lufs": round(integrated_lufs, 1),
            "lra_lu":          round(lra, 1),
            "crest_factor_db": crest,
            "bands_lufs_normalised_dbfs": bands,
        },
        "checks": checks,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "style_check.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "style_check.txt").write_text(
        _format_text_report(result, profile), encoding="utf-8"
    )
    return result


def _bar(measured: float, target: float, scale_db: float = 6.0, width: int = 30) -> str:
    """Two-sided ASCII bar centered on 0 (target). One char ≈ scale_db / (width/2) dB."""
    half = width // 2
    delta = measured - target
    pos = int(round((delta / scale_db) * half))
    pos = max(-half, min(half, pos))
    if pos < 0:
        return " " * (half + pos) + "█" * (-pos) + "│" + " " * half
    if pos > 0:
        return " " * half + "│" + "█" * pos + " " * (half - pos)
    return " " * half + "│" + " " * half


_COLOR_TAG = {"GREEN": "[OK]", "YELLOW": "[~~]", "RED": "[!!]"}
_BORDERLINE_TAG = "[OK*]"  # green but near the yellow threshold (severity >= 0.7)


def _tag_for(c: dict) -> str:
    if c.get("borderline"):
        return _BORDERLINE_TAG
    return _COLOR_TAG[c["verdict"]]


def _format_text_report(result: dict, profile: dict) -> str:
    lines: list[str] = []
    lines.append(f"STYLE CHECK — {result['style_profile']} (profile v{result['profile_version']})")
    lines.append("=" * 70)
    lines.append(f"Mix: {result['mix_file']}")
    borderline_part = ""
    if result['counts'].get('borderline'):
        borderline_part = f"   [{result['counts']['borderline']} borderline]"
    lines.append(f"Verdict: {result['verdict']}    Score: {result['score']}/100    "
                 f"({result['counts']['green']} green / "
                 f"{result['counts']['yellow']} yellow / "
                 f"{result['counts']['red']} red){borderline_part}")
    lines.append("")
    lines.append(profile.get("description", ""))
    lines.append("")
    lines.append("LOUDNESS / DYNAMICS")
    lines.append("-" * 70)
    for c in result["checks"][:3]:
        tag = _tag_for(c)
        if "range" in c:
            r = c["range"]
            lines.append(f"  {tag} {c['name']:<18}  measured {c['measured']:>6.1f}   "
                         f"target {c['target']:>5.1f}   range [{r[0]}..{r[1]}]")
        else:
            d = c.get("delta", 0.0)
            lines.append(f"  {tag} {c['name']:<18}  measured {c['measured']:>6.1f}   "
                         f"target {c['target']:>5.1f}   delta {d:+.1f}   (±{c['tolerance']})")
    lines.append("")
    lines.append("TONAL BALANCE (LUFS-normalised, dBFS RMS, bar = measured-vs-target ±6 dB)")
    lines.append("-" * 70)
    for c in result["checks"][3:]:
        tag = _tag_for(c)
        bar = _bar(c["measured"], c["target"])
        name = c["name"].replace("band_", "")
        lines.append(f"  {tag} {name:<18}  {c['measured']:>6.1f} dB  [{bar}] "
                     f"Δ{c['delta']:+5.1f}  (±{c['tolerance']})")
    lines.append("")
    # Borderline GREEN warnings (severity >= 0.7) — call them out as heads-up
    borderlines = [c for c in result["checks"] if c.get("borderline")]
    if borderlines:
        lines.append("BORDERLINE (GREEN but close to YELLOW — worth noting)")
        lines.append("-" * 70)
        for c in borderlines:
            name = c["name"].replace("band_", "")
            sev = c.get("severity", 0)
            if "delta" in c:
                lines.append(f"  [OK*] {name:<18}  delta {c['delta']:+.1f}  (severity {sev:.2f}, tolerance ±{c.get('tolerance', '?')})")
            else:
                lines.append(f"  [OK*] {name:<18}  measured {c['measured']} (severity {sev:.2f}, range {c.get('range', '?')})")
        lines.append("")
    # Recommendations from each RED / YELLOW band
    recs: list[str] = []
    for c in result["checks"]:
        if c["verdict"] == "GREEN":
            continue
        if c["name"].startswith("band_"):
            band = c["name"].replace("band_", "")
            d = c["delta"]
            direction = "cut" if d > 0 else "boost"
            recs.append(f"  - {band}: {direction} ~{abs(d):.1f} dB to reach style target")
        elif c["name"] == "integrated_lufs":
            d = c["delta"]
            if d < 0:
                recs.append(f"  - integrated LUFS {d:+.1f} below target — increase loudness")
            else:
                recs.append(f"  - integrated LUFS {d:+.1f} above target — back off the limiter")
        elif c["name"] == "lra_lu":
            r = c["range"]
            if c["measured"] < r[0]:
                recs.append(f"  - LRA {c['measured']} below {r[0]} — over-compressed for this style")
            else:
                recs.append(f"  - LRA {c['measured']} above {r[1]} — too dynamic, more compression")
        elif c["name"] == "crest_factor_db":
            r = c["range"]
            if c["measured"] < r[0]:
                recs.append(f"  - crest {c['measured']} below {r[0]} — over-limited")
            else:
                recs.append(f"  - crest {c['measured']} above {r[1]} — under-processed")
    if recs:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        lines.extend(recs)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade a stereo mix against a named style profile.",
    )
    parser.add_argument("mix", nargs="?", type=Path, help="Path to mix WAV (e.g. mix.wav, master_spotify.wav)")
    parser.add_argument("--style", type=str, help="Style profile name (modern_rock / classic_rock / pop / hip_hop / jazz_acoustic)")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--list-styles", action="store_true", help="List built-in profiles")
    args = parser.parse_args()

    if args.list_styles:
        for name in list_profiles():
            try:
                p = load_profile(name)
                print(f"  {name:<18}  v{p.get('version','?'):<5}  {p.get('description', '')[:60]}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {name}  (ERROR: {exc})")
        return

    if not args.mix or not args.style:
        parser.error("Both <mix> and --style are required (use --list-styles to see profiles)")
    if not args.mix.exists():
        print(f"FATAL: mix file not found: {args.mix}", file=sys.stderr)
        sys.exit(2)

    profile = load_profile(args.style)
    result = check_style(args.mix, profile, args.output_dir)
    print(f"Style check: {result['style_profile']}  →  {result['verdict']}  ({result['score']}/100)")
    print(f"  {result['counts']['green']} green / {result['counts']['yellow']} yellow / {result['counts']['red']} red")
    print(f"  Report: {args.output_dir / 'style_check.txt'}")
    sys.exit(0 if result["verdict"] != "RED" else 1)


if __name__ == "__main__":
    main()
