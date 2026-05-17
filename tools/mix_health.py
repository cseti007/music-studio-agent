"""Session-level mix health scorecard.

Runs after `render_mix --render` and produces a single-page go/no-go view:
the rendered mix's loudness, dynamics, stereo image, tonal balance vs. an
optional reference, masking pair counts, and pumping detection on the bus
stems.

This is the agent's primary "are we done?" decision tool. Each metric gets
a green / yellow / red verdict so the agent can speak in concrete terms
instead of "the mix sounds nice" hand-waving.

Inputs:
  - The session directory (auto-discovers <session>/mixes/mix.wav and the
    per-bus stems under <session>/stems/)
  - Optional reference WAV for tonal balance comparison

Outputs:
  - mix_health.json
  - mix_health.txt (the scorecard you actually read)

Usage:
  python mix_health.py output/terido [--reference ref.wav] [--lufs-target -14]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly, welch


# Verdict thresholds (rock/pop modern target). Yellow = slightly off, red = needs work.
_TARGETS = {
    "lufs_tolerance_db": 0.5,         # within ±0.5 of target = green; ±1.5 = yellow; else red
    "true_peak_ceiling_dbtp": -1.0,   # green if <= ceiling
    "lra_min_lu": 6.0,                # green if >= 6 LU; yellow 4-6; red < 4
    "ms_width_min": 0.10,             # green if >= 0.10; yellow 0.05-0.10; red < 0.05
    "ms_width_max": 0.50,             # green if <= 0.50; yellow 0.50-0.65; red > 0.65
    "mono_corr_low_freq_min": 0.6,    # below 100 Hz, near-mono is needed
    "spectral_delta_yellow_db": 2.0,  # band deltas this much vs ref = yellow
    "spectral_delta_red_db":    4.0,  # this much = red
}


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------

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
# Measurements
# ---------------------------------------------------------------------------

def _true_peak_dbfs(signal: np.ndarray, oversample: int = 4) -> float:
    up = resample_poly(signal, oversample, 1)
    return float(20.0 * np.log10(max(float(np.max(np.abs(up))), 1e-12)))


def _stereo_metrics(data: np.ndarray, sr: int) -> dict:
    if data.shape[1] < 2:
        return {"channels": data.shape[1], "ms_width_ratio": None,
                "lr_correlation": None, "low_freq_correlation": None}
    L = data[:, 0].astype(np.float64)
    R = data[:, 1].astype(np.float64)

    rms_L = float(np.sqrt(np.mean(L ** 2) + 1e-12))
    rms_R = float(np.sqrt(np.mean(R ** 2) + 1e-12))
    balance_db = 20.0 * np.log10((rms_L + 1e-12) / (rms_R + 1e-12))

    M = (L + R) * 0.5
    S = (L - R) * 0.5
    rms_M = float(np.sqrt(np.mean(M ** 2) + 1e-12))
    rms_S = float(np.sqrt(np.mean(S ** 2) + 1e-12))
    ms_width = rms_S / max(rms_M, 1e-12)

    corr = float(np.corrcoef(L, R)[0, 1]) if rms_L > 1e-10 and rms_R > 1e-10 else 1.0

    # Low-frequency correlation (below 200 Hz) — bass should be near-mono for
    # both phase coherence and mono compatibility on club PA / phone.
    from scipy.signal import butter, sosfilt
    sos = butter(4, 200.0 / (sr / 2.0), btype="low", output="sos")
    L_low = sosfilt(sos, L)
    R_low = sosfilt(sos, R)
    if float(np.std(L_low)) > 1e-10 and float(np.std(R_low)) > 1e-10:
        low_corr = float(np.corrcoef(L_low, R_low)[0, 1])
    else:
        low_corr = 1.0

    return {
        "channels": 2,
        "balance_db": round(balance_db, 2),
        "ms_width_ratio": round(ms_width, 3),
        "lr_correlation": round(corr, 3),
        "low_freq_correlation": round(low_corr, 3),
    }


def _detect_pumping_quick(signal: np.ndarray, sr: int) -> dict:
    """Compact version of analyze._detect_pumping for the stems.

    Depth (p95/p5) measured on ACTIVE frames only (RMS > -40 dBFS) — without
    that gate, silent gaps between musical sections collapse p5 to ~0 and
    hide real pumping on intermittent material.
    """
    hop = max(1, int(sr * 0.01))
    n_frames = len(signal) // hop
    if n_frames < 200:
        return {"pumping_detected": False, "pump_rate_hz": None,
                "modulation_depth_db": None, "lf_excess_db": None,
                "active_frame_ratio": None}

    env = np.array([
        np.sqrt(np.mean(signal[i * hop:(i + 1) * hop] ** 2) + 1e-20)
        for i in range(n_frames)
    ])
    env_sr = sr / hop
    from scipy.signal import butter, sosfilt
    sos_hp = butter(2, 0.5 / (env_sr / 2.0), btype="high", output="sos")
    env_hp = sosfilt(sos_hp, env)

    nperseg = min(len(env_hp), 1024)
    freqs, psd = welch(env_hp, fs=env_sr, nperseg=nperseg)

    def _band_peak_db(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            return -120.0
        return 10.0 * np.log10(float(np.max(psd[m])) + 1e-20)

    pump_db = _band_peak_db(1.0, 5.0)
    ref_db = _band_peak_db(5.0, 15.0)
    excess = pump_db - ref_db

    active_threshold_lin = 10.0 ** (-40.0 / 20.0)
    active_mask = env > active_threshold_lin
    n_active = int(np.sum(active_mask))
    active_ratio = n_active / max(n_frames, 1)

    if n_active < 50:
        depth_db = 0.0
    else:
        active_env = env[active_mask]
        p95 = float(np.percentile(active_env, 95))
        p5 = float(np.percentile(active_env, 5))
        depth_db = 20.0 * np.log10(p95 / max(p5, 1e-12)) if p5 > 1e-9 else 0.0

    pumping = bool(excess >= 6.0 and depth_db >= 5.0)

    mask = (freqs >= 1.0) & (freqs < 5.0)
    rate = float(freqs[mask][int(np.argmax(psd[mask]))]) if mask.any() else None

    return {
        "pumping_detected": pumping,
        "pump_rate_hz": round(rate, 2) if rate is not None else None,
        "modulation_depth_db": round(depth_db, 1),
        "lf_excess_db": round(excess, 1),
        "active_frame_ratio": round(active_ratio, 3),
    }


def _third_octave_psd_db(mono: np.ndarray, sr: int) -> dict[float, float]:
    nperseg = min(len(mono), 32768)
    freqs, psd = welch(mono, fs=sr, nperseg=nperseg, average="mean")
    psd_db = 10.0 * np.log10(psd + 1e-20)
    out: dict[float, float] = {}
    f = 20.0
    while f <= min(sr / 2.0, 20000.0):
        lo = f / 2.0 ** (1.0 / 6.0)
        hi = f * 2.0 ** (1.0 / 6.0)
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            out[round(f, 1)] = float(np.mean(psd_db[m]))
        f *= 2.0 ** (1.0 / 3.0)
    return out


# ---------------------------------------------------------------------------
# Section reports
# ---------------------------------------------------------------------------

def _loudness_section(mono: np.ndarray, data: np.ndarray, sr: int,
                      lufs_target: float, tp_ceiling: float) -> dict:
    meter = pyln.Meter(sr)
    try:
        lufs = float(meter.integrated_loudness(data if data.shape[1] > 1 else mono))
    except Exception:
        lufs = -120.0
    try:
        lra = float(meter.loudness_range(data if data.shape[1] > 1 else mono))
    except Exception:
        lra = 0.0
    tp = _true_peak_dbfs(mono)

    # Verdicts
    lufs_err = abs(lufs - lufs_target)
    lufs_v = _verdict(
        green=lufs_err <= _TARGETS["lufs_tolerance_db"],
        yellow=lufs_err <= 1.5,
    )
    tp_v = _verdict(green=tp <= tp_ceiling, yellow=tp <= tp_ceiling + 1.0)
    lra_v = _verdict(green=lra >= _TARGETS["lra_min_lu"], yellow=lra >= 4.0)

    return {
        "integrated_lufs": round(lufs, 2),
        "lufs_target": lufs_target,
        "lufs_delta": round(lufs - lufs_target, 2),
        "lufs_verdict": lufs_v,
        "true_peak_dbtp": round(tp, 2),
        "true_peak_ceiling": tp_ceiling,
        "true_peak_verdict": tp_v,
        "lra_lu": round(lra, 2),
        "lra_verdict": lra_v,
    }


def _stereo_section(data: np.ndarray, sr: int) -> dict:
    stereo = _stereo_metrics(data, sr)
    if stereo["ms_width_ratio"] is None:
        return {"width_verdict": GREEN, **stereo, "mono_compat_verdict": GREEN}
    width = stereo["ms_width_ratio"]
    width_v = _verdict(
        green=(_TARGETS["ms_width_min"] <= width <= _TARGETS["ms_width_max"]),
        yellow=(0.05 <= width <= 0.65),
    )
    low_corr = stereo["low_freq_correlation"]
    mono_v = _verdict(
        green=low_corr >= _TARGETS["mono_corr_low_freq_min"],
        yellow=low_corr >= 0.4,
    )
    return {**stereo, "width_verdict": width_v, "mono_compat_verdict": mono_v}


def _tonal_balance_section(mono: np.ndarray, sr: int, ref_path: Path | None) -> dict:
    if ref_path is None or not ref_path.exists():
        return {"available": False, "verdict": GREEN, "note": "no reference supplied"}

    ref_data, ref_sr = sf.read(str(ref_path), always_2d=True)
    ref_mono = ref_data.mean(axis=1).astype(np.float64)

    meter = pyln.Meter(sr)
    try:
        tgt_lufs = float(meter.integrated_loudness(mono))
    except Exception:
        tgt_lufs = -120.0
    meter_r = pyln.Meter(ref_sr)
    try:
        ref_lufs = float(meter_r.integrated_loudness(ref_mono))
    except Exception:
        ref_lufs = -120.0
    offset = ref_lufs - tgt_lufs if ref_lufs > -100 and tgt_lufs > -100 else 0.0

    tgt_bands = _third_octave_psd_db(mono, sr)
    ref_bands = _third_octave_psd_db(ref_mono, ref_sr)
    common = sorted(set(tgt_bands) & set(ref_bands))

    deltas = []
    for hz in common:
        d = (tgt_bands[hz] + offset) - ref_bands[hz]
        deltas.append({"hz": hz, "delta_db": round(d, 1)})

    abs_max = max((abs(d["delta_db"]) for d in deltas), default=0.0)
    verdict = _verdict(
        green=abs_max < _TARGETS["spectral_delta_yellow_db"],
        yellow=abs_max < _TARGETS["spectral_delta_red_db"],
    )

    # Roll up into 3 regions for human reading
    def region_avg(lo, hi):
        vals = [d["delta_db"] for d in deltas if lo <= d["hz"] < hi]
        return round(float(np.mean(vals)), 1) if vals else 0.0

    return {
        "available": True,
        "reference": str(ref_path),
        "loudness_match_offset_db": round(offset, 2),
        "region_delta_db": {
            "bottom_20_250": region_avg(20, 250),
            "mids_250_4k":   region_avg(250, 4000),
            "top_4k_20k":    region_avg(4000, 20000),
        },
        "max_band_delta_db": round(abs_max, 1),
        "verdict": verdict,
    }


def _stems_section(stems_dir: Path | None) -> dict:
    if stems_dir is None or not stems_dir.is_dir():
        return {"available": False, "note": "no stems/ directory — run render_mix --stems first"}

    stem_files = sorted(stems_dir.glob("stem_*.wav"))
    if not stem_files:
        return {"available": False, "note": f"no stem_*.wav in {stems_dir}"}

    results = []
    any_pumping = False
    for f in stem_files:
        data, sr = sf.read(str(f), always_2d=True)
        mono = data.mean(axis=1).astype(np.float64)
        pump = _detect_pumping_quick(mono, sr)
        if pump["pumping_detected"]:
            any_pumping = True
        results.append({
            "bus": f.stem.replace("stem_", ""),
            "pumping": pump,
        })
    verdict = RED if any_pumping else GREEN
    return {"available": True, "verdict": verdict, "buses": results}


def _masking_section(session_dir: Path) -> dict:
    report = session_dir / "analysis" / "masking_report.json"
    if not report.exists():
        return {"available": False, "verdict": GREEN, "note": "no masking_report.json — run detect_masking first"}
    data = json.loads(report.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    crit = int(summary.get("critical", 0))
    high = int(summary.get("high", 0))
    verdict = _verdict(green=(crit == 0 and high <= 2), yellow=(crit == 0 and high <= 5))
    return {
        "available": True,
        "critical": crit,
        "high": high,
        "moderate": int(summary.get("moderate", 0)),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def _render_text(report: dict) -> str:
    lines = ["MIX HEALTH REPORT", "=" * 60, f"  Mix: {report['mix_file']}", ""]

    L = report["loudness"]
    lines.append("LOUDNESS")
    lines.append("-" * 60)
    lines.append(f"  {L['lufs_verdict']} Integrated LUFS  : {L['integrated_lufs']:+.2f}  "
                 f"(target {L['lufs_target']}, delta {L['lufs_delta']:+.2f})")
    lines.append(f"  {L['true_peak_verdict']} True peak        : {L['true_peak_dbtp']:+.2f} dBTP  "
                 f"(ceiling {L['true_peak_ceiling']})")
    lines.append(f"  {L['lra_verdict']} LRA              : {L['lra_lu']:.1f} LU  "
                 f"(target >= {_TARGETS['lra_min_lu']} LU)")

    S = report["stereo"]
    lines.append("")
    lines.append("STEREO IMAGE")
    lines.append("-" * 60)
    if S.get("ms_width_ratio") is None:
        lines.append("  (mono mix — stereo metrics skipped)")
    else:
        lines.append(f"  {S['width_verdict']} M/S width ratio  : {S['ms_width_ratio']}  "
                     f"(green {_TARGETS['ms_width_min']}-{_TARGETS['ms_width_max']})")
        lines.append(f"  {S['mono_compat_verdict']} Low-freq L/R corr: {S['low_freq_correlation']}  "
                     f"(green >= {_TARGETS['mono_corr_low_freq_min']})")
        lines.append(f"      L/R correlation  : {S['lr_correlation']}")
        lines.append(f"      L/R balance      : {S['balance_db']:+.2f} dB")

    T = report["tonal_balance"]
    lines.append("")
    lines.append("TONAL BALANCE vs REFERENCE")
    lines.append("-" * 60)
    if not T["available"]:
        lines.append(f"  (skipped — {T.get('note', 'no reference')})")
    else:
        lines.append(f"  {T['verdict']} Reference: {T['reference']}")
        r = T["region_delta_db"]
        lines.append(f"      Bottom (20-250 Hz)  : {r['bottom_20_250']:+.1f} dB")
        lines.append(f"      Mids (250-4 kHz)    : {r['mids_250_4k']:+.1f} dB")
        lines.append(f"      Top (4-20 kHz)      : {r['top_4k_20k']:+.1f} dB")
        lines.append(f"      Max single-band delta: {T['max_band_delta_db']:.1f} dB")

    M = report["masking"]
    lines.append("")
    lines.append("FREQUENCY MASKING")
    lines.append("-" * 60)
    if not M["available"]:
        lines.append(f"  (skipped — {M.get('note', '')})")
    else:
        lines.append(f"  {M['verdict']} Critical pairs : {M['critical']}")
        lines.append(f"      High pairs     : {M['high']}")
        lines.append(f"      Moderate pairs : {M['moderate']}")

    P = report["stems"]
    lines.append("")
    lines.append("STEM PUMPING / OVER-COMPRESSION")
    lines.append("-" * 60)
    if not P["available"]:
        lines.append(f"  (skipped — {P.get('note', '')})")
    else:
        for b in P["buses"]:
            pump = b["pumping"]
            tag = RED if pump["pumping_detected"] else GREEN
            detail = f"{pump['pump_rate_hz']} Hz, depth {pump['modulation_depth_db']} dB" \
                if pump["pumping_detected"] else "clean"
            lines.append(f"  {tag} {b['bus']:<10}  : {detail}")

    # Overall
    verdicts = [L["lufs_verdict"], L["true_peak_verdict"], L["lra_verdict"]]
    if S.get("ms_width_ratio") is not None:
        verdicts += [S["width_verdict"], S["mono_compat_verdict"]]
    if T["available"]:
        verdicts.append(T["verdict"])
    if M["available"]:
        verdicts.append(M["verdict"])
    if P["available"]:
        verdicts.append(P["verdict"])

    n_green = verdicts.count(GREEN)
    n_yellow = verdicts.count(YELLOW)
    n_red = verdicts.count(RED)
    total = len(verdicts)
    overall = GREEN if n_red == 0 and n_yellow <= 1 else (YELLOW if n_red == 0 else RED)

    lines += ["", "=" * 60,
              f"OVERALL: {overall}  {n_green} green, {n_yellow} yellow, {n_red} red  ({total} checks)"]
    if overall == GREEN:
        lines.append("  Mix is ready.")
    elif overall == YELLOW:
        lines.append("  Mix is close — address the yellow items.")
    else:
        lines.append("  Mix needs work — red items must be fixed before delivery.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def mix_health(session_dir: Path, output_dir: Path,
               reference: Path | None = None,
               lufs_target: float = -14.0,
               tp_ceiling: float = -1.0) -> dict:
    # Locate the mix
    mix_candidates = [
        session_dir / "mixes" / "mix.wav",
        session_dir / "mix.wav",
    ]
    mix_path = next((p for p in mix_candidates if p.exists()), None)
    if mix_path is None:
        raise FileNotFoundError(
            f"No mix.wav found. Tried: {[str(p) for p in mix_candidates]}. "
            f"Run render_mix --render first."
        )

    print(f"Reading {mix_path}...", flush=True)
    data, sr = sf.read(str(mix_path), always_2d=True)
    mono = data.mean(axis=1).astype(np.float64)

    print("  [1/5] Loudness...", flush=True)
    loudness = _loudness_section(mono, data, sr, lufs_target, tp_ceiling)
    print("  [2/5] Stereo...", flush=True)
    stereo = _stereo_section(data, sr)
    print("  [3/5] Tonal balance...", flush=True)
    tonal = _tonal_balance_section(mono, sr, reference)
    print("  [4/5] Masking summary...", flush=True)
    masking = _masking_section(session_dir)
    print("  [5/5] Stem pumping...", flush=True)
    stems = _stems_section(session_dir / "stems")

    report = {
        "session_dir": str(session_dir),
        "mix_file": str(mix_path),
        "loudness": loudness,
        "stereo": stereo,
        "tonal_balance": tonal,
        "masking": masking,
        "stems": stems,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mix_health.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = _render_text(report)
    (output_dir / "mix_health.txt").write_text(text, encoding="utf-8")
    print()
    print(text)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session-level mix health scorecard. Run after render_mix --render.",
    )
    parser.add_argument("session_dir", type=Path, help="Session output directory (e.g. output/terido)")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write mix_health.json/.txt (default: <session_dir>/analysis)",
    )
    parser.add_argument("--reference", type=Path, default=None,
                        help="Optional reference WAV for tonal-balance comparison")
    parser.add_argument("--lufs-target", type=float, default=-14.0,
                        help="LUFS target (default: -14 for streaming)")
    parser.add_argument("--tp-ceiling", type=float, default=-1.0,
                        help="True peak ceiling in dBTP (default: -1.0)")
    args = parser.parse_args()

    if not args.session_dir.is_dir():
        print(json.dumps({"error": f"Not a directory: {args.session_dir}"}), file=sys.stderr)
        sys.exit(1)
    output_dir = args.output_dir if args.output_dir is not None else args.session_dir / "analysis"

    mix_health(
        session_dir=args.session_dir,
        output_dir=output_dir,
        reference=args.reference,
        lufs_target=args.lufs_target,
        tp_ceiling=args.tp_ceiling,
    )


if __name__ == "__main__":
    main()
