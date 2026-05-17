"""Detect frequency masking between stems in a session.

For each 1/3-octave band, finds stem pairs that compete within a configurable
dB threshold — indicating that one stem is likely masking the other in the mix.

All stems are LUFS-normalized to -18 dBFS before comparison so that the analysis
reflects mix-ready levels rather than raw recording gain differences.

Severity levels (based on level gap between competing stems in a band):
  CRITICAL  < 3 dB gap — strong masking, almost certainly audible
  HIGH      3–6 dB gap — significant masking, likely audible
  MODERATE  6–10 dB gap — moderate masking, may be audible on some playback systems

Output:
  masking_report.json  — full per-band data, all competing pairs
  masking_report.txt   — human-readable heatmap + ranked masking list

Usage:
  # Auto-discover stems from session output directory
  python detect_masking.py output/terido --output-dir output/terido

  # Specify stage (raw, eq, comp, fx)
  python detect_masking.py output/terido --stage comp --output-dir output/terido

  # Specific files only
  python detect_masking.py kick.wav snare.wav bass.wav --output-dir output/terido

  # Lower threshold to see more masking pairs
  python detect_masking.py output/terido --threshold 6 --output-dir output/terido
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import welch

# Stage → filename candidates (first match wins)
_STAGE_FILES = {
    "raw":  ["assembled.wav"],
    "eq":   ["assembled_eq.wav"],
    "comp": ["assembled_eq_comp.wav", "assembled_gained_eq_comp.wav"],
    "fx":   [],  # discovered via glob
}

# 10-band heatmap regions (label, lo_hz, hi_hz)
_HEATMAP_BANDS = [
    ("SUB  ", 20,    60),
    ("LBASS", 60,   120),
    ("BASS ", 120,  250),
    ("UBASS", 250,  500),
    ("LOMID", 500, 1000),
    ("MID  ", 1000, 2000),
    ("UMID ", 2000, 4000),
    ("PRES ", 4000, 8000),
    ("AIR  ", 8000, 12000),
    ("HIAIR", 12000, 20000),
]

_SEVERITY_THRESHOLDS = [
    ("CRITICAL", 0.0, 3.0),
    ("HIGH",     3.0, 6.0),
    ("MODERATE", 6.0, 10.0),
]

_BLOCKS = " ░▒▓█"


# ---------------------------------------------------------------------------
# Stem discovery
# ---------------------------------------------------------------------------

def _find_stem_file(track_dir: Path, stage: str) -> Path | None:
    if stage == "fx":
        # Last alphabetically among _sat, _amp, _delay, _reverb suffixes
        candidates = sorted(track_dir.glob("assembled_*_*.wav"))
        # Prefer amp/reverb over plain _comp; skip _comp and _eq only files
        fx_files = [f for f in candidates if f.stem.count("_") >= 3]
        if fx_files:
            return fx_files[-1]
        # Fall back to comp
        return _find_stem_file(track_dir, "comp")

    for name in _STAGE_FILES.get(stage, []):
        p = track_dir / name
        if p.exists():
            return p
    return None


def discover_stems(session_dir: Path, stage: str) -> dict[str, Path]:
    """Find the best assembled WAV per track in session_dir/tracks/."""
    tracks_dir = session_dir / "tracks"
    if not tracks_dir.exists():
        return {}

    stems: dict[str, Path] = {}
    for track_dir in sorted(tracks_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        f = _find_stem_file(track_dir, stage)
        if f is None:
            # Fall back through stages
            for fallback in ("comp", "eq", "raw"):
                if fallback == stage:
                    continue
                f = _find_stem_file(track_dir, fallback)
                if f:
                    break
        if f:
            stems[track_dir.name] = f
    return stems


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------

def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    return data.mean(axis=1).astype(np.float64), sr


def _lufs_normalize(mono: np.ndarray, sr: int, target_lufs: float = -18.0) -> np.ndarray:
    meter = pyln.Meter(sr)
    try:
        current = float(meter.integrated_loudness(mono))
    except Exception:
        return mono
    if current < -70.0:
        return mono
    gain_lin = 10.0 ** ((target_lufs - current) / 20.0)
    return mono * gain_lin


def _third_octave_psd_db(mono: np.ndarray, sr: int) -> list[dict]:
    nperseg = min(len(mono), 32768)
    freqs, psd = welch(mono, fs=sr, nperseg=nperseg, average="mean")
    psd_db = 10.0 * np.log10(psd + 1e-20)

    centers: list[float] = []
    f = 20.0
    while f <= min(sr / 2.0, 20000.0):
        centers.append(f)
        f *= 2.0 ** (1.0 / 3.0)

    bands = []
    for fc in centers:
        lo = fc / 2.0 ** (1.0 / 6.0)
        hi = fc * 2.0 ** (1.0 / 6.0)
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            bands.append({"hz": round(fc, 1), "db": round(float(np.mean(psd_db[mask])), 1)})
    return bands


def _band_region_db(bands: list[dict], lo_hz: float, hi_hz: float) -> float:
    vals = [b["db"] for b in bands if lo_hz <= b["hz"] < hi_hz]
    return round(float(np.mean(vals)), 1) if vals else -120.0


# ---------------------------------------------------------------------------
# Masking detection
# ---------------------------------------------------------------------------

def _severity(gap_db: float) -> str:
    for label, lo, hi in _SEVERITY_THRESHOLDS:
        if lo <= gap_db < hi:
            return label
    return ""


def find_masking_pairs(
    stem_bands: dict[str, list[dict]],
    threshold_db: float = 6.0,
    floor_db: float = -45.0,
) -> list[dict]:
    """For each 1/3-octave band, find stem pairs competing within threshold_db."""
    # Build hz → {stem: db} index
    hz_index: dict[float, dict[str, float]] = {}
    for stem, bands in stem_bands.items():
        for b in bands:
            hz = b["hz"]
            db = b["db"]
            if db < floor_db:
                continue
            hz_index.setdefault(hz, {})[stem] = db

    masking_events: list[dict] = []
    for hz in sorted(hz_index):
        active = hz_index[hz]
        if len(active) < 2:
            continue

        # Sort stems by level in this band (loudest first)
        ranked = sorted(active.items(), key=lambda x: -x[1])

        # Check all pairs
        seen = set()
        for i, (stem_a, db_a) in enumerate(ranked):
            for stem_b, db_b in ranked[i + 1:]:
                gap = abs(db_a - db_b)
                if gap >= threshold_db:
                    continue
                sev = _severity(gap)
                if not sev:
                    continue
                pair_key = tuple(sorted([stem_a, stem_b]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                masking_events.append({
                    "hz": hz,
                    "severity": sev,
                    "gap_db": round(gap, 1),
                    "stems": [
                        {"name": stem_a, "db": round(db_a, 1)},
                        {"name": stem_b, "db": round(db_b, 1)},
                    ],
                })

    # Sort: severity first, then hz
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2}
    masking_events.sort(key=lambda e: (sev_order.get(e["severity"], 9), e["hz"]))
    return masking_events


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _hz_label(hz: float) -> str:
    return f"{hz:6.0f} Hz" if hz < 1000 else f"{hz / 1000:5.2f} kHz"


def _heatmap(stem_bands: dict[str, list[dict]], stem_names: list[str]) -> str:
    col_w = 5
    name_w = max(len(n) for n in stem_names) + 1

    header = f"{'Stem':<{name_w}}" + "".join(f"{b[0]:^{col_w}}" for b in _HEATMAP_BANDS)
    sep = "─" * len(header)
    lines = ["", "STEM ENERGY HEATMAP (per frequency region, loudness-normalized)", sep, header, sep]

    for name in stem_names:
        bands = {b["hz"]: b["db"] for b in stem_bands.get(name, [])}
        row = f"{name:<{name_w}}"
        for _, lo, hi in _HEATMAP_BANDS:
            # Average dB in this region
            vals = [db for hz, db in bands.items() if lo <= hz < hi]
            if not vals:
                row += f"{'':^{col_w}}"
                continue
            avg_db = float(np.mean(vals))
            # Map -60..0 dBFS to 0..4 block index
            level = int(np.clip((avg_db + 60) / 15, 0, 4))
            row += f"{_BLOCKS[level]:^{col_w}}"
        lines.append(row)

    lines.append(sep)
    lines.append("  Energy scale: (space)=silent  ░=low  ▒=mid  ▓=high  █=dominant")
    return "\n".join(lines)


def _masking_section(events: list[dict], threshold_db: float) -> str:
    if not events:
        return "\nNo masking pairs detected above threshold."

    lines = [f"\nMASKING PAIRS  (threshold: ±{threshold_db:.0f} dB)", "=" * 60]
    current_sev = None

    moderate_shown = 0
    moderate_limit = 15

    for e in events:
        if e["severity"] != current_sev:
            current_sev = e["severity"]
            sev_desc = {
                "CRITICAL": "CRITICAL  (< 3 dB gap — strong masking)",
                "HIGH":     "HIGH      (3–6 dB gap — significant masking)",
                "MODERATE": f"MODERATE  (6–10 dB gap — top {moderate_limit} shown)",
            }.get(current_sev, current_sev)
            lines += ["", f"── {sev_desc} ──────────────────────"]

        if e["severity"] == "MODERATE":
            if moderate_shown >= moderate_limit:
                continue
            moderate_shown += 1

        hz_lbl = _hz_label(e["hz"])
        a, b = e["stems"][0], e["stems"][1]
        lines.append(
            f"  {hz_lbl}  {a['name']} ({a['db']:+.1f} dBFS)  vs  "
            f"{b['name']} ({b['db']:+.1f} dBFS)  [{e['gap_db']:.1f} dB gap]"
        )

    lines += [
        "",
        "RECOMMENDATIONS",
        "─" * 60,
        "  For each competing pair: decide which stem is the 'hero' at that frequency.",
        "  Cut the non-hero stem 2–4 dB in the flagged band using apply_eq.py:",
        "    --filter '{\"type\":\"peak\",\"hz\":64,\"db\":-3,\"q\":1.5}'",
        "  CRITICAL pairs should be addressed first — they cause the most audible masking.",
        "  Sidechain compression (apply_compression.py --sidechain) can also reduce",
        "  transient masking between kick and bass in the 50–120 Hz range.",
    ]
    return "\n".join(lines)


def _summary_counts(events: list[dict]) -> str:
    counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0}
    for e in events:
        counts[e["severity"]] = counts.get(e["severity"], 0) + 1
    parts = [f"{v} {k.lower()}" for k, v in counts.items() if v > 0]
    return ", ".join(parts) if parts else "none"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def detect_masking(
    stems: dict[str, Path],
    output_dir: Path,
    threshold_db: float = 10.0,
    lufs_target: float = -18.0,
    stage: str = "comp",
) -> dict:
    if not stems:
        print("No stems found.", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(stems)} stems...", flush=True)
    stem_bands: dict[str, list[dict]] = {}
    stem_names = list(stems.keys())

    for i, (name, path) in enumerate(stems.items(), 1):
        print(f"  [{i}/{len(stems)}] {name}", flush=True)
        mono, sr = _load_mono(path)
        mono = _lufs_normalize(mono, sr, lufs_target)
        stem_bands[name] = _third_octave_psd_db(mono, sr)

    print("Detecting masking pairs...", flush=True)
    events = find_masking_pairs(stem_bands, threshold_db=threshold_db)

    report = {
        "stems": {name: str(path) for name, path in stems.items()},
        "stage": stage,
        "threshold_db": threshold_db,
        "lufs_normalization_target": lufs_target,
        "masking_pairs": events,
        "summary": {
            "total_pairs": len(events),
            "critical": sum(1 for e in events if e["severity"] == "CRITICAL"),
            "high":     sum(1 for e in events if e["severity"] == "HIGH"),
            "moderate": sum(1 for e in events if e["severity"] == "MODERATE"),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "masking_report.json"
    txt_path  = output_dir / "masking_report.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_line = _summary_counts(events)
    header = "\n".join([
        "FREQUENCY MASKING REPORT",
        "=" * 60,
        f"  Stage     : {stage}",
        f"  Stems     : {len(stems)}",
        f"  Threshold : ±{threshold_db:.0f} dB",
        f"  Masking   : {summary_line}",
    ])

    full_text = (
        header
        + _heatmap(stem_bands, stem_names)
        + _masking_section(events, threshold_db)
    )
    txt_path.write_text(full_text, encoding="utf-8")
    print(full_text)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect frequency masking between stems in a session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="Session output directory (auto-discovers stems) or individual WAV files",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for reports")
    parser.add_argument(
        "--stage", choices=["raw", "eq", "comp", "fx"], default="comp",
        help="Which processing stage to use for auto-discovery (default: comp)",
    )
    parser.add_argument(
        "--threshold", type=float, default=6.0, metavar="DB",
        help="Maximum dB gap between competing stems to flag (default: 6.0 = HIGH+CRITICAL only; use 10 for MODERATE too)",
    )
    parser.add_argument(
        "--lufs", type=float, default=-18.0, metavar="LUFS",
        help="Normalize all stems to this LUFS before comparison (default: -18.0)",
    )
    args = parser.parse_args()

    stems: dict[str, Path] = {}
    stage = args.stage

    if len(args.inputs) == 1 and args.inputs[0].is_dir():
        # Auto-discover from session directory
        session_dir = args.inputs[0]
        stems = discover_stems(session_dir, stage)
        if not stems:
            print(
                f"No stems found in {session_dir}/tracks/ for stage '{stage}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-discovered {len(stems)} stems (stage: {stage}):", file=sys.stderr)
        for name in stems:
            print(f"  {name}", file=sys.stderr)
    else:
        # Explicit file list
        stage = "manual"
        for p in args.inputs:
            if not p.exists():
                print(json.dumps({"error": f"Not found: {p}"}), file=sys.stderr)
                sys.exit(1)
            stems[p.stem] = p

    detect_masking(
        stems,
        output_dir=args.output_dir,
        threshold_db=args.threshold,
        lufs_target=args.lufs,
        stage=stage,
    )


if __name__ == "__main__":
    main()
