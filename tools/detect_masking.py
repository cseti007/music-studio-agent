"""Find candidate frequency overlap between stems in a session.

For each 1/3-octave band, lists stem pairs whose band power lies within a
configurable dB gap. These are hypotheses about where two parts may compete;
check them by listening in context before any EQ decision.

All stems are LUFS-normalized to a common level (default -18 LUFS) before
comparison. This removes raw recording-gain differences but also discards
the actual mix balance: a flagged pair may be well separated by the faders,
and a part that is masked in the mix may not be flagged at all.

Band levels are 1/3-octave band power (PSD integrated over the band) in dB
relative to digital full scale (a full-scale sine reads -3 dB).

Severity levels (level gap between two stems in a band):
  CRITICAL  < 3 dB gap
  HIGH      3-6 dB gap
  MODERATE  6-10 dB gap
A smaller gap means more similar normalized band power; it does not by
itself establish audible masking.

Pairs are time-gated with the overlap coefficient of their activity
envelopes (shared active frames / active frames of the shorter part), so a
short part that plays entirely over a long part is still compared.

In session mode, tracks marked "active": false in mix_config.json are
skipped, and the fx stage uses each track's mix_config "file" when present.

Output:
  masking_report.json  - full per-band data, all candidate pairs
  masking_report.txt   - human-readable heatmap + ranked candidate list

Usage:
  # Auto-discover stems from session output directory
  python detect_masking.py output/<session> --output-dir output/<session>

  # Specify stage (raw, eq, comp, fx)
  python detect_masking.py output/<session> --stage comp --output-dir output/<session>

  # Specific files only
  python detect_masking.py kick.wav snare.wav bass.wav --output-dir output/<session>

  # Lower threshold to see more masking pairs
  python detect_masking.py output/<session> --threshold 6 --output-dir output/<session>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import welch

# Need to add the tools dir to sys.path so we can import _stages
sys.path.insert(0, str(Path(__file__).parent))
from _stages import STAGE_CANDIDATES as _SHARED_STAGE_CANDIDATES  # noqa: E402

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
        # Processing tools append a suffix to their input's stem, so the final
        # file of a chain is a leaf: no other file extends its name. Send-mode
        # outputs (*_send.wav) are wet-only returns and never a track's signal.
        files = [f for f in track_dir.glob("assembled*.wav") if not f.stem.endswith("_send")]
        leaves = [f for f in files if not any(o.stem.startswith(f.stem + "_") for o in files)]
        if not leaves:
            return None
        if len(leaves) > 1:
            print(f"  WARNING: {track_dir.name}: several processing branches "
                  f"({', '.join(sorted(f.name for f in leaves))}); using the longest chain",
                  file=sys.stderr)
        return max(leaves, key=lambda f: (f.stem.count("_"), f.name))

    for name in _SHARED_STAGE_CANDIDATES.get(stage, []):
        p = track_dir / name
        if p.exists():
            return p
    return None


def _mix_config_tracks(session_dir: Path) -> dict[str, dict]:
    """Track entries of session_dir/mix_config.json by name ({} if absent)."""
    path = session_dir / "mix_config.json"
    if not path.exists():
        return {}
    config = json.loads(path.read_text(encoding="utf-8"))
    return {t["name"]: t for t in config.get("tracks", []) if "name" in t}


def discover_stems(session_dir: Path, stage: str) -> dict[str, Path]:
    """Find the best assembled WAV per track in session_dir/tracks/."""
    tracks_dir = session_dir / "tracks"
    if not tracks_dir.exists():
        return {}

    config_tracks = _mix_config_tracks(session_dir)
    stems: dict[str, Path] = {}
    for track_dir in sorted(tracks_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        entry = config_tracks.get(track_dir.name)
        if entry is not None and not entry.get("active", True):
            continue
        if stage == "fx" and entry is not None and Path(entry.get("file", "")).is_file():
            stems[track_dir.name] = Path(entry["file"])
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


_ACTIVITY_FRAME_MS = 100.0
_ACTIVITY_THRESHOLD_DB = -45.0


def _activity_envelope(mono: np.ndarray, sr: int) -> np.ndarray:
    """Boolean array (one per ~100ms frame): True if stem is active in that frame.

    "Active" = frame RMS above _ACTIVITY_THRESHOLD_DB. Used to detect frames
    where two stems play simultaneously — only those frames matter for masking.
    """
    frame_len = int(_ACTIVITY_FRAME_MS * sr / 1000.0)
    n_frames = len(mono) // frame_len
    if n_frames == 0:
        return np.zeros(0, dtype=bool)
    thresh_lin = 10.0 ** (_ACTIVITY_THRESHOLD_DB / 20.0)
    rms = np.array([
        np.sqrt(np.mean(mono[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])
    return rms > thresh_lin


def _gated_audio(mono: np.ndarray, sr: int, active: np.ndarray) -> np.ndarray:
    """Concatenate only the active frames into a single buffer for PSD analysis."""
    frame_len = int(_ACTIVITY_FRAME_MS * sr / 1000.0)
    chunks = [mono[i * frame_len:(i + 1) * frame_len] for i, on in enumerate(active) if on]
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks)


def _third_octave_psd_db(mono: np.ndarray, sr: int) -> list[dict]:
    """1/3-octave band power in dB re full scale (PSD summed over band bins).

    Band power, unlike the mean PSD density, does not depend on the FFT bin
    width, so absolute floors are meaningful: a full-scale sine reads -3 dB.
    """
    if len(mono) < 1024:
        return []
    nperseg = min(len(mono), 32768)
    freqs, psd = welch(mono, fs=sr, nperseg=nperseg, average="mean")
    bin_hz = freqs[1] - freqs[0]

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
            power = float(np.sum(psd[mask]) * bin_hz)
            bands.append({"hz": round(fc, 1), "db": round(10.0 * np.log10(power + 1e-20), 1)})
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


_MIN_COACTIVITY_RATIO = 0.15
# Bands below this 1/3-octave band power (dB re full scale, after LUFS
# normalization) are treated as inactive for that stem.
_DEFAULT_FLOOR_DB = -50.0


def _coactivity_ratio(env_a: np.ndarray, env_b: np.ndarray) -> float:
    """Overlap coefficient: shared active frames / active frames of the shorter part.

    A solo playing 10% of the song entirely over a sustained bass scores 1.0
    (it is always accompanied), while verse-only vs chorus-only parts score 0.
    """
    n = min(len(env_a), len(env_b))
    if n == 0:
        return 0.0
    a = env_a[:n]
    b = env_b[:n]
    smaller = min(int(np.sum(a)), int(np.sum(b)))
    if smaller == 0:
        return 0.0
    intersect = int(np.sum(a & b))
    return float(intersect / smaller)


def find_masking_pairs(
    stem_bands: dict[str, list[dict]],
    stem_activity: dict[str, np.ndarray] | None = None,
    threshold_db: float = 6.0,
    floor_db: float = _DEFAULT_FLOOR_DB,
) -> list[dict]:
    """For each 1/3-octave band, find stem pairs within threshold_db of each other.

    Band levels are band power in dB; bands below floor_db are ignored.
    When stem_activity is supplied, a pair is suppressed if the two stems
    don't co-occur in time often enough (overlap coefficient below
    _MIN_COACTIVITY_RATIO), e.g. "rhythm guitar (verses only) vs lead vocal
    (choruses only)".
    """
    # Build hz → {stem: db} index
    hz_index: dict[float, dict[str, float]] = {}
    for stem, bands in stem_bands.items():
        for b in bands:
            hz = b["hz"]
            db = b["db"]
            if db < floor_db:
                continue
            hz_index.setdefault(hz, {})[stem] = db

    # Pre-compute pair co-activity once per pair
    coactivity: dict[tuple[str, str], float] = {}
    if stem_activity:
        names = list(stem_activity.keys())
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                coactivity[tuple(sorted([a, b]))] = _coactivity_ratio(
                    stem_activity[a], stem_activity[b]
                )

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
                # Time-gating: require the pair to actually co-occur
                if coactivity:
                    co = coactivity.get(pair_key, 1.0)
                    if co < _MIN_COACTIVITY_RATIO:
                        continue
                seen.add(pair_key)
                event = {
                    "hz": hz,
                    "severity": sev,
                    "gap_db": round(gap, 1),
                    "stems": [
                        {"name": stem_a, "db": round(db_a, 1)},
                        {"name": stem_b, "db": round(db_b, 1)},
                    ],
                }
                if coactivity:
                    event["coactivity"] = round(coactivity.get(pair_key, 0.0), 2)
                masking_events.append(event)

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
    lines = ["", "STEM ENERGY HEATMAP (mean 1/3-octave band power per region, after LUFS normalization)",
             sep, header, sep]

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
            # Map -60..-20 dB band power to 0..4 block index
            level = int(np.clip((avg_db + 60) / 10, 0, 4))
            row += f"{_BLOCKS[level]:^{col_w}}"
        lines.append(row)

    lines.append(sep)
    lines.append("  Band power scale (dB re full scale): (space)<-50  ░ -50..-40  ▒ -40..-30  ▓ -30..-20  █ >=-20")
    return "\n".join(lines)


def _masking_section(events: list[dict], threshold_db: float) -> str:
    if not events:
        return "\nNo candidate overlap pairs within the threshold."

    lines = [f"\nCANDIDATE OVERLAP PAIRS  (gap threshold: {threshold_db:.0f} dB)", "=" * 60]
    current_sev = None

    moderate_shown = 0
    moderate_limit = 15

    for e in events:
        if e["severity"] != current_sev:
            current_sev = e["severity"]
            sev_desc = {
                "CRITICAL": "CRITICAL  (< 3 dB gap)",
                "HIGH":     "HIGH      (3-6 dB gap)",
                "MODERATE": f"MODERATE  (6-10 dB gap, top {moderate_limit} shown)",
            }.get(current_sev, current_sev)
            lines += ["", f"── {sev_desc} ──────────────────────"]

        if e["severity"] == "MODERATE":
            if moderate_shown >= moderate_limit:
                continue
            moderate_shown += 1

        hz_lbl = _hz_label(e["hz"])
        a, b = e["stems"][0], e["stems"][1]
        co_lbl = f"  co={e['coactivity']:.2f}" if "coactivity" in e else ""
        lines.append(
            f"  {hz_lbl}  {a['name']} ({a['db']:+.1f} dB)  vs  "
            f"{b['name']} ({b['db']:+.1f} dB)  [{e['gap_db']:.1f} dB gap]{co_lbl}"
        )

    lines += [
        "",
        "HOW TO USE THESE CANDIDATES",
        "─" * 60,
        "  Each pair is a hypothesis: two parts with similar normalized band power",
        "  that play at the same time. Listen to the flagged passage in the actual",
        "  mix balance before changing anything; many pairs are intentional blends.",
        "  If an overlap is heard as a problem, compare a small bounded change",
        "  (balance, a narrow EQ cut, or a dynamic EQ keyed from the other part)",
        "  against the unprocessed version at matched loudness.",
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

_NORMALIZATION_NOTE = (
    "Every stem was normalized to {lufs:.0f} LUFS before comparison, so the actual "
    "mix balance is not represented; listen in context before acting on a pair."
)

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
    stem_activity: dict[str, np.ndarray] = {}
    stem_names = list(stems.keys())

    for i, (name, path) in enumerate(stems.items(), 1):
        print(f"  [{i}/{len(stems)}] {name}", flush=True)
        mono, sr = _load_mono(path)
        mono = _lufs_normalize(mono, sr, lufs_target)
        activity = _activity_envelope(mono, sr)
        stem_activity[name] = activity
        # PSD on the active portion only — silent regions don't contribute
        gated = _gated_audio(mono, sr, activity)
        if len(gated) > 0:
            stem_bands[name] = _third_octave_psd_db(gated, sr)
        else:
            stem_bands[name] = []

    print("Comparing band power between stems...", flush=True)
    events = find_masking_pairs(stem_bands, stem_activity=stem_activity, threshold_db=threshold_db)

    report = {
        "stems": {name: str(path) for name, path in stems.items()},
        "stage": stage,
        "threshold_db": threshold_db,
        "lufs_normalization_target": lufs_target,
        "band_level_unit": "1/3-octave band power, dB re full scale",
        "normalization_note": _NORMALIZATION_NOTE.format(lufs=lufs_target),
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
        "FREQUENCY OVERLAP CANDIDATES",
        "=" * 60,
        f"  Stage     : {stage}",
        f"  Stems     : {len(stems)}",
        f"  Threshold : {threshold_db:.0f} dB gap",
        f"  Pairs     : {summary_line}",
        f"  Note      : {_NORMALIZATION_NOTE.format(lufs=lufs_target)}",
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
        description="Find candidate frequency overlap between stems; hypotheses to check by listening.",
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
