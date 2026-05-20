#!/usr/bin/env python3
"""
render_mix.py -- sum processed stems into a stereo mix

Usage:
  render_mix.py output/<session> --generate-config [--config mix_config.json]
  render_mix.py mix_config.json --render [--output mix.wav]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import Compressor, Gain, Limiter, Pedalboard, Reverb
from scipy.signal import butter as _butter, resample_poly as _resample_poly, sosfilt as _sosfilt, sosfiltfilt as _sosfiltfilt

# Optional reverb support — loaded from apply_reverb in same directory
try:
    _tools_dir = str(Path(__file__).parent)
    if _tools_dir not in sys.path:
        sys.path.insert(0, _tools_dir)
    from apply_reverb import PRESETS as _REVERB_PRESETS, _gate as _reverb_gate
    _HAS_REVERB = True
except ImportError:
    _HAS_REVERB = False

try:
    from apply_eq import _hp_sos, _lp_sos, _lowshelf_sos, _highshelf_sos, _peak_sos
    _HAS_EQ = True
except ImportError:
    _HAS_EQ = False

from _stages import STAGE_CANDIDATES as _STAGE_CANDIDATE_LISTS, STAGE_NAMES

PRESETS_DIR = Path(__file__).parent / "presets"

# Guitar guitarist prefixes in specificity order (most specific first)
_GUITARIST_PREFIXES = ["GTR 1", "GTR LACI", "GTR TERKA", "GTR"]

_DRUM_KEYWORDS = ["KICK", "SN ", " SN", "OH ", " OH", "TOM", "HIHAT", "HI-HAT",
                  "CRASH", "RIDE", "ROOM", "CYMBAL"]


def _detect_bus(name: str) -> str:
    u = name.upper()
    if any(k in u for k in _DRUM_KEYWORDS):
        return "drums"
    if "BASS" in u:
        return "bass"
    for prefix in _GUITARIST_PREFIXES:
        if u.startswith(prefix):
            return prefix.lower().replace(" ", "_")
    return "master"


def _detect_pan(name: str) -> float:
    u = name.upper()
    if re.search(r'\bL\b', u):
        return -0.7
    if re.search(r'\bR\b', u):
        return 0.7
    return 0.0


def _guitarist_prefix(name: str) -> str | None:
    u = name.upper()
    for p in _GUITARIST_PREFIXES:
        if u.startswith(p):
            return p
    return None


def _mic_type(name: str, prefix: str) -> str:
    # "GTR 1 FENDER.dup1.06" -> strip prefix -> "FENDER.dup1.06" -> strip version -> "FENDER"
    rest = name[len(prefix):].strip()
    mic = re.sub(r'\.?(dup\d+\.)?\d+$', '', rest).strip()
    return mic if mic else "DI"


def _find_final_file(track_dir: Path) -> str | None:
    for name in ["assembled_aligned_eq_comp.wav", "assembled_eq_comp.wav",
                 "assembled_aligned_eq.wav", "assembled_eq.wav",
                 "assembled_aligned.wav", "assembled.wav"]:
        p = track_dir / name
        if p.exists():
            return str(p)
    return None


def _resolve_stage_file(config_file: str, stage: str) -> str:
    """Return the stem file for the requested processing stage.

    Searches for the stage-appropriate file in the same directory as config_file.
    Falls back to config_file with a warning if the stage file doesn't exist.
    The "fx" stage uses the config file as-is (final stem, before bus processing).
    """
    if stage == "fx":
        return config_file
    candidates = _STAGE_CANDIDATE_LISTS.get(stage, [])
    track_dir = Path(config_file).parent
    for name in candidates:
        p = track_dir / name
        if p.exists():
            return str(p)
    print(f"    WARNING: no '{stage}' stage file found in {track_dir} — using config file")
    return config_file


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _load_style_bus_defaults(style: str | None) -> dict[str, float]:
    """Look up `default_bus_volume_db` from the named style profile.

    Returns {} if `style` is None or the profile doesn't exist / lacks
    the field. Callers should fall back to 0.0 for any bus not in the
    returned dict.
    """
    if not style:
        return {}
    profile_path = Path(__file__).resolve().parent / "style_profiles" / f"{style}.json"
    if not profile_path.exists():
        print(f"WARNING: style profile '{style}' not found at {profile_path} — "
              f"using neutral 0 dB bus defaults", file=sys.stderr)
        return {}
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read style profile '{style}': {exc}", file=sys.stderr)
        return {}
    return profile.get("default_bus_volume_db", {})


def generate_config(session_dir: Path, output_path: Path, style: str | None = None) -> None:
    sr = 48000
    session_json = session_dir / "session.json"
    if session_json.exists():
        with open(session_json) as f:
            sr = json.load(f).get("sample_rate", 48000)

    style_bus_defaults = _load_style_bus_defaults(style)
    if style_bus_defaults:
        print(f"Using style '{style}' bus defaults: "
              + ", ".join(f"{k}={v:+.1f}" for k, v in style_bus_defaults.items()))

    tracks_root = session_dir / "tracks"
    scan_dir = tracks_root if tracks_root.is_dir() else session_dir

    tracks = []
    for track_dir in sorted(scan_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        final_file = _find_final_file(track_dir)
        if not final_file:
            continue
        name = track_dir.name
        tracks.append({
            "name": name,
            "file": final_file,
            "active": True,
            "bus": _detect_bus(name),
            "blend_group": None,
            "volume_db": 0.0,
            "pan": _detect_pan(name),
        })

    # Assign blend groups and normalize volumes for guitar multi-mic tracks.
    # Count distinct mic types per guitarist (not dup variants) to determine
    # the normalization factor: volume = -20*log10(n_mic_types).
    gtr_mic_types: dict[str, set] = defaultdict(set)
    for t in tracks:
        prefix = _guitarist_prefix(t["name"])
        if prefix:
            gtr_mic_types[prefix].add(_mic_type(t["name"], prefix))

    for t in tracks:
        prefix = _guitarist_prefix(t["name"])
        if prefix:
            slug = prefix.lower().replace(" ", "_")
            t["blend_group"] = f"{slug}_mics"
            n = len(gtr_mic_types[prefix])
            t["volume_db"] = round(-20.0 * float(np.log10(max(n, 1))), 1)

    # Build bus hierarchy: per-guitarist sub-buses feed into a shared guitar bus
    gtr_sub_buses = {}
    for prefix in _GUITARIST_PREFIXES:
        slug = prefix.lower().replace(" ", "_")
        if any(t["bus"] == slug for t in tracks):
            gtr_sub_buses[slug] = {
                "volume_db": 0.0,
                "comp_preset": None,
                "parent_bus": "guitar",
            }

    # Top-level bus volume_db: use the style profile default if given, else 0 dB
    def _bus_default(name: str) -> float:
        return float(style_bus_defaults.get(name, 0.0))

    buses = {
        "drums":  {"volume_db": _bus_default("drums"),  "comp_preset": "comp_drum_bus", "parent_bus": None},
        "bass":   {"volume_db": _bus_default("bass"),   "comp_preset": None,            "parent_bus": None},
        **gtr_sub_buses,
        "guitar": {"volume_db": _bus_default("guitar"), "comp_preset": None,            "parent_bus": None},
    }

    mixes_dir = session_dir / "mixes"
    config = {
        "session_dir": str(session_dir),
        "output_dir": str(mixes_dir),
        "sample_rate": sr,
        "master": {
            "lufs_target": -14.0,
            "true_peak_dbfs": -2.0,
        },
        "buses": buses,
        "tracks": tracks,
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    n_gtr = sum(1 for t in tracks if _guitarist_prefix(t["name"]))
    n_drum = sum(1 for t in tracks if t["bus"] == "drums")
    n_bass = sum(1 for t in tracks if t["bus"] == "bass")
    print(f"Config written: {output_path}")
    print(f"  {len(tracks)} tracks: {n_drum} drums, {n_bass} bass, {n_gtr} guitar")
    print("  Guitar blend normalization:")
    for prefix, mics in sorted(gtr_mic_types.items()):
        n = len(mics)
        db = round(-20.0 * float(np.log10(n)), 1)
        print(f"    {prefix}: {sorted(mics)} ({n} mic types) -> {db:+.1f} dB/track")
    print(f"\n  Edit {output_path} to:")
    print("    - Set active=false for alternate takes you don't want (dup versions)")
    print("    - Adjust volume_db per track for mic blend (e.g. DI lower than amp mic)")
    print("    - Adjust pan per track (-1.0 full L, 0.0 center, 1.0 full R)")
    print("    - Adjust bus volume_db for relative group levels")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _topo_order(buses: dict) -> list[str]:
    """Return bus names in processing order: children before parents (post-order DFS)."""
    children: dict[str, list] = {name: [] for name in buses}
    for name, cfg in buses.items():
        parent = cfg.get("parent_bus")
        if parent and parent in children:
            children[parent].append(name)

    order: list[str] = []
    visited: set = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        for child in children.get(name, []):
            visit(child)
        order.append(name)

    for name in buses:
        visit(name)
    return order


def _load_as_stereo(path: Path, length: int, sr: int) -> np.ndarray:
    """Load WAV file as (2, length) float64 stereo array.

    Mono files are duplicated to both channels. Stereo (and multi-channel)
    files keep channels 0 and 1; any further channels are dropped.
    """
    data, file_sr = sf.read(str(path), always_2d=True)
    if file_sr != sr:
        raise ValueError(f"SR mismatch: {path} is {file_sr} Hz, expected {sr}")

    if data.shape[1] == 1:
        left = data[:, 0]
        right = left
    else:
        left = data[:, 0]
        right = data[:, 1]

    n = len(left)
    if n < length:
        pad = np.zeros(length - n)
        left = np.concatenate([left, pad])
        right = np.concatenate([right, pad])
    else:
        left = left[:length]
        right = right[:length]

    return np.vstack([left, right])  # (2, length)


def _pan(buf: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power pan. pan: -1.0 (full L) to 1.0 (full R). buf: (2, N)."""
    angle = (pan + 1.0) / 2.0 * (np.pi / 2.0)
    result = buf.copy()
    result[0] *= np.cos(angle)
    result[1] *= np.sin(angle)
    return result


def _apply_comp_preset(buf: np.ndarray, preset_name: str, sr: int) -> np.ndarray:
    path = PRESETS_DIR / f"{preset_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    with open(path) as f:
        p = json.load(f)
    s = p["settings"]
    makeup = float(s.get("makeup_db") or 0.0)
    board = Pedalboard([
        Compressor(
            threshold_db=float(s["threshold_db"]),
            ratio=float(s["ratio"]),
            attack_ms=float(s["attack_ms"]),
            release_ms=float(s["release_ms"]),
        ),
        Gain(gain_db=makeup),
    ])
    return board(buf.astype(np.float32), sr).astype(np.float64)


def _write_stem(buf: np.ndarray, path: Path, sr: int, lufs_target: float = -18.0) -> None:
    """Write a single bus buffer as a normalized stem WAV."""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(buf.T)
    if np.isfinite(loudness):
        buf = buf * 10.0 ** ((lufs_target - loudness) / 20.0)
    peak = np.max(np.abs(buf))
    if peak > 1.0:
        buf = buf / peak
    sf.write(str(path), buf.T, sr, subtype="PCM_24")


def _apply_bus_reverb(buf: np.ndarray, sr: int, preset_name: str, wet: float) -> np.ndarray:
    """Apply reverb send to a bus buffer. Returns wet-only reverb return.

    buf: (2, N) float64 stereo. Returns (2, N) reverb return scaled by wet.
    Preset parameters (room_size, damping, pre_delay_ms, hp_hz, lp_hz, gate)
    are loaded from apply_reverb.PRESETS.
    """
    p = _REVERB_PRESETS[preset_name]
    data = buf.T.copy()  # (N, 2) — filter processing uses axis=0

    pre_delay_samples = int(p.get("pre_delay_ms", 0.0) * sr / 1000.0)
    if pre_delay_samples > 0:
        delayed = np.concatenate([np.zeros((pre_delay_samples, 2)), data], axis=0)
    else:
        delayed = data

    board = Pedalboard([
        Reverb(
            room_size=p["room_size"],
            damping=p["damping"],
            wet_level=1.0,
            dry_level=0.0,
            width=p["width"],
        )
    ])
    reverb_out = board(delayed.T.astype(np.float32), sr).T.astype(np.float64)  # (N+pre, 2)

    if pre_delay_samples > 0:
        reverb_out = reverb_out[pre_delay_samples:]
    reverb_out = reverb_out[:len(data)]
    if len(reverb_out) < len(data):
        reverb_out = np.concatenate([reverb_out, np.zeros((len(data) - len(reverb_out), 2))], axis=0)

    hp_hz = p.get("hp_hz")
    lp_hz = p.get("lp_hz")
    if hp_hz:
        sos = _butter(2, hp_hz / (sr / 2.0), btype="high", output="sos")
        reverb_out = _sosfilt(sos, reverb_out, axis=0)
    if lp_hz:
        sos = _butter(2, lp_hz / (sr / 2.0), btype="low", output="sos")
        reverb_out = _sosfilt(sos, reverb_out, axis=0)

    gate_hold = p.get("gate_hold_ms")
    gate_release = p.get("gate_release_ms")
    if gate_hold and gate_release:
        reverb_out = _reverb_gate(reverb_out, sr, hold_ms=gate_hold, release_ms=gate_release)

    return (reverb_out * wet).T  # (2, N)


def _bus_parallel_sat_relevance_check(buf: np.ndarray, sr: int, bus_name: str) -> dict:
    """Decide whether parallel saturation on a drum bus would help.

    Conditions:
      - Bus crest factor > 10 dB (transient life left to saturate)
      - Bus is the drum bus (other instruments don't benefit the same way)
    """
    rms = float(np.sqrt(np.mean(buf ** 2)))
    peak = float(np.max(np.abs(buf)))
    crest_db = 20.0 * np.log10(peak / max(rms, 1e-10)) if rms > 1e-10 else 0.0

    meter = pyln.Meter(sr)
    try:
        lra = float(meter.loudness_range(buf.T))
    except Exception:
        lra = 0.0

    issues = []
    if crest_db < 10.0:
        issues.append(f"crest {crest_db:.1f} dB < 10 — bus already squashed; parallel sat adds fuzz, not punch")
    if lra < 4.0:
        issues.append(f"LRA {lra:.1f} LU < 4 — too compressed for parallel sat to add life")
    if bus_name.lower() != "drums":
        issues.append(f"bus is '{bus_name}', not 'drums' — parallel sat preset tuned for drum kit")

    return {
        "tool": "drum_bus_parallel_sat",
        "bus": bus_name,
        "crest_db": round(crest_db, 1),
        "lra_lu": round(lra, 1),
        "recommend_skip": bool(issues),
        "issues": issues,
    }


def _parallel_saturate(buf: np.ndarray, mode: str, drive: float, mix: float) -> np.ndarray:
    """Blend a saturated copy of `buf` back into the dry signal.

    mode: 'tube'   = asymmetric tanh (even harmonics, warmth)
          'tape'   = symmetric tanh   (odd+even, smooth)
          'clipper'= cubic soft clip  (odd, presence)
    drive: 0.0-1.0, amount of saturation push
    mix: 0.0-1.0, blend amount of the saturated copy on top of the dry
    """
    if drive <= 0 or mix <= 0:
        return buf
    in_rms = np.sqrt(np.mean(buf ** 2) + 1e-12)
    x = buf * (1.0 + drive * 3.0)
    if mode == "tube":
        # Asymmetric: positive side soft-clips earlier
        sat = np.where(x >= 0, np.tanh(x * 1.2), np.tanh(x))
    elif mode == "clipper":
        # Cubic soft clip — odd harmonics
        clipped = np.clip(x, -1.0, 1.0)
        sat = clipped - (clipped ** 3) / 3.0
    else:  # tape
        sat = np.tanh(x)
    out_rms = np.sqrt(np.mean(sat ** 2) + 1e-12)
    if out_rms > 1e-10:
        sat = sat * (in_rms / out_rms)
    return buf + sat * mix


def _tape_saturate(buf: np.ndarray, drive: float) -> np.ndarray:
    """Symmetric tanh soft clipping (tape saturation). buf: (2, N). RMS-normalized.

    Symmetric clipping generates odd-order harmonics (2nd, 3rd) — tape-like warmth
    without the tonal shift of asymmetric tube saturation. RMS normalization preserves
    level: only the spectral content changes.
    """
    if drive <= 0.0:
        return buf
    in_rms = np.sqrt(np.mean(buf ** 2) + 1e-12)
    x = buf * (1.0 + drive * 3.0)
    out = np.tanh(x)
    out_rms = np.sqrt(np.mean(out ** 2) + 1e-12)
    if out_rms > 1e-10:
        out = out * (in_rms / out_rms)
    return out


def _soft_clip(buf: np.ndarray, threshold_db: float, knee_db: float = 1.5) -> np.ndarray:
    """Cubic soft clipper. Above threshold the signal is gradually rounded
    over a knee_db transition; far above threshold it asymptotes to the
    ceiling. Generates mostly odd-order harmonics, like a console clipper.

    Use BEFORE the brick-wall limiter for modern loudness without "squashed"
    feel: clipper handles the peaks musically, limiter just catches strays.
    """
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    knee_lin = 10.0 ** (knee_db / 20.0)
    out = buf.copy()
    abs_buf = np.abs(buf)

    # Three zones: below knee start, inside knee, above knee
    knee_start = threshold_lin / knee_lin
    knee_end = threshold_lin * knee_lin

    # Knee region: smooth cubic transition
    in_knee = (abs_buf >= knee_start) & (abs_buf < knee_end)
    if in_knee.any():
        # Normalise position in the knee to [0, 1]
        x = (abs_buf[in_knee] - knee_start) / (knee_end - knee_start)
        # Cubic ease-out that asymptotes near the ceiling
        gain = 1.0 - x * x * (3.0 - 2.0 * x) * (1.0 - threshold_lin / abs_buf[in_knee])
        out[in_knee] = np.sign(buf[in_knee]) * abs_buf[in_knee] * gain

    # Above-knee region: hard ceiling at threshold
    above = abs_buf >= knee_end
    if above.any():
        out[above] = np.sign(buf[above]) * threshold_lin

    return out


def _hard_clip(buf: np.ndarray, threshold_db: float) -> np.ndarray:
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    return np.clip(buf, -threshold_lin, threshold_lin)


def _clipper_relevance_check(master: np.ndarray, sr: int) -> dict:
    """Decide whether a clipper would do anything useful BEFORE applying it.

    Conditions for use:
      - Sample peak headroom > 2 dB (so the clipper has somewhere to work)
      - LRA > 4 LU after glue comp (so we're not crushing already-flat material)
    """
    sample_peak_db = 20.0 * np.log10(float(np.max(np.abs(master))) + 1e-12)
    meter = pyln.Meter(sr)
    try:
        lra = float(meter.loudness_range(master.T))
    except Exception:
        lra = 0.0

    issues = []
    if sample_peak_db < -10:
        issues.append(f"peak {sample_peak_db:.1f} dBFS — nothing for clipper to clip")
    if lra < 4.0:
        issues.append(f"LRA {lra:.1f} LU — already too compressed, clipper will add fatigue")

    return {
        "tool": "master_clipper",
        "sample_peak_dbfs": round(sample_peak_db, 2),
        "lra_lu": round(lra, 1),
        "recommend_skip": bool(issues),
        "issues": issues,
    }


def _measure_true_peak_dbfs(master: np.ndarray, oversample: int = 4) -> float:
    """4x-oversampled true peak in dBFS for a (2, N) stereo buffer.

    pedalboard.Limiter only constrains the sample peak; inter-sample peaks
    can still exceed the ceiling after codec encoding (Spotify Ogg/Vorbis,
    Apple AAC). This second-pass measurement reveals them.
    """
    up_l = _resample_poly(master[0], oversample, 1)
    up_r = _resample_poly(master[1], oversample, 1)
    tp = max(float(np.max(np.abs(up_l))), float(np.max(np.abs(up_r))))
    return 20.0 * np.log10(max(tp, 1e-12))


def _ms_encode(master: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """L/R stereo → (mid, side). Energy-preserving (no 1/sqrt(2) factor)."""
    mid = (master[0] + master[1]) * 0.5
    side = (master[0] - master[1]) * 0.5
    return mid, side


def _ms_decode(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    """(mid, side) → L/R stereo."""
    L = mid + side
    R = mid - side
    return np.vstack([L, R])


def _ms_relevance_check(master: np.ndarray, ms_cfg: dict, sr: int) -> dict:
    """Decide whether M/S processing would help.

    Conditions:
      - The mix has actual side content (rms(side) / rms(mid) > 0.05).
        A near-mono mix won't benefit and could just create stereo problems.
      - If side gain > 0 dB is configured, the existing width can't already be huge
        (ms_width_ratio < 0.5), or we risk breaking mono compatibility.
    """
    mid, side = _ms_encode(master)
    rms_m = float(np.sqrt(np.mean(mid ** 2) + 1e-12))
    rms_s = float(np.sqrt(np.mean(side ** 2) + 1e-12))
    width_ratio = rms_s / max(rms_m, 1e-10)

    issues = []
    if width_ratio < 0.05:
        issues.append(
            f"M/S width {width_ratio:.3f} < 0.05 — mix is near-mono, M/S processing has no audible target"
        )
    side_eq = ms_cfg.get("side_eq", [])
    side_boost = any(f.get("db", 0) > 0 for f in side_eq)
    if side_boost and width_ratio > 0.5:
        issues.append(
            f"side EQ boost requested but width {width_ratio:.3f} > 0.5 — risk of mono-compat breakage"
        )

    return {
        "tool": "ms_processing",
        "ms_width_ratio": round(width_ratio, 3),
        "recommend_skip": bool(issues),
        "issues": issues,
    }


def _ms_apply_eq(channel: np.ndarray, sr: int, filters: list) -> np.ndarray:
    """Apply EQ chain to a single mono channel (mid or side). Zero-phase."""
    if not _HAS_EQ or not filters:
        return channel
    out = channel.copy()
    for f in filters:
        ftype = f.get("type", "")
        if ftype == "highpass":
            sos = _hp_sos(f["hz"], f.get("order", 2), sr)
        elif ftype == "lowpass":
            sos = _lp_sos(f["hz"], f.get("order", 2), sr)
        elif ftype == "highshelf":
            sos = _highshelf_sos(f["hz"], f["db"], f.get("slope", 1.0), sr)
        elif ftype == "lowshelf":
            sos = _lowshelf_sos(f["hz"], f["db"], f.get("slope", 1.0), sr)
        elif ftype == "peak":
            sos = _peak_sos(f["hz"], f.get("q", 1.0), f["db"], sr)
        else:
            continue
        out = _sosfiltfilt(sos, out)
    return out


def _apply_ms_processing(master: np.ndarray, sr: int, ms_cfg: dict) -> tuple[np.ndarray, dict]:
    """Run M/S processing on the master chain. Returns (processed_master, report)."""
    rel = _ms_relevance_check(master, ms_cfg, sr)
    if rel["recommend_skip"]:
        return master, {"settings": ms_cfg, "relevance_check": rel, "applied": False}

    mid, side = _ms_encode(master)

    # Mid/Side independent gains
    mid_gain_db = float(ms_cfg.get("mid_gain_db", 0.0))
    side_gain_db = float(ms_cfg.get("side_gain_db", 0.0))
    if mid_gain_db != 0.0:
        mid = mid * 10.0 ** (mid_gain_db / 20.0)
    if side_gain_db != 0.0:
        side = side * 10.0 ** (side_gain_db / 20.0)

    # Mid/Side independent EQ
    mid = _ms_apply_eq(mid, sr, ms_cfg.get("mid_eq", []))
    side = _ms_apply_eq(side, sr, ms_cfg.get("side_eq", []))

    return _ms_decode(mid, side), {
        "settings": ms_cfg,
        "relevance_check": rel,
        "applied": True,
    }


def _apply_eq_chain(buf: np.ndarray, sr: int, filters: list, label: str = "EQ") -> np.ndarray:
    """Apply a chain of EQ filters to a stereo buffer. buf: (2, N). Zero-phase (sosfiltfilt).

    `label` is used only for warning prints (e.g. "master EQ" or "bus 'drums' EQ").
    """
    if not _HAS_EQ:
        print(f"  WARNING: apply_eq.py not importable — skipping {label}")
        return buf
    out = buf.copy()
    for f in filters:
        ftype = f.get("type", "")
        if ftype == "highpass":
            sos = _hp_sos(f["hz"], f.get("order", 2), sr)
        elif ftype == "lowpass":
            sos = _lp_sos(f["hz"], f.get("order", 2), sr)
        elif ftype == "highshelf":
            sos = _highshelf_sos(f["hz"], f["db"], f.get("slope", 1.0), sr)
        elif ftype == "lowshelf":
            sos = _lowshelf_sos(f["hz"], f["db"], f.get("slope", 1.0), sr)
        elif ftype == "peak":
            sos = _peak_sos(f["hz"], f.get("q", 1.0), f["db"], sr)
        else:
            print(f"  WARNING: unknown {label} filter type '{ftype}' — skipped")
            continue
        out = _sosfiltfilt(sos, out, axis=1)
    return out


# Backwards-compat alias — older code calls _apply_master_eq
def _apply_master_eq(master: np.ndarray, sr: int, filters: list) -> np.ndarray:
    return _apply_eq_chain(master, sr, filters, label="master EQ")


def render_mix(config_path: Path, output_wav: Path | None = None, render_stems: bool = False, stage: str | None = None) -> None:
    with open(config_path) as f:
        config = json.load(f)

    sr: int = config["sample_rate"]
    master_cfg: dict = config["master"]
    buses_cfg: dict = config["buses"]
    active = [t for t in config["tracks"] if t.get("active", True)]

    if not active:
        raise ValueError("No active tracks in config")

    print(f"Active tracks: {len(active)} / {len(config['tracks'])}")

    if stage:
        print(f"Stage: '{stage}'")
        resolved = []
        for t in active:
            sf_ = _resolve_stage_file(t["file"], stage)
            resolved.append({**t, "file": sf_} if sf_ != t["file"] else t)
        subs = sum(1 for a, b in zip(active, resolved) if a["file"] != b["file"])
        active = resolved
        print(f"  {subs}/{len(active)} stems substituted to '{stage}' stage files")
        print()

    max_length = max(sf.info(t["file"]).frames for t in active)
    print(f"Mix length: {max_length / sr:.1f}s ({max_length} samples at {sr} Hz)")
    print()

    # Load tracks and sum into bus buffers
    bus_buffers: dict[str, np.ndarray] = {}
    for t in active:
        name = t["name"]
        bus = t.get("bus", "master")
        vol = 10.0 ** (t.get("volume_db", 0.0) / 20.0)
        p = t.get("pan", 0.0)
        bg = t.get("blend_group") or ""
        bg_label = f" [{bg}]" if bg else ""
        polarity = -1.0 if t.get("polarity_flip", False) else 1.0
        pol_label = " [FLIP]" if polarity < 0 else ""
        print(f"  {name}{bg_label}{pol_label}")
        print(f"    bus:{bus}  vol:{t.get('volume_db', 0.0):+.1f}dB  pan:{p:+.2f}")

        stereo = _load_as_stereo(Path(t["file"]), max_length, sr)
        stereo = _pan(stereo * vol * polarity, p)

        if bus not in bus_buffers:
            bus_buffers[bus] = np.zeros((2, max_length))
        bus_buffers[bus] += stereo

    print()

    # Process buses in topological order (leaves first, parents after)
    processed: dict[str, np.ndarray] = {}
    for bus_name in _topo_order(buses_cfg):
        cfg = buses_cfg[bus_name]

        buf = bus_buffers.get(bus_name, np.zeros((2, max_length))).copy()

        # Add processed child buses that route into this bus
        for child, child_cfg in buses_cfg.items():
            if child_cfg.get("parent_bus") == bus_name and child in processed:
                buf += processed[child]

        if not np.any(buf):
            continue

        buf *= 10.0 ** (cfg.get("volume_db", 0.0) / 20.0)

        bus_pan = cfg.get("pan", 0.0)
        if bus_pan != 0.0:
            buf = _pan(buf, bus_pan)

        bus_eq = cfg.get("eq")
        if bus_eq:
            buf = _apply_eq_chain(buf, sr, bus_eq, label=f"bus '{bus_name}' EQ")
            filter_summary = ", ".join(
                f"{f.get('type', '?')}@{f.get('hz', '?')}Hz"
                + (f" {f.get('db', 0):+.1f}dB" if f.get("db") is not None else "")
                for f in bus_eq
            )
            print(f"  Bus '{bus_name}': + EQ ({filter_summary})")

        preset = cfg.get("comp_preset")
        if preset:
            peak_in = 20.0 * np.log10(np.max(np.abs(buf)) + 1e-12)
            print(f"  Bus '{bus_name}': compressing with preset '{preset}' (peak in: {peak_in:.1f} dBFS)...",
                  end=" ", flush=True)
            buf = _apply_comp_preset(buf, preset, sr)
            peak_out = 20.0 * np.log10(np.max(np.abs(buf)) + 1e-12)
            print(f"-> {peak_out:.1f} dBFS")
        else:
            peak = 20.0 * np.log10(np.max(np.abs(buf)) + 1e-12)
            print(f"  Bus '{bus_name}': peak {peak:.1f} dBFS")

        sat_cfg = cfg.get("saturation")
        if sat_cfg:
            sat_drive = float(sat_cfg.get("drive", 0.3))
            buf = _tape_saturate(buf, sat_drive)
            print(f"  Bus '{bus_name}': + tape saturation drive={sat_drive}")

        # Parallel saturation: blend a saturated copy back in. Make-it-hit
        # tool — guarded by relevance_check (drum bus only, must have crest).
        psat_cfg = cfg.get("parallel_saturation")
        if psat_cfg:
            rel = _bus_parallel_sat_relevance_check(buf, sr, bus_name)
            if rel["recommend_skip"]:
                print(f"  Bus '{bus_name}': parallel sat SKIPPED — {'; '.join(rel['issues'])}")
            else:
                mode = str(psat_cfg.get("mode", "tube"))
                drive = float(psat_cfg.get("drive", 0.5))
                mix = float(psat_cfg.get("mix", 0.2))
                buf = _parallel_saturate(buf, mode, drive, mix)
                print(f"  Bus '{bus_name}': + parallel sat ({mode}, drive={drive}, mix={mix})")

        reverb_cfg = cfg.get("reverb_send")
        if reverb_cfg:
            if not _HAS_REVERB:
                print(f"  WARNING: apply_reverb.py not importable — skipping reverb send for '{bus_name}'")
            else:
                rv_preset = reverb_cfg["preset"]
                rv_wet = float(reverb_cfg.get("wet", 0.15))
                reverb_return = _apply_bus_reverb(buf, sr, rv_preset, rv_wet)
                buf = buf + reverb_return
                peak_rv = 20.0 * np.log10(np.max(np.abs(reverb_return)) + 1e-12)
                print(f"  Bus '{bus_name}': + reverb send '{rv_preset}' wet={rv_wet} (return peak {peak_rv:.1f} dBFS)")

        processed[bus_name] = buf

    # Sum top-level buses (parent_bus: null) into master
    master = np.zeros((2, max_length))
    for bus_name, cfg in buses_cfg.items():
        if not cfg.get("parent_bus") and bus_name in processed:
            master += processed[bus_name]

    peak_pre = 20.0 * np.log10(np.max(np.abs(master)) + 1e-12)
    print(f"\n  Master pre-processing: {peak_pre:.1f} dBFS")

    meter = pyln.Meter(sr)

    # Master bus glue compressor (optional)
    master_comp = master_cfg.get("comp")
    if master_comp:
        comp_board = Pedalboard([
            Compressor(
                threshold_db=float(master_comp["threshold_db"]),
                ratio=float(master_comp.get("ratio", 2.0)),
                attack_ms=float(master_comp.get("attack_ms", 10.0)),
                release_ms=float(master_comp.get("release_ms", 300.0)),
            ),
            Gain(gain_db=float(master_comp.get("makeup_db", 0.0))),
        ])
        lufs_pre_comp = meter.integrated_loudness(master.T)
        master = comp_board(master.astype(np.float32), sr).astype(np.float64)

        lufs_post_comp = meter.integrated_loudness(master.T)
        peak_post_comp = 20.0 * np.log10(np.max(np.abs(master)) + 1e-12)
        gr_lufs = lufs_post_comp - lufs_pre_comp
        print(f"  Master comp: {master_comp['threshold_db']}dB threshold  {master_comp.get('ratio', 2.0)}:1  "
              f"att={master_comp.get('attack_ms', 10.0)}ms  rel={master_comp.get('release_ms', 300.0)}ms  "
              f"-> GR {gr_lufs:+.1f} LUFS  peak {peak_post_comp:.1f} dBFS")

    # Master clipper (optional). Place between glue comp and EQ — clipper
    # shapes peaks musically; the brick-wall limiter at the end only catches
    # what's left. Skipped automatically if relevance_check flags issues.
    clipper_cfg = master_cfg.get("clipper")
    clipper_report: dict | None = None
    if clipper_cfg:
        rel = _clipper_relevance_check(master, sr)
        clipper_report = {"settings": clipper_cfg, "relevance_check": rel}
        if rel["recommend_skip"]:
            print(f"  Clipper: SKIPPED — {'; '.join(rel['issues'])}")
        else:
            mode = str(clipper_cfg.get("mode", "soft")).lower()
            threshold = float(clipper_cfg.get("threshold_db", -1.0))
            knee = float(clipper_cfg.get("knee_db", 1.5))
            peak_pre = 20.0 * np.log10(np.max(np.abs(master)) + 1e-12)
            if mode == "hard":
                master = _hard_clip(master, threshold)
            else:
                master = _soft_clip(master, threshold, knee)
            peak_post = 20.0 * np.log10(np.max(np.abs(master)) + 1e-12)
            print(f"  Master clipper ({mode}): threshold {threshold:.1f} dBFS  "
                  f"knee {knee:.1f} dB  -> peak {peak_pre:.1f} → {peak_post:.1f} dBFS")
            clipper_report["applied"] = True

    # M/S processing (optional). Place AFTER glue comp + clipper, BEFORE
    # final master EQ — gives M/S the cleanest input to widen, then the
    # master EQ still has the last spectral say.
    ms_cfg = master_cfg.get("ms")
    ms_report: dict | None = None
    if ms_cfg:
        master, ms_report = _apply_ms_processing(master, sr, ms_cfg)
        if ms_report.get("applied"):
            mid_g = ms_cfg.get("mid_gain_db", 0.0)
            side_g = ms_cfg.get("side_gain_db", 0.0)
            print(f"  M/S: mid {mid_g:+.1f} dB, side {side_g:+.1f} dB  "
                  f"(width ratio {ms_report['relevance_check']['ms_width_ratio']:.3f})")
        else:
            issues = ms_report["relevance_check"]["issues"]
            print(f"  M/S: SKIPPED — {'; '.join(issues)}")

    # Master bus EQ (optional, zero-phase)
    master_eq = master_cfg.get("eq")
    if master_eq:
        peak_pre_eq = 20.0 * np.log10(np.max(np.abs(master)) + 1e-12)
        master = _apply_master_eq(master, sr, master_eq)
        peak_post_eq = 20.0 * np.log10(np.max(np.abs(master)) + 1e-12)
        filter_desc = ", ".join(
            f"{f.get('type','?')} {f.get('hz', '')}Hz"
            + (f" {f['db']:+.1f}dB" if "db" in f else "")
            for f in master_eq
        )
        print(f"  Master EQ: {filter_desc}  -> peak {peak_post_eq:.1f} dBFS")

    # LUFS normalization
    target_lufs = master_cfg.get("lufs_target", -14.0)
    loudness = meter.integrated_loudness(master.T)
    print(f"  Integrated loudness: {loudness:.1f} LUFS -> target {target_lufs} LUFS")
    if np.isfinite(loudness):
        gain_db = target_lufs - loudness
        master *= 10.0 ** (gain_db / 20.0)
        print(f"  Applied gain: {gain_db:+.1f} dB")

    # True peak limiting: pedalboard.Limiter handles sample peak; we then
    # measure ISP at 4x oversampling and scale down if the codec-relevant
    # true peak still exceeds the ceiling.
    tp_limit = master_cfg.get("true_peak_dbfs", -2.0)
    board = Pedalboard([Limiter(threshold_db=float(tp_limit), release_ms=100.0)])
    master = board(master.astype(np.float32), sr).astype(np.float64)

    sample_peak_after = 20.0 * np.log10(np.max(np.abs(master)) + 1e-12)
    tp_measured = _measure_true_peak_dbfs(master)
    if tp_measured > tp_limit:
        isp_scale_db = tp_limit - tp_measured
        master *= 10.0 ** (isp_scale_db / 20.0)
        tp_final = tp_limit
        print(f"  Peak after limiter: sample {sample_peak_after:.2f} dBFS  "
              f"true {tp_measured:.2f} dBTP -> {tp_final:.2f} dBTP "
              f"(ISP correction {isp_scale_db:+.2f} dB)")
    else:
        tp_final = tp_measured
        print(f"  Peak after limiter: sample {sample_peak_after:.2f} dBFS  "
              f"true {tp_final:.2f} dBTP")
    peak_final = tp_final

    # Post-limiter LUFS correction. pedalboard.Limiter applies automatic
    # makeup gain that pushes sample peak up to the threshold, so the
    # post-limiter loudness drifts above target. Re-measure and re-apply
    # gain if the actual LUFS deviates by more than 0.3 LU, while keeping
    # true peak within the ceiling.
    loudness_post = meter.integrated_loudness(master.T)
    if np.isfinite(loudness_post):
        delta_post = target_lufs - loudness_post
        if abs(delta_post) > 0.3:
            current_tp = _measure_true_peak_dbfs(master)
            headroom_db = tp_limit - current_tp
            corrected_gain_db = delta_post if delta_post < 0 else min(delta_post, headroom_db)
            master *= 10.0 ** (corrected_gain_db / 20.0)
            print(f"  Post-limiter LUFS: {loudness_post:.2f} -> applying {corrected_gain_db:+.2f} dB "
                  f"(target {target_lufs}, headroom {headroom_db:+.2f} dB)")
            loudness_post = meter.integrated_loudness(master.T)
            peak_final = _measure_true_peak_dbfs(master)
    loudness = loudness_post

    # Write output
    if output_wav is None:
        out_dir = Path(config.get("output_dir", "."))
        if stage:
            stages_dir = out_dir / "stages"
            stages_dir.mkdir(parents=True, exist_ok=True)
            output_wav = stages_dir / f"mix_stage_{stage}.wav"
        else:
            output_wav = out_dir / "mix.wav"
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), master.T, sr, subtype="PCM_24")

    report = {
        "output": str(output_wav),
        "stage": stage,
        "active_tracks": len(active),
        "buses": list(processed.keys()),
        "lufs_target": target_lufs,
        "true_peak_limit_dbfs": tp_limit,
        "integrated_lufs": round(loudness, 2) if np.isfinite(loudness) else None,
        "true_peak_dbtp": round(peak_final, 2),
        "sample_peak_dbfs": round(sample_peak_after, 2),
        "duration_s": round(max_length / sr, 3),
        "clipper": clipper_report,
        "ms": ms_report,
    }
    report_path = output_wav.with_name(output_wav.stem + "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nOutput: {output_wav}")
    print(f"Report: {report_path}")

    if render_stems:
        stems_dir = output_wav.parent.parent / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nStems -> {stems_dir}/")
        for bus_name, buf in processed.items():
            stem_path = stems_dir / f"stem_{bus_name}.wav"
            _write_stem(buf.copy(), stem_path, sr)
            lufs = pyln.Meter(sr).integrated_loudness(buf.T)
            print(f"  stem_{bus_name}.wav  ({lufs:.1f} LUFS -> -18 LUFS)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render processed stems into a stereo mix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  render_mix.py output/<session> --generate-config
  render_mix.py output/<session>/mix_config.json --render
  render_mix.py output/<session>/mix_config.json --render --output my_mix.wav
        """,
    )
    parser.add_argument("input", help="Session output dir (--generate-config) or mix_config.json (--render)")
    parser.add_argument("--generate-config", action="store_true", help="Scan session dir and write mix_config.json")
    parser.add_argument("--style", default=None,
                        help="Style profile for genre-aware bus volume defaults during --generate-config "
                             "(modern_rock / classic_rock / pop / hip_hop / jazz_acoustic). "
                             "Without it, every bus starts at 0 dB.")
    parser.add_argument("--config", default=None, help="Config output path (default: <session_dir>/mix_config.json)")
    parser.add_argument("--render", action="store_true", help="Render mix from config")
    parser.add_argument("--output", default=None, help="Output WAV path (default: <output_dir>/mix.wav)")
    parser.add_argument("--stems", action="store_true", help="Also write per-bus stem WAVs to <output_dir>/stems/")
    parser.add_argument(
        "--stage",
        choices=STAGE_NAMES,
        default=None,
        help="Use stem files from this processing stage. raw=assembled, eq=after EQ, comp=after comp, fx=config file (default). Bus and master chain always run.",
    )
    args = parser.parse_args()

    if args.generate_config:
        session_dir = Path(args.input)
        if not session_dir.is_dir():
            parser.error(f"Not a directory: {session_dir}")
        config_path = Path(args.config) if args.config else session_dir / "mix_config.json"
        generate_config(session_dir, config_path, style=args.style)

    elif args.render:
        config_path = Path(args.input)
        if not config_path.is_file():
            parser.error(f"Config file not found: {config_path}")
        output_wav = Path(args.output) if args.output else None
        render_mix(config_path, output_wav, render_stems=args.stems, stage=args.stage)

    else:
        parser.error("Specify --generate-config or --render")


if __name__ == "__main__":
    main()
