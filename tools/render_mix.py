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

# Backing vocal keywords first (more specific) so "BG VOX" doesn't fall into the
# generic VOX/VOC bucket.
_VOCAL_BG_KEYWORDS = ["BG VOX", "BG VOC", "BACKING", "HARMONY", "AD-LIB", "ADLIB", "DOUBLE"]
_VOCAL_LEAD_KEYWORDS = ["LEAD VOX", "LEAD VOC", "VOX", "VOC", "WHISPER", "TALK"]


def _detect_bus(name: str) -> str:
    u = name.upper()
    if any(k in u for k in _DRUM_KEYWORDS):
        return "drums"
    if "BASS" in u:
        return "bass"
    for prefix in _GUITARIST_PREFIXES:
        if u.startswith(prefix):
            return prefix.lower().replace(" ", "_")
    if any(k in u for k in _VOCAL_BG_KEYWORDS):
        return "vocal_bg"
    if any(k in u for k in _VOCAL_LEAD_KEYWORDS):
        return "vocal_lead"
    return "master"


# Drum kit panning conventions (audience perspective — modern mixing default).
# Pitches sweep across the stereo field; cymbals placed where they physically
# sit relative to the drummer. The drummer's right-hand cymbals (hi-hat, ride)
# end up on the audience's LEFT side of the stereo image — the convention is
# "audience perspective" so the panning matches what a listener facing the
# kit on stage would hear.
#
# Sources: standard rock-mix references (Producer Society, iZotope, Sound on
# Sound LCR / hard-pan articles). Tweak via mix_config.json per-track `pan`
# field if a particular session has unusual kit placement.
#
# More specific keys (e.g. "RACK TOM 2") MUST come before less specific ones
# (e.g. "RACK TOM") so the first match wins.
_DRUM_PAN_DEFAULTS: list[tuple[str, float]] = [
    # Toms — left-to-right pitch sweep
    ("RACK TOM 1",  -0.4),
    ("RACK TOM 2",  -0.15),
    ("RACK TOM",    -0.4),    # generic single rack
    ("FLOOR TOM",   +0.5),
    ("FLOOR",       +0.5),    # short-form
    # Cymbals — drummer's physical layout (audience perspective)
    ("HIHAT",       -0.2),
    ("HI-HAT",      -0.2),
    ("RIDE",        +0.3),
    # Kick / snare / generic crash → center
    ("KICK",         0.0),
    ("SN ",          0.0),
    (" SN",          0.0),
    ("SNARE",        0.0),
    ("CRASH",        0.0),
]


def _detect_pan(name: str) -> float:
    """Detect a sensible default pan for a track name.

    Order of resolution:
      1. OH / ROOM stereo pairs use the L/R suffix (existing behaviour) —
         each pair contributes hard stereo width.
      2. Mono drum-kit pieces (toms, hi-hat, ride, kick, snare) get the
         audience-perspective default pan from `_DRUM_PAN_DEFAULTS`.
      3. Any other name with a standalone L or R word → ±0.7 (stereo pair).
      4. Otherwise center.

    Used by `--generate-config` to populate per-track `pan` fields. Override
    per-track in mix_config.json if a session has unusual kit / mic placement.
    """
    u = name.upper()

    # Stereo pairs (OH / ROOM): use L/R suffix as before
    if "OH " in u or " OH" in u or "OVERHEAD" in u or "ROOM " in u or " ROOM" in u:
        if re.search(r'\bL\b', u):
            return -0.7
        if re.search(r'\bR\b', u):
            return 0.7
        # Stereo pair without explicit L/R suffix — leave at center
        return 0.0

    # Drum-kit pieces (mono mics) — apply audience-perspective spread
    for keyword, pan in _DRUM_PAN_DEFAULTS:
        if keyword in u:
            return pan

    # Generic L/R suffix detection (stereo pairs on non-drum tracks)
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


def _load_style_bus_pans(style: str | None) -> dict[str, float]:
    """Look up `default_bus_pan` from the named style profile.

    Returns {} if `style` is None or the profile doesn't exist / lacks
    the field. Callers should fall back to 0.0 (center) for any bus not
    in the returned dict.

    Genre conventions (typical 2 rhythm guitarist setup):
      - modern_rock:        gtr_1 -0.6,  gtr_laci +0.6  (wide L/R)
      - punchy_modern_rock: gtr_1 -0.7,  gtr_laci +0.7  (very wide)
      - classic_rock:       gtr_1 -0.5,  gtr_laci +0.5  (band-feel)
      - pop:                gtr_1 -0.4,  gtr_laci +0.4  (conservative)
      - hip_hop:            gtr_1  0,    gtr_laci  0    (centered, drum-led)
      - jazz_acoustic:      gtr_1 -0.3,  gtr_laci +0.3  (narrow, intimate)
    """
    if not style:
        return {}
    profile_path = Path(__file__).resolve().parent / "style_profiles" / f"{style}.json"
    if not profile_path.exists():
        return {}
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return profile.get("default_bus_pan", {})


def generate_config(session_dir: Path, output_path: Path, style: str | None = None) -> None:
    sr = 48000
    session_json = session_dir / "session.json"
    if session_json.exists():
        with open(session_json) as f:
            sr = json.load(f).get("sample_rate", 48000)

    style_bus_defaults = _load_style_bus_defaults(style)
    style_bus_pans = _load_style_bus_pans(style)
    if style_bus_defaults:
        print(f"Using style '{style}' bus volume defaults: "
              + ", ".join(f"{k}={v:+.1f}" for k, v in style_bus_defaults.items()))
    if style_bus_pans:
        # Only print non-zero entries — center is the default and not informative.
        non_center = {k: v for k, v in style_bus_pans.items() if v != 0.0}
        if non_center:
            print(f"Using style '{style}' bus pan defaults: "
                  + ", ".join(f"{k}={v:+.2f}" for k, v in non_center.items()))

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

    # Top-level bus volume_db: use the style profile default if given, else 0 dB.
    # Bus pan: same idea — style profile says e.g. modern_rock gtr_1 -0.6.
    # auto_trim_db is computed below by measuring active stem dry-sums.
    def _bus_default(name: str) -> float:
        return float(style_bus_defaults.get(name, 0.0))

    def _bus_pan(name: str) -> float:
        return float(style_bus_pans.get(name, 0.0))

    buses = {
        # Default to comp_drum_bus_gentle (2:1, -8 dB, ~2 dB GR) per modern
        # industry guidance (Music Guy Mixing 2025: "1-2 dB GR average, 2-3 dB
        # max at busiest moments"). The harder comp_drum_bus (4:1, -10 dB,
        # ~5 dB GR) is too aggressive for a default — switch manually if a
        # session genuinely needs heavier glue.
        "drums":  {"volume_db": _bus_default("drums"),  "pan": _bus_pan("drums"),  "auto_trim_db": 0.0, "comp_preset": "comp_drum_bus_gentle", "parent_bus": None},
        "bass":   {"volume_db": _bus_default("bass"),   "pan": _bus_pan("bass"),   "auto_trim_db": 0.0, "comp_preset": None,            "parent_bus": None},
        **{name: {**cfg, "pan": _bus_pan(name), "auto_trim_db": 0.0} for name, cfg in gtr_sub_buses.items()},
        "guitar": {"volume_db": _bus_default("guitar"), "pan": _bus_pan("guitar"), "auto_trim_db": 0.0, "comp_preset": None,            "parent_bus": None},
    }

    # Vocal sub-buses if vocal tracks were detected. Both feed into a shared
    # `vocal` parent so a single bus-level processing pass (master glue, gentle
    # comp, etc.) can sit downstream of both lead and BG.
    has_vocal_lead = any(t["bus"] == "vocal_lead" for t in tracks)
    has_vocal_bg = any(t["bus"] == "vocal_bg" for t in tracks)
    if has_vocal_lead or has_vocal_bg:
        if has_vocal_lead:
            buses["vocal_lead"] = {"volume_db": _bus_default("vocal_lead"), "pan": _bus_pan("vocal_lead"), "auto_trim_db": 0.0, "comp_preset": None, "parent_bus": "vocal"}
        if has_vocal_bg:
            buses["vocal_bg"] = {"volume_db": _bus_default("vocal_bg"), "pan": _bus_pan("vocal_bg"), "auto_trim_db": 0.0, "comp_preset": None, "parent_bus": "vocal"}
        buses["vocal"] = {"volume_db": _bus_default("vocal"), "pan": _bus_pan("vocal"), "auto_trim_db": 0.0, "comp_preset": None, "parent_bus": None}

    mixes_dir = session_dir / "mixes"
    config = {
        "session_dir": str(session_dir),
        "output_dir": str(mixes_dir),
        "sample_rate": sr,
        "master": {
            # Premaster mode: render_mix produces a CLEAN mix.wav with peak
            # headroom at peak_target_dbfs (no limiter, no LUFS norm). The
            # master phase (master_mix.py) owns LUFS target + true peak
            # ceiling per delivery format. Industry standard handoff.
            "premaster_mode": True,
            "peak_target_dbfs": -3.0,
        },
        "buses": buses,
        "tracks": tracks,
    }

    # Per-bus auto-trim: load active stems, measure dry-sum LUFS per bus, and
    # set auto_trim_db so each bus output sits at -18 LUFS with volume_db = 0.
    # This is the "calibration anchor" — keeps the master sum at a sane level
    # regardless of how many stems each bus has (drums 15+, bass 2, vocal 1...).
    print("\nComputing per-bus auto-trim (target -18 LUFS per bus):")
    auto_trims = _compute_bus_auto_trims(config, target_lufs=-18.0, verbose=True)
    for name, trim in auto_trims.items():
        config["buses"][name]["auto_trim_db"] = trim

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


def _compute_bus_auto_trims(
    config: dict,
    target_lufs: float = -18.0,
    verbose: bool = True,
) -> dict[str, float]:
    """Per-bus auto-trim calibration.

    Walks every bus in topological order (children before parents). For each:
      1. Sums active stems (with per-track volume_db + pan + polarity_flip)
      2. Adds child bus outputs (post-(auto_trim + volume_db))
      3. Measures integrated LUFS of that dry sum
      4. Computes auto_trim_db = target_lufs - measured_lufs

    Returns {bus_name: auto_trim_db}. At render time the effective bus gain is
    auto_trim_db + volume_db. With volume_db = 0 every bus output sits at
    target_lufs regardless of stem count — fixes the "drum bus sums to
    +5 dBFS while bass sums to -1 dBFS" pile-up problem.

    Buses with no audible content get 0.0 (no-op). The function does NOT
    mutate `config`; callers write the result back into buses[name]["auto_trim_db"].
    """
    sr = int(config.get("sample_rate", 48000))
    active_tracks = [t for t in config["tracks"] if t.get("active", True)]
    if not active_tracks:
        return {name: 0.0 for name in config["buses"]}

    try:
        max_length = max(sf.info(t["file"]).frames for t in active_tracks)
    except Exception as exc:
        if verbose:
            print(f"  auto-trim: could not stat stem files ({exc}) — leaving auto_trim_db = 0",
                  file=sys.stderr)
        return {name: 0.0 for name in config["buses"]}

    meter = pyln.Meter(sr)

    bus_buffers: dict[str, np.ndarray] = {}
    for t in active_tracks:
        bus = t.get("bus", "master")
        if bus not in config["buses"]:
            continue
        vol = 10.0 ** (t.get("volume_db", 0.0) / 20.0)
        pan_v = t.get("pan", 0.0)
        polarity = -1.0 if t.get("polarity_flip", False) else 1.0
        try:
            stereo = _load_as_stereo(Path(t["file"]), max_length, sr)
        except Exception as exc:
            if verbose:
                print(f"  auto-trim: skip {t['file']} ({exc})", file=sys.stderr)
            continue
        stereo = _pan(stereo * vol * polarity, pan_v)
        if bus not in bus_buffers:
            bus_buffers[bus] = np.zeros((2, max_length), dtype=np.float32)
        bus_buffers[bus] += stereo

    auto_trims: dict[str, float] = {}
    bus_outputs: dict[str, np.ndarray] = {}

    for bus_name in _topo_order(config["buses"]):
        cfg = config["buses"][bus_name]
        buf = bus_buffers.get(bus_name, np.zeros((2, max_length), dtype=np.float32)).copy()
        for child_name, child_cfg in config["buses"].items():
            if child_cfg.get("parent_bus") == bus_name and child_name in bus_outputs:
                buf = buf + bus_outputs[child_name]

        if not np.any(buf):
            auto_trims[bus_name] = 0.0
            bus_outputs[bus_name] = buf
            if verbose:
                print(f"  Bus '{bus_name}': no audio  -> auto_trim_db = +0.0")
            continue

        # Apply the bus's own pan BEFORE measuring LUFS so the calibration
        # matches the post-pan state the renderer produces (the renderer
        # applies eff_gain then bus_pan; gain and constant-power pan
        # commute, so the LUFS landing point is identical). Without this,
        # hard-panned buses lose ~3 dB LUFS at render that the calibration
        # never sees, producing too much downstream attenuation.
        bus_pan = float(cfg.get("pan", 0.0))
        if bus_pan != 0.0:
            buf = _pan(buf, bus_pan)

        try:
            lufs_in = float(meter.integrated_loudness(buf.T))
        except Exception:
            lufs_in = float("-inf")

        if not np.isfinite(lufs_in):
            auto_trims[bus_name] = 0.0
            bus_outputs[bus_name] = buf
            if verbose:
                print(f"  Bus '{bus_name}': LUFS unmeasurable -> auto_trim_db = +0.0")
            continue

        auto_trim = round(target_lufs - lufs_in, 1)
        auto_trims[bus_name] = auto_trim

        eff_gain_db = auto_trim + float(cfg.get("volume_db", 0.0))
        bus_outputs[bus_name] = buf * (10.0 ** (eff_gain_db / 20.0))

        if verbose:
            print(f"  Bus '{bus_name}': dry-sum {lufs_in:+6.1f} LUFS  -> "
                  f"auto_trim {auto_trim:+5.1f} dB  (volume_db {cfg.get('volume_db', 0.0):+.1f})")

    return auto_trims


def _load_as_stereo(path: Path, length: int, sr: int) -> np.ndarray:
    """Load WAV file as (2, length) float32 stereo array.

    Mono files are duplicated to both channels. Stereo (and multi-channel)
    files keep channels 0 and 1; any further channels are dropped.

    Returns float32 (vs float64): halves memory bandwidth on big buffers
    (e.g. 41 stems × 7 min @ 48 kHz × 2 ch × 4 byte = ~2 GB vs ~4 GB) and
    matches what pedalboard expects natively — eliminates the float32 cast
    overhead later in the bus chain.
    """
    data, file_sr = sf.read(str(path), always_2d=True, dtype="float32")
    if file_sr != sr:
        raise ValueError(f"SR mismatch: {path} is {file_sr} Hz, expected {sr}")

    n = data.shape[0]
    out = np.zeros((2, length), dtype=np.float32)
    take = min(n, length)
    if data.shape[1] == 1:
        out[0, :take] = data[:take, 0]
        out[1, :take] = data[:take, 0]
    else:
        out[0, :take] = data[:take, 0]
        out[1, :take] = data[:take, 1]
    return out


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
    return board(buf.astype(np.float32), sr).astype(np.float32)


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
    reverb_out = board(delayed.T.astype(np.float32), sr).T.astype(np.float32)  # (N+pre, 2)

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


def _bus_tape_sat_relevance_check(buf: np.ndarray, sr: int, bus_name: str) -> dict:
    """Decide whether tape saturation on a bus would help.

    Conditions for SKIP:
      - Bus crest factor < 8 dB (signal already squashed; tanh adds buzz, not warmth)
      - Bus LRA < 4 LU (no dynamic headroom — tape sat compounds the squash)

    No bus-name restriction: tape sat is general purpose. Threshold values are
    slightly more permissive than parallel_sat (10 dB / 4 LU) because tape sat
    is less drastic — it just shapes harmonics rather than blending an extra
    saturated copy on top.
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
    if crest_db < 8.0:
        issues.append(f"crest {crest_db:.1f} dB < 8 — bus already squashed; tape sat adds buzz, not warmth")
    if lra < 4.0:
        issues.append(f"LRA {lra:.1f} LU < 4 — too compressed for tape sat to add life")

    return {
        "tool": "tape_saturate",
        "bus": bus_name,
        "crest_db": round(crest_db, 1),
        "lra_lu": round(lra, 1),
        "recommend_skip": bool(issues),
        "issues": issues,
    }


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


def _measure_true_peak_dbfs(master: np.ndarray, oversample: int = 4,
                            fast_skip_db: float = -3.0) -> float:
    """4x-oversampled true peak in dBFS for a (2, N) stereo buffer.

    pedalboard.Limiter only constrains the sample peak; inter-sample peaks
    can still exceed the ceiling after codec encoding (Spotify Ogg/Vorbis,
    Apple AAC). This second-pass measurement reveals them.

    Fast path: if the sample peak is below `fast_skip_db` (default -3 dBFS)
    we skip the resample_poly call and approximate TP ≈ sample_peak + 0.5 dB.
    The approximation is always conservative — the actual TP-vs-sample-peak
    gap can't exceed ~0.5 dB for normal stereo audio, and at -3 dBFS the
    TP can't reach the -1 dBTP ceiling regardless. Saves ~0.8s per call;
    a typical render measures 8-11 buses, only the master output is hot
    enough to actually need the oversampled measurement.
    """
    sample_peak = float(np.max(np.abs(master)))
    sample_peak_db = 20.0 * np.log10(max(sample_peak, 1e-12))
    if sample_peak_db < fast_skip_db:
        return sample_peak_db + 0.5  # conservative TP approximation
    up_l = _resample_poly(master[0], oversample, 1)
    up_r = _resample_poly(master[1], oversample, 1)
    tp = max(float(np.max(np.abs(up_l))), float(np.max(np.abs(up_r))))
    return 20.0 * np.log10(max(tp, 1e-12))


def _peak_db(buf: np.ndarray) -> float:
    return float(20.0 * np.log10(np.max(np.abs(buf)) + 1e-12))


def _peak_verdict(peak_db: float, true_peak_db: float | None = None) -> str:
    """DAW-style channel warning level based on worst of sample/true peak.

    [OK]   peak  < -6 dBFS         — comfortable headroom
    [WARN] -6 ≤ peak < -1 dBFS    — close to ceiling
    [CLIP] peak ≥ -1 dBFS or TP   — risk of inter-sample clipping
    """
    worst = peak_db if true_peak_db is None else max(peak_db, true_peak_db)
    if worst >= -1.0:
        return "[CLIP]"
    if worst >= -6.0:
        return "[WARN]"
    return "[OK]"


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
    # Shared reverb buses — top-level `reverb_buses` block of mix_config.json.
    # Each named bus accumulates pre-fader sends from any track listing it in
    # its `reverb_sends`. After the track loop, each reverb bus is rendered
    # once and the wet return is summed into master.
    reverb_buses_cfg: dict = config.get("reverb_buses", {})
    reverb_send_buffers: dict[str, np.ndarray] = {
        rb_name: np.zeros((2, max_length), dtype=np.float32) for rb_name in reverb_buses_cfg
    }

    bus_buffers: dict[str, np.ndarray] = {}
    # Per-bus per-track downsampled mono buffers for phase-correlation checks.
    # Only buses with >=2 active tracks get pairwise-correlated afterward —
    # this catches polarity-flipped duplicates and delayed-copy phase issues
    # that no per-clip / source-hash audit can see (different files, same source).
    bus_track_mono: dict[str, list[tuple[str, np.ndarray]]] = {}
    _PHASE_DECIM = 48  # 48 kHz -> 1 kHz; plenty for low-freq correlation
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
            bus_buffers[bus] = np.zeros((2, max_length), dtype=np.float32)
        bus_buffers[bus] += stereo

        # Capture pre-mix mono (before pan/vol scaling) for phase-correlation
        # check between tracks on the same bus. Decimated to 1 kHz to keep
        # memory bounded (41 tracks × 7-min @ 48 kHz = 13.5 GB otherwise).
        bus_track_mono.setdefault(bus, []).append(
            (name, stereo.mean(axis=0)[::_PHASE_DECIM].copy())
        )

        # Track-level sends to shared reverb buses (post-fader, post-pan).
        for send in t.get("reverb_sends", []):
            rb_name = send.get("bus")
            if rb_name not in reverb_send_buffers:
                print(f"    WARNING: reverb_send target '{rb_name}' not in reverb_buses — skipped")
                continue
            level_db = float(send.get("level_db", -6.0))
            reverb_send_buffers[rb_name] += stereo * (10.0 ** (level_db / 20.0))

    print()

    # Per-bus pairwise phase-correlation scan. Flags pairs where two active
    # tracks on the same bus are correlated (potential phase-coherent +6 dB
    # doubling) or anti-correlated (polarity-flipped duplicate or comb
    # filtering from delayed copy). Threshold tuned to surface the obvious
    # cases without false-positive noise from incidentally-similar mics.
    #
    # Vectorised pass: per-track per-window RMS is computed once (not per
    # pair), then pair joint-active masks are a bitwise AND. With 18-track
    # drum buses (18 choose 2 = 153 pairs) the old python-loop approach
    # spent ~5s here; the vectorised version is < 200 ms.
    PHASE_WARN_THRESHOLD = 0.4   # |corr| > 0.4 flags a warning
    THRESHOLD_RMS_LIN = 10.0 ** (-45.0 / 20.0)  # -45 dBFS
    phase_warnings: list[dict] = []
    for bus, items in bus_track_mono.items():
        if len(items) < 2:
            continue
        sr_decim = sr // _PHASE_DECIM
        win = sr_decim  # 1 second of decimated samples
        n_min = min(len(it[1]) for it in items)
        if n_min < win:
            continue
        n_win = n_min // win
        usable = n_win * win

        # Stack tracks into (n_tracks, n_min) then reshape to (n_tracks, n_win, win)
        # for vectorised per-window RMS. Trim to usable length.
        signals = np.stack([it[1][:usable] for it in items])  # (T, N)
        framed = signals.reshape(len(items), n_win, win)
        per_win_rms = np.sqrt(np.mean(framed ** 2, axis=2) + 1e-12)  # (T, n_win)
        per_track_active = per_win_rms > THRESHOLD_RMS_LIN  # (T, n_win)

        # Pairwise scan — but with cheap O(n_win) joint mask per pair.
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                joint = per_track_active[i] & per_track_active[j]
                overlap_sec = float(joint.sum())
                if overlap_sec < 5:  # need >= 5 sec of mutual activity
                    continue
                sample_mask = np.repeat(joint, win)
                a_olap = signals[i, :usable][sample_mask]
                b_olap = signals[j, :usable][sample_mask]
                an = (a_olap - a_olap.mean()) / (a_olap.std() + 1e-12)
                bn = (b_olap - b_olap.mean()) / (b_olap.std() + 1e-12)
                corr = float(np.mean(an * bn))
                if abs(corr) >= PHASE_WARN_THRESHOLD:
                    kind = "anti-correlated (cancellation / polarity)" if corr < 0 else "correlated (constructive sum / phase-coherent doubling)"
                    phase_warnings.append({
                        "bus": bus,
                        "track_a": items[i][0],
                        "track_b": items[j][0],
                        "correlation": round(corr, 3),
                        "overlap_sec": overlap_sec,
                        "kind": kind,
                    })

    if phase_warnings:
        print("  [!] Per-bus phase-correlation warnings (|corr| ≥ 0.4):")
        for w in phase_warnings:
            print(f"    {w['bus']}: '{w['track_a']}' vs '{w['track_b']}'  "
                  f"corr={w['correlation']:+.2f}  overlap={w['overlap_sec']:.0f}s  — {w['kind']}")
        print()

    # Process buses in topological order (leaves first, parents after).
    # bus_peaks captures the sample peak after each chain stage and the true
    # peak on the final bus output — the per-channel "meter bridge" for the
    # mix_report.
    processed: dict[str, np.ndarray] = {}
    bus_peaks: dict[str, dict] = {}
    for bus_name in _topo_order(buses_cfg):
        cfg = buses_cfg[bus_name]

        buf = bus_buffers.get(bus_name, np.zeros((2, max_length), dtype=np.float32)).copy()

        # Add processed child buses that route into this bus
        for child, child_cfg in buses_cfg.items():
            if child_cfg.get("parent_bus") == bus_name and child in processed:
                buf += processed[child]

        if not np.any(buf):
            continue

        stage_peaks: dict[str, float | None] = {
            "sum_in": None, "after_vol_pan": None, "after_eq": None,
            "after_comp": None, "after_sat": None, "after_parallel_sat": None,
            "after_reverb": None,
        }
        stage_peaks["sum_in"] = round(_peak_db(buf), 2)

        # Effective bus gain = auto_trim_db (calibration so dry-sum hits -18 LUFS)
        # + volume_db (style/user offset on top). Default 0 for both if absent.
        auto_trim_db = float(cfg.get("auto_trim_db", 0.0))
        volume_db = float(cfg.get("volume_db", 0.0))
        eff_gain_db = auto_trim_db + volume_db
        buf *= 10.0 ** (eff_gain_db / 20.0)
        if auto_trim_db != 0.0:
            print(f"  Bus '{bus_name}': auto_trim {auto_trim_db:+.1f} dB + volume {volume_db:+.1f} dB "
                  f"= {eff_gain_db:+.1f} dB")

        bus_pan = cfg.get("pan", 0.0)
        if bus_pan != 0.0:
            buf = _pan(buf, bus_pan)
        stage_peaks["after_vol_pan"] = round(_peak_db(buf), 2)

        bus_eq = cfg.get("eq")
        if bus_eq:
            buf = _apply_eq_chain(buf, sr, bus_eq, label=f"bus '{bus_name}' EQ")
            filter_summary = ", ".join(
                f"{f.get('type', '?')}@{f.get('hz', '?')}Hz"
                + (f" {f.get('db', 0):+.1f}dB" if f.get("db") is not None else "")
                for f in bus_eq
            )
            peak_eq = _peak_db(buf)
            stage_peaks["after_eq"] = round(peak_eq, 2)
            print(f"  Bus '{bus_name}': + EQ ({filter_summary})  -> {peak_eq:.1f} dBFS {_peak_verdict(peak_eq)}")

        preset = cfg.get("comp_preset")
        if preset:
            peak_in = _peak_db(buf)
            print(f"  Bus '{bus_name}': compressing with preset '{preset}' (peak in: {peak_in:.1f} dBFS)...",
                  end=" ", flush=True)
            buf = _apply_comp_preset(buf, preset, sr)
            peak_out = _peak_db(buf)
            stage_peaks["after_comp"] = round(peak_out, 2)
            print(f"-> {peak_out:.1f} dBFS {_peak_verdict(peak_out)}")
        else:
            peak = _peak_db(buf)
            print(f"  Bus '{bus_name}': peak {peak:.1f} dBFS {_peak_verdict(peak)}")

        sat_cfg = cfg.get("saturation")
        if sat_cfg:
            sat_drive = float(sat_cfg.get("drive", 0.3))
            sat_force = bool(sat_cfg.get("force", False))
            rel = _bus_tape_sat_relevance_check(buf, sr, bus_name)
            if rel["recommend_skip"] and not sat_force:
                print(f"  Bus '{bus_name}': tape saturation SKIPPED — {'; '.join(rel['issues'])}")
            else:
                buf = _tape_saturate(buf, sat_drive)
                peak_sat = _peak_db(buf)
                stage_peaks["after_sat"] = round(peak_sat, 2)
                note = " [FORCED]" if (rel["recommend_skip"] and sat_force) else ""
                print(f"  Bus '{bus_name}': + tape saturation drive={sat_drive}  -> {peak_sat:.1f} dBFS {_peak_verdict(peak_sat)}{note}")

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
                peak_psat = _peak_db(buf)
                stage_peaks["after_parallel_sat"] = round(peak_psat, 2)
                print(f"  Bus '{bus_name}': + parallel sat ({mode}, drive={drive}, mix={mix})  -> {peak_psat:.1f} dBFS {_peak_verdict(peak_psat)}")

        reverb_cfg = cfg.get("reverb_send")
        if reverb_cfg:
            if not _HAS_REVERB:
                print(f"  WARNING: apply_reverb.py not importable — skipping reverb send for '{bus_name}'")
            else:
                rv_preset = reverb_cfg["preset"]
                rv_wet = float(reverb_cfg.get("wet", 0.15))
                reverb_return = _apply_bus_reverb(buf, sr, rv_preset, rv_wet)
                buf = buf + reverb_return
                peak_rv = _peak_db(reverb_return)
                peak_after = _peak_db(buf)
                stage_peaks["after_reverb"] = round(peak_after, 2)
                print(f"  Bus '{bus_name}': + reverb send '{rv_preset}' wet={rv_wet} (return peak {peak_rv:.1f} dBFS, bus {peak_after:.1f} dBFS {_peak_verdict(peak_after)})")

        final_peak = _peak_db(buf)
        final_tp = _measure_true_peak_dbfs(buf)
        bus_peaks[bus_name] = {
            **stage_peaks,
            "final": round(final_peak, 2),
            "true_peak_final": round(final_tp, 2),
            "verdict": _peak_verdict(final_peak, final_tp),
        }
        processed[bus_name] = buf

    # Render each shared reverb bus and collect the wet returns
    reverb_returns: dict[str, np.ndarray] = {}
    for rb_name, rb_cfg in reverb_buses_cfg.items():
        if not _HAS_REVERB:
            print(f"  WARNING: apply_reverb.py not importable — skipping reverb_bus '{rb_name}'")
            continue
        send_buf = reverb_send_buffers[rb_name]
        if not np.any(send_buf):
            continue
        rv_preset = rb_cfg["preset"]
        rv_wet = float(rb_cfg.get("wet", 1.0))
        return_db = float(rb_cfg.get("return_volume_db", -6.0))
        rv_return = _apply_bus_reverb(send_buf, sr, rv_preset, rv_wet)
        rv_return *= (10.0 ** (return_db / 20.0))
        # Optional return pan
        return_pan = float(rb_cfg.get("return_pan", 0.0))
        if return_pan != 0.0:
            rv_return = _pan(rv_return, return_pan)
        reverb_returns[rb_name] = rv_return
        peak_rv = 20.0 * np.log10(np.max(np.abs(rv_return)) + 1e-12)
        print(f"  Reverb bus '{rb_name}': preset='{rv_preset}' return_db={return_db:+.1f} "
              f"(return peak {peak_rv:.1f} dBFS)")

    # Per-bus peak summary table — DAW-style channel meter snapshot.
    if bus_peaks:
        print()
        print("  Per-bus peak summary (dBFS):")
        print(f"    {'bus':<14}{'sum':>7}{'vol/pan':>9}{'eq':>7}{'comp':>7}{'sat':>7}{'p-sat':>7}{'reverb':>8}{'final':>8}{'TP':>7}  verdict")
        for bn, bp in bus_peaks.items():
            def _fmt(v): return f"{v:>7.1f}" if v is not None else f"{'----':>7}"
            print(f"    {bn:<14}"
                  f"{_fmt(bp['sum_in'])}{_fmt(bp['after_vol_pan']).rjust(9)}"
                  f"{_fmt(bp['after_eq'])}{_fmt(bp['after_comp'])}"
                  f"{_fmt(bp['after_sat'])}{_fmt(bp['after_parallel_sat'])}"
                  f"{_fmt(bp['after_reverb']).rjust(8)}"
                  f"{bp['final']:>8.1f}{bp['true_peak_final']:>7.1f}  {bp['verdict']}")

    # Sum top-level buses (parent_bus: null) into master
    master = np.zeros((2, max_length), dtype=np.float32)
    for bus_name, cfg in buses_cfg.items():
        if not cfg.get("parent_bus") and bus_name in processed:
            master += processed[bus_name]

    # Sum shared reverb returns into master
    for rb_name, rv_ret in reverb_returns.items():
        master += rv_ret

    # master_peaks captures sample peak at each master-chain stage. Final
    # row also carries true peak — the "master strip" of the meter bridge.
    master_peaks: dict[str, float | None] = {
        "sum_in": None, "after_comp": None, "after_clipper": None,
        "after_ms": None, "after_eq": None, "after_lufs_norm": None,
        "after_limiter": None,
    }
    peak_pre = _peak_db(master)
    master_peaks["sum_in"] = round(peak_pre, 2)
    print(f"\n  Master pre-processing: {peak_pre:.1f} dBFS {_peak_verdict(peak_pre)}")

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
        master = comp_board(master.astype(np.float32), sr).astype(np.float32)

        lufs_post_comp = meter.integrated_loudness(master.T)
        peak_post_comp = _peak_db(master)
        master_peaks["after_comp"] = round(peak_post_comp, 2)
        gr_lufs = lufs_post_comp - lufs_pre_comp
        print(f"  Master comp: {master_comp['threshold_db']}dB threshold  {master_comp.get('ratio', 2.0)}:1  "
              f"att={master_comp.get('attack_ms', 10.0)}ms  rel={master_comp.get('release_ms', 300.0)}ms  "
              f"-> GR {gr_lufs:+.1f} LUFS  peak {peak_post_comp:.1f} dBFS {_peak_verdict(peak_post_comp)}")

    # Premaster mode: the mix output is intended as a clean handoff to the
    # master phase. Industry best practice (SOS, LANDR, iZotope, Mat Leffler-
    # Schulman) — premaster should NOT have a brick-wall limiter, clipper,
    # M/S, or LUFS normalization to -14 baked in. Those are mastering's job;
    # stacking them here creates a two-stage limiter cascade that flattens
    # transients twice. Premaster target: integrated ~-18 LUFS (emergent
    # from the autotrim + glue comp), peak at master.peak_target_dbfs
    # (default -3 dBFS, configurable). Opt out with `"premaster_mode": false`
    # in master_cfg to get the legacy combined mix+master chain.
    premaster_mode = bool(master_cfg.get("premaster_mode", True))
    clipper_cfg = master_cfg.get("clipper")
    ms_cfg = master_cfg.get("ms")
    clipper_report: dict | None = None
    ms_report: dict | None = None

    if premaster_mode:
        # Warn if mastering-style options leaked in from a legacy config —
        # they are silently ignored in premaster mode to keep the handoff clean.
        if clipper_cfg or ms_cfg or master_cfg.get("lufs_target") is not None \
                or master_cfg.get("true_peak_dbfs") is not None:
            print("  [premaster] clipper/ms/lufs_target/true_peak_dbfs in config "
                  "— IGNORED (these are mastering-phase options). "
                  "Set master.premaster_mode=false to apply them.")

    # Master clipper (legacy / opt-in only — premaster_mode skips it)
    if not premaster_mode and clipper_cfg:
        rel = _clipper_relevance_check(master, sr)
        clipper_report = {"settings": clipper_cfg, "relevance_check": rel}
        if rel["recommend_skip"]:
            print(f"  Clipper: SKIPPED — {'; '.join(rel['issues'])}")
        else:
            mode = str(clipper_cfg.get("mode", "soft")).lower()
            threshold = float(clipper_cfg.get("threshold_db", -1.0))
            knee = float(clipper_cfg.get("knee_db", 1.5))
            peak_pre = _peak_db(master)
            if mode == "hard":
                master = _hard_clip(master, threshold)
            else:
                master = _soft_clip(master, threshold, knee)
            peak_post = _peak_db(master)
            master_peaks["after_clipper"] = round(peak_post, 2)
            print(f"  Master clipper ({mode}): threshold {threshold:.1f} dBFS  "
                  f"knee {knee:.1f} dB  -> peak {peak_pre:.1f} → {peak_post:.1f} dBFS {_peak_verdict(peak_post)}")
            clipper_report["applied"] = True

    # M/S processing (legacy / opt-in only)
    if not premaster_mode and ms_cfg:
        master, ms_report = _apply_ms_processing(master, sr, ms_cfg)
        if ms_report.get("applied"):
            mid_g = ms_cfg.get("mid_gain_db", 0.0)
            side_g = ms_cfg.get("side_gain_db", 0.0)
            peak_ms = _peak_db(master)
            master_peaks["after_ms"] = round(peak_ms, 2)
            print(f"  M/S: mid {mid_g:+.1f} dB, side {side_g:+.1f} dB  "
                  f"(width ratio {ms_report['relevance_check']['ms_width_ratio']:.3f})  -> peak {peak_ms:.1f} dBFS {_peak_verdict(peak_ms)}")
        else:
            issues = ms_report["relevance_check"]["issues"]
            print(f"  M/S: SKIPPED — {'; '.join(issues)}")

    # Master bus EQ (optional, zero-phase). Subtle tonal shaping is OK in
    # premaster — it's not loudness-focused processing.
    master_eq = master_cfg.get("eq")
    if master_eq:
        master = _apply_master_eq(master, sr, master_eq)
        peak_post_eq = _peak_db(master)
        master_peaks["after_eq"] = round(peak_post_eq, 2)
        filter_desc = ", ".join(
            f"{f.get('type','?')} {f.get('hz', '')}Hz"
            + (f" {f['db']:+.1f}dB" if "db" in f else "")
            for f in master_eq
        )
        print(f"  Master EQ: {filter_desc}  -> peak {peak_post_eq:.1f} dBFS {_peak_verdict(peak_post_eq)}")

    # Branch: premaster vs legacy master chain
    if premaster_mode:
        # Peak normalize to the configured target (default -3 dBFS) — a single
        # scalar gain across the buffer, NO limiter / clipper. Preserves
        # transients perfectly; just lowers the level so the master phase
        # receives clean headroom.
        peak_target = float(master_cfg.get("peak_target_dbfs", -3.0))
        peak_target_lin = 10.0 ** (peak_target / 20.0)
        current_peak_lin = float(np.max(np.abs(master)))
        if current_peak_lin > 1e-10:
            scale = peak_target_lin / current_peak_lin
            scale_db = 20.0 * np.log10(scale)
            master = (master * scale).astype(np.float32)
            peak_final_sample = _peak_db(master)
            print(f"  Premaster peak normalize: scale {scale_db:+.2f} dB -> "
                  f"peak {peak_final_sample:.2f} dBFS")
        loudness_final = float(meter.integrated_loudness(master.T))
        tp_final = float(_measure_true_peak_dbfs(master))
        sample_peak_after = _peak_db(master)
        master_peaks["after_lufs_norm"] = None
        master_peaks["after_limiter"] = None
        master_peaks["final_sample_peak"] = round(sample_peak_after, 2)
        master_peaks["final_true_peak"] = round(tp_final, 2)
        # Verdict for a premaster: peak should be in the -6..-3 dBFS window
        # (industry standard handoff). Use a relaxed verdict here.
        if sample_peak_after > -1.0:
            verdict = "[CLIP]"
        elif sample_peak_after > -2.0:
            verdict = "[WARN]"
        else:
            verdict = "[OK]"
        master_peaks["verdict"] = verdict
        print(f"  Premaster verdict: {verdict}  "
              f"(sample {sample_peak_after:.2f} dBFS, true {tp_final:.2f} dBTP, "
              f"integrated {loudness_final:.2f} LUFS)")
        loudness = loudness_final
        peak_final = tp_final
    else:
        # LEGACY MASTER CHAIN — kept for opt-in via premaster_mode=false.
        # Identical to the pre-premaster-mode behavior.
        target_lufs = master_cfg.get("lufs_target", -14.0)
        loudness = meter.integrated_loudness(master.T)
        print(f"  Integrated loudness: {loudness:.1f} LUFS -> target {target_lufs} LUFS")
        if np.isfinite(loudness):
            gain_db = target_lufs - loudness
            master *= 10.0 ** (gain_db / 20.0)
            peak_post_norm = _peak_db(master)
            master_peaks["after_lufs_norm"] = round(peak_post_norm, 2)
            print(f"  Applied gain: {gain_db:+.1f} dB  -> peak {peak_post_norm:.1f} dBFS {_peak_verdict(peak_post_norm)}")

        tp_limit = master_cfg.get("true_peak_dbfs", -2.0)
        board = Pedalboard([Limiter(threshold_db=float(tp_limit), release_ms=100.0)])
        master = board(master.astype(np.float32), sr).astype(np.float32)

        sample_peak_after = _peak_db(master)
        tp_measured = _measure_true_peak_dbfs(master)
        if tp_measured > tp_limit:
            isp_scale_db = tp_limit - tp_measured
            master *= 10.0 ** (isp_scale_db / 20.0)
            tp_final = tp_limit
            sample_peak_after = _peak_db(master)
            print(f"  Peak after limiter: sample {sample_peak_after:.2f} dBFS  "
                  f"true {tp_measured:.2f} dBTP -> {tp_final:.2f} dBTP "
                  f"(ISP correction {isp_scale_db:+.2f} dB)")
        else:
            tp_final = tp_measured
            print(f"  Peak after limiter: sample {sample_peak_after:.2f} dBFS  "
                  f"true {tp_final:.2f} dBTP")
        master_peaks["after_limiter"] = round(sample_peak_after, 2)
        peak_final = tp_final

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
                sample_peak_after = _peak_db(master)
                master_peaks["after_limiter"] = round(sample_peak_after, 2)
        loudness = loudness_post

        master_peaks["final_sample_peak"] = round(sample_peak_after, 2)
        master_peaks["final_true_peak"] = round(peak_final, 2)
        master_peaks["verdict"] = _peak_verdict(sample_peak_after, peak_final)
        print(f"  Master verdict: {master_peaks['verdict']}  (sample {sample_peak_after:.1f} dBFS, true {peak_final:.1f} dBTP)")

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
        "mix_stage": "premaster" if premaster_mode else "master",
        "active_tracks": len(active),
        "buses": list(processed.keys()),
        # In premaster mode lufs_target / true_peak_limit_dbfs are not applied;
        # mastering owns those numbers. Reported here only for legacy mode.
        "lufs_target": (None if premaster_mode
                        else master_cfg.get("lufs_target", -14.0)),
        "true_peak_limit_dbfs": (None if premaster_mode
                                 else master_cfg.get("true_peak_dbfs", -2.0)),
        "peak_target_dbfs": (float(master_cfg.get("peak_target_dbfs", -3.0))
                             if premaster_mode else None),
        "integrated_lufs": round(loudness, 2) if np.isfinite(loudness) else None,
        "true_peak_dbtp": round(peak_final, 2),
        "sample_peak_dbfs": round(sample_peak_after, 2),
        "duration_s": round(max_length / sr, 3),
        "clipper": clipper_report,
        "ms": ms_report,
        "bus_peaks": bus_peaks,
        "master_peaks": master_peaks,
        "phase_warnings": phase_warnings,
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
    parser.add_argument("input", help="Session output dir (--generate-config) or mix_config.json (--render / --recompute-autotrim)")
    parser.add_argument("--generate-config", action="store_true", help="Scan session dir and write mix_config.json")
    parser.add_argument("--style", default=None,
                        help="Style profile for genre-aware bus volume defaults during --generate-config "
                             "(modern_rock / classic_rock / pop / hip_hop / jazz_acoustic). "
                             "Without it, every bus starts at 0 dB.")
    parser.add_argument("--config", default=None, help="Config output path (default: <session_dir>/mix_config.json)")
    parser.add_argument("--render", action="store_true", help="Render mix from config")
    parser.add_argument("--recompute-autotrim", action="store_true",
                        help="Surgical: read existing mix_config.json, re-measure per-bus dry sums, "
                             "and write auto_trim_db back. Preserves all other fields (active flags, "
                             "volume_db, pan, presets, sends). Use when active tracks change.")
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

    elif args.recompute_autotrim:
        config_path = Path(args.input)
        if not config_path.is_file():
            parser.error(f"Config file not found: {config_path}")
        with open(config_path) as f:
            config = json.load(f)
        print(f"Recomputing per-bus auto-trim (target -18 LUFS per bus) for {config_path}")
        auto_trims = _compute_bus_auto_trims(config, target_lufs=-18.0, verbose=True)
        for name, trim in auto_trims.items():
            if name in config["buses"]:
                config["buses"][name]["auto_trim_db"] = trim
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nUpdated auto_trim_db on {len(auto_trims)} buses in {config_path}")

    elif args.render:
        config_path = Path(args.input)
        if not config_path.is_file():
            parser.error(f"Config file not found: {config_path}")
        output_wav = Path(args.output) if args.output else None
        render_mix(config_path, output_wav, render_stems=args.stems, stage=args.stage)

    else:
        parser.error("Specify --generate-config, --recompute-autotrim, or --render")


if __name__ == "__main__":
    main()
