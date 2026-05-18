"""Parse a DAW session file into a canonical clip-layout JSON.

Supported formats:
  Pro Tools  .ptx / .pts / .ptf  -- via ptftool binary
  Ableton Live  .als              -- gzip-compressed XML (stdlib only)

Canonical output (session.json):
{
  "session_file": str,
  "daw": "protools" | "ableton",
  "sample_rate": int,
  "duration_samples": int,
  "tracks": [
    {
      "name": str,
      "clips": [
        {
          "source_file": str,
          "timeline_start_sample": int,
          "source_offset_sample": int,
          "length_samples": int
        }
      ]
    }
  ]
}
"""

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Override with the PTFTOOL_PATH environment variable if installed elsewhere.
PTFTOOL = Path(os.environ.get("PTFTOOL_PATH", "/tmp/ptformat/ptftool"))


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".ptx", ".pts", ".ptf"):
        return "protools"
    if ext == ".als":
        return "ableton"
    raise ValueError(f"Unsupported session format: '{ext}' — supported: .ptx .pts .ptf .als")


# ---------------------------------------------------------------------------
# Pro Tools
# ---------------------------------------------------------------------------

_PT_SR_RE = re.compile(r"Samplerate\s*=\s*(\d+)Hz")
_PT_CLIP_RE = re.compile(
    r"^`(.+?)`\s+t\(\d+\)\s+\((.+?)\)\s+@\s+(\d+)\s+\+\s+(\d+),\s+(\d+)"
)


def _resolve_audio(filename: str, audio_dir: Path | None) -> str:
    if audio_dir is None:
        return filename
    matches = list(audio_dir.glob(filename))
    if matches:
        return str(matches[0])
    # Unicode-normalized fallback: ptftool output (NFC, common on Linux)
    # and disk names (NFD, common on macOS-originated filesystems) can
    # represent the same character with different code-point sequences.
    import unicodedata
    target_nfc = unicodedata.normalize("NFC", filename)
    target_nfd = unicodedata.normalize("NFD", filename)
    lower = filename.lower()
    for p in audio_dir.iterdir():
        disk_nfc = unicodedata.normalize("NFC", p.name)
        if disk_nfc == target_nfc or disk_nfc == target_nfd:
            return str(p)
        if p.name.lower() == lower:
            return str(p)
    return filename  # unresolved — return as-is


def _parse_protools(session_path: Path, audio_dir: Path | None) -> dict:
    if not PTFTOOL.exists():
        raise FileNotFoundError(
            f"ptftool not found at {PTFTOOL}.\n"
            "Build it:\n"
            "  git clone https://github.com/zamaudio/ptformat.git /tmp/ptformat\n"
            "  cd /tmp/ptformat && CXX=g++ make all INCL='-I.'\n"
            "Or set PTFTOOL_PATH to point at an existing binary."
        )

    proc = subprocess.run(
        [str(PTFTOOL), str(session_path)],
        capture_output=True, text=True, errors="replace",
    )
    output = proc.stdout + proc.stderr

    sample_rate = 48000
    sr_m = _PT_SR_RE.search(output)
    if sr_m:
        sample_rate = int(sr_m.group(1))

    tracks: dict[str, list] = {}
    max_end = 0

    for line in output.splitlines():
        m = _PT_CLIP_RE.match(line.strip())
        if not m:
            continue
        track_name       = m.group(1)
        source_filename  = m.group(2)
        timeline_start   = int(m.group(3))
        source_offset    = int(m.group(4))
        length           = int(m.group(5))

        tracks.setdefault(track_name, []).append({
            "source_file":            _resolve_audio(source_filename, audio_dir),
            "timeline_start_sample":  timeline_start,
            "source_offset_sample":   source_offset,
            "length_samples":         length,
        })
        max_end = max(max_end, timeline_start + length)

    track_list = sorted(
        [
            {"name": name, "clips": sorted(clips, key=lambda c: c["timeline_start_sample"])}
            for name, clips in tracks.items()
        ],
        key=lambda t: t["name"],
    )

    return {
        "session_file":    str(session_path),
        "daw":             "protools",
        "sample_rate":     sample_rate,
        "duration_samples": max_end,
        "tracks":          track_list,
    }


# ---------------------------------------------------------------------------
# Ableton Live
# ---------------------------------------------------------------------------

def _find_val(el: ET.Element, tag: str, default: float = 0.0) -> float:
    child = el.find(tag)
    if child is not None and child.get("Value") is not None:
        return float(child.get("Value"))
    return default


def _ableton_bpm(root: ET.Element) -> float:
    el = root.find(".//MasterTrack//Tempo/Manual")
    if el is not None and el.get("Value"):
        return float(el.get("Value"))
    return 120.0


def _ableton_sample_rate(root: ET.Element) -> int:
    el = root.find(".//SampleRef/SampleRate")
    if el is not None and el.get("Value"):
        return int(float(el.get("Value")))
    return 44100


def _ableton_source_file(clip: ET.Element, session_dir: Path) -> str:
    ref = clip.find(".//SampleRef/FileRef")
    if ref is None:
        return ""
    abs_el = ref.find("Path")
    if abs_el is not None and abs_el.get("Value"):
        return abs_el.get("Value")
    rel_el = ref.find("RelativePath")
    if rel_el is not None and rel_el.get("Value"):
        return str((session_dir / rel_el.get("Value")).resolve())
    return ""


def _beats_to_samples(beats: float, bpm: float, sr: int) -> int:
    return int(beats * (60.0 / bpm) * sr)


def _parse_ableton(session_path: Path) -> dict:
    with gzip.open(str(session_path), "rb") as f:
        root = ET.parse(f).getroot()

    bpm         = _ableton_bpm(root)
    sample_rate = _ableton_sample_rate(root)
    session_dir = session_path.parent

    track_list = []
    max_end    = 0

    for audio_track in root.findall(".//AudioTrack"):
        name_el    = audio_track.find("Name/EffectiveName")
        track_name = name_el.get("Value", "Unknown") if name_el is not None else "Unknown"

        clips = []
        clip_path = (
            "DeviceChain/MainSequencer/ClipTimeable"
            "/ArrangerAutomation/Events/AudioClip"
        )
        for clip in audio_track.findall(clip_path):
            time_beats       = float(clip.get("Time", 0))
            start_rel_beats  = _find_val(clip, "StartRelative")
            out_marker_beats = _find_val(clip, "OutMarker")
            length_beats     = out_marker_beats - start_rel_beats

            if length_beats <= 0:
                continue

            tl_start  = _beats_to_samples(time_beats, bpm, sample_rate)
            src_offset = _beats_to_samples(start_rel_beats, bpm, sample_rate)
            length_s  = _beats_to_samples(length_beats, bpm, sample_rate)

            clips.append({
                "source_file":            _ableton_source_file(clip, session_dir),
                "timeline_start_sample":  tl_start,
                "source_offset_sample":   src_offset,
                "length_samples":         length_s,
            })
            max_end = max(max_end, tl_start + length_s)

        if clips:
            track_list.append({
                "name":  track_name,
                "clips": sorted(clips, key=lambda c: c["timeline_start_sample"]),
            })

    track_list.sort(key=lambda t: t["name"])

    return {
        "session_file":     str(session_path),
        "daw":              "ableton",
        "sample_rate":      sample_rate,
        "bpm":              bpm,
        "duration_samples": max_end,
        "tracks":           track_list,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_session(
    session_path: Path,
    output_dir: Path,
    audio_dir: Path | None = None,
) -> dict:
    fmt = _detect_format(session_path)

    if fmt == "protools":
        data = _parse_protools(session_path, audio_dir)
    else:
        data = _parse_ableton(session_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "session.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    data["_output"] = str(out_path)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a DAW session file into canonical clip-layout JSON."
    )
    parser.add_argument("session", type=Path, help="Session file (.ptx .pts .ptf .als)")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Where to write session.json (default: ./output)",
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=None,
        help="Directory containing the audio files (resolves filenames to full paths)",
    )
    args = parser.parse_args()

    if not args.session.exists():
        print(json.dumps({"error": f"Session file not found: {args.session}"}), file=sys.stderr)
        sys.exit(1)

    result = parse_session(args.session, args.output_dir, args.audio_dir)

    summary = {
        "session_file":     result["session_file"],
        "daw":              result["daw"],
        "sample_rate":      result["sample_rate"],
        "duration_samples": result["duration_samples"],
        "duration_sec":     round(result["duration_samples"] / result["sample_rate"], 2),
        "track_count":      len(result["tracks"]),
        "total_clips":      sum(len(t["clips"]) for t in result["tracks"]),
        "output":           result.get("_output"),
    }
    if "bpm" in result:
        summary["bpm"] = result["bpm"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
