"""Measure the snare sound of a song: a descriptor profile from isolated hits.

The song is separated into drums / no_drums with demucs (htdemucs, two
stems), isolated snare-like hits are picked from the drum stem, and their
descriptors (tuning, 1/3-octave shape, wire/body ratio, crack, decay, tail;
see _snare.py) are aggregated into a median/MAD profile. The profile is a
matching TARGET: hits cut from a commercial reference are written to
private_ref/ for private listening only and must not be used in a release.

Use --compare-to to measure the distance D of this song's snare (e.g. our own
mix) to a reference: a snare_profile.json, or a snare_match.json (its merged
multi-song profile). D ranks hypotheses; listening decides.

Separation runs in the Python environment that has demucs installed, given
by MUSIC_MIX_AGENT_SEPARATOR_PYTHON (default: the current interpreter).
Stems are cached in <output-dir>/separated/<file hash>/.

Usage:
  python tools/snare_profile.py song.flac --output-dir output/ref/stinkfist
  python tools/snare_profile.py our_mix.wav --output-dir output/ours \\
      --compare-to output/ref/stinkfist/snare_profile.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import _snare as S
from _recall import file_hash

SCHEMA = "snare-profile/1"
MODEL = "htdemucs"


def separate(song: Path, cache_root: Path) -> tuple[Path, Path, dict]:
    """Return (drums, no_drums) stem paths, separating with demucs if not cached."""
    python = os.environ.get("MUSIC_MIX_AGENT_SEPARATOR_PYTHON", sys.executable)
    cache = cache_root / file_hash(song)[:16]
    drums, rest = cache / "drums.wav", cache / "no_drums.wav"
    info = {"python": python, "model": MODEL, "cache": str(cache)}
    if drums.exists() and rest.exists():
        return drums, rest, info
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [python, "-m", "demucs", "-n", MODEL, "--two-stems", "drums", "--float32", "-o", tmp, str(song)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"demucs failed ({proc.returncode}): {proc.stderr.strip()[-1500:]}")
        out = Path(tmp) / MODEL / song.stem
        cache.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out / "drums.wav"), drums)
        shutil.move(str(out / "no_drums.wav"), rest)
    return drums, rest, info


def _save_private_hits(drums: np.ndarray, hits: list[dict], out_dir: Path, count: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.txt").write_text(
        "Snare hits cut from a reference recording for private listening and comparison only.\n"
        "They are not licensed for release: never place them in a mix that leaves this machine.\n",
        encoding="utf-8")
    files = []
    for i, hit in enumerate(hits[:count], 1):
        seg = drums[hit["onset"] - S.ms(1):hit["onset"] + S.ms(450)].copy()
        fade = S.ms(80)
        seg[-fade:] *= np.linspace(1.0, 0.0, fade) ** 2
        seg *= 10.0 ** (-1.0 / 20.0) / (np.abs(seg).max() + 1e-12)
        path = out_dir / f"hit_{i:02d}_{hit['time_s']:.1f}s.wav"
        sf.write(str(path), seg, S.SR, subtype="PCM_24")
        files.append(str(path))
    return files


def snare_profile(song: Path, output_dir: Path, drums_stem: Path | None = None,
                  no_drums_stem: Path | None = None, min_hits: int = 8, save_hits: int = 6,
                  compare_to: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if drums_stem and no_drums_stem:
        separator = {"python": None, "model": "given stems"}
    else:
        drums_stem, no_drums_stem, separator = separate(song, output_dir / "separated")
    drums, rest = S.load_mono(drums_stem), S.load_mono(no_drums_stem)
    n = min(len(drums), len(rest))
    hits = S.find_reference_hits(drums[:n], rest[:n])

    report = {"schema": SCHEMA, "extractor": S.EXTRACTOR_VERSION, "source": str(song), "source_hash": file_hash(song),
              "separator": separator, "n_hits": len(hits),
              "status": "ok" if len(hits) >= min_hits else "insufficient_evidence"}
    if hits:
        descs = [h["desc"] for h in hits]
        report["profile"] = S.aggregate(descs)
        if len(descs) >= 3:
            loo = [S.distance(descs[i], S.aggregate(descs[:i] + descs[i + 1:]))[0] for i in range(len(descs))]
            report["loo_d_p75"] = round(float(np.percentile(loo, 75)), 2)
        report["private_ref_hits"] = _save_private_hits(drums, hits, output_dir / "private_ref", save_hits)
    report["hits"] = [{k: v for k, v in h.items() if k != "desc"} for h in hits]
    if compare_to and "profile" in report:
        ref = json.loads(Path(compare_to).read_text(encoding="utf-8"))
        target = ref.get("merged_profile") or ref["profile"]  # snare_match.json or snare_profile.json
        d, z = S.distance(S.as_descriptor(report["profile"]), target)
        report["compare"] = {"reference": str(compare_to), "D": round(d, 2), "z": z,
                             "reference_accept_d": ref.get("accept_d", ref.get("loo_d_p75"))}
    (output_dir / "snare_profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a song's snare sound as a descriptor profile.")
    parser.add_argument("song", type=Path, help="Song or mix (wav/flac/aiff)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drums-stem", type=Path, help="Use this drum stem instead of separating")
    parser.add_argument("--no-drums-stem", type=Path, help="Matching no_drums stem (with --drums-stem)")
    parser.add_argument("--min-hits", type=int, default=8, help="Hits needed for status ok (default 8)")
    parser.add_argument("--save-hits", type=int, default=6, help="Best hits written to private_ref/ (default 6)")
    parser.add_argument("--compare-to", type=Path, help="Reference snare_profile.json or snare_match.json to measure D against")
    args = parser.parse_args()
    if not args.song.exists():
        print(json.dumps({"error": f"File not found: {args.song}"}), file=sys.stderr)
        sys.exit(1)
    if bool(args.drums_stem) != bool(args.no_drums_stem):
        parser.error("--drums-stem and --no-drums-stem go together")
    try:
        report = snare_profile(args.song, args.output_dir, args.drums_stem, args.no_drums_stem,
                               args.min_hits, args.save_hits, args.compare_to)
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)
    summary = {k: report.get(k) for k in ("status", "n_hits", "loo_d_p75", "compare")}
    summary["profile"] = {k: v for k, v in report.get("profile", {}).items() if k not in ("shape", "shape_hz")}
    summary["report"] = str(args.output_dir / "snare_profile.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
