"""Shared stage -> stem-file resolution used by render_mix and detect_masking.

Each stage maps to an ordered list of filename candidates. The first existing
file in the track directory wins. Earlier names are more specific (later
processing chain) so that aligned/eq/comp variants are preferred over the raw
assembled output.

The "fx" stage is special — its meaning differs per consumer (render_mix uses
the file from mix_config.json as-is; detect_masking globs for chain suffixes),
so it is not represented here.
"""

from pathlib import Path

# Stage -> ordered filename candidates (first match wins).
STAGE_CANDIDATES: dict[str, list[str]] = {
    "raw": [
        "assembled.wav",
    ],
    "eq": [
        "assembled_aligned_eq.wav",
        "assembled_eq.wav",
        "assembled.wav",
    ],
    "comp": [
        "assembled_aligned_eq_comp.wav",
        "assembled_eq_comp.wav",
        "assembled_aligned_eq.wav",
        "assembled_eq.wav",
        "assembled.wav",
    ],
}

# Stage names that consumers expose on the CLI (in addition to "fx").
STAGE_NAMES = list(STAGE_CANDIDATES.keys()) + ["fx"]


def find_stage_file(track_dir: Path, stage: str) -> Path | None:
    """Return the best matching stem file in track_dir for the given stage.

    Returns None if no candidate exists (and the caller must decide what to do).
    Unknown stages return None.
    """
    for name in STAGE_CANDIDATES.get(stage, []):
        p = track_dir / name
        if p.exists():
            return p
    return None
