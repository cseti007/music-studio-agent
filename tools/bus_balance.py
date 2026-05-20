"""Per-bus loudness balance report.

For a finished mix render with --stems, reads each bus stem
(output/<session>/stems/stem_<bus>.wav), applies the bus volume_db
from mix_config.json (including parent-bus chain), and measures
the resulting LUFS. The numbers show how loud each bus actually
contributes to the final mix sum — the data behind perception
questions like "is the bass too loud?" or "are drums too quiet?".

Note: only top-level buses (parent_bus == null) are directly summed
into master. Sub-buses are shown for reference but their LUFS goes
through their parent's processing.

Usage:
    python tools/bus_balance.py path/to/mix_config.json
"""

import json
import sys
import soundfile as sf
import numpy as np
import pyloudnorm as pyln
from pathlib import Path


def _effective_volume_db(bus_name: str, buses_cfg: dict) -> tuple[float, bool]:
    """Sum volume_db up the parent chain. Returns (vol_db, is_toplevel)."""
    cfg = buses_cfg.get(bus_name, {})
    vol = float(cfg.get("volume_db", 0))
    parent = cfg.get("parent_bus")
    is_toplevel = parent is None
    while parent:
        pcfg = buses_cfg.get(parent, {})
        vol += float(pcfg.get("volume_db", 0))
        parent = pcfg.get("parent_bus")
    return vol, is_toplevel


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: bus_balance.py <mix_config.json>", file=sys.stderr)
        sys.exit(1)
    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    session_dir = Path(config.get("session_dir", config_path.parent))
    stems_dir = session_dir / "stems"

    if not stems_dir.exists():
        print(f"ERROR: stems/ directory not found at {stems_dir}.", file=sys.stderr)
        print("Run `render_mix --render --stems` first.", file=sys.stderr)
        sys.exit(1)

    buses_cfg = config["buses"]
    results = []

    for stem_file in sorted(stems_dir.glob("stem_*.wav")):
        bus = stem_file.stem.replace("stem_", "")
        data, sr = sf.read(stem_file, always_2d=True)
        vol_db, is_toplevel = _effective_volume_db(bus, buses_cfg)
        scaled = data * (10 ** (vol_db / 20.0))
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(scaled)
        sample_peak = 20 * np.log10(max(np.max(np.abs(scaled)), 1e-12))
        results.append((bus, vol_db, lufs, sample_peak, is_toplevel))

    results.sort(key=lambda r: -r[2])
    max_lufs = max(r[2] for r in results)

    print("PER-BUS LOUDNESS CONTRIBUTION (effective LUFS in mix)")
    print(f"  config : {config_path}")
    print(f"  stems  : {stems_dir}")
    print()
    print(f'{"BUS":<12s} {"top?":>5s} {"vol_db":>7s} {"LUFS_eff":>9s} {"peak_dBFS":>10s}  ratio (loudest=0dB)')
    print("-" * 88)
    for bus, vol, lufs, peak, is_toplevel in results:
        delta = lufs - max_lufs
        bar_width = max(0, 30 + int(delta))
        bar = "#" * bar_width
        top_marker = "[T]" if is_toplevel else " - "
        print(f"  {bus:<10s} {top_marker:>5s} {vol:+6.1f} {lufs:>9.2f} {peak:>10.2f}  {delta:+5.1f} dB  {bar}")

    print()
    print("[T] = top-level bus (summed directly into master).")
    print("Other rows are sub-buses for reference; their contribution")
    print("flows through the parent's processing.")


if __name__ == "__main__":
    main()
