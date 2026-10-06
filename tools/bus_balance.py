"""Per-bus loudness balance report.

For a finished mix render with --stems, reports the LUFS each bus actually
contributes to the master sum — the data behind perception questions like
"is the bass too loud?" or "are the guitars too quiet?".

`render_mix --render --stems` writes `stems/stem_<bus>.wav` as 32-bit float
at the bus output level (after auto_trim_db + volume_db, pan and the bus
chain; no renormalization). The stem's measured LUFS and peak therefore ARE
its level in the mix, before the master chain (glue comp, peak
normalization), which scales all buses together. Stage renders
(`--stage X --stems`) write to `stems/<stage>/` and are not read here.

Note: only top-level buses (parent_bus == null) are summed directly into
master. Sub-buses are shown for reference but their contribution flows
through the parent's processing chain.

Usage:
    python tools/bus_balance.py path/to/mix_config.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def _is_toplevel(bus_name: str, buses_cfg: dict) -> bool:
    return buses_cfg.get(bus_name, {}).get("parent_bus") is None


def _effective_volume_db(bus_name: str, buses_cfg: dict) -> float:
    """Sum volume_db up the parent chain (for the display column only)."""
    cfg = buses_cfg.get(bus_name, {})
    vol = float(cfg.get("volume_db", 0))
    parent = cfg.get("parent_bus")
    while parent:
        pcfg = buses_cfg.get(parent, {})
        vol += float(pcfg.get("volume_db", 0))
        parent = pcfg.get("parent_bus")
    return vol


def measure_buses(config_path: Path) -> list[dict]:
    """Measure every stems/stem_<bus>.wav, loudest first."""
    config = json.loads(Path(config_path).read_text())
    session_dir = Path(config.get("session_dir", Path(config_path).parent))
    stems_dir = session_dir / "stems"
    if not stems_dir.exists():
        raise FileNotFoundError(f"stems/ directory not found at {stems_dir}; "
                                "run `render_mix --render --stems` first")
    buses_cfg = config["buses"]
    rows = []
    for stem_file in sorted(stems_dir.glob("stem_*.wav")):
        bus = stem_file.stem.replace("stem_", "")
        data, sr = sf.read(stem_file, always_2d=True)
        rows.append({
            "bus": bus,
            "volume_db": _effective_volume_db(bus, buses_cfg),
            "lufs": float(pyln.Meter(sr).integrated_loudness(data)),
            "peak_dbfs": float(20 * np.log10(max(np.max(np.abs(data)), 1e-12))),
            "top": _is_toplevel(bus, buses_cfg),
        })
    rows.sort(key=lambda r: -r["lufs"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-bus loudness contribution measured from <session>/stems/stem_<bus>.wav.")
    parser.add_argument("config", type=Path, help="mix_config.json of a render made with --stems")
    args = parser.parse_args()
    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    try:
        results = measure_buses(args.config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    finite = [r["lufs"] for r in results if np.isfinite(r["lufs"])]
    max_lufs = max(finite) if finite else 0.0

    print("PER-BUS LOUDNESS CONTRIBUTION (measured stem LUFS, before the master chain)")
    print(f"  config : {args.config}")
    print()
    print(f'{"BUS":<12s} {"top?":>5s} {"vol_db":>7s} {"LUFS_eff":>9s}'
          f' {"peak_dBFS":>10s}  ratio (loudest=0dB)')
    print("-" * 88)
    for r in results:
        lufs, peak = r["lufs"], r["peak_dbfs"]
        if not np.isfinite(lufs):
            delta_str = "  n/a"
            bar = ""
        else:
            delta = lufs - max_lufs
            delta_str = f"{delta:+5.1f} dB"
            bar = "#" * max(0, 30 + int(delta))
        top_marker = "[T]" if r["top"] else " - "
        lufs_str = f"{lufs:>9.2f}" if np.isfinite(lufs) else "      n/a"
        peak_str = f"{peak:>10.2f}" if np.isfinite(peak) else "       n/a"
        print(f"  {r['bus']:<10s} {top_marker:>5s} {r['volume_db']:+6.1f} {lufs_str}"
              f" {peak_str}  {delta_str}  {bar}")

    print()
    print("[T] = top-level bus (summed directly into master).")
    print("Other rows are sub-buses for reference; their contribution")
    print("flows through the parent's processing.")


if __name__ == "__main__":
    main()
