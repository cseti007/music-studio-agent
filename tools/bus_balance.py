"""Per-bus loudness balance report.

For a finished mix render with --stems, reports the LUFS each bus actually
contributes to the master sum — the data behind perception questions like
"is the bass too loud?" or "are the guitars too quiet?".

The on-disk stem files in `stems/stem_<bus>.wav` are renormalized to
-18 LUFS by `_write_stem()` (with an optional peak limit), so they cannot
be read directly to get the in-mix loudness. This tool reconstructs the
true pre-renormalization LUFS using the peak data recorded in
`mixes/mix_report.json` (`bus_peaks[<bus>]["final"]`):

    in_mix_lufs = disk_lufs - disk_peak_dbfs + bus_peaks[bus]["final"]

Derivation: `_write_stem` is a linear scaling (LUFS-norm then optional
peak-limit). Since linear scaling shifts LUFS and peak by the same dB
amount, `disk_lufs - input_lufs = disk_peak - input_peak`. Rearranging
gives the formula above.

Note: only top-level buses (parent_bus == null) are summed directly into
master. Sub-buses are shown for reference but their contribution flows
through the parent's processing chain.

Usage:
    python tools/bus_balance.py path/to/mix_config.json
"""

import json
import sys
import soundfile as sf
import numpy as np
import pyloudnorm as pyln
from pathlib import Path


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
    mix_report_path = session_dir / "mixes" / "mix_report.json"

    if not stems_dir.exists():
        print(f"ERROR: stems/ directory not found at {stems_dir}.", file=sys.stderr)
        print("Run `render_mix --render --stems` first.", file=sys.stderr)
        sys.exit(1)

    bus_peaks: dict = {}
    if mix_report_path.exists():
        try:
            mix_report = json.loads(mix_report_path.read_text())
            bus_peaks = mix_report.get("bus_peaks", {}) or {}
        except Exception as exc:
            print(f"WARNING: could not parse {mix_report_path} ({exc}); "
                  f"falling back to approximate measurement.", file=sys.stderr)
    else:
        print(f"WARNING: {mix_report_path} not found; falling back to "
              f"approximate measurement (results will be inaccurate for "
              f"peak-limited stems and ignore auto_trim).", file=sys.stderr)

    buses_cfg = config["buses"]
    results = []

    for stem_file in sorted(stems_dir.glob("stem_*.wav")):
        bus = stem_file.stem.replace("stem_", "")
        data, sr = sf.read(stem_file, always_2d=True)
        meter = pyln.Meter(sr)
        disk_lufs = float(meter.integrated_loudness(data))
        disk_peak = 20 * np.log10(max(np.max(np.abs(data)), 1e-12))

        bp = bus_peaks.get(bus)
        if bp and bp.get("final") is not None:
            # Reconstruct the true in-mix LUFS via the inverse-write formula.
            in_mix_lufs = disk_lufs - disk_peak + float(bp["final"])
            in_mix_peak = float(bp["final"])
            method = "exact"
        else:
            # Fallback: best-effort approximation using only the on-disk
            # state + volume_db (NOT pan-aware, NOT auto_trim-aware).
            vol_db = _effective_volume_db(bus, buses_cfg)
            in_mix_lufs = disk_lufs + vol_db
            in_mix_peak = disk_peak + vol_db
            method = "approx"

        is_top = _is_toplevel(bus, buses_cfg)
        vol_for_display = _effective_volume_db(bus, buses_cfg)
        results.append((bus, vol_for_display, in_mix_lufs, in_mix_peak,
                        is_top, method))

    # Sort by in_mix_lufs descending (loudest at top)
    results.sort(key=lambda r: -r[2])
    finite = [r[2] for r in results if np.isfinite(r[2])]
    max_lufs = max(finite) if finite else 0.0

    print("PER-BUS LOUDNESS CONTRIBUTION (effective LUFS in mix)")
    print(f"  config : {config_path}")
    print(f"  stems  : {stems_dir}")
    if mix_report_path.exists() and bus_peaks:
        print(f"  source : {mix_report_path.name} (exact)")
    else:
        print(f"  source : disk stems only (approximate — re-render with"
              f" --stems to get exact numbers)")
    print()
    print(f'{"BUS":<12s} {"top?":>5s} {"vol_db":>7s} {"LUFS_eff":>9s}'
          f' {"peak_dBFS":>10s}  ratio (loudest=0dB)')
    print("-" * 88)
    for bus, vol, lufs, peak, is_top, method in results:
        if not np.isfinite(lufs):
            delta_str = "  n/a"
            bar = ""
        else:
            delta = lufs - max_lufs
            delta_str = f"{delta:+5.1f} dB"
            bar_width = max(0, 30 + int(delta))
            bar = "#" * bar_width
        top_marker = "[T]" if is_top else " - "
        flag = "" if method == "exact" else " ~"
        lufs_str = f"{lufs:>9.2f}" if np.isfinite(lufs) else "      n/a"
        peak_str = f"{peak:>10.2f}" if np.isfinite(peak) else "       n/a"
        print(f"  {bus:<10s} {top_marker:>5s} {vol:+6.1f} {lufs_str}"
              f" {peak_str}  {delta_str}  {bar}{flag}")

    print()
    print("[T] = top-level bus (summed directly into master).")
    print("Other rows are sub-buses for reference; their contribution")
    print("flows through the parent's processing.")
    if any(r[5] == "approx" for r in results):
        print("~ = approximate (mix_report.json missing or bus_peaks unavailable).")


if __name__ == "__main__":
    main()
