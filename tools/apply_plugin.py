"""Process a stem through an external VST3 (or macOS Audio Unit) plugin.

Loads the plugin with pedalboard, optionally restores a saved plugin state,
sets parameters by name, renders offline (pedalboard compensates the latency
the plugin reports) and writes a float WAV. The report and operation record
keep the plugin path, identity, version, a hash of the plugin binary or
bundle, the full parameter snapshot after setting values, and the plugin's
raw state saved next to the output, so the call can be replayed and audited.

Limits: plugins that report latency incorrectly, use randomness or
time-dependent modulation, or depend on host tempo will not replay
bit-identically; replay then reports a hash mismatch instead of overwriting.
Ableton's own devices (EQ Eight, Glue Compressor, ...) are not plugins and
cannot be hosted here. Use --list-params to see the parameter names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from _recall import content_hash, record_operation


def _load(plugin_path: Path, plugin_name: str | None):
    from pedalboard import load_plugin
    return load_plugin(str(plugin_path), plugin_name=plugin_name)


def _identity(plugin) -> dict:
    keys = ("name", "descriptive_name", "manufacturer_name", "version", "identifier", "category")
    return {k: getattr(plugin, k, None) for k in keys}


def _snapshot(plugin) -> dict:
    """Every parameter's current value as the plugin reports it."""
    snapshot = {}
    for name, parameter in plugin.parameters.items():
        value = getattr(plugin, name)
        snapshot[name] = value if isinstance(value, (bool, int, float, str)) else str(parameter)
    return snapshot


@record_operation("apply_plugin")
def apply_plugin(input_path: Path, output_dir: Path, plugin_path: Path,
                 params: dict | None = None, plugin_name: str | None = None,
                 state_path: Path | None = None, mix: float = 1.0,
                 tail_sec: float = 0.0, suffix: str = "plugin") -> dict:
    if not 0.0 <= mix <= 1.0 or not 0.0 <= tail_sec <= 60.0:
        raise ValueError("mix must be within 0..1 and tail_sec within 0..60 s")
    if not suffix.replace("_", "").isalnum():
        raise ValueError("suffix may contain only letters, digits and underscores")
    plugin = _load(plugin_path, plugin_name)
    if state_path is not None:
        plugin.raw_state = state_path.read_bytes()
    unknown = sorted(set(params or {}) - set(plugin.parameters))
    if unknown:
        raise ValueError(f"Unknown plugin parameters {unknown}; available: {sorted(plugin.parameters)}")
    for name, value in (params or {}).items():
        setattr(plugin, name, value)

    data, sr = sf.read(input_path, always_2d=True, dtype="float32")
    if not np.isfinite(data).all():
        raise ValueError("Input contains non-finite samples")
    dry = np.pad(data, ((0, round(tail_sec * sr)), (0, 0)))
    wet = np.asarray(plugin.process(dry.T, sr, reset=True), dtype=np.float32).T
    if wet.shape[0] != dry.shape[0]:
        raise ValueError(f"Plugin returned {wet.shape[0]} frames for {dry.shape[0]}; it buffers audio")
    if wet.shape[1] != dry.shape[1]:
        if dry.shape[1] != 1:
            raise ValueError(f"Plugin changed the channel count {dry.shape[1]} -> {wet.shape[1]}")
        dry = np.repeat(dry, wet.shape[1], axis=1)  # mono into a stereo plugin
    if not np.isfinite(wet).all():
        raise ValueError("Plugin output contains non-finite samples")
    result = (1.0 - mix) * dry + mix * wet

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{input_path.stem}_{suffix}.wav"
    sf.write(output, result, sr, subtype="FLOAT")
    state_out = output.with_suffix(".plugin_state.bin")
    state_out.write_bytes(bytes(plugin.raw_state))
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    report = {"input": str(input_path), "output": str(output),
              "plugin": {"path": str(plugin_path), "hash": content_hash(plugin_path),
                         "plugin_name": plugin_name, **_identity(plugin),
                         "reported_latency_samples": getattr(plugin, "reported_latency_samples", None)},
              "params_requested": params or {}, "parameters_after": _snapshot(plugin),
              "state_file": str(state_out), "state_sha256": content_hash(state_out),
              "state_restored_from": str(state_path) if state_path else None,
              "mix": mix, "tail_sec": tail_sec,
              "output_peak_dbfs": round(20 * np.log10(max(peak, 1e-12)), 2),
              "output_exceeds_0dbfs": peak > 1.0,
              "listening_review": {"status": "pending", "performed_by_tool": False}}
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, nargs="?", help="Input WAV")
    parser.add_argument("--plugin", type=Path, required=True, help="Path to the .vst3 (or .component) plugin")
    parser.add_argument("--plugin-name", help="Plugin to load from a multi-plugin bundle")
    parser.add_argument("--output-dir", type=Path, help="Output directory (required unless --list-params)")
    parser.add_argument("--params", type=Path, help="JSON object of parameter name -> value")
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                        help="Set one parameter (repeatable; overrides --params)")
    parser.add_argument("--state", type=Path, help="Restore a saved raw plugin state before setting parameters")
    parser.add_argument("--mix", type=float, default=1.0, help="Dry/wet blend, 1.0 = fully processed")
    parser.add_argument("--tail-sec", type=float, default=0.0,
                        help="Seconds of silence appended so reverb/delay tails are rendered")
    parser.add_argument("--suffix", default="plugin", help="Output name suffix: <input>_<suffix>.wav")
    parser.add_argument("--list-params", action="store_true", help="Print the plugin's parameters and exit")
    args = parser.parse_args()

    if args.list_params:
        plugin = _load(args.plugin, args.plugin_name)
        print(json.dumps({"plugin": _identity(plugin), "parameters": {
            name: {"value": value, "range": str(plugin.parameters[name])}
            for name, value in _snapshot(plugin).items()}}, indent=2, default=str))
        return
    if args.input is None or args.output_dir is None:
        parser.error("input and --output-dir are required unless --list-params")
    params = json.loads(args.params.read_text(encoding="utf-8")) if args.params else {}
    for item in args.param:
        name, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--param expects NAME=VALUE, got {item!r}")
        params[name] = _parse_value(value)
    report = apply_plugin(args.input, args.output_dir, args.plugin, params or None, args.plugin_name,
                          args.state, args.mix, args.tail_sec, args.suffix)
    print(json.dumps({k: report[k] for k in ("output", "output_peak_dbfs", "state_file")}, indent=2))


if __name__ == "__main__":
    main()
