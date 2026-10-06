"""Record file-processing calls without losing resolved arguments or dependencies."""

import functools
import hashlib
import inspect
import json
from pathlib import Path


CALLS = {
    "apply_dynamic_eq": ("apply_dynamic_eq",),
    "apply_automation": ("apply_automation",),
    "level_notes": ("level_notes_file",),
    "compare_reference": ("_apply_eq_to_target",),
    "apply_gain": ("apply_gain_per_clip", "apply_gain_per_channel"),
    "align_phase": ("align_phase",),
    "apply_eq": ("apply_eq",),
    "apply_compression": ("apply_compression",),
    "apply_gate": ("apply_gate",),
    "apply_amp": ("apply_amp",),
    "apply_reverb": ("apply_reverb",),
    "apply_transient": ("apply_transient_file",),
    "apply_saturation": ("apply_saturation",),
    "apply_delay": ("apply_delay",),
    "apply_deesser": ("apply_deesser",),
    "apply_pitch_correct": ("apply_pitch_correct",),
    "apply_octaver": ("apply_octaver",),
    "apply_subharm": ("apply_subharm",),
    "apply_haas": ("apply_haas",),
    "apply_exciter": ("apply_exciter",),
    "apply_multiband_comp": ("apply_multiband_comp",),
}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_hash(path):
    """Hash audio samples and format without container timestamps or tags."""
    if Path(path).suffix.lower() not in {".wav", ".wave", ".aif", ".aiff", ".flac", ".ogg"}:
        return file_hash(path)
    import soundfile as sf
    with sf.SoundFile(path) as stream:
        digest = hashlib.sha256(json.dumps({"sample_rate": stream.samplerate,
            "channels": stream.channels, "frames": stream.frames,
            "subtype": stream.subtype}, sort_keys=True).encode())
        for block in stream.blocks(blocksize=65536, dtype="float64", always_2d=True):
            digest.update(block.astype("<f8", copy=False).tobytes())
    return "audio-sha256:" + digest.hexdigest()


def record_operation(module):
    """Capture complete callable arguments and per-output recall metadata.

    Human-readable report aliases remain available. Operation files are
    attached to individual outputs, so a second EQ cannot erase the first.
    """
    def decorate(function):
        if function.__name__ not in CALLS[module]:
            raise ValueError("Unregistered audio operation")
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = dict(bound.arguments)
            path_keys = [k for k, v in values.items() if isinstance(v, Path)]
            for key in path_keys:
                values[key] = str(values[key].resolve())
            # Validate serializability before processing or writing audio.
            json.dumps(values, allow_nan=False)
            dependencies = {}
            tool_root = Path(__file__).parent
            for source in tool_root.glob("*.py"):
                dependencies[str(source.resolve())] = file_hash(source)
            for key in path_keys:
                if key not in ("output_dir", "output_path") and Path(values[key]).is_file():
                    dependencies[values[key]] = content_hash(values[key])
            if "session_json" in values:
                session = json.loads(Path(values["session_json"]).read_text())
                selected = values.get("track_names")
                for track in session["tracks"]:
                    if selected and track["name"] not in selected:
                        continue
                    for clip in track.get("clips", []):
                        source = Path(clip["source_file"]).resolve()
                        if str(source) not in dependencies and source.is_file():
                            dependencies[str(source)] = content_hash(source)
            # Presets and config may supply defaults inside the callable.
            preset = values.get("preset") or values.get("preset_name")
            preset_path = Path(__file__).parent / "presets" / f"{preset}.json"
            for path in (preset_path, Path("config.toml")):
                if path.is_file():
                    dependencies[str(path.resolve())] = file_hash(path)
            result = function(*args, **kwargs)
            for report in result if isinstance(result, list) else [result]:
                if not report or not report.get("output"):
                    continue
                output = Path(report["output"]).resolve()
                call_values = dict(values)
                if "track_names" in values and "track" in report:
                    call_values.update(track_names=[report["track"]], all_tracks=False)
                inputs = [values[k] for k in ("input_path", "file_path", "target_path", "session_json") if k in values]
                operation = {
                    "schema_version": 1,
                    "step": "recorded_call",
                    "input": inputs[0] if inputs else None,
                    "output": str(output),
                    "args": {"module": module, "function": function.__name__,
                             "kwargs": call_values, "path_keys": path_keys},
                    "dependencies": dependencies,
                    "output_hash": content_hash(output),
                }
                output.with_suffix(".operation.json").write_text(
                    json.dumps(operation, indent=2, allow_nan=False), encoding="utf-8")
            return result
        return wrapped
    return decorate
