"""Record file-processing calls without losing resolved arguments or dependencies.

Schema 2 operation records keep data dependencies (inputs, sources, presets)
separate from "engine" hashes of the local tool modules loaded when the call
ran. Schema 1 records mixed both into "dependencies"; split_dependencies()
classifies tools/*.py paths there as engine files.
"""

import contextlib
import functools
import hashlib
import inspect
import json
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 2
SCHEMA_VERSIONS = (1, 2)
_suppressed = False

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


def is_engine_path(path):
    """Tool modules (tools/*.py) are engine code, not data dependencies."""
    path = Path(path)
    return path.suffix == ".py" and path.parent.name == TOOLS_DIR.name


def _config_was_used(operation):
    """Schema 1 recorded config.toml for every call; only an unresolved
    per-clip target with normalization enabled actually read it."""
    args = operation.get("args", {})
    kwargs = args.get("kwargs", {})
    return (args.get("function") == "apply_gain_per_clip" and kwargs.get("normalize")
            and kwargs.get("target_lufs") is None)


def split_dependencies(operation):
    """Return (data, engine) hashes for schema 1 and schema 2 records.

    Engine entries (tool code, and schema 1 config.toml the call never read)
    only warn on replay; data entries must match.
    """
    recorded = operation.get("dependencies", {})
    data, engine = {}, dict(operation.get("engine", {}))
    for path, digest in recorded.items():
        unused_config = Path(path).name == "config.toml" and not _config_was_used(operation)
        (engine if is_engine_path(path) or unused_config else data)[path] = digest
    return data, engine


def _engine_hashes():
    """Hash local tool modules loaded in this process, including lazy imports.

    Over-inclusive when several tools share one interpreter (tests, replay),
    exact for a single CLI run.
    """
    files = {Path(module.__file__).resolve() for module in list(sys.modules.values())
             if isinstance(getattr(module, "__file__", None), str)}
    return {str(path): file_hash(path) for path in sorted(files)
            if path.parent == TOOLS_DIR and path.suffix == ".py"}


@contextlib.contextmanager
def recording_suppressed():
    """Run recorded callables without writing operation files (verifying replay)."""
    global _suppressed
    previous, _suppressed = _suppressed, True
    try:
        yield
    finally:
        _suppressed = previous


def _is_path_parameter(parameter):
    annotation = parameter.annotation
    return "Path" in (annotation if isinstance(annotation, str) else repr(annotation))


def _session_sources(session_json, track_name, selected, cache):
    """Hash only the clip sources of the track that produced this output."""
    session = json.loads(Path(session_json).read_text(encoding="utf-8"))
    dependencies = {}
    for track in session["tracks"]:
        if track_name is not None and track["name"] != track_name:
            continue
        if track_name is None and selected and track["name"] not in selected:
            continue
        for clip in track.get("clips", []):
            source = str(Path(clip["source_file"]).resolve())
            if source not in cache and Path(source).is_file():
                cache[source] = content_hash(source)
            if source in cache:
                dependencies[source] = cache[source]
    return dependencies


def record_operation(module, resolve=None):
    """Capture complete callable arguments and per-output recall metadata.

    Human-readable report aliases remain available. Operation files are
    attached to individual outputs, so a second EQ cannot erase the first.
    `resolve(arguments)` may replace config or mode-dependent defaults in the
    bound arguments, so the record holds the values actually used.
    """
    def decorate(function):
        if function.__name__ not in CALLS[module]:
            raise ValueError("Unregistered audio operation")
        signature = inspect.signature(function)
        path_parameters = {name for name, parameter in signature.parameters.items()
                           if _is_path_parameter(parameter)}

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            # Path-annotated parameters given as str are paths too: hash and restore them.
            for key in path_parameters:
                if isinstance(bound.arguments.get(key), str):
                    bound.arguments[key] = Path(bound.arguments[key])
            if resolve is not None:
                resolve(bound.arguments)
            if _suppressed:
                return function(*bound.args, **bound.kwargs)
            values = dict(bound.arguments)
            path_keys = [k for k, v in values.items() if isinstance(v, Path)]
            for key in path_keys:
                values[key] = str(Path(values[key]).resolve())
            # Validate serializability before processing or writing audio.
            for key, value in values.items():
                try:
                    json.dumps(value, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Argument {key!r} of {function.__name__} cannot be "
                                     f"recorded for recall (non-finite or non-JSON value): {exc}") from None
            dependencies = {}
            for key in path_keys:
                if key not in ("output_dir", "output_path") and Path(values[key]).is_file():
                    dependencies[values[key]] = content_hash(values[key])
            # Presets may supply defaults inside the callable.
            preset = values.get("preset") or values.get("preset_name")
            preset_path = TOOLS_DIR / "presets" / f"{preset}.json"
            if preset and preset_path.is_file():
                dependencies[str(preset_path)] = file_hash(preset_path)
            result = function(*bound.args, **bound.kwargs)
            reports = [r for r in (result if isinstance(result, list) else [result])
                       if r and r.get("output")]
            engine = _engine_hashes() if reports else {}
            source_cache = {}
            for report in reports:
                output = Path(report["output"]).resolve()
                call_values = dict(values)
                call_dependencies = dict(dependencies)
                if "track_names" in values and "track" in report:
                    call_values.update(track_names=[report["track"]], all_tracks=False)
                if "session_json" in values:
                    call_dependencies.update(_session_sources(
                        values["session_json"], report.get("track"), values.get("track_names"), source_cache))
                inputs = [values[k] for k in ("input_path", "file_path", "target_path", "session_json") if k in values]
                operation = {
                    "schema_version": SCHEMA_VERSION,
                    "step": "recorded_call",
                    "input": inputs[0] if inputs else None,
                    "output": str(output),
                    "args": {"module": module, "function": function.__name__,
                             "kwargs": call_values, "path_keys": path_keys},
                    "dependencies": call_dependencies,
                    "engine": engine,
                    "output_hash": content_hash(output),
                }
                output.with_suffix(".operation.json").write_text(
                    json.dumps(operation, indent=2, allow_nan=False), encoding="utf-8")
            return result
        return wrapped
    return decorate
