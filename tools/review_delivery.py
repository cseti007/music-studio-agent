"""Combine current-file checks with explicitly scoped human listening records.

Evidence records are attestations, not authenticated proof of listening. Agents
must copy actual feedback and its source; this tool cannot infer or invent it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from _dsp import worst_channel_true_peak_dbfs
from _recall import content_hash


SCOPES = ("vocal_balance", "section", "full_song", "codec_roundtrip")


def listening_status(artifact_hash: str, evidence: list[dict],
                     require_codec_review: bool = False) -> dict:
    """Ignore stale approvals and prevent narrow feedback from approving a song."""
    if not isinstance(evidence, list):
        raise ValueError("Evidence must be a list of listening records")
    approved = set()
    revision_requested = False
    applicable, stale = [], 0
    for entry in evidence:
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(key), str) and entry[key].strip()
            for key in ("artifact_hash", "reviewer", "source", "quote", "scope", "decision")
        ):
            raise ValueError("Each record needs artifact_hash, reviewer, source, quote, scope and decision")
        if entry["scope"] not in SCOPES or entry["decision"] not in ("approved", "revision_requested"):
            raise ValueError("Unsupported approval scope or decision")
        if entry["artifact_hash"] != artifact_hash:
            stale += 1
            continue
        applicable.append(entry)
        if entry["decision"] == "revision_requested":
            approved.clear()
            revision_requested = True
        else:
            approved.add(entry["scope"])
            if entry["scope"] == "full_song":
                revision_requested = False
    required = {"full_song"} | ({"codec_roundtrip"} if require_codec_review else set())
    missing = sorted(required - approved)
    return {"status": "revision_requested" if revision_requested else ("pending" if missing else "approved"),
            "missing_scopes": missing, "applicable_records": applicable,
            "stale_records_ignored": stale,
            "evidence_limit": "Externally supplied human feedback; authenticity is not verified by this tool"}


def review_delivery(audio: Path, output_dir: Path, evidence: list[dict] | None = None,
                    tp_ceiling: float | None = None, sample_rate: int | None = None,
                    bit_depth: int | None = None, target_lufs: float | None = None,
                    lufs_tolerance: float = 0.5, require_codec_review: bool = False) -> dict:
    for value in (tp_ceiling, target_lufs, lufs_tolerance):
        if value is not None and not np.isfinite(value):
            raise ValueError("Delivery targets must be finite")
    if (tp_ceiling is not None and tp_ceiling > 0) or lufs_tolerance < 0:
        raise ValueError("Ceiling must be <= 0 dBTP and loudness tolerance nonnegative")
    info = sf.info(audio)
    data, sr = sf.read(audio, always_2d=True)
    if info.channels not in (1, 2) or len(data) < 0.4 * sr or not np.isfinite(data).all():
        raise ValueError("Review requires at least 400 ms of finite mono/stereo audio")
    loudness = float(pyln.Meter(sr).integrated_loudness(data))
    peak = float(worst_channel_true_peak_dbfs(data.T))
    checks = {"measurable_signal": bool(np.isfinite(loudness)),
              "sample_peak_below_full_scale": bool(np.max(np.abs(data)) < 1)}
    if tp_ceiling is not None:
        checks["true_peak_ceiling"] = peak <= tp_ceiling
    if sample_rate is not None:
        checks["sample_rate"] = sr == sample_rate
    if bit_depth is not None:
        checks["bit_depth"] = info.subtype == f"PCM_{bit_depth}"
    if target_lufs is not None:
        checks["contractual_loudness"] = bool(np.isfinite(loudness) and abs(loudness - target_lufs) <= lufs_tolerance)
    artifact_hash = content_hash(audio)
    listening = listening_status(artifact_hash, [] if evidence is None else evidence, require_codec_review)
    technical_status = "failed" if not all(checks.values()) else ("incomplete" if tp_ceiling is None else "passed")
    if technical_status != "passed":
        status = f"technical_{technical_status}"
    elif listening["status"] != "approved":
        status = "revision_requested" if listening["status"] == "revision_requested" else "listening_pending"
    else:
        status = "ready_for_delivery"
    report = {"audio": str(audio.resolve()), "artifact_hash": artifact_hash,
              "status": status, "delivery_ready": status == "ready_for_delivery",
              "assessment_scope": "Specified technical checks and recorded human approval; no professional-quality certification",
              "technical": {"status": technical_status, "checks": checks,
                            "true_peak_dbtp": peak, "integrated_lufs": loudness if np.isfinite(loudness) else None,
                            "sample_rate": sr, "subtype": info.subtype, "duration_sec": info.duration,
                            "targets": {"tp_ceiling": tp_ceiling, "sample_rate": sample_rate,
                                        "bit_depth": bit_depth, "contractual_lufs": target_lufs}},
              "listening": listening}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "delivery_review.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, help="JSON list of actual human feedback, oldest first")
    parser.add_argument("--tp-ceiling", type=float)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--bit-depth", type=int, choices=(16, 24, 32))
    parser.add_argument("--target-lufs", type=float, help="Contractual requirement only, not a streaming preference")
    parser.add_argument("--lufs-tolerance", type=float, default=0.5)
    parser.add_argument("--require-codec-review", action="store_true")
    args = vars(parser.parse_args())
    if args["evidence"] is not None:
        args["evidence"] = json.loads(args["evidence"].read_text(encoding="utf-8"))
    report = review_delivery(**args)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["delivery_ready"] else 1)


if __name__ == "__main__":
    main()
