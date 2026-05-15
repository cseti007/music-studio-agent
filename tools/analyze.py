"""Analyze a single audio stem — outputs JSON stats and saves a MEL spectrogram PNG."""

import argparse
import json
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt

DEFAULT_TARGET_LUFS = -23.0


def _rms_db(signal: np.ndarray) -> float:
    rms = np.sqrt(np.mean(signal ** 2))
    return float(20 * np.log10(max(rms, 1e-10)))


def _band_rms_db(signal: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    nyq = sr / 2.0
    high_norm = min(high_hz / nyq, 0.999)
    low_norm = low_hz / nyq

    if low_norm <= 0.001:
        sos = butter(4, high_norm, btype="low", output="sos")
    else:
        sos = butter(4, [low_norm, high_norm], btype="band", output="sos")

    return _rms_db(sosfilt(sos, signal))


def _noise_floor_db(signal: np.ndarray, sr: int, frame_sec: float = 0.1) -> float:
    frame_len = int(sr * frame_sec)
    frames = [
        signal[i : i + frame_len]
        for i in range(0, len(signal) - frame_len, frame_len)
    ]
    rms_vals = [np.sqrt(np.mean(f ** 2)) for f in frames]
    rms_vals = [v for v in rms_vals if v > 1e-10]
    if not rms_vals:
        return -120.0
    return float(20 * np.log10(np.percentile(rms_vals, 5)))


def _dynamic_range_db(signal: np.ndarray, sr: int, frame_sec: float = 0.1) -> float:
    frame_len = int(sr * frame_sec)
    frames = [
        signal[i : i + frame_len]
        for i in range(0, len(signal) - frame_len, frame_len)
    ]
    rms_vals = [np.sqrt(np.mean(f ** 2)) for f in frames if np.max(np.abs(f)) > 1e-6]
    if len(rms_vals) < 10:
        return 0.0
    p10 = np.percentile(rms_vals, 10)
    p95 = np.percentile(rms_vals, 95)
    return float(20 * np.log10((p95 + 1e-10) / (p10 + 1e-10)))


_BLOCKS = " ░▒▓█"
_TEXT_COLS = 60
_TEXT_BANDS = [
    ("SUB  ", 20,    60),
    ("BASS ", 60,    250),
    ("MID  ", 250,   2000),
    ("HIGH ", 2000,  8000),
    ("AIR  ", 8000,  20000),
]


def _text_spectrogram(signal: np.ndarray, sr: int) -> str:
    fmax = min(sr // 2, 20000)
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)  # shape: (n_mels, n_frames), range ~[-80, 0]
    freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=fmax)

    n_frames = S_db.shape[1]
    col_size = max(n_frames // _TEXT_COLS, 1)

    duration = len(signal) / sr
    time_labels = []
    for col in range(_TEXT_COLS):
        t = int(col * duration / _TEXT_COLS)
        time_labels.append(f"{t//60}:{t%60:02d}" if t >= 60 else f"{t}s")

    # header: time markers every 10 columns
    header_parts = ["       "]
    for i, lbl in enumerate(time_labels):
        if i % 10 == 0:
            header_parts.append(f"{lbl:<10}")
    header = "".join(header_parts).rstrip()

    rows = []
    for label, lo, hi in _TEXT_BANDS:
        band_mask = (freqs >= lo) & (freqs < hi)
        if not band_mask.any():
            continue
        band_S = S_db[band_mask, :]  # (band_bins, n_frames)
        # average energy per column
        chars = []
        for col in range(_TEXT_COLS):
            start = col * col_size
            end = min(start + col_size, n_frames)
            energy = float(np.mean(band_S[:, start:end]))
            # map dB range [-60, 0] to 5 levels
            level = int(np.clip((energy + 60) / 12, 0, 4))
            chars.append(_BLOCKS[level])
        rows.append(f"{label}  {''.join(chars)}")

    rows.reverse()  # high frequencies on top
    return "\n".join([header] + rows)


def _save_png_spectrogram(signal: np.ndarray, sr: int, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=min(sr // 2, 20000))
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)


def _save_outputs(
    stats: dict,
    text_spec: str,
    signal: np.ndarray,
    sr: int,
    stem_name: str,
    output_dir: Path,
) -> dict:
    """Save all analysis artifacts into output_dir/<stem_name>/."""
    stem_dir = output_dir / stem_name
    stem_dir.mkdir(parents=True, exist_ok=True)

    png_path = stem_dir / "spectrogram.png"
    txt_path = stem_dir / "spectrogram.txt"
    json_path = stem_dir / "analysis.json"

    _save_png_spectrogram(signal, sr, png_path, title=stem_name)
    txt_path.write_text(text_spec, encoding="utf-8")

    result = {
        **stats,
        "outputs": {
            "analysis_json": str(json_path),
            "spectrogram_png": str(png_path),
            "spectrogram_txt": str(txt_path),
        },
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def analyze(file_path: Path, output_dir: Path, target_lufs: float = DEFAULT_TARGET_LUFS) -> dict:
    data, sr = sf.read(str(file_path), always_2d=True)
    mono = data.mean(axis=1).astype(np.float64)
    channels = data.shape[1]
    duration = len(mono) / sr

    meter = pyln.Meter(sr)
    try:
        lufs_input = data if channels > 1 else mono
        integrated_lufs = float(meter.integrated_loudness(lufs_input))
    except Exception:
        integrated_lufs = -120.0

    true_peak = float(20 * np.log10(max(np.max(np.abs(mono)), 1e-10)))

    freq_bands = {
        "sub_60hz":     (0,    60),
        "low_60_250hz": (60,   250),
        "mid_250_2khz": (250,  2000),
        "high_2_8khz":  (2000, 8000),
        "air_8khz_plus":(8000, min(sr // 2, 20000)),
    }
    frequency_bands = {
        f"{name}_rms_db": round(_band_rms_db(mono, sr, lo, hi), 1)
        for name, (lo, hi) in freq_bands.items()
    }

    noise_floor = round(_noise_floor_db(mono, sr), 1)
    dynamic_range = round(_dynamic_range_db(mono, sr), 1)
    recommended_gain = round(target_lufs - integrated_lufs, 1) if integrated_lufs > -100 else 0.0
    text_spec = _text_spectrogram(mono, sr)

    stats = {
        "file": str(file_path),
        "duration_sec": round(duration, 2),
        "sample_rate": sr,
        "channels": channels,
        "loudness": {
            "integrated_lufs": round(integrated_lufs, 1),
            "true_peak_dbfs": round(true_peak, 1),
            "dynamic_range_db": dynamic_range,
        },
        "frequency_bands": frequency_bands,
        "noise_floor_dbfs": noise_floor,
        "recommended_gain_db": recommended_gain,
        "spectrogram_text": text_spec,
    }

    return _save_outputs(stats, text_spec, mono, sr, file_path.stem, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an audio stem.")
    parser.add_argument("file", type=Path, help="Audio file path (wav/flac/mp3/aiff)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to write analysis artifacts (default: ./output)",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=DEFAULT_TARGET_LUFS,
        help=f"Target loudness for gain recommendation (default: {DEFAULT_TARGET_LUFS})",
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(json.dumps({"error": f"File not found: {args.file}"}), file=sys.stderr)
        sys.exit(1)

    result = analyze(args.file, output_dir=args.output_dir, target_lufs=args.target_lufs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
