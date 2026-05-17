"""Analyze a single audio stem — outputs JSON stats and saves a MEL spectrogram PNG."""

import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt, welch

_CONFIG_PATH = Path("config.toml")
_FALLBACK_TARGET_LUFS = -18.0


def _config_target_lufs() -> float:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
        return float(cfg.get("analyze", {}).get("default_target_lufs", _FALLBACK_TARGET_LUFS))
    return _FALLBACK_TARGET_LUFS


DEFAULT_TARGET_LUFS = _config_target_lufs()


def _rms_db(signal: np.ndarray) -> float:
    rms = np.sqrt(np.mean(signal ** 2))
    return float(20 * np.log10(max(rms, 1e-10)))


def _true_peak_dbfs(signal: np.ndarray, oversample: int = 4) -> float:
    """ITU-R BS.1770-4 style true peak via polyphase upsampling.

    Inter-sample peaks emerge after DAC reconstruction or codec encoding
    (Ogg Vorbis, AAC). Oversampling reveals them before they cause clipping.
    """
    up = resample_poly(signal, oversample, 1)
    peak = float(np.max(np.abs(up)))
    return float(20 * np.log10(max(peak, 1e-10)))


def _band_filter(signal: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyq = sr / 2.0
    high_norm = min(high_hz / nyq, 0.999)
    low_norm = low_hz / nyq
    if low_norm <= 0.001:
        sos = butter(4, high_norm, btype="low", output="sos")
    else:
        sos = butter(4, [low_norm, high_norm], btype="band", output="sos")
    return sosfilt(sos, signal)


def _band_rms_db(signal: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    return _rms_db(_band_filter(signal, sr, low_hz, high_hz))


def _band_crest_db(signal: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    """Peak-to-RMS ratio (crest factor) within a frequency band.

    High crest = transient-rich / dynamic (room to compress).
    Low crest  = sustained / already compressed.

    Used to decide whether multiband comp / sub-synth / parallel sat would
    actually do anything useful in that band.
    """
    band = _band_filter(signal, sr, low_hz, high_hz)
    rms = float(np.sqrt(np.mean(band ** 2)))
    peak = float(np.max(np.abs(band)))
    if rms < 1e-10:
        return 0.0
    return round(20.0 * np.log10(peak / rms), 1)


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


def _detect_hum(signal: np.ndarray, sr: int, prominence_threshold_db: float = 12.0) -> dict:
    """Detect mains hum (50/60 Hz and harmonics) via Welch PSD narrow-band peak analysis.

    A frequency is flagged as hum if its peak is >= prominence_threshold_db above the
    local median of surrounding bins (±40 Hz window, excluding ±3 Hz around the peak).
    Only non-silent frames are analysed to avoid false negatives from silence gaps.
    """
    # Analyse only non-silent frames to avoid silence diluting hum peaks
    frame_len = 4096
    chunks = [
        signal[i : i + frame_len]
        for i in range(0, len(signal) - frame_len, frame_len)
        if np.sqrt(np.mean(signal[i : i + frame_len] ** 2)) > 1e-5
    ]
    if not chunks:
        return {"hum_detected": False, "dominant_mains_hz": None, "harmonics": {}, "recommendation": "No audio content"}

    audio = np.concatenate(chunks)

    # Welch PSD — nperseg tuned for ~1.5 Hz/bin resolution
    nperseg = min(len(audio), 32768)
    freqs, psd = welch(audio, fs=sr, nperseg=nperseg, average="median")
    psd_db = 10.0 * np.log10(psd + 1e-20)
    bin_hz = freqs[1] - freqs[0]

    def _peak_prominence(target_hz: float) -> float | None:
        idx = int(round(target_hz / bin_hz))
        if idx >= len(psd_db):
            return None
        peak_db = psd_db[idx]
        surround = int(round(40.0 / bin_hz))
        exclude = max(1, int(round(3.0 / bin_hz)))
        lo, hi = max(0, idx - surround), min(len(psd_db), idx + surround + 1)
        mask = np.ones(hi - lo, dtype=bool)
        c = idx - lo
        mask[max(0, c - exclude) : c + exclude + 1] = False
        neighbours = psd_db[lo:hi][mask]
        if len(neighbours) == 0:
            return None
        return float(peak_db - np.median(neighbours))

    detected: dict[str, list] = {}
    for mains_hz in (50, 60):
        harmonics = []
        for n in range(1, 7):
            target = mains_hz * n
            if target >= sr / 2:
                break
            prominence = _peak_prominence(float(target))
            if prominence is not None and prominence >= prominence_threshold_db:
                harmonics.append({"frequency_hz": target, "prominence_db": round(prominence, 1)})
        if harmonics:
            detected[f"{mains_hz}hz"] = harmonics

    hum_detected = bool(detected)
    dominant = None
    if "50hz" in detected and "60hz" in detected:
        s50 = sum(h["prominence_db"] for h in detected["50hz"])
        s60 = sum(h["prominence_db"] for h in detected["60hz"])
        dominant = 50 if s50 >= s60 else 60
    elif "50hz" in detected:
        dominant = 50
    elif "60hz" in detected:
        dominant = 60

    if hum_detected:
        rec = f"High-pass filter at {dominant - 10} Hz, or notch at {dominant} Hz and harmonics"
    else:
        rec = "No hum detected"

    return {
        "hum_detected": hum_detected,
        "dominant_mains_hz": dominant,
        "harmonics": detected,
        "recommendation": rec,
    }


_BLOCKS = " ░▒▓█"

_TEXT_BANDS = [
    ("SUB   ", 20,     60),
    ("LOBASS", 60,     120),
    ("BASS  ", 120,    250),
    ("UPBASS", 250,    500),
    ("LOMID ", 500,    1000),
    ("MID   ", 1000,   2000),
    ("UPMID ", 2000,   4000),
    ("PRES  ", 4000,   8000),
    ("AIR   ", 8000,   12000),
    ("HIAIR ", 12000,  20000),
]

_LABEL_WIDTH = 8  # label (6) + "  " (2)


def _text_cols(duration_sec: float) -> int:
    term_cols = shutil.get_terminal_size(fallback=(120, 40)).columns
    max_cols = max(60, term_cols - _LABEL_WIDTH - 2)
    return min(max(60, int(duration_sec)), max_cols)


def _text_spectrogram(signal: np.ndarray, sr: int) -> str:
    duration = len(signal) / sr
    n_cols = _text_cols(duration)

    fmax = min(sr // 2, 20000)
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)
    freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=fmax)

    n_frames = S_db.shape[1]
    col_size = max(n_frames // n_cols, 1)

    # time header — ~10 evenly spaced markers
    label_step = max(10, n_cols // 10)
    header = " " * _LABEL_WIDTH
    for i in range(0, n_cols, label_step):
        t = int(i * duration / n_cols)
        lbl = f"{t//60}:{t%60:02d}" if t >= 60 else f"{t}s"
        header += f"{lbl:<{label_step}}"
    header = header.rstrip()

    rows = []
    for label, lo, hi in _TEXT_BANDS:
        band_mask = (freqs >= lo) & (freqs < hi)
        if not band_mask.any():
            continue
        band_S = S_db[band_mask, :]
        chars = []
        for col in range(n_cols):
            start = col * col_size
            end = min(start + col_size, n_frames)
            energy = float(np.mean(band_S[:, start:end]))
            level = int(np.clip((energy + 60) / 12, 0, 4))
            chars.append(_BLOCKS[level])
        rows.append(f"{label}  {''.join(chars)}")

    rows.reverse()  # high frequencies on top

    # RMS amplitude waveform — shows dynamic envelope over time
    samples_per_col = max(1, len(signal) // n_cols)
    rms_chars = []
    for col in range(n_cols):
        start = col * samples_per_col
        end = min(start + samples_per_col, len(signal))
        rms_db = float(20 * np.log10(max(np.sqrt(np.mean(signal[start:end] ** 2)), 1e-10)))
        level = int(np.clip((rms_db + 60) / 12, 0, 4))
        rms_chars.append(_BLOCKS[level])

    rows.append("-" * (n_cols + _LABEL_WIDTH))
    rows.append(f"{'RMS   '}  {''.join(rms_chars)}")

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


def _frequency_response(signal: np.ndarray, sr: int) -> list[dict]:
    """1/3-octave smoothed frequency response from Welch PSD."""
    nperseg = min(len(signal), 32768)
    freqs, psd = welch(signal.astype(np.float64), fs=sr, nperseg=nperseg, average="mean")
    psd_db = 10.0 * np.log10(psd + 1e-20)

    centers: list[float] = []
    f = 20.0
    while f <= min(sr / 2.0, 20000.0):
        centers.append(f)
        f *= 2.0 ** (1.0 / 3.0)

    result = []
    for fc in centers:
        lo = fc / 2.0 ** (1.0 / 6.0)
        hi = fc * 2.0 ** (1.0 / 6.0)
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            result.append({"hz": round(fc, 1), "db": round(float(np.mean(psd_db[mask])), 1)})

    return result


def _freq_response_text(freq_response: list[dict]) -> str:
    if not freq_response:
        return ""
    db_vals = [r["db"] for r in freq_response]
    db_max = max(db_vals)
    db_floor = db_max - 60.0
    lines = ["", "1/3-OCTAVE FREQUENCY RESPONSE", "-" * 54]
    for r in freq_response:
        hz, db = r["hz"], r["db"]
        bar_len = int(np.clip((db - db_floor) / 60.0 * 40, 0, 40))
        hz_label = f"{hz:6.0f} Hz" if hz < 1000 else f"{hz / 1000:5.2f} kHz"
        lines.append(f"{hz_label}  {'█' * bar_len:<40}  {db:+.1f} dB")
    return "\n".join(lines)


def _lra(data: np.ndarray, sr: int, meter: pyln.Meter) -> float:
    """EBU R128 Loudness Range (LRA) in LU. Returns 0.0 if too short for measurement."""
    try:
        return round(float(meter.loudness_range(data)), 1)
    except Exception:
        return 0.0


def _detect_pumping(signal: np.ndarray, sr: int) -> dict:
    """Detect compressor pumping artifact via low-frequency envelope modulation.

    Pumping = audible periodic dipping of the signal level caused by a release
    time tuned wrong (or hit too hard) on a compressor. Spectrally it shows up
    as energy in the signal's amplitude envelope around 1-5 Hz.

    Method:
      1. Compute a short-window RMS envelope (10 ms hop).
      2. High-pass the envelope above 0.5 Hz to remove the overall level.
      3. Look at envelope spectral peak in 1-5 Hz vs. the 5-15 Hz reference.
      4. Pumping is detected if the 1-5 Hz peak exceeds the reference by
         >= 6 dB and the absolute modulation depth exceeds ~5%.
    """
    hop_ms = 10.0
    hop = max(1, int(sr * hop_ms / 1000.0))
    n_frames = len(signal) // hop
    if n_frames < 200:
        return {"pumping_detected": False, "pump_rate_hz": None, "modulation_depth_db": None,
                "note": "signal too short for pumping analysis"}

    env = np.array([
        np.sqrt(np.mean(signal[i * hop:(i + 1) * hop] ** 2) + 1e-20)
        for i in range(n_frames)
    ])
    env_sr = sr / hop  # ~100 Hz

    # Remove DC and very slow drift (< 0.5 Hz)
    from scipy.signal import butter as _butter_local
    sos_hp = _butter_local(2, 0.5 / (env_sr / 2.0), btype="high", output="sos")
    env_hp = sosfilt(sos_hp, env)

    # Spectrum of the envelope
    nperseg = min(len(env_hp), 1024)
    freqs, psd = welch(env_hp, fs=env_sr, nperseg=nperseg)

    def _band_peak_db(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            return -120.0
        peak = float(np.max(psd[mask]))
        return 10.0 * np.log10(peak + 1e-20)

    pump_peak_db = _band_peak_db(1.0, 5.0)
    ref_peak_db = _band_peak_db(5.0, 15.0)
    excess_db = pump_peak_db - ref_peak_db

    # Modulation depth: peak-to-trough swing in envelope (dB), as a sanity gate
    p95 = float(np.percentile(env, 95))
    p5 = float(np.percentile(env, 5))
    if p5 < 1e-9 or p95 < 1e-9:
        depth_db = 0.0
    else:
        depth_db = 20.0 * np.log10(p95 / p5)

    pumping = bool(excess_db >= 6.0 and depth_db >= 5.0)

    # Identify the actual pump rate within 1-5 Hz
    mask = (freqs >= 1.0) & (freqs < 5.0)
    pump_rate = float(freqs[mask][int(np.argmax(psd[mask]))]) if mask.any() else None

    return {
        "pumping_detected": pumping,
        "pump_rate_hz": round(pump_rate, 2) if pump_rate is not None else None,
        "modulation_depth_db": round(depth_db, 1),
        "lf_excess_db": round(excess_db, 1),
    }


def _crest_factor_db(signal: np.ndarray) -> float:
    """Peak-to-RMS ratio in dB. High = dynamic material, low = compressed."""
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    if rms < 1e-10:
        return 0.0
    return round(float(20 * np.log10(peak / rms)), 1)


def _stereo_metrics(data: np.ndarray) -> dict:
    """L/R balance, LR correlation, and M/S width from a (N, 2) array."""
    L = data[:, 0].astype(np.float64)
    R = data[:, 1].astype(np.float64)
    rms_L = np.sqrt(np.mean(L ** 2))
    rms_R = np.sqrt(np.mean(R ** 2))
    balance_db = round(float(20 * np.log10((rms_L + 1e-10) / (rms_R + 1e-10))), 1)
    if rms_L > 1e-10 and rms_R > 1e-10:
        correlation = round(float(np.corrcoef(L, R)[0, 1]), 3)
    else:
        correlation = 1.0
    M = L + R
    S = L - R
    rms_M = np.sqrt(np.mean(M ** 2))
    rms_S = np.sqrt(np.mean(S ** 2))
    ms_width = round(float(rms_S / (rms_M + 1e-10)), 3)
    return {
        "balance_db": balance_db,
        "lr_correlation": correlation,
        "ms_width_ratio": ms_width,
    }


def _transient_density(signal: np.ndarray, sr: int) -> float:
    """Onset events per second — higher = more transient, lower = sustained."""
    onset_env = librosa.onset.onset_strength(y=signal.astype(np.float32), sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration = len(signal) / sr
    if duration < 0.1:
        return 0.0
    return round(float(len(onset_frames) / duration), 2)


def _spectral_centroid_hz(signal: np.ndarray, sr: int) -> float:
    """Mean spectral centroid in Hz — tonal brightness indicator."""
    centroid = librosa.feature.spectral_centroid(y=signal.astype(np.float32), sr=sr)
    return round(float(np.mean(centroid)), 0)


def _transient_profile(signal: np.ndarray, sr: int) -> dict:
    """Per-onset attack prominence and decay time — indicates whether transient shaping is needed.

    prominence_db: attack peak (first 5ms after onset) vs sustain RMS (5-150ms), in dB.
      > 8 dB = strong attack already; 4-8 dB = moderate; < 4 dB = attack buried in sustain.
    decay_time_ms: ms from the peak sample until signal drops -20 dB below peak.
      Short = tight; long = boomy/washy.
    """
    onset_frames = librosa.onset.onset_detect(y=signal.astype(np.float32), sr=sr)
    onset_samples = librosa.frames_to_samples(onset_frames)

    attack_end = int(0.005 * sr)   # 5 ms
    sustain_end = int(0.150 * sr)  # 150 ms
    search_end = int(0.500 * sr)   # 500 ms decay search window

    prominences: list[float] = []
    decay_times: list[float] = []

    for onset in onset_samples:
        if onset + sustain_end >= len(signal):
            continue

        attack_win = np.abs(signal[onset:onset + attack_end])
        if len(attack_win) == 0:
            continue
        attack_peak = float(np.max(attack_win))
        if attack_peak < 1e-6:
            continue

        sustain_win = signal[onset + attack_end:onset + sustain_end]
        sustain_rms = float(np.sqrt(np.mean(sustain_win ** 2) + 1e-12))
        prominences.append(20.0 * np.log10(attack_peak / sustain_rms))

        # Envelope-based decay: smooth with 5ms RMS window to avoid zero-crossing artifacts
        local_win = signal[onset:min(onset + search_end, len(signal))]
        env_win = max(1, int(0.005 * sr))
        envelope = np.sqrt(np.convolve(local_win ** 2, np.ones(env_win) / env_win, mode="full")[:len(local_win)])
        peak_env_idx = int(np.argmax(envelope))
        peak_env_val = float(envelope[peak_env_idx])
        if peak_env_val < 1e-6:
            continue
        threshold = peak_env_val * 10.0 ** (-20.0 / 20.0)
        after_peak_env = envelope[peak_env_idx:]
        below = np.where(after_peak_env < threshold)[0]
        decay_ms = (float(below[0]) if len(below) > 0 else float(len(after_peak_env))) * 1000.0 / sr
        decay_times.append(decay_ms)

    if not prominences:
        return {
            "onset_count": 0,
            "transient_prominence_db": None,
            "transient_prominence_std_db": None,
            "decay_time_ms": None,
            "decay_time_std_ms": None,
        }

    return {
        "onset_count": len(prominences),
        "transient_prominence_db": round(float(np.mean(prominences)), 1),
        "transient_prominence_std_db": round(float(np.std(prominences)), 1),
        "decay_time_ms": round(float(np.mean(decay_times)), 1),
        "decay_time_std_ms": round(float(np.std(decay_times)), 1),
    }


def _stats_summary_text(stats: dict) -> str:
    lines = ["", "STATS SUMMARY", "-" * 54]
    loud = stats.get("loudness", {})
    lufs = loud.get("integrated_lufs", "n/a")
    lra = loud.get("loudness_range_lu", "n/a")
    crest = loud.get("crest_factor_db", "n/a")
    lines.append(f"  LUFS: {lufs} LUFS  |  LRA: {lra} LU  |  Crest factor: {crest} dB")
    td = stats.get("transient_density_per_sec", "n/a")
    sc = stats.get("spectral_centroid_hz", "n/a")
    lines.append(f"  Transient density: {td} /s  |  Spectral centroid: {sc} Hz")
    tp = stats.get("transient_profile", {})
    if tp and tp.get("onset_count", 0) > 0:
        prom = tp.get("transient_prominence_db", "n/a")
        prom_std = tp.get("transient_prominence_std_db", "n/a")
        decay = tp.get("decay_time_ms", "n/a")
        decay_std = tp.get("decay_time_std_ms", "n/a")
        lines.append(f"  Transient profile: prominence {prom} dB (±{prom_std})  |  decay {decay} ms (±{decay_std})")
    stereo = stats.get("stereo")
    if stereo:
        bal = stereo.get("balance_db", 0)
        side = "L>R" if bal > 0 else ("R>L" if bal < 0 else "balanced")
        corr = stereo.get("lr_correlation", "n/a")
        width = stereo.get("ms_width_ratio", "n/a")
        lines.append(f"  Stereo: balance {abs(bal):.1f} dB ({side})  |  LR corr {corr}  |  M/S width {width}")
    pump = stats.get("pumping")
    if pump and pump.get("pumping_detected"):
        rate = pump.get("pump_rate_hz", "n/a")
        depth = pump.get("modulation_depth_db", "n/a")
        lines.append(f"  [!] Pumping: {rate} Hz, depth {depth} dB — likely over-compressed")
    return "\n".join(lines)


def _save_outputs(
    stats: dict,
    text_spec: str,
    freq_response: list[dict],
    signal: np.ndarray,
    sr: int,
    stem_name: str,
    output_dir: Path,
) -> dict:
    """Save all analysis artifacts directly into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "spectrogram.png"
    txt_path = output_dir / "spectrogram.txt"
    json_path = output_dir / "analysis.json"

    _save_png_spectrogram(signal, sr, png_path, title=stem_name)
    txt_path.write_text(
        text_spec + "\n" + _freq_response_text(freq_response) + "\n" + _stats_summary_text(stats),
        encoding="utf-8",
    )

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

    print(f"Analyzing: {file_path.name}  ({duration:.1f}s, {channels}ch, {sr}Hz)", flush=True)

    print("  [1/5] Loudness metrics (LUFS, LRA, crest factor)...", flush=True)
    meter = pyln.Meter(sr)
    lufs_input = data if channels > 1 else mono
    try:
        integrated_lufs = float(meter.integrated_loudness(lufs_input))
    except Exception:
        integrated_lufs = -120.0
    lra = _lra(lufs_input, sr, meter)
    sample_peak = float(20 * np.log10(max(np.max(np.abs(mono)), 1e-10)))
    true_peak = _true_peak_dbfs(mono)
    crest_factor = _crest_factor_db(mono)
    noise_floor = round(_noise_floor_db(mono, sr), 1)
    dynamic_range = round(_dynamic_range_db(mono, sr), 1)
    recommended_gain = round(target_lufs - integrated_lufs, 1) if integrated_lufs > -100 else 0.0
    stereo = _stereo_metrics(data) if channels >= 2 else None

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
    frequency_bands_crest_db = {
        f"{name}_crest_db": _band_crest_db(mono, sr, lo, hi)
        for name, (lo, hi) in freq_bands.items()
    }

    print("  [2/5] Spectral analysis (mel spectrogram)...", flush=True)
    text_spec = _text_spectrogram(mono, sr)

    print("  [3/5] Frequency response (1/3-octave Welch PSD)...", flush=True)
    freq_response = _frequency_response(mono, sr)

    print("  [4/6] Hum detection...", flush=True)
    hum = _detect_hum(mono, sr)

    print("  [5/6] Onset detection + transient profile...", flush=True)
    transient_density = _transient_density(mono, sr)
    spec_centroid = _spectral_centroid_hz(mono, sr)
    transient_prof = _transient_profile(mono, sr)

    print("  [6/6] Pumping / over-compression detection...", flush=True)
    pumping = _detect_pumping(mono, sr)

    stats = {
        "file": str(file_path),
        "duration_sec": round(duration, 2),
        "sample_rate": sr,
        "channels": channels,
        "loudness": {
            "integrated_lufs": round(integrated_lufs, 1),
            "loudness_range_lu": lra,
            "true_peak_dbfs": round(true_peak, 1),
            "sample_peak_dbfs": round(sample_peak, 1),
            "dynamic_range_db": dynamic_range,
            "crest_factor_db": crest_factor,
        },
        "frequency_bands": frequency_bands,
        "frequency_bands_crest_db": frequency_bands_crest_db,
        "noise_floor_dbfs": noise_floor,
        "hum_detection": hum,
        "frequency_response": freq_response,
        "transient_density_per_sec": transient_density,
        "spectral_centroid_hz": spec_centroid,
        "transient_profile": transient_prof,
        "pumping": pumping,
        "recommended_gain_db": recommended_gain,
        "spectrogram_text": text_spec,
    }

    if stereo is not None:
        stats["stereo"] = stereo

    result = _save_outputs(stats, text_spec, freq_response, mono, sr, file_path.stem, output_dir)
    print(f"  Done -> {output_dir / 'analysis.json'}", flush=True)
    return result


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
