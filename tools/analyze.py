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
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt, welch

sys.path.insert(0, str(Path(__file__).parent))
from _dsp import worst_channel_true_peak_dbfs  # noqa: E402

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"
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


def _headroom_verdict(sample_peak_db: float, true_peak_db: float) -> str:
    """DAW-style channel warning level based on worst of sample/true peak.

    [OK]   peak  < -6 dBFS         — comfortable headroom
    [WARN] -6 ≤ peak < -1 dBFS    — close to ceiling
    [CLIP] peak ≥ -1 dBFS or TP   — risk of inter-sample clipping
    """
    worst = max(sample_peak_db, true_peak_db)
    if worst >= -1.0:
        return "[CLIP]"
    if worst >= -6.0:
        return "[WARN]"
    return "[OK]"


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


_HUM_BLOCK_SEC = 0.5      # integer number of 50 and 60 Hz cycles: hum phase stays continuous
_HUM_SEGMENT_SEC = 4.0    # Welch segment -> 0.25 Hz bins
_HUM_TOLERANCE_HZ = 0.5   # line must lie within this distance of n * mains
_HUM_FLANK_HZ = (0.75, 3.0)
_HUM_MIN_HARMONICS = 2


def _detect_hum(signal: np.ndarray, sr: int, prominence_threshold_db: float = 12.0) -> dict:
    """Detect mains hum (50/60 Hz and harmonics) as narrow, persistent spectral lines.

    Method:
      1. Split into 0.5 s blocks, drop digital silence, and keep the quietest
         quarter of the blocks (at least 8 s): hum is constant, music is not.
      2. Median-averaged Welch PSD with 4 s segments (0.25 Hz bins). The median
         across segments only keeps lines present in most of the quiet audio.
      3. A harmonic n * mains counts when the strongest bin within +-0.5 Hz of it
         exceeds the strongest bin 0.75-3 Hz away on either side by
         prominence_threshold_db. A musical note 1 Hz or more away (e.g. G1 at
         49 Hz) dominates its own flank and is rejected.
      4. Hum requires at least two harmonics of the same mains series.
    Only notch filters are recommended; a high-pass would remove musical bass.
    """
    block = int(_HUM_BLOCK_SEC * sr)
    n_blocks = len(signal) // block
    need_blocks = int(np.ceil(2 * _HUM_SEGMENT_SEC / _HUM_BLOCK_SEC))
    blocks = signal[: n_blocks * block].reshape(n_blocks, block) if n_blocks else np.zeros((0, block))
    rms = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1)) if n_blocks else np.zeros(0)
    candidates = np.where(rms > 1e-6)[0]
    if len(candidates) < need_blocks:
        return {"hum_detected": False, "dominant_mains_hz": None, "harmonics": {},
                "recommendation": "Not analysed: less than 8 s of non-silent audio"}

    order = candidates[np.argsort(rms[candidates], kind="stable")]
    n_quiet = max(need_blocks, int(round(0.25 * len(candidates))))
    audio = blocks[np.sort(order[:n_quiet])].ravel()

    freqs, psd = welch(audio, fs=sr, nperseg=int(_HUM_SEGMENT_SEC * sr), average="median")
    psd_db = 10.0 * np.log10(psd + 1e-30)

    def _line_prominence(target_hz: float) -> float | None:
        near = np.abs(freqs - target_hz) <= _HUM_TOLERANCE_HZ
        if not near.any():
            return None
        idx = np.flatnonzero(near)[int(np.argmax(psd_db[near]))]
        dist = freqs - freqs[idx]
        flanks = []
        for side in (-1, 1):
            mask = (side * dist >= _HUM_FLANK_HZ[0]) & (side * dist <= _HUM_FLANK_HZ[1])
            if not mask.any():
                return None
            flanks.append(float(np.max(psd_db[mask])))
        return float(psd_db[idx] - max(flanks))

    detected: dict[str, list] = {}
    for mains_hz in (50, 60):
        harmonics = []
        for n in range(1, 7):
            target = mains_hz * n
            if target + _HUM_FLANK_HZ[1] >= sr / 2:
                break
            prominence = _line_prominence(float(target))
            if prominence is not None and prominence >= prominence_threshold_db:
                harmonics.append({"frequency_hz": target, "prominence_db": round(prominence, 1)})
        if len(harmonics) >= _HUM_MIN_HARMONICS:
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
        lines = ", ".join(f"{h['frequency_hz']} Hz" for h in detected[f"{dominant}hz"])
        rec = (f"Candidate narrow notch filters at {lines}; audition against the "
               f"unfiltered stem (narrow mains lines do not justify a broad low cut)")
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


def _lra(data: np.ndarray, sr: int, meter: pyln.Meter) -> float | None:
    """EBU R128 Loudness Range (LRA) in LU. None if not measurable (too short or silent)."""
    try:
        return _finite_round(meter.loudness_range(data))
    except Exception:
        return None


def _detect_pumping(signal: np.ndarray, sr: int) -> dict:
    """Detect compressor pumping artifact via low-frequency envelope modulation.

    Pumping = audible periodic dipping of the signal level caused by a release
    time tuned wrong (or hit too hard) on a compressor. Spectrally it shows up
    as energy in the signal's amplitude envelope around 1-5 Hz.

    Method:
      1. Compute a short-window RMS envelope (10 ms hop).
      2. High-pass the envelope above 0.5 Hz to remove the overall level.
      3. Look at envelope spectral peak in 1-5 Hz vs. the 5-15 Hz reference.
      4. Modulation depth (p95/p5) is computed on **active** frames only —
         silent gaps between hits or songs sections push p5 to ~0 and falsely
         hide pumping; gating to active frames keeps the depth measurement
         meaningful on intermittent material (kick mics, gtr w/ pauses, etc.).
      5. Pumping is detected if the 1-5 Hz peak exceeds the 5-15 Hz reference
         by >= 6 dB AND the modulation depth on active frames exceeds ~5 dB.
    """
    hop_ms = 10.0
    hop = max(1, int(sr * hop_ms / 1000.0))
    n_frames = len(signal) // hop
    if n_frames < 200:
        return {"pumping_detected": False, "pump_rate_hz": None, "modulation_depth_db": None,
                "lf_excess_db": None, "active_frame_ratio": None,
                "note": "signal too short for pumping analysis"}

    env = np.array([
        np.sqrt(np.mean(signal[i * hop:(i + 1) * hop] ** 2) + 1e-20)
        for i in range(n_frames)
    ])
    env_sr = sr / hop  # ~100 Hz

    # Remove DC and very slow drift (< 0.5 Hz). Rate detection uses this.
    sos_hp = butter(2, 0.5 / (env_sr / 2.0), btype="high", output="sos")
    env_hp = sosfilt(sos_hp, env)

    # Spectrum of the envelope (full timeline — pumping rate detection wants
    # the entire periodic signature, including the silent dips).
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

    # Modulation depth: peak-to-trough swing on ACTIVE frames only.
    # An "active" frame is one whose RMS is above -40 dBFS — i.e. the player
    # is actually playing. This is the gate that fixes the silent-gap bug:
    # without it, p5 dives to ~0 on intermittent material and depth_db = 0.
    active_threshold_lin = 10.0 ** (-40.0 / 20.0)
    active_mask = env > active_threshold_lin
    n_active = int(np.sum(active_mask))
    active_ratio = n_active / max(n_frames, 1)

    if n_active < 50:
        depth_db = 0.0
    else:
        active_env = env[active_mask]
        p95 = float(np.percentile(active_env, 95))
        p5 = float(np.percentile(active_env, 5))
        depth_db = 20.0 * np.log10(p95 / max(p5, 1e-12)) if p5 > 1e-9 else 0.0

    pumping = bool(excess_db >= 6.0 and depth_db >= 5.0)

    # Identify the actual pump rate within 1-5 Hz
    mask = (freqs >= 1.0) & (freqs < 5.0)
    pump_rate = float(freqs[mask][int(np.argmax(psd[mask]))]) if mask.any() else None

    return {
        "pumping_detected": pumping,
        "pump_rate_hz": round(pump_rate, 2) if pump_rate is not None else None,
        "modulation_depth_db": round(depth_db, 1),
        "lf_excess_db": round(excess_db, 1),
        "active_frame_ratio": round(active_ratio, 3),
    }


def _crest_factor_db(signal: np.ndarray) -> float:
    """Peak-to-RMS ratio in dB. High = dynamic material, low = compressed.

    For multichannel (N, C) input, peak and RMS come from the same channel:
    the one with the highest sample peak. A mono downmix would understate a
    hard-panned peak by 6 dB and mix RMS from different channels.
    """
    if signal.ndim == 2:
        signal = signal[:, int(np.argmax(np.max(np.abs(signal), axis=0)))]
    signal = signal.astype(np.float64)
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    if rms < 1e-10:
        return 0.0
    return round(float(20 * np.log10(peak / rms)), 1)


# Channels with RMS below this (-90 dBFS) are treated as silent for stereo metrics.
_STEREO_SILENT_RMS = 10.0 ** (-90.0 / 20.0)


def _stereo_metrics(data: np.ndarray) -> dict:
    """L/R balance, LR correlation, and M/S width from a (N, 2) array.

    balance_db and lr_correlation are None when either channel is silent
    (below -90 dBFS RMS): a level ratio against silence and a correlation
    with a constant are undefined.
    """
    L = data[:, 0].astype(np.float64)
    R = data[:, 1].astype(np.float64)
    rms_L = np.sqrt(np.mean(L ** 2))
    rms_R = np.sqrt(np.mean(R ** 2))
    if rms_L > _STEREO_SILENT_RMS and rms_R > _STEREO_SILENT_RMS:
        balance_db = round(float(20 * np.log10(rms_L / rms_R)), 1)
        correlation = round(float(np.corrcoef(L, R)[0, 1]), 3)
    else:
        balance_db = None
        correlation = None
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


def _transient_density(signal: np.ndarray, sr: int, onset_env: np.ndarray | None = None) -> float:
    """Onset events per second — higher = more transient, lower = sustained.

    `onset_env` may be passed in to avoid recomputing librosa's onset_strength
    when the caller has already computed it (e.g. for spectral_flux).
    """
    if onset_env is None:
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


def _transient_profile(signal: np.ndarray, sr: int, onset_env: np.ndarray | None = None) -> dict:
    """Per-onset attack prominence and decay time — indicates whether transient shaping is needed.

    prominence_db: attack peak (first 5ms after onset) vs sustain RMS (5-150ms), in dB.
      > 8 dB = strong attack already; 4-8 dB = moderate; < 4 dB = attack buried in sustain.
    decay_time_ms: ms from the peak sample until signal drops -20 dB below peak.
      Short = tight; long = boomy/washy.
    """
    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=signal.astype(np.float32), sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
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


def _onsets_sec(signal: np.ndarray, sr: int, onset_env: np.ndarray | None = None) -> list[float]:
    """Onset times in seconds — same detector as _transient_density, exposed as raw list."""
    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=signal.astype(np.float32), sr=sr)
    frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    times = librosa.frames_to_time(frames, sr=sr)
    return [round(float(t), 3) for t in times]


def _tempo_bpm(signal: np.ndarray, sr: int, onset_env: np.ndarray | None = None) -> float | None:
    """Estimated tempo in BPM. Returns None for short signals or unstable estimates."""
    if len(signal) / sr < 4.0:
        return None
    try:
        if onset_env is None:
            onset_env = librosa.onset.onset_strength(y=signal.astype(np.float32), sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        if not np.isfinite(bpm) or bpm < 30.0 or bpm > 300.0:
            return None
        return round(bpm, 1)
    except Exception:
        return None


def _rms_envelope_db_per_sec(signal: np.ndarray, sr: int) -> list[float]:
    """RMS envelope downsampled to 1 Hz, expressed in dBFS.

    Length matches int(duration_sec) — one sample per second. Useful for
    spotting section-level dynamics (intro / verse / chorus loudness shifts).
    """
    duration = len(signal) / sr
    n_seconds = max(1, int(duration))
    samples_per_sec = sr
    out: list[float] = []
    for i in range(n_seconds):
        chunk = signal[i * samples_per_sec : (i + 1) * samples_per_sec]
        if len(chunk) == 0:
            break
        rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
        out.append(round(20.0 * np.log10(max(rms, 1e-10)), 1))
    return out


def _lufs_short_term(data: np.ndarray, sr: int) -> list[float]:
    """BS.1770 short-term LUFS (3 s window) sampled every 1 s.

    Returns the loudness curve over time — complements integrated_lufs (single
    number) by showing where in the track the loudness sits. The 1 s step
    keeps cost bounded on long stems (long takes used to spawn 4000+ meter
    calls at the previous 0.1 s spacing).
    """
    duration = len(data) / sr
    if duration < 3.0:
        return []

    meter = pyln.Meter(sr, block_size=3.0)
    step_samples = sr  # 1 s step
    block_samples = int(3.0 * sr)
    out: list[float] = []
    pos = 0
    while pos + block_samples <= len(data):
        block = data[pos : pos + block_samples]
        try:
            lufs = float(meter.integrated_loudness(block))
        except Exception:
            lufs = -120.0
        if not np.isfinite(lufs):
            lufs = -120.0
        out.append(round(lufs, 1))
        pos += step_samples
    return out


def _spectral_flux_per_sec(signal: np.ndarray, sr: int, onset_env: np.ndarray | None = None) -> list[float]:
    """Spectral flux (onset strength) downsampled to 1 Hz.

    Useful for section detection — flux peaks at intro→verse→chorus boundaries
    where the spectral content changes substantially. Reuses `onset_env` if
    the caller already computed it.
    """
    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=signal.astype(np.float32), sr=sr)
    # librosa default hop = 512 → frames-per-sec ≈ sr/512
    frames_per_sec = sr / 512.0
    duration = len(signal) / sr
    n_seconds = max(1, int(duration))
    out: list[float] = []
    for i in range(n_seconds):
        lo = int(i * frames_per_sec)
        hi = int((i + 1) * frames_per_sec)
        if lo >= len(onset_env):
            break
        out.append(round(float(np.mean(onset_env[lo:hi])), 3))
    return out


# Krumhansl-Schmuckler key profiles (major and minor pitch-class weights)
_KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimated_key(signal: np.ndarray, sr: int) -> dict:
    """Krumhansl-Schmuckler key estimation from a chroma profile.

    Correlates the mean chroma vector against all 24 rotated major/minor
    Krumhansl profiles. The best-matching rotation is the key; confidence is
    the Pearson correlation coefficient (both vectors mean-subtracted, -1..1).
    Plain cosine similarity of non-negative vectors is near 1 even for white
    noise. Tonal stems give high values; noise and drums stay low.
    """
    if len(signal) / sr < 4.0:
        return {"key": None, "mode": None, "confidence": 0.0}

    try:
        # chroma_stft is 5-10× faster than chroma_cqt on long stems and is
        # accurate enough for rock-style tonal estimation.
        chroma = librosa.feature.chroma_stft(y=signal.astype(np.float32), sr=sr)
    except Exception:
        return {"key": None, "mode": None, "confidence": 0.0}

    profile = chroma.mean(axis=1)
    if profile.sum() < 1e-6:
        return {"key": None, "mode": None, "confidence": 0.0}

    def _unit(v: np.ndarray) -> np.ndarray:
        v = v - np.mean(v)
        return v / (np.linalg.norm(v) + 1e-12)

    profile = _unit(profile)
    maj_n = _unit(_KRUMHANSL_MAJOR)
    min_n = _unit(_KRUMHANSL_MINOR)

    best_score = -1.0
    best_key = 0
    best_mode = "major"
    for shift in range(12):
        rolled = np.roll(profile, -shift)
        for mode_name, ref in (("major", maj_n), ("minor", min_n)):
            score = float(np.dot(rolled, ref))
            if score > best_score:
                best_score = score
                best_key = shift
                best_mode = mode_name

    return {
        "key": _PITCH_NAMES[best_key],
        "mode": best_mode,
        "confidence": round(best_score, 2),
    }


_VOCAL_METRICS_NONVOCAL_HINTS = (
    "KICK", "SN ", " SN", "SNARE", "OH ", " OH", "OVERHEAD", "TOM ", " TOM",
    "HIHAT", "HI-HAT", "CYMBAL", "CRASH", "RIDE", "ROOM",
    "BASS", "DI ", " DI", "AMP", "FENDER", "ORANGE", "MARSHALL", "DI CLEAN",
    "GTR ", " GTR", "GUITAR", "STRING", "PIANO", "KEYS",
)


def _looks_vocal(file_name: str) -> bool:
    """Heuristic: should `_vocal_metrics` run its expensive pyin pitch detection
    on this stem? Returns True if the filename has a vocal-ish marker, False if
    it has a clear non-vocal marker. Default (no marker either way) → True
    (safer for unknown content).

    Pitch detection via `librosa.pyin` + viterbi is ~15-25s per stem (the
    single biggest hotspot in analyze.py). Skipping it on drum / bass / guitar
    stems drops analyze.py from ~30s to ~5s per non-vocal stem.
    """
    u = file_name.upper()
    if any(k in u for k in ("VOX", "VOC", "VOCAL", "LEAD", "HARMONY",
                            "BG VOX", "WHISPER", "AD-LIB", "ADLIB")):
        return True
    if any(k in u for k in _VOCAL_METRICS_NONVOCAL_HINTS):
        return False
    return True


_PYIN_HOP = 512                 # pyin frame hop (librosa default for frame_length 2048)
_NOTE_SPLIT_CENTS = 80.0        # frame-to-frame jump that starts a new note
_PITCH_SMOOTH_SEC = 0.2         # ~one vibrato cycle; removes vibrato and tracker jitter
_VIBRATO_MIN_NOTE_SEC = 0.5
_PLOSIVE_WINDOW_SEC = 0.02
_PLOSIVE_REL_DB = -12.0         # LF burst within 12 dB of the stem's loud level
_PLOSIVE_REFRACTORY_SEC = 0.15


def _note_segments(f0: np.ndarray, min_frames: int = 1) -> list[np.ndarray]:
    """Cents (re A440) of each note: voiced runs split at large pitch jumps."""
    voiced = np.isfinite(f0) & (f0 > 0)
    edges = np.diff(np.concatenate(([0], voiced.astype(np.int8), [0])))
    notes = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        cents = 1200.0 * np.log2(f0[start:end] / 440.0)
        cuts = np.flatnonzero(np.abs(np.diff(cents)) > _NOTE_SPLIT_CENTS) + 1
        notes.extend(n for n in np.split(cents, cuts) if len(n) >= min_frames)
    return notes


def _pitch_stats(f0: np.ndarray, frame_rate: float) -> dict:
    """Intonation of voiced frames relative to per-note semitone targets.

    1. Global tuning offset = circular mean of the cents (re A440) modulo 100;
       it is removed first, so a consistently sharp/flat but internally
       consistent performance is not penalised (the offset is reported).
    2. Notes = voiced runs split at frame-to-frame jumps > 80 cents. Each note
       is smoothed over 200 ms (removes vibrato and tracker jitter) and its
       target is the semitone nearest to the note's median pitch.
    3. Deviation = smoothed pitch - note target; not wrapped, so drifts and
       scoops larger than 50 cents count in full.

    cents_std is the RMS deviation in cents (field name kept for
    compatibility). Note-centre errors alone cannot exceed 50 cents (a note
    50 cents off is closer to the neighbouring semitone), so values near 29
    are what uniformly random note centres would give; well-intoned singing
    is typically well below 15.
    """
    voiced = np.isfinite(f0) & (f0 > 0)
    if not voiced.any():
        return {}
    phase = np.exp(2j * np.pi * (1200.0 * np.log2(f0[voiced] / 440.0)) / 100.0)
    offset = float(np.angle(np.mean(phase)) * 100.0 / (2.0 * np.pi))
    window = max(1, int(round(_PITCH_SMOOTH_SEC * frame_rate)))
    deviations = []
    notes = _note_segments(f0)
    for cents in notes:
        cents = cents - offset
        smooth = uniform_filter1d(cents, min(window, len(cents)), mode="nearest")
        deviations.append(smooth - np.round(np.median(cents) / 100.0) * 100.0)
    dev = np.abs(np.concatenate(deviations))
    return {
        "cents_std": round(float(np.sqrt(np.mean(dev ** 2))), 1),
        "cents_mad": round(float(np.median(dev)), 1),
        "fraction_over_25_cents": round(float(np.mean(dev > 25.0)), 3),
        "tuning_offset_cents": round(offset, 1),
        "note_count": len(notes),
    }


def _vibrato_stats(f0: np.ndarray, frame_rate: float) -> dict:
    """Vibrato rate and semi-extent from notes lasting at least 0.5 s.

    Each note's cents curve is linearly detrended, Hann-windowed and
    zero-padded to 0.05 Hz resolution; the strongest 4-7 Hz component gives
    the rate (Hz) and its sinusoidal amplitude, the semi-extent (+-cents).
    Medians across notes are reported. A semi-extent below ~10 cents means
    no meaningful vibrato.
    """
    rates, extents = [], []
    for cents in _note_segments(f0, int(np.ceil(_VIBRATO_MIN_NOTE_SEC * frame_rate))):
        x = np.arange(len(cents))
        cents = cents - np.polyval(np.polyfit(x, cents, 1), x)
        win = np.hanning(len(cents))
        n_fft = max(len(cents), int(frame_rate / 0.05))
        spec = np.abs(np.fft.rfft(cents * win, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / frame_rate)
        band = np.flatnonzero((freqs >= 4.0) & (freqs <= 7.0))
        if len(band) == 0:
            continue
        k = band[int(np.argmax(spec[band]))]
        rates.append(float(freqs[k]))
        extents.append(float(2.0 * spec[k] / np.sum(win)))
    if not rates:
        return {}
    return {
        "rate_hz": round(float(np.median(rates)), 2),
        "extent_cents": round(float(np.median(extents)), 1),
        "notes_analyzed": len(rates),
    }


def _plosive_stats(signal: np.ndarray, sr: int) -> dict:
    """Low-frequency (< 100 Hz) bursts, counted on a smoothed envelope.

    Envelope = 20 ms moving RMS of the 100 Hz low-passed signal. An event
    starts when it rises above a level 12 dB below the stem's loud level
    (95th percentile of the full-band 20 ms RMS over frames above -60 dBFS);
    rises within 150 ms of the previous event are merged into it.
    """
    win = max(1, int(_PLOSIVE_WINDOW_SEC * sr))
    kernel = np.ones(win) / win
    lf = sosfilt(butter(4, 100.0, btype="low", fs=sr, output="sos"), signal.astype(np.float64))
    lf_env = np.sqrt(np.maximum(np.convolve(lf ** 2, kernel, mode="same"), 0.0))
    full_env = np.sqrt(np.maximum(np.convolve(signal.astype(np.float64) ** 2, kernel, mode="same"), 0.0))
    minutes = len(signal) / sr / 60.0
    peak = float(np.max(np.abs(lf))) if len(lf) else 0.0
    peak_db = round(20.0 * np.log10(peak), 1) if peak > 1e-6 else None
    active = full_env > 10.0 ** (-60.0 / 20.0)
    if not active.any() or minutes <= 0:
        return {"events_count": 0, "events_per_minute": 0.0, "peak_db": peak_db}
    threshold = float(np.percentile(full_env[active], 95)) * 10.0 ** (_PLOSIVE_REL_DB / 20.0)
    above = np.concatenate(([0], (lf_env > threshold).astype(np.int8)))
    refractory = int(_PLOSIVE_REFRACTORY_SEC * sr)
    events, last = 0, -refractory
    for i in np.flatnonzero(np.diff(above) > 0):
        if i - last >= refractory:
            events += 1
        last = i
    return {
        "events_count": events,
        "events_per_minute": round(events / minutes, 2),
        "peak_db": peak_db,
    }


def _vocal_metrics(signal: np.ndarray, sr: int, run_pitch: bool = True) -> dict:
    """Vocal-specific metrics: sibilance, plosive, pitch stats, vibrato, breath.

    None of these are vocal-only — they're meaningful for any monophonic
    tonal source — but they're primarily useful for deciding vocal-chain
    parameters (de-esser threshold, pitch-correct strength, etc.).

    `run_pitch=False` skips the librosa.pyin + viterbi block (the slow part).
    Sibilance / plosive / breath still compute — they're cheap band-filter +
    envelope work and are sometimes informative on non-vocal sources too.

    plosive.events_count is a total over the stem; plosive.events_per_minute
    normalises it by duration (see _plosive_stats). pitch.cents_std is an RMS
    deviation from per-note targets (see _pitch_stats); vibrato.extent_cents
    is a semi-extent in cents (see _vibrato_stats).
    """
    out: dict = {
        "sibilance": {},
        "plosive": {},
        "pitch": {},
        "vibrato": {},
        "breath": {},
    }

    # Sibilance: 5-8 kHz band peak + per-second event density
    sib_low, sib_high = 5500.0, min(8500.0, sr / 2 - 1)
    if sib_high > sib_low:
        sib_sos = butter(4, [sib_low, sib_high], btype="band", fs=sr, output="sos")
        sib_band = sosfilt(sib_sos, signal)
        sib_peak = float(np.max(np.abs(sib_band)))
        out["sibilance"]["peak_db"] = round(20.0 * np.log10(max(sib_peak, 1e-10)), 1)
        # event density: count peaks > -25 dBFS per second
        threshold_lin = 10 ** (-25 / 20)
        win = max(1, int(0.005 * sr))  # 5 ms window
        env = np.sqrt(np.convolve(sib_band ** 2, np.ones(win) / win, mode="same"))
        crossings = np.diff((env > threshold_lin).astype(int))
        n_events = int(np.sum(crossings > 0))
        duration = len(signal) / sr
        out["sibilance"]["density_per_sec"] = round(n_events / max(duration, 0.01), 2)

    out["plosive"] = _plosive_stats(signal, sr)

    # Pitch tracking — librosa.pyin (probabilistic YIN). The single biggest
    # hotspot in analyze.py (~15-25s on a 7-min stem due to viterbi). Skipped
    # entirely when run_pitch=False (typical for non-vocal stems).
    duration_sec = len(signal) / sr
    if run_pitch and duration_sec >= 2.0:
        try:
            f0, _, _ = librosa.pyin(
                signal.astype(np.float32), fmin=80, fmax=600, sr=sr, hop_length=_PYIN_HOP,
            )
            voiced_f0 = f0[~np.isnan(f0)]
            if len(voiced_f0) > 0:
                frame_rate = sr / _PYIN_HOP
                out["pitch"]["mean_hz"] = round(float(np.mean(voiced_f0)), 1)
                out["pitch"]["median_hz"] = round(float(np.median(voiced_f0)), 1)
                out["pitch"].update(_pitch_stats(f0, frame_rate))
                out["pitch"]["voiced_ratio"] = round(float(len(voiced_f0)) / len(f0), 2)
                out["vibrato"] = _vibrato_stats(f0, frame_rate)
            else:
                out["pitch"]["mean_hz"] = None
        except Exception:
            out["pitch"]["mean_hz"] = None
    elif not run_pitch:
        out["pitch"]["skipped_reason"] = "non-vocal stem (filename heuristic)"

    # Breath / silence ratio (frames below -45 dBFS)
    frame_n = max(1, int(0.05 * sr))  # 50 ms frames
    n_frames = len(signal) // frame_n
    silent = 0
    for i in range(n_frames):
        chunk = signal[i * frame_n:(i + 1) * frame_n]
        rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
        if 20 * np.log10(max(rms, 1e-10)) < -45:
            silent += 1
    out["breath"]["silence_ratio"] = round(silent / max(n_frames, 1), 2)

    return out


def _stats_summary_text(stats: dict) -> str:
    lines = ["", "STATS SUMMARY", "-" * 54]
    loud = stats.get("loudness", {})
    lufs = loud.get("integrated_lufs")
    lra = loud.get("loudness_range_lu")
    crest = loud.get("crest_factor_db")
    lufs, lra, crest = ("n/a" if v is None else v for v in (lufs, lra, crest))
    lines.append(f"  LUFS: {lufs} LUFS  |  LRA: {lra} LU  |  Crest factor: {crest} dB")
    sample_peak = loud.get("sample_peak_dbfs")
    true_peak = loud.get("true_peak_dbfs")
    verdict = loud.get("headroom_verdict")
    if sample_peak is not None and true_peak is not None and verdict is not None:
        lines.append(f"  Headroom: {verdict} sample {sample_peak} dBFS / true {true_peak} dBTP  (target < -6 dBFS)")
    td = stats.get("transient_density_per_sec", "n/a")
    sc = stats.get("spectral_centroid_hz", "n/a")
    lines.append(f"  Transient density: {td} /s  |  Spectral centroid: {sc} Hz")
    tempo = stats.get("tempo_bpm")
    key = stats.get("estimated_key", {}) or {}
    key_str = f"{key.get('key')} {key.get('mode')}" if key.get("key") else "n/a"
    key_conf = key.get("confidence")
    tempo_str = f"{tempo} BPM" if tempo is not None else "n/a"
    if key_conf is not None and key.get("key"):
        lines.append(f"  Tempo: {tempo_str}  |  Estimated key: {key_str} (conf {key_conf})")
    else:
        lines.append(f"  Tempo: {tempo_str}  |  Estimated key: {key_str}")
    tp = stats.get("transient_profile", {})
    if tp and tp.get("onset_count", 0) > 0:
        prom = tp.get("transient_prominence_db", "n/a")
        prom_std = tp.get("transient_prominence_std_db", "n/a")
        decay = tp.get("decay_time_ms", "n/a")
        decay_std = tp.get("decay_time_std_ms", "n/a")
        lines.append(f"  Transient profile: prominence {prom} dB (±{prom_std})  |  decay {decay} ms (±{decay_std})")
    stereo = stats.get("stereo")
    if stereo:
        bal = stereo.get("balance_db")
        corr = stereo.get("lr_correlation")
        width = stereo.get("ms_width_ratio", "n/a")
        if bal is None:
            bal_str, corr = "n/a (one channel silent)", "n/a"
        else:
            side = "L>R" if bal > 0 else ("R>L" if bal < 0 else "balanced")
            bal_str = f"{abs(bal):.1f} dB ({side})"
        lines.append(f"  Stereo: balance {bal_str}  |  LR corr {corr}  |  M/S width {width}")
    pump = stats.get("pumping")
    if pump and pump.get("pumping_detected"):
        rate = pump.get("pump_rate_hz", "n/a")
        depth = pump.get("modulation_depth_db", "n/a")
        lines.append(f"  [!] Pumping: {rate} Hz, depth {depth} dB — candidate: compressor artifact or musical pulse; compare before/after by ear")
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
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def _json_safe(value):
    """Recursively replace non-finite floats (NaN, +-inf) with None for strict JSON."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _finite_round(value: float | None, ndigits: int = 1) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), ndigits)


_CACHE_VERSION = 3  # bump when the analysis schema changes


def _cache_signature(file_path: Path, target_lufs: float, force_vocal_metrics: bool) -> dict:
    """Stable signature describing the inputs to an analyze() call.

    Cache key = (input WAV mtime + size + args). If a previous analysis.json has
    a matching `_cache` block, we can skip the work entirely. Bypass with
    `use_cache=False` or by deleting analysis.json.
    """
    stat = file_path.stat()
    return {
        "version": _CACHE_VERSION,
        "input_mtime_ns": int(stat.st_mtime_ns),
        "input_size_bytes": int(stat.st_size),
        "target_lufs": float(target_lufs),
        "force_vocal_metrics": bool(force_vocal_metrics),
    }


def _try_use_cached_analysis(
    output_dir: Path,
    signature: dict,
) -> dict | None:
    """If analysis.json + spectrogram.{txt,png} exist with a matching cache key,
    return the cached dict. Else return None.
    """
    analysis_path = output_dir / "analysis.json"
    if not analysis_path.exists():
        return None
    try:
        cached = json.loads(analysis_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if cached.get("_cache") != signature:
        return None
    # Side artifacts must also exist for the cache to count as valid
    spec_txt = output_dir / "spectrogram.txt"
    spec_png = output_dir / "spectrogram.png"
    if not (spec_txt.exists() and spec_png.exists()):
        return None
    return cached


def analyze(
    file_path: Path,
    output_dir: Path,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    force_vocal_metrics: bool = False,
    use_cache: bool = True,
) -> dict:
    if use_cache and file_path.exists() and output_dir.exists():
        sig = _cache_signature(file_path, target_lufs, force_vocal_metrics)
        cached = _try_use_cached_analysis(output_dir, sig)
        if cached is not None:
            print(f"Analyzing: {file_path.name}  -> CACHE HIT  ({output_dir/'analysis.json'})", flush=True)
            return cached

    data, sr = sf.read(str(file_path), always_2d=True, dtype="float32")
    if not np.isfinite(data).all():
        raise ValueError(f"{file_path}: audio contains non-finite samples (NaN/Inf); "
                         "repair or re-export the file before analysis")
    mono = data.mean(axis=1).astype(np.float32)
    channels = data.shape[1]
    duration = len(mono) / sr

    print(f"Analyzing: {file_path.name}  ({duration:.1f}s, {channels}ch, {sr}Hz)", flush=True)

    print("  [1/9] Loudness metrics (LUFS, LRA, crest factor)...", flush=True)
    meter = pyln.Meter(sr)
    lufs_input = data if channels > 1 else mono
    # Silent or too-short input: pyloudnorm returns -inf or raises -> None.
    try:
        integrated_lufs = _finite_round(meter.integrated_loudness(lufs_input))
    except Exception:
        integrated_lufs = None
    lra = _lra(lufs_input, sr, meter)
    # Worst individual channel, not the monosum — a hard-panned full-scale
    # peak is ~6 dB quieter after .mean() and would under-report.
    sample_peak = float(20 * np.log10(max(float(np.max(np.abs(data))), 1e-10)))
    true_peak = worst_channel_true_peak_dbfs(data)
    crest_factor = _crest_factor_db(data)
    noise_floor = round(_noise_floor_db(mono, sr), 1)
    dynamic_range = round(_dynamic_range_db(mono, sr), 1)
    recommended_gain = round(target_lufs - integrated_lufs, 1) if integrated_lufs is not None else None
    stereo = _stereo_metrics(data) if channels >= 2 else None

    # Band RMS / band crest are measured on the mono downmix (L+R)/2: a
    # hard-panned source reads 6 dB lower and anti-phase content cancels.
    # Peaks and the overall crest factor above use individual channels.
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

    print("  [2/9] Spectral analysis (mel spectrogram)...", flush=True)
    text_spec = _text_spectrogram(mono, sr)

    print("  [3/9] Frequency response (1/3-octave Welch PSD)...", flush=True)
    freq_response = _frequency_response(mono, sr)

    print("  [4/9] Hum detection...", flush=True)
    hum = _detect_hum(mono, sr)

    print("  [5/9] Onset detection + transient profile...", flush=True)
    # Compute onset_env once and share it across transient_density + spectral_flux
    onset_env_shared = librosa.onset.onset_strength(y=mono.astype(np.float32), sr=sr)
    transient_density = _transient_density(mono, sr, onset_env=onset_env_shared)
    spec_centroid = _spectral_centroid_hz(mono, sr)
    transient_prof = _transient_profile(mono, sr, onset_env=onset_env_shared)
    onsets = _onsets_sec(mono, sr, onset_env=onset_env_shared)

    print("  [6/9] Pumping / over-compression detection...", flush=True)
    pumping = _detect_pumping(mono, sr)

    print("  [7/9] Tempo + key estimation...", flush=True)
    tempo_bpm = _tempo_bpm(mono, sr, onset_env=onset_env_shared)
    estimated_key = _estimated_key(mono, sr)

    print("  [8/9] Envelopes (RMS dB, LUFS short-term, spectral flux)...", flush=True)
    rms_env = _rms_envelope_db_per_sec(mono, sr)
    lufs_st = _lufs_short_term(lufs_input, sr)
    spec_flux = _spectral_flux_per_sec(mono, sr, onset_env=onset_env_shared)

    # Use parent dir name too — assembled.wav files live under per-track dirs like
    # `output/<session>/tracks/KICK IN.05/assembled.wav`, and the meaningful identifier
    # is in the parent dir name, not the basename.
    stem_hint = f"{file_path.parent.name} {file_path.name}"
    run_pitch = force_vocal_metrics or _looks_vocal(stem_hint)
    pitch_note = "" if run_pitch else " — pitch skipped (non-vocal filename heuristic)"
    print(f"  [9/9] Vocal metrics (sibilance, plosive, pitch, vibrato, breath){pitch_note}...", flush=True)
    vocal = _vocal_metrics(mono, sr, run_pitch=run_pitch)

    stats = {
        "file": str(file_path),
        "duration_sec": round(duration, 2),
        "sample_rate": sr,
        "channels": channels,
        "loudness": {
            "integrated_lufs": integrated_lufs,
            "loudness_range_lu": lra,
            "true_peak_dbfs": round(true_peak, 1),
            "sample_peak_dbfs": round(sample_peak, 1),
            "dynamic_range_db": dynamic_range,
            "crest_factor_db": crest_factor,
            "headroom_verdict": _headroom_verdict(sample_peak, true_peak),
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
        "tempo_bpm": tempo_bpm,
        "estimated_key": estimated_key,
        "onsets_sec": onsets,
        "envelopes": {
            "rms_db_per_second": rms_env,
            "lufs_short_term": lufs_st,
            "spectral_flux_per_second": spec_flux,
        },
        "vocal": vocal,
        "recommended_gain_db": recommended_gain,
    }

    if stereo is not None:
        stats["stereo"] = stereo

    # Cache key so the next analyze() call can short-circuit if WAV is unchanged
    stats["_cache"] = _cache_signature(file_path, target_lufs, force_vocal_metrics)

    stats = _json_safe(stats)
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
    parser.add_argument(
        "--force-vocal-metrics",
        action="store_true",
        help="Run librosa.pyin pitch detection even if the filename looks non-vocal. "
             "Default heuristic skips pyin on drum/bass/guitar stems (~20s saving).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip the mtime-based cache and force re-analysis even if "
             "analysis.json + spectrogram.{txt,png} already exist with a "
             "matching cache key.",
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(json.dumps({"error": f"File not found: {args.file}"}), file=sys.stderr)
        sys.exit(1)

    try:
        result = analyze(
            args.file,
            output_dir=args.output_dir,
            target_lufs=args.target_lufs,
            force_vocal_metrics=args.force_vocal_metrics,
            use_cache=not args.no_cache,
        )
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
