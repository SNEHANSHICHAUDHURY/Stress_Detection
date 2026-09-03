import collections
import numpy as np
from scipy import signal as sps


def pos_algorithm(rgb_signal, fps):
    rgb_signal = np.asarray(rgb_signal, dtype=np.float64)
    n = rgb_signal.shape[0]
    window_len = int(fps * 1.6)
    if n < window_len:
        return np.zeros(n)

    pulse = np.zeros(n)
    weights = np.zeros(n)

    for start in range(0, n - window_len + 1):
        window = rgb_signal[start:start + window_len]
        mean_rgb = np.mean(window, axis=0)
        mean_rgb[mean_rgb == 0] = 1e-6
        normalized = window / mean_rgb

        Xs = 3 * normalized[:, 0] - 2 * normalized[:, 1]
        Ys = 1.5 * normalized[:, 0] + normalized[:, 1] - 1.5 * normalized[:, 2]

        std_x = np.std(Xs)
        std_y = np.std(Ys)
        alpha = std_x / std_y if std_y > 1e-8 else 0

        S = Xs - alpha * Ys
        pulse[start:start + window_len] += (S - np.mean(S))
        weights[start:start + window_len] += 1.0

    return np.divide(pulse, weights, out=np.zeros_like(pulse), where=weights > 0)


def bandpass_filter(sig, fps, low_hz=0.7, high_hz=3.5, order=3):
    sig = np.asarray(sig, dtype=np.float64)
    n = len(sig)
    if n < 15:
        return sig - np.mean(sig)

    nyq = fps / 2.0
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    b, a = sps.butter(order, [low, high], btype="band")
    
    padlen = min(n - 1, 3 * max(len(a), len(b)))
    return sps.filtfilt(b, a, sig, padlen=padlen)

def estimate_hr_and_hrv(pulse_signal, fps):
    pulse_signal = np.asarray(pulse_signal, dtype=np.float64)
    n = len(pulse_signal)
    
    empty_result = {"hr_bpm": None, "sdnn_ms": None, "rmssd_ms": None}
    
    if n < int(fps * 3) or np.std(pulse_signal) < 1e-6:
        return empty_result

    # 1. Frequency-Domain HR via Welch's Periodogram (Far more stable than peak-picking)
    freqs, psd = sps.welch(pulse_signal - np.mean(pulse_signal), fs=fps, nperseg=min(n, int(fps * 6)))
    valid_mask = (freqs >= 0.75) & (freqs <= 3.0)  # 45 to 180 BPM
    
    if not np.any(valid_mask):
        return empty_result
        
    peak_freq = freqs[valid_mask][np.argmax(psd[valid_mask])]
    hr_bpm = peak_freq * 60.0

    # 2. Time-Domain Peak Detection with Adaptive Prominence
    win_len = min(9, n if n % 2 == 1 else n - 1)
    smoothed = sps.savgol_filter(pulse_signal, window_length=win_len, polyorder=min(2, win_len - 1)) if win_len > 3 else pulse_signal

    # Distance constrained by the dominant frequency found above (within 25% tolerance)
    expected_peak_dist = int(fps / peak_freq)
    min_dist = max(int(expected_peak_dist * 0.75), 1)
    
    peaks, _ = sps.find_peaks(smoothed, distance=min_dist, prominence=0.35 * np.std(smoothed))

    if len(peaks) < 3:
        return {"hr_bpm": round(float(hr_bpm), 1), "sdnn_ms": None, "rmssd_ms": None}

    ibi_ms = (np.diff(peaks) / fps) * 1000.0
    
    # 3. Outlier Filtering: Reject IBIs that deviate > 20% from median interval
    med_ibi = np.median(ibi_ms)
    clean_ibi = ibi_ms[(ibi_ms >= 0.80 * med_ibi) & (ibi_ms <= 1.20 * med_ibi)]

    if len(clean_ibi) < 2:
        return {"hr_bpm": round(float(hr_bpm), 1), "sdnn_ms": None, "rmssd_ms": None}

    sdnn_ms = float(np.std(clean_ibi))
    rmssd_ms = float(np.sqrt(np.mean(np.diff(clean_ibi) ** 2))) if len(clean_ibi) > 2 else sdnn_ms

    # Hard physiological sanity clamp (prevents impossible RMSSD readouts)
    if rmssd_ms > 150.0:
        rmssd_ms = None

    return {
        "hr_bpm": round(float(hr_bpm), 1),
        "sdnn_ms": round(sdnn_ms, 1) if sdnn_ms else None,
        "rmssd_ms": round(rmssd_ms, 1) if rmssd_ms else None,
    }

def eye_aspect_ratio(eye_pts):
    p1, p2, p3, p4, p5, p6 = eye_pts
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0


def mouth_aspect_ratio(mouth_pts):
    # mouth_pts: [left_corner, top_lip, right_corner, bottom_lip]
    p1, p2, p3, p4 = mouth_pts
    v = np.linalg.norm(p2 - p4)
    h = np.linalg.norm(p1 - p3)
    return v / h if h > 1e-6 else 0.0


def calculate_comprehensive_stress(rmssd_ms, blink_rate_per_min, brow_distance_ratio, head_jitter):
    """
    Computes a balanced 0-100 stress score across 4 multi-modal cues.
    """
    # 1. HRV Score (RMSSD: 20ms = high stress, 80ms = low stress)
    if rmssd_ms is not None:
        clipped_rmssd = max(20.0, min(80.0, rmssd_ms))
        s_hrv = (1.0 - (clipped_rmssd - 20.0) / 60.0) * 100.0
    else:
        s_hrv = 50.0  # Fallback neutral baseline

    # 2. Blink Rate Score (Normal: 12-20 bpm; High stress: >25 bpm)
    s_blink = np.clip((blink_rate_per_min - 14.0) / (32.0 - 14.0) * 100.0, 0, 100)

    # 3. Brow Furrow / Facial Tension (Lower distance ratio = furrowed brows = stress)
    # Baseline normal ratio is typically around 0.18 - 0.25
    s_brow = np.clip((0.22 - brow_distance_ratio) / (0.22 - 0.14) * 100.0, 0, 100)

    # 4. Head Jitter / Restlessness (Higher std of head pose rotation = stress)
    s_motion = np.clip((head_jitter - 0.5) / (3.5 - 0.5) * 100.0, 0, 100)

    total_stress = (0.40 * s_hrv) + (0.25 * s_blink) + (0.20 * s_brow) + (0.15 * s_motion)
    return round(float(total_stress), 1), {
        "hrv_subscore": round(float(s_hrv), 1),
        "blink_subscore": round(float(s_blink), 1),
        "brow_subscore": round(float(s_brow), 1),
        "motion_subscore": round(float(s_motion), 1),
    }
