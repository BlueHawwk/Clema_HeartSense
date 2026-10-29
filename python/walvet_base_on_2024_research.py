"""
live_wavelet.py — wavelet-based motion-artifact reduction for Clema HR prototype.

Input from firmware:
t,cx,cy,cz,dx,dy,dz,bx,by,bz

Default mapping (EDIT IF NEEDED):
C = finger, D = front chest, B = silicone channel.
Target for HR = front chest (D); references = finger (C) and silicone channel (B).

Method:
1. Use the chest sensor's 3 axes as the target signal.
2. Use stationary wavelet transform (SWT) to decompose each target axis.
3. Estimate a conservative noise threshold using robust detail-coefficient
   statistics from target and reference sensors.
4. Reconstruct denoised chest axes.
5. Extract a 5–25 Hz vibration band, compute its short-time energy envelope,
   then estimate HR from envelope autocorrelation and compare with envelope FFT.
6. Show target raw/clean envelope and candidate HR.

This is a research prototype, not a medical device. Reference sensors can
contain cardiac vibration too; excessive denoising can remove useful signal.
Validate against Polar H10 during rest, walking, bending and running.

Install:
    pip install numpy scipy pyserial pyqtgraph PyQt6 PyWavelets
"""

import sys
import time
import datetime
from collections import deque

import numpy as np
import serial
import pyqtgraph as pg
import pywt
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import butter, sosfiltfilt, find_peaks


# ---------------- SETTINGS ----------------
PORT = "COM5"
BAUD = 921600
FS = 500
WINDOW_SECONDS = 8
PLOT_SECONDS = 5
UPDATE_SECONDS = 1.0

# Input columns are C XYZ, D XYZ, B XYZ after timestamp.
SENSOR_NAMES = [
    "Palec (C)",
    "Przód klatki piersiowej (D)",
    "Kanał silikonowy (B)",
]
TARGET_SENSOR = 1               # D = chest
REFERENCE_SENSORS = (0, 2)      # C and B; used only as noise indicators

WAVELET = "db4"
SWT_LEVEL = 5
REFERENCE_THRESHOLD_GAIN = 0.8  # raise to suppress more reference-correlated noise
THRESHOLD_GAIN = 0.65            # global denoising strength; tune with recordings

CARDIAC_BAND = (5.0, 25.0)       # mechanical vibration band for envelope extraction
ENVELOPE_WINDOW_S = 0.08
HR_MIN = 40
HR_MAX = 150

WINDOW_N = WINDOW_SECONDS * FS
PLOT_N = PLOT_SECONDS * FS
ENV_FS = 100

band_sos = butter(4, CARDIAC_BAND, btype="bandpass", fs=FS, output="sos")


def parse_lines(buf):
    rows = []
    parts = buf.split("\n")
    remainder = parts.pop() if parts else ""
    for line in parts:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) != 10:
            continue
        try:
            rows.append([float(v) for v in fields])
        except ValueError:
            continue
    return rows, remainder


def robust_sigma(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med)) / 0.6745 + 1e-12)


def make_swt_length(x, level):
    # SWT requires the input length to be divisible by 2**level.
    block = 2 ** level
    extra = (-len(x)) % block
    if extra:
        x = np.pad(x, (0, extra), mode="edge")
    return x, extra


def wavelet_denoise_target(target_xyz, reference_xyz):
    """
    Denoise each target axis using SWT.
    reference_xyz shape: (N, 6), from the two other sensors' axes.
    Returns cleaned_xyz and original_xyz.
    """
    n = len(target_xyz)
    cleaned = np.zeros_like(target_xyz, dtype=float)

    for axis in range(3):
        target = np.asarray(target_xyz[:, axis], dtype=float)
        target = target - np.mean(target)
        padded, extra = make_swt_length(target, SWT_LEVEL)
        coeffs = pywt.swt(padded, WAVELET, level=SWT_LEVEL, trim_approx=False)

        # coeffs are ordered [(cA_level,cD_level), ...].
        # Keep the approximation; shrink detail coefficients at each scale.
        new_coeffs = []
        for level_idx, (cA, cD) in enumerate(coeffs):
            target_noise = robust_sigma(cD)

            ref_sigmas = []
            for ref_axis in range(reference_xyz.shape[1]):
                ref = np.asarray(reference_xyz[:, ref_axis], dtype=float)
                ref = ref - np.mean(ref)
                ref_pad, ref_extra = make_swt_length(ref, SWT_LEVEL)
                ref_coeffs = pywt.swt(ref_pad, WAVELET, level=SWT_LEVEL, trim_approx=False)
                ref_sigmas.append(robust_sigma(ref_coeffs[level_idx][1]))

            ref_noise = float(np.median(ref_sigmas)) if ref_sigmas else 0.0
            # Universal threshold plus reference-derived contribution.
            threshold = THRESHOLD_GAIN * target_noise * np.sqrt(2.0 * np.log(max(len(cD), 2)))
            threshold += REFERENCE_THRESHOLD_GAIN * ref_noise
            cD_clean = pywt.threshold(cD, threshold, mode="soft")
            new_coeffs.append((cA, cD_clean))

        reconstructed = pywt.iswt(new_coeffs, WAVELET)
        if extra:
            reconstructed = reconstructed[:-extra]
        cleaned[:, axis] = reconstructed[:n]

    return cleaned


def autocorr_hr(envelope, fs=ENV_FS):
    x = np.asarray(envelope, dtype=float)
    if len(x) < 4 * fs:
        return None, 0.0
    x = x - np.mean(x)
    if np.std(x) < 1e-8:
        return None, 0.0

    # FFT autocorrelation for efficiency.
    nfft = 1 << int(np.ceil(np.log2(2 * len(x) - 1)))
    spec = np.fft.rfft(x, n=nfft)
    corr = np.fft.irfft(spec * np.conj(spec), n=nfft)[:len(x)]
    corr /= corr[0] + 1e-12

    min_lag = int(60.0 / HR_MAX * fs)
    max_lag = min(int(60.0 / HR_MIN * fs), len(corr) - 1)
    if min_lag >= max_lag:
        return None, 0.0

    # Local maxima are preferable to simply selecting any broad shoulder.
    segment = corr[min_lag:max_lag + 1]
    peaks, _ = find_peaks(segment)
    if len(peaks):
        candidates = peaks + min_lag
        idx = int(candidates[np.argmax(corr[candidates])])
    else:
        idx = int(np.argmax(segment)) + min_lag

    strength = float(corr[idx])
    if strength < 0.08:
        return None, strength
    return float(60.0 * fs / idx), strength


def fft_hr(envelope, fs=ENV_FS):
    x = np.asarray(envelope, dtype=float)
    if len(x) < 4 * fs:
        return None, 0.0
    x = x - np.mean(x)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    mask = (freqs >= HR_MIN / 60.0) & (freqs <= HR_MAX / 60.0)
    if not np.any(mask):
        return None, 0.0
    p = spectrum[mask]
    f = freqs[mask]
    idx = int(np.argmax(p))
    confidence = float(p[idx] / (np.sum(p) + 1e-12))
    return float(f[idx] * 60.0), confidence


def extract_envelope(xyz):
    # Bandpass each axis, then vector magnitude of the filtered vibration.
    filtered_axes = np.zeros_like(xyz, dtype=float)
    for axis in range(3):
        filtered_axes[:, axis] = sosfiltfilt(band_sos, xyz[:, axis])
    vibration = np.sqrt(np.sum(filtered_axes ** 2, axis=1))

    # RMS envelope, then downsample to 100 Hz.
    k = max(3, int(ENVELOPE_WINDOW_S * FS))
    kernel = np.ones(k, dtype=float) / k
    env = np.sqrt(np.maximum(np.convolve(vibration ** 2, kernel, mode="same"), 0.0))
    env = env[::5]
    return vibration, env


def estimate_window(arr):
    xyz = arr[:, 1:].reshape(len(arr), 3, 3)
    target = xyz[:, TARGET_SENSOR, :]
    refs = np.concatenate([xyz[:, i, :] for i in REFERENCE_SENSORS], axis=1)

    # Remove approximate gravity/offset before SWT.
    target = target - np.mean(target, axis=0, keepdims=True)
    refs = refs - np.mean(refs, axis=0, keepdims=True)
    cleaned_xyz = wavelet_denoise_target(target, refs)

    raw_vib, raw_env = extract_envelope(target)
    clean_vib, clean_env = extract_envelope(cleaned_xyz)

    raw_ac, raw_ac_q = autocorr_hr(raw_env)
    clean_ac, clean_ac_q = autocorr_hr(clean_env)
    clean_fft, clean_fft_q = fft_hr(clean_env)

    # Prefer agreement between two independent periodicity estimates.
    if clean_ac is not None and clean_fft is not None:
        diff = abs(clean_ac - clean_fft)
        agreement = max(0.0, 1.0 - diff / 25.0)
        quality = 0.6 * clean_ac_q + 0.4 * agreement
        if diff > 20:
            quality *= 0.5
        candidate = clean_ac if clean_ac_q >= 0.08 else clean_fft
    elif clean_ac is not None:
        candidate, quality = clean_ac, 0.65 * clean_ac_q
    elif clean_fft is not None:
        candidate, quality = clean_fft, 0.35 * clean_fft_q
    else:
        candidate, quality = None, 0.0

    return {
        "raw_env": raw_env,
        "clean_env": clean_env,
        "raw_vib": raw_vib,
        "clean_vib": clean_vib,
        "raw_ac": raw_ac,
        "clean_ac": clean_ac,
        "clean_fft": clean_fft,
        "candidate": candidate,
        "quality": float(quality),
    }


class Tracker:
    def __init__(self):
        self.hr = None
        self.last_good = 0.0

    def update(self, candidate, quality, now):
        if candidate is None or quality < 0.10:
            if self.hr is not None and now - self.last_good <= 3.0:
                return self.hr, "podtrzymanie poprzedniego wyniku"
            return self.hr, "sygnał niewiarygodny"

        if self.hr is None:
            self.hr = float(candidate)
        else:
            jump = abs(float(candidate) - self.hr)
            if jump > 20.0 and quality < 0.45:
                if now - self.last_good <= 3.0:
                    return self.hr, "odrzucono skok HR"
                return self.hr, "kandydat niepewny"
            alpha = 0.35 if quality >= 0.35 else 0.15
            self.hr = (1.0 - alpha) * self.hr + alpha * float(candidate)

        self.last_good = now
        return self.hr, "OK"


def main():
    print(f"Łączenie z {PORT} @ {BAUD}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.01)
    except serial.SerialException as exc:
        print(f"Nie można otworzyć portu {PORT}: {exc}")
        print("Sprawdź PORT i zamknij Arduino Serial Monitor.")
        return

    log_name = "log_wavelet_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    log = open(log_name, "w", buffering=1, encoding="utf-8")
    rows = deque(maxlen=WINDOW_N)
    text_buffer = ""
    tracker = Tracker()
    last_estimate = 0.0
    last_result = None

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(title="Clema — Wavelet motion-artifact reduction")
    win.resize(1200, 850)
    win.show()

    p_raw = win.addPlot(row=0, col=0, title="Obwiednia przed denoisingiem")
    p_raw.setLabel("bottom", "Czas", units="s")
    p_raw.showGrid(x=True, y=True, alpha=0.25)
    c_raw = p_raw.plot(pen=pg.mkPen(width=1.3))

    p_clean = win.addPlot(row=1, col=0, title="Obwiednia po wavelet denoising")
    p_clean.setLabel("bottom", "Czas", units="s")
    p_clean.showGrid(x=True, y=True, alpha=0.25)
    c_clean = p_clean.plot(pen=pg.mkPen(width=1.3))

    status = QtWidgets.QLabel("Czekam na dane...")
    status.setStyleSheet("font-size: 15px; padding: 8px;")
    win.addItem(status, row=0, col=1, rowspan=2)

    details = QtWidgets.QLabel("Po zebraniu 8 sekund pojawią się oszacowania.")
    details.setStyleSheet("font-size: 13px; padding: 8px;")
    win.addItem(details, row=2, col=0, colspan=2)

    def update():
        nonlocal text_buffer, last_estimate, last_result
        data = ser.read(8192)
        if data:
            decoded = data.decode("ascii", "ignore")
            log.write(decoded)
            text_buffer += decoded
            parsed, text_buffer = parse_lines(text_buffer)
            for row in parsed:
                rows.append(row)

        if len(rows) < WINDOW_N:
            status.setText(f"Zbieranie danych: {len(rows)}/{WINDOW_N} próbek")
            return

        now = time.monotonic()
        if now - last_estimate >= UPDATE_SECONDS:
            arr = np.asarray(rows, dtype=float)
            try:
                last_result = estimate_window(arr)
                hr, track_status = tracker.update(
                    last_result["candidate"], last_result["quality"], now
                )
                last_result["tracked_hr"] = hr
                last_result["track_status"] = track_status
            except Exception as exc:
                status.setText(f"Błąd wavelet processing: {exc}")
                return
            last_estimate = now

        if last_result is None:
            return

        raw_env = last_result["raw_env"][-int(PLOT_SECONDS * ENV_FS):]
        clean_env = last_result["clean_env"][-int(PLOT_SECONDS * ENV_FS):]
        x = np.arange(len(raw_env)) / float(ENV_FS)
        c_raw.setData(x, raw_env)
        c_clean.setData(np.arange(len(clean_env)) / float(ENV_FS), clean_env)

        hr = last_result.get("tracked_hr")
        hr_text = f"{hr:.1f} BPM" if hr is not None else "-- BPM"
        status.setText(
            f"<b>HR: {hr_text}</b><br>"
            f"Status: {last_result.get('track_status')}<br>"
            f"Jakość heurystyczna: {last_result.get('quality', 0):.2f}"
        )
        details.setText(
            f"RAW AC: {fmt(last_result.get('raw_ac'))} BPM | "
            f"CLEAN AC: {fmt(last_result.get('clean_ac'))} BPM | "
            f"CLEAN FFT: {fmt(last_result.get('clean_fft'))} BPM<br>"
            f"Target: {SENSOR_NAMES[TARGET_SENSOR]} | "
            f"References: {', '.join(SENSOR_NAMES[i] for i in REFERENCE_SENSORS)}<br>"
            f"Wavelet: {WAVELET}, level={SWT_LEVEL}, "
            f"threshold gain={THRESHOLD_GAIN:.2f}, log: {log_name}"
        )

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(30)

    def close_event(event):
        try:
            timer.stop()
            ser.close()
            log.close()
        finally:
            event.accept()

    win.closeEvent = close_event
    sys.exit(app.exec())


def fmt(value):
    return "--" if value is None else f"{float(value):.1f}"


if __name__ == "__main__":
    main()
