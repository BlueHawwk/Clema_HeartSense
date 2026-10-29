"""
live_fusion.py

Eksperymentalny algorytm fuzji sensorów dla projektu Clema:
- 3 akcelerometry, strumień:
  t,cx,cy,cz,dx,dy,dz,bx,by,bz
- osobna analiza osi i modułu przyspieszenia każdego sensora,
- filtr pasmowy 0.7–3.0 Hz (42–180 BPM),
- adaptacyjne odejmowanie składowych przewidywalnych z referencji metodą LMS,
- kandydaci HR z autokorelacji i FFT,
- ocena zgodności, kara za możliwą kadencję ruchu,
- tracker wygładzający wynik i sygnalizujący niską jakość.

WAŻNE:
1. To prototyp do badań, nie urządzenie medyczne.
2. Założone przypisanie domyślne:
   C = palec, D = przód klatki, B = kanał silikonowy.
   Sprawdź kolejność kolumn z firmware.
3. Nie zakładamy, że sensor na kanale jest czystą referencją ruchu:
   LMS jest ograniczany do części sygnału przewidywalnej z referencji,
   ale może też usuwać część sygnału serca, jeśli jest z nimi skorelowany.
4. Wymaga porównania z Polar H10; HR może być błędne podczas ruchu.
"""

import sys
import time
import datetime
from collections import deque

import numpy as np
import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import butter, sosfilt


# ----------------------------- SETTINGS -----------------------------
PORT = "COM5"
BAUD = 921600
FS = 500                         # musi zgadzać się z firmware
WINDOW_SECONDS = 8
PLOT_SECONDS = 5
UPDATE_SECONDS = 1.0

BPM_MIN = 40.0
BPM_MAX = 150.0
BAND_LOW_HZ = BPM_MIN / 60.0
BAND_HIGH_HZ = BPM_MAX / 60.0

# LMS: mały krok = wolniejsza, zwykle stabilniejsza adaptacja.
LMS_TAPS = 12                    # liczba opóźnionych próbek na kanał referencyjny
LMS_MU = 0.0005                  # do strojenia; zależy od skali wejścia
LMS_EPS = 1e-6
LMS_WARMUP_SECONDS = 1.0
LMS_MAX_WEIGHT_NORM = 5.0

# Odrzucanie słabych / niewiarygodnych kandydatów.
MIN_PEAK_CONTRAST = 1.20
MIN_AUTOCORR_STRENGTH = 0.10
TRACK_MAX_JUMP_BPM = 18.0
TRACK_SMOOTHING = 0.30
HOLD_SECONDS = 3.0

SENSOR_NAMES = [
    "Palec (sensor 1)",
    "Przód klatki (sensor 2)",
    "Kanał silikonowy (sensor 3)",
]

# Wiersz: timestamp + 3 osie sensora C + 3 osie D + 3 osie B.
# Zwracane przez parser kolumny przyspieszeń indeksowane są 0..8:
# sensor 0 = C (0:3), sensor 1 = D (3:6), sensor 2 = B (6:9).
TARGET_SENSOR = 0                # domyślnie palec jako sygnał docelowy
REFERENCE_SENSORS = (1, 2)       # przód klatki i kanał silikonowy jako referencje

WINDOW_N = int(WINDOW_SECONDS * FS)
PLOT_N = int(PLOT_SECONDS * FS)

band_sos = butter(
    4,
    [BAND_LOW_HZ, BAND_HIGH_HZ],
    btype="bandpass",
    fs=FS,
    output="sos",
)
motion_sos = butter(
    2,
    [0.5, 4.0],
    btype="bandpass",
    fs=FS,
    output="sos",
)


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


def make_features(accel):
    """
    accel shape: (N, 3 sensory, 3 osie).
    Zwraca moduły i osie, używane do budowy celu/referencji.
    """
    magnitudes = np.sqrt(np.sum(accel * accel, axis=2))
    return magnitudes


def safe_bandpass(x, sos):
    x = np.asarray(x, dtype=float)
    if len(x) < 30:
        return np.zeros_like(x)
    x = np.nan_to_num(x)
    # Dla strumienia live używamy filtru przyczynowego.
    y, _ = sosfilt(sos, x)
    return y


class LMSCanceller:
    """
    Adaptacyjny filtr LMS z wieloma referencjami i opóźnieniami.
    Pracuje na sygnałach pasmowych (3 osie każdego sensora referencyjnego).
    """
    def __init__(self, n_refs, taps, mu, eps):
        self.n_refs = n_refs
        self.taps = taps
        self.mu = mu
        self.eps = eps
        self.w = np.zeros(n_refs * taps, dtype=float)
        self.hist = np.zeros((n_refs, taps), dtype=float)

    def reset(self):
        self.w[:] = 0.0
        self.hist[:] = 0.0

    def step(self, refs):
        # refs: wektor długości n_refs; aktualna próbka każdego kanału.
        self.hist[:, 1:] = self.hist[:, :-1]
        self.hist[:, 0] = refs
        u = self.hist.reshape(-1)
        y_hat = float(np.dot(self.w, u))
        # Normalized LMS ogranicza wrażliwość na skalę sygnału.
        norm = float(np.dot(u, u)) + self.eps
        error = float(refs[0] - y_hat)  # tylko do wyznaczania wymiaru? nadpisywane przez process
        return y_hat, u, norm


class AdaptiveCanceller:
    """
    NLMS wielokanałowy dla jednego sygnału docelowego.
    Referencje: 3 osie * 2 sensory (6 kanałów).
    """
    def __init__(self, n_refs, taps=12, mu=0.0005, eps=1e-6):
        self.n_refs = n_refs
        self.taps = taps
        self.mu = mu
        self.eps = eps
        self.w = np.zeros(n_refs * taps, dtype=float)
        self.hist = np.zeros((n_refs, taps), dtype=float)

    def step(self, desired, refs):
        self.hist[:, 1:] = self.hist[:, :-1]
        self.hist[:, 0] = refs
        u = self.hist.reshape(-1)
        estimate = float(np.dot(self.w, u))
        error = float(desired - estimate)
        norm = float(np.dot(u, u)) + self.eps
        self.w += (self.mu / norm) * error * u
        wnorm = float(np.linalg.norm(self.w))
        if wnorm > LMS_MAX_WEIGHT_NORM:
            self.w *= LMS_MAX_WEIGHT_NORM / wnorm
        return error, estimate, norm


def autocorr_candidate(x):
    """
    Kandydat HR z autokorelacji sygnału w zakresie 40–150 BPM.
    Zwraca bpm i strength.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 4 * FS:
        return None, 0.0
    x = x - np.mean(x)
    # Obwiednia energii zmniejsza wrażliwość na zmianę znaku przebiegu.
    kernel = max(3, int(0.06 * FS))
    energy = np.convolve(x * x, np.ones(kernel) / kernel, mode="same")
    env = np.sqrt(np.maximum(energy, 0.0))[::5]  # 100 Hz
    env -= np.mean(env)
    if np.std(env) < 1e-8:
        return None, 0.0

    corr = np.correlate(env, env, mode="full")[len(env) - 1:]
    if corr[0] <= 1e-12:
        return None, 0.0
    corr = corr / corr[0]
    min_lag = int(60.0 / BPM_MAX * 100.0)
    max_lag = int(60.0 / BPM_MIN * 100.0)
    if max_lag >= len(corr):
        max_lag = len(corr) - 1
    if min_lag >= max_lag:
        return None, 0.0

    section = corr[min_lag:max_lag + 1]
    idx = int(np.argmax(section)) + min_lag
    strength = float(corr[idx])
    if strength < MIN_AUTOCORR_STRENGTH:
        return None, strength

    # Prosta interpolacja paraboliczna dla ułamkowej próbki.
    lag = float(idx)
    if 1 <= idx < len(corr) - 1:
        a, b, c = corr[idx - 1], corr[idx], corr[idx + 1]
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-12:
            lag += float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))
    bpm = 60.0 * 100.0 / lag
    return float(bpm), strength


def fft_candidate(x):
    """Kandydat HR z dominującej częstotliwości w paśmie."""
    x = np.asarray(x, dtype=float)
    if len(x) < 4 * FS:
        return None, 0.0
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / FS)
    mask = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    if not np.any(mask):
        return None, 0.0
    bp = spec[mask]
    bf = freqs[mask]
    i = int(np.argmax(bp))
    bpm = float(bf[i] * 60.0)
    confidence = float(bp[i] / (np.sum(bp) + 1e-12))
    return bpm, confidence


class HRTracker:
    def __init__(self):
        self.value = None
        self.last_good_time = None

    def update(self, candidate, quality, now):
        if candidate is None or quality < 0.08:
            if self.last_good_time is not None and now - self.last_good_time <= HOLD_SECONDS:
                return self.value, "podtrzymanie poprzedniego wyniku"
            return self.value, "sygnał niewiarygodny"

        candidate = float(candidate)
        if self.value is None:
            self.value = candidate
            self.last_good_time = now
            return self.value, "inicjalizacja"

        if abs(candidate - self.value) > TRACK_MAX_JUMP_BPM:
            if quality < 0.45:
                if self.last_good_time is not None and now - self.last_good_time <= HOLD_SECONDS:
                    return self.value, "odrzucono skok HR"
                return self.value, "kandydat odrzucony"
            # Silny nowy kandydat może oznaczać rzeczywistą zmianę HR.

        self.value = (1.0 - TRACK_SMOOTHING) * self.value + TRACK_SMOOTHING * candidate
        self.last_good_time = now
        return self.value, "OK"


class FusionProcessor:
    def __init__(self):
        self.rows = deque(maxlen=WINDOW_N)
        # Filtry przyczynowe mają własny stan na każdą oś.
        self.band_zi = None
        self.motion_zi = None
        self.filtered_target = deque(maxlen=WINDOW_N)
        self.filtered_raw = deque(maxlen=WINDOW_N)
        self.filtered_refs = deque(maxlen=WINDOW_N)
        self.last_results = {}
        self.tracker = HRTracker()
        self.last_update = 0.0
        self.warmup_samples = 0

        # 6 referencji: XYZ z dwóch sensorów referencyjnych.
        self.canceller = AdaptiveCanceller(
            n_refs=6, taps=LMS_TAPS, mu=LMS_MU, eps=LMS_EPS
        )

        self._target_sos_zi = None
        self._ref_sos_zi = None
        self._motion_sos_zi = None
        self._raw_target_buffer = deque(maxlen=WINDOW_N)
        self._clean_target_buffer = deque(maxlen=WINDOW_N)
        self._ref_energy_buffer = deque(maxlen=WINDOW_N)

    def push(self, rows):
        arr = np.asarray(rows, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 10:
            return

        accel = arr[:, 1:].reshape(len(arr), 3, 3)
        for row_idx in range(len(arr)):
            if not np.all(np.isfinite(accel[row_idx])):
                continue

            # W tym prototypie target = moduł przyspieszenia sensora docelowego.
            target_xyz = accel[row_idx, TARGET_SENSOR]
            if np.allclose(target_xyz, 0.0):
                # Pomijamy próbki zerowe zamiast uczyć filtr na brakujących danych.
                continue

            target_mag = float(np.linalg.norm(target_xyz))
            target_mag -= 9.80665  # przybliżone usunięcie g; orientacja może się zmieniać
            refs_xyz = accel[row_idx, list(REFERENCE_SENSORS), :].reshape(-1)
            if np.allclose(refs_xyz, 0.0):
                refs_xyz = np.zeros(6, dtype=float)

            # Filtr pasmowy celu i referencji; osobno dla każdej próbki.
            # Stan filtra utrzymujemy między próbkami.
            if self._target_sos_zi is None:
                self._target_sos_zi = np.zeros((band_sos.shape[0], 2), dtype=float)
                self._ref_sos_zi = np.zeros((band_sos.shape[0], 2, 6), dtype=float)
                self._motion_sos_zi = np.zeros((motion_sos.shape[0], 2, 6), dtype=float)

            y_target, self._target_sos_zi = self._sos_one(
                band_sos, target_mag, self._target_sos_zi
            )

            refs_bp = []
            refs_motion = []
            for j in range(6):
                val, self._ref_sos_zi[:, :, j] = self._sos_one(
                    band_sos, refs_xyz[j], self._ref_sos_zi[:, :, j]
                )
                mv, self._motion_sos_zi[:, :, j] = self._sos_one(
                    motion_sos, refs_xyz[j], self._motion_sos_zi[:, :, j]
                )
                refs_bp.append(val)
                refs_motion.append(mv)

            refs_bp = np.asarray(refs_bp, dtype=float)
            refs_motion = np.asarray(refs_motion, dtype=float)
            motion_energy = float(np.sqrt(np.mean(refs_motion * refs_motion)))

            # LMS uczy się tylko gdy referencje wykazują ruch ponad mały próg.
            # Próg jest heurystyczny i musi być zweryfikowany na logach.
            moving = motion_energy > 0.015
            if moving and self.warmup_samples >= int(LMS_WARMUP_SECONDS * FS):
                clean, removed, norm = self.canceller.step(y_target, refs_bp)
            else:
                clean, removed = y_target, 0.0
                # Podczas ciszy nie adaptujemy współczynników, żeby ograniczyć
                # ryzyko odejmowania sygnału serca.
                norm = float(np.dot(refs_bp, refs_bp)) + LMS_EPS

            self._raw_target_buffer.append(y_target)
            self._clean_target_buffer.append(clean)
            self._ref_energy_buffer.append(motion_energy)
            self.warmup_samples += 1

        self.rows.extend(arr.tolist())

    @staticmethod
    def _sos_one(sos, x, zi):
        from scipy.signal import sosfilt
        y, zf = sosfilt(sos, [x], zi=zi)
        return float(y[0]), zf

    def estimate(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_update < UPDATE_SECONDS:
            return self.last_results
        self.last_update = now

        if len(self._raw_target_buffer) < int(4 * FS):
            self.last_results = {"status": "Zbieranie danych", "n": len(self._raw_target_buffer)}
            return self.last_results

        raw = np.asarray(self._raw_target_buffer, dtype=float)
        clean = np.asarray(self._clean_target_buffer, dtype=float)
        refs_energy = np.asarray(self._ref_energy_buffer, dtype=float)

        raw_ac, raw_ac_q = autocorr_candidate(raw)
        clean_ac, clean_ac_q = autocorr_candidate(clean)
        clean_fft, clean_fft_q = fft_candidate(clean)

        # Jeśli AC i FFT zgadzają się, wzrasta wiarygodność kandydata.
        candidates = []
        if clean_ac is not None:
            q = 0.45 * clean_ac_q
            if clean_fft is not None:
                diff = abs(clean_ac - clean_fft)
                agreement = max(0.0, 1.0 - diff / 20.0)
                q = 0.55 * clean_ac_q + 0.45 * agreement
                if diff > 18.0:
                    q *= 0.5
            candidates.append((clean_ac, q, "AC"))
        if clean_fft is not None:
            candidates.append((clean_fft, 0.35 * clean_fft_q, "FFT"))
        if raw_ac is not None:
            candidates.append((raw_ac, 0.25 * raw_ac_q, "RAW-AC"))

        if candidates:
            # Wybieramy kandydata o najwyższej ocenie; unikamy średniej
            # bezkrytycznie łączącej różne rytmy.
            candidate, quality, method = max(candidates, key=lambda item: item[1])
        else:
            candidate, quality, method = None, 0.0, "brak"

        tracked, status = self.tracker.update(candidate, quality, now)
        self.last_results = {
            "status": status,
            "hr": tracked,
            "candidate": candidate,
            "quality": quality,
            "method": method,
            "raw_ac": raw_ac,
            "clean_ac": clean_ac,
            "clean_fft": clean_fft,
            "raw_ac_q": raw_ac_q,
            "clean_ac_q": clean_ac_q,
            "clean_fft_q": clean_fft_q,
            "moving_fraction": float(np.mean(refs_energy > 0.015)),
            "raw": raw,
            "clean": clean,
        }
        return self.last_results


def main():
    print(f"Łączenie z {PORT} @ {BAUD}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.01)
    except serial.SerialException as exc:
        print(f"Nie można otworzyć portu {PORT}: {exc}")
        print("Sprawdź PORT i zamknij Arduino Serial Monitor.")
        return

    log_name = "log_fusion_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    log = open(log_name, "w", buffering=1, encoding="utf-8")
    processor = FusionProcessor()
    text_buffer = ""

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(title="Clema — Multi-Sensor Adaptive HR Fusion")
    win.resize(1250, 900)
    win.show()

    p_raw = win.addPlot(row=0, col=0, title="Sygnał docelowy — sensor palca")
    p_raw.setLabel("bottom", "Czas", units="s")
    p_raw.setLabel("left", "Przyspieszenie filtrowane", units="m/s²")
    p_raw.showGrid(x=True, y=True, alpha=0.25)
    c_raw = p_raw.plot(pen=pg.mkPen(width=1.2))

    p_clean = win.addPlot(row=1, col=0, title="Sygnał po adaptacyjnym usuwaniu ruchu (LMS)")
    p_clean.setLabel("bottom", "Czas", units="s")
    p_clean.setLabel("left", "Sygnał", units="m/s²")
    p_clean.showGrid(x=True, y=True, alpha=0.25)
    c_clean = p_clean.plot(pen=pg.mkPen(width=1.2))

    p_ref = win.addPlot(row=2, col=0, title="Energia ruchu z sensorów referencyjnych")
    p_ref.setLabel("bottom", "Czas", units="s")
    p_ref.setLabel("left", "RMS", units="m/s²")
    p_ref.showGrid(x=True, y=True, alpha=0.25)
    c_ref = p_ref.plot(pen=pg.mkPen(width=1.2))

    status = QtWidgets.QLabel("Czekam na dane...")
    status.setStyleSheet("font-size: 15px; padding: 6px;")
    win.addItem(status, row=0, col=1, rowspan=2)
    detail = QtWidgets.QLabel("Kandydaci HR pojawią się po zebraniu okna danych.")
    detail.setStyleSheet("font-size: 13px; padding: 6px;")
    win.addItem(detail, row=2, col=1)

    def update():
        nonlocal text_buffer
        data = ser.read(8192)
        if data:
            decoded = data.decode("ascii", "ignore")
            log.write(decoded)
            text_buffer += decoded
            rows, text_buffer = parse_lines(text_buffer)
            if rows:
                processor.push(rows)

        result = processor.estimate()
        if "raw" not in result:
            status.setText(f"<b>{result.get('status', 'Czekam')}</b>")
            return

        raw = result["raw"][-PLOT_N:]
        clean = result["clean"][-PLOT_N:]
        x = np.arange(len(raw)) / float(FS)
        c_raw.setData(x, raw)
        c_clean.setData(x, clean)

        energy = np.asarray(processor._ref_energy_buffer, dtype=float)[-PLOT_N:]
        c_ref.setData(np.arange(len(energy)) / float(FS), energy)

        hr = result.get("hr")
        hr_text = f"{hr:.1f} BPM" if hr is not None else "-- BPM"
        cand = result.get("candidate")
        cand_text = f"{cand:.1f}" if cand is not None else "--"
        status.setText(
            f"<b>HR: {hr_text}</b><br>"
            f"Status: {result.get('status')}<br>"
            f"Metoda: {result.get('method')}<br>"
            f"Jakość (heurystyczna): {result.get('quality', 0):.2f}"
        )
        detail.setText(
            f"Kandydat: {cand_text} BPM<br>"
            f"RAW AC: {fmt(result.get('raw_ac'))} BPM<br>"
            f"CLEAN AC: {fmt(result.get('clean_ac'))} BPM<br>"
            f"CLEAN FFT: {fmt(result.get('clean_fft'))} BPM<br>"
            f"Udział próbek z wykrytym ruchem: {result.get('moving_fraction', 0):.2f}<br>"
            f"Log: {log_name}"
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
