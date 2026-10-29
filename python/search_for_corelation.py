"""
live_fft.py — estymacja HR przez FFT dla 3 akcelerometrów.

Format strumienia z firmware:
t,cx,cy,cz,dx,dy,dz,bx,by,bz

Domyślne mapowanie sensorów:
C = palec, D = przód klatki piersiowej, B = kanał silikonowy.
Sprawdź kolejność w swoim firmware.

Algorytm:
1. Dla każdego sensora osobno liczy moduł przyspieszenia.
2. Usuwa średnią i filtruje sygnał pasmowo 0.5–3 Hz.
3. Liczy widmo FFT w oknie czasowym.
4. Szuka dominującej częstotliwości w zakresie 40–150 BPM.
5. Pokazuje BPM i wykresy dla wszystkich trzech sensorów.

To prototyp badawczy, nie urządzenie medyczne. Dominująca częstotliwość
może pochodzić z kroków lub innych ruchów zamiast z pracy serca.
"""

import sys
import time
import datetime
from collections import deque

import numpy as np
import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import butter, sosfiltfilt


# -------------------- USTAWIENIA --------------------
PORT = "COM5"
BAUD = 921600
FS = 500                       # musi odpowiadać firmware
WINDOW_SECONDS = 8             # okno analizy
PLOT_SECONDS = 5               # długość wykresu
BPM_MIN = 40
BPM_MAX = 150
ESTIMATE_EVERY_S = 1.0

BAND_LOW_HZ = BPM_MIN / 60.0
BAND_HIGH_HZ = BPM_MAX / 60.0

SENSOR_NAMES = [
    "Palec (sensor 1)",
    "Przód klatki (sensor 2)",
    "Kanał silikonowy (sensor 3)",
]

# Format: timestamp, Cx,Cy,Cz, Dx,Dy,Dz, Bx,By,Bz
SENSOR_SLICES = [(0, 3), (3, 6), (6, 9)]

WINDOW_N = int(WINDOW_SECONDS * FS)
PLOT_N = int(PLOT_SECONDS * FS)

sos = butter(
    4,
    [BAND_LOW_HZ, BAND_HIGH_HZ],
    btype="bandpass",
    fs=FS,
    output="sos",
)


def parse_lines(buf):
    """Parsuje pełne linie CSV, zachowując niepełny ostatni wiersz."""
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
            rows.append([float(x) for x in fields])
        except ValueError:
            continue

    return rows, remainder


def get_magnitude_window(arr, sensor_index):
    """Moduł przyspieszenia z osi XYZ dla jednego sensora."""
    xyz = arr[:, 1:].reshape(len(arr), 3, 3)[:, sensor_index, :]
    mag = np.sqrt(np.sum(xyz * xyz, axis=1))
    return mag


def estimate_fft(signal):
    """
    Zwraca (bpm, filtrowany sygnał, częstotliwość dominująca, pewność).
    Pewność jest tylko względną miarą stosunku mocy dominującego piku do mocy
    w badanym paśmie — nie jest prawdopodobieństwem poprawnego HR.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) < WINDOW_N:
        return None, signal, None, 0.0

    if not np.all(np.isfinite(signal)):
        signal = np.nan_to_num(signal)

    signal = signal - np.mean(signal)

    try:
        filtered = sosfiltfilt(sos, signal)
    except ValueError:
        return None, signal, None, 0.0

    # Okno Hann ogranicza przeciek widma.
    windowed = filtered * np.hanning(len(filtered))
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / FS)

    mask = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    if not np.any(mask):
        return None, filtered, None, 0.0

    band_power = spectrum[mask]
    band_freqs = freqs[mask]
    peak_index = int(np.argmax(band_power))
    peak_freq = float(band_freqs[peak_index])

    if peak_freq <= 0:
        return None, filtered, None, 0.0

    bpm = peak_freq * 60.0
    total_power = float(np.sum(band_power)) + 1e-12
    confidence = float(band_power[peak_index] / total_power)

    # Interpolacja paraboliczna dla lepszej rozdzielczości częstotliwości.
    if 0 < peak_index < len(band_power) - 1:
        y0, y1, y2 = np.log(band_power[peak_index - 1:peak_index + 2] + 1e-12)
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            delta = float(np.clip(delta, -1.0, 1.0))
            peak_freq += delta * (FS / len(windowed))
            bpm = peak_freq * 60.0

    return float(bpm), filtered, peak_freq, confidence


class Processor:
    def __init__(self):
        self.rows = deque(maxlen=WINDOW_N)
        self.last_estimate_time = 0.0
        self.results = [None, None, None]
        self.filtered = [np.array([]) for _ in range(3)]
        self.confidence = [0.0, 0.0, 0.0]
        self.frequency = [None, None, None]

    def push(self, rows):
        for row in rows:
            if len(row) == 10:
                self.rows.append(row)

    def estimate(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_estimate_time < ESTIMATE_EVERY_S:
            return
        self.last_estimate_time = now

        if len(self.rows) < WINDOW_N:
            return

        arr = np.asarray(self.rows, dtype=float)
        for idx in range(3):
            mag = get_magnitude_window(arr, idx)
            bpm, filt, freq, conf = estimate_fft(mag)
            self.results[idx] = bpm
            self.filtered[idx] = filt
            self.frequency[idx] = freq
            self.confidence[idx] = conf

    def plot_data(self, idx):
        arr = np.asarray(self.rows, dtype=float)
        if len(arr) == 0:
            return np.array([]), np.array([])

        mag = get_magnitude_window(arr, idx)
        _, filt, _, _ = estimate_fft(mag)
        if len(filt) == 0:
            return np.array([]), np.array([])

        filt = filt[-PLOT_N:]
        x = np.arange(len(filt)) / float(FS)
        return x, filt


def main():
    print(f"Łączenie z {PORT} @ {BAUD}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.01)
    except serial.SerialException as exc:
        print(f"Nie można otworzyć {PORT}: {exc}")
        print("Sprawdź PORT i zamknij Arduino Serial Monitor.")
        return

    log_name = "log_fft_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    log = open(log_name, "w", buffering=1, encoding="utf-8")
    processor = Processor()
    text_buffer = ""

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(title="Clema — estymacja HR metodą FFT")
    win.resize(1200, 850)
    win.show()

    curves = []
    labels = []
    for idx, name in enumerate(SENSOR_NAMES):
        plot = win.addPlot(row=idx, col=0, title=name)
        plot.setLabel("bottom", "Czas w oknie", units="s")
        plot.setLabel("left", "Przyspieszenie filtrowane", units="m/s²")
        plot.showGrid(x=True, y=True, alpha=0.25)
        curves.append(plot.plot(pen=pg.mkPen(width=1.5)))

        label = QtWidgets.QLabel(f"{name}: czekam na dane")
        label.setStyleSheet("font-size: 14px; padding: 6px;")
        win.addItem(label, row=idx, col=1)
        labels.append(label)

    footer = QtWidgets.QLabel(f"Log strumienia: {log_name}")
    win.addItem(footer, row=3, col=0, colspan=2)

    last_ui_update = 0.0

    def update():
        nonlocal text_buffer, last_ui_update

        data = ser.read(8192)
        if data:
            decoded = data.decode("ascii", "ignore")
            log.write(decoded)
            text_buffer += decoded
            rows, text_buffer = parse_lines(text_buffer)
            if rows:
                processor.push(rows)

        processor.estimate()

        arr = np.asarray(processor.rows, dtype=float)
        if len(arr) >= WINDOW_N:
            for idx in range(3):
                x, y = processor.plot_data(idx)
                curves[idx].setData(x, y)

                bpm = processor.results[idx]
                conf = processor.confidence[idx]
                freq = processor.frequency[idx]
                bpm_text = f"{bpm:.1f} BPM" if bpm is not None else "-- BPM"
                freq_text = f"{freq:.3f} Hz" if freq is not None else "-- Hz"
                labels[idx].setText(
                    f"<b>{SENSOR_NAMES[idx]}</b><br>"
                    f"HR: {bpm_text} | f: {freq_text} | "
                    f"względna koncentracja widma: {conf:.2f}"
                )
        else:
            count = len(processor.rows)
            for idx in range(3):
                labels[idx].setText(
                    f"<b>{SENSOR_NAMES[idx]}</b><br>"
                    f"Zbieranie danych: {count}/{WINDOW_N} próbek"
                )

        last_ui_update = time.monotonic()

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


if __name__ == "__main__":
    main()
