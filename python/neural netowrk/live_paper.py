"""
live_paper.py

Eksperymentalna implementacja metody inspirowanej artykułem:
"Heart Rate Measurement Using the Built-In Triaxial Accelerometer
from a Commercial Digital Writing Device" (Sensors, 2024).

Dostosowanie do projektu Clema:
- ESP8266 wysyła wiersze: t,cx,cy,cz,dx,dy,dz,bx,by,bz
- sensor 1: palec
- sensor 2: przód klatki piersiowej
- sensor 3: kanał silikonowy

Algorytm dla KAŻDEGO sensora niezależnie:
1. oblicz moduł przyspieszenia sqrt(ax² + ay² + az²),
2. filtr dolnoprzepustowy Butterwortha 4. rzędu, fc = 2 Hz,
3. wykryj piki i oblicz odstępy między nimi,
4. BPM = 60 / mediana odstępów.

To jest prototyp badawczy, a nie urządzenie medyczne. W artykule pomiar
był wykonywany podczas spokojnego pisania; wyniki podczas biegania mogą
być znacznie gorsze. Parametry detekcji pików mogą wymagać strojenia.
"""

import sys
import time
import datetime
from collections import deque

import numpy as np
import serial
import pyqtgraph as pg
from scipy.signal import butter, sosfiltfilt, find_peaks
from pyqtgraph.Qt import QtCore, QtWidgets


# ---------- Ustawienia ----------
PORT = "COM5"
BAUD = 921600
FS = 500                    # musi odpowiadać częstotliwości firmware
CUTOFF_HZ = 2.0              # filtr zbliżony do artykułu
FILTER_ORDER = 4
WINDOW_SECONDS = 10          # długość analizowanego fragmentu
PLOT_SECONDS = 5             # ile sekund pokazywać na wykresie
BPM_MIN = 40
BPM_MAX = 150
ESTIMATE_EVERY_S = 1.0

# Strojenie detekcji pików:
# Prominence jest wyliczane automatycznie na podstawie rozrzutu sygnału.
MIN_PROMINENCE_ABS = 0.015   # m/s²; dolne ograniczenie, do strojenia
PROMINENCE_STD_FACTOR = 0.35
MIN_PEAK_DISTANCE_FACTOR = 0.80  # część minimalnego okresu dla BPM_MAX


SENSOR_NAMES = [
    "Palec (sensor 1)",
    "Przód klatki (sensor 2)",
    "Kanał silikonowy (sensor 3)",
]

# Kolumny osi w wierszu X:
# timestamp, cx,cy,cz, dx,dy,dz, bx,by,bz
SENSOR_SLICES = [(0, 3), (3, 6), (6, 9)]

sos = butter(
    FILTER_ORDER,
    CUTOFF_HZ,
    btype="lowpass",
    fs=FS,
    output="sos",
)


def parse_lines(buf):
    """Parsuje wiersze CSV z firmware; zachowuje niepełny ostatni wiersz."""
    rows = []
    lines = buf.split("\n")
    remainder = lines.pop() if lines else ""

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(",")
        if len(parts) != 10:
            continue

        try:
            # Pierwsza kolumna to timestamp w mikrosekundach.
            values = [float(v) for v in parts]
            rows.append(values)
        except ValueError:
            continue

    return rows, remainder


def calculate_magnitude(rows, start, end):
    """Zwraca moduł przyspieszenia dla jednego sensora."""
    xyz = rows[:, 1 + start:1 + end]
    return np.sqrt(np.sum(xyz * xyz, axis=1))


def estimate_hr(signal, fs=FS):
    """
    Estymuje HR z jednego modułu przyspieszenia.
    Zwraca (bpm, filtrowany_sygnal, piki, jakosc, opis).
    """
    signal = np.asarray(signal, dtype=float)

    if len(signal) < int(4 * fs):
        return None, signal, np.array([], dtype=int), 0.0, "Za mało danych"

    if not np.all(np.isfinite(signal)):
        signal = np.nan_to_num(signal)

    # Filtrowanie całego okna. sosfiltfilt nie przesuwa fazy,
    # ale na krańcach okna mogą wystąpić artefakty.
    try:
        filtered = sosfiltfilt(sos, signal)
    except ValueError:
        return None, signal, np.array([], dtype=int), 0.0, "Filtrowanie niemożliwe"

    # Usuń stały poziom dla czytelniejszej detekcji pików.
    centered = filtered - np.median(filtered)
    spread = 1.4826 * np.median(np.abs(centered - np.median(centered)))
    if not np.isfinite(spread) or spread < 1e-8:
        spread = float(np.std(centered))

    prominence = max(MIN_PROMINENCE_ABS, PROMINENCE_STD_FACTOR * spread)
    min_distance = int((60.0 / BPM_MAX) * fs * MIN_PEAK_DISTANCE_FACTOR)
    max_distance = int((60.0 / BPM_MIN) * fs * 1.15)

    peaks, properties = find_peaks(
        centered,
        distance=max(1, min_distance),
        prominence=prominence,
    )

    if len(peaks) < 3:
        return None, filtered, peaks, 0.0, "Za mało pików"

    intervals = np.diff(peaks) / float(fs)
    bpm_each = 60.0 / intervals
    valid = (bpm_each >= BPM_MIN) & (bpm_each <= BPM_MAX)
    valid_intervals = intervals[valid]

    if len(valid_intervals) < 2:
        return None, filtered, peaks, 0.0, "Brak stabilnego rytmu"

    # Mediana jest mniej wrażliwa na pojedyncze błędnie wykryte piki.
    bpm = 60.0 / float(np.median(valid_intervals))

    # Prosta miara regularności odstępów: 1 = bardzo regularnie.
    cv = float(np.std(valid_intervals) / max(np.mean(valid_intervals), 1e-9))
    quality = float(np.clip(1.0 - cv, 0.0, 1.0))

    if not (BPM_MIN <= bpm <= BPM_MAX):
        return None, filtered, peaks, quality, "Poza zakresem HR"

    note = "OK" if quality >= 0.35 else "Nieregularny sygnał"
    return bpm, filtered, peaks, quality, note


class PaperProcessor:
    def __init__(self):
        self.window_n = int(WINDOW_SECONDS * FS)
        self.plot_n = int(PLOT_SECONDS * FS)
        self.timestamps = deque(maxlen=self.window_n)
        self.magnitudes = [deque(maxlen=self.window_n) for _ in range(3)]
        self.last_estimates = [None, None, None]
        self.last_quality = [0.0, 0.0, 0.0]
        self.last_notes = ["Czekam na dane"] * 3
        self.last_filtered = [np.array([]) for _ in range(3)]
        self.last_peaks = [np.array([], dtype=int) for _ in range(3)]
        self.last_time = 0.0

    def push(self, rows):
        arr = np.asarray(rows, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 10:
            return

        # Przyjmujemy, że dane przychodzą z firmware w m/s².
        for row in arr:
            self.timestamps.append(row[0])
            for idx, (start, end) in enumerate(SENSOR_SLICES):
                mag = calculate_magnitude(row.reshape(1, -1), start, end)[0]
                self.magnitudes[idx].append(float(mag))

    def estimate(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_time < ESTIMATE_EVERY_S:
            return
        self.last_time = now

        for idx in range(3):
            values = np.asarray(self.magnitudes[idx], dtype=float)
            bpm, filtered, peaks, quality, note = estimate_hr(values)
            self.last_estimates[idx] = bpm
            self.last_filtered[idx] = filtered
            self.last_peaks[idx] = peaks
            self.last_quality[idx] = quality
            self.last_notes[idx] = note

    def plot_data(self, idx):
        values = np.asarray(self.magnitudes[idx], dtype=float)
        if len(values) == 0:
            return np.array([]), np.array([]), np.array([], dtype=int)

        # Pokazujemy ostatnie PLOT_SECONDS sekund.
        values = values[-self.plot_n:]
        n = len(values)
        x = np.arange(n) / float(FS)

        # Dla wykresu używamy tej samej metody co estymator.
        _, filtered, peaks, _, _ = estimate_hr(
            np.asarray(self.magnitudes[idx], dtype=float)
        )
        if len(filtered) >= n:
            filtered = filtered[-n:]
            peaks = peaks[peaks >= max(0, len(filtered) - n)]
        else:
            filtered = values
            peaks = np.array([], dtype=int)

        # Piki przeliczamy do indeksów widocznego fragmentu.
        full_len = len(self.magnitudes[idx])
        start_full = max(0, full_len - self.plot_n)
        peaks_visible = peaks[peaks >= start_full] - start_full
        peaks_visible = peaks_visible[(peaks_visible >= 0) & (peaks_visible < n)]
        return x, filtered, peaks_visible


def main():
    print(f"Łączenie z {PORT} @ {BAUD}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.01)
    except serial.SerialException as exc:
        print(f"Nie można otworzyć portu {PORT}: {exc}")
        print("Sprawdź PORT w pliku oraz czy Serial Monitor Arduino jest zamknięty.")
        return

    log_name = "log_paper_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    log = open(log_name, "w", buffering=1, encoding="utf-8")
    processor = PaperProcessor()
    buffer = ""

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(title="Clema — metoda z artykułu (moduł + LP 2 Hz + piki)")
    win.resize(1200, 900)
    win.show()

    plots = []
    curves = []
    peak_scatter = []
    status_labels = []

    for idx, name in enumerate(SENSOR_NAMES):
        plot = win.addPlot(row=idx, col=0, title=name)
        plot.setLabel("bottom", "Czas", units="s")
        plot.setLabel("left", "Przyspieszenie filtrowane", units="m/s²")
        plot.showGrid(x=True, y=True, alpha=0.25)
        curve = plot.plot(pen=pg.mkPen(width=1.5))
        scatter = pg.ScatterPlotItem(size=9, symbol="o", pen=pg.mkPen(width=1.5))
        plot.addItem(scatter)
        plots.append(plot)
        curves.append(curve)
        peak_scatter.append(scatter)

        label = QtWidgets.QLabel(f"{name}: czekam na dane")
        label.setStyleSheet("font-size: 14px; padding: 4px;")
        win.addItem(label, row=idx, col=1)
        status_labels.append(label)

    buf_status = QtWidgets.QLabel(f"Log: {log_name}")
    win.addItem(buf_status, row=3, col=0, colspan=2)

    last_estimate_time = 0.0

    def update():
        nonlocal buffer, last_estimate_time
        data = ser.read(8192)
        if data:
            decoded = data.decode("ascii", "ignore")
            log.write(decoded)
            buffer += decoded
            rows, buffer = parse_lines(buffer)
            if rows:
                processor.push(rows)

        now = time.monotonic()
        if now - last_estimate_time >= ESTIMATE_EVERY_S:
            processor.estimate(force=True)
            last_estimate_time = now

            for idx in range(3):
                x, filtered, peaks = processor.plot_data(idx)
                curves[idx].setData(x, filtered)

                if len(peaks):
                    peak_scatter[idx].setData(
                        x= x[peaks],
                        y= filtered[peaks],
                    )
                else:
                    peak_scatter[idx].setData([], [])

                bpm = processor.last_estimates[idx]
                quality = processor.last_quality[idx]
                note = processor.last_notes[idx]
                bpm_text = f"{bpm:.1f} BPM" if bpm is not None else "-- BPM"
                status_labels[idx].setText(
                    f"<b>{SENSOR_NAMES[idx]}</b><br>"
                    f"HR: {bpm_text} &nbsp; | &nbsp; "
                    f"jakość/regularność: {quality:.2f} &nbsp; | &nbsp; {note}"
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


if __name__ == "__main__":
    main()
