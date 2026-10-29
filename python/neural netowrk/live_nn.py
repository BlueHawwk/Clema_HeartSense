"""
live_nn.py — estymacja HR siecią neuronową (MLP) na podstawie 3 akcelerometrów.

Oczekiwany strumień firmware:
t,cx,cy,cz,dx,dy,dz,bx,by,bz

Kolejność domyślna:
C = palec, D = przód klatki piersiowej, B = kanał silikonowy.
Sprawdź, czy odpowiada kolejności w firmware.

Model nie jest dołączony: najpierw wytrenuj go skryptem train_nn.py na
własnych danych zsynchronizowanych z tętnem referencyjnym (np. Polar H10).
"""

import sys
import time
import datetime
from collections import deque
from pathlib import Path

import numpy as np
import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import butter, sosfiltfilt
import joblib


# ---------- Ustawienia ----------
PORT = "COM5"
BAUD = 921600
FS = 500
WINDOW_SECONDS = 8
PLOT_SECONDS = 5
MODEL_PATH = "hr_mlp.joblib"
ESTIMATE_EVERY_S = 1.0

SENSOR_NAMES = [
    "Palec (sensor 1)",
    "Przód klatki (sensor 2)",
    "Kanał silikonowy (sensor 3)",
]

# Zmiana pasma ogranicza składową stałą i bardzo wolne zmiany orientacji.
# Cechy FFT są liczone w paśmie 0.5–3 Hz (30–180 BPM).
BAND_LOW_HZ = 0.5
BAND_HIGH_HZ = 3.0
FFT_BINS = 32
sos = butter(4, [BAND_LOW_HZ, BAND_HIGH_HZ], btype="bandpass", fs=FS, output="sos")

WINDOW_N = int(WINDOW_SECONDS * FS)
PLOT_N = int(PLOT_SECONDS * FS)


def parse_lines(buf):
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
            rows.append([float(v) for v in parts])
        except ValueError:
            pass
    return rows, remainder


def feature_vector(window_rows):
    """
    Cechy dla jednego okna: dla każdego sensora moduł przyspieszenia,
    filtrowanie pasmowe, a następnie znormalizowane widmo FFT w 32 koszykach.
    Zwraca wektor 3*FFT_BINS cech.
    """
    arr = np.asarray(window_rows, dtype=float)
    xyz = arr[:, 1:].reshape(len(arr), 3, 3)
    feats = []

    for sensor in range(3):
        block = xyz[:, sensor, :]
        if not np.all(np.isfinite(block)) or np.max(np.abs(block)) < 1e-8:
            feats.extend([0.0] * FFT_BINS)
            continue

        mag = np.sqrt(np.sum(block * block, axis=1))
        mag = mag - np.mean(mag)

        try:
            filt = sosfiltfilt(sos, mag)
        except ValueError:
            feats.extend([0.0] * FFT_BINS)
            continue

        # Okno Hann ogranicza przeciekanie widma.
        spec = np.abs(np.fft.rfft(filt * np.hanning(len(filt))))
        freqs = np.fft.rfftfreq(len(filt), d=1.0 / FS)
        mask = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
        band = spec[mask]

        if len(band) == 0:
            feats.extend([0.0] * FFT_BINS)
            continue

        # Interpolacja widma do stałej liczby cech.
        old_x = np.linspace(0.0, 1.0, len(band))
        new_x = np.linspace(0.0, 1.0, FFT_BINS)
        band = np.interp(new_x, old_x, band)
        band = band / (np.linalg.norm(band) + 1e-12)
        feats.extend(band.tolist())

    return np.asarray(feats, dtype=float)


def main():
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        print(f"Brak modelu: {MODEL_PATH}")
        print("Najpierw przygotuj nagrania z etykietami HR i uruchom train_nn.py.")
        print("Model musi być wytrenowany na cechach generowanych przez ten sam kod.")
        return

    try:
        model = joblib.load(model_file)
    except Exception as exc:
        print(f"Nie udało się wczytać modelu {MODEL_PATH}: {exc}")
        return

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.01)
    except serial.SerialException as exc:
        print(f"Nie można otworzyć portu {PORT}: {exc}")
        print("Sprawdź PORT i zamknij Arduino Serial Monitor.")
        return

    log_name = "log_nn_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    log = open(log_name, "w", buffering=1, encoding="utf-8")
    rows_buffer = deque(maxlen=WINDOW_N)
    text_buffer = ""

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(title="Clema — estymacja HR siecią neuronową")
    win.resize(1100, 750)
    win.show()

    plot = win.addPlot(row=0, col=0, colspan=2, title="Moduł przyspieszenia — 3 sensory")
    plot.setLabel("bottom", "Czas w oknie", units="s")
    plot.setLabel("left", "Moduł przyspieszenia", units="m/s²")
    plot.showGrid(x=True, y=True, alpha=0.25)
    curves = [
        plot.plot(pen=pg.mkPen(width=1.5), name=SENSOR_NAMES[i])
        for i in range(3)
    ]
    status = QtWidgets.QLabel("Czekam na dane...")
    status.setStyleSheet("font-size: 16px; padding: 8px;")
    win.addItem(status, row=1, col=0, colspan=2)
    log_label = QtWidgets.QLabel(f"Surowy strumień zapisywany do: {log_name}")
    win.addItem(log_label, row=2, col=0, colspan=2)

    last_estimate = 0.0
    last_bpm = None
    last_features = None

    def update():
        nonlocal text_buffer, last_estimate, last_bpm, last_features
        data = ser.read(8192)
        if data:
            decoded = data.decode("ascii", "ignore")
            log.write(decoded)
            text_buffer += decoded
            rows, text_buffer = parse_lines(text_buffer)
            for row in rows:
                rows_buffer.append(row)

        if len(rows_buffer) < WINDOW_N:
            status.setText(f"Zbieranie danych: {len(rows_buffer)}/{WINDOW_N} próbek")
            return

        # Wykres odświeżany przy każdym wywołaniu; inferencja co sekundę.
        arr = np.asarray(rows_buffer, dtype=float)
        xyz = arr[:, 1:].reshape(len(arr), 3, 3)
        for i in range(3):
            mag = np.sqrt(np.sum(xyz[:, i, :] ** 2, axis=1))
            curves[i].setData(np.arange(len(mag)) / FS, mag)

        now = time.monotonic()
        if now - last_estimate >= ESTIMATE_EVERY_S:
            try:
                x = feature_vector(arr).reshape(1, -1)
                prediction = float(np.asarray(model.predict(x)).ravel()[0])
                if 30.0 <= prediction <= 200.0:
                    last_bpm = prediction
                    last_features = x
                last_estimate = now
            except Exception as exc:
                status.setText(f"Błąd inferencji modelu: {exc}")
                return

        if last_bpm is None:
            status.setText("Model zwrócił wynik poza zakresem 30–200 BPM.")
        else:
            status.setText(
                f"<b>HR z sieci: {last_bpm:.1f} BPM</b> — "
                f"okno {WINDOW_SECONDS} s. Wynik zależy od jakości i zakresu danych treningowych."
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
