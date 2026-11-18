"""
live_scg.py — estymacja HR z akcelerometrów metodą SCG + autokorelacja + tracker HMM.

Format strumienia:
t,cx,cy,cz,dx,dy,dz,bx,by,bz

Mapowanie sensorów jak w repo (README, firmware 3_sensors.ino):
C (kolumny cx,cy,cz) = klatka piersiowa / sygnał SCG
D (kolumny dx,dy,dz) = urządzenie / referencja ruchu
B (kolumny bx,by,bz) = plecy / referencja ruchu

UWAGA:
Jeżeli firmware ma inną kolejność sensorów, zmień CHEST_SENSOR_INDEX
i REF_SENSOR_INDICES poniżej. Indeksy: C=0, D=1, B=2.

Algorytm:
1. Filtr 5–25 Hz + RLS (Recursive Least Squares) usuwa ruch z osi sensora SCG,
   z pozostałymi dwoma akcelerometrami jako referencją. Wynik RLS służy TYLKO
   do wykresu RAW vs CLEAN.
2. HR liczone jest osobno, z pasma 20–60 Hz. Poniżej ~20 Hz dominują uderzenia
   stóp, a klik zastawek serca ma dużo energii wyżej.
3. Obwiednia energii (RMS 60 ms) i jej autokorelacja — dla KAŻDEJ z 9 osi
   (wszystkie trzy sensory widzą bicie serca), autokorelacje są uśredniane.
   Nie odejmujemy referencji od sygnału HR, bo referencje też zawierają serce.
4. Wynik autokorelacji dla każdego tempa 40–150 BPM:
   0.75 * ac(okres) + 0.25 * ac(2 * okres), jak w oryginalnym kandydacie.
5. Tracker HMM (filtr probabilistyczny): co sekundę rozkład prawdopodobieństwa
   tętna jest lekko rozmywany (tętno zmienia się powoli) i mnożony przez nowy
   wynik autokorelacji. Piki od kroków skaczą, serce stoi — więc serce wygrywa.

Wyniki na nagraniach z repo (raw_data, porównanie z Polar H10, odtwarzanie
tym plikiem):
  chodzenie: średni błąd ~3 BPM (wcześniej ~25 BPM — łapał rytm kroków ~50/min)
  siedzenie / stanie: ~1.5 BPM

To prototyp badawczy, nie urządzenie medyczne. Wynik BPM wymaga walidacji
z niezależnym pomiarem referencyjnym (np. ECG/pas piersiowy).
"""

import sys
import time
import datetime
from collections import deque

import numpy as np
import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import butter, sosfilt, sosfilt_zi


# ----------------------------- USTAWIENIA -----------------------------

PORT = "COM5"
BAUD = 921600
FS = 500

# Mapowanie sensorów w strumieniu: C=0, D=1, B=2 (jak w README i firmware).
CHEST_SENSOR_INDEX = 0
REF_SENSOR_INDICES = (1, 2)
SENSOR_NAMES = ("klatka (C)", "urządzenie (D)", "plecy (B)")

BPM_MIN = 40
BPM_MAX = 150

SCG_BAND = (5.0, 25.0)
MOTION_BAND = (0.8, 4.0)

WINDOW_SECONDS = 10
PLOT_SECONDS = 5
ESTIMATE_EVERY_S = 1.0

# RLS
TAPS = 6
LAMBDA = 0.998
RLS_INITIAL_P = 1e3

# Detekcja ruchu
MOVE_RMS = 0.03
MOVE_FACTOR = 2.5
MOTION_HOLD_SECONDS = 1.5

# Estymator HR
HR_BAND = (20.0, 60.0)        # pasmo do HR (nie 5–25 Hz: tam dominują kroki)
HR_WINDOW_SECONDS = 10        # okno autokorelacji
ENVELOPE_SECONDS = 0.06
ENVELOPE_FS = 100
TWICE_LAG_WEIGHT = 0.25       # waga ac(2 * okres)
TRACK_SIGMA_BPM = 1.0         # o ile BPM tętno może "dryfować" na sekundę
TRACK_BETA = 2.0              # jak mocno ufać pojedynczemu oknu (mniej = więcej ciągłości)
TRACK_MIX = 0.002             # mała szansa skoku w dowolne miejsce (wyjście ze złego zaczepienia)
LOCK_MASS = 0.5               # "zablokowany", gdy tyle prawdopodobieństwa jest w ±3 BPM
WARMUP_UPDATES = 5            # ile aktualizacji zanim wynik może być "zablokowany"

# Krok aktualizacji wykresu
UI_INTERVAL_MS = 30

WINDOW_N = int(WINDOW_SECONDS * FS)
HR_WINDOW_N = int(HR_WINDOW_SECONDS * FS)
PLOT_N = int(PLOT_SECONDS * FS)
HOLD_SAMPLES = int(MOTION_HOLD_SECONDS * FS)


# ----------------------------- FILTRY -----------------------------

SCG_SOS = butter(
    4, SCG_BAND, btype="bandpass", fs=FS, output="sos"
)
MOTION_SOS = butter(
    2, MOTION_BAND, btype="bandpass", fs=FS, output="sos"
)
HR_SOS = butter(
    4, HR_BAND, btype="bandpass", fs=FS, output="sos"
)


def moving_average(x, size):
    """Średnia ruchoma z odbiciem próbek na krawędziach."""
    x = np.asarray(x, dtype=float)
    size = max(int(size), 1)
    if size == 1 or len(x) < 2:
        return x.copy()
    size = min(size, len(x))
    left = size // 2
    right = size - 1 - left
    padded = np.pad(x, (left, right), mode="reflect")
    csum = np.cumsum(np.insert(padded, 0, 0.0))
    return (csum[size:] - csum[:-size]) / size


def init_filter_state(sos, x0):
    """Stan początkowy filtra dla stałego wejścia x0."""
    zi = sosfilt_zi(sos)
    return zi[:, :, None] * np.asarray(x0)[None, None, :]


# ----------------------------- PARSOWANIE -----------------------------

def parse_lines(buffer):
    """Parsuje pełne linie CSV; zachowuje niepełną końcową linię."""
    rows = []
    parts = buffer.split(b"\n")
    remainder = parts[-1]

    for line in parts[:-1]:
        line = line.strip()
        if not line or line.startswith(b"#"):
            continue
        fields = line.split(b",")
        if len(fields) != 10:
            continue
        try:
            rows.append([float(v) for v in fields])
        except ValueError:
            continue

    return rows, remainder


# ----------------------------- AUTOKORELACJA HR -----------------------------

def beat_autocorrelation(signal):
    """
    Oblicza autokorelację obwiedni energii SCG.
    Wejście: sygnał jednej osi, próbkowany z FS.
    Wynik: znormalizowana autokorelacja przy ENVELOPE_FS.
    """
    x = np.asarray(signal, dtype=float)
    if len(x) < 3 * FS or not np.all(np.isfinite(x)):
        return None

    # Ograniczenie pojedynczych impulsów, które mogłyby zdominować energię.
    scale = np.median(np.abs(x)) / 0.6745 + 1e-12
    x = np.clip(x, -4.0 * scale, 4.0 * scale)

    # Obwiednia energii: RMS w oknie około 60 ms.
    energy = np.sqrt(
        moving_average(x * x, max(1, int(ENVELOPE_SECONDS * FS)))
    )

    # 500 Hz -> 100 Hz.
    envelope = energy[:: max(1, int(FS / ENVELOPE_FS))]
    if len(envelope) < 10:
        return None

    # Ograniczenie wpływu pojedynczego dużego impulsu i usunięcie trendu.
    cap = 3.0 * np.median(envelope) + 1e-12
    envelope = np.minimum(envelope, cap)
    envelope -= moving_average(envelope, ENVELOPE_FS)

    n = len(envelope)
    spectrum = np.fft.rfft(envelope, n=2 * n)
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    ac /= (n - np.arange(n))
    ac /= (ac[0] + 1e-12)
    return ac


def autocorrelation_score(ac, bpm_grid):
    """
    Wynik autokorelacji dla każdego tempa z bpm_grid
    (interpolacja między próbkami opóźnienia):
    0.75 * ac(okres) + 0.25 * max(ac(2 * okres), 0).
    """
    lags = np.arange(len(ac))
    lag = 60.0 / bpm_grid * ENVELOPE_FS
    lag2 = 2.0 * lag
    s1 = np.interp(lag, lags, ac)
    s2 = np.where(
        lag2 < len(ac) - 1,
        np.interp(np.minimum(lag2, len(ac) - 1), lags, ac),
        0.0,
    )
    return (1.0 - TWICE_LAG_WEIGHT) * s1 + TWICE_LAG_WEIGHT * np.maximum(s2, 0.0)


# ----------------------------- KADENCJA -----------------------------

def estimate_cadence(low_refs):
    """
    Szacuje okres kroków z referencji w paśmie 0.8–4 Hz.
    low_refs ma kształt (liczba_próbek, 6).
    """
    if low_refs is None or low_refs.size == 0:
        return None, 0.0

    autocorrelations = []
    for channel in range(low_refs.shape[1]):
        v = low_refs[::5, channel]  # 500 -> 100 Hz
        if np.std(v) < 1e-8:
            continue

        v = v - np.mean(v)
        n = len(v)
        f = np.fft.rfft(v, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n]
        ac /= (n - np.arange(n))
        ac /= ac[0] + 1e-12
        autocorrelations.append(ac)

    if not autocorrelations:
        return None, 0.0

    ac_mean = np.mean(autocorrelations, axis=0)

    # 0.3–1.2 s odpowiada około 50–200 krokom/min.
    lo, hi = 30, min(120, len(ac_mean) - 1)
    if hi <= lo:
        return None, 0.0

    index = lo + int(np.argmax(ac_mean[lo:hi]))
    return index / 100.0, float(ac_mean[index])


# ----------------------------- TRACKER HR -----------------------------

class HRTracker:
    """
    Tracker HMM na siatce BPM. Co sekundę:
      1. rozmycie rozkładu (tętno może się zmienić o ~TRACK_SIGMA_BPM),
      2. mnożenie przez exp(TRACK_BETA * wynik_autokorelacji),
      3. wynik = maksimum rozkładu.
    Wymaga wywołania update() raz na sekundę.
    """

    def __init__(self):
        self.grid = np.arange(BPM_MIN, BPM_MAX + 0.25, 0.5)
        d = self.grid[:, None] - self.grid[None, :]
        T = np.exp(-0.5 * (d / TRACK_SIGMA_BPM) ** 2)
        T /= T.sum(axis=0, keepdims=True)
        self.transition = (1.0 - TRACK_MIX) * T + TRACK_MIX / len(self.grid)
        self.reset()

    def reset(self):
        self.p = np.ones(len(self.grid)) / len(self.grid)
        self.updates = 0
        self.hr = None
        self.confidence = 0.0

    def update(self, score):
        if score is None:
            return
        self.p = self.transition @ self.p
        self.p *= np.exp(TRACK_BETA * (score - np.max(score)))
        self.p /= self.p.sum()
        self.updates += 1
        i = int(np.argmax(self.p))
        self.hr = float(self.grid[i])
        self.confidence = float(self.p[np.abs(self.grid - self.hr) <= 3.0].sum())

    def state(self):
        if self.hr is None:
            return None, "unreliable", 0.0
        if self.updates >= WARMUP_UPDATES and self.confidence >= LOCK_MASS:
            return self.hr, "locked", self.confidence
        return self.hr, "guess", self.confidence


def format_hr(state):
    hr, status, confidence = state
    if hr is None:
        return "--"
    if status == "guess":
        return f"~{hr:.0f} BPM (niska pewność {confidence:.2f})"
    if status == "holding":
        return f"{hr:.0f} BPM (utrzymany)"
    return f"{hr:.0f} BPM (zablokowany, {confidence:.2f})"


# ----------------------------- PROCESSOR -----------------------------

class SCGProcessor:
    def __init__(self):
        self.rows = deque(maxlen=WINDOW_N)

        self.started = False
        self.scg_zi = None
        self.motion_zi = None

        self.last_raw = np.zeros(9, dtype=float)
        self.sensor_alive = np.zeros(3, dtype=bool)

        self.ref_count = 6
        self.rls_dim = self.ref_count * TAPS
        self.P = np.eye(self.rls_dim) * RLS_INITIAL_P
        self.W = np.zeros((3, self.rls_dim), dtype=float)
        self.ref_history = np.zeros((TAPS, self.ref_count), dtype=float)

        self.motion_energy = 0.0
        self.motion_baseline = deque(maxlen=600)
        self.tick = 0
        self.motion_hold = 0
        self.moving = False

        self.last_raw_scg = np.empty((0, 3))
        self.last_clean_scg = np.empty((0, 3))
        self.last_refs = np.empty((0, 6))
        self.last_moving = np.zeros(0, dtype=bool)
        self.last_energy = np.zeros(0)
        self.last_threshold = np.zeros(0)
        self.last_low_refs = np.empty((0, 6))

        self.raw_buffer = np.zeros((WINDOW_N, 3))
        self.clean_buffer = np.zeros((WINDOW_N, 3))
        self.low_buffer = np.zeros((WINDOW_N, 6))
        self.filled = 0

        self.hr_zi = None
        self.hr_buffer = np.zeros((HR_WINDOW_N, 9))   # 20–60 Hz, wszystkie 9 osi
        self.hr_filled = 0
        self.tracker = HRTracker()
        self.cadence = (None, 0.0)

        self.last_estimate_time = 0.0
        self.hr_state = (None, "unreliable", 0.0)

    def push(self, rows):
        """Przetwarza paczkę wierszy; zwraca sygnały RAW i CLEAN."""
        x = np.asarray(rows, dtype=float)
        if x.ndim != 2 or x.shape[1] != 10 or len(x) == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        # Kolumny 1..9 to XYZ trzech sensorów; kolumna 0 to timestamp.
        xyz = x[:, 1:10].copy()

        # Zachowaj ostatnią poprawną próbkę, gdy sensor chwilowo wysyła same zera.
        for i in range(len(xyz)):
            for sensor in range(3):
                sl = slice(sensor * 3, sensor * 3 + 3)
                if np.any(xyz[i, sl] != 0):
                    self.last_raw[sl] = xyz[i, sl]
                    self.sensor_alive[sensor] = True
                else:
                    xyz[i, sl] = self.last_raw[sl]

        for sensor in range(3):
            if not np.any(self.last_raw[sensor * 3:sensor * 3 + 3]):
                self.sensor_alive[sensor] = False

        if not self.started:
            self.scg_zi = init_filter_state(SCG_SOS, xyz[0])
            ref_cols0 = [c for s in REF_SENSOR_INDICES for c in range(3 * s, 3 * s + 3)]
            self.motion_zi = init_filter_state(MOTION_SOS, xyz[0, ref_cols0])
            self.hr_zi = init_filter_state(HR_SOS, xyz[0])
            self.started = True

        # Filtr SCG 5–25 Hz na wszystkich kanałach.
        hp, self.scg_zi = sosfilt(
            SCG_SOS, xyz, axis=0, zi=self.scg_zi
        )

        # Pasmo 20–60 Hz na wszystkich 9 osiach — z tego liczone jest HR.
        hr_band, self.hr_zi = sosfilt(HR_SOS, xyz, axis=0, zi=self.hr_zi)
        m = min(len(hr_band), HR_WINDOW_N)
        self.hr_buffer = np.roll(self.hr_buffer, -m, axis=0)
        self.hr_buffer[-m:] = hr_band[-m:]
        self.hr_filled = min(self.hr_filled + len(hr_band), HR_WINDOW_N)

        # Filtr niskopasmowy referencji do wykrywania kadencji.
        ref_columns = []
        for sensor_index in REF_SENSOR_INDICES:
            ref_columns.extend(range(sensor_index * 3, sensor_index * 3 + 3))
        ref_raw = xyz[:, ref_columns]
        low, self.motion_zi = sosfilt(
            MOTION_SOS, ref_raw, axis=0, zi=self.motion_zi
        )

        chest_columns = list(
            range(CHEST_SENSOR_INDEX * 3, CHEST_SENSOR_INDEX * 3 + 3)
        )
        chest = hp[:, chest_columns]
        refs = hp[:, ref_columns]

        # Wyłącz referencję, jeśli dany sensor nie dostarczył żadnych danych.
        ref_alive = np.array(
            [self.sensor_alive[idx] for idx in REF_SENSOR_INDICES],
            dtype=bool
        )
        ref_mask = np.repeat(ref_alive, 3).astype(float)
        refs = refs * ref_mask[None, :]
        low = low * ref_mask[None, :]

        clean = np.empty_like(chest)
        moving_flags = np.zeros(len(chest), dtype=bool)
        energy_trace = np.zeros(len(chest))
        threshold_trace = np.zeros(len(chest))
        alpha = 1.0 / (0.2 * FS)

        for i in range(len(chest)):
            r = refs[i]
            self.ref_history = np.roll(self.ref_history, 1, axis=0)
            self.ref_history[0] = r
            u = self.ref_history.ravel()

            active_channels = max(int(np.sum(ref_mask)), 1)
            instantaneous_energy = float(r @ r) / active_channels
            self.motion_energy += alpha * (
                instantaneous_energy - self.motion_energy
            )

            self.tick += 1
            if self.tick % 50 == 0:
                self.motion_baseline.append(self.motion_energy)

            quiet_level = (
                float(np.percentile(self.motion_baseline, 10))
                if len(self.motion_baseline) > 100 else 0.0
            )
            threshold = max(
                MOVE_RMS ** 2,
                (MOVE_FACTOR ** 2) * quiet_level
            )

            if self.motion_energy > threshold and np.any(ref_mask):
                self.motion_hold = HOLD_SAMPLES
            elif self.motion_hold > 0:
                self.motion_hold -= 1

            self.moving = self.motion_hold > 0
            moving_flags[i] = self.moving
            energy_trace[i] = self.motion_energy
            threshold_trace[i] = threshold

            y = chest[i]

            if self.moving and np.any(ref_mask):
                Pu = self.P @ u
                denominator = LAMBDA + float(u @ Pu)
                k = Pu / max(denominator, 1e-12)

                error = y - self.W @ u
                self.W += np.outer(error, k)
                self.P = (self.P - np.outer(k, Pu)) / LAMBDA
                self.P = 0.5 * (self.P + self.P.T)

                trace = float(np.trace(self.P))
                if trace > 1e5:
                    self.P *= 1e5 / trace

                clean[i] = error
            else:
                # Gdy użytkownik jest nieruchomy, zamrażamy adaptację RLS.
                clean[i] = y

        self.last_raw_scg = chest
        self.last_clean_scg = clean
        self.last_refs = refs
        self.last_low_refs = low
        # Dla live_debug.py: energia ruchu i próg dla każdej próbki tej paczki.
        self.last_moving = moving_flags
        self.last_energy = energy_trace
        self.last_threshold = threshold_trace
        self._append(chest, clean, low)

        return chest, clean

    def _append(self, raw, clean, low):
        n = len(raw)
        if n >= WINDOW_N:
            raw, clean, low = raw[-WINDOW_N:], clean[-WINDOW_N:], low[-WINDOW_N:]
            n = WINDOW_N

        self.raw_buffer = np.roll(self.raw_buffer, -n, axis=0)
        self.clean_buffer = np.roll(self.clean_buffer, -n, axis=0)
        self.low_buffer = np.roll(self.low_buffer, -n, axis=0)

        self.raw_buffer[-n:] = raw
        self.clean_buffer[-n:] = clean
        self.low_buffer[-n:] = low
        self.filled = min(self.filled + n, WINDOW_N)

    def _hr_score(self):
        """Uśredniona autokorelacja obwiedni z osi działających sensorów -> wynik na siatce BPM."""
        autocorrelations = []
        for sensor in range(3):
            if not self.sensor_alive[sensor]:
                continue
            for axis in range(3):
                ac = beat_autocorrelation(self.hr_buffer[:, sensor * 3 + axis])
                if ac is not None:
                    autocorrelations.append(ac)
        if not autocorrelations:
            return None
        return autocorrelation_score(np.mean(autocorrelations, axis=0), self.tracker.grid)

    def estimate(self, force=False):
        """
        Wołać raz na sekundę (tracker zakłada 1 s między aktualizacjami).
        Zwraca (bpm albo None, "locked"/"guess"/"unreliable", pewność) albo None,
        gdy jeszcze za mało danych.
        """
        now = time.monotonic()
        if not force and now - self.last_estimate_time < ESTIMATE_EVERY_S:
            return None

        self.last_estimate_time = now
        if self.hr_filled < HR_WINDOW_N:
            return None

        if any(self.sensor_alive[idx] for idx in REF_SENSOR_INDICES) and self.filled >= 6 * FS:
            self.cadence = estimate_cadence(self.low_buffer[-self.filled:])
        else:
            self.cadence = (None, 0.0)

        self.tracker.update(self._hr_score())
        self.hr_state = self.tracker.state()
        return self.hr_state

    def plot_data(self):
        if self.filled == 0:
            return np.array([]), np.array([]), np.array([])

        raw = self.raw_buffer[-self.filled:, 2]
        clean = self.clean_buffer[-self.filled:, 2]

        raw = raw[-PLOT_N:]
        clean = clean[-PLOT_N:]
        x = np.arange(len(raw)) / float(FS)
        return x, raw, clean

    def hr_plot_data(self):
        """
        Sygnał, z którego naprawdę liczone jest HR: oś Z klatki w paśmie 20–60 Hz
        oraz jego obwiednia energii (RMS 60 ms) — ostatnie PLOT_SECONDS sekund.
        """
        n = min(self.hr_filled, PLOT_N)
        if n == 0:
            return np.array([]), np.array([]), np.array([])
        signal = self.hr_buffer[-n:, CHEST_SENSOR_INDEX * 3 + 2]
        envelope = np.sqrt(moving_average(signal * signal, int(ENVELOPE_SECONDS * FS)))
        x = np.arange(n) / float(FS)
        return x, signal, envelope


# ----------------------------- REPLAY LOGU -----------------------------

def load_log(path):
    """
    Wczytuje surowy log z portu (t,cx,...,bz, bez nagłówka; linie '#' pomijane)
    albo nagranie z nagłówkiem (t_us/t_s, cx..bz, opcjonalnie polar_bpm).
    Zwraca (dane (n, 10), polar albo None).
    """
    import pandas as pd

    first = ""
    with open(path) as f:
        for first in f:
            if first.strip() and not first.startswith("#"):
                break

    if first.split(",")[0].strip().replace("_", "").isalpha():
        cols = ["cx", "cy", "cz", "dx", "dy", "dz", "bx", "by", "bz"]
        d = pd.read_csv(path, low_memory=False)
        for c in cols + ["polar_bpm"]:
            if c in d:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=cols)
        t = d["t_us"].values if "t_us" in d else np.arange(len(d)) * 1e6 / FS
        data = np.column_stack([t, d[cols].values]).astype(float)
        polar = None
        if "polar_bpm" in d and d["polar_bpm"].notna().any():
            polar = d["polar_bpm"].ffill().values
        return data, polar

    with open(path, "rb") as f:
        rows, _ = parse_lines(f.read() + b"\n")
    return np.array(rows, dtype=float), None


def replay_log(path):
    data, polar = load_log(path)
    if data.ndim != 2 or data.shape[1] != 10:
        print("Log musi zawierać 10 kolumn: timestamp + 9 osi.")
        return

    processor = SCGProcessor()
    chunk_size = FS
    errors = []

    for start in range(0, len(data) - chunk_size + 1, chunk_size):
        processor.push(data[start:start + chunk_size])
        state = processor.estimate(force=True)
        if state:
            line = (
                f"{(start + chunk_size) / FS:6.1f}s | "
                f"HR: {format_hr(state):32s} | "
                f"{'RUCH  ' if processor.moving else 'SPOKÓJ'}"
            )
            if polar is not None:
                ref = polar[start + chunk_size - 1]
                line += f" | Polar: {ref:4.0f}"
                if state[0] is not None:
                    errors.append((abs(state[0] - ref), state[1] == "locked"))
            print(line)

    if errors:
        e = np.array([x[0] for x in errors])
        locked = np.array([x[1] for x in errors])
        print(
            f"\nPorównanie z Polarem: średni błąd {e.mean():.1f} BPM, "
            f"mediana {np.median(e):.1f} BPM, w ±5 BPM {100 * np.mean(e <= 5):.0f}% sekund, "
            f"zablokowany {100 * locked.mean():.0f}% czasu"
        )


# ----------------------------- APLIKACJA LIVE -----------------------------

def main():
    print(f"Łączenie z {PORT} @ {BAUD}...")
    print(
        f"Sensor SCG: {SENSOR_NAMES[CHEST_SENSOR_INDEX]} | "
        f"referencje: {', '.join(SENSOR_NAMES[i] for i in REF_SENSOR_INDICES)}"
    )

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.01)
    except serial.SerialException as exc:
        print(f"Nie można otworzyć {PORT}: {exc}")
        print("Sprawdź PORT i zamknij Arduino Serial Monitor.")
        return

    log_name = (
        "log_scg_" +
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") +
        ".csv"
    )
    log = open(log_name, "w", buffering=1, encoding="utf-8")

    processor = SCGProcessor()
    byte_buffer = b""

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(title="Clema — SCG: sygnał i HR")
    win.resize(1200, 900)
    win.show()

    header = win.addLabel("Oczekiwanie na dane...", row=0, col=0, colspan=2)
    plot = win.addPlot(
        row=1, col=0, colspan=2,
        title=f"1  {SENSOR_NAMES[CHEST_SENSOR_INDEX]}, oś Z, 5–25 Hz: RAW i CLEAN po RLS (tylko podgląd)"
    )
    plot.setLabel("bottom", "Czas w oknie", units="s")
    plot.setLabel("left", "Przyspieszenie filtrowane", units="m/s²")
    plot.showGrid(x=True, y=True, alpha=0.25)

    raw_curve = plot.plot(pen=pg.mkPen((220, 180, 0), width=1))
    clean_curve = plot.plot(pen=pg.mkPen((0, 220, 100), width=1.5))
    plot.addLegend()
    raw_curve.setPen(pg.mkPen((220, 180, 0), width=1), name="RAW")
    clean_curve.setPen(pg.mkPen((0, 220, 100), width=1.5), name="CLEAN (RLS)")

    hr_plot = win.addPlot(
        row=2, col=0, colspan=2,
        title=f"2  Z tego liczone jest HR: {SENSOR_NAMES[CHEST_SENSOR_INDEX]}, oś Z, 20–60 Hz "
              f"(szary) i obwiednia energii (czerwona) — HR to rytm garbów obwiedni"
    )
    hr_plot.setLabel("bottom", "Czas w oknie", units="s")
    hr_plot.setLabel("left", "Przyspieszenie filtrowane", units="m/s²")
    hr_plot.showGrid(x=True, y=True, alpha=0.25)
    hr_signal_curve = hr_plot.plot(pen=pg.mkPen((150, 150, 150), width=1))
    hr_envelope_curve = hr_plot.plot(pen=pg.mkPen((255, 70, 70), width=2))

    details = win.addLabel("", row=3, col=0, colspan=2)
    log_label = win.addLabel(f"Log: {log_name}", row=4, col=0, colspan=2)

    rate_count = 0
    rate_start = time.monotonic()
    sample_rate = 0.0

    def update():
        nonlocal byte_buffer, rate_count, rate_start, sample_rate

        incoming = ser.read(8192)
        if incoming:
            log.write(incoming.decode("ascii", "ignore"))
            byte_buffer += incoming

        rows, byte_buffer = parse_lines(byte_buffer)
        if rows:
            processor.push(rows)
            rate_count += len(rows)

        now = time.monotonic()
        if now - rate_start >= 1.0:
            sample_rate = rate_count / (now - rate_start)
            rate_count = 0
            rate_start = now

        result = processor.estimate()

        x, raw, clean = processor.plot_data()
        if len(x):
            raw_curve.setData(x, raw)
            clean_curve.setData(x, clean)

        hx, hr_signal, hr_envelope = processor.hr_plot_data()
        if len(hx):
            hr_signal_curve.setData(hx, hr_signal)
            hr_envelope_curve.setData(hx, hr_envelope)

        if result is None:
            if processor.hr_filled < HR_WINDOW_N:
                header.setText(
                    f"Zbieranie danych: {processor.hr_filled}/{HR_WINDOW_N} próbek"
                )
        else:

            cadence_period, cadence_strength = processor.cadence
            cadence_text = (
                f"{60.0 / cadence_period:.0f} kroków/min"
                if cadence_period else "--"
            )

            header.setText(f"HR: {format_hr(result)}")

            details.setText(
                f"Stan: {'RUCH' if processor.moving else 'SPOKÓJ'} | "
                f"Kadencja: {cadence_text} (siła {cadence_strength:.2f}) | "
                + "".join(
                    f"{SENSOR_NAMES[i]}: {'ON' if processor.sensor_alive[i] else 'OFF'} | "
                    for i in range(3)
                ) +
                f"Transmisja: {sample_rate:.0f} próbek/s"
            )

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(UI_INTERVAL_MS)

    def close_event(event):
        timer.stop()
        try:
            ser.close()
        finally:
            log.close()
        event.accept()

    win.closeEvent = close_event
    sys.exit(app.exec())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        replay_log(sys.argv[1])
    else:
        main()
