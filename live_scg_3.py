"""
live_scg_en.py - live heart rate from the 4-sensor ESP32 (4_sensors_csv.ino): chest SCG + autocorrelation + probabilistic tracker.

Run live:   py live_scg_en.py COM7            (chest-2 is the chest sensor)
            py live_scg_en.py COM7 c1         (use chest-1 instead)
Replay:     py live_scg_en.py recording.csv   (prints one line per second and an error summary if the file has polar_bpm)

Stream format (4_sensors_csv.ino, 500 lines/s):  t_us,c1x,c1y,c1z,c2x,c2y,c2z,dx,dy,dz,bx,by,bz
   chest = chest-1 or chest-2 (your choice), device (d) and back (b) = motion references.

HOW IT WORKS
1. 5-25 Hz band-pass + RLS (recursive least squares) removes motion from the chest signal, using the device and back
   sensors as references. This result is ONLY for the top plot (RAW vs CLEAN); it is not used for the heart rate.
2. The heart rate is measured separately in the 20-60 Hz band. Below ~20 Hz the footstep impacts dominate,
   while the heart valve clicks have a lot of energy higher up.
3. Energy envelope (60 ms RMS) and its autocorrelation, for EVERY one of the 9 axes (all three sensors feel the
   heartbeat), averaged. The references are not subtracted, because they contain the heartbeat too.
4. Score for each rate 40-150 bpm:  0.75 * ac(period) + 0.25 * ac(2 * period).
5. Probabilistic tracker (HMM): once a second the probability distribution of the heart rate is blurred slightly
   (the rate changes slowly) and multiplied by the new score. Step peaks jump around, the heart is steady,
   so the heart wins.
6. Improvements over the first version (see settings below):
   - the walking rhythm (step rate, half and double) is penalised in the score,
   - the tracker starts from a prior around normal heart rates instead of a flat distribution,
   - everything restarts after a gap in the data (rows lost), because the signal is no longer continuous.

import sys
import time
import traceback
import datetime
from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


# ----------------------------- SETTINGS -----------------------------

BAUD = 921600
FS = 500
NCOL = 13                     # t_us + 4 sensors x XYZ (what the ESP32 sends)
MAX_ABS = 41.0                # m/s^2: anything bigger is a corrupted line (range is +-2 g = 19.6, or +-4 g = 39.2)

# which sensor is the chest sensor: first column index (after t_us) of its XYZ in the 13-column stream
CHEST_COLS = {"c1": 0, "c2": 3}
DEVICE_COLS = 6
BACK_COLS = 9

# Sensors used: NSENS = 4 uses all four (the chosen chest sensor, the device, the back and the OTHER chest sensor),
# NSENS = 3 ignores the other chest sensor. Order inside the processor: chest = 0, device = 1, back = 2, other chest = 3.
# The motion references (RLS preview and step rate) are all sensors except the chest sensor;
# the heart rate is computed from the energy envelope of every axis of every sensor.
NSENS = 4
SENSOR_NAMES = ("chest", "device", "back", "other chest")
CHEST_SENSOR_INDEX = 0
OTHER_CHEST_COLS = {"c1": 3, "c2": 0}           # the chest sensor that is not selected


def ref_sensors():
    return tuple(range(1, NSENS))

BPM_MIN = 40
BPM_MAX = 150

SCG_BAND = (5.0, 25.0)        # for the RAW/CLEAN preview and the motion detection
MOTION_BAND = (0.8, 4.0)      # for the step-rate (cadence) measurement
WINDOW_SECONDS = 10
PLOT_SECONDS = 5
ESTIMATE_EVERY_S = 1.0

# RLS
TAPS = 6
LAMBDA = 0.998
RLS_INITIAL_P = 1e3

# motion detection
MOVE_RMS = 0.03
MOVE_FACTOR = 2.5
MOTION_HOLD_SECONDS = 1.5

# heart-rate estimator
HR_BAND = (20.0, 60.0)        # band for the heart rate (not 5-25 Hz: footsteps dominate there)
HR_WINDOW_SECONDS = 10        # autocorrelation window
ENVELOPE_SECONDS = 0.06
ENVELOPE_FS = 100
TWICE_LAG_WEIGHT = 0.25       # weight of ac(2 * period)
TRACK_SIGMA_BPM = 1.0         # how far (bpm) the rate may drift per second
TRACK_BETA = 2.0              # how much to trust one window (lower = more continuity)
TRACK_MIX = 0.002             # small chance of a jump anywhere (escape from a wrong lock)
LOCK_MASS = 0.5               # "locked" when this much probability is within +-3 bpm
WARMUP_UPDATES = 5            # updates before a result may be called "locked"

# improvements, tested on your walking and sitting recordings (set CADENCE_PENALTY = 0, PRIOR_MEAN = None,
# QUALITY_MIN_SCORE = 0 to get the original behaviour back; QUALITY_MIN_SCORE 0.08 shows more numbers, 0.13 fewer but more accurate)
CADENCE_PENALTY = 0.3         # subtract this from the score near the step rate (and half / double); 0 = off
CADENCE_MIN_STRENGTH = 0.3    # the step rhythm must be at least this clear (0..1) to be used
CADENCE_TOL = 0.06            # +-6 % around the step rate
PRIOR_MEAN = 80               # e.g. 80: start the tracker from a bell around this rate instead of flat
PRIOR_SD = 25.0
QUALITY_MIN_SCORE = 0.10      # if the best autocorrelation score (average of the last 5 s) is below this there is no clear rhythm
                              # and the new reading is not trusted; 0 = always trust the tracker's best guess
HOLD_LAST_GOOD = True         # when the reading is not trusted, keep showing the LAST GOOD reading (marked "holding")
                              # instead of "--", so there is never a gap; False = show "--" (status "unreliable")

UI_INTERVAL_MS = 30
LABELS = {"1": "sitting", "2": "standing", "3": "walk_slow", "4": "walk_fast", "5": "arm_swing",
          "6": "turning", "7": "stairs", "8": "carrying", "9": "talking", "0": "other"}

WINDOW_N = int(WINDOW_SECONDS * FS)
HR_WINDOW_N = int(HR_WINDOW_SECONDS * FS)
PLOT_N = int(PLOT_SECONDS * FS)
HOLD_SAMPLES = int(MOTION_HOLD_SECONDS * FS)


# ----------------------------- FILTERS -----------------------------

SCG_SOS = butter(4, SCG_BAND, btype="bandpass", fs=FS, output="sos")
MOTION_SOS = butter(2, MOTION_BAND, btype="bandpass", fs=FS, output="sos")
HR_SOS = butter(4, HR_BAND, btype="bandpass", fs=FS, output="sos")


def moving_average(x, size):
    """Moving average with reflected edges."""
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
    """Initial filter state for a constant input x0 (no start-up transient)."""
    zi = sosfilt_zi(sos)
    return zi[:, :, None] * np.asarray(x0)[None, None, :]


# ----------------------------- INPUT PARSING -----------------------------

class Feed:
    """Turns serial bytes into valid rows and measures real elapsed time from the ESP timestamps.
    A row is (t_us, sec, raw 12 values, row, gap) where row = [t, chest xyz, device xyz, back xyz, other chest xyz]."""

    def __init__(self, chest="c2"):
        self.chest = CHEST_COLS[chest]
        self.other = OTHER_CHEST_COLS[chest]
        self.buf = b""
        self.last_t = None
        self.sec = 0.0
        self.lost_s = 0.0
        self.bad = 0
        self.gap = False                 # set to True when rows were lost; the caller resets the filter and clears it

    def parse(self, data):
        self.buf += data
        parts = self.buf.split(b"\n")
        self.buf = parts[-1]
        out = []
        for p in parts[:-1]:
            p = p.strip()
            if not p or p[:1] == b"#" or p.startswith(b"t_us"):
                continue
            f = p.split(b",")
            if len(f) < NCOL:
                self.bad += 1
                continue
            try:
                t = int(f[0])
                v = [float(x) for x in f[1:NCOL]]
            except ValueError:
                self.bad += 1                    # garbled line (a byte was lost on the USB link)
                continue
            if max(abs(x) for x in v) > MAX_ABS:
                self.bad += 1
                continue
            if self.last_t is None:
                dt = 0.0
            else:
                dt = (t - self.last_t) / 1e6
                if 0 < dt <= 0.1:
                    pass
                elif 0.1 < dt < 3600:            # a real hole: rows were lost (e.g. the console was clicked)
                    self.lost_s += dt
                    self.gap = True
                else:                            # garbled or repeated timestamp: assume the normal 2 ms
                    dt = 0.002
                    t = self.last_t + 2000
            self.last_t = t
            self.sec += dt
            c = self.chest
            o = self.other
            row = [t] + v[c:c + 3] + v[DEVICE_COLS:DEVICE_COLS + 3] + v[BACK_COLS:BACK_COLS + 3] + v[o:o + 3]
            out.append((t, self.sec, v, row[:1 + 3 * NSENS], self.gap))
            self.gap = False
        return out


# ----------------------------- HEART-RATE AUTOCORRELATION -----------------------------

def beat_autocorrelation(signal):
    """Autocorrelation of the SCG energy envelope. Input: one axis sampled at FS. Output: normalised autocorrelation at ENVELOPE_FS."""
    x = np.asarray(signal, dtype=float)
    if len(x) < 3 * FS or not np.all(np.isfinite(x)):
        return None
    scale = np.median(np.abs(x)) / 0.6745 + 1e-12            # limit single shocks that could dominate the energy
    x = np.clip(x, -4.0 * scale, 4.0 * scale)
    energy = np.sqrt(moving_average(x * x, max(1, int(ENVELOPE_SECONDS * FS))))
    envelope = energy[:: max(1, int(FS / ENVELOPE_FS))]      # 500 Hz -> 100 Hz
    if len(envelope) < 10:
        return None
    cap = 3.0 * np.median(envelope) + 1e-12                  # one big burst cannot dominate; remove the trend
    envelope = np.minimum(envelope, cap)
    envelope -= moving_average(envelope, ENVELOPE_FS)
    n = len(envelope)
    spectrum = np.fft.rfft(envelope, n=2 * n)
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    ac /= (n - np.arange(n))
    ac /= (ac[0] + 1e-12)
    return ac


def autocorrelation_score(ac, bpm_grid):
    """Score for every rate in bpm_grid: 0.75 * ac(period) + 0.25 * max(ac(2 * period), 0)."""
    lags = np.arange(len(ac))
    lag = 60.0 / bpm_grid * ENVELOPE_FS
    lag2 = 2.0 * lag
    s1 = np.interp(lag, lags, ac)
    s2 = np.where(lag2 < len(ac) - 1, np.interp(np.minimum(lag2, len(ac) - 1), lags, ac), 0.0)
    return (1.0 - TWICE_LAG_WEIGHT) * s1 + TWICE_LAG_WEIGHT * np.maximum(s2, 0.0)


# ----------------------------- CADENCE (STEP RATE) -----------------------------

def estimate_cadence(low_refs):
    """Step period from the references in the 0.8-4 Hz band. low_refs has shape (samples, 6). Returns (period_s, strength)."""
    if low_refs is None or low_refs.size == 0:
        return None, 0.0
    autocorrelations = []
    for channel in range(low_refs.shape[1]):
        v = low_refs[::5, channel]                           # 500 -> 100 Hz
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
    lo, hi = 30, min(120, len(ac_mean) - 1)                  # 0.3-1.2 s = about 50-200 steps/min
    if hi <= lo:
        return None, 0.0
    index = lo + int(np.argmax(ac_mean[lo:hi]))
    return index / 100.0, float(ac_mean[index])


# ----------------------------- HEART-RATE TRACKER -----------------------------

class HRTracker:
    """HMM tracker on a bpm grid. Once a second:
      1. blur the distribution (the rate can change by ~TRACK_SIGMA_BPM),
      2. multiply by exp(TRACK_BETA * autocorrelation score),
      3. result = maximum of the distribution.
    update() must be called once per second."""

    def __init__(self):
        self.grid = np.arange(BPM_MIN, BPM_MAX + 0.25, 0.5)
        d = self.grid[:, None] - self.grid[None, :]
        T = np.exp(-0.5 * (d / TRACK_SIGMA_BPM) ** 2)
        T /= T.sum(axis=0, keepdims=True)
        self.transition = (1.0 - TRACK_MIX) * T + TRACK_MIX / len(self.grid)
        self.reset()

    def reset(self):
        if PRIOR_MEAN is None:
            self.p = np.ones(len(self.grid)) / len(self.grid)
        else:
            self.p = np.exp(-0.5 * ((self.grid - PRIOR_MEAN) / PRIOR_SD) ** 2)
            self.p /= self.p.sum()
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


def format_hr(state, hold_age=None):
    hr, status, confidence = state
    if hr is None:
        return "--"
    if status == "holding":
        return f"{hr:.0f} bpm (holding last good reading" + (f", {hold_age} s old)" if hold_age is not None else ")")
    if status == "guess":
        return f"~{hr:.0f} bpm (low confidence {confidence:.2f})"
    return f"{hr:.0f} bpm (locked, {confidence:.2f})"


# ----------------------------- PROCESSOR -----------------------------

class SCGProcessor:
    def __init__(self):
        self.rows = deque(maxlen=WINDOW_N)
        self.started = False
        self.scg_zi = None
        self.motion_zi = None
        self.hr_zi = None
        self.last_raw = np.zeros(3 * NSENS, dtype=float)
        self.sensor_alive = np.zeros(NSENS, dtype=bool)

        self.ref_count = 3 * len(ref_sensors())
        self.rls_dim = self.ref_count * TAPS
        self.P = np.eye(self.rls_dim) * RLS_INITIAL_P
        self.W = np.zeros((3, self.rls_dim), dtype=float)
        self.ref_history = np.zeros((TAPS, self.ref_count), dtype=float)

        self.motion_energy = 0.0
        self.motion_baseline = deque(maxlen=600)
        self.tick = 0
        self.motion_hold = 0
        self.moving = False

        self.raw_buffer = np.zeros((WINDOW_N, 3))
        self.clean_buffer = np.zeros((WINDOW_N, 3))
        self.low_buffer = np.zeros((WINDOW_N, self.ref_count))
        self.filled = 0

        self.hr_buffer = np.zeros((HR_WINDOW_N, 3 * NSENS))      # 20-60 Hz, all axes of all sensors in use
        self.hr_filled = 0
        self.tracker = HRTracker()
        self.cadence = (None, 0.0)
        self.quality = deque(maxlen=5)
        self.last_good = None                            # last trusted heart rate
        self.hold_age = 0                                # seconds since it was measured
        self.last_estimate_time = 0.0
        self.hr_state = (None, "unreliable", 0.0)

    def push(self, rows):
        """Processes a block of rows (n, 1 + 3 * NSENS) = [t, chest xyz, device xyz, back xyz, (other chest xyz)].
        Returns RAW and CLEAN chest signals (5-25 Hz)."""
        x = np.asarray(rows, dtype=float)
        if x.ndim != 2 or x.shape[1] != 1 + 3 * NSENS or len(x) == 0:
            return np.empty((0, 3)), np.empty((0, 3))
        xyz = x[:, 1:1 + 3 * NSENS].copy()

        for i in range(len(xyz)):                        # keep the last good value if a sensor sends all zeros
            for sensor in range(NSENS):
                sl = slice(sensor * 3, sensor * 3 + 3)
                if np.any(xyz[i, sl] != 0):
                    self.last_raw[sl] = xyz[i, sl]
                    self.sensor_alive[sensor] = True
                else:
                    xyz[i, sl] = self.last_raw[sl]
        for sensor in range(NSENS):
            if not np.any(self.last_raw[sensor * 3:sensor * 3 + 3]):
                self.sensor_alive[sensor] = False

        ref_columns = [c for s in ref_sensors() for c in range(3 * s, 3 * s + 3)]
        if not self.started:
            self.scg_zi = init_filter_state(SCG_SOS, xyz[0])
            self.motion_zi = init_filter_state(MOTION_SOS, xyz[0, ref_columns])
            self.hr_zi = init_filter_state(HR_SOS, xyz[0])
            self.started = True

        hp, self.scg_zi = sosfilt(SCG_SOS, xyz, axis=0, zi=self.scg_zi)

        hr_band, self.hr_zi = sosfilt(HR_SOS, xyz, axis=0, zi=self.hr_zi)     # 20-60 Hz on all 9 axes -> heart rate
        m = min(len(hr_band), HR_WINDOW_N)
        self.hr_buffer = np.roll(self.hr_buffer, -m, axis=0)
        self.hr_buffer[-m:] = hr_band[-m:]
        self.hr_filled = min(self.hr_filled + len(hr_band), HR_WINDOW_N)

        low, self.motion_zi = sosfilt(MOTION_SOS, xyz[:, ref_columns], axis=0, zi=self.motion_zi)

        chest = hp[:, 3 * CHEST_SENSOR_INDEX:3 * CHEST_SENSOR_INDEX + 3]
        refs = hp[:, ref_columns]
        ref_alive = np.array([self.sensor_alive[idx] for idx in ref_sensors()], dtype=bool)
        ref_mask = np.repeat(ref_alive, 3).astype(float)
        refs = refs * ref_mask[None, :]
        low = low * ref_mask[None, :]

        clean = np.empty_like(chest)
        alpha = 1.0 / (0.2 * FS)
        for i in range(len(chest)):
            r = refs[i]
            self.ref_history = np.roll(self.ref_history, 1, axis=0)
            self.ref_history[0] = r
            u = self.ref_history.ravel()

            active = max(int(np.sum(ref_mask)), 1)
            self.motion_energy += alpha * (float(r @ r) / active - self.motion_energy)
            self.tick += 1
            if self.tick % 50 == 0:
                self.motion_baseline.append(self.motion_energy)
            quiet = float(np.percentile(self.motion_baseline, 10)) if len(self.motion_baseline) > 100 else 0.0
            threshold = max(MOVE_RMS ** 2, (MOVE_FACTOR ** 2) * quiet)

            if self.motion_energy > threshold and np.any(ref_mask):
                self.motion_hold = HOLD_SAMPLES
            elif self.motion_hold > 0:
                self.motion_hold -= 1
            self.moving = self.motion_hold > 0

            y = chest[i]
            if self.moving and np.any(ref_mask):
                Pu = self.P @ u
                k = Pu / max(LAMBDA + float(u @ Pu), 1e-12)
                error = y - self.W @ u
                self.W += np.outer(error, k)
                self.P = (self.P - np.outer(k, Pu)) / LAMBDA
                self.P = 0.5 * (self.P + self.P.T)
                trace = float(np.trace(self.P))
                if trace > 1e5:
                    self.P *= 1e5 / trace
                clean[i] = error
            else:
                clean[i] = y                              # still: RLS adaptation is frozen

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
        """Averaged envelope autocorrelation of all working axes -> score on the bpm grid."""
        autocorrelations = []
        for sensor in range(NSENS):
            if not self.sensor_alive[sensor]:
                continue
            for axis in range(3):
                ac = beat_autocorrelation(self.hr_buffer[:, sensor * 3 + axis])
                if ac is not None:
                    autocorrelations.append(ac)
        if not autocorrelations:
            return None
        score = autocorrelation_score(np.mean(autocorrelations, axis=0), self.tracker.grid)
        self.quality.append(float(np.max(score)))
        period, strength = self.cadence
        if CADENCE_PENALTY > 0 and period and strength >= CADENCE_MIN_STRENGTH:
            step_bpm = 60.0 / period
            for mult in (0.5, 1.0, 2.0):
                near = np.abs(self.tracker.grid - mult * step_bpm) <= CADENCE_TOL * mult * step_bpm
                score = np.where(near, score - CADENCE_PENALTY * strength, score)
        return score

    def estimate(self, force=False):
        """Call once a second (the tracker assumes 1 s between updates).
        Returns (bpm or None, "locked"/"guess"/"unreliable", confidence) or None when there is not enough data yet."""
        now = time.monotonic()
        if not force and now - self.last_estimate_time < ESTIMATE_EVERY_S:
            return None
        self.last_estimate_time = now
        if self.hr_filled < HR_WINDOW_N:
            return None
        if any(self.sensor_alive[idx] for idx in ref_sensors()) and self.filled >= 6 * FS:
            self.cadence = estimate_cadence(self.low_buffer[-self.filled:])
        else:
            self.cadence = (None, 0.0)
        self.tracker.update(self._hr_score())
        self.hr_state = self.tracker.state()
        trusted = not (QUALITY_MIN_SCORE > 0 and self.quality and np.mean(self.quality) < QUALITY_MIN_SCORE)
        if trusted and self.hr_state[0] is not None:
            self.last_good = self.hr_state[0]
            self.hold_age = 0
        elif not trusted:                                # no clear rhythm: do not trust the new number
            self.hold_age += 1
            if HOLD_LAST_GOOD and self.last_good is not None:
                self.hr_state = (self.last_good, "holding", self.hr_state[2])      # keep showing the last good reading
            else:                                        # nothing good measured yet (only right after the start): "--"
                self.hr_state = (None, "unreliable", self.hr_state[2])
        return self.hr_state

    def plot_data(self):
        if self.filled == 0:
            return np.array([]), np.array([]), np.array([])
        raw = self.raw_buffer[-self.filled:, 2][-PLOT_N:]
        clean = self.clean_buffer[-self.filled:, 2][-PLOT_N:]
        return np.arange(len(raw)) / float(FS), raw, clean

    def hr_plot_data(self):
        """The signal the heart rate is really computed from: chest Z axis in the 20-60 Hz band and its energy envelope."""
        n = min(self.hr_filled, PLOT_N)
        if n == 0:
            return np.array([]), np.array([]), np.array([])
        signal = self.hr_buffer[-n:, CHEST_SENSOR_INDEX * 3 + 2]
        envelope = np.sqrt(moving_average(signal * signal, int(ENVELOPE_SECONDS * FS)))
        return np.arange(n) / float(FS), signal, envelope


# ----------------------------- REPLAY -----------------------------

def replay_log(path, chest="c2", quiet=False):
    """Runs a recording through the processor. The file needs the 13 sensor columns (header or not).
    The filter restarts after every gap in the data. Returns a list of (sec, hr, status, confidence, polar)."""
    import pandas as pd
    d = pd.read_csv(path, low_memory=False)
    if "t_us" not in d.columns:
        d = pd.read_csv(path, header=None, low_memory=False)
        d.columns = ["t_us"] + [f"v{i}" for i in range(1, d.shape[1])]
    polar = pd.to_numeric(d["polar_bpm"], errors="coerce").values if "polar_bpm" in d.columns else None
    label = d["label"].astype(str).values if "label" in d.columns else None
    raw = d.iloc[:, :NCOL].to_csv(index=False, header=False, float_format="%.3f").encode()
    feed = Feed(chest)
    rows = feed.parse(raw + b"\n")
    X = np.array([r[3] for r in rows])
    secs = np.array([r[1] for r in rows])
    gaps = np.array([r[4] for r in rows])
    idx_valid = np.arange(len(rows))
    starts = [0] + [i for i in range(1, len(rows)) if gaps[i]]
    ends = starts[1:] + [len(rows)]
    result = []
    for a, b in zip(starts, ends):
        if b - a < (HR_WINDOW_SECONDS + 2) * FS:
            continue
        proc = SCGProcessor()
        for i in range(a, b - FS + 1, FS):
            proc.push(X[i:i + FS])
            st = proc.estimate(force=True)
            if st:
                j = i + FS - 1
                pol = np.nan
                if polar is not None:
                    pol = polar[min(j, len(polar) - 1)]
                result.append((secs[j], st[0], st[1], st[2], pol, int(proc.moving)))
                if not quiet:
                    extra = f" | polar {pol:4.0f}" if not np.isnan(pol) else ""
                    print(f"{secs[j]:7.1f}s | HR: {format_hr(st):34s} | {'MOVING' if proc.moving else 'still ':6s}{extra}")
    if result and polar is not None:
        r = np.array([[x[1], x[4]] for x in result if x[1] is not None and not np.isnan(x[4])], dtype=float)
        if len(r):
            e = np.abs(r[:, 0] - r[:, 1])
            locked = np.array([x[2] == "locked" for x in result if x[1] is not None and not np.isnan(x[4])])
            print(f"\nCompared with the Polar: mean error {e.mean():.1f} bpm, median {np.median(e):.1f} bpm, "
                  f"within +-5 bpm {100 * np.mean(e <= 5):.0f}% of the seconds, locked {100 * locked.mean():.0f}% of the time"
                  + (f", error when locked {e[locked].mean():.1f} bpm" if locked.any() else ""))
    return result


# ----------------------------- LIVE APP -----------------------------

def main(port, chest):
    import serial
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

    print(f"Connecting to {port} @ {BAUD} ...")
    try:
        ser = serial.Serial(port, BAUD, timeout=0.01)
    except serial.SerialException as exc:
        print(f"Cannot open {port}: {exc}")
        print("Check the port and close the Arduino Serial Monitor.")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_name, hr_name = f"scg_data_{stamp}.csv", f"scg_hr_{stamp}.csv"
    fdata = open(data_name, "w", buffering=1, encoding="utf-8")
    fdata.write("t_us,c1x,c1y,c1z,c2x,c2y,c2z,dx,dy,dz,bx,by,bz,label,polar_bpm\n")
    fhr = open(hr_name, "w", buffering=1, encoding="utf-8")
    fhr.write("sec,label,polar_bpm,hr,status,confidence,moving,step_rate_per_min\n")

    feed = Feed(chest)
    proc = SCGProcessor()
    app = QtWidgets.QApplication(sys.argv)
    state = {"label": "none", "polar": "", "typing": None}

    class Win(pg.GraphicsLayoutWidget):
        def keyPressEvent(self, ev):
            k = ev.text()
            if state["typing"] is not None:
                if ev.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                    if state["typing"].isdigit():
                        state["polar"] = state["typing"]
                    state["typing"] = None
                elif ev.key() == QtCore.Qt.Key.Key_Backspace:
                    state["typing"] = state["typing"][:-1]
                elif k.isdigit():
                    state["typing"] += k
            elif k.lower() == "q":
                app.quit()
            elif k.lower() == "h":
                state["typing"] = ""
            elif k in LABELS:
                state["label"] = LABELS[k]

    win = Win(title=f"live_scg_en - {chest}")
    win.resize(1200, 900)
    win.show()
    header = win.addLabel("Waiting for data...", row=0, col=0, colspan=2)
    plot = win.addPlot(row=1, col=0, colspan=2,
                       title="1  chest Z axis, 5-25 Hz: RAW and CLEAN after RLS (preview only)")
    plot.setLabel("bottom", "Time in window", units="s")
    plot.setLabel("left", "Filtered acceleration", units="m/s^2")
    plot.showGrid(x=True, y=True, alpha=0.25)
    plot.addLegend()
    raw_curve = plot.plot(pen=pg.mkPen((220, 180, 0), width=1), name="RAW")
    clean_curve = plot.plot(pen=pg.mkPen((0, 220, 100), width=1.5), name="CLEAN (RLS)")

    hr_plot = win.addPlot(row=2, col=0, colspan=2,
                          title="2  what the heart rate is computed from: chest Z, 20-60 Hz (grey) and its energy envelope (red) - the heart rate is the rhythm of the humps")
    hr_plot.setLabel("bottom", "Time in window", units="s")
    hr_plot.setLabel("left", "Filtered acceleration", units="m/s^2")
    hr_plot.showGrid(x=True, y=True, alpha=0.25)
    hr_signal_curve = hr_plot.plot(pen=pg.mkPen((150, 150, 150), width=1))
    hr_envelope_curve = hr_plot.plot(pen=pg.mkPen((255, 70, 70), width=2))
    details = win.addLabel("", row=3, col=0, colspan=2)
    win.addLabel(f"Files: {data_name}, {hr_name}", row=4, col=0, colspan=2)

    counters = {"n": 0, "t": time.monotonic(), "rate": 0.0, "rows": 0, "err": ""}

    def update():
        try:
            _update()
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if msg != counters["err"]:
                traceback.print_exc()
                counters["err"] = msg
            header.setText(f"ERROR in update(): {msg}  (full traceback in the console)")

    def _update():
        rows = feed.parse(ser.read(8192))
        for t, sec, v, row10, gap in rows:
            fdata.write(f"{t}," + ",".join(f"{x:.3f}" for x in v) + f",{state['label']},{state['polar']}\n")
        if rows:
            # split the block at gaps so each part is processed continuously
            block = []
            for t, sec, v, row10, gap in rows:
                if gap:                                    # rows were lost: the signal is not continuous any more
                    if block:
                        proc.push(block)
                        block = []
                    keep = (proc.last_good, proc.hold_age)
                    proc.__init__()                        # restart filters and tracker
                    proc.last_good, proc.hold_age = keep   # but keep showing the last good reading until a new one is found
                block.append(row10)
            if block:
                proc.push(block)
            counters["n"] += len(rows)
            counters["rows"] += len(rows)

        now = time.monotonic()
        if now - counters["t"] >= 1.0:
            counters["rate"] = counters["n"] / (now - counters["t"])
            counters["n"] = 0
            counters["t"] = now

        result = proc.estimate()
        x, raw, clean = proc.plot_data()
        if len(x):
            raw_curve.setData(x, raw)
            clean_curve.setData(x, clean)
        hx, hs, he = proc.hr_plot_data()
        if len(hx):
            hr_signal_curve.setData(hx, hs)
            hr_envelope_curve.setData(hx, he)

        lost = f" | DATA LOST {feed.lost_s:.0f} s" if feed.lost_s > 0.5 else ""
        typing = f" | typing Polar: {state['typing']}_" if state["typing"] is not None else ""
        if result is None:
            if proc.hr_filled < HR_WINDOW_N:
                header.setText(f"Collecting data: {proc.hr_filled}/{HR_WINDOW_N} samples | rows OK: {counters['rows']}, "
                               f"bad lines: {feed.bad}{lost}")
        else:
            period, strength = proc.cadence
            step_rate = 60.0 / period if period else None
            header.setText(f"HR: {format_hr(result, proc.hold_age)}   |   label: {state['label']}   polar: {state['polar'] or '-'}{typing}{lost}")
            details.setText(
                f"State: {'MOVING' if proc.moving else 'still'} | "
                f"Step rate: {'--' if step_rate is None else f'{step_rate:.0f}/min'} (strength {strength:.2f}) | "
                + "".join(f"{SENSOR_NAMES[i]}: {'ON' if proc.sensor_alive[i] else 'OFF'} | " for i in range(NSENS))
                + f"{counters['rate']:.0f} rows/s | bad lines {feed.bad}")
            fhr.write(f"{feed.sec:.1f},{state['label']},{state['polar']},"
                      f"{'' if result[0] is None else round(result[0], 1)},{result[1]},{result[2]:.2f},{int(proc.moving)},"
                      f"{'' if step_rate is None else round(step_rate, 1)}\n")

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(UI_INTERVAL_MS)
    app.exec()
    timer.stop()
    ser.close()
    fdata.close()
    fhr.close()
    print(f"Saved {data_name} and {hr_name}")


if __name__ == "__main__":
    import os
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit()
    chest_arg = sys.argv[2] if len(sys.argv) > 2 else "c2"
    if os.path.isfile(sys.argv[1]):
        replay_log(sys.argv[1], chest_arg)
    else:
        main(sys.argv[1], chest_arg)
