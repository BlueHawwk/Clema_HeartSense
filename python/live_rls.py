"""
live_rls.py  -  heart rate from chest SCG with motion cancellation (device + back sensors as references)

Stream format from the ESP8266 (921600 baud):  t,cx,cy,cz,dx,dy,dz,bx,by,bz   (500 Hz)
   chest = columns 1-3, device = 4-6, back = 7-9.   Lines starting with '#' are ignored.
   A sensor that is down prints zeros: it is detected and simply not used.

Run live:      py live_rls.py
Replay a log:  py live_rls.py log_rls_xxxx.csv        (prints HR once per second, no plot)

What is different from the earlier scripts
  1. RLS (recursive least squares) canceller instead of NLMS: converges in about a second, not minutes.
     6 reference channels (device xyz + back xyz) x 6 taps = 36 weights per chest axis.
  2. It learns ONLY while the references say you are moving (adaptive threshold), and is frozen when still,
     so it cannot learn to subtract your heartbeat that leaks into the references.
  3. Heart rate comes from a beat-energy envelope + autocorrelation on each of the 3 chest axes,
     candidates are scored, then a TRACKER with memory picks the one consistent with the previous rate
     (rate cannot jump 30 bpm in a second) and holds the value for a few seconds when the signal is bad.
  4. Step cadence is measured from the references and candidates that sit exactly on the walking
     rhythm (or its double/half) are penalised.
  5. Never shows an old number as if it were current: status says locked / holding / unreliable.
"""
import sys, time, datetime
from collections import deque
import numpy as np
from scipy.signal import butter, sosfilt


def uniform_filter1d(x, size):
    """moving average, same length, edges reflected (numpy only, so no extra scipy modules are needed)"""
    x = np.asarray(x, float)
    size = max(int(size), 1)
    if size == 1 or len(x) < 2:
        return x.copy()
    size = min(size, len(x))
    left = size // 2
    right = size - 1 - left
    xp = np.pad(x, (left, right), mode="reflect")
    c = np.cumsum(np.insert(xp, 0, 0.0))
    return (c[size:] - c[:-size]) / size

# ----------------------------------------------------------------- settings
PORT, BAUD = "COM5", 921600
FS = 500
BAND = (5.0, 25.0)            # SCG band, Hz
WIN = 10 * FS                 # analysis window
L = 6                         # taps per reference channel
LAM = 0.998                   # RLS forgetting factor (memory ~ 1/(1-LAM) = 500 samples = 1 s)
MOVE_RMS = 0.03               # absolute floor for "moving" (m/s^2 of reference RMS in the SCG band)
MOVE_FACTOR = 2.5             # moving = reference RMS > MOVE_FACTOR x its own quiet level
HOLD = int(1.5 * FS)          # keep cancelling this long after motion stops
BPM_MIN, BPM_MAX = 40, 150
ACCEPT = 0.30                 # score needed to lock on
KEEP = 0.20                   # score needed to keep following
HOLD_S = 8                    # seconds to keep showing the last rate when the signal is bad


# ----------------------------------------------------------------- estimator pieces
def beat_autocorr(x):
    """x: 1-D band-passed chest signal at FS. Returns autocorrelation of the beat-energy envelope at 100 Hz."""
    s = np.median(np.abs(x)) / 0.6745 + 1e-12
    x = np.clip(x, -4 * s, 4 * s)                               # tame shocks
    e = np.sqrt(uniform_filter1d(x * x, int(0.06 * FS)))        # 60 ms energy envelope
    e = e[::5]                                                  # 100 Hz
    e = np.minimum(e, 3 * np.median(e) + 1e-12)                 # a single big burst cannot dominate
    e = e - uniform_filter1d(e, 100)                            # remove 1 s trend
    n = len(e)
    f = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (n - np.arange(n))                                # unbiased
    return ac / (ac[0] + 1e-12)


def candidates(ac):
    """peaks of ac in the heart-rate range -> list of (bpm, score, lag_seconds)"""
    lo, hi = int(60 / BPM_MAX * 100), int(60 / BPM_MIN * 100)
    out = []
    for i in range(lo, min(hi, len(ac) // 2 - 1)):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.1:
            a, b, c = ac[i - 1], ac[i], ac[i + 1]
            d = (a - 2 * b + c)
            off = 0.5 * (a - c) / d if d != 0 else 0.0           # parabolic interpolation
            lag = (i + off) / 100.0
            score = 0.75 * b + 0.25 * max(ac[2 * i], 0.0)       # a real rhythm repeats at 2x the lag too
            out.append((60.0 / lag, float(score), lag))
    return out


def cadence(low_refs):
    """step period from the references' low band (0.8-4 Hz). low_refs: (n, 6) at FS. -> (period_s, strength)"""
    acs = []
    for k in range(low_refs.shape[1]):
        v = low_refs[::5, k]
        if np.std(v) < 1e-6:
            continue
        v = v - np.mean(v)
        f = np.fft.rfft(v, 2 * len(v))
        a = np.fft.irfft(f * np.conj(f))[:len(v)]
        a = a / (len(v) - np.arange(len(v)))
        acs.append(a / (a[0] + 1e-12))
    if not acs:
        return None, 0.0
    a = np.mean(acs, axis=0)
    lo, hi = 30, 120                                            # 0.3 - 1.2 s
    i = lo + int(np.argmax(a[lo:hi]))
    return i / 100.0, float(a[i])


class Tracker:
    def __init__(self):
        self.hr = None; self.conf = 0.0; self.miss = 0
        self.lock = []; self.jump = []
        self.last_good = None; self.guess = None; self.gscore = 0.0

    def update(self, cands):
        self._make_guess(cands)
        cands = [c for c in cands if c[1] >= 0.15]
        if self.hr is None:
            good = [c for c in cands if c[1] >= ACCEPT]
            if good:
                self.lock = (self.lock + [max(good, key=lambda c: c[1])[0]])[-3:]
                if len(self.lock) == 3 and max(self.lock) - min(self.lock) <= 8:
                    self.hr = float(np.median(self.lock)); self.conf = max(c[1] for c in good)
                    self.miss = 0; self.lock = []
            else:
                self.lock = []
            return
        gate = min(8 + 2 * self.miss, 40)
        near = [c for c in cands if abs(c[0] - self.hr) <= gate and c[1] >= KEEP]
        if near:
            b = max(near, key=lambda c: c[1] - 0.004 * abs(c[0] - self.hr))
            g = 0.6 if b[1] > 0.6 else 0.3
            self.hr += g * (b[0] - self.hr); self.conf = b[1]; self.miss = 0; self.jump = []
        else:
            self.miss += 1
            strong = [c for c in cands if c[1] >= 0.6]
            if strong:                                           # re-lock only if the same new rate repeats
                self.jump = (self.jump + [max(strong, key=lambda c: c[1])[0]])[-3:]
                if len(self.jump) == 3 and max(self.jump) - min(self.jump) <= 8:
                    self.hr = float(np.median(self.jump)); self.miss = 0; self.jump = []
            else:
                self.jump = []
            if self.miss > HOLD_S:
                self.hr = None; self.lock = []; self.conf = 0.0

    def _make_guess(self, cands):
        """best-effort value for when there is no lock: strongest candidate, pulled towards the last good rate"""
        if not cands:
            self.guess = None
            return
        ref = self.hr if self.hr is not None else self.last_good
        def val(c):
            return c[1] - (0.004 * abs(c[0] - ref) if ref is not None else 0.0)
        b = max(cands, key=val)
        self.guess, self.gscore = b[0], b[1]

    def state(self):
        if self.hr is not None:
            self.last_good = self.hr
        if self.hr is None:
            if self.guess is None:
                return None, "unreliable", 0.0
            return self.guess, "guess", self.gscore
        return self.hr, ("locked" if self.miss == 0 else "holding"), self.conf


# ----------------------------------------------------------------- main processor
class Processor:
    def __init__(self):
        self.sos = butter(4, BAND, btype="band", fs=FS, output="sos")
        self.sos_low = butter(2, [0.8, 4.0], btype="band", fs=FS, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2, 9))
        self.zi_low = np.zeros((self.sos_low.shape[0], 2, 6))
        self.started = False
        self.last = np.zeros(9)                    # last good raw value per column
        self.alive = np.zeros(3, bool)             # chest / device / back
        # RLS
        self.nu = 6 * L
        self.P = np.eye(self.nu) * 1e3
        self.W = np.zeros((3, self.nu))
        self.hist = np.zeros((L, 6))
        # motion gate
        self.es = 0.0
        self.base = deque(maxlen=600)
        self.tick = 0
        self.hold = 0
        self.moving = False
        # buffers
        self.raw = np.zeros((WIN, 3)); self.clean = np.zeros((WIN, 3)); self.low = np.zeros((WIN, 6))
        self.filled = 0
        self.tr_raw, self.tr_clean = Tracker(), Tracker()
        self.cad = (None, 0.0)

    # ---- one block of raw rows (n, 9) -> returns (raw_hp (n,3), clean (n,3))
    def push(self, X):
        X = np.array(X, float)
        n = len(X)
        for i in range(n):                                       # hold last good value when a sensor reads all zero
            for s in range(3):
                sl = slice(3 * s, 3 * s + 3)
                if np.any(X[i, sl] != 0):
                    self.last[sl] = X[i, sl]; self.alive[s] = True
                else:
                    X[i, sl] = self.last[sl]
        # a sensor that has never produced data stays zero; mark dead ones
        for s in range(3):
            if not np.any(self.last[3 * s:3 * s + 3]):
                self.alive[s] = False
        if not self.started:
            self.zi = self.zi * 0 + sosfilt_init(self.sos, X[0])  # start without a step transient
            self.zi_low = self.zi_low * 0 + sosfilt_init(self.sos_low, X[0, 3:])
            self.started = True
        hp, self.zi = sosfilt(self.sos, X, axis=0, zi=self.zi)
        lo, self.zi_low = sosfilt(self.sos_low, X[:, 3:], axis=0, zi=self.zi_low)
        chest, refs = hp[:, :3], hp[:, 3:]
        use_ref = np.array([self.alive[1]] * 3 + [self.alive[2]] * 3)   # which of the 6 ref channels are real
        refs = refs * use_ref
        clean = np.empty_like(chest)
        mv = np.zeros(n, bool); es_arr = np.zeros(n); thr_arr = np.zeros(n)
        a = 1.0 / (0.2 * FS)
        for i in range(n):
            r = refs[i]
            self.hist = np.roll(self.hist, 1, axis=0); self.hist[0] = r
            u = self.hist.ravel()
            nact = max(use_ref.sum(), 1)
            self.es += a * (float(r @ r) / nact - self.es)
            self.tick += 1
            if self.tick % 50 == 0:
                self.base.append(self.es)
            q = np.percentile(self.base, 10) if len(self.base) > 100 else 0.0
            thr = max(MOVE_RMS ** 2, (MOVE_FACTOR ** 2) * q)
            if self.es > thr:
                self.hold = HOLD
            elif self.hold > 0:
                self.hold -= 1
            self.moving = self.hold > 0
            mv[i] = self.moving; es_arr[i] = self.es; thr_arr[i] = thr
            y = chest[i]
            if self.moving and use_ref.any():
                Pu = self.P @ u
                k = Pu / (LAM + u @ Pu)
                e = y - self.W @ u                               # a-priori error = cleaned chest
                self.W += np.outer(e, k)
                self.P = (self.P - np.outer(k, Pu)) / LAM
                self.P = 0.5 * (self.P + self.P.T)
                tr = np.trace(self.P)
                if tr > 1e5:                                      # covariance wind-up protection
                    self.P *= 1e5 / tr
                clean[i] = e
            else:
                clean[i] = y
        self._append(chest, clean, lo)
        self.last_refs, self.last_mv, self.last_es, self.last_thr = refs, mv, es_arr, thr_arr   # for the debug viewer
        return chest, clean

    def _append(self, chest, clean, lo):
        n = len(chest)
        if n >= WIN:
            chest, clean, lo, n = chest[-WIN:], clean[-WIN:], lo[-WIN:], WIN
        self.raw = np.roll(self.raw, -n, axis=0); self.raw[-n:] = chest
        self.clean = np.roll(self.clean, -n, axis=0); self.clean[-n:] = clean
        self.low = np.roll(self.low, -n, axis=0); self.low[-n:] = lo
        self.filled = min(self.filled + n, WIN)

    # ---- call about once a second
    def estimate(self):
        if self.filled < 6 * FS:
            return None
        m = self.filled
        self.cad = cadence(self.low[-m:]) if (self.alive[1] or self.alive[2]) else (None, 0.0)
        for tr, buf in ((self.tr_raw, self.raw), (self.tr_clean, self.clean)):
            seg = buf[-m:]
            sigs = [seg[:, 0], seg[:, 1], seg[:, 2]]
            u, s, vt = np.linalg.svd(seg - seg.mean(0), full_matrices=False)
            sigs.append(seg @ vt[0])                              # strongest combination of the 3 axes
            cands = []
            for x in sigs:
                for bpm, sc, lag in candidates(beat_autocorr(x)):
                    if self.cad[0] and self.cad[1] > 0.4:        # sits on the walking rhythm -> less trust
                        for mult in (0.5, 1.0, 2.0):
                            if abs(lag - mult * self.cad[0]) / lag < 0.05:
                                sc *= 0.6; break
                    cands.append((bpm, sc, lag))
            tr.update(cands)
        return self.tr_raw.state(), self.tr_clean.state()


def sosfilt_init(sos, x0):
    """filter state for a constant input x0 (no start-up transient). x0 shape (c,) -> (sections, 2, c)"""
    from scipy.signal import sosfilt_zi
    z = sosfilt_zi(sos)                                           # (sections, 2)
    return z[:, :, None] * np.asarray(x0)[None, None, :]


# ----------------------------------------------------------------- parsing / IO
def parse_lines(buf):
    rows = []
    parts = buf.split(b"\n")
    rest = parts[-1]
    for p in parts[:-1]:
        p = p.strip()
        if not p or p.startswith(b"#"):
            continue
        f = p.split(b",")
        if len(f) != 10:
            continue
        try:
            rows.append([float(v) for v in f[1:]])
        except ValueError:
            pass
    return rows, rest


def fmt(st):
    hr, status, conf = st
    if hr is None:
        return "--"
    if status == "guess":
        return f"~{hr:.0f} bpm (low confidence {conf:.2f})"
    return f"{hr:.0f} bpm ({status}, {conf:.2f})" if status == "locked" else f"{hr:.0f} bpm (holding)"


def replay(path):
    import pandas as pd
    d = pd.read_csv(path, header=None, comment="#").values
    d = d[:, 1:10] if d.shape[1] >= 10 else d
    p = Processor()
    step = FS
    for i in range(0, len(d) - step, step):
        p.push(d[i:i + step])
        r = p.estimate()
        if r:
            print(f"{(i + step) / FS:6.0f}s  raw: {fmt(r[0]):26s} clean: {fmt(r[1]):26s} {'MOVING' if p.moving else 'still'}")


def main():
    import serial
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
    ser = serial.Serial(PORT, BAUD, timeout=0.01)
    log = open("log_rls_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv", "w")
    p = Processor()
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(title="chest SCG: raw (yellow) vs cleaned (green)")
    win.resize(1000, 600); win.show()
    head = win.addLabel("starting...", row=0, col=0)
    plot = win.addPlot(row=1, col=0)
    SHOW = 5 * FS
    plot.setYRange(-0.15, 0.15)
    c_raw = plot.plot(pen=pg.mkPen((200, 200, 0), width=1))
    c_cln = plot.plot(pen=pg.mkPen((0, 255, 0), width=1.5))
    buf = b""; rate_n = 0; rate_t = time.time(); rate = 0
    last_est = time.time(); text = "measuring..."
    shown_raw = np.zeros(SHOW); shown_cln = np.zeros(SHOW)

    def update():
        nonlocal buf, rate_n, rate_t, rate, last_est, text, shown_raw, shown_cln
        data = ser.read(4096)
        if data:
            log.write(data.decode("ascii", "ignore")); buf += data
        rows, buf = parse_lines(buf)
        if rows:
            raw, cln = p.push(rows)
            rate_n += len(rows)
            shown_raw = np.concatenate([shown_raw, raw[:, 2]])[-SHOW:]
            shown_cln = np.concatenate([shown_cln, cln[:, 2]])[-SHOW:]
            c_raw.setData(shown_raw); c_cln.setData(shown_cln)
        now = time.time()
        if now - rate_t >= 1:
            rate = rate_n / (now - rate_t); rate_n = 0; rate_t = now
        if now - last_est >= 1:
            last_est = now
            r = p.estimate()
            if r:
                best = r[1] if (p.moving or r[1][0] is not None) else r[0]   # while moving never fall back to the raw value
                text = (f"HR: {fmt(best)}   |   raw: {fmt(r[0])}   clean: {fmt(r[1])}   |   "
                        f"{'MOVING' if p.moving else 'still'}   device {'ON' if p.alive[1] else 'OFF'}  "
                        f"back {'ON' if p.alive[2] else 'OFF'}   {rate:.0f}/s")
            else:
                text = f"measuring... {rate:.0f}/s"
            head.setText(text)

    t = QtCore.QTimer(); t.timeout.connect(update); t.start(30)
    app.exec()
    log.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        replay(sys.argv[1])
    else:
        main()