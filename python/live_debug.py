"""
live_debug.py - see every signal separately and what the RLS filter subtracts.
Needs live_scg.py in the same folder (port, sensors and processing come from there).
Run:  py live_debug.py       (PORT / BAUD from live_scg.py)

Works with the 3-sensor stream (t,C,D,B) and the 4-sensor stream (t,C1,C2,D,B);
the number of sensors is detected from the data and the panel titles are filled in then.

Panels (last 5 seconds, all band-passed 5-25 Hz, same units m/s^2):
  1    CHEST (C / C1) raw (yellow)  and  the MOTION ESTIMATE being subtracted by RLS (red)
  2    RESULT = chest - estimate (green)  = the CLEAN curve of live_scg.py
       (only for viewing: the heart rate is computed from the 20-60 Hz band, see live_scg.py)
  3-5  REFERENCE sensors x/y/z (D and B, plus C2 with 4 sensors)
  6    Motion gate: reference RMS (white) vs threshold (red dashed). RLS only learns/subtracts when white is above red.
Keys: 1/2/3 choose which chest axis is shown (x/y/z).
"""
import time, datetime
import numpy as np
import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import live_scg as M

PORT, BAUD, FS = M.PORT, M.BAUD, M.FS
SHOW = 5 * FS
AXIS = 2                                        # 0=x 1=y 2=z of the chest
MAX_REFS = 3                                    # 4 sensors -> 3 reference sensors

ser = serial.Serial(PORT, BAUD, timeout=0.01)
log = open("log_dbg_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv", "w")
p = M.SCGProcessor()

app = QtWidgets.QApplication([])
class Win(pg.GraphicsLayoutWidget):
    def keyPressEvent(self, ev):
        global AXIS
        k = ev.text()
        if k in ("1", "2", "3"):
            AXIS = int(k) - 1
win = Win(title="SCG debug view")
win.resize(1100, 1000); win.show()
head = win.addLabel("starting...", row=0, col=0)

def mkplot(row, title, n_curves, colors, yr=None):
    pl = win.addPlot(row=row, col=0, title=title)
    pl.showGrid(x=False, y=True, alpha=0.3)
    if yr: pl.setYRange(*yr)
    pl.setXRange(0, SHOW)
    curves = [pl.plot(pen=pg.mkPen(c, width=1.3)) for c in colors]
    return pl, curves

XYZ = [(255, 120, 120), (120, 255, 120), (120, 160, 255)]
pl1, (c_raw, c_sub) = mkplot(1, "1  CHEST raw (yellow)  and motion estimate being subtracted (red)", 2, [(220, 220, 0), (255, 60, 60)])
pl2, (c_cln,) = mkplot(2, "2  RESULT = chest - estimate (green), view only", 1, [(0, 255, 0)], (-0.1, 0.1))
ref_plots = [mkplot(3 + i, f"{3 + i}  reference sensor (waiting for data)", 3, XYZ) for i in range(MAX_REFS)]
for pl, _ in ref_plots[1:]:
    pl.setYLink(ref_plots[0][0])                 # same scale for all references, easy to compare
pl_gate, (c_es, c_thr) = mkplot(3 + MAX_REFS, f"{3 + MAX_REFS}  Motion gate: reference RMS (white) vs threshold (red) - subtracts only when white is above red", 2,
                                [(255, 255, 255), (255, 60, 60)])
c_thr.setPen(pg.mkPen((255, 60, 60), width=1.3, style=QtCore.Qt.PenStyle.DashLine))

def blank(k=1):
    return np.zeros((SHOW, k)) if k > 1 else np.zeros(SHOW)
B = dict(raw=blank(), sub=blank(), cln=blank(), es=blank(), thr=blank())
for i in range(MAX_REFS):
    B[f"ref{i}"] = blank(3)
def push(key, new):
    B[key] = np.concatenate([B[key], new])[-SHOW:]

titled = False
def set_titles():
    """called once the number of sensors is known"""
    names = p.sensor_names
    pl1.setTitle(f"1  CHEST {names[p.chest_sensor]} raw (yellow)  and motion estimate being subtracted (red)")
    for i, (pl, _) in enumerate(ref_plots):
        if i < len(p.ref_sensors):
            pl.setTitle(f"{3 + i}  {names[p.ref_sensors[i]]} x / y / z (reference)")
        else:
            pl.setTitle(f"{3 + i}  (no sensor - 3-sensor stream)")

buf = b""; rate_n = 0; rate_t = time.time(); rate = 0; last_est = time.time(); text = "measuring..."
def update():
    global buf, rate_n, rate_t, rate, last_est, text, titled
    data = ser.read(8192)
    if data:
        log.write(data.decode("ascii", "ignore")); buf += data
    rows, buf = M.parse_lines(buf)
    if rows:
        raw, cln = p.push(rows)
        if len(raw) == 0:
            return
        if not titled:
            set_titles(); titled = True
        rate_n += len(raw)
        push("raw", raw[:, AXIS]); push("cln", cln[:, AXIS]); push("sub", raw[:, AXIS] - cln[:, AXIS])
        for i in range(min(len(p.ref_sensors), MAX_REFS)):
            push(f"ref{i}", p.last_refs[:, 3 * i:3 * i + 3])
        push("es", np.sqrt(p.last_energy)); push("thr", np.sqrt(p.last_threshold))
        c_raw.setData(B["raw"]); c_sub.setData(B["sub"]); c_cln.setData(B["cln"])
        for i, (_, curves) in enumerate(ref_plots):
            for k in range(3):
                curves[k].setData(B[f"ref{i}"][:, k])
        c_es.setData(B["es"]); c_thr.setData(B["thr"])
        # fixed-ish scale for the chest so you can see the real size; auto for the rest
        m = max(0.05, float(np.percentile(np.abs(B["raw"]), 98)) * 1.2)
        pl1.setYRange(-m, m)
    now = time.time()
    if now - rate_t >= 1:
        rate = rate_n / (now - rate_t); rate_n = 0; rate_t = now
    if now - last_est >= 1:
        last_est = now
        r = p.estimate(force=True)
        ax = "xyz"[AXIS]
        n = p.n_sensors or 0
        alive = "  ".join(f"{p.sensor_names[s]} {'ON' if p.sensor_alive[s] else 'OFF'}" for s in range(n))
        ref_txt = f"   HR reference: {p.sensor_names[p.noise_ref]}" if p.noise_ref is not None else ""
        if r:
            text = (f"HR: {M.format_hr(r)}   |   {'MOVING' if p.moving else 'still'}   {alive}{ref_txt}   "
                    f"{rate:.0f}/s   | chest axis shown: {ax} (keys 1/2/3)")
        else:
            text = f"measuring... {p.hr_filled}/{M.HR_WINDOW_N} samples  {n or '?'} sensors  {rate:.0f}/s"
        head.setText(text)

t = QtCore.QTimer(); t.timeout.connect(update); t.start(30)
app.exec()
log.close()
