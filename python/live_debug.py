"""
live_debug.py - see every signal separately and what the RLS filter subtracts.
Needs live_scg.py in the same folder (port, sensor mapping and processing come from there).
Run:  py live_debug.py       (PORT / BAUD from live_scg.py)

Panels (last 5 seconds, all band-passed 5-25 Hz, same units m/s^2):
  1  CHEST raw (yellow)  and  the MOTION ESTIMATE being subtracted by RLS (red)
  2  RESULT = chest - estimate (green)  = the CLEAN curve of live_scg.py
     (only for viewing: the heart rate is computed from the 20-60 Hz band of all 9 axes, see live_scg.py)
  3  DEVICE sensor (D) x/y/z   (reference)
  4  BACK sensor (B) x/y/z     (reference)
  5  Motion gate: reference RMS (white) vs threshold (red dashed). RLS only learns/subtracts when white is above red.
Which sensor is the chest is set by CHEST_SENSOR_INDEX / REF_SENSOR_INDICES in live_scg.py.
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

ser = serial.Serial(PORT, BAUD, timeout=0.01)
log = open("log_dbg_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv", "w")
p = M.SCGProcessor()
NAMES = M.SENSOR_NAMES                          # C=chest, D=device, B=back (as in the README / firmware)
CHEST, REF1, REF2 = (NAMES[M.CHEST_SENSOR_INDEX],) + tuple(NAMES[i] for i in M.REF_SENSOR_INDICES)

app = QtWidgets.QApplication([])
class Win(pg.GraphicsLayoutWidget):
    def keyPressEvent(self, ev):
        global AXIS
        k = ev.text()
        if k in ("1", "2", "3"):
            AXIS = int(k) - 1
win = Win(title="SCG debug view")
win.resize(1100, 900); win.show()
head = win.addLabel("starting...", row=0, col=0)

def mkplot(row, title, n_curves, colors, yr=None):
    pl = win.addPlot(row=row, col=0, title=title)
    pl.showGrid(x=False, y=True, alpha=0.3)
    if yr: pl.setYRange(*yr)
    pl.setXRange(0, SHOW)
    curves = [pl.plot(pen=pg.mkPen(c, width=1.3)) for c in colors]
    return pl, curves

pl1, (c_raw, c_sub) = mkplot(1, f"1  {CHEST} raw (yellow)  and motion estimate being subtracted (red)", 2, [(220, 220, 0), (255, 60, 60)])
pl2, (c_cln,) = mkplot(2, "2  RESULT = chest - estimate (green), view only", 1, [(0, 255, 0)], (-0.1, 0.1))
pl3, dev = mkplot(3, f"3  {REF1} x / y / z (reference)", 3, [(255, 120, 120), (120, 255, 120), (120, 160, 255)])
pl4, bck = mkplot(4, f"4  {REF2} x / y / z (reference)", 3, [(255, 120, 120), (120, 255, 120), (120, 160, 255)])
pl5, (c_es, c_thr) = mkplot(5, "5  Motion gate: reference RMS (white) vs threshold (red) - subtracts only when white is above red", 2,
                            [(255, 255, 255), (255, 60, 60)])
c_thr.setPen(pg.mkPen((255, 60, 60), width=1.3, style=QtCore.Qt.PenStyle.DashLine))
pl3.setYLink(pl4)                                # same scale for device and back, easy to compare
for pl in (pl2,):
    pl.setYLink(pl1) if False else None

def blank(k=1):
    return np.zeros((SHOW, k)) if k > 1 else np.zeros(SHOW)
B = dict(raw=blank(), sub=blank(), cln=blank(), dev=blank(3), bck=blank(3), es=blank(), thr=blank())
def push(key, new):
    B[key] = np.concatenate([B[key], new])[-SHOW:]

buf = b""; rate_n = 0; rate_t = time.time(); rate = 0; last_est = time.time(); text = "measuring..."
def update():
    global buf, rate_n, rate_t, rate, last_est, text
    data = ser.read(4096)
    if data:
        log.write(data.decode("ascii", "ignore")); buf += data
    rows, buf = M.parse_lines(buf)
    if rows:
        raw, cln = p.push(rows)
        if len(raw) == 0:
            return
        rate_n += len(rows)
        push("raw", raw[:, AXIS]); push("cln", cln[:, AXIS]); push("sub", raw[:, AXIS] - cln[:, AXIS])
        push("dev", p.last_refs[:, 0:3]); push("bck", p.last_refs[:, 3:6])
        push("es", np.sqrt(p.last_energy)); push("thr", np.sqrt(p.last_threshold))
        c_raw.setData(B["raw"]); c_sub.setData(B["sub"]); c_cln.setData(B["cln"])
        for k in range(3):
            dev[k].setData(B["dev"][:, k]); bck[k].setData(B["bck"][:, k])
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
        alive = "  ".join(f"{NAMES[s]} {'ON' if p.sensor_alive[s] else 'OFF'}" for s in range(3))
        if r:
            text = (f"HR: {M.format_hr(r)}   |   {'MOVING' if p.moving else 'still'}   {alive}   "
                    f"{rate:.0f}/s   | chest axis shown: {ax} (keys 1/2/3)")
        else:
            text = f"measuring... {p.hr_filled}/{M.HR_WINDOW_N} samples  {rate:.0f}/s"
        head.setText(text)

t = QtCore.QTimer(); t.timeout.connect(update); t.start(30)
app.exec()
log.close()