"""
live_debug.py - see every signal separately and what is being subtracted.
Needs live_rls.py in the same folder.   Run:  py live_debug.py       (COM5, 921600)

Panels (last 5 seconds, all band-passed 5-25 Hz, same units m/s^2):
  1  CHEST raw (yellow)  and  the MOTION ESTIMATE being subtracted (red)
  2  RESULT = chest - estimate (green)   <- this is what the heart rate is computed from
  3  DEVICE sensor x/y/z   (reference)
  4  BACK sensor x/y/z     (reference)
  5  Motion gate: reference RMS (white) vs threshold (red dashed). Filter only learns/subtracts when white is above red.
Keys: 1/2/3 choose which chest axis is shown (x/y/z).
"""
import time, datetime
import numpy as np
import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import live_rls as M

PORT, BAUD, FS = M.PORT, M.BAUD, M.FS
SHOW = 5 * FS
AXIS = 2                                        # 0=x 1=y 2=z of the chest

ser = serial.Serial(PORT, BAUD, timeout=0.01)
log = open("log_dbg_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv", "w")
p = M.Processor()

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

pl1, (c_raw, c_sub) = mkplot(1, "1  CHEST raw (yellow)  and motion estimate being subtracted (red)", 2, [(220, 220, 0), (255, 60, 60)])
pl2, (c_cln,) = mkplot(2, "2  RESULT = chest - estimate (green)", 1, [(0, 255, 0)], (-0.1, 0.1))
pl3, dev = mkplot(3, "3  DEVICE sensor x / y / z (reference)", 3, [(255, 120, 120), (120, 255, 120), (120, 160, 255)])
pl4, bck = mkplot(4, "4  BACK sensor x / y / z (reference)", 3, [(255, 120, 120), (120, 255, 120), (120, 160, 255)])
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
        rate_n += len(rows)
        push("raw", raw[:, AXIS]); push("cln", cln[:, AXIS]); push("sub", raw[:, AXIS] - cln[:, AXIS])
        push("dev", p.last_refs[:, 0:3]); push("bck", p.last_refs[:, 3:6])
        push("es", np.sqrt(p.last_es)); push("thr", np.sqrt(p.last_thr))
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
        r = p.estimate()
        ax = "xyz"[AXIS]
        if r:
            best = r[1] if (p.moving or r[1][0] is not None) else r[0]
            text = (f"HR: {M.fmt(best)}   |   {'MOVING' if p.moving else 'still'}   device {'ON' if p.alive[1] else 'OFF'} "
                    f"back {'ON' if p.alive[2] else 'OFF'}   {rate:.0f}/s   | chest axis shown: {ax} (keys 1/2/3)")
        else:
            text = f"measuring... {rate:.0f}/s"
        head.setText(text)

t = QtCore.QTimer(); t.timeout.connect(update); t.start(30)
app.exec()
log.close()