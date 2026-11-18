# Clema HeartSense

Estimating heart rate through clothing with accelerometers, for wearable.

A small accelerometer on the chest picks up the mechanical vibration of each heartbeat
(seismocardiography, SCG). The problem is that body motion is 10-100x larger than the heartbeat.
This project uses two extra accelerometers (on the device and on the back) as motion references
and subtracts the motion they see from the chest signal with an adaptive filter.

> Hobby / research project. \*\*Not a medical device\*\* and not for health decisions.

## Status

* At rest: heart rate matches a Polar H10 chest strap.
* Moving: cancellation removes most of the motion in the signal, but the heart rate is not yet reliable in every movement. Local motion that only the chest feels (muscles, strap shake) cannot be cancelled by reference sensors.
* `live\_scg.py` (autocorrelation + HMM tracker) on the recordings in `raw\_data from accelerometers`, compared with the Polar H10:

|Recording|`live\_rls.py` (old)|`live\_scg.py`|
|-|-|-|
|walking 5 min|24.8 bpm mean error, 12% of seconds within 5 bpm|**3.3 bpm, 87%** (3.1 bpm, 92% after the first minute)|
|sitting 5 min|2.1 bpm, 94%|**1.5 bpm, 99%**|
|standing 5 min|-|**1.7 bpm, 99%**|

* 4 sensors (C1, C2, D, B), 15 min of walking (`walking\_realtime\_with\_polar.csv`, 50 MB, not in the repository; Polar 83-103 bpm), replayed with `live\_scg.py`:

|Variant|Whole recording|After the first minute of each section|
|-|-|-|
|C1 subtracted from C2, D, B (default)|**4.0 bpm, 90% within 5 bpm**|**1.9 bpm, 97%, correlation with Polar 0.60**|
|all 4 sensors averaged, no subtraction|5.6 bpm, 84%|2.8 bpm, 92%, correlation 0.08|

  With the subtraction the reading follows the Polar; without it it sits near the average. Subtracting C2 instead of C1 barely helps. Most of the remaining error is the start: for the first ~15-25 s the tracker can sit on the step rhythm (about 55-60/min).

  The old estimator locked onto the walking rhythm (about 50/min). Only one walking recording with a reference exists so far and the heart rate in it is nearly constant (88 ± 3.5 bpm), so fast heart-rate changes are not tested yet. Recordings with a Polar reference and a changing rate (stairs, faster walking) are very welcome.

## Hardware

|Part|Notes|
|-|-|
|ESP8266 NodeMCU (ESP-12E)|reads the sensors, streams over USB serial|
|3x Adafruit LSM6DS3TR-C (STEMMA QT)|chest (main), device (reference), back (reference)|
|Polar H10 (optional)|ground truth for checking the heart rate|

### Wiring (I2C, software I2C on two buses)

|Sensor|VIN|GND|SDA|SCL|DO|
|-|-|-|-|-|-|
|chest|3V|G|D2|D1|not connected (address 0x6A)|
|device|3V|G|D2|D1|to 3V (address 0x6B)|
|back|3V|G|D6|D5|not connected (address 0x6A)|

Chest and device share one bus (different addresses), the back sensor has its own bus because of its longer cable.
If a sensor misbehaves, use shorter wires or lower the I2C clock in the firmware.

## Repository layout

```
firmware/3\_sensors/3\_sensors.ino   ESP8266 firmware (500 Hz, 3 sensors, tolerates a dead sensor)
python/live\_scg.py                 live heart rate: SCG autocorrelation + HMM tracker, 3 or 4 sensors (recommended)
python/live\_debug.py               live view of every signal and what RLS subtracts (uses live\_scg.py)
python/live\_rls.py                 previous version (5-25 Hz autocorrelation + rule-based tracker)
raw\_data from accelerometers/      5 min recordings; \*\_merged.csv include the Polar H10 heart rate
requirements.txt                   Python packages
```

## Quick start

### 1\. Firmware

1. Arduino IDE, board **NodeMCU 1.0 (ESP-12E Module)**, select the COM port.
2. Open `firmware/3\_sensors/3\_sensors.ino` and upload.
3. Serial Monitor at **921600 baud** should print `# chest sensor OK` and then lines like:

```
t,cx,cy,cz,dx,dy,dz,bx,by,bz
```

With four sensors the line has 13 fields: `t,c1x,c1y,c1z,c2x,c2y,c2z,dx,dy,dz,bx,by,bz`. `live\_scg.py` and `live\_debug.py` detect 3 or 4 sensors from the line length.

Columns: time in microseconds, then x/y/z acceleration in m/s^2 for chest, device and back.
A sensor that is down prints zeros and the firmware retries it every 500 ms.
Lines starting with `#` are status messages.

### 2\. Python

```
py -m pip install -r requirements.txt
cd python
py live\_scg.py
```

* Close the Arduino Serial Monitor first, only one program can use the COM port.
* Set `PORT` (default `COM7`) and `BAUD` (default `921600`) at the top of `live\_scg.py`.
* Sensor roles: 3 sensors C = chest, D = device, B = back; 4 sensors C1, C2, D, B. `CHEST\_SENSOR\_INDEX = 0` (C / C1) is the sensor shown in the plots and cleaned by RLS.
* With 4 sensors `HR\_NOISE\_REF\_4 = 0`: C1 is subtracted from C2, D and B before the heart rate is computed. Set it to `None` to switch the subtraction off (or to another index to use a different reference). With 3 sensors there is no subtraction (`HR\_NOISE\_REF\_3 = None`).
* The window shows (1) chest raw vs RLS-cleaned, 5-25 Hz, view only, and (2) the signal the heart rate is computed from: chest 20-60 Hz and its energy envelope. The heart rate is the rhythm of the envelope bumps.
* Replay a log: `py live\_scg.py log\_scg\_YYYYMMDD\_HHMMSS.csv` (prints heart rate once per second). A gap of more than 1 s in the time column starts the processing again, as a reconnect would.
* Replay a merged recording and compare with the Polar: `py live\_scg.py "../raw\_data from accelerometers/walking\_5min\_merged.csv"`.
* `py live\_debug.py` shows five panels: chest raw vs the estimate RLS subtracts, the result, both reference sensors, and the motion gate. Needs `live\_scg.py` in the same folder.

Every live run writes a timestamped `log\_\*.csv` with the raw stream.

## How it works (`live\_scg.py`)

Heart rate:

1. **Band-pass 20-60 Hz** on all axes. Below about 20 Hz the footsteps dominate; the valve "click" of the heartbeat has a lot of energy above that. This change alone took walking from 25 to about 6.5 bpm error.
2. **4 sensors only: subtract C1.** In every 10 s window, each axis of C2, D and B is cleaned with a least-squares fit of C1 (3 axes x 8 delays); C1 itself is used as it is. On the 4-sensor walking recording this is what makes the reading follow the Polar. With 3 sensors this step is skipped: there all sensors carry the heartbeat and subtracting did not help.
3. **Beat envelope** (60 ms RMS, 100 Hz) and its **autocorrelation** over a 10 s window, for every axis of every working sensor, averaged. All three sensors pick up the heartbeat (checked on the sitting/standing recordings), so they are added, not subtracted.
4. **Score for every rate 40-150 bpm**: 0.75 x autocorrelation at the beat period + 0.25 x at twice the period.
5. **HMM tracker**: once a second the probability over rates is spread a little (the rate may drift about 1 bpm/s) and multiplied by the new score. Footstep peaks jump around, the heartbeat stays put, so it accumulates. Status `locked` = at least half of the probability within 3 bpm, otherwise `\~NN BPM (niska pewność)`.

The RLS canceller (below) is still there, but only for the RAW/CLEAN plot and `live\_debug.py`: subtracting the references also subtracts the heartbeat they contain.

## How the previous version works (`live\_rls.py`)

1. **Band-pass 5-25 Hz** (4th-order Butterworth) on every channel keeps the heartbeat vibration and drops slow body motion.
2. **Motion gate**: the filter only learns and subtracts while the reference sensors show motion above an adaptive threshold (2.5x their own quiet level). At rest the output is the raw signal, so the filter cannot learn to remove the heartbeat that leaks into the references.
3. **RLS canceller**: for each chest axis it predicts the chest motion from the last 6 samples of all 6 reference channels (36 weights) and subtracts it. Forgetting factor 0.998, about 1 s memory.
4. **Beat envelope and autocorrelation** on the 3 chest axes plus their strongest combination give heart rate candidates (40-150 bpm).
5. **Step cadence rejection**: the walking rhythm is measured from the references, and candidates sitting exactly on it are penalised.
6. **Tracker**: picks the candidate consistent with the previous rate, holds the last value for up to 8 s when the signal is bad, and shows a low-confidence guess when there is no lock.

On-screen status: `locked`, `holding`, `\~NN bpm (low confidence)`, or `--`.

## Tuning

At the top of `live\_scg.py`:

|Setting|Meaning|
|-|-|
|`HR\_BAND`|band used for the heart rate (20-60 Hz)|
|`HR\_WINDOW\_SECONDS`|autocorrelation window|
|`HR\_NOISE\_REF\_4`|4 sensors: which sensor is subtracted from the others (0 = C1, `None` = off)|
|`HR\_CANCEL\_TAPS`|delays used in that subtraction|
|`TRACK\_SIGMA\_BPM`|how fast the rate may change per second; raise it if the reading lags behind real changes|
|`TRACK\_BETA`|how much one window is trusted; lower = smoother, higher = jumpier|
|`LOCK\_MASS`|confidence needed for `locked`|
|`MOVE\_RMS`|minimum reference RMS counted as motion (RLS / plot only)|

At the top of `live\_rls.py`:

|Setting|Meaning|
|-|-|
|`MOVE\_RMS`|minimum reference RMS counted as motion; lower it if the title says "still" while you move|
|`LAM`|RLS memory; lower is faster but noisier|
|`L`|taps per reference channel|
|`ACCEPT`, `KEEP`|score needed to lock on / keep following a rate|

## Tips

* Mount the sensors firmly. A loose back sensor measures motion the chest never feels, and cancellation cannot fix that.
* Check roles with a tap test: tap each sensor and see which columns react (chest columns 2-4, device 5-7, back 8-10 in the serial stream).
* For ground truth, note the Polar H10 value while recording each movement.



