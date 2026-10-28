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
* The tracker so far has been tested on simulated data and informal runs, not on a proper dataset. Contributions with recordings plus reference heart rates are very welcome.

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
python/live\_rls.py                 live heart rate with motion cancellation
python/live\_debug.py               live view of every signal and what is subtracted
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

Columns: time in microseconds, then x/y/z acceleration in m/s^2 for chest, device and back.
A sensor that is down prints zeros and the firmware retries it every 500 ms.
Lines starting with `#` are status messages.

### 2\. Python

```
py -m pip install -r requirements.txt
cd python
py live\_rls.py
```

* Close the Arduino Serial Monitor first, only one program can use the COM port.
* Set `PORT` (default `COM5`) and `BAUD` (default `921600`) at the top of `live\_rls.py`.
* Replay a recording: `py live\_rls.py log\_rls\_YYYYMMDD\_HHMMSS.csv` (prints heart rate once per second).
* `py live\_debug.py` shows five panels: chest raw vs the estimate being subtracted, the result, device axes, back axes, and the motion gate. Needs `live\_rls.py` in the same folder.

Every live run writes a timestamped `log\_\*.csv` with the raw stream.

## How it works

1. **Band-pass 5-25 Hz** (4th-order Butterworth) on every channel keeps the heartbeat vibration and drops slow body motion.
2. **Motion gate**: the filter only learns and subtracts while the reference sensors show motion above an adaptive threshold (2.5x their own quiet level). At rest the output is the raw signal, so the filter cannot learn to remove the heartbeat that leaks into the references.
3. **RLS canceller**: for each chest axis it predicts the chest motion from the last 6 samples of all 6 reference channels (36 weights) and subtracts it. Forgetting factor 0.998, about 1 s memory.
4. **Beat envelope and autocorrelation** on the 3 chest axes plus their strongest combination give heart rate candidates (40-150 bpm).
5. **Step cadence rejection**: the walking rhythm is measured from the references, and candidates sitting exactly on it are penalised.
6. **Tracker**: picks the candidate consistent with the previous rate, holds the last value for up to 8 s when the signal is bad, and shows a low-confidence guess when there is no lock.

On-screen status: `locked`, `holding`, `\~NN bpm (low confidence)`, or `--`.

## Tuning

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



