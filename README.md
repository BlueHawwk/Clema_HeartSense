# Heartbeat sensor over a t-shirt (4 sensors, ESP32)

Heart rate through clothing from chest accelerometers (SCG), for a TEC cooling garment. Four LSM6DS3TR-C sensors are read by an ESP32; `live_scg_3.py` computes the heart rate live and is checked against a Polar H10. Research prototype, not a medical device.

## Files
- `4_sensors/4_sensors.ino`: ESP32 firmware, prints CSV (500 lines/s)
- `live_scg_3.py`: live heart rate, logging, and replay of recordings

Install once: `py -m pip install pyserial pyqtgraph PyQt6 scipy numpy pandas`

## Wiring (software SPI)
Every sensor: VIN to 3V3, GND to GND, SCL to GPIO18, DO to GPIO19, SDA to GPIO23, and its own CS pin:

| Sensor | CS pin |
|---|---|
| chest-1 | GPIO17 |
| chest-2 | GPIO4 |
| device | GPIO5 |
| back | GPIO16 |

## Firmware
Upload `4_sensors.ino` (ESP32 board, your COM port). Close the Serial Monitor afterwards, it blocks the port.
- Output at 921600 baud: `t_us,c1x,c1y,c1z,c2x,c2y,c2z,dx,dy,dz,bx,by,bz` in m/s², 500 lines/s. Lines starting with `#` are messages.
- Each sensor is read at 6.66 kHz and averaged into every 2 ms line. A missing sensor prints `0.000`.
- Range is ±2 g. For ±4 g set `CTRL1_XL_VALUE 0xA8` and `MMS2_PER_COUNT 1.196f` together.

## Live filter
```
py live_scg_3.py COM7          live (chest-2 is the chest sensor)
py live_scg_3.py COM7 c1       live (chest-1)
py live_scg_3.py file.csv      replay a recording, prints the error vs Polar if the file has polar_bpm
```
Keys (click the plot window, **never the black console**, it freezes and loses data): `1` sitting, `2` standing, `3` walk_slow, `4` walk_fast, `5` arm_swing, `6` turning, `7` stairs, `8` carrying, `9` talking, `0` other, `H` + digits + Enter = Polar bpm, `Q` = quit and save.
Output: `scg_data_*.csv` (all rows + label + polar, can be replayed) and `scg_hr_*.csv` (one row per second).

How it works: heart rate is measured in the 20-60 Hz band (above the footstep impacts) from the energy envelope of every sensor axis, autocorrelated and averaged, then followed by a probabilistic tracker. When there is no clear rhythm it shows the last good reading ("holding, N s old").

Main settings (top of the file): `NSENS` (4 or 3 sensors), `HR_SENSORS` and `CADENCE_SENSORS` (which sensors give the heart rate / act as references), `QUALITY_MIN_SCORE` (0.10; lower shows more numbers), `HOLD_LAST_GOOD`.
Best walking result in the tests: chest-1 + chest-2 for the heart rate, device + back as references: `HR_SENSORS = (0, 3)`, `CADENCE_SENSORS = (1, 2)`.

## Results (mean error vs Polar)
| | Walking | Sitting |
|---|---|---|
| default (all 4 sensors) | 4.3 bpm | 1.1 bpm |
| chest-1 + chest-2 only | 3.6 bpm | 1.1 bpm |
| chest-2 only | 5.1 bpm | 2.8 bpm |

One sitting (10 min) and one walking (16 min) recording, one person; settings were chosen on the same data. The first minute after a start is weak. Heart rate hardly changed during the walk (85-100 bpm), so bigger changes are untested.

## Troubleshooting
- `can't open file`: run the command in the folder that contains the file.
- Only usage text appears: you forgot the port (`COM7`).
- `could not open port`: close the Serial Monitor, check the COM number.
- "DATA LOST" or fewer than 500 rows/s: you clicked the console window.
