#include <Arduino.h>
#include "soc/soc.h"
#include "soc/gpio_reg.h"

// Four LSM6DS3TR-C on an ESP32 -> CSV lines on the USB serial port, for record_4.py.
// Same fast software SPI and detection as 4_sensors_detect / 4_sensors_test.
//
// Each sensor is read at 6.66 kHz (only when it has new data). Every OUTPUT_PERIOD_US
// the readings of that period are averaged and ONE line is printed:
//     t_us,c1x,c1y,c1z,c2x,c2y,c2z,dx,dy,dz,bx,by,bz        (m/s^2, 3 decimals)
// 2000 us = 500 lines/s (the same rate as your earlier recordings).
// Lines starting with '#' are messages (sensor found/lost); the recorder skips them.
// A sensor that is missing or lost prints 0.000 in its columns.
//
//   Shared lines:  SCK = GPIO18   MISO = GPIO19   MOSI = GPIO23
//   Chip selects:  chest-1 = GPIO17   chest-2 = GPIO4   device = GPIO5   back = GPIO16

#define PIN_SCK  18
#define PIN_MISO 19
#define PIN_MOSI 23

#define SENSOR_COUNT 4
const char*   ROLE[SENSOR_COUNT]    = {"chest-1", "chest-2", "device", "back"};
const uint8_t CS_PINS[SENSOR_COUNT] = {17, 4, 5, 16};

#define BAUD             921600
#define OUTPUT_PERIOD_US 2000          // 2000 = 500 Hz, 1000 = 1000 Hz (needs ~110 B/line: fine up to ~800 lines/s at 921600)

// +-2 g : 0xA0, scale 0.598 mm/s^2 per count.   +-4 g : 0xA8, scale 1.196.
// Use +-4 g if you walk (values above 19.6 m/s^2 would otherwise clip).
#define CTRL1_XL_VALUE 0xA0
#define MMS2_PER_COUNT 0.598f          // set 1.196f together with 0xA8

#define REG_WHO_AM_I   0x0F
#define REG_CTRL1_XL   0x10
#define REG_CTRL2_G    0x11
#define REG_CTRL3_C    0x12
#define REG_CTRL4_C    0x13
#define REG_STATUS_REG 0x1E
#define REG_OUTX_L_XL  0x28

#define SCK_MASK  (1UL << PIN_SCK)
#define MISO_MASK (1UL << PIN_MISO)
#define MOSI_MASK (1UL << PIN_MOSI)
#define GPIO_HIGH(mask) REG_WRITE(GPIO_OUT_W1TS_REG, mask)
#define GPIO_LOW(mask)  REG_WRITE(GPIO_OUT_W1TC_REG, mask)
#define GPIO_READ(mask) ((REG_READ(GPIO_IN_REG) & mask) != 0)

bool     sensorUp[SENSOR_COUNT];
uint8_t  badReads[SENSOR_COUNT];
int32_t  sum[SENSOR_COUNT][3];
uint16_t cnt[SENSOR_COUNT];
uint32_t startUs, nextOut;
uint32_t lines = 0;

inline void selectSensor(uint8_t i)   { GPIO_LOW(1UL << CS_PINS[i]); }
inline void deselectSensor(uint8_t i) { GPIO_HIGH(1UL << CS_PINS[i]); }

inline uint8_t spiTransferFast(uint8_t data) {
  uint8_t received = 0;
  for (uint8_t mask = 0x80; mask; mask >>= 1) {
    GPIO_LOW(SCK_MASK);
    if (data & mask) GPIO_HIGH(MOSI_MASK); else GPIO_LOW(MOSI_MASK);
    GPIO_HIGH(SCK_MASK);
    received <<= 1;
    if (GPIO_READ(MISO_MASK)) received |= 1;
  }
  GPIO_LOW(SCK_MASK);
  return received;
}

uint8_t readRegister(uint8_t i, uint8_t reg) {
  selectSensor(i);
  spiTransferFast(reg | 0x80);
  uint8_t v = spiTransferFast(0x00);
  deselectSensor(i);
  return v;
}

void writeRegister(uint8_t i, uint8_t reg, uint8_t value) {
  selectSensor(i);
  spiTransferFast(reg & 0x7F);
  spiTransferFast(value);
  deselectSensor(i);
}

bool detectAndInit(uint8_t i) {
  if (readRegister(i, REG_WHO_AM_I) != 0x6A) return false;
  writeRegister(i, REG_CTRL3_C, 0x01);   // software reset
  delay(20);
  writeRegister(i, REG_CTRL3_C, 0x44);   // BDU + IF_INC
  writeRegister(i, REG_CTRL4_C, 0x04);   // I2C off
  writeRegister(i, REG_CTRL2_G, 0x00);   // gyro off
  writeRegister(i, REG_CTRL1_XL, CTRL1_XL_VALUE);
  delay(5);
  return readRegister(i, REG_CTRL1_XL) == CTRL1_XL_VALUE;
}

void pollSensor(uint8_t i) {
  if (!(readRegister(i, REG_STATUS_REG) & 0x01)) return;      // no new sample yet
  uint8_t b[6];
  selectSensor(i);
  spiTransferFast(REG_OUTX_L_XL | 0x80);
  for (uint8_t k = 0; k < 6; k++) b[k] = spiTransferFast(0x00);
  deselectSensor(i);

  bool allFF = true, all00 = true;
  for (uint8_t k = 0; k < 6; k++) { if (b[k] != 0xFF) allFF = false; if (b[k] != 0x00) all00 = false; }
  if (allFF || all00) {
    if (++badReads[i] >= 5 && sensorUp[i]) { sensorUp[i] = false; Serial.printf("# %s LOST\n", ROLE[i]); }
    return;
  }
  badReads[i] = 0;
  sum[i][0] += (int16_t)((b[1] << 8) | b[0]);
  sum[i][1] += (int16_t)((b[3] << 8) | b[2]);
  sum[i][2] += (int16_t)((b[5] << 8) | b[4]);
  cnt[i]++;
}

// writes value (in 0.001 m/s^2) as text with 3 decimals, returns the new end of the buffer
char* putMilli(char* p, int32_t v) {
  if (v < 0) { *p++ = '-'; v = -v; }
  uint32_t whole = v / 1000, frac = v % 1000;
  char tmp[8]; uint8_t n = 0;
  do { tmp[n++] = '0' + whole % 10; whole /= 10; } while (whole);
  while (n) *p++ = tmp[--n];
  *p++ = '.';
  *p++ = '0' + frac / 100; *p++ = '0' + (frac / 10) % 10; *p++ = '0' + frac % 10;
  return p;
}

void emitLine(uint32_t tUs) {
  char line[160]; char* p = line;
  uint32_t t = tUs; char tmp[12]; uint8_t n = 0;
  do { tmp[n++] = '0' + t % 10; t /= 10; } while (t);
  while (n) *p++ = tmp[--n];
  for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
    for (uint8_t a = 0; a < 3; a++) {
      *p++ = ',';
      int32_t milli = 0;
      if (sensorUp[i] && cnt[i]) milli = (int32_t)lroundf((float)sum[i][a] / cnt[i] * MMS2_PER_COUNT);
      p = putMilli(p, milli);
    }
    sum[i][0] = sum[i][1] = sum[i][2] = 0; cnt[i] = 0;
  }
  *p++ = '\n';
  Serial.write((uint8_t*)line, p - line);
}

void startRecordingClock() {
  startUs = micros();
  nextOut = startUs + OUTPUT_PERIOD_US;
}

void setup() {
  Serial.setTxBufferSize(8192);
  Serial.begin(BAUD);
  delay(500);
  pinMode(PIN_SCK, OUTPUT); pinMode(PIN_MOSI, OUTPUT); pinMode(PIN_MISO, INPUT);
  GPIO_LOW(SCK_MASK);
  for (uint8_t i = 0; i < SENSOR_COUNT; i++) { pinMode(CS_PINS[i], OUTPUT); deselectSensor(i); }
  delay(100);

  uint8_t found = 0;
  for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
    sensorUp[i] = detectAndInit(i);
    if (sensorUp[i]) found++;
    Serial.printf("# %s (CS %u): %s\n", ROLE[i], CS_PINS[i], sensorUp[i] ? "OK" : "NOT FOUND");
  }
  Serial.printf("# %u of %u sensors, %u lines/s\n", found, SENSOR_COUNT, 1000000UL / OUTPUT_PERIOD_US);
  Serial.println("t_us,c1x,c1y,c1z,c2x,c2y,c2z,dx,dy,dz,bx,by,bz");
  startRecordingClock();
}

void loop() {
  for (uint8_t i = 0; i < SENSOR_COUNT; i++) if (sensorUp[i]) pollSensor(i);
  if ((int32_t)(micros() - nextOut) >= 0) {
    emitLine(nextOut - startUs);
    nextOut += OUTPUT_PERIOD_US;
    if (++lines % 5000 == 0) {                 // every ~10 s: try to bring back lost sensors
      for (uint8_t i = 0; i < SENSOR_COUNT; i++) if (!sensorUp[i] && detectAndInit(i)) {
        sensorUp[i] = true; badReads[i] = 0; Serial.printf("# %s back\n", ROLE[i]);
      }
    }
  }
}
