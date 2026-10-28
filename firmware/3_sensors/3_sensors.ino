#include <ESP8266WiFi.h>
#include <Wire.h>

const uint8_t NS = 3;
const char*   ROLE[NS]    = {"chest", "device", "back"};
const uint8_t BUS_OF[NS]  = {0, 0, 1};
const uint8_t ADDR_OF[NS] = {0x6A, 0x6B, 0x6A};    // device: DO tied to 3V gives 0x6B
const uint8_t SDA_PIN[2]  = {D2, D6};
const uint8_t SCL_PIN[2]  = {D1, D5};

const uint32_t PERIOD_US = 2000;       // 500 Hz
const float    ACC_SCALE = 0.000598f;  // m/s^2 per count (+-2 g)

int8_t   curBus = -1;
bool     up[NS];
uint8_t  fails[NS];
uint32_t retryAt[NS];
uint32_t nextT;

void selectBus(uint8_t b) {
  if (curBus == (int8_t)b) return;
  Wire.begin(SDA_PIN[b], SCL_PIN[b]);
  Wire.setClock(400000);
  curBus = b;
}

void busClear(uint8_t b) {             // free a stuck bus by clocking SCL
  pinMode(SDA_PIN[b], INPUT_PULLUP);
  for (uint8_t k = 0; k < 9; k++) {
    pinMode(SCL_PIN[b], OUTPUT);
    digitalWrite(SCL_PIN[b], LOW);
    delayMicroseconds(5);
    pinMode(SCL_PIN[b], INPUT_PULLUP);
    delayMicroseconds(5);
  }
  curBus = -1;
}

bool wr(uint8_t i, uint8_t reg, uint8_t val) {
  selectBus(BUS_OF[i]);
  Wire.beginTransmission(ADDR_OF[i]);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool rd(uint8_t i, uint8_t reg, uint8_t* buf, uint8_t n) {
  selectBus(BUS_OF[i]);
  Wire.beginTransmission(ADDR_OF[i]);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(ADDR_OF[i], n) != n) return false;
  for (uint8_t k = 0; k < n; k++) buf[k] = Wire.read();
  return true;
}

bool tryInit(uint8_t i) {
  uint8_t id = 0;
  if (!rd(i, 0x0F, &id, 1) || id != 0x6A) return false;
  wr(i, 0x12, 0x01);                   // software reset
  delay(20);
  return wr(i, 0x12, 0x44) && wr(i, 0x10, 0x70);   // BDU + auto-increment, 833 Hz +-2 g
}

void setup() {
  WiFi.mode(WIFI_OFF);
  Serial.begin(921600);
  delay(300);
  Serial.println();
  for (uint8_t i = 0; i < NS; i++) {
    up[i] = tryInit(i);
    fails[i] = 0;
    retryAt[i] = millis() + 500;
    Serial.print("# ");
    Serial.print(ROLE[i]);
    Serial.println(up[i] ? " sensor OK" : " sensor NOT FOUND");
  }
  Serial.println("t,cx,cy,cz,dx,dy,dz,bx,by,bz");
  nextT = micros();
}

void loop() {
  if ((int32_t)(micros() - nextT) < 0) return;
  nextT += PERIOD_US;
  if ((int32_t)(micros() - nextT) > 5 * (int32_t)PERIOD_US) nextT = micros() + PERIOD_US;

  float    v[NS][3] = {};
  int8_t   event[NS] = {};
  uint32_t now = millis();

  for (uint8_t i = 0; i < NS; i++) {
    if (up[i]) {
      uint8_t b[6];
      if (rd(i, 0x28, b, 6)) {
        fails[i] = 0;
        for (uint8_t k = 0; k < 3; k++)
          v[i][k] = (int16_t)(b[2 * k] | (b[2 * k + 1] << 8)) * ACC_SCALE;
      } else if (++fails[i] >= 3) {
        up[i] = false;
        retryAt[i] = now + 500;
        event[i] = -1;
        busClear(BUS_OF[i]);
      }
    } else if (now >= retryAt[i]) {
      if (tryInit(i)) { up[i] = true; fails[i] = 0; event[i] = 1; }
      else retryAt[i] = millis() + 500;
    }
  }

  for (uint8_t i = 0; i < NS; i++) {
    if (event[i]) {
      Serial.print("# ");
      Serial.print(ROLE[i]);
      Serial.println(event[i] > 0 ? " sensor back" : " sensor LOST");
    }
  }

  char line[160];
  int n = snprintf(line, sizeof(line),
                   "%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   (unsigned long)micros(),
                   v[0][0], v[0][1], v[0][2],
                   v[1][0], v[1][1], v[1][2],
                   v[2][0], v[2][1], v[2][2]);
  Serial.write((const uint8_t*)line, n);
}