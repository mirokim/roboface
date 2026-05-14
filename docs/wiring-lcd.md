# 2.4" LCD (ILI9341) SPI 결선 가이드

**모듈**: Waveshare 2.4" LCD Display Module — ILI9341 4-wire SPI, 240×320 (가로 사용 시 320×240)

## 핀맵 (LCD 모듈 ↔ Pi 5)

| LCD 핀 | 기능 | Pi 5 BCM | T-Cobbler 라벨 | 물리 핀 |
|---|---|---|---|---|
| **VCC** | 전원 (3.3V) | — | `3V3` | 1 또는 17 |
| **GND** | 그라운드 | — | `GND` | 6/9/etc |
| **DIN** | SPI MOSI (데이터) | GPIO 10 | `MOSI` 또는 `G10` | 19 |
| **CLK** | SPI 클럭 | GPIO 11 | `SCLK` 또는 `G11` | 23 |
| **CS** | Chip Select | **GPIO 5** ⚠️ | `GPIO5` 또는 `G5` | 29 |
| **DC** | Data/Command | GPIO 25 | `G25` | 22 |
| **RST** | Reset | GPIO 27 | `G27` | 13 |
| **BL** | Backlight | GPIO 18 | `G18` | 12 |

총 **8 가닥**.

## 결선 시 주의

1. **VCC는 3.3V** — 5V 가도 동작은 하지만 일부 모듈은 손상 가능. 3.3V 권장.
2. **CS는 GPIO 5 사용** — Pi 5에서 CE0(GPIO 8)은 spidev 드라이버가 점유해서
   digitalio로 잡으면 "GPIO busy" 에러. GPIO 5는 free하고 가까이 있음.
3. **점퍼선 색 권장**:
   - VCC: 빨강
   - GND: 검정
   - 데이터/제어: 노랑/초록/주황/파랑 (구분 위해 다양하게)

## 결선 절차

1. Pi 안전 종료: `sudo shutdown -h now` → 녹색 LED 꺼지면 USB-C 분리
2. LCD 핀 ↔ T-Cobbler 행 점퍼선 연결 (위 표 참고)
3. 두 번 확인 (특히 VCC/GND, MOSI/MISO 혼동 X)
4. 전원 인가 → SSH 접속

## 첫 동작 확인

SPI 활성화돼있는지 (이미 해뒀어야):
```bash
ls -la /dev/spi*
# /dev/spidev0.0 가 보여야 함
```

LCD 라이브러리 설치:
```bash
cd ~/roboface
source .venv/bin/activate
pip install adafruit-circuitpython-rgb-display pillow
```

테스트 스크립트로 빨간/녹색/파란 화면 띄우기:
```bash
python -m src.face.lcd_test
```

성공 시 LCD에 RED → GREEN → BLUE 순차 표시.

그 다음:
```bash
python -m src.main_robot
```

→ 진짜 얼굴 표시.

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| 화면 새까만데 백라이트 안 켜짐 | BL 핀(GPIO 18) 결선 또는 VCC/GND 확인 |
| 흰 화면 (전원만 들어옴) | DIN/CLK/CS 중 하나 잘못 연결, SPI 활성화 안 됨 |
| 부분만 그려짐 / 깨짐 | rotation 잘못 설정 또는 baudrate 너무 높음 |
| `No module named 'board'` | `pip install adafruit-blinka` |
| `No SPI device` | `sudo raspi-config` → Interface → SPI Enable |
