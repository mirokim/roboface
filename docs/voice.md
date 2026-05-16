# 음성 어시스턴트 (Voice Assistant)

USB 스피커폰 + 웨이크 워드 + OpenAI Whisper(STT) + OpenAI TTS + Anthropic Claude.

## 흐름

```
[USB 스피커폰 마이크]
    ↓ 16kHz/mono PCM
[VAD + Wake Word (Porcupine "jarvis")]
    ↓ wake 감지
[VAD 발화 녹음 (silence 0.7s)]
    ↓ WAV
[OpenAI Whisper API]
    ↓ 한국어 텍스트
[Claude (대화 히스토리 포함)]
    ↓ 1~2문장 응답
[OpenAI TTS (mp3 → PCM)]
    ↓ RMS 윈도우 → mouth_state.talk_amplitude
[USB 스피커폰 출력 + LCD 립싱크]
```

## 의존성

```bash
# Pi 5에서
sudo apt install -y ffmpeg portaudio19-dev
source ~/roboface/.venv/bin/activate
pip install -r requirements-pi.txt
```

필수 시스템 패키지:
- `ffmpeg` — TTS mp3 → PCM 디코딩 (pydub 백엔드)
- `portaudio19-dev` — sounddevice 빌드용

## 환경 변수 (`.env`)

```bash
# OpenAI (Whisper STT + TTS)
OPENAI_API_KEY=sk-...

# Picovoice Porcupine — https://console.picovoice.ai 무료 발급
PORCUPINE_ACCESS_KEY=...

# 빌트인 키워드 (대소문자 무관)
# jarvis | computer | hey google | hey siri | alexa | americano | bumblebee
# blueberry | grapefruit | grasshopper | picovoice | porcupine | terminator
PORCUPINE_KEYWORD=jarvis

# 한국어 커스텀 키워드 사용 시
# PORCUPINE_KEYWORD_PATH=/home/miro/roboface/data/wake_words/안녕로보_ko_raspberry-pi.ppn

# Anthropic (대화)
ANTHROPIC_API_KEY=sk-ant-...

# 마이크 디바이스 — 비우면 시스템 기본
# 디바이스 인덱스 또는 이름의 일부 (sounddevice가 substring 매칭)
# AUDIO_INPUT_DEVICE=USB
```

## 마이크/스피커 확인

```bash
# 디바이스 목록
python -c "import sounddevice; print(sounddevice.query_devices())"

# 기본 마이크 5초 녹음 → 재생
python -c "
import sounddevice as sd, soundfile as sf
rec = sd.rec(5 * 16000, samplerate=16000, channels=1)
sd.wait()
sf.write('/tmp/test.wav', rec, 16000)
print('녹음 완료. aplay /tmp/test.wav 로 재생 확인')
"
```

스피커폰이 인식 안 되면 `arecord -l` / `aplay -l` 로 ALSA 레벨 확인.

## 동작 확인

```bash
# 로그 확인
journalctl -u roboface -f | grep -E "voice_assistant|wake_word|stt|tts"

# 발화 트리거 → 응답 흐름
# 1. "Jarvis" 외침
# 2. LCD 표정 FOCUSED (LISTENING)
# 3. 질문 ("오늘 몇 시야?")
# 4. LCD 입 움직임 + 스피커폰에서 응답 음성
```

## 트러블슈팅

### Wake word 감지 안 됨
- `PORCUPINE_ACCESS_KEY` 누락? `journalctl -u roboface | grep Porcupine`
- 마이크 게인 낮음 → `alsamixer` 에서 Capture 볼륨 올리기
- 키워드 발음 또렷이. `sensitivity` 기본 0.6 → 코드에서 0.7~0.8로 올려도 됨

### "마이크 초기화 실패"
- `portaudio19-dev` 설치 확인
- 디바이스 충돌 — 다른 프로세스가 점유하고 있을 수 있음 (`fuser /dev/snd/*`)

### TTS 음성 안 나옴 / mp3 디코딩 실패
- `ffmpeg` 시스템 설치 필요 (`which ffmpeg`)
- `pydub` 설치 확인

### STT 결과 항상 빈 문자열
- 마이크 입력은 잘 되는데 OpenAI가 인식 못하는 경우
- `record_utterance`가 너무 짧게 자르는지 확인 — `silence_ms` 늘려보기

### 비용 (대략)
- Whisper: $0.006/min
- TTS (gpt-4o-mini-tts): $0.015/1K chars
- Claude Haiku 4.5: $1/M input + $5/M output tokens (prompt caching 적용)
- 하루 30턴 대화 가정 시 < $0.50/일

## 끄고 싶다면

`.env`에서 `OPENAI_API_KEY` 또는 `PORCUPINE_ACCESS_KEY` 비우면 voice_assistant task가
자동으로 종료됨 (다른 task는 영향 없음).
