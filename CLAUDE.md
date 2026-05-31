# Roboface — 프로젝트 가이드 (Claude 전용)

라즈베리파이 5 + 2.4" LCD + Pan/Tilt 서보 + AI 카메라(IMX500) 기반의
책상 위 동반자 로봇. 시뮬레이터(Pygame) ↔ 실기(LCD) 양쪽 모드.

이 파일은 Claude(에이전트 / Claude Code 세션)가 매번 처음부터 코드 다 읽지 않아도
로봇 스펙, 아키텍처, 도구, 변경 시 주의점을 빠르게 파악하도록 정리한 SSOT 인덱스다.

---

## 0. 모드와 진입점

| 모드 | 진입점 | 환경변수 |
|---|---|---|
| simulator | `python -m src.main` | `ROBOFACE_MODE=simulator` |
| robot (Pi 5) | `python -m src.main_robot` | `ROBOFACE_MODE=robot` |

`is_simulator()` / `is_robot()` ([src/config.py](src/config.py))로 분기.

---

## 1. 하드웨어 스펙 (robot 모드)

| 부품 | 모델 | 인터페이스 |
|---|---|---|
| SBC | Raspberry Pi 5 (8GB) | — |
| 디스플레이 | Waveshare 2.4" LCD (ILI9341, 320×240) | SPI0 (DC=25, RST=27, BL=18, CS=GPIO5 워크어라운드) |
| 서보 | SG92R 270° × 2 (Pan + Tilt) | PCA9685 PWM 드라이버 (I2C, addr 0x40, ch 0/1) |
| 모션센서 | S3KM1110 24GHz mmWave | UART `/dev/serial0` @ 115200 |
| 온습도 | DHT22 | GPIO 22 |
| 카메라 | Raspberry Pi AI Camera (Sony IMX500 + RP2040) | CSI mini-22pin |
| 마이크 | COMS CM421 USB 컨퍼런스 마이크 (전방위) | USB-A. 신호 약함(peak 2-4k) — software gain + OpenAI Whisper로 보정 |
| 스피커 | **없음** | TTS는 비활성(`TTS_DISABLED=1`), LCD 말풍선만 |

### 서보 가동 범위 (소프트 리밋, [src/config.py](src/config.py))
- Pan: `PAN_MIN_DEG=45° ~ PAN_CENTER_DEG=135° ~ PAN_MAX_DEG=225°`
- Tilt: `TILT_MIN_DEG=50° ~ TILT_CENTER_DEG=90° ~ TILT_MAX_DEG=130°`
- 리밋은 [src/motion/servos.py](src/motion/servos.py)의 `_clamp()`가 `set_angles()`에서 강제.

---

## 2. 모듈 구조 (한 줄 요약)

```
src/
├── config.py              ← 하드웨어 핀맵 + BehaviorConfig (행동 파라미터 SSOT)
├── main.py                ← simulator 진입점 (pygame 윈도우)
├── main_robot.py          ← robot 진입점 (LCD + 센서)
├── face/                  ← 표정 렌더링
│   ├── expressions.py     ← EXPRESSIONS_BY_NAME (22종 표정 SSOT)
│   ├── eyes.py, mouth.py  ← 눈/입 그리기
│   ├── extras.py          ← 폰트, 말풍선, 땀, 떨림
│   ├── renderer.py        ← PygameRenderer + FaceState
│   └── lcd_renderer.py    ← ILI9341 SPI 출력 (robot 전용)
├── brain/                 ← 두뇌
│   ├── state_machine.py   ← State(IDLE/WATCHING/GREETING/TALKING/ALERTING/LISTENING) + StateContext
│   ├── perception.py      ← PerceptionState (사용자 위치, 온도)
│   ├── memory.py          ← SQLite (work_sessions, conversation_log, learned_facts, env_log…) WAL 모드
│   ├── triggers.py        ← ProactiveTrigger 평가기 + TRIGGER_EXPRESSIONS SSOT
│   ├── conversation.py    ← Claude API 래퍼 + SYSTEM_PROMPT (image 첨부 지원)
│   ├── conversation_templates.py  ← 제스처 즉시 응답 멘트 SSOT (hands_up/nod/shake/gaze)
│   ├── agent.py           ← RobotAgent (Claude tool-use + multi-turn + vision)
│   ├── image_encoding.py  ← numpy RGB → JPEG base64 (agent vision용)
│   └── time_of_day.py     ← 시간대(morning/lunch/afternoon/evening/late) SSOT
├── vision/                ← AI Camera 처리
│   ├── camera.py          ← IMX500 wrapper (VISION_MODE=detect/pose)
│   ├── person_detector.py ← 사람 감지 debouncing
│   ├── pose_gestures.py   ← wave / hands_up / nod / shake / gaze 인식
│   ├── face_memory.py     ← 얼굴 등록/인식
│   ├── emotion_mirror.py  ← 사용자 표정 → 로봇 표정 거울
│   └── photo_memory.py    ← 표정 캡처 + 통계
├── audio/                 ← 음성 (마이크 도착 시 활성)
│   ├── mic.py             ← Microphone (sounddevice + webrtcvad)
│   ├── wake_word.py       ← Porcupine
│   ├── stt.py             ← OpenAI Whisper
│   ├── tts.py             ← OpenAI TTS + 립싱크 (mouth.shape_for_amp 공유)
│   ├── fake_tts.py        ← 시뮬용 가짜 발화
│   └── audio_monitor.py   ← 박수 / 음악 비트 감지
├── motion/                ← 서보
│   ├── servos.py          ← ServoController (Mock/PCA9685)
│   └── poses.py           ← nod, shake, greeting, dance
├── sensors/               ← 환경 센서
│   ├── base.py            ← Sensor ABC + SensorEventType
│   ├── manager.py         ← 폴링 + 이벤트 deque
│   ├── mmwave.py          ← S3KM1110 UART 파서
│   └── environment.py     ← DHT22 + 시뮬레이션
├── tasks/                 ← 백그라운드 비동기 태스크
│   ├── proactive_speaker.py  ← 트리거 → 멘트 → 발화
│   ├── behavior_speaker.py   ← 이벤트 즉시 멘트 (wave_back/reappear)
│   ├── voice_assistant.py    ← wake → STT → Claude → TTS
│   ├── voice_commands.py     ← STT 트리거 시스템 명령 ("디버그 모드"→폰 테더링)
│   ├── work_tracker.py       ← 작업 세션 + 휴식 임계
│   ├── posture_monitor.py    ← 자세 감시 + perception.posture_category 갱신
│   ├── activity_monitor.py   ← 60초 윈도우 keypoint 변동량 → activity_level
│   ├── ambient_listener.py   ← 주변 STT (현재 MockSTT)
│   ├── schedule_extractor.py ← 대화에서 일정 추출 → ThinkTank
│   ├── journal_writer.py     ← 저널 동기화
│   ├── vision_task.py        ← 카메라 메인 루프 (gaze_target/frame cache 포함)
│   ├── head_tracker.py       ← 서보로 머리 추적 + perception.head_pan/tilt 갱신
│   ├── eye_tracker.py        ← 시선 동공 이동
│   ├── reactive_face.py      ← flash_expression
│   ├── audio_reactive.py     ← 박수/음악 → 표정/모션
│   ├── thermal_state.py      ← 온도 → 떨림/땀 + face.env_* 미러
│   ├── mood_drift.py         ← 장기 기분 드리프트
│   └── idle_animation.py     ← idle gaze + ambient motion
├── integrations/
│   ├── thinktank/            ← ThinkTank(외부 시스템) 통합
│   │   ├── client.py         ← HTTP + 재시도
│   │   └── offline_queue.py  ← SQLite 오프라인 큐
│   └── weather.py            ← OpenWeather TTL 캐시 클라이언트 (agent 컨텍스트용)
└── utils/logger.py
```

---

## 3. SSOT (Single Source of Truth) 인덱스

값/매핑을 바꿀 때 **이 파일들만** 건드린다. 다른 곳에 같은 상수 박지 말 것.

| 무엇 | 어디 | 비고 |
|---|---|---|
| 모드/핀맵/API 키 | [src/config.py](src/config.py) | 환경변수 우선 |
| 음성/STT 토글 env | [src/config.py](src/config.py) | `TTS_DISABLED`, `AMBIENT_LISTEN`, `WAKE_DISABLED`, `STT_BACKEND` (auto/local/openai), `STT_LOCAL_MODEL` (tiny/base/small) |
| 행동 파라미터 (대화 빈도, 휴식 임계, 깜빡임, agent vision 등) | [src/config.py](src/config.py) `BehaviorConfig` | 모든 task가 `BEHAVIOR.*`로 참조 |
| 입 모양 ↔ 음량 임계 | `BehaviorConfig.mouth_amp_thresholds` | mouth.py/tts.py 공유 — [src/face/mouth.py](src/face/mouth.py) `shape_for_amp()` |
| 표정 정의 | [src/face/expressions.py](src/face/expressions.py) `EXPRESSIONS_BY_NAME` | agent enum 자동 도출 |
| 상태 → 기본 표정 | [src/brain/state_machine.py](src/brain/state_machine.py) `_DEFAULT_EXPRESSIONS` | |
| 트리거 → 표정 | [src/brain/triggers.py](src/brain/triggers.py) `TRIGGER_EXPRESSIONS` | 누락 시 `expression_for()`가 KeyError |
| 제스처 응답 멘트 | [src/brain/conversation_templates.py](src/brain/conversation_templates.py) `GESTURE_POOLS` | hands_up/nod/shake/gaze. wave는 behavior_speaker가 SSOT |
| 시간대 분류 | [src/brain/time_of_day.py](src/brain/time_of_day.py) `period_for()` | morning/lunch/afternoon/evening/late |
| 시스템 프롬프트(캐릭터 보이스) | [src/brain/conversation.py](src/brain/conversation.py) `SYSTEM_PROMPT` + [src/brain/agent.py](src/brain/agent.py) `_AGENT_SYSTEM` | agent용은 후자가 우선 |
| 에이전트 도구 스키마 | [src/brain/agent.py](src/brain/agent.py) `_TOOLS` | speak/set_expression/dance/do_nothing/**recall**/**remember_fact** |
| 활동 추론 신호 | [src/brain/perception.py](src/brain/perception.py) `PerceptionState` | gaze_target/activity_level/posture_category/current_emotion/head_pan_deg/head_tilt_deg/last_frame |
| 시간대 힌트 (식사/오후 슬럼프 등) | [src/brain/agent.py](src/brain/agent.py) `_time_hint()` | agent 프롬프트에 주입 |
| 날씨 (OpenWeather) | [src/integrations/weather.py](src/integrations/weather.py) `WeatherClient` + `get_client()` 싱글톤 | `OPENWEATHER_API_KEY` 없으면 snapshot()→None (skip). `WEATHER_LAT/LON/LOCATION_NAME` env override. `BEHAVIOR.weather_cache_sec`(1800s) TTL + stale-while-error. agent `_tick`이 매번 await — 캐시 hit면 비용 0. `_build_situation_suffix(weather_line=...)` 한 줄 주입 |
| 센서 이벤트 enum | [src/sensors/base.py](src/sensors/base.py) `SensorEventType` | |
| DB 스키마 | [src/brain/memory.py](src/brain/memory.py) `SCHEMA_SQL` + `_MIGRATIONS` | WAL 모드. 테이블: work_sessions, conversation_log, proactive_log, schedules, env_log, user_patterns, face_snapshots, command_queue, **learned_facts** (category/source/last_used_at) |
| STT hallucination 필터 | [src/tasks/ambient_listener.py](src/tasks/ambient_listener.py) `_HALLUCINATION_PATTERNS` | 백엔드 무관 — "구독/좋아요/감사합니다/시청해" 등 한국어 Whisper 흔한 패턴. peak<400 가드도 같이 |
| 비언어 이벤트 (agent context 노이즈) | [src/brain/agent.py](src/brain/agent.py) `_NONVERBAL_KINDS` | orientation/presence/distance/gesture_* 카운트로 요약, 줄별 표시 X |

---

## 4. Claude가 로봇을 제어하는 두 가지 경로

### (a) 자율 결정 — Claude **에이전트**가 매 15초마다 도구 호출
[src/brain/agent.py](src/brain/agent.py) `RobotAgent`가 ANTHROPIC_API_KEY 있을 때만 활성.

행동 도구:
- `speak(text, expression?)` — 사용자에게 한두 문장 발화 (90초 쿨다운)
- `set_expression(expression)` — 표정만 변경 (22종 중 1)
- `dance(beats?, bpm?)` — 짧은 댄스
- `do_nothing()` — 침묵 유지 (기본은 이게 정답, 5번 중 3-4번)
- `remember_fact(key, value)` — 사용자에 대해 알게 된 사실 저장 (learned_facts 테이블)

정보 도구 (multi-turn — 결과 받고 행동 다시 결정):
- `recall(keyword, days?)` — conversation_log 검색. 오래된 대화 회상용. 최대 3 round 안에 행동 도구 호출해야 종료.

이미지 첨부 (agent vision):
- `BehaviorConfig.agent_vision_enabled=True` + 사용자 표정/활동성/시선 변화 시
  또는 max_interval_sec(10분)마다 한 번 — perception.last_frame을 JPEG로 인코딩해 첨부.
- min_interval_sec(60초) 안엔 무조건 skip. 비용 절감.

표정 이름은 `EXPRESSIONS_BY_NAME` 키 (대소문자 무관). 새 표정 추가하면 agent 도구
스키마가 자동으로 따라옴 (`_EXPRESSION_NAMES` 도출).

캐릭터 보이스: [src/brain/agent.py](src/brain/agent.py) `_AGENT_SYSTEM` 프롬프트 참조.
조용함·사려깊음·반말·이모지 X·1-2문장. 자기 머리/표정/카메라 일체 인식.

컨텍스트(매 tick `_build_situation`):
- 시각/사용자 이름/존재/거리/온도, 오늘 첫 등장/누적 작업, 최근 트리거
- 사용자 표정/시선 타깃/활동성/자세 (활동 추론 신호)
- 내 상태/표정/머리 방향
- 학습된 facts (최근 20개)
- 시간대 힌트 (점심/오후 슬럼프 등)
- 최근 대화 로그
- 24h 표정 통계 + 로봇 컨디션

### (b) 외부 수동 제어 — `scripts/robot_cli.py`
시뮬레이터/로봇이 떠 있을 때 별도 프로세스에서 명령. (Phase 5.2에서 작성)

---

## 5. 변경 시 규칙

- **하드코딩 상수는 BehaviorConfig에 넣고 거기서만 읽어라.** 모듈 상단에 매직 넘버 두지 말 것.
- **표정 추가**: [src/face/expressions.py](src/face/expressions.py)에 추가하고 `EXPRESSIONS_BY_NAME`에도 등록. 그 외 변경 불필요 — agent enum / 키 매핑 모두 자동.
- **새 트리거 추가**: [src/brain/triggers.py](src/brain/triggers.py)에 `check_*` 함수 + `TRIGGER_EXPRESSIONS`에 표정 매핑 추가. 누락 시 `expression_for()`가 즉시 KeyError로 알려줌.
- **제스처 응답 추가**: [src/brain/conversation_templates.py](src/brain/conversation_templates.py) `GESTURE_POOLS`에 추가.
- **DB 스키마 변경**: [src/brain/memory.py](src/brain/memory.py) `SCHEMA_SQL`은 idempotent. 컬럼 추가는 ALTER로 직접. 마이그레이션 시스템 아직 없음.
- **마이크 graceful fallback**: [src/main_robot.py](src/main_robot.py)는 `MicCaptureError` 잡아 audio_reactive task를 빼고, voice_assistant는 자체 graceful exit. 이 패턴 유지.
- **마이크 native rate 자동**: [src/audio/mic.py](src/audio/mic.py) `Microphone._negotiate_format()` — 디바이스가 16k mono 직접 지원 안 하면 native(예: 44.1k stereo)로 캡처 후 callback에서 numpy resample/mono 변환.
- **STT 백엔드 추상화**: [src/audio/stt.py](src/audio/stt.py) `create_stt()` 팩토리 — `STT_BACKEND=local`(faster-whisper)/`openai`(API)/`auto`(local→openai fallback). 동일 `.transcribe(wav_bytes) -> str` 인터페이스.
- **Wake word 백엔드 추상화**: [src/audio/wake_word.py](src/audio/wake_word.py) `create_wake_word()` — `PORCUPINE_ACCESS_KEY` 있으면 Picovoice, 없으면 openWakeWord(번들 onnx). 동일 `.process_pcm(bytes) -> bool` 인터페이스.

---

## 6. 테스트 / 디버그

```bash
pytest                      # 전체 테스트
pytest tests/test_emotion_mirror.py -v  # 특정 모듈
python -m src.face.lcd_test    # 실기에서 LCD 단독 확인 (robot 모드 필요)
python scripts/show_conversation.py  # 최근 대화 로그 출력
python scripts/robot_cli.py status   # 현재 상태 조회 (Phase 5.2)
```

로그: `LOG_LEVEL=DEBUG`로 자세히. 모듈별 prefix(`agent`, `proactive`, `vision`…)로 grep.

---

## 7. 현재 진행 (2026-05-31 기준)

### 음성 풀스택 가동

- 마이크 (CM421) → VAD → 로컬 faster-whisper or OpenAI Whisper → conversation_log → agent → LCD 말풍선
- `.env` 권장 셋: `AMBIENT_LISTEN=1`, `WAKE_DISABLED=1`, `TTS_DISABLED=1`, `STT_BACKEND=openai` (마이크 약해서 local small도 정확도 낮음)
- 박수/음악 비트는 audio_reactive가 별도 처리 — 박수 시 SURPRISED 표정만(끄덕 모션 X)
- **STT 청취 인디케이터**: VAD 발화 감지 시 `face.recording=True`(LCD 우상단 빨간 dot 깜빡임), 발화 끝 자동 off. `WhisperVADStreamer(face=...)` 주입, `VADRecorder(on_speech_start/on_speech_end)` 콜백 (finally로 stuck on 방지). (transcript echo는 거슬려서 제거 — recording dot만 유지)
- **음성 시스템 명령** ([src/tasks/voice_commands.py](src/tasks/voice_commands.py)): ambient transcript handler로 등록. "디버그 모드" 들리면 `nmcli connection up jhS26u`로 폰 테더링 wifi 전환. 회사선 폰 켜고 음성 한 번으로 전환 가능. 30s cooldown, FOCUSED→HAPPY(연결)/WORRIED(폰 꺼져있음/실패) 시각 피드백. NetworkManager 권한은 polkit rule(`/etc/polkit-1/rules.d/50-nm-roboface.rules`)로 miro 사용자 부여 — repo 외부 설치, robot setup 시 한 번만

### 인프라

- **무선 자동 업데이트**: `roboface-update.timer` 5분 주기로 git fetch → ff-pull → restart. 보너스: web UI에서 즉시 트리거. 자세한 건 [docs/setup-auto-update.md](docs/setup-auto-update.md)
- **wifi 우선순위**: 폰 핫스팟(`jhS26u`) priority 100, 회사 wifi autoconnect off — 회사선 폰 켜야 봇 온라인

### Agent 응답성

- `perception.last_user_speech_at` — STT 결과 들어오면 agent 2초 안에 tick
- speak cooldown(90s) bypass — 사용자 발화 8초 안엔 무조건 응답
- quiet hours(22~7시) bypass — 사용자 직접 발화엔 quiet 무시
- `_AGENT_SYSTEM` 최상단 hard rule — `📍 현재 대화 흐름`에 "아직 응답 안 함" 보이면 speak 강제
- `{MIC_GUIDANCE}` placeholder — `AMBIENT_LISTEN_ACTIVE`에 따라 마이크 안내 동적

### 메모리

- `learned_facts`에 `category`(self/user)/`source`(seed/agent/cli)/`last_used_at` 추가
- `remember_fact`가 schedule-like 감지 시 `schedules` 테이블로 라우팅
- agent prompt가 facts 노출할 때마다 `mark_facts_used()` — 활용도 추적
- 비언어 이벤트(orientation/presence/distance/gesture)는 카운트로 요약 — "들락거리네" 같은 헛소리 차단

### Vision/모션 안정화

- `WristWaveDetector`: orientation=front gate 추가 + amp/zc 임계 강화 (false positive 79/일 → 거의 0)
- `HeadOscillationDetector` (nod/shake): history 1.8→2.2s, cooldown 20→30s, zc 6→8
- `head_tracker`: bbox jump>0.40 reject (+ 연속 3회 시 강제 통과), SMOOTHING 0.10, MAX_STEP 2.5°, DEADZONE 6°, median bbox 스무딩
- `run_ambient_motion`: 사용자 있을 때 빈도 ↓ (40-120s), 진폭 1.5-2.5° (휙 돌림 완화)
- `poses.sway`: base를 **현재 servos.position**으로 (CENTER 강제 X) — head_tracker가 옆을 보고 있어도 그 위치 기준으로 흔들고 그 위치로 복귀. CENTER 기준이던 시절엔 사용자 따라가던 위치 ↔ CENTER 양방향 휙 발생
- `head_tracker`: motion 양보(motion_busy/TALKING/GREETING/LISTENING) 풀린 다음 cycle에 `pan_current/tilt_current/stable_target/last_sent`을 `servos.position`으로 resync — 다른 모션이 옮긴 위치와 stale 내부 상태 어긋남 방어
- `HeadOscillationDetector.reset()`: `_shoulder_axis_history`도 함께 clear (외부 reset 후 stale 어깨 데이터와 새 nose 섞이는 버그 fix)
- `mic._convert_native_to_target`: int() → round() — odd native rate에서 frame size off-by-one으로 VAD가 silently drop 회피

### 다음 후보

- 더 좋은 USB 마이크/헤드셋 (현재 peak 2-4k라 OpenAI API 의존 — 신호 강하면 local로 무료 전환)
- 한국어 wake word 학습 ("로보야") — openWakeWord 합성 데이터 학습
- face thread-safety helper / agent.py 모듈 분리

스펙/설치는 [docs/parts-list.md](docs/parts-list.md), [docs/setup-pi5.md](docs/setup-pi5.md), [docs/wiring-lcd.md](docs/wiring-lcd.md), [docs/setup-auto-update.md](docs/setup-auto-update.md) 참조.
