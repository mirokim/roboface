# Roboface 아키텍처 및 구현 계획

**작성일**: 2026-05-10 17:42 KST
**상태**: Draft v0.1
**관련 문서**: [parts-list.md](parts-list.md)

---

## 1. 개요

데스크탑에 두는 캐릭터 로봇. 단순한 명령-응답 봇이 아니라 **사용자를 곁에서 관찰하며 적절한 순간에 개입하는 동반자**가 목표.

### 핵심 컨셉: Passive Observer Companion

```
[사용자가 일하는 동안]
  └─ 로봇은 조용히 곁에 앉아
     ├─ 자세를 본다
     ├─ 작업 시간을 잰다
     ├─ 환경을 느낀다 (온/습도)
     ├─ 주변 대화를 듣는다
     └─ 적절한 순간에만 입을 연다
        ("벌써 2시간 앉아 계셨네요" / "내일 회의 잊지 마세요")
```

### 차별점

- ❌ "헤이 시리" 같은 wake word 기반 즉답형 X
- ✅ **수동적 관찰 → 능동적 개입** 전환형
- ✅ 장기 컨텍스트 누적 (일정, 작업 패턴, 환경 학습)
- ✅ 가벼운 존재감 — 말 거는 빈도 < 침묵 시간

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    SENSORS (입력)                       │
│  mmWave (존재/거리)  DHT22 (온/습도)                    │
│  AI Camera (시각)    USB Mic (소리)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  PERCEPTION (인지)                      │
│  - 사용자 존재/부재                                     │
│  - 자세 분석 (Pose Estimation)                          │
│  - 음성 인식 (Whisper STT)                              │
│  - 환경 상태                                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              BRAIN (판단/추론)                          │
│  - 상태 머신 (현재 행동 모드)                           │
│  - 트리거 평가기 (개입 조건 판단)                       │
│  - LLM 추론 (Claude API) — 대사 생성, 일정 추출         │
│  - 메모리 (단기 + 장기 + 일정 DB)                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                ACTUATORS (출력)                         │
│  LCD (얼굴 표정)  Servo×2 (머리 움직임)                 │
│  Speaker (음성)                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 소프트웨어 모듈 구조

```
roboface/
├── docs/                          # 문서
│   ├── parts-list.md
│   └── 2026-05-10_1742_architecture-plan.md
├── src/
│   ├── config.py                  # 핀맵, 상수, API 키
│   ├── main.py                    # 메인 루프
│   │
│   ├── face/                      # 얼굴 렌더링
│   │   ├── eyes.py                # 눈 모양/깜빡임
│   │   ├── mouth.py               # 입 모양/립싱크
│   │   ├── expressions.py         # 감정 표정 프리셋
│   │   └── renderer.py            # LCD/Pygame 추상화
│   │
│   ├── motion/                    # 머리 움직임
│   │   ├── servos.py              # PCA9685 + SG92R 제어
│   │   ├── poses.py               # 미리 정의된 동작 (인사, 끄덕임)
│   │   └── tracking.py            # 사용자 추적 (PID 등)
│   │
│   ├── sensors/                   # 센서 입력
│   │   ├── mmwave.py              # S3KM1110 UART 파싱
│   │   ├── environment.py         # DHT22
│   │   └── manager.py             # 센서 통합 관리
│   │
│   ├── vision/                    # AI Camera
│   │   ├── camera.py              # picamera2 wrapper
│   │   ├── face_detector.py       # 얼굴 위치 (트래킹용)
│   │   └── pose_estimator.py      # 자세 분석
│   │
│   ├── audio/                     # 음성
│   │   ├── stt.py                 # Whisper (로컬)
│   │   ├── tts.py                 # Piper / ElevenLabs
│   │   └── mic.py                 # 항시 녹음 + VAD
│   │
│   ├── brain/                     # 판단/추론
│   │   ├── state_machine.py       # 행동 모드 전이
│   │   ├── triggers.py            # 개입 조건 평가
│   │   ├── conversation.py        # Claude API 호출
│   │   └── memory.py              # 단기/장기 메모리
│   │
│   ├── tasks/                     # 능동 작업 (백그라운드)
│   │   ├── posture_monitor.py     # 자세 감시
│   │   ├── work_tracker.py        # 작업시간 추적
│   │   ├── ambient_listener.py    # 주변 대화 듣기
│   │   └── schedule_extractor.py  # 일정 추론
│   │
│   └── data/                      # 영속 데이터
│       ├── work_log.db            # 작업 시간 로그 (SQLite)
│       ├── schedule.db            # 추론된 일정
│       └── memory.json            # 사용자 패턴
│
├── tests/                         # 단위 테스트 + 시뮬레이터
│   └── face_simulator.py          # Pygame으로 얼굴 미리보기
└── README.md
```

---

## 4. 센서 활용 전략

### 4.1 mmWave (S3KM1110)

| 데이터 | 활용 |
|---|---|
| 존재 감지 (있음/없음) | 사용자 등장/이탈 → 인사/배웅 트리거 |
| 거리 (cm 단위) | 가까이 오면 적극 반응, 멀어지면 조용히 |
| 미세 동작 (호흡 등) | "정적 존재" 판단 → 책상에 가만히 있다고 인식 |
| 신규 감지 이벤트 | "방금 들어왔다" 정확 시점 |

**용도**: 작업시간 추적의 코어, 인사 트리거

### 4.2 DHT22 (온습도)

| 데이터 | 활용 |
|---|---|
| 온도 | 25°C↑ "더우시죠?", 18°C↓ "춥지 않으세요?" |
| 습도 | 30%↓ "건조해요, 물 드세요", 70%↑ "습해서 쾌적하지 않을 텐데" |
| 변화율 | 급격한 변화 감지 → 환기/에어컨 코멘트 |

**용도**: 환경 대화 소재 (자연스러운 멘트 제공)

### 4.3 AI Camera (IMX500 + RP2040)

| 데이터 | 활용 |
|---|---|
| 얼굴 위치/방향 | 사용자 얼굴 추적 → 머리 회전 |
| Pose Estimation | 어깨 기울기, 목 각도, 거북목 감지 |
| 시선 방향 (선택) | 모니터 응시 시간 추정 |
| 표정 인식 (선택) | 사용자 기분에 맞춰 반응 변경 |

**용도**: 자세 확인의 코어. 시각적 인지 전반

**참고**: 온칩 NPU에서 객체 감지 결과만 전송받아 Pi CPU 부담 최소화. 자세 분석은 MediaPipe Pose를 Pi에서 실행하거나 IMX500용 변환 모델 사용.

### 4.4 마이크 (오디오)

| 데이터 | 활용 |
|---|---|
| Voice Activity Detection (VAD) | 대화 중인지 감지 |
| Whisper STT (스트리밍) | 발화 텍스트화 |
| 환경 음향 | 키보드 타이핑 = 작업 중 신호 |

**용도**: 주변 대화 듣기 + 직접 명령 + 일정 추출

⚠️ 오디오 부품 미정 — Phase 3에서 결정/통합

---

## 5. 핵심 동작 (4가지 아이디어)

### 5.1 자세 확인

```
AI Camera → Pose Estimation → 거북목/굽은 어깨 분석
                                  ↓
                              N분 지속?
                                  ↓
                          소프트 알림 → (무시 시) 점진적 강화
```

**구현 단계**:
1. AI Camera 실시간 frame 수신
2. Pose 모델로 keypoint 추출 (코, 어깨, 귀)
3. 각도 계산 (목 기울기 = 귀-어깨 각도)
4. 30초 단위 평균, 5분 단위 분석
5. 임계치 초과 + 지속 → 알림

**알림 정책 (점진적)**:
- 1차 (10분): 표정만 살짝 (걱정 표정 + 머리 기울임)
- 2차 (20분): 짧은 음성 ("자세 좀...")
- 3차 (30분): 명확한 음성 + 큰 표정 변화

### 5.2 작업시간 추적

```
mmWave 존재 감지 → SQLite에 시작 시간 기록
                       ↓
              사용자 자리 비움 감지
                       ↓
              종료 시간 기록 → 누적
                       ↓
   1시간/2시간 도달 → 휴식 권유 멘트
```

**저장 스키마**:
```sql
CREATE TABLE work_sessions (
  id INTEGER PRIMARY KEY,
  start_time DATETIME,
  end_time DATETIME,
  duration_seconds INT,
  break_count INT,
  posture_warnings INT
);
```

**행동 트리거**:
- 1시간 누적: 부드러운 권유
- 2시간 무휴식: 강한 권유 + 표정 변화
- 4시간 무휴식: 걱정 표정 + 적극 만류
- 매일 패턴 학습 → 사용자 평소 패턴과 비교

### 5.3 가끔 말걸기

**개입 조건 (트리거)**:

| 트리거 | 조건 | 빈도 |
|---|---|---|
| 환경 변화 | 온/습도 급변 | 1일 0~3회 |
| 시간대 마일스톤 | 점심/저녁/퇴근 시간 | 1일 1~2회 |
| 작업 마일스톤 | 1시간/2시간 작업 도달 | 트리거됨 |
| 정적 존재 (오래 가만) | 30분+ 미동작 | 1시간에 1회 |
| 사용자 등장/복귀 | mmWave 신규 감지 | 부재 후 5분+ 지나면 |
| 자세 알림 | 위 5.1 참조 | 지속 시 |
| 무조건 침묵 시간 | 4시간+ 말 안 걸음 | 안전장치 |

**LLM 프롬프트 패턴**:
```
당신은 책상 위에 있는 작은 로봇 캐릭터입니다.
현재 상황:
- 시간: {time}
- 사용자 작업 시간: {work_duration}
- 마지막 휴식: {last_break}
- 온도: {temp}, 습도: {humidity}
- 최근 추출된 일정: {schedules}

지금 이 사용자에게 무슨 한마디 짧게 건네면 자연스러울지,
1~2문장으로만 답하세요. 너무 자주 말 거는 잔소리꾼은 X.
```

### 5.4 주변 이야기 듣고 일정 추론

```
마이크 → VAD로 대화 구간 감지
            ↓
        Whisper STT (로컬)
            ↓
       텍스트 → LLM (Claude)
            ↓
   {일정/약속/할일} JSON 추출
            ↓
       schedule.db 저장
            ↓
   다음 날 아침 / 시간 임박 시 알림
```

**LLM 추출 프롬프트**:
```
다음 발화에서 일정/약속/할일이 언급되면 JSON으로 추출:

발화: "{transcript}"

스키마:
{
  "events": [
    {"type": "meeting|deadline|reminder",
     "datetime": "YYYY-MM-DDTHH:MM",
     "description": "...",
     "confidence": 0.0~1.0}
  ]
}
없으면 빈 배열.
```

**주의사항**:
- ⚠️ **프라이버시**: 항시 녹음 → 사용자에게 명확히 알림 + ON/OFF 가능
- ⚠️ 발화 텍스트는 일정 추출 후 폐기 (저장 X)
- ⚠️ 친구/가족 대화는 무시 (사용자 본인 발화만 분석)

---

## 6. 데이터 흐름 (실시간)

```
[Sensors] ─────────┐
                   ▼
            ┌──────────────┐
            │ Sensor Bus    │ (이벤트 큐)
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ Trigger      │ (조건 평가, 1Hz)
            │ Evaluator    │
            └──────┬───────┘
                   ▼ (개입 결정)
            ┌──────────────┐
            │ State        │
            │ Machine      │
            └──┬─────────┬─┘
               ▼         ▼
         [Face]      [Voice]
        [Motion]
```

**주기**:
- 센서 폴링: 10Hz (mmWave, IMU)
- DHT22: 0.1Hz (10초마다)
- 카메라 프레임: 5~15Hz
- LLM 호출: 이벤트 기반 (트리거 시만)
- 트리거 평가: 1Hz

---

## 7. 상태 머신

```
                        ┌──────────┐
                        │   IDLE   │ ←──── 부재 5분+
                        └────┬─────┘
                             │ 사용자 감지
                             ▼
                        ┌──────────┐
                        │ GREETING │ (인사)
                        └────┬─────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │   WATCHING       │ ←─┐
                  │ (관찰 모드, 기본) │   │
                  └─┬──┬──┬──┬───────┘   │
                    │  │  │  │           │
       ┌────────────┘  │  │  └──────┐    │
       ▼               ▼  ▼         ▼    │
  ┌─────────┐ ┌────────────┐ ┌──────────┐│
  │ TALKING │ │ ALERTING   │ │ LISTENING ││
  │(대화중) │ │ (자세/휴식) │ │ (수동청취)││
  └────┬────┘ └─────┬──────┘ └─────┬────┘│
       │            │              │     │
       └────────────┴──────────────┴─────┘
                  반응 종료 → WATCHING
```

**모드별 행동**:

| 모드 | LCD | 머리 | 음성 |
|---|---|---|---|
| IDLE | 절전 (눈 감김) | 정면 | 무음 |
| WATCHING | 깜빡이는 눈 | 가끔 회전 | 무음 |
| GREETING | 환영 표정 | 사용자 추적 | "안녕하세요!" |
| TALKING | 입 모양 변화 | 사용자 응시 | TTS 출력 |
| ALERTING | 걱정 표정 | 사용자 응시 | 알림 멘트 |
| LISTENING | 집중 표정 | 음원 방향 | 무음 (분석 중) |

---

## 8. LLM 사용 전략

### 호출 위치
- 능동 멘트 생성 (5.3)
- 일정 추출 (5.4)
- 사용자 직접 대화 응답
- 주간/일간 작업 패턴 요약

### 비용/지연 최적화
- **로컬 LLM (Llama 3.2 3B)**: 일정 추출 (정형 출력)
- **Claude API**: 자연스러운 대사 생성 (질 우선)
- **Prompt caching**: 시스템 프롬프트 재사용
- **Whisper**: 로컬 small/medium 모델 (Pi 5 8GB 충분)

### 비용 추정 (월)
- Claude API (Haiku): 일 50회 × 2k 토큰 = ~$2/월
- 로컬: $0
- → 합리적 운영 가능

---

## 9. 구현 단계 (Phase Plan)

### Phase 1: 얼굴 (현재~D+3)
- [ ] 프로젝트 구조 셋업
- [ ] config.py 핀맵 정의
- [ ] Pygame 기반 얼굴 렌더러
- [ ] 눈 깜빡임 + 기본 표정 5종
- [ ] LCD로 이식 (부품 도착 후)

### Phase 2: 움직임 + 환경 인지 (D+4 ~ D+10)
- [ ] PCA9685 + 서보 제어
- [ ] 미리 정의된 동작 (인사, 끄덕, 두리번)
- [ ] mmWave UART 파싱
- [ ] DHT22 읽기
- [ ] WATCHING 상태 + 환경 멘트 (텍스트만)

### Phase 3: AI Camera (D+11 ~ D+17)
- [ ] picamera2 통합
- [ ] 얼굴 검출 → 머리 추적
- [ ] Pose Estimation 통합
- [ ] 자세 알림 첫 동작

### Phase 4: 음성 (D+18 ~ D+25)
- [ ] 오디오 부품 결정 + 통합
- [ ] Whisper STT
- [ ] TTS (Piper 또는 ElevenLabs)
- [ ] 입 모양 립싱크

### Phase 5: 두뇌 (D+26 ~ D+35)
- [ ] 상태 머신 완성
- [ ] 트리거 평가기
- [ ] Claude API 통합
- [ ] 작업시간 추적 (SQLite)

### Phase 6: 능동 행동 (D+36 ~ D+45)
- [ ] 가끔 말걸기 트리거
- [ ] 일정 추출 파이프라인
- [ ] 장기 메모리 + 패턴 학습

### Phase 7: 외관 (병행)
- [ ] 받침대 (베이스 플레이트)
- [ ] 3D 프린팅 외피
- [ ] 케이블 정리

---

## 10. 미해결 사항 (Open Questions)

| 항목 | 상태 |
|---|---|
| 오디오 부품 (USB vs I2S) | TBD — Phase 4 전까지 결정 |
| 받침대 재질/크기 | TBD — 부품 도착 실측 후 결정 |
| 진동 댐퍼 사용 여부 | TBD — 동작 후 진동 측정 |
| LLM 분배 (로컬 vs API) | 1차는 Haiku, 향후 Llama 추가 |
| 자세 모델 (MediaPipe vs IMX500 변환) | 일단 MediaPipe |
| TTS 품질 vs 비용 (Piper vs ElevenLabs) | Piper로 시작, 만족 안 되면 변경 |
| 프라이버시 / 녹음 인디케이터 | LCD에 항상 표시 (눈 모서리 점) |
| 외관 캐릭터 디자인 | 동작 검증 후 |

---

## 11. ThinkTank1 통합 연동

### 11.1 통합 동기

ThinkTank1 (`C:\Dev\thinktank1`)은 이미 운영 중인 **Life OS 플랫폼**:
- React + Vite 프론트엔드, Express + Supabase Postgres 백엔드
- Vault(노트), Journal, Calendar, People, Insights, Recall, Curation
- Anthropic Claude SDK, Web Push (VAPID), Capacitor Android 앱
- 26개 API 라우트 가동 중

Roboface 단독으로는 **장기 저장/시각화/모바일 접근**이 약하고, ThinkTank는 **물리적 입출력 채널**이 없음. 두 시스템이 합쳐지면 서로의 약점 보완.

### 11.2 통합 아키텍처

```
┌──────────────────────────────────────────────────┐
│  ThinkTank1 (Server: Express on port 3001)        │
│  ────────────────────────────────────────────    │
│  Routes (26):                                    │
│   /api/journal      /api/vault    /api/calendar  │
│   /api/people       /api/insights /api/recall    │
│   /api/curation     /api/push     /api/chat      │
│   /api/news         /api/weather  /api/health    │
│  ────────────────────────────────────────────    │
│  Storage: Supabase Postgres                      │
│  Auth: tt_auth 쿠키 (브라우저) + 서비스 토큰     │
└──────────────────┬───────────────────────────────┘
                   │ HTTPS (REST)
                   │ + WebSocket (실시간 옵션)
                   │
┌──────────────────┴───────────────────────────────┐
│  Roboface (Pi 5)                                  │
│  ────────────────────────────────────────────    │
│  src/integrations/thinktank/                     │
│   ├── client.py       # HTTP 클라이언트          │
│   ├── auth.py         # 서비스 토큰 관리         │
│   ├── journal.py      # 자동 일기 작성           │
│   ├── calendar.py     # 일정 추가/조회           │
│   ├── vault.py        # 검색/RAG                 │
│   ├── people.py       # 얼굴 ↔ 사람 매칭         │
│   └── push_listener.py # 푸시 수신/음성 변환     │
└───────────────────────────────────────────────────┘
```

### 11.3 인증 전략

ThinkTank의 기본 인증은 **`tt_auth` httpOnly 쿠키**(브라우저용). Roboface는 헤드리스 디바이스라 별도 방식 필요:

| 방식 | 평가 |
|---|---|
| **서비스 토큰** (`X-Robot-Token` 헤더) | ⭐ 추천. ThinkTank에 미들웨어 추가 |
| OAuth Client Credentials | 과한 복잡도 |
| Pre-auth된 쿠키 직접 사용 | 만료/갱신 번거로움 |

**제안 ThinkTank 측 변경**:
```javascript
// server/middleware/requireAuth.js 확장
export function requireAuth(req, res, next) {
  // 기존 쿠키 검사 ...
  // 추가: 서비스 토큰
  const robotToken = req.headers['x-robot-token'];
  if (robotToken && robotToken === process.env.ROBOT_SERVICE_TOKEN) {
    return next();
  }
  // ...
}
```

`.env`에 `ROBOT_SERVICE_TOKEN` 추가 후 양쪽에서 공유.

### 11.4 핵심 통합 시나리오

#### 🥇 Tier 1 (즉시 가치, Phase 5~6에 통합)

**A. 자동 저널링 (Auto-journal)**
```
Roboface 마이크 → STT → LLM 요약 → 의미 있는 발화 추출
                                          ↓
                          POST /api/journal (서비스 토큰)
                                          ↓
                              ThinkTank Journal에 저장
```
- 트리거: 사용자 발화 감정 강도 ↑, 약속/사건 언급
- 보관: ThinkTank가 알아서 (Strata 노트 자동 동기화 보유)

**B. 일정 추출 → Calendar 직접 입력**
```
"내일 3시 김 부장님과 미팅" → LLM JSON 추출
                              → POST /api/calendar/events
                              → ThinkTank가 푸시 예약
                              → 시간 임박 시 푸시 → Roboface가 음성 출력
```
- Roboface에 별도 일정 DB 안 둬도 됨 (ThinkTank가 SoT)

**C. Vault를 장기 기억으로 사용 (RAG)**
```
사용자: "지난주에 OOO 관련해서 뭐 적었지?"
   ↓
Roboface STT → 검색어 추출
   → POST /api/vault/search { q: "OOO" }
   → 관련 노트 받기 (top 3)
   → Claude API에 컨텍스트로 + 사용자 질문
   → 자연어 답변 생성 → TTS
```

**D. Push 알림 → 음성 출력**
```
ThinkTank 푸시 발송 (모바일과 동일)
   ↓
Roboface가 폴링 또는 WebSocket으로 수신
   ↓
중요도 분류 → 음성/표정으로 전달
```
- 새 엔드포인트: `GET /api/push/pending` (로봇용 큐)
- 또는 web-push 직접 구독 (VAPID 키 공유)

**E. 얼굴 인식 + People DB**
```
AI Camera 얼굴 감지
   ↓
얼굴 임베딩 → POST /api/people/match
   ↓
일치하는 person + 마지막 만남/메모 조회
   ↓
LLM에 컨텍스트 → "OO님 한 달 만이네요"
```

#### 🥈 Tier 2 (안정화 후)

**F. 작업 패턴 → Insights 누적**
- Roboface가 일/시간 단위 통계 → POST `/api/insights`
- ThinkTank 대시보드에서 시각화 (Recharts)
- Weekly/Monthly 자동 인사이트 (ThinkTank의 `runWeeklyInsight` 활용)

**G. Health Connect 통합**
- ThinkTank의 `capacitor-health-connect` 데이터(걸음/수면/심박)와
- Roboface 관찰(자세/작업시간) 결합
- 통합 인사이트: "수면 부족 + 자세 나쁨 = 오늘 컨디션 ↓"

**H. ThinkTank Android 앱 = Roboface 컴패니언**
- 별도 앱 안 만들고 ThinkTank 앱에 로보페이스 모듈만 추가
- 로봇 상태 / 원격 제어 / 통계 보기

#### 🥉 Tier 3 (재미 단계)

**I. Recall 모듈 활용**
- "1년 전 오늘?" → ThinkTank Recall API → 음성으로 회상

**J. Curation → 능동 멘트 소재**
- ThinkTank의 News/Curation 결과를 아침에 한 줄 정도

**K. 음성으로 Vault에 메모 추가**
- "이거 메모해줘: ..." → POST `/api/vault/notes`

### 11.5 ThinkTank1 측에 추가할 엔드포인트

기존 라우트로 90% 커버 가능. **로봇 전용 추가 권장**:

```
POST   /api/robot/observation
       { type: "posture"|"work_session"|"presence",
         data: {...}, timestamp }
       
GET    /api/robot/notifications/pending
       → 음성 변환 대기 중인 푸시 큐

WS     /api/robot/realtime
       → 양방향 실시간 (선택, Phase 7+)

POST   /api/robot/face-embedding
       { embedding: [512 float], confidence }
       → People 매칭 결과 반환

GET    /api/robot/context-bundle
       → 현재 시각 기준 한 묶음:
         · 다음 일정 1~3개
         · 미확인 알림
         · 오늘 작업 통계
         · 최근 대화 요약
       (Roboface가 멘트 생성 시 단일 호출로 컨텍스트 확보)
```

### 11.6 데이터 흐름 (통합 후)

```
[사용자 발화 / 행동]
         │
         ▼
[Roboface 센서/마이크/카메라]
         │
         ├─ 단순 응답 → 자체 처리 (LLM 로컬)
         │
         ├─ 저장 가치 있음
         │   ↓
         │ ThinkTank API
         │   ├─ Journal
         │   ├─ Calendar
         │   ├─ Vault (메모)
         │   └─ Insights (통계)
         │
         └─ 컨텍스트 필요
             ↓
            ThinkTank API
             ├─ Vault 검색
             ├─ People 조회
             ├─ Calendar 조회
             └─ Recall (과거 회상)
```

### 11.7 통신 프로토콜 정리

| 항목 | 방식 |
|---|---|
| 기본 통신 | HTTPS REST |
| 인증 | `X-Robot-Token` 헤더 |
| 데이터 형식 | JSON |
| 실시간 알림 (선택) | WebSocket 또는 SSE |
| 오디오/이미지 업로드 | multipart/form-data 또는 base64 |
| 폴링 주기 | 알림 큐 5초, 컨텍스트 번들 30초 |
| 오프라인 처리 | Roboface 로컬 큐 → 복구 시 재전송 |

### 11.8 환경 변수 추가

**Roboface `src/config.py`**:
```python
THINKTANK_BASE_URL = "http://localhost:3001"  # 또는 배포 URL
THINKTANK_ROBOT_TOKEN = os.getenv("THINKTANK_ROBOT_TOKEN")
THINKTANK_TIMEOUT = 5  # seconds
THINKTANK_RETRY = 3
```

**ThinkTank `.env`**:
```
ROBOT_SERVICE_TOKEN=<32바이트 랜덤 hex>
ROBOT_ALLOWED_IPS=192.168.0.0/24  # (선택, 추가 보안)
```

### 11.9 단계별 통합 로드맵

| Phase | Roboface 측 | ThinkTank 측 |
|---|---|---|
| **5.1** | `integrations/thinktank/client.py` 기본 HTTP | `requireAuth` 서비스 토큰 지원 |
| **5.2** | Journal 자동 저장 | `/api/journal` 그대로 사용 |
| **5.3** | Calendar 추가 | `/api/calendar` 그대로 사용 |
| **6.1** | Push 수신 (폴링) | `/api/robot/notifications/pending` 추가 |
| **6.2** | Vault 검색 (RAG) | `/api/vault/search` 그대로 사용 |
| **6.3** | People 매칭 | `/api/robot/face-embedding` 추가 |
| **7.1** | Insights 누적 | `/api/robot/observation` 추가 |
| **7.2** | Health 통합 | 기존 health 라우트 활용 |
| **8.0** | WebSocket 실시간 (선택) | `/api/robot/realtime` 추가 |

### 11.10 위험 요소

| 리스크 | 완화 |
|---|---|
| ThinkTank 서버 다운 | 로컬 큐에 쌓아두고 복구 시 동기화 |
| 네트워크 지연으로 음성 응답 늦음 | 즉답 가능한 간단 케이스는 로컬 LLM, 복잡한 건 ThinkTank/Claude |
| 프라이버시 (저널링 자동화) | "저장 모드" 명시적 ON/OFF, LCD 인디케이터 |
| 토큰 유출 | `.env` 격리, 로컬 네트워크에서만 접근 권장, IP allow-list |
| 양쪽 동시 수정 충돌 | ThinkTank가 SoT — Roboface는 캐시만 |

### 11.11 첫 통합 작업 (Phase 5 시작 시)

1. ThinkTank `.env`에 `ROBOT_SERVICE_TOKEN` 추가 + `requireAuth` 확장 (1시간)
2. Roboface `integrations/thinktank/client.py` 작성 + 헬스체크 (1시간)
3. Journal 자동 저장 PoC: "테스트 메모" → `/api/journal` POST (30분)
4. ThinkTank UI에서 저장된 메모 확인 (10분)

이 4단계만 검증되면 나머지는 패턴 반복.

---

## 12. 다음 즉시 작업

부품 도착 전 (~D+3) 동안:

1. **프로젝트 폴더 구조 생성** ← 다음 단계
2. **config.py 작성** (핀맵 + 상수)
3. **Pygame 얼굴 시뮬레이터** 시작
4. **눈 깜빡임 애니메이션** 구현
5. (가능하면) Claude API 키 발급, 간단 호출 테스트

도착 후 즉시:
1. Pi OS 설치 + SSH 설정
2. SPI/I2C/UART 활성화
3. LCD에 첫 그림 띄우기
4. 서보 1개 회전 테스트
