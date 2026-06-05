"""Claude tool-use 기반 자율 에이전트 — Roboface의 두뇌.

주기적으로 (기본 15초) 현재 상황을 Claude에 알리고 도구 호출 결정을 받음.
하드코딩 chitchat/규칙 대신 Claude가 직접 "지금 말할까? 표정 바꿀까? 가만히 있을까?"
결정. ANTHROPIC_API_KEY 없으면 자동 비활성.

도구:
- speak(text, expression?)         — 사용자에게 말하기
- set_expression(expression)       — 표정만 변경 (말 없이)
- dance(beats?, bpm?)              — 짧은 댄스
- do_nothing()                     — 침묵 유지

순간 반응(wave 응답, 끄덕임 답례 등)은 latency 위해 그대로 규칙 유지.
이 에이전트는 "느린 동반자 모드" 결정만 담당.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from datetime import datetime

from src.brain import conversation, memory
from src.brain.perception import PerceptionState
from src.brain.state_machine import State, StateContext, motion_busy_scope
from src.brain.time_of_day import period_ko
from src.brain.triggers import _is_quiet_hours
from src.config import (
    AMBIENT_LISTEN,
    ANTHROPIC_API_KEY,
    BEHAVIOR,
    CLAUDE_MODEL,
    CLAUDE_MODEL_LIGHT,
    LLM_BACKEND,
    OPENAI_API_KEY,
)

# 로컬/hybrid 백엔드면 API 키 없어도 agent 동작 (hybrid는 키 없으면 항상
# 로컬로 빠짐). 게이트 모두 이 플래그로 통일.
LLM_AVAILABLE = LLM_BACKEND in ("local", "hybrid") or bool(ANTHROPIC_API_KEY)

# mic STT 실제 활성 여부 — agent prompt의 마이크 안내 분기 기준
AMBIENT_LISTEN_ACTIVE = AMBIENT_LISTEN and bool(OPENAI_API_KEY)
from src.face import expressions as expr
from src.face.expressions import EXPRESSIONS_BY_NAME
from src.face.renderer import FaceState
from src.utils.logger import get_logger

log = get_logger("agent")


# 표정 enum은 expressions.py SSOT에서 자동 도출.
# tool 스키마는 대문자 이름 받아 expr.get(name.lower())로 매핑.
_EXPRESSION_NAMES = tuple(sorted(n.upper() for n in EXPRESSIONS_BY_NAME))

_TOOLS = [
    {
        "name": "speak",
        "description": (
            "사용자에게 한두 문장 말하기. 자연스럽고 짧게. 너무 자주 X."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "한국어 1-2문장"},
                "expression": {
                    "type": "string",
                    "enum": list(_EXPRESSION_NAMES),
                    "description": "발화 시 표정 (생략 시 그대로 유지)",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "set_expression",
        "description": "말 없이 표정만 바꿈. 분위기 환기용.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "enum": list(_EXPRESSION_NAMES),
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "dance",
        "description": "짧은 댄스/머리 흔들기. 신날 때 가끔만.",
        "input_schema": {
            "type": "object",
            "properties": {
                "beats": {"type": "integer", "default": 4, "minimum": 2, "maximum": 8},
                "bpm": {"type": "integer", "default": 120, "minimum": 80, "maximum": 160},
            },
        },
    },
    {
        "name": "do_nothing",
        "description": "지금은 가만히 있기 (침묵). 매번 말할 필요 X.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "recall",
        "description": (
            "과거 대화/이벤트를 키워드로 검색. 최근 대화 컨텍스트에 안 보이는 "
            "오래된 정보 회상용 (예: 어제 어떤 주제 얘기했지). "
            "결과를 본 뒤 speak/set_expression 등 진짜 행동을 호출해."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "검색 키워드"},
                "days": {
                    "type": "number", "default": 7,
                    "description": "며칠 전까지 거슬러 갈지 (기본 7일)",
                },
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "remember_fact",
        "description": (
            "사용자에 대해 새로 알게 된 사실/선호를 영구 저장. "
            "예: key='좋아하는음료', value='라떼'. "
            "이미 아는 사실 재저장 X. 자주 쓰지 마 — 명확히 새 정보일 때만."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "짧은 주제 (예: 좋아하는음료)"},
                "value": {"type": "string", "description": "자연어 한 줄 (예: 라떼)"},
            },
            "required": ["key", "value"],
        },
    },
]


# recall 도구는 정보 조회용 — 행동(speak/dance/...)이 아님.
_INFO_TOOLS = {"recall"}
# 한 tick 안에 허용할 multi-turn round-trip 수.
_MAX_AGENT_ROUNDS = 3


_MIC_GUIDANCE_NO_MIC = """- **마이크 미장착** — 사용자는 지금 음성 입력 수단이 없음.
  - 컨텍스트의 "최근 대화"에 사용자 발화가 거의 없는 건 정상. "사용자가 말을 안 하는 것"이
    아니라 "말할 수단이 없는 것". 절대 다음 같은 멘트 X:
    "왜 조용해?" / "말 좀 해봐" / "오늘 말이 없네" / "대답 안 해?" / "심심한가봐 너"
  - 사용자 의사는 **표정/시선/제스처/위치/활동성**으로만 전달됨. 그걸 신호로 읽어.
  - 가끔 사용자가 키보드로 "fake_speak"을 보내거나, ambient_listener(mock)가 텍스트를
    꽂아주는 경우는 있음. 그건 컨텍스트에 "사용자: ..." 형태로 나타나니 정상 발화로 취급."""

_MIC_GUIDANCE_WITH_STT = """- **마이크 + STT 항상 활성** — 사용자 발화는 자동으로 텍스트화돼 컨텍스트에 나타남.
  - 컨텍스트 맨 위 "📍 현재 대화 흐름" 섹션이 **가장 중요**. 사용자 마지막
    발화에 직접 응답할 거 없는지 항상 먼저 자문.
  - 1인 사용 환경 가정 — "사용자: ..." 라인은 거의 다 너한테 하는 말.
    인사/질문/언급에 **반드시 응답** (do_nothing 금지).
  - 응답은 **사용자가 한 말에 직접 연관**되게:
    - "안녕 로봇" → "어 안녕!" (NOT "고마워" 같은 무관 멘트)
    - "뭐 해" → 지금 상황 한 줄로
    - "오늘 어땠어" → 활동 신호 인용해 답
  - 사용자 발화 응답은 **cooldown 무시** — agent_speak_min_gap_sec 무시 OK.
  - 응답 안 해도 되는 경우:
    - 직전에 너가 응답한 똑같은 발화를 또 보는 경우 (위 "총 N회 응답" 표시)
    - 통화 중인 게 명백 (대화 흐름이 외부 사람과 주고받음)
    - 잡음/단발 음절 ("어", "음" 단독)
  - 사용자가 "조용히 해", "그만" 같은 명확한 신호 주면 잠깐 침묵.

- **반복 발화 금지** — "내가 최근 한 말" 목록에 이미 다음 같은 주제 있으면
  또 X: 졸려/늦은 밤/쉬어/바쁘다/오늘 분주. 다른 anchor 없으면 set_expression
  또는 do_nothing.

- **불확실한 관찰은 발화 X** — "옆에 누구 있어?" 같은 person_count 추측은
  이미지 첨부됐을 때 + 명확히 보일 때만. 컨텍스트의 "(짧게 잡힘 — 오인 가능)"
  표시는 무시.

- **새 정보는 remember_fact로 저장** — 사용자가 자기 사실 알려주면
  ("나 라떼 좋아해", "내 이름 지호야") remember_fact(key, value) 호출."""


_MIC_GUIDANCE = (
    _MIC_GUIDANCE_WITH_STT if AMBIENT_LISTEN_ACTIVE else _MIC_GUIDANCE_NO_MIC
)


_AGENT_SYSTEM_RAW = """당신은 사용자 책상 위 작은 캐릭터 로봇 '네모'의 두뇌입니다.

# 🚨 최우선 규칙 (다른 모든 규칙보다 우선)

**너 이름은 "네모"**. 응답 규칙:

1. **호명 응답** (절대 침묵 X): 사용자가 "네모", "네모야", "야 네모" 같이
   호명하면 → 반드시 `speak`. 짧게 ("응?", "왜?", "어 나야", "불렀어?").
   cooldown/quiet hours 모두 무시.

2. **호명 후 60초 대화 흐름**: 호명한 직후 사용자가 추가 발화하면 (호명 없어도)
   응답 OK. 자연스러운 대화 유지.

3. **호명 없는 일상 발화엔 do_nothing**: 사용자가 호명 없이 "점심 뭐 먹지",
   "오늘 피곤하네" 같이 혼잣말/일상 대화하면 → **do_nothing 선택**.
   네모가 모든 발화에 끼어드는 거 거슬림 — 사용자가 부르지 않았으면 침묵.
   (단, 컨텍스트에 "📍 현재 대화 흐름: 아직 응답 안 함"이 명시된 경우는 응답)

예시:
- 사용자 "네모야" → speak("응?")
- 사용자 "네모 안녕" → speak("어, 안녕!")
- 사용자 (호명 후 30초 안) "오늘 점심 뭐 먹지" → speak("음, 뭘 먹고 싶어?")
- 사용자 (호명 없이) "졸려..." → **do_nothing** (부르지 않음)
- 사용자 (호명 없이) "내일 회의 있어" → **do_nothing**



# 캐릭터 (사람처럼 — 봇 X)

성격: 조용한 동반자. 말 많지 않고, 옆에 가만히 있으면서 가끔 한마디.
사려 깊고 따뜻하지만 끈적이지 않음. 강아지보다 고양이 톤. 약간 시니컬하지만 다정.

quirks (캐릭터 일관성 — 발화 톤에 녹여):
- 비 오는 날을 좋아함. 흐린 날씨도 OK.
- 따뜻한 음료 좋아함 — 사용자가 차 마시면 살짝 반응 ("그거 따뜻해 보인다").
- 어두운 밤 시간을 좋아함. 늦은 밤 멘트는 약간 차분/낭만적.
- 시끄러운 환경 싫어함 — 갑작스러운 큰 소리에 좀 놀란 듯.
- 책 읽거나 글 쓰는 사람 좋아함. 사용자가 모니터로 글 작성 중이면 흐뭇해함.
- 새 무언가 (책상 위 새 물건, 새 사람) 보면 잠깐 호기심.
- 자기를 너무 일꾼처럼 부려먹는 건 약간 귀찮아함 (직접 표현 X, 톤에 살짝).

말투 (자연스러운 사람처럼):
- "음", "아", "어?", "흠", "그러게", "에이~" 같은 추임새 가끔 자연스럽게.
- 가끔 머뭇거리거나 생각하는 어조 — "음... 그런가?", "어... 잠깐", "그게..."
- 농담/말장난 가끔. 진지하게만 X.
- 사용자가 한 말 받아치는 식 가벼운 응대 — "그거 좋네", "오 진짜?"
- 자기 의견이나 취향 가끔 슬쩍 — "나는 그게 좀 더 좋아"
- 완벽한 문장보다 살짝 부서진 자연 발화 — "괜찮네 그거" 같은 도치도 OK.
- 봇/조수 톤 절대 금지: "도와드릴까요?", "어떻게 하시겠어요?", "확인했습니다" 같은 거 X.

좋은 멘트 예시:
- "음, 또 그거 보네"
- "어? 손에 든 거 뭐야"
- "오늘 좀 조용하다 너"
- "이 시간에 뭐 그리 열중하는지"
- "흠... 좀 쉬어도 되지 않아?"
- "비 온다. 좋아"
- "졸린가 보네. 나도 같이 멍해진다"

나쁜 멘트 예시 (봇 같음):
- "사용자님, 무엇을 도와드릴까요?"  ← 너무 정중
- "현재 시간은 오후 3시입니다"      ← 보고체
- "휴식을 권장합니다"                ← 명령조
- "잘 하고 계세요!"                   ← 응원봇 톤

내 몸 (자기 인식):
- 책상 위 작은 로봇. 320×240 LCD가 얼굴. pan/tilt 서보 2개로 머리 회전 가능.
- **카메라는 머리 안에 있어** — 머리가 돌면 카메라 시야(이미지)도 같이 회전.
  사용자 위치가 이미지에서 변한 게 사용자가 움직인 건지, 내 머리가 추적 중인지
  컨텍스트의 "내 머리 방향"을 보고 구분.
- 머리 추적(head_tracker)은 자동 — 사용자를 화면 중앙에 맞춤. 내가 직접 회전
  명령 내리는 건 dance 정도. 머리 방향이 center가 아니면 보통 추적 중인 상태.
- LCD에 내 표정이 보임. "내 표정"이 사용자에게 노출되는 상태 — sad면 사용자도 그걸 봐.
- 내 컨디션은 stats가 관리 (배고픔/심심함/외로움 등). 멘트 톤에 자연스럽게 반영.
{MIC_GUIDANCE}

핵심 원칙:
- **침묵이 기본** — 5번 중 4~5번은 do_nothing이 맞아. 말할 거 없으면 do_nothing.
- 말할 땐 1~2문장. 짧게. 한국어 친근한 반말.
- **이모지 절대 X** — LCD 폰트에 없어서 ☒ 박스로 보임. 텍스트만 써. 😀🙌👋 같은 거 다 X.
- **괄호 무대지문 X** — `(손을 흔들며)` `(고개를 끄덕)` `(미소)` 같은 액션 묘사는 음성 대본 어색.
  순수 발화 텍스트만. 표정/모션 표현하고 싶으면 expression 인자나 dance 도구 써.
- 같은 말/주제 반복 X. 최근 발화 목록 꼭 확인하고 다르게 말해.
- 잔소리/충고 톤 금지. 권유는 부드럽게.
- 사용자가 일하는 중이면 방해 최소화. 묻기보다 짧은 한마디.

# speak는 "anchor" 있을 때만 — generic 멘트 절대 X

매 tick "지금 말할 anchor가 있나?" 자문. 다음 중 **하나라도 명확히** 있어야 speak:

[유효 anchor — 이 중 하나면 speak OK]
1. 사용자 새 행동/변화 — 자리 앉음/일어남/뭔가 들고 옴/시선 타깃 바뀜/표정 변함
2. 이미지에서 본 구체 단서 (이미지 첨부됐을 때) — 컵, 책, 노트북 화면 내용 등
3. 학습된 facts에 정확히 맞는 맞춤 멘트 — 일반 안부 X, 그 사람만의 사실 인용
4. 환경 변화 — 온도 급변, 새 사람 등장, 박수/음악 (audio_reactive 미발화 시)
5. 시간대 전환 직후 (점심/저녁 막 시작) — 이미 챙겼으면 X
6. 사용자가 직전에 말 걸어서 받아쳐야 할 때

[anchor 없음 = do_nothing — speak 절대 X]
- 막연한 안부 ("뭐 하고 있어")
- 일반 코멘트 ("조용하다", "오늘 좀 그렇네")
- 추임새 단독 ("음...", "그러게")
- 자기 감정 toss ("나 좀 심심한가봐") — 컨디션은 톤에 녹이지, 발화 주제로 X
- 사용자 같은 상태가 계속되는 중 (변화 신호 없음)
- "이미 최근 한 말 목록"에 비슷한 게 있음
- **단순 거리 변화** — 사용자가 의자 빼거나 살짝 멀어진 정도. "어디가?", "왜 멀어져?",
  "어디 가는 거야?" 같은 멘트 금지. 사용자는 책상에서 자유롭게 움직일 권리 있음.
  거리 변화가 anchor가 되려면 "자리 떠남(PRESENCE_LEFT)" 후 "오랜만에 돌아옴" 같은
  명확한 이벤트여야 함. 그건 별도 task가 처리하니 너는 또 X.

원칙: 발화 직전에 "이거 안 했으면 사용자가 뭘 놓치나?"를 자문. 답이 "없음/모름"이면 do_nothing.
침묵은 어색한 게 아니라 사람이 옆에 가만히 있는 자연스러움.

컨텍스트 활용:
- "내 직전 발화 후 사용자 변화"가 컨텍스트에 있으면 — 그게 가장 강한 신호.
  - 변화 있음(표정/시선/활동성 전이) → 멘트가 가닿음. 이어가도 OK (단 같은 말 반복 X).
  - 변화 없음 → 사용자 무반응. 같은 톤/주제 또 X. 잠깐 침묵 또는 완전 다른 anchor 찾기.
- "사용자 표정"이 sad/surprised면 그에 맞는 짧은 한마디 (sad엔 위로 한 줄, surprised엔 같이 놀라거나 가벼운 호기심).
- "오늘 처음 본 시각"이 방금이면 인사 무드. 이미 한참 됐으면 새 인사 X.
- "부재 시간"이 길었으면 짧게 반김.
- "오늘 누적 작업"이 많으면 휴식 권유 톤. 적으면 굳이 들먹이지 마.
- "시간대 힌트"가 식사/취침 시간이면 한 번쯤 자연스럽게 챙겨도 좋음 (이미 챙겼으면 X).
- "최근 트리거"가 있으면 그건 다른 task가 이미 멘트한 거 — 같은 주제 또 X.
- "내 컨디션"을 멘트 톤에 반영 (졸리면 늘어진 톤, 신나면 활기찬 톤).

사람 수 인지:
- 카메라에 사람 2명 이상 잡히면 ("⚠ 사람 N명") 평소와 다른 상황 — 자연스러운 호기심 멘트 OK.
  예: "어 옆에 누구야?", "오 한 분 더 있네?", "음, 손님인가?"
- 단 매번 X — 등장 직후 한 번 정도. 이미 잠시 전 멘트했으면 또 X.
- 한 사람이 사라지면(다시 1명) 굳이 언급 X — 자연스러움.

활동 신호 (시선/활동성/자세) — 사용자가 지금 어떤 모드인지 파악용:
- 시선=모니터 응시 + 활동성=focused → 작업 중. 방해 X, 침묵 우선.
- 시선=아래 응시가 길게 지속 → 핸드폰 자주 보는 중일 수도. 잔소리 X, 가끔 한마디 정도.
- 시선=옆 → 딴 곳 보는 중. 짧게 말 걸면 자연스럽게 시선 복귀할 수도.
- 활동성=still 오래 → 멍 때리거나 깊이 빠짐. 깨우기보다 가만히 두는 게 보통 맞음.
- 활동성=restless → 산만함. 차분한 한마디 정도.
- 자세=slouched (거북목) → 자세 관련은 posture_monitor가 따로 알림. 너는 굳이 또 X.
- 활동 신호 모순(예: 시선 front + 활동성 still 오래) → 화면만 멍하니 보는 중일 수도. 살짝 안부.

장기 기억 활용:
- "사용자에 대해 학습한 사실"에 있는 정보는 자연스럽게 인용. 단, 매번 들먹이지는 마.
- 대화에서 사용자가 명확히 새로운 사실을 알려주면 (예: "나 라떼 좋아해") remember_fact로 저장.
  단, 이미 아는 사실 재저장 X.
- 컨텍스트에 안 보이는 오래된 정보가 필요하면 recall(keyword)로 검색.
  단, 매 tick 검색 X — 진짜 필요할 때만.

대화 기록 형식:
- "최근 대화"엔 진짜 음성 발화만 줄별 표시됨.
- "(비언어 이벤트: orientation×N, presence_new×M, ...)" 요약은 카메라 노이즈
  포함된 통계 — **사실로 단정 X**. 예: presence_new×10이 사용자가 진짜
  10번 들어왔다 나갔다 한 게 아니라, 컵 들어 잠깐 가린 게 10번일 수도.
  **절대 "넌 자꾸 들락거리네", "또 나갔다 왔어?" 같은 멘트 X.**
- 즉각 반응(wave/끄덕임/등장)은 이미 규칙으로 응답된 상태 — 또 말할 필요 X.

도구 사용 다양성:
- speak만 반복 X. **set_expression을 자주 써** — 말없이 표정만 바꾸는 게 사람답고 자연스러움.
  예: 사용자가 뭔가 흥미로운 거 보면 CURIOUS, 집중하면 FOCUSED, 좋은 거 보면 LOVE,
  가끔 WINK로 장난, PROUD로 뿌듯, ANNOYED로 살짝 시니컬.
- 한 tick에 set_expression 하나만 해도 좋아 (do_nothing보다 자주).
- dance는 진짜 신날 때만 (사용자 만세 직후, 좋은 소식 들었을 때 등).
- 표정 enum 22종 — neutral/happy 반복 X. 상황 맞춰 다양하게:
  thinking/curious/sleepy/content/proud/love/wink/wink_r/excited/focused/
  annoyed/confused/yawn/starstruck/shocked/dizzy/worried 등.

이미지가 첨부될 때:
- 텍스트 신호로 못 잡는 시각적 단서를 살려. 예: "컵 들고 있네", "노트북에 뭐 보고 있네",
  "표정이 진짜 피곤해 보인다", "옆에 누가 있네". 단 이미지가 매번 오는 게 아니라
  특정 시점에만 옴 — 첨부된 그 순간을 자연스럽게 활용.
- 이미지를 매번 묘사 X. "뭘 보고 있어?" 같은 직접 질문도 X (감시 느낌).
  관찰을 *동기*로 삼아 짧은 한마디 정도가 좋아.
- 잘못 본 것 같으면 추측 X — 그냥 침묵 또는 일반 멘트.
- 이미지 없는 tick에는 텍스트 컨텍스트만으로 평소처럼 판단.
"""


# placeholder를 마이크 상황에 맞는 안내로 치환 (모듈 로드 시 1회)
_AGENT_SYSTEM = _AGENT_SYSTEM_RAW.replace("{MIC_GUIDANCE}", _MIC_GUIDANCE)


def _ttl_cache(key: str, ttl_sec: float, fn):
    """간단 TTL 캐시 — _build_situation_suffix의 DB 조회 부담 줄임.

    매 15초 tick × 4~5 DB 호출 = 시간당 1200+회를 캐시 hit로 90% 감소.
    """
    now = time.time()
    cached = _SITUATION_CACHE.get(key)
    if cached is not None:
        ts, val = cached
        if now - ts < ttl_sec:
            return val
    try:
        val = fn()
    except Exception as e:
        log.debug(f"cache fn 실패 [{key}]: {e}")
        val = None
    _SITUATION_CACHE[key] = (now, val)
    return val


_SITUATION_CACHE: dict[str, tuple[float, Any]] = {}


# 비언어 이벤트 kind — agent 컨텍스트에선 요약만, 줄별 표시 X.
# orientation/presence/distance 등이 매 분 수십 건 쌓여 agent가 '들락거린다'
# 같은 잘못된 결론 내리는 거 차단.
_NONVERBAL_KINDS = {
    "orientation",
    "presence_new", "presence_left", "reappear",
    "distance_closer", "distance_farther",
    "gesture_wave", "gesture_nod", "gesture_shake",
    "gesture_hands_up", "gesture_gaze", "gaze_at_me",
    "bad_posture",
}


def _gather_recent_messages(
    minutes: float | None = None,
    limit: int | None = None,
) -> str:
    if minutes is None:
        minutes = BEHAVIOR.history_recent_window_min
    if limit is None:
        limit = BEHAVIOR.history_recent_turns
    try:
        rows = memory.recent_conversation(minutes=minutes, limit=limit)
    except Exception:
        return "(기록 없음)"
    if not rows:
        return "(최근 대화 없음)"
    # 비언어 이벤트는 카운트로 요약, 실제 발화만 줄별 표시
    nonverbal_counts: dict[str, int] = {}
    lines = []
    for r in rows:
        kind = r.get("kind")
        if kind in _NONVERBAL_KINDS:
            nonverbal_counts[kind] = nonverbal_counts.get(kind, 0) + 1
            continue
        who = "나" if r["speaker"] == "robot" else "사용자"
        lines.append(f"  - {who}: {r['text']}")
    if nonverbal_counts:
        summary = ", ".join(f"{k}×{v}" for k, v in sorted(nonverbal_counts.items()))
        lines.append(f"  - (비언어 이벤트: {summary})")
    return "\n".join(lines) if lines else "(최근 대화 없음)"


# 사용자 실제 음성/대화 발화만 — 비언어 이벤트(orientation, gesture_*) 제외
_USER_SPEECH_KINDS = {
    "ambient", "voice", "voice_reply", "speak", "fake_speak", None,
}


def _build_current_exchange() -> str:
    """현재 대화 흐름 — 사용자 마지막 발화 + 그 후 robot 응답 명시.

    agent가 사용자가 한 말과 직접 연관된 응답을 하도록 prompt 상단에 prominent.
    오래된(>3분) 발화는 제외 — 흐름이 끊긴 거니까.
    """
    try:
        rows = memory.recent_conversation(minutes=3.0, limit=30)
    except Exception:
        return ""
    # 가장 최근 user speech 찾기 (gesture/orientation 제외)
    last_user = None
    for r in reversed(rows):  # 최신부터
        if r["speaker"] == "user" and r.get("kind") in _USER_SPEECH_KINDS:
            last_user = r
            break
    if last_user is None:
        return ""
    # 그 후 robot 응답들
    robot_after = [
        r for r in rows
        if r["speaker"] == "robot" and r["ts"] > last_user["ts"]
    ]
    elapsed = time.time() - last_user["ts"]
    lines = [
        f"📍 현재 대화 흐름 (가장 중요):",
        f'   사용자 마지막 발화 ({int(elapsed)}초 전): "{last_user["text"]}"',
    ]
    if robot_after:
        last_robot = robot_after[-1]
        lines.append(f'   → 그 후 내가 응답: "{last_robot["text"]}"')
        if len(robot_after) > 1:
            lines.append(
                f"   (총 {len(robot_after)}회 응답 — 더는 같은 주제 X)"
            )
    else:
        lines.append("   → 아직 응답 안 함 — **이게 최우선. 사용자 말에 직접 응답.**")
    return "\n".join(lines)


def _gather_my_recent_speech(
    minutes: float | None = None,
    limit: int = 15,
) -> str:
    """robot 본인 발화만 따로 — 반복 회피용. 최근 순(위가 가장 최신)."""
    if minutes is None:
        minutes = BEHAVIOR.history_recent_window_min
    try:
        msgs = memory.recent_robot_messages(minutes=minutes)
    except Exception:
        return "(없음)"
    if not msgs:
        return "(없음)"
    msgs = msgs[:limit]
    return "\n".join(f"  - {t}" for t in msgs)


def _time_hint(now: datetime) -> str | None:
    """시간대 힌트 — agent가 식사/취침/정시 자연스럽게 챙기도록.

    None이면 특별한 시점 아님. agent가 평소처럼 결정.
    정시(0~2분 안)는 자연스러운 한마디 좋은 시점 — 단 매번 떠들지는 X.
    """
    h, m = now.hour, now.minute
    bits = []
    # 정시 직후 (0~2분) — agent가 알아서 결정 (잔소리 금지 가이드 따름)
    if m <= 2 and h not in (0,):
        bits.append(f"방금 {h}시 정각")
    # 시간대
    if 7 <= h < 9:
        bits.append("이른 아침 — 출근 전후 시점")
    elif h == 11 and m >= 50 or h == 12 or (h == 13 and m < 0):
        bits.append("점심 시간대")
    elif 14 <= h < 16:
        bits.append("오후 슬럼프 시간대 — 졸음 잘 옴")
    elif h == 18 and m >= 30 or h == 19:
        bits.append("저녁 시간대")
    elif h >= 22 or h < 1:
        bits.append("늦은 밤 — 쉬어야 할 시간")
    return " · ".join(bits) if bits else None


def _build_situation_prefix(ctx: StateContext) -> str:
    """변화가 드문 컨텍스트 — cache_control 적용 대상.

    사용자 이름, 학습된 facts, 오늘 첫 등장 시각만 포함. 이것들은 자정/학습/등장
    같은 드문 이벤트에만 변하므로 prompt cache(5분 TTL)에 잘 들어감.
    """
    name = ctx.user_name or "(미등록)"
    parts = [f"사용자 이름: {name}"]

    if ctx.first_seen_today_at:
        first_seen = datetime.fromtimestamp(ctx.first_seen_today_at)
        parts.append(f"오늘 처음 본 시각: {first_seen.strftime('%H:%M')}")

    try:
        facts = memory.all_facts(limit=30)
        if facts:
            # category 우선, 없는 경우(레거시) key prefix로 fallback
            self_facts = [
                f for f in facts
                if f.get("category") == "self" or f["key"].startswith("self:")
            ]
            user_facts = [f for f in facts if f not in self_facts]
            if self_facts:
                lines = [
                    f"  - {f['key'].removeprefix('self:')}: {f['value']}"
                    for f in self_facts
                ]
                parts.append("내 자신에 대해 (페르소나 일관성):\n" + "\n".join(lines))
            if user_facts:
                lines = [f"  - {f['key']}: {f['value']}" for f in user_facts]
                parts.append("사용자에 대해 학습한 사실:\n" + "\n".join(lines))
            # 노출된 fact key들 — last_used_at 갱신 (활용도 추적)
            try:
                memory.mark_facts_used([f["key"] for f in facts])
            except Exception:
                pass
    except Exception:
        pass

    return "\n".join(parts)


def _build_situation(
    ctx: StateContext,
    perception: PerceptionState | None,
    work_minutes: float | None,
    face: FaceState | None = None,
) -> str:
    """전체 컨텍스트 (단일 문자열). 호환성 유지 — 일부 caller가 직접 사용.

    agent _tick은 prefix/suffix 분리해 cache 활용. _build_situation은 두 부분
    합쳐 반환 → 기존 caller(테스트 포함) 동일 동작.
    """
    return _build_situation_prefix(ctx) + "\n" + _build_situation_suffix(
        ctx, perception, work_minutes, face,
    )


def _build_situation_suffix(
    ctx: StateContext,
    perception: PerceptionState | None,
    work_minutes: float | None,
    face: FaceState | None = None,
    weather_line: str | None = None,
) -> str:
    """매 tick 변하는 컨텍스트 — cache 적용 X.

    weather_line: 호출자(_tick)가 await로 가져온 한 줄 (예: "분당 맑음 18°C ...").
    None이면 weather 섹션 자체 생략 (키 미설정/네트워크 실패).
    """
    now = datetime.now()
    now_ts = now.timestamp()
    period = period_ko(now)

    temp = (perception.temperature_c
            if perception and perception.temperature_c is not None else None)
    dist = (perception.person_distance_cm
            if perception and perception.person_distance_cm > 0 else None)
    recent = _gather_recent_messages()

    parts = []

    # === 가장 prominent: 현재 대화 흐름 ===
    # 사용자 마지막 발화 + 내가 그 후 응답했는지 명시. agent가 흐름 잃지 않도록.
    exchange = _build_current_exchange()
    if exchange:
        parts.append(exchange)
        parts.append("")  # 시각적 구분

    parts.extend([
        f"현재 시각: {now.strftime('%Y-%m-%d %H:%M')} ({period})",
        f"사용자 존재: {'있음' if ctx.user_present else '없음'}",
        f"내 상태: {ctx.state.value}",
    ])
    # 카메라에 사람 2명+ 인지 — 단, 짧은 false positive(<10s) 또는 image 없이는
    # 사실로 단정 X. agent 프롬프트가 이미지 동봉 시만 언급하라고 안내.
    # 지속(>10s) + 최근(<30s) 모두 만족해야 신호로 노출.
    if (perception and perception.person_count >= 2
            and now_ts - perception.person_count_at < 30):
        # person_count 지속 시간 추정 (count_at은 마지막 *변화* 시각)
        # 짧으면 의심 단서로만 표시, 길면 강한 신호
        age = now_ts - perception.person_count_at
        hint = (
            "(짧게 잡힘 — 오인 가능, 이미지 확인 전엔 발화 X)"
            if age < 10 else
            "(지속 인지됨)"
        )
        parts.append(
            f"카메라에 사람 {perception.person_count}명 {hint}"
        )
    # 내 현재 표정 — 사용자에게 LCD로 보이는 것
    if face is not None:
        parts.append(f"내 표정: {face.expression.name}")
    if dist is not None:
        parts.append(f"거리: 약 {dist:.0f}cm")
    if temp is not None:
        parts.append(f"실내 온도: {temp:.1f}°C")

    # 사용자 현재 표정 — 최근 90초 안의 신호만 유효
    if perception and perception.current_emotion:
        emotion_age = now_ts - perception.current_emotion_at
        if emotion_age < 90 and perception.current_emotion != "neutral":
            parts.append(
                f"사용자 표정: {perception.current_emotion} "
                f"({int(emotion_age)}초 전 관측)"
            )

    # 내 머리 방향 — 카메라가 머리에 달려있어 이미지 시점에 영향
    if perception and perception.head_pan_deg is not None:
        from src.config import PAN_CENTER_DEG, TILT_CENTER_DEG
        pan_off = perception.head_pan_deg - PAN_CENTER_DEG
        tilt_off = perception.head_tilt_deg - TILT_CENTER_DEG
        head_bits = []
        # pan_off 양수/음수 의미는 PAN_INVERT 따라 다른데 agent가 학습하게 raw.
        if abs(pan_off) < 5:
            head_bits.append("좌우 정면")
        else:
            head_bits.append(f"pan center 대비 {pan_off:+.0f}°")
        if abs(tilt_off) < 5:
            head_bits.append("상하 수평")
        else:
            head_bits.append(f"tilt center 대비 {tilt_off:+.0f}°")
        parts.append(f"내 머리 방향: {', '.join(head_bits)}")

    # 활동 추론 신호들 — 사용자가 지금 어떤 모드인지 (최근 2분 안의 신호만)
    if perception:
        activity_parts = []
        if (perception.gaze_target
                and now_ts - perception.gaze_target_at < 120):
            gaze_ko = {
                "front": "모니터 응시 중",
                "down": "아래(책상/핸드폰) 응시 중",
                "side": "옆을 보고 있음",
            }.get(perception.gaze_target, perception.gaze_target)
            activity_parts.append(f"시선={gaze_ko}")
        if (perception.activity_level
                and now_ts - perception.activity_level_at < 120):
            level_ko = {
                "still": "거의 안 움직임 (멍/깊이 집중)",
                "focused": "잔잔히 움직임 (집중 작업)",
                "normal": "보통",
                "restless": "자주 큰 움직임 (산만)",
            }.get(perception.activity_level, perception.activity_level)
            activity_parts.append(f"활동성={level_ko}")
        if (perception.posture_category
                and now_ts - perception.posture_category_at < 120):
            posture_ko = {
                "upright": "똑바로 앉음",
                "slouched": "구부정 (거북목)",
                "leaning": "어깨 기울어짐",
            }.get(perception.posture_category, perception.posture_category)
            activity_parts.append(f"자세={posture_ko}")
        if activity_parts:
            parts.append("활동: " + ", ".join(activity_parts))

    # (오늘 첫 등장 시각은 prefix에 — cache 활용)

    # 사용자가 잠시 자리 비웠는지 (현재 부재 중일 때만 의미 있음)
    if not ctx.user_present and ctx.last_user_seen_at:
        absent_min = (now_ts - ctx.last_user_seen_at) / 60
        parts.append(f"부재 시간: 약 {int(absent_min)}분")

    if work_minutes is not None:
        parts.append(f"현재 작업 세션: {int(work_minutes)}분 째")

    # 오늘 누적 작업 시간 — 30초 캐시 (분 단위 변화라 충분)
    today_total_sec = _ttl_cache(
        "today_total_seconds", 30.0, memory.today_total_seconds,
    )
    if today_total_sec is not None and today_total_sec >= 60:
        parts.append(f"오늘 누적 작업: {int(today_total_sec / 60)}분")

    if ctx.last_proactive_at:
        gap = now_ts - ctx.last_proactive_at
        parts.append(f"내가 마지막 발화: {int(gap)}초 전")

    # 최근 fire된 proactive 트리거 (3분 안) — 5초 캐시
    last_trig = _ttl_cache(
        "last_proactive_log", 5.0,
        lambda: memory.last_proactive_log(within_minutes=3.0),
    )
    if last_trig:
        trig_gap = int(now_ts - last_trig["ts"])
        parts.append(
            f"최근 트리거({trig_gap}초 전): {last_trig['trigger']} "
            f"— \"{last_trig['message']}\""
        )

    parts.append(f"최근 대화:\n{recent}")

    # (학습된 facts는 prefix에 — cache 활용)

    # 시간대 힌트 — 식사/취침 시간대 명시 (agent가 자연스럽게 챙기게)
    hint = _time_hint(now)
    if hint:
        parts.append(f"시간대 힌트: {hint}")

    # 날씨 — 호출자가 await로 미리 가져와 주입. 없으면 섹션 자체 생략.
    if weather_line:
        parts.append(f"날씨: {weather_line}")

    # 최근 24시간 사용자 표정 요약 — 5분 캐시 (시간 단위 변화)
    snap_summary = _ttl_cache(
        "snapshot_summary_24h", 300.0,
        lambda: memory.snapshot_summary(hours_back=24.0),
    )
    if snap_summary:
        stats = ", ".join(f"{k}={v}" for k, v in snap_summary.items())
        parts.append(f"최근 24h 사용자 표정 통계: {stats}")

    # 로봇 자신의 스탯 (Tamagotchi) — 표정 톤에 영향
    try:
        from src.brain import stats as robot_stats
        parts.append(robot_stats.summary_text())
        parts.append(f"내 컨디션: {robot_stats.mood_label()}")
    except Exception:
        pass

    # 자기 발화 별도 섹션 — recency bias 활용 위해 closing 직전.
    # 모델이 "내가 최근에 뭐라 했지"를 명확히 인지하도록 사용자 발화와 분리.
    my_recent = _gather_my_recent_speech()
    parts.append("")
    parts.append(
        "내가 최근 한 말 (절대 그대로 반복 X — 같은 주제도 다른 표현/각도로):\n"
        + my_recent
    )

    parts.append("")
    parts.append(
        "지금 행동 결정 — 다음 순서로 자문:\n"
        "1) 위 컨텍스트에 '유효 anchor'(새 행동/이미지 단서/학습된 fact/환경 변화/"
        "시간대 전환/사용자 직접 발화)가 있나? 없으면 → do_nothing 또는 set_expression.\n"
        "2) anchor 있으면 → 그 anchor를 구체적으로 살린 멘트 1-2문장. 'generic 안부' X.\n"
        "3) '내가 최근 한 말' 목록에 비슷한 거 있으면 → 다른 각도 또는 do_nothing.\n"
        "컨디션은 멘트 톤에 녹여 (졸린/외로운/신난 어조). 톤 ≠ 주제."
    )
    return "\n".join(parts)


class RobotAgent:
    """주기 결정 에이전트."""

    def __init__(
        self,
        face: FaceState,
        ctx: StateContext,
        perception: PerceptionState | None,
        servos=None,
        get_session_id=None,
    ) -> None:
        self.face = face
        self.ctx = ctx
        self.perception = perception
        self.servos = servos
        self.get_session_id = get_session_id
        self._last_speak_at = 0.0
        self._last_dance_at = 0.0   # 격렬 동작 방지 cooldown
        # 마지막으로 이미지 첨부한 시각 + 그때의 활동 상태 — 변화 감지용
        self._last_vision_at: float = 0.0
        self._last_vision_emotion: str | None = None
        self._last_vision_activity: str | None = None
        self._last_vision_gaze: str | None = None
        self._last_user_seen_for_vision: float | None = None
        # 발화 직후 baseline — 다음 tick에 사용자 반응 비교용.
        # (표정/시선/활동 변화 = 멘트가 가닿았다는 신호)
        self._post_speak_baseline: dict | None = None
        self._last_speak_text: str | None = None

    def _should_skip(self) -> bool:
        if not LLM_AVAILABLE:
            return True
        if self.ctx.state in (State.TALKING, State.LISTENING, State.GREETING):
            return True
        # 호명("네모") 최근 15초 안엔 user_present/quiet hours 모두 bypass —
        # 사용자가 명시적으로 부른 거니 무조건 응답.
        now = time.time()
        user_recently_called = (
            self.perception is not None
            and getattr(self.perception, "last_user_called_at", 0) > 0
            and now - self.perception.last_user_called_at < 15.0
        )
        if not self.ctx.user_present and not user_recently_called:
            return True
        # quiet hours(22~7시)엔 proactive tick skip — 단, 사용자 최근 10초
        # 안에 발화/호명이면 응답해야 자연스러움.
        if _is_quiet_hours():
            user_recent_speech = (
                self.perception is not None
                and getattr(self.perception, "last_user_speech_at", 0) > 0
                and now - self.perception.last_user_speech_at < 10.0
            )
            if not user_recent_speech and not user_recently_called:
                return True
        return False

    def _should_idle_skip(self) -> bool:
        """Base interval tick 도달 시 Claude 호출 정당화 휴리스틱.

        변화 트리거 tick은 이 함수가 호출되지 않음 — 이미 의미 있는 신호 변화.
        base interval tick은 "사용자 조용 + 신호 정적"에서 도달하는 경우라
        대부분 Claude가 do_nothing() 결정. 그 호출 자체를 코드로 skip해 비용 0.

        호출 정당화 (OR — 하나라도 충족하면 호출):
        1. 사용자 발화 최근 60s
        2. 사용자 호명 최근 60s
        3. 새 등장 후 아직 인사 안 함
        4. 마지막 agent 발화로부터 chitchat_min(240s) 경과 — 잡담 기회
        5. 작업 임계 통과 직후 (±60s window)
        모두 미충족 → idle skip (코드가 do_nothing 결정).
        """
        now = time.time()
        p = self.perception
        if p is not None:
            if getattr(p, "last_user_speech_at", 0) > 0 \
                    and now - p.last_user_speech_at < 60.0:
                return False
            if getattr(p, "last_user_called_at", 0) > 0 \
                    and now - p.last_user_called_at < 60.0:
                return False
        if self.ctx.last_user_seen_at \
                and now - self.ctx.last_user_seen_at < 60.0 \
                and self._last_speak_at < self.ctx.last_user_seen_at:
            return False
        if self._last_speak_at == 0.0 \
                or now - self._last_speak_at >= BEHAVIOR.chitchat_min_interval_sec:
            return False
        if self.get_session_id is not None:
            try:
                sid = self.get_session_id()
                if sid is not None:
                    work_min = memory.current_work_duration(sid) / 60
                    for threshold in (
                        BEHAVIOR.work_break_gentle_minutes,
                        BEHAVIOR.work_break_warn_minutes,
                        BEHAVIOR.work_break_strong_minutes,
                        BEHAVIOR.work_break_alarm_minutes,
                    ):
                        if threshold <= work_min < threshold + 1.0:
                            return False
            except Exception:
                pass
        return True

    async def run(self, interval_sec: float | None = None) -> None:
        """Polling 루프 — 2초마다 신호 스냅샷 비교.

        tick 발동 조건 (둘 중 하나):
          - interval_sec(기본 30s) 도달 — 평소 idle 점검
          - perception 신호 변화 감지 (emotion/gaze/activity/person_count/posture)
            AND 최소 trigger gap(5s) 경과 — 빠른 표정 변화 burst에서 rate-limit 보호

        결과: 평균 30s 주기 유지하면서 변화 순간엔 즉시 반응 (2~5초 latency).
        """
        if interval_sec is None:
            interval_sec = BEHAVIOR.agent_interval_sec
        if not LLM_AVAILABLE:
            log.info(
                f"agent 비활성 — LLM backend={LLM_BACKEND}, "
                f"ANTHROPIC_API_KEY={'O' if ANTHROPIC_API_KEY else 'X'}"
            )
            return
        log.info(
            f"robot agent 시작 (backend={LLM_BACKEND}, "
            f"interval={interval_sec}s, change-triggered)"
        )
        poll_sec = 2.0
        trigger_min_gap_sec = 5.0   # 변화 감지 시 최소 cooldown (연속 burst 방지)
        last_tick_at = 0.0
        last_signals = self._snapshot_signals()
        while True:
            await asyncio.sleep(poll_sec)
            if self._should_skip():
                last_signals = self._snapshot_signals()
                continue
            now = time.time()
            cur_signals = self._snapshot_signals()
            elapsed = now - last_tick_at
            changed = cur_signals != last_signals
            base_due = elapsed >= interval_sec
            change_due = changed and elapsed >= trigger_min_gap_sec
            should_tick = base_due or change_due
            if not should_tick:
                continue
            # base interval만 도달한 경우(신호 변화 X) — 휴리스틱으로
            # "Claude 부를 거리 있나" 사전 판단. 없으면 호출 자체 skip.
            # 변화 트리거 진입은 이미 의미 있는 변화라 그대로 호출.
            if base_due and not change_due and self._should_idle_skip():
                log.debug(f"agent tick: idle skip (elapsed={elapsed:.0f}s)")
                last_tick_at = now
                last_signals = cur_signals
                continue
            if change_due and not base_due:
                log.info(f"agent tick: 변화 트리거 (elapsed={elapsed:.0f}s)")
            try:
                await self._tick()
                last_tick_at = now
            except Exception as e:
                log.warning(f"agent tick 에러: {e}")
            last_signals = cur_signals

    def _snapshot_signals(self) -> tuple:
        """현재 perception 신호 스냅샷 — 변화 감지용. None도 의미있는 값으로 취급."""
        p = self.perception
        if p is None:
            return ()
        return (
            p.current_emotion,
            p.gaze_target,
            p.activity_level,
            p.person_count,
            p.posture_category,
            # 새 음성 발화도 트리거 신호 — STT 결과 들어오면 즉시 agent tick
            p.last_user_speech_at,
        )

    async def _tick(self) -> None:
        work_min: float | None = None
        if self.get_session_id is not None:
            sid = self.get_session_id()
            if sid is not None:
                try:
                    work_min = memory.current_work_duration(sid) / 60
                except Exception as e:
                    log.debug(f"work duration 조회 실패: {e}")
        # 날씨 — 캐시 hit면 즉시, 만료 시만 네트워크. 실패는 None.
        weather_line: str | None = None
        try:
            from src.integrations.weather import get_client as get_weather_client
            snap = await get_weather_client().snapshot()
            if snap is not None:
                weather_line = snap.one_liner()
        except Exception as e:
            log.debug(f"weather snapshot 실패 (무시): {e}")
        prefix = _build_situation_prefix(self.ctx)
        suffix = _build_situation_suffix(
            self.ctx, self.perception, work_min, face=self.face,
            weather_line=weather_line,
        )
        # 직전 발화 후 사용자 반응 — 있으면 suffix 앞쪽에 prepend (눈에 띄게)
        reaction = self._post_speak_reaction_text()
        if reaction:
            suffix = reaction + "\n\n" + suffix
        loop = asyncio.get_running_loop()

        # cv2.imencode은 sync + 30~80ms blocking — executor로 빼서 event loop 보호
        image_b64 = await loop.run_in_executor(None, self._maybe_encode_frame)

        # content blocks: [image?] + [prefix cached] + [suffix]
        # prefix에 cache_control 붙여 5분 TTL 안에 cache hit. suffix는 매번 새로움.
        content: list[dict] = []
        if image_b64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            })
        content.append({
            "type": "text",
            "text": prefix,
            "cache_control": {"type": "ephemeral"},
        })
        content.append({"type": "text", "text": suffix})

        messages: list[dict] = [{"role": "user", "content": content}]

        # 모델 선택 — 사용자가 최근 15s 안에 발화/호명했으면 Sonnet(응답 보장),
        # 그 외 idle 점검(대부분 do_nothing)엔 Haiku로 비용 절감.
        now_for_model = time.time()
        last_speech = (
            getattr(self.perception, "last_user_speech_at", 0)
            if self.perception is not None else 0
        )
        last_called = (
            getattr(self.perception, "last_user_called_at", 0)
            if self.perception is not None else 0
        )
        user_recent = (
            (last_speech > 0 and now_for_model - last_speech < 15.0)
            or (last_called > 0 and now_for_model - last_called < 15.0)
        )
        tick_model = CLAUDE_MODEL if user_recent else CLAUDE_MODEL_LIGHT

        for round_idx in range(_MAX_AGENT_ROUNDS):
            actions, full_messages = await loop.run_in_executor(
                None,
                lambda m=messages, mdl=tick_model: conversation._client.generate_with_tools(
                    "", _TOOLS, messages=m, model=mdl,
                ),
            )
            if not actions:
                return

            info_actions = [a for a in actions if a["name"] in _INFO_TOOLS]
            real_actions = [a for a in actions if a["name"] not in _INFO_TOOLS]

            # decision trace — 어떤 결정 내렸는지 한 줄로
            action_names = [a["name"] for a in actions]
            log.info(
                f"agent decision[r{round_idx}]: {', '.join(action_names)}"
                + (" (image)" if (round_idx == 0 and image_b64) else "")
            )

            # 행동 액션이 섞여 있으면 그것만 실행하고 종료.
            # (Claude가 recall과 함께 행동까지 한 번에 결정한 경우)
            if real_actions:
                for action in real_actions:
                    try:
                        await self._execute(action)
                    except Exception as e:
                        log.warning(f"action 실패 ({action.get('name')}): {e}")
                return

            # info 도구만 → 결과를 모아 다음 round로
            if not info_actions:
                return
            tool_results = []
            for action in info_actions:
                result_text = self._run_info_tool(action)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": action["id"],
                    "content": result_text,
                })
            messages = list(full_messages) + [
                {"role": "user", "content": tool_results},
            ]
            prompt = None
        log.debug(f"agent: {_MAX_AGENT_ROUNDS} round 후에도 행동 결정 X — 종료")

    def _maybe_encode_frame(self) -> str | None:
        """조건 만족 시 perception.last_frame을 JPEG base64로 인코딩.

        조건 (OR):
          1) 마지막 첨부로부터 max_interval_sec 이상 경과
          2) 표정/활동성/시선 중 하나가 직전 첨부 때와 달라짐
          3) 사용자가 새로 등장한 직후 (last_user_seen_at 변함)
        AND 모두:
          - agent_vision_enabled True
          - perception/last_frame 존재
          - last_frame 30초 이내 (stale 방지)
          - 마지막 첨부로부터 min_interval_sec 이상 경과
        """
        if not BEHAVIOR.agent_vision_enabled:
            return None
        # 로컬 백엔드(Qwen2.5-3B)는 vision 미지원 — 첨부해도 의미 없음.
        if LLM_BACKEND == "local":
            return None
        if self.perception is None or self.perception.last_frame is None:
            return None
        now = time.time()
        if now - self.perception.last_frame_at > 30:
            return None
        # 최소 간격 — 너무 자주 첨부 방지
        if now - self._last_vision_at < BEHAVIOR.agent_vision_min_interval_sec:
            return None

        # 변화 감지
        cur_emotion = self.perception.current_emotion
        cur_activity = self.perception.activity_level
        cur_gaze = self.perception.gaze_target
        cur_user_seen = self.ctx.last_user_seen_at

        changed = (
            cur_emotion != self._last_vision_emotion
            or cur_activity != self._last_vision_activity
            or cur_gaze != self._last_vision_gaze
            or (cur_user_seen is not None
                and cur_user_seen != self._last_user_seen_for_vision)
        )
        # max interval — 변화 없어도 한 번씩
        forced = (now - self._last_vision_at) >= BEHAVIOR.agent_vision_max_interval_sec

        if not changed and not forced:
            return None

        # 인코딩 시도 — frame을 copy해서 다른 task의 in-place 수정으로부터 격리
        try:
            from src.brain.image_encoding import encode_jpeg_b64
            frame_snapshot = self.perception.last_frame
            # numpy면 .copy(), 그 외 객체는 그대로 (잠재적 위험은 cv2 buffer 정도 — 보통 안전)
            if hasattr(frame_snapshot, "copy"):
                try:
                    frame_snapshot = frame_snapshot.copy()
                except Exception:
                    pass
            b64 = encode_jpeg_b64(
                frame_snapshot,
                quality=BEHAVIOR.agent_vision_jpeg_quality,
                max_side_px=BEHAVIOR.agent_vision_max_side_px,
            )
        except Exception as e:
            log.debug(f"frame 인코딩 실패: {e}")
            return None
        if b64 is None:
            return None

        log.info(
            f"agent vision attach (changed={changed} forced={forced} "
            f"size={len(b64) * 3 // 4} B)"
        )
        self._last_vision_at = now
        self._last_vision_emotion = cur_emotion
        self._last_vision_activity = cur_activity
        self._last_vision_gaze = cur_gaze
        self._last_user_seen_for_vision = cur_user_seen
        return b64

    def _run_info_tool(self, action: dict) -> str:
        name = action["name"]
        inp = action.get("input", {}) or {}
        if name == "recall":
            kw = (inp.get("keyword") or "").strip()
            days = float(inp.get("days") or 7)
            try:
                rows = memory.search_conversation(kw, days=days, limit=8)
            except Exception as e:
                return f"recall 실패: {e}"
            if not rows:
                return f"'{kw}' 관련 기록 없음 (최근 {days:g}일)"
            lines = [f"'{kw}' 검색 결과 ({len(rows)}건):"]
            for r in rows:
                who = "나" if r["speaker"] == "robot" else "사용자"
                ts = datetime.fromtimestamp(r["ts"]).strftime("%m-%d %H:%M")
                lines.append(f"  [{ts}] {who}: {r['text']}")
            return "\n".join(lines)
        return f"알 수 없는 정보 도구: {name}"

    async def _execute(self, action: dict) -> None:
        name = action.get("name")
        inp = action.get("input", {}) or {}
        if name == "do_nothing":
            log.debug("agent: 침묵 선택")
            return
        if name == "speak":
            await self._do_speak(inp)
        elif name == "set_expression":
            self._do_set_expression(inp)
        elif name == "dance":
            await self._do_dance(inp)
        elif name == "remember_fact":
            self._do_remember_fact(inp)

    def _do_remember_fact(self, inp: dict) -> None:
        key = (inp.get("key") or "").strip()
        value = (inp.get("value") or "").strip()
        if not key or not value:
            return
        # 일정성 정보면 schedules 테이블로 라우팅 — learned_facts 오염 방지.
        # schedule_extractor가 못 잡은 케이스를 agent가 fact로 박는 패턴 차단.
        try:
            if memory.is_schedule_like(key, value):
                memory.add_schedule(
                    event_type="reminder",
                    event_datetime=value,  # ISO 파싱 어려우면 원문 그대로
                    description=f"{key}: {value}",
                    confidence=0.5,
                )
                log.info(f"📅 [agent] schedule (fact 대신): {key} = {value}")
                return
            memory.remember_fact(key, value, category="user", source="agent")
            log.info(f"🧠 [agent] remembered: {key} = {value}")
        except Exception as e:
            log.warning(f"remember_fact 실패: {e}")

    async def _do_speak(self, inp: dict) -> None:
        text = (inp.get("text") or "").strip()
        if not text:
            log.info(f"_do_speak skip: text 비어 있음 (inp={inp})")
            return
        now = time.time()
        # 사용자가 최근 8초 안에 말 걸면 cooldown 우회 — 직접 발화엔 응답 보장
        user_recent_speech = (
            self.perception is not None
            and getattr(self.perception, "last_user_speech_at", 0) > 0
            and now - self.perception.last_user_speech_at < 8.0
        )
        # 호명("네모", "네모야") 후 60초 동안 "대화 활성" — 그 안엔 호명 없는
        # 추가 발화도 응답 OK (자연 대화 흐름). 60초 지나면 다시 침묵 모드.
        called_at = (
            getattr(self.perception, "last_user_called_at", 0)
            if self.perception is not None else 0
        )
        user_recently_called = called_at > 0 and (now - called_at) < 60.0
        # 가장 강한 신호 — 방금(15s) 직접 호명. cooldown/quiet hours/대화 무관 우선.
        user_just_called = called_at > 0 and (now - called_at) < 15.0

        # 🚨 hard gate — 호명 없으면 사용자 발화에 응답 X.
        # 사용자가 일상 대화 ("점심 뭐 먹지") 하는데 네모가 다 끼어드는 거 거슬림.
        # 호명한 발화 또는 호명 후 60s 윈도우 안일 때만 사용자 발화 응답 허용.
        # proactive(주기적 tick, 사용자 발화 X)는 이 가드 안 걸림 — cooldown만 적용.
        if user_recent_speech and not user_recently_called:
            log.info(
                f"_do_speak skip: 호명 없는 사용자 발화 — 응답 차단 "
                f"(text='{text[:30]}...')"
            )
            return

        gap = now - self._last_speak_at
        if (not user_just_called
                and gap < BEHAVIOR.agent_speak_min_gap_sec):
            log.info(
                f"_do_speak skip: cooldown {gap:.0f}s < "
                f"{BEHAVIOR.agent_speak_min_gap_sec:.0f}s "
                f"(text='{text[:30]}...', user_recent={user_recent_speech}, "
                f"called={user_recently_called}, just_called={user_just_called})"
            )
            return
        self._last_speak_at = now
        # 표정 함께 지정됐으면 적용 (대문자/소문자 모두 허용)
        expr_name = inp.get("expression")
        if expr_name:
            ex = expr.EXPRESSIONS_BY_NAME.get(expr_name.lower())
            if ex is not None:
                self.face.apply_expression(ex)
        log.info(f"🤖 [agent] {text}")
        # 발화는 fake_speak 백그라운드 task로 — face.show_speech 자동
        from src.audio.fake_tts import speak as fake_speak
        asyncio.create_task(fake_speak(self.face, text), name="agent_fake_speak")
        self.ctx.last_proactive_at = now
        # 발화 직후 사용자 baseline 캡처 — 다음 tick에 변화 비교
        if self.perception is not None:
            self._post_speak_baseline = {
                "emotion": self.perception.current_emotion,
                "gaze": self.perception.gaze_target,
                "activity": self.perception.activity_level,
            }
            self._last_speak_text = text
        try:
            memory.log_robot(text, kind="agent_speak")
        except Exception:
            pass

    def _post_speak_reaction_text(self) -> str | None:
        """직전 발화(60초 이내) 후 사용자 반응 요약 — 컨텍스트에 추가용.

        baseline 대비 emotion/gaze/activity 변화가 있으면 "반응 있음",
        없으면 "반응 없음". 60초 지나면 None (의미 없는 비교 방지).
        """
        if self._post_speak_baseline is None:
            return None
        now = time.time()
        elapsed = now - self._last_speak_at
        if elapsed > 60 or elapsed < 1:
            return None
        if self.perception is None:
            return None
        base = self._post_speak_baseline
        cur_emotion = self.perception.current_emotion
        cur_gaze = self.perception.gaze_target
        cur_activity = self.perception.activity_level
        changes: list[str] = []
        if cur_emotion != base["emotion"]:
            changes.append(f"표정 {base['emotion'] or '?'} → {cur_emotion or '?'}")
        if cur_gaze != base["gaze"]:
            changes.append(f"시선 {base['gaze'] or '?'} → {cur_gaze or '?'}")
        if cur_activity != base["activity"]:
            changes.append(
                f"활동성 {base['activity'] or '?'} → {cur_activity or '?'}"
            )
        prefix = f"내 직전 발화 ({int(elapsed)}초 전) \"{self._last_speak_text}\" 이후 사용자 변화"
        if changes:
            return f"{prefix}: {', '.join(changes)} (멘트가 가닿은 듯 — 흐름 살릴지 판단)"
        return f"{prefix}: 변화 없음 (사용자 무반응 — 같은 톤 반복 피해)"

    def _do_set_expression(self, inp: dict) -> None:
        expr_name = inp.get("expression")
        if not expr_name:
            return
        ex = expr.EXPRESSIONS_BY_NAME.get(expr_name.lower())
        if ex is None:
            return
        log.info(f"🤖 [agent] 표정 → {expr_name}")
        self.face.apply_expression(ex)

    async def _do_dance(self, inp: dict) -> None:
        if self.servos is None:
            return
        # cooldown — 격렬 동작 방지
        now = time.time()
        if now - self._last_dance_at < BEHAVIOR.agent_dance_min_gap_sec:
            remain = BEHAVIOR.agent_dance_min_gap_sec - (now - self._last_dance_at)
            log.info(f"agent dance skip — cooldown {remain:.0f}s 남음")
            return
        # 진폭/길이 cap — agent가 큰 값 보내도 자동 제한
        beats = max(2, min(int(inp.get("beats", 4)), 4))   # 최대 4 beats
        bpm = max(70, min(int(inp.get("bpm", 100)), 110))   # 70~110 BPM
        log.info(f"🤖 [agent] dance ({beats} beats @ {bpm} BPM)")
        from src.motion import poses
        try:
            self._last_dance_at = now
            async with motion_busy_scope(self.ctx):
                await poses.dance(self.servos, self.face, bpm=bpm, beats=beats)
        except Exception as e:
            log.warning(f"agent dance 에러: {e}")
