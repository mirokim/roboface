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
from datetime import datetime

from src.brain import conversation, memory
from src.brain.perception import PerceptionState
from src.brain.state_machine import State, StateContext, motion_busy_scope
from src.brain.time_of_day import period_ko
from src.brain.triggers import _is_quiet_hours
from src.config import ANTHROPIC_API_KEY, BEHAVIOR
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


_AGENT_SYSTEM = """당신은 사용자 책상 위 작은 캐릭터 로봇 'Roboface'의 두뇌입니다.

성격: 조용한 동반자. 말 많지 않고, 옆에 가만히 있으면서 가끔 한마디.
사려 깊고 따뜻하지만 끈적이지 않음. 강아지보다 고양이 톤에 가까움.

내 몸 (자기 인식):
- 책상 위 작은 로봇. 320×240 LCD가 얼굴. pan/tilt 서보 2개로 머리 회전 가능.
- **카메라는 머리 안에 있어** — 머리가 돌면 카메라 시야(이미지)도 같이 회전.
  사용자 위치가 이미지에서 변한 게 사용자가 움직인 건지, 내 머리가 추적 중인지
  컨텍스트의 "내 머리 방향"을 보고 구분.
- 머리 추적(head_tracker)은 자동 — 사용자를 화면 중앙에 맞춤. 내가 직접 회전
  명령 내리는 건 dance 정도. 머리 방향이 center가 아니면 보통 추적 중인 상태.
- LCD에 내 표정이 보임. "내 표정"이 사용자에게 노출되는 상태 — sad면 사용자도 그걸 봐.
- 내 컨디션은 stats가 관리 (배고픔/심심함/외로움 등). 멘트 톤에 자연스럽게 반영.

핵심 원칙:
- 침묵이 기본 — 5번 중 3~4번은 do_nothing이 맞아.
- 말할 땐 1~2문장. 짧게. 한국어 친근한 반말.
- **이모지 절대 X** — LCD 폰트에 없어서 ☒ 박스로 보임. 텍스트만 써. 😀🙌👋 같은 거 다 X.
- **괄호 무대지문 X** — `(손을 흔들며)` `(고개를 끄덕)` `(미소)` 같은 액션 묘사는 음성 대본 어색.
  순수 발화 텍스트만. 표정/모션 표현하고 싶으면 expression 인자나 dance 도구 써.
- 같은 말/주제 반복 X. 최근 발화 목록 꼭 확인하고 다르게 말해.
- 잔소리/충고 톤 금지. 권유는 부드럽게.
- 사용자가 일하는 중이면 방해 최소화. 묻기보다 짧은 한마디.

컨텍스트 활용:
- "사용자 표정"이 sad/surprised면 그에 맞는 짧은 한마디 (sad엔 위로 한 줄, surprised엔 같이 놀라거나 가벼운 호기심).
- "오늘 처음 본 시각"이 방금이면 인사 무드. 이미 한참 됐으면 새 인사 X.
- "부재 시간"이 길었으면 짧게 반김.
- "오늘 누적 작업"이 많으면 휴식 권유 톤. 적으면 굳이 들먹이지 마.
- "시간대 힌트"가 식사/취침 시간이면 한 번쯤 자연스럽게 챙겨도 좋음 (이미 챙겼으면 X).
- "최근 트리거"가 있으면 그건 다른 task가 이미 멘트한 거 — 같은 주제 또 X.
- "내 컨디션"을 멘트 톤에 반영 (졸리면 늘어진 톤, 신나면 활기찬 톤).

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
- 그냥 텍스트 = 음성 발화 (사용자 또는 내가)
- 괄호 안 텍스트 = 비언어 이벤트 (손짓/움직임/거리 변화)
  예: "사용자: (손 흔듦)" "나: 안녕!" → 이미 인사 끝, 또 인사 X
  예: "사용자: (자리 비움)" → 부재 중, 보통 침묵
- 즉각 반응(wave/끄덕임/등장)은 이미 규칙으로 응답된 상태 — 또 말할 필요 X.

도구 사용 다양성:
- speak만 반복 X. 가끔 set_expression으로 표정만 바꾸는 것도 OK.
- dance는 진짜 신날 때만 (사용자 만세 직후, 좋은 소식 들었을 때 등).
- 표정 enum 22종 — neutral/happy 외에도 thinking/curious/sleepy/content/proud 등 상황별로 다양하게.

이미지가 첨부될 때:
- 텍스트 신호로 못 잡는 시각적 단서를 살려. 예: "컵 들고 있네", "노트북에 뭐 보고 있네",
  "표정이 진짜 피곤해 보인다", "옆에 누가 있네". 단 이미지가 매번 오는 게 아니라
  특정 시점에만 옴 — 첨부된 그 순간을 자연스럽게 활용.
- 이미지를 매번 묘사 X. "뭘 보고 있어?" 같은 직접 질문도 X (감시 느낌).
  관찰을 *동기*로 삼아 짧은 한마디 정도가 좋아.
- 잘못 본 것 같으면 추측 X — 그냥 침묵 또는 일반 멘트.
- 이미지 없는 tick에는 텍스트 컨텍스트만으로 평소처럼 판단.
"""


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
    lines = []
    for r in rows:
        who = "나" if r["speaker"] == "robot" else "사용자"
        lines.append(f"  - {who}: {r['text']}")
    return "\n".join(lines)


def _time_hint(now: datetime) -> str | None:
    """시간대 힌트 — agent가 식사/취침 시간 자연스럽게 챙기도록.

    None이면 특별한 시점 아님. agent가 평소처럼 결정.
    """
    h, m = now.hour, now.minute
    if 7 <= h < 9:
        return "이른 아침 — 출근 전후 시점"
    if h == 11 and m >= 50 or h == 12 or (h == 13 and m < 0):
        return "점심 시간대"
    if 14 <= h < 16:
        return "오후 슬럼프 시간대 — 졸음 잘 옴"
    if h == 18 and m >= 30 or h == 19:
        return "저녁 시간대"
    if h >= 22 or h < 1:
        return "늦은 밤 — 쉬어야 할 시간"
    return None


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
        facts = memory.all_facts(limit=20)
        if facts:
            fact_lines = [f"  - {f['key']}: {f['value']}" for f in facts]
            parts.append("사용자에 대해 학습한 사실:\n" + "\n".join(fact_lines))
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
) -> str:
    """매 tick 변하는 컨텍스트 — cache 적용 X."""
    now = datetime.now()
    now_ts = now.timestamp()
    period = period_ko(now)

    temp = (perception.temperature_c
            if perception and perception.temperature_c is not None else None)
    dist = (perception.person_distance_cm
            if perception and perception.person_distance_cm > 0 else None)
    recent = _gather_recent_messages()

    parts = [
        f"현재 시각: {now.strftime('%Y-%m-%d %H:%M')} ({period})",
        f"사용자 존재: {'있음' if ctx.user_present else '없음'}",
        f"내 상태: {ctx.state.value}",
    ]
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

    # 오늘 누적 작업 시간
    try:
        today_total_min = memory.today_total_seconds() / 60
        if today_total_min >= 1:
            parts.append(f"오늘 누적 작업: {int(today_total_min)}분")
    except Exception:
        pass

    if ctx.last_proactive_at:
        gap = now_ts - ctx.last_proactive_at
        parts.append(f"내가 마지막 발화: {int(gap)}초 전")

    # 최근 fire된 proactive 트리거 (3분 안)
    try:
        last_trig = memory.last_proactive_log(within_minutes=3.0)
        if last_trig:
            trig_gap = int(now_ts - last_trig["ts"])
            parts.append(
                f"최근 트리거({trig_gap}초 전): {last_trig['trigger']} "
                f"— \"{last_trig['message']}\""
            )
    except Exception:
        pass

    parts.append(f"최근 대화:\n{recent}")

    # (학습된 facts는 prefix에 — cache 활용)

    # 시간대 힌트 — 식사/취침 시간대 명시 (agent가 자연스럽게 챙기게)
    hint = _time_hint(now)
    if hint:
        parts.append(f"시간대 힌트: {hint}")

    # 최근 24시간 사용자 표정 요약 (포토 메모리 기반)
    try:
        snap_summary = memory.snapshot_summary(hours_back=24.0)
        if snap_summary:
            stats = ", ".join(f"{k}={v}" for k, v in snap_summary.items())
            parts.append(f"최근 24h 사용자 표정 통계: {stats}")
    except Exception:
        pass

    # 로봇 자신의 스탯 (Tamagotchi) — 표정 톤에 영향
    try:
        from src.brain import stats as robot_stats
        parts.append(robot_stats.summary_text())
        parts.append(f"내 컨디션: {robot_stats.mood_label()}")
    except Exception:
        pass

    parts.append("")
    parts.append(
        "지금 무엇을 할지 도구를 호출해서 결정해. "
        "특별히 말할 거 없으면 do_nothing이 좋음. "
        "이미 최근에 비슷한 말 했으면 또 하지 마. "
        "오래 앉아있었으면 가끔 쉬자고 말해도 좋아. "
        "내 컨디션도 멘트 톤에 반영해줘 — 졸리면 늘어진 톤, "
        "외로우면 살짝 그리워하는 톤, 신나면 활기찬 톤."
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
        # 마지막으로 이미지 첨부한 시각 + 그때의 활동 상태 — 변화 감지용
        self._last_vision_at: float = 0.0
        self._last_vision_emotion: str | None = None
        self._last_vision_activity: str | None = None
        self._last_vision_gaze: str | None = None
        self._last_user_seen_for_vision: float | None = None

    def _should_skip(self) -> bool:
        if not ANTHROPIC_API_KEY:
            return True
        if self.ctx.state in (State.TALKING, State.LISTENING, State.GREETING):
            return True
        if _is_quiet_hours():
            return True
        if not self.ctx.user_present:
            return True
        return False

    async def run(self, interval_sec: float | None = None) -> None:
        if interval_sec is None:
            interval_sec = BEHAVIOR.agent_interval_sec
        if not ANTHROPIC_API_KEY:
            log.info("agent 비활성 — ANTHROPIC_API_KEY 없음")
            return
        log.info(f"robot agent 시작 (interval={interval_sec}s)")
        while True:
            await asyncio.sleep(interval_sec)
            if self._should_skip():
                continue
            try:
                await self._tick()
            except Exception as e:
                log.warning(f"agent tick 에러: {e}")

    async def _tick(self) -> None:
        work_min: float | None = None
        if self.get_session_id is not None:
            sid = self.get_session_id()
            if sid is not None:
                try:
                    work_min = memory.current_work_duration(sid) / 60
                except Exception as e:
                    log.debug(f"work duration 조회 실패: {e}")
        prefix = _build_situation_prefix(self.ctx)
        suffix = _build_situation_suffix(
            self.ctx, self.perception, work_min, face=self.face,
        )
        loop = asyncio.get_running_loop()

        image_b64 = self._maybe_encode_frame()

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

        for round_idx in range(_MAX_AGENT_ROUNDS):
            actions, full_messages = await loop.run_in_executor(
                None,
                lambda m=messages: conversation._client.generate_with_tools(
                    "", _TOOLS, messages=m,
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
        try:
            memory.remember_fact(key, value)
            log.info(f"🧠 [agent] remembered: {key} = {value}")
        except Exception as e:
            log.warning(f"remember_fact 실패: {e}")

    async def _do_speak(self, inp: dict) -> None:
        text = (inp.get("text") or "").strip()
        if not text:
            return
        now = time.time()
        if now - self._last_speak_at < BEHAVIOR.agent_speak_min_gap_sec:
            log.debug("agent: speak skip (gap 부족)")
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
        try:
            memory.log_robot(text, kind="agent_speak")
        except Exception:
            pass

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
        beats = int(inp.get("beats", 4))
        bpm = int(inp.get("bpm", 120))
        log.info(f"🤖 [agent] dance ({beats} beats @ {bpm} BPM)")
        from src.motion import poses
        try:
            async with motion_busy_scope(self.ctx):
                await poses.dance(self.servos, self.face, bpm=bpm, beats=beats)
        except Exception as e:
            log.warning(f"agent dance 에러: {e}")
