"""Vision task — IMX500 카메라 스트림 → person presence + perception 업데이트."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable

from src.brain.perception import PerceptionState
from src.brain.state_machine import StateContext
from src.config import DATA_DIR, VISION_MODE
from src.face.renderer import FaceState
from src.sensors.base import SensorEvent, SensorEventType
from src.tasks import behavior_speaker
from src.tasks.reactive_face import flash_expression
from src.utils.logger import get_logger
from src.vision.emotion_mirror import EMOTION_SMILE, EmotionMirror
from src.vision.face_memory import FaceMemory, detect_face_crop
from src.vision.person_detector import PersonDetector
from src.vision.pose_gestures import (
    HandsUpDetector, HeadNodDetector, HeadShakeDetector,
)
from src.vision.pose_stabilizer import PoseStabilizer
from src.vision.wave_detector import WaveDetector
from src.vision.wrist_wave_detector import WristWaveDetector

log = get_logger("vision_task")


async def run_vision(
    emit_event: Callable[[SensorEvent], None],
    perception: PerceptionState | None = None,
    face: FaceState | None = None,
    ctx: StateContext | None = None,
) -> None:
    """카메라 스트림 → person detector → sensor events + perception 업데이트."""
    try:
        from src.vision.camera import IMX500Camera
    except Exception as e:
        log.warning(f"vision 모듈 import 실패 (헤드리스 모드 유지): {e}")
        return

    try:
        cam = IMX500Camera(mode=VISION_MODE)
    except Exception as e:
        log.warning(f"AI Camera 초기화 실패 (mode={VISION_MODE}) — vision task 건너뜀: {e}")
        return

    detector = PersonDetector(
        min_confidence=0.1 if VISION_MODE == "pose" else 0.5,
        # pose 모드는 score가 자주 깜빡거려 5초로는 LEFT 토글 잦음 → 15초로 완화
        away_timeout_sec=15.0 if VISION_MODE == "pose" else 5.0,
    )
    fps = getattr(cam, "target_fps", 10.0)
    wave_detector: WaveDetector | WristWaveDetector
    hands_up_detector: HandsUpDetector | None = None
    head_nod_detector: HeadNodDetector | None = None
    head_shake_detector: HeadShakeDetector | None = None
    pose_stab: PoseStabilizer | None = None
    if VISION_MODE == "pose":
        wave_detector = WristWaveDetector(fps=fps)
        hands_up_detector = HandsUpDetector(fps=fps)
        head_nod_detector = HeadNodDetector(fps=fps)
        head_shake_detector = HeadShakeDetector(fps=fps)
        pose_stab = PoseStabilizer(fps=fps)
    else:
        wave_detector = WaveDetector(fps=fps)
    emotion_mirror = EmotionMirror() if face is not None else None
    face_memory = FaceMemory(DATA_DIR / "faces.db") if ctx is not None else None
    last_recognized: str | None = None
    last_recognize_at = 0.0
    last_person_bbox: tuple[float, float, float, float] | None = None
    last_person_at = 0.0
    last_keypoints = None
    # 거리 변화 감지 — 안정화 위해 최근 N개 median으로 비교
    last_distance_for_comment: float | None = None
    dist_window: deque[float] = deque(maxlen=8)
    last_diag_log_at = 0.0
    log.info(f"vision task 시작 (mode={VISION_MODE} + wave + emotion + face memory)")

    try:
        # pose 모드 점수는 매우 낮은 경우 많음 (HigherHRNet 특성)
        person_filter_conf = 0.1 if VISION_MODE == "pose" else 0.5

        async for detections in cam.stream():
            events = detector.process(detections)
            for ev in events:
                emit_event(ev)

            # PerceptionState 업데이트 — 사람 위치/거리 매 프레임 추적
            person_bbox: tuple[float, float, float, float] | None = None
            if perception is not None:
                person_dets = [
                    d for d in detections
                    if d.class_name == "person" and d.confidence >= person_filter_conf
                ]
                # 5초마다 진단 로그 (DEBUG)
                now_dt = time.time()
                if now_dt - last_diag_log_at > 5.0:
                    last_diag_log_at = now_dt
                    has_kp = any(d.keypoints is not None for d in person_dets)
                    log.debug(
                        f"vision: raw={len(detections)} "
                        f"person_dets={len(person_dets)} (필터≥{person_filter_conf}) "
                        f"keypoints={'있음' if has_kp else '없음'}"
                    )
                if person_dets:
                    biggest = max(
                        person_dets,
                        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                    )
                    cur_dist = detector._last_distance or -1.0
                    perception.update_person(
                        bbox=biggest.bbox,
                        distance_cm=cur_dist,
                    )
                    person_bbox = biggest.bbox
                    # pose 모드: stabilizer로 keypoints 스무딩
                    if pose_stab is not None:
                        last_keypoints = pose_stab.update(
                            biggest.keypoints, biggest.confidence,
                        )
                    else:
                        last_keypoints = biggest.keypoints

                    # 거리 변화 멘트 — 8프레임 median으로 안정화 후 비교
                    # (pose bbox가 매 프레임 흔들려 single sample 비교는 노이즈)
                    if cur_dist > 0:
                        dist_window.append(cur_dist)
                    locked_ok = pose_stab is None or pose_stab.is_locked
                    if (len(dist_window) >= dist_window.maxlen
                            and last_distance_for_comment is not None
                            and locked_ok
                            and face is not None and ctx is not None):
                        sorted_w = sorted(dist_window)
                        smoothed = sorted_w[len(sorted_w) // 2]   # median
                        delta = smoothed - last_distance_for_comment
                        if delta < -60:
                            try:
                                from src.brain import memory as _mem
                                _mem.log_user(
                                    f"(가까이 옴 — {int(smoothed)}cm)",
                                    kind="distance_closer",
                                )
                            except Exception:
                                pass
                            behavior_speaker.say(
                                face, ctx,
                                behavior_speaker.closer_message(),
                                kind="distance_closer",
                                cooldown_sec=180.0,
                            )
                            last_distance_for_comment = smoothed
                        elif delta > 60:
                            try:
                                from src.brain import memory as _mem
                                _mem.log_user(
                                    f"(멀어짐 — {int(smoothed)}cm)",
                                    kind="distance_farther",
                                )
                            except Exception:
                                pass
                            behavior_speaker.say(
                                face, ctx,
                                behavior_speaker.farther_message(),
                                kind="distance_farther",
                                cooldown_sec=180.0,
                            )
                            last_distance_for_comment = smoothed
                    elif (len(dist_window) >= dist_window.maxlen
                            and last_distance_for_comment is None):
                        sorted_w = sorted(dist_window)
                        last_distance_for_comment = sorted_w[len(sorted_w) // 2]
                else:
                    # 사람 없음 — pose stab에 0 push해서 lock 자연스럽게 풀림
                    # 거리 윈도우도 비움 (재등장 시 기준 새로 잡기)
                    if pose_stab is not None:
                        pose_stab.update(None, 0.0)
                    dist_window.clear()
                    if (perception.person_present
                            and time.time() - perception.last_person_seen_at
                            > detector.away_timeout_sec):
                        perception.clear_person()

            # person_bbox 잠깐 끊겨도 1.5초까지는 이전 bbox 유지 — wave detector가
            # 손 흔들기 중 person 인식 깜빡임으로 reset되는 거 방지.
            # 단 wave 감지는 person이 실제로 잡힌 프레임에서만 — grace 동안에는
            # bbox 영역에 다른 모션(커튼 등)이 wave로 잘못 인식되는 거 방지.
            now_t = time.time()
            if person_bbox is not None:
                last_person_bbox = person_bbox
                last_person_at = now_t
                effective_bbox = person_bbox
                person_confirmed_this_frame = True
            elif last_person_bbox is not None and now_t - last_person_at < 1.5:
                effective_bbox = last_person_bbox
                person_confirmed_this_frame = False
            else:
                effective_bbox = None
                last_person_bbox = None
                person_confirmed_this_frame = False

            # 손 흔들기 + 표정 거울 + 얼굴 인식 — 사람이 보일 때, frame 1회 캡처
            if effective_bbox is not None:
                try:
                    frame = cam.get_main_frame()
                    # 제스처 emit 게이트: pose 모드는 stabilizer가 lock된 상태에서만.
                    # detect 모드는 항상 person_confirmed 시 통과.
                    gesture_gate_ok = person_confirmed_this_frame and (
                        pose_stab is None or pose_stab.is_locked
                    )
                    # wave 감지 — pose 모드는 wrist keypoint, detect 모드는 motion
                    wave_detected = False
                    if gesture_gate_ok:
                        if isinstance(wave_detector, WristWaveDetector):
                            wave_detected = wave_detector.process(last_keypoints)
                        else:
                            wave_detected = wave_detector.process(
                                frame, effective_bbox,
                            )
                    if wave_detected:
                        emit_event(SensorEvent(
                            type=SensorEventType.GESTURE_WAVE,
                            data={"bbox": effective_bbox},
                        ))
                    # pose 모드: 추가 제스처들 (lock된 상태에서만)
                    if (gesture_gate_ok and last_keypoints is not None
                            and hands_up_detector is not None):
                        if hands_up_detector.process(last_keypoints):
                            emit_event(SensorEvent(
                                type=SensorEventType.GESTURE_HANDS_UP, data={},
                            ))
                        if head_nod_detector is not None and head_nod_detector.process(
                            last_keypoints,
                        ):
                            emit_event(SensorEvent(
                                type=SensorEventType.GESTURE_HEAD_NOD, data={},
                            ))
                        if head_shake_detector is not None and head_shake_detector.process(
                            last_keypoints,
                        ):
                            emit_event(SensorEvent(
                                type=SensorEventType.GESTURE_HEAD_SHAKE, data={},
                            ))
                    if emotion_mirror is not None and face is not None:
                        emotion = emotion_mirror.process(frame, effective_bbox)
                        if emotion == EMOTION_SMILE:
                            # 사용자가 웃으면 같이 웃음 (짧게)
                            from src.face.expressions import HAPPY
                            flash_expression(face, HAPPY, 1.5)
                    # 얼굴 인식 — 매 2초 한 번 (CPU 절약)
                    if (face_memory is not None and ctx is not None
                            and time.time() - last_recognize_at > 2.0):
                        last_recognize_at = time.time()
                        face_crop = detect_face_crop(frame, effective_bbox)
                        if face_crop is not None:
                            # 1) pending register 처리 우선
                            if ctx.pending_register_name:
                                name = ctx.pending_register_name
                                if face_memory.register(name, face_crop):
                                    ctx.user_name = name
                                    log.info(f"🎉 {name} 등록 완료")
                                    if face is not None:
                                        from src.face.expressions import STARSTRUCK
                                        flash_expression(face, STARSTRUCK, 1.5)
                                ctx.pending_register_name = None
                            else:
                                # 2) 인식 시도
                                match = face_memory.recognize(face_crop)
                                if match is not None:
                                    if match.name != last_recognized:
                                        last_recognized = match.name
                                        ctx.user_name = match.name
                                        log.info(
                                            f"😊 인식: {match.name} "
                                            f"(conf={match.confidence:.3f})"
                                        )
                                        try:
                                            from src.brain import memory as _mem
                                            _mem.log_user(
                                                f"(얼굴 인식: {match.name})",
                                                kind="face_recognized",
                                            )
                                        except Exception:
                                            pass
                                        if face is not None and ctx is not None:
                                            from src.face.expressions import HAPPY
                                            flash_expression(face, HAPPY, 1.0)
                                            behavior_speaker.say(
                                                face, ctx,
                                                behavior_speaker.face_greeting_message(
                                                    match.name,
                                                ),
                                                kind="face_recognize",
                                                cooldown_sec=120.0,
                                            )
                                else:
                                    last_recognized = None
                                    # 알 수 없는 사람 — user_name clear
                                    if ctx.user_name:
                                        ctx.user_name = None
                except Exception as e:
                    log.warning(f"vision frame 분석 에러: {e}")
            else:
                wave_detector.reset()
                if pose_stab is not None:
                    pose_stab.reset()
                if hands_up_detector is not None:
                    hands_up_detector.reset()
                last_recognized = None
                last_distance_for_comment = None
                if ctx is not None and ctx.user_name:
                    ctx.user_name = None
    except asyncio.CancelledError:
        log.info("vision task 취소 요청")
        raise
    except Exception as e:
        log.warning(f"vision task 에러: {e}")
    finally:
        try:
            cam.close()
        except Exception:
            pass
