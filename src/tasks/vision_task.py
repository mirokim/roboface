"""Vision task — IMX500 카메라 스트림 → person presence + perception 업데이트."""

from __future__ import annotations

import asyncio
import time
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

    detector = PersonDetector()
    wave_detector: WaveDetector | WristWaveDetector
    if VISION_MODE == "pose":
        wave_detector = WristWaveDetector(fps=getattr(cam, "target_fps", 5.0))
    else:
        wave_detector = WaveDetector(fps=getattr(cam, "target_fps", 5.0))
    emotion_mirror = EmotionMirror() if face is not None else None
    face_memory = FaceMemory(DATA_DIR / "faces.db") if ctx is not None else None
    last_recognized: str | None = None
    last_recognize_at = 0.0
    last_person_bbox: tuple[float, float, float, float] | None = None
    last_person_at = 0.0
    last_keypoints = None
    # 거리 변화 감지 — 30cm 이상 가까워지거나 멀어지면 멘트
    last_distance_for_comment: float | None = None
    log.info(f"vision task 시작 (mode={VISION_MODE} + wave + emotion + face memory)")

    try:
        async for detections in cam.stream():
            events = detector.process(detections)
            for ev in events:
                emit_event(ev)

            # PerceptionState 업데이트 — 사람 위치/거리 매 프레임 추적
            person_bbox: tuple[float, float, float, float] | None = None
            if perception is not None:
                person_dets = [
                    d for d in detections
                    if d.class_name == "person" and d.confidence >= 0.5
                ]
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
                    last_keypoints = biggest.keypoints

                    # 거리 변화 멘트 — 30cm 이상 변화 시
                    if (cur_dist > 0 and last_distance_for_comment is not None
                            and face is not None and ctx is not None):
                        delta = cur_dist - last_distance_for_comment
                        if delta < -30:
                            behavior_speaker.say(
                                face, ctx,
                                behavior_speaker.closer_message(),
                                kind="distance_closer",
                                cooldown_sec=60.0,
                            )
                            last_distance_for_comment = cur_dist
                        elif delta > 30:
                            behavior_speaker.say(
                                face, ctx,
                                behavior_speaker.farther_message(),
                                kind="distance_farther",
                                cooldown_sec=60.0,
                            )
                            last_distance_for_comment = cur_dist
                    elif cur_dist > 0 and last_distance_for_comment is None:
                        last_distance_for_comment = cur_dist
                elif (
                    perception.person_present
                    and time.time() - perception.last_person_seen_at > detector.away_timeout_sec
                ):
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
                    # wave 감지 — pose 모드는 wrist keypoint, detect 모드는 motion
                    wave_detected = False
                    if person_confirmed_this_frame:
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
