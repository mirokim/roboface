"""Roboface Web UI — aiohttp 단일 페이지 대시보드.

기능:
- 실시간 face preview JPG (1초마다 갱신)
- 상태/스탯/대화 로그/스냅샷 갤러리
- 원격 명령: 말 시키기, 표정 변경, 댄스, 스냅샷, 얼굴 등록
- SSE로 상태 변화 푸시

인증: WEB_UI_PASSWORD 환경변수. 빈 문자열이면 서버 비활성화.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import os
import secrets
import time
from pathlib import Path

from aiohttp import web
import aiohttp_jinja2
import jinja2

from src.brain import memory, stats as robot_stats
from src.brain.perception import PerceptionState
from src.brain.state_machine import StateContext
from src.config import DATA_DIR, WEB_UI_PASSWORD, WEB_UI_PORT
from src.face import expressions as exprs
from src.face.renderer import FaceState, draw_face_to_surface
from src.utils.logger import get_logger

log = get_logger("web")

_HERE = Path(__file__).resolve().parent
STATIC_DIR = _HERE / "static"
TEMPLATES_DIR = _HERE / "templates"

SESSION_COOKIE = "roboface_session"
SESSION_TTL_SEC = 7 * 86400   # 1주일


# === 세션 토큰 (HMAC) ===

def _session_secret() -> bytes:
    """비밀번호 + 머신 ID 결합. 비번 바뀌면 모든 세션 무효화."""
    secret = WEB_UI_PASSWORD + str(DATA_DIR)
    return hashlib.sha256(secret.encode()).digest()


def _make_token(expires_at: float) -> str:
    payload = f"{expires_at:.0f}"
    sig = hmac.new(_session_secret(), payload.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{payload}.{sig}"


def _verify_token(token: str) -> bool:
    try:
        payload, sig = token.split(".", 1)
        expires_at = float(payload)
        if time.time() > expires_at:
            return False
        expected = hmac.new(
            _session_secret(), payload.encode(), hashlib.sha256,
        ).hexdigest()[:16]
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False


def _is_authed(request: web.Request) -> bool:
    if not WEB_UI_PASSWORD:
        return True   # 비번 없으면 무조건 통과 (LAN 신뢰 모드)
    token = request.cookies.get(SESSION_COOKIE, "")
    return _verify_token(token)


@web.middleware
async def auth_middleware(request: web.Request, handler):
    public = (
        request.path == "/login"
        or request.path.startswith("/static/")
    )
    if public or _is_authed(request):
        return await handler(request)
    # API는 401, 페이지는 redirect
    if request.path.startswith("/api/") or request.path == "/events":
        return web.json_response({"error": "unauthorized"}, status=401)
    raise web.HTTPFound("/login")


# === 라우트 ===

@aiohttp_jinja2.template("index.html")
async def page_index(request):
    return {}


@aiohttp_jinja2.template("login.html")
async def page_login(request):
    return {"error": request.query.get("error", "")}


async def post_login(request):
    data = await request.post()
    if data.get("password", "") == WEB_UI_PASSWORD and WEB_UI_PASSWORD:
        token = _make_token(time.time() + SESSION_TTL_SEC)
        resp = web.HTTPFound("/")
        resp.set_cookie(
            SESSION_COOKIE, token,
            max_age=SESSION_TTL_SEC, httponly=True, samesite="Lax",
        )
        raise resp
    raise web.HTTPFound("/login?error=1")


async def post_logout(request):
    resp = web.HTTPFound("/login")
    resp.del_cookie(SESSION_COOKIE)
    raise resp


# === API: 읽기 ===

async def api_state(request):
    app = request.app
    face: FaceState = app["face"]
    ctx: StateContext = app["ctx"]
    perception: PerceptionState = app["perception"]
    s = robot_stats.get()
    # API 사용량 (Claude 호출 비용)
    try:
        from src.brain.conversation import _usage
        usage_summary = {
            "calls": _usage.calls,
            "image_attaches": _usage.image_attaches,
            "input_tokens": _usage.input_tokens,
            "cache_read_tokens": _usage.cache_read_tokens,
            "output_tokens": _usage.output_tokens,
            "total_usd": round(_usage.estimated_usd(), 4),
            "hourly_usd": round(_usage.hourly_estimate_usd(), 3),
        }
    except Exception:
        usage_summary = None
    return web.json_response({
        "state": ctx.state.value,
        "user_present": ctx.user_present,
        "user_name": ctx.user_name,
        "distance_cm": perception.person_distance_cm if perception.person_distance_cm > 0 else None,
        "temperature_c": perception.temperature_c,
        "expression": face.expression.name,
        "stats": {
            "energy": round(s.energy, 1),
            "mood": round(s.mood, 1),
            "social": round(s.social, 1),
            "curiosity": round(s.curiosity, 1),
            "label": robot_stats.mood_label(),
        },
        "api_usage": usage_summary,
    })


async def api_conversation(request):
    minutes = float(request.query.get("minutes", "30"))
    limit = int(request.query.get("limit", "50"))
    rows = memory.recent_conversation(minutes=minutes, limit=limit)
    return web.json_response(rows)


async def api_snapshots(request):
    limit = int(request.query.get("limit", "20"))
    rows = memory.recent_snapshots(days=7.0, limit=limit)
    # photo_path → URL로
    for r in rows:
        path = Path(r["photo_path"])
        r["photo_url"] = f"/api/snapshot/{r['ts']:.3f}"  # ts로 라우팅
        del r["photo_path"]
    # ts → photo_path 매핑 캐시 (라우트에서 검색)
    return web.json_response(rows)


async def api_snapshot_file(request):
    ts_str = request.match_info.get("ts", "")
    try:
        ts = float(ts_str)
    except ValueError:
        return web.Response(status=400)
    # DB에서 ts ± 0.5초 안의 행 찾기
    with memory.db() as conn:
        row = conn.execute(
            "SELECT photo_path FROM face_snapshots WHERE ABS(ts - ?) < 1.0 LIMIT 1",
            (ts,),
        ).fetchone()
    if not row:
        return web.Response(status=404)
    p = Path(row["photo_path"])
    if not p.exists():
        return web.Response(status=404)
    return web.FileResponse(p, headers={"Cache-Control": "max-age=3600"})


async def api_face_preview(request):
    """현재 얼굴 상태를 JPG로 렌더해서 반환."""
    app = request.app
    face: FaceState = app["face"]
    # pygame Surface 한 장 만들기 (메인 LCD render와 별개)
    import pygame
    if not pygame.get_init():
        pygame.init()
        pygame.display.set_mode((1, 1))
    from src.config import DISPLAY_HEIGHT, DISPLAY_WIDTH
    canvas = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    draw_face_to_surface(canvas, face)
    raw = pygame.image.tostring(canvas, "RGB")
    from PIL import Image
    img = Image.frombytes("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), raw)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=78)
    buf.seek(0)
    return web.Response(
        body=buf.read(),
        content_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


async def sse_events(request):
    """Server-Sent Events — 5초마다 상태 push."""
    resp = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )
    await resp.prepare(request)
    app = request.app
    face: FaceState = app["face"]
    ctx: StateContext = app["ctx"]
    perception: PerceptionState = app["perception"]
    try:
        while True:
            s = robot_stats.get()
            payload = {
                "state": ctx.state.value,
                "user_present": ctx.user_present,
                "user_name": ctx.user_name,
                "expression": face.expression.name,
                "distance_cm": (
                    perception.person_distance_cm
                    if perception.person_distance_cm > 0 else None
                ),
                "temperature_c": perception.temperature_c,
                "stats": {
                    "energy": round(s.energy, 1),
                    "mood": round(s.mood, 1),
                    "social": round(s.social, 1),
                    "curiosity": round(s.curiosity, 1),
                    "label": robot_stats.mood_label(),
                },
                "ts": time.time(),
            }
            await resp.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode())
            await asyncio.sleep(2.0)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


# === API: 쓰기 (명령 큐) ===

async def api_speak(request):
    data = await request.json()
    text = (data.get("text") or "").strip()
    if not text:
        return web.json_response({"error": "text required"}, status=400)
    expr_name = data.get("expression")
    args = {"text": text}
    if expr_name:
        args["expression"] = expr_name
    cmd_id = memory.enqueue_command("speak", args)
    return web.json_response({"ok": True, "id": cmd_id})


async def api_expression(request):
    data = await request.json()
    name = data.get("name", "").upper()
    if not hasattr(exprs, name):
        return web.json_response({"error": "unknown expression"}, status=400)
    cmd_id = memory.enqueue_command("expression", {"name": name})
    return web.json_response({"ok": True, "id": cmd_id})


async def api_dance(request):
    data = await request.json()
    args = {}
    if "beats" in data:
        args["beats"] = int(data["beats"])
    if "bpm" in data:
        args["bpm"] = int(data["bpm"])
    cmd_id = memory.enqueue_command("dance", args)
    return web.json_response({"ok": True, "id": cmd_id})


async def api_register_face(request):
    data = await request.json()
    name = (data.get("name") or "").strip()
    if not name:
        return web.json_response({"error": "name required"}, status=400)
    app = request.app
    ctx: StateContext = app["ctx"]
    ctx.pending_register_name = name
    return web.json_response({
        "ok": True,
        "msg": f"다음 프레임에서 '{name}'으로 얼굴 등록 시도.",
    })


async def api_snapshot_now(request):
    """즉시 스냅샷 — vision_task의 debug_snapshot 트리거."""
    try:
        from src.vision import debug_snapshot
        debug_snapshot.request_snapshot("web_ui")
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def api_expressions_list(request):
    """사용 가능한 표정 이름 목록."""
    names = [
        n for n in dir(exprs)
        if not n.startswith("_") and n.isupper()
        and not n.endswith(("_PATH", "PATTERN"))
    ]
    # 실제 Expression 객체인지 확인
    filtered = []
    for n in names:
        try:
            obj = getattr(exprs, n)
            if hasattr(obj, "name"):
                filtered.append(n)
        except Exception:
            pass
    return web.json_response(sorted(filtered))


# === 앱 생성 + 실행 ===

def _create_app(face, ctx, perception):
    app = web.Application(middlewares=[auth_middleware])
    app["face"] = face
    app["ctx"] = ctx
    app["perception"] = perception

    aiohttp_jinja2.setup(
        app, loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
    )

    app.router.add_get("/", page_index)
    app.router.add_get("/login", page_login)
    app.router.add_post("/login", post_login)
    app.router.add_post("/logout", post_logout)

    app.router.add_get("/api/state", api_state)
    app.router.add_get("/api/conversation", api_conversation)
    app.router.add_get("/api/snapshots", api_snapshots)
    app.router.add_get(r"/api/snapshot/{ts}", api_snapshot_file)
    app.router.add_get("/api/face-preview.jpg", api_face_preview)
    app.router.add_get("/api/expressions", api_expressions_list)
    app.router.add_get("/events", sse_events)

    app.router.add_post("/api/speak", api_speak)
    app.router.add_post("/api/expression", api_expression)
    app.router.add_post("/api/dance", api_dance)
    app.router.add_post("/api/register-face", api_register_face)
    app.router.add_post("/api/snapshot", api_snapshot_now)

    app.router.add_static("/static/", path=str(STATIC_DIR))
    return app


async def run_web_server(face, ctx, perception) -> None:
    """asyncio bg task로 실행."""
    if not WEB_UI_PASSWORD:
        log.warning("WEB_UI_PASSWORD 비어있음 — Web UI 비활성화. "
                    ".env에 WEB_UI_PASSWORD 설정하세요.")
        return
    app = _create_app(face, ctx, perception)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", WEB_UI_PORT)
    try:
        await site.start()
        log.info(f"🌐 Web UI: http://0.0.0.0:{WEB_UI_PORT}")
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()
