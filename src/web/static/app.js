// Roboface Web UI — vanilla JS
'use strict';

const $ = (id) => document.getElementById(id);

// ─── 상태 표시 ───
function applyState(d) {
  $('s-state').textContent = d.state || '—';
  $('s-expr').textContent = d.expression || '—';
  $('s-user').textContent = d.user_name || (d.user_present ? '있음 (미등록)' : '없음');
  $('s-dist').textContent = d.distance_cm != null ? `${Math.round(d.distance_cm)}cm` : '—';
  $('s-temp').textContent = d.temperature_c != null ? `${d.temperature_c.toFixed(1)}°C` : '—';

  const s = d.stats || {};
  $('s-label').textContent = s.label || '—';
  for (const k of ['energy', 'mood', 'social', 'curiosity']) {
    const v = s[k] ?? 0;
    const bar = $(`s-${k === 'curiosity' ? 'curious' : k}`);
    const val = $(`s-${k === 'curiosity' ? 'curious' : k}-val`);
    if (bar) bar.style.width = `${v}%`;
    if (val) val.textContent = Math.round(v);
  }
}

async function fetchState() {
  try {
    const r = await fetch('/api/state');
    if (!r.ok) return;
    applyState(await r.json());
  } catch (e) { /* network */ }
}

// ─── face preview 주기 갱신 ───
function refreshFace() {
  const img = $('face-img');
  if (img) img.src = `/api/face-preview.jpg?t=${Date.now()}`;
}

// ─── 대화 로그 ───
async function loadConversation() {
  try {
    const r = await fetch('/api/conversation?minutes=60&limit=80');
    if (!r.ok) return;
    const rows = await r.json();
    const list = $('conv-list');
    list.innerHTML = '';
    // 최근이 아래로
    for (const row of rows) {
      const d = new Date(row.ts * 1000);
      const ts = `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
      const div = document.createElement('div');
      div.className = `conv-row ${row.speaker}`;
      div.innerHTML = `
        <span class="ts">${ts}</span>
        <span class="who">${row.speaker === 'robot' ? '🗣️' : '👤'}</span>
        <span class="text"></span>
      `;
      div.querySelector('.text').textContent = row.text;
      list.appendChild(div);
    }
    list.scrollTop = list.scrollHeight;
  } catch (e) { /* network */ }
}

// ─── 사진 ───
async function loadSnapshots() {
  try {
    const r = await fetch('/api/snapshots?limit=24');
    if (!r.ok) return;
    const rows = await r.json();
    const grid = $('snap-grid');
    grid.innerHTML = '';
    if (rows.length === 0) {
      grid.innerHTML = '<p style="color:var(--dim);font-size:13px">사진 없음.</p>';
      return;
    }
    for (const row of rows) {
      const d = new Date(row.ts * 1000);
      const time = `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
      const cell = document.createElement('div');
      cell.className = 'snap-cell';
      const meta = `${time} · ${row.emotion || '?'}`;
      cell.innerHTML = `
        <img src="${row.photo_url}" loading="lazy" alt="">
        <div class="meta"></div>
      `;
      cell.querySelector('.meta').textContent = meta;
      grid.appendChild(cell);
    }
  } catch (e) { /* network */ }
}

// ─── 표정 목록 ───
async function loadExpressions() {
  try {
    const r = await fetch('/api/expressions');
    if (!r.ok) return;
    const names = await r.json();
    const sel = $('cmd-expr-select');
    sel.innerHTML = names.map(n => `<option value="${n}">${n}</option>`).join('');
  } catch (e) { /* network */ }
}

// ─── 명령 ───
function showResult(msg, ok = true) {
  const el = $('cmd-result');
  el.textContent = msg;
  el.className = 'result ' + (ok ? 'ok' : 'err');
  setTimeout(() => { el.textContent = ''; el.className = 'result'; }, 3500);
}

async function post(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

$('cmd-speak-btn').addEventListener('click', async () => {
  const text = $('cmd-speak-text').value.trim();
  if (!text) return;
  try {
    await post('/api/speak', { text });
    $('cmd-speak-text').value = '';
    showResult(`발화: "${text}"`);
    setTimeout(loadConversation, 1500);
  } catch (e) { showResult(e.message, false); }
});

$('cmd-speak-text').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') $('cmd-speak-btn').click();
});

$('cmd-expr-btn').addEventListener('click', async () => {
  const name = $('cmd-expr-select').value;
  try {
    await post('/api/expression', { name });
    showResult(`표정: ${name}`);
    setTimeout(refreshFace, 500);
  } catch (e) { showResult(e.message, false); }
});

$('cmd-dance-btn').addEventListener('click', async () => {
  try {
    await post('/api/dance', { beats: 4, bpm: 110 });
    showResult('댄스 큐잉됨');
  } catch (e) { showResult(e.message, false); }
});

$('cmd-snapshot-btn').addEventListener('click', async () => {
  try {
    await post('/api/snapshot', {});
    showResult('스냅샷 요청 — 다음 프레임에서 캡처');
    setTimeout(loadSnapshots, 2500);
  } catch (e) { showResult(e.message, false); }
});

$('cmd-face-btn').addEventListener('click', async () => {
  const name = $('cmd-face-name').value.trim();
  if (!name) { showResult('이름 입력', false); return; }
  try {
    const r = await post('/api/register-face', { name });
    showResult(r.msg || '등록 요청');
    $('cmd-face-name').value = '';
  } catch (e) { showResult(e.message, false); }
});

$('cmd-update-btn').addEventListener('click', async () => {
  if (!confirm('git pull 받고 변경 있으면 봇 재시작. 진행?')) return;
  const btn = $('cmd-update-btn');
  btn.disabled = true;
  const orig = btn.textContent;
  btn.textContent = '⟳ 업데이트 중...';
  try {
    const r = await post('/api/update', {});
    showResult(r.msg || '업데이트 트리거됨');
  } catch (e) {
    showResult(e.message, false);
  } finally {
    setTimeout(() => {
      btn.disabled = false;
      btn.textContent = orig;
    }, 5000);
  }
});

$('refresh-conv').addEventListener('click', loadConversation);
$('refresh-snaps').addEventListener('click', loadSnapshots);

// ─── SSE 실시간 업데이트 ───
function connectSSE() {
  const es = new EventSource('/events');
  es.onmessage = (ev) => {
    try { applyState(JSON.parse(ev.data)); } catch (e) {}
  };
  es.onerror = () => { es.close(); setTimeout(connectSSE, 3000); };
}

// ─── 초기 + 주기 ───
fetchState();
loadConversation();
loadSnapshots();
loadExpressions();
connectSSE();

setInterval(refreshFace, 1500);   // 얼굴 1.5초마다
setInterval(loadConversation, 15000);   // 대화 15초마다
setInterval(loadSnapshots, 60000);       // 사진 1분마다
