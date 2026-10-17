import { FilesetResolver, PoseLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision';
import './style.css';
import './fit.css';
import './portrait.css';

type Mode = 'camera' | 'demo';
type Phase = 'ready' | 'calibrating' | 'countdown' | 'playing' | 'finished';

const ROUND_SECONDS = 15;
const MAX_HEIGHT_CM = 300;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const SCORE_KEY = 'high-jump-arena-scores-v2';

const personStanding = `<svg viewBox="0 0 80 100" aria-hidden="true"><circle cx="40" cy="16" r="10"/><path d="M29 31h22l5 34H45v29H35V65H24z"/></svg>`;
const personReady = `<svg viewBox="0 0 80 100" aria-hidden="true"><circle cx="49" cy="20" r="10"/><path d="m34 34 20 5-4 24-18-3-13 28-9-5 17-35zM48 60l18 25-8 7-23-28z"/></svg>`;
const personJump = `<svg viewBox="0 0 80 100" aria-hidden="true"><circle cx="43" cy="25" r="10"/><path d="m34 39 19 3-2 28-20-2zM48 43 64 8l8 4-13 38zM36 66 21 91l-8-5 15-28zM48 67l16 22-8 6-19-27z"/></svg>`;

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="app-shell">
    <header class="topbar">
      <div class="brand"><span class="brand-slash"></span><strong>ジャンプ！</strong><small>HIGH REACH CHALLENGE</small></div>
      <div class="top-actions">
        <span class="system-state"><i></i><b id="system-state">準備OK</b></span>
        <button id="settings-button" class="icon-button" type="button">設定</button>
        <button id="layout-button" class="icon-button" type="button" aria-pressed="false">縦画面</button>
        <button id="fullscreen-button" class="icon-button" type="button">全画面</button>
      </div>
    </header>
    <main class="game-layout">
      <section class="play-field" aria-label="ジャンプ到達点表示">
        <div class="paper-texture"></div>
        <div class="title-block"><span class="challenge-label">JUMP &amp; REACH</span><h1 id="main-title">手を上にのばそう</h1><p id="main-message">立つ位置を合わせたら、スタートを押してね</p></div>
        <div id="ruler" class="ruler" aria-label="高さの目盛り"></div>
        <div class="reach-line" id="reach-line"><span class="hand">✋</span><b id="reach-line-value">-- cm</b></div>
        <div class="steps" aria-label="遊び方">
          <div class="step"><span class="step-number">1</span>${personStanding}<b>立つ</b></div><span class="step-arrow">›</span>
          <div class="step"><span class="step-number">2</span>${personReady}<b>かまえる</b></div><span class="step-arrow">›</span>
          <div class="step"><span class="step-number">3</span>${personJump}<b>ジャンプ</b></div>
        </div>
        <div class="jump-zone"><span class="footprints">● ●</span><strong>ここに立つ</strong></div><div class="floor-line"></div>
        <div class="camera-preview" id="camera-preview"><div class="camera-label"><span>CAMERA</span><b id="camera-state">OFF</b></div><video id="camera-video" autoplay playsinline muted></video><canvas id="pose-canvas"></canvas><div class="camera-empty">カメラ映像</div></div>
      </section>
      <aside class="score-panel">
        <div class="phase-chip" id="phase-chip">READY</div>
        <section class="score-card"><span>今回の記録</span><div><strong id="current-score">---</strong><small>cm</small></div><p id="score-note">最高到達点を測ります</p></section>
        <section class="best-card"><span>最高記録</span><strong id="best-score">---</strong><small>cm</small></section>
        <section class="ranking-card"><header><span>ランキング</span><small>この端末</small></header><ol id="ranking"></ol></section>
        <section class="timer-card"><div class="timer-ring" id="timer-ring"><strong id="timer">${ROUND_SECONDS}</strong><small>秒</small></div><p id="jump-count">ジャンプ 0回</p></section>
        <button id="start-button" class="start-button" type="button"><span id="start-text">スタート</span><b>→</b></button>
        <button id="demo-jump-button" class="demo-jump-button" type="button" hidden>ジャンプ！</button>
      </aside>
    </main>
    <div class="settings-backdrop" id="settings-backdrop" hidden></div>
    <aside class="settings-drawer" id="settings-drawer" aria-hidden="true">
      <header><div><small>GAME SETUP</small><h2>プレイ設定</h2></div><button id="settings-close" type="button" aria-label="設定を閉じる">×</button></header>
      <section><h3>計測方法</h3><div class="mode-switch"><button class="mode active" data-mode="camera" type="button"><b>カメラ計測</b><small>実際にジャンプ</small></button><button class="mode" data-mode="demo" type="button"><b>デモ</b><small>動作確認用</small></button></div></section>
      <section><h3>からだの設定</h3><label>身長<span><input id="body-height" type="number" min="100" max="230" value="150" inputmode="numeric"> cm</span></label><label>立ったまま手を伸ばした高さ<span><input id="standing-reach" type="number" min="120" max="280" value="190" inputmode="numeric"> cm</span></label><p>床から指先までを測って入力すると、精度が上がります。</p></section>
      <section class="camera-help"><h3>カメラの置き方</h3><ul><li>頭から足まで映す</li><li>カメラを動かさない</li><li>壁と平行に置く</li></ul></section>
      <button id="clear-scores" class="clear-button" type="button">ランキングを消去</button>
    </aside>
    <div id="toast" class="toast" role="status" aria-live="polite"></div>
  </div>`;

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const video = $<HTMLVideoElement>('#camera-video');
const canvas = $<HTMLCanvasElement>('#pose-canvas');
const context = canvas.getContext('2d')!;
const state = {
  mode: 'camera' as Mode, phase: 'ready' as Phase, pose: null as PoseLandmarker | null, stream: null as MediaStream | null,
  startAt: 0, roundStartedAt: 0, calibration: [] as { foot: number; body: number; hand: number }[], baselineFoot: 0,
  bodyPixels: 0, reachPixels: 0, bestReach: 0, liveReach: 0, jumps: 0, airborne: false, lastVideoTime: -1,
  lastDetection: 0, demoJump: null as { started: number; height: number } | null, countdownTimer: 0,
};

function numeric(selector: string, fallback: number): number {
  const value = Number($<HTMLInputElement>(selector).value);
  if (!Number.isFinite(value)) return fallback;
  const [min, max] = selector === '#body-height' ? [100, 230] : [120, 280];
  return Math.max(min, Math.min(max, value));
}
function scores(): number[] { try { const saved = JSON.parse(localStorage.getItem(SCORE_KEY) ?? '[]'); return Array.isArray(saved) ? saved.filter(Number.isFinite).slice(0, 10) : []; } catch { return []; } }
function saveScore(value: number) { localStorage.setItem(SCORE_KEY, JSON.stringify([...scores(), Math.round(value)].sort((a, b) => b - a).slice(0, 10))); }
function renderRuler() { $('#ruler').innerHTML = Array.from({ length: 31 }, (_, index) => { const cm = index * 10; const major = cm % 50 === 0; return `<div class="ruler-mark ${major ? 'major' : ''}" style="bottom:${cm / MAX_HEIGHT_CM * 100}%">${major ? `<b>${cm}</b><small>cm</small>` : ''}</div>`; }).join(''); }
function renderRanking() {
  const list = scores().slice(0, 3);
  $('#ranking').innerHTML = [0, 1, 2].map((index) => `<li><span>${index + 1}</span><b>${list[index] ?? '---'}</b><small>cm</small></li>`).join('');
  $('#best-score').textContent = list[0] ? String(list[0]) : '---';
}
function setPhase(phase: Phase, title: string, message: string) {
  state.phase = phase;
  $('#phase-chip').textContent = phase === 'ready' ? 'READY' : phase === 'calibrating' ? 'CALIBRATION' : phase === 'countdown' ? 'GET READY' : phase === 'playing' ? 'JUMP!' : 'FINISH';
  $('#main-title').textContent = title; $('#main-message').textContent = message;
  $('#system-state').textContent = phase === 'playing' ? '計測中' : phase === 'calibrating' ? '調整中' : '準備OK';
  document.body.dataset.phase = phase;
}
function setReachLine(value: number, visible: boolean) {
  $('#reach-line').style.bottom = `${Math.max(0, Math.min(97, value / MAX_HEIGHT_CM * 100))}%`;
  $('#reach-line').classList.toggle('visible', visible); $('#reach-line-value').textContent = `${Math.round(value)} cm`;
}
function render() {
  const visible = state.bestReach > 0 || state.phase === 'calibrating' || state.phase === 'playing';
  const displayReach = state.phase === 'playing' && state.liveReach > 0 ? state.liveReach : state.bestReach;
  $('#current-score').textContent = state.bestReach > 0 ? String(Math.round(state.bestReach)) : '---';
  $('#jump-count').textContent = `ジャンプ ${state.jumps}回`; setReachLine(displayReach || numeric('#standing-reach', 190), visible);
  if (state.phase === 'playing') {
    const remaining = Math.max(0, Math.ceil(ROUND_SECONDS - (performance.now() - state.roundStartedAt) / 1000));
    $('#timer').textContent = String(remaining); $('#timer-ring').style.setProperty('--progress', `${remaining / ROUND_SECONDS * 360}deg`);
    if (remaining <= 0) finishRound();
  }
}
function validPoint(point?: NormalizedLandmark, visibility = 0.5): point is NormalizedLandmark { return !!point && point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1 && (point.visibility ?? 1) > visibility; }
function measure(points: NormalizedLandmark[]) {
  const nose = points[0]; const feet = [points[27], points[28]].filter((point) => validPoint(point, .55)); const hands = [points[15], points[16], points[19], points[20]].filter((point) => validPoint(point, .45));
  if (!validPoint(nose, .55) || !feet.length || !hands.length) return null;
  const foot = Math.max(...feet.map((point) => point.y)); const hand = Math.min(...hands.map((point) => point.y)); const body = (foot - nose.y) / .94;
  return body >= .28 && foot - hand >= .3 ? { foot, body, hand } : null;
}
function drawPose(points: NormalizedLandmark[]) {
  canvas.width = video.videoWidth; canvas.height = video.videoHeight; context.clearRect(0, 0, canvas.width, canvas.height); context.fillStyle = '#ff6b17'; context.strokeStyle = '#fff'; context.lineWidth = 3;
  for (const index of [0, 15, 16, 19, 20, 27, 28]) { const point = points[index]; if (!validPoint(point, .4)) continue; context.beginPath(); context.arc(point.x * canvas.width, point.y * canvas.height, index >= 15 && index <= 20 ? 7 : 5, 0, Math.PI * 2); context.fill(); context.stroke(); }
}
function processPose(points: NormalizedLandmark[]) {
  drawPose(points); const measurement = measure(points); if (!measurement) { $('#camera-state').textContent = '全身を映す'; return; } $('#camera-state').textContent = 'TRACKING';
  if (state.phase === 'calibrating') {
    state.calibration.push(measurement);
    if (state.calibration.length >= 35 && performance.now() - state.startAt >= 2600) {
      const median = (values: number[]) => values.sort((a, b) => a - b)[Math.floor(values.length / 2)];
      state.baselineFoot = median(state.calibration.map((item) => item.foot)); state.bodyPixels = median(state.calibration.map((item) => item.body)); state.reachPixels = median(state.calibration.map((item) => item.foot - item.hand)); beginCountdown();
    }
    return;
  }
  if (state.phase !== 'playing') return;
  const standingReach = numeric('#standing-reach', 190); const bodyHeight = numeric('#body-height', 150);
  const scale = standingReach / Math.max(.2, state.reachPixels) * .75 + bodyHeight / Math.max(.2, state.bodyPixels) * .25;
  const measuredReach = Math.max(0, Math.min(MAX_HEIGHT_CM + 20, (state.baselineFoot - measurement.hand) * scale));
  state.liveReach = state.liveReach ? state.liveReach * .62 + measuredReach * .38 : measuredReach;
  const jumpHeight = Math.max(0, (state.baselineFoot - measurement.foot) * bodyHeight / Math.max(.2, state.bodyPixels));
  if (jumpHeight > 5 && jumpHeight < 110) {
    if (!state.airborne) { state.airborne = true; state.jumps += 1; }
    if (state.liveReach > state.bestReach) { state.bestReach = state.liveReach; $('#score-note').textContent = '記録更新！'; }
  } else if (jumpHeight < 2.5) { state.airborne = false; $('#score-note').textContent = 'もっと高く！'; }
}
async function cameraLoop() {
  if (!state.stream) return; requestAnimationFrame(cameraLoop);
  if (state.phase === 'calibrating' && performance.now() - state.startAt > 14000) { resetAfterError('調整できませんでした', '全身と上げた手が映る位置で、もう一度試してください'); return; }
  if (!state.pose || video.readyState < 2) return; const now = performance.now(); if (now - state.lastDetection < 60 || video.currentTime === state.lastVideoTime) return; state.lastDetection = now; state.lastVideoTime = video.currentTime;
  try { const result = state.pose.detectForVideo(video, now); if (result.landmarks[0]) processPose(result.landmarks[0]); else { context.clearRect(0, 0, canvas.width, canvas.height); $('#camera-state').textContent = '人を探しています'; } }
  catch (error) { console.error(error); resetAfterError('計測エラー', 'ページを再読み込みして試してください'); }
}
function showToast(message: string) { $('#toast').textContent = message; $('#toast').classList.add('show'); window.setTimeout(() => $('#toast').classList.remove('show'), 3500); }
async function startCamera() {
  state.bestReach = 0; state.liveReach = 0; state.jumps = 0; state.airborne = false; $('#start-button').setAttribute('disabled', 'true'); setPhase('calibrating', '手を上げて、そのまま', '全身を映して3秒間止まってね'); $('#start-text').textContent = '調整中';
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('カメラを使うにはHTTPSまたはlocalhostで開いてください');
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } }); video.srcObject = state.stream; await video.play(); $('#camera-preview').classList.add('on');
    if (!state.pose) { const vision = await FilesetResolver.forVisionTasks(WASM_URL); try { state.pose = await PoseLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' }, runningMode: 'VIDEO', numPoses: 1 }); } catch { state.pose = await PoseLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' }, runningMode: 'VIDEO', numPoses: 1 }); } }
    state.calibration = []; state.startAt = performance.now(); requestAnimationFrame(cameraLoop);
  } catch (error) { resetAfterError('カメラを開始できません', error instanceof Error ? error.message : 'カメラの許可を確認してください'); }
}
function beginCountdown() {
  $('#start-button').setAttribute('disabled', 'true');
  setPhase('countdown', 'ジャンプの準備！', 'カウントが0になったらスタート'); let count = 3; $('#timer').textContent = String(count); $('#timer-ring').style.setProperty('--progress', '360deg'); window.clearInterval(state.countdownTimer);
  state.countdownTimer = window.setInterval(() => { count -= 1; $('#timer').textContent = count > 0 ? String(count) : 'GO'; if (count <= 0) { window.clearInterval(state.countdownTimer); window.setTimeout(beginRound, 450); } }, 800);
}
function beginRound() { state.roundStartedAt = performance.now(); $('#start-text').textContent = 'プレイ中'; $('#score-note').textContent = 'もっと高く！'; $('#demo-jump-button').hidden = state.mode !== 'demo'; setPhase('playing', '手を上にのばそう', '15秒間、何度でもジャンプ！'); render(); }
function startDemo() { state.bestReach = 0; state.liveReach = numeric('#standing-reach', 190); state.jumps = 0; state.airborne = false; beginCountdown(); }
function demoJump() {
  if (state.mode !== 'demo' || state.phase !== 'playing' || state.demoJump) return; state.jumps += 1; state.demoJump = { started: performance.now(), height: 25 + Math.random() * 35 };
  const animate = () => { if (!state.demoJump || state.phase !== 'playing') return; const progress = Math.min(1, (performance.now() - state.demoJump.started) / 720); state.liveReach = numeric('#standing-reach', 190) + state.demoJump.height * Math.sin(progress * Math.PI); state.bestReach = Math.max(state.bestReach, state.liveReach); $('#score-note').textContent = state.liveReach >= state.bestReach - 1 ? '記録更新！' : 'もっと高く！'; if (progress < 1) requestAnimationFrame(animate); else { state.liveReach = numeric('#standing-reach', 190); state.demoJump = null; } }; requestAnimationFrame(animate);
}
function stopCamera() { state.stream?.getTracks().forEach((track) => track.stop()); state.stream = null; video.srcObject = null; $('#camera-preview').classList.remove('on'); $('#camera-state').textContent = 'OFF'; }
function resetAfterError(title: string, message: string) { stopCamera(); window.clearInterval(state.countdownTimer); setPhase('ready', title, message); $('#start-button').removeAttribute('disabled'); $('#start-text').textContent = 'もう一度'; showToast(message); }
function finishRound() { if (state.phase !== 'playing') return; setPhase('finished', 'チャレンジ終了！', state.bestReach > 0 ? `記録は ${Math.round(state.bestReach)} cm` : 'もう一度チャレンジしよう'); stopCamera(); $('#timer').textContent = '0'; $('#score-note').textContent = state.bestReach > 0 ? 'ナイスジャンプ！' : 'もう一度挑戦しよう'; $('#demo-jump-button').hidden = true; $('#start-button').removeAttribute('disabled'); $('#start-text').textContent = 'もう一度'; if (state.bestReach > 0) saveScore(state.bestReach); renderRanking(); }
function openSettings(open: boolean) { $('#settings-drawer').classList.toggle('open', open); $('#settings-drawer').setAttribute('aria-hidden', String(!open)); $('#settings-backdrop').hidden = !open; }
function setLayout(portrait: boolean) {
  $('.app-shell').classList.toggle('portrait', portrait);
  document.body.classList.toggle('portrait-mode', portrait);
  $('#layout-button').textContent = portrait ? '横画面' : '縦画面';
  $('#layout-button').setAttribute('aria-pressed', String(portrait));
  localStorage.setItem('high-jump-layout-v1', portrait ? 'portrait' : 'landscape');
}

document.querySelectorAll<HTMLButtonElement>('.mode').forEach((button) => button.addEventListener('click', () => { if (['playing', 'calibrating', 'countdown'].includes(state.phase)) return; state.mode = button.dataset.mode as Mode; document.querySelectorAll('.mode').forEach((element) => element.classList.toggle('active', element === button)); $('#camera-preview').classList.toggle('demo', state.mode === 'demo'); setPhase('ready', '手を上にのばそう', state.mode === 'camera' ? '立つ位置を合わせたら、スタートを押してね' : 'スタート後、ジャンプボタンかスペースキーを押してね'); $('#start-text').textContent = 'スタート'; }));
$('#start-button').addEventListener('click', () => { if (state.phase === 'playing') return; openSettings(false); if (state.mode === 'camera') void startCamera(); else startDemo(); });
$('#demo-jump-button').addEventListener('click', demoJump);
document.addEventListener('keydown', (event) => { if (event.code === 'Space' && state.mode === 'demo' && state.phase === 'playing' && !(event.target instanceof HTMLInputElement)) { event.preventDefault(); demoJump(); } });
$('#settings-button').addEventListener('click', () => openSettings(true)); $('#settings-close').addEventListener('click', () => openSettings(false)); $('#settings-backdrop').addEventListener('click', () => openSettings(false));
$('#layout-button').addEventListener('click', () => setLayout(!$('.app-shell').classList.contains('portrait')));
$('#fullscreen-button').addEventListener('click', async () => { try { if (!document.fullscreenElement) await document.documentElement.requestFullscreen(); else await document.exitFullscreen(); } catch { showToast('全画面表示を開始できませんでした'); } });
$('#clear-scores').addEventListener('click', () => { localStorage.removeItem(SCORE_KEY); renderRanking(); showToast('ランキングを消去しました'); });
window.addEventListener('beforeunload', stopCamera);
setLayout(localStorage.getItem('high-jump-layout-v1') === 'portrait');
renderRuler(); renderRanking(); setPhase('ready', '手を上にのばそう', '立つ位置を合わせたら、スタートを押してね'); setInterval(render, 80);
