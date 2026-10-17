import { FilesetResolver, PoseLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision';
import './style.css';
import './portrait.css';
import './light.css';

type Mode = 'camera' | 'demo';
type Phase = 'ready' | 'calibrating' | 'countdown' | 'playing' | 'finished';

const ROUND_SECONDS = 15;
const MAX_HEIGHT_CM = 300;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const SCORE_KEY = 'high-jump-arena-scores-v2';
const LANDMARKS = [
  { cm: 200, label: 'ドアの高さ' },
  { cm: 224, label: 'バレーネット 女子' },
  { cm: 243, label: 'バレーネット 男子' },
  { cm: 280, label: 'すごい！' },
];

const handIcon = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 5 11h4.5v10h5V11H19z"/></svg>`;

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="app-shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark">${handIcon}</span><div><strong>ジャンプ！</strong><small>HIGH REACH CHALLENGE</small></div></div>
      <div class="top-actions">
        <span class="system-state"><i></i><b id="system-state">準備OK</b></span>
        <button id="settings-button" class="icon-button" type="button">設定</button>
        <button id="theme-button" class="icon-button" type="button" aria-pressed="false">ライト</button>
        <button id="layout-button" class="icon-button" type="button" aria-pressed="false">縦画面</button>
        <button id="fullscreen-button" class="icon-button" type="button">全画面</button>
      </div>
    </header>
    <main class="game-layout">
      <section class="play-field" aria-label="ジャンプ到達点表示">
        <div class="field-glow"></div>
        <div class="scale" id="scale">
          <div id="ruler" class="ruler" aria-label="高さの目盛り"></div>
          <div id="landmarks" class="landmarks"></div>
          <div class="mark-line best-line" id="best-line"><span>BEST</span><b id="best-line-value">--</b></div>
          <div class="mark-line peak-line" id="peak-line"><span>MAX</span><b id="peak-line-value">--</b></div>
          <div class="reach-line" id="reach-line"><span class="hand">${handIcon}</span><b id="reach-line-value">-- cm</b></div>
          <div class="floor-line"></div>
        </div>
        <div class="title-block"><span class="challenge-label">JUMP &amp; REACH</span><h1 id="main-title">手を上にのばそう</h1><p id="main-message">立つ位置を合わせたら、スタートを押してね</p></div>
        <div class="jump-zone"><span class="footprints"><i></i><i></i></span><strong>ここに立つ</strong></div>
        <div class="countdown" id="countdown" aria-hidden="true"><b id="countdown-number">3</b></div>
        <div class="result" id="result" aria-live="polite">
          <span class="result-label" id="result-label">RESULT</span>
          <div class="result-score"><strong id="result-score">---</strong><small>cm</small></div>
          <p id="result-message">ナイスジャンプ！</p>
        </div>
        <div class="camera-preview" id="camera-preview"><div class="camera-label"><span>CAMERA</span><b id="camera-state">OFF</b></div><video id="camera-video" autoplay playsinline muted></video><canvas id="pose-canvas"></canvas><div class="camera-empty">カメラ映像</div></div>
      </section>
      <aside class="score-panel">
        <div class="panel-head"><div class="phase-chip" id="phase-chip">READY</div><section class="timer-card"><p id="jump-count">ジャンプ 0回</p><div class="timer-ring" id="timer-ring"><strong id="timer">${ROUND_SECONDS}</strong><small>秒</small></div></section></div>
        <section class="score-card" id="score-card"><span>今回の記録</span><div class="score-value"><strong id="current-score">---</strong><small>cm</small></div><div class="meter"><i id="score-meter"></i><em id="meter-best"></em></div><p id="score-note">最高到達点を測ります</p></section>
        <section class="best-card"><span>この端末の最高記録</span><div><strong id="best-score">---</strong><small>cm</small></div></section>
        <section class="ranking-card"><header><span>ランキング</span><small>TOP 5</small></header><ol id="ranking"></ol></section>
        <div class="panel-actions">
          <button id="start-button" class="start-button" type="button"><span id="start-text">スタート</span><b>▶</b></button>
          <button id="demo-jump-button" class="demo-jump-button" type="button" hidden>ジャンプ！<small>SPACE</small></button>
        </div>
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
  lastDetection: 0, demoJump: null as { started: number; height: number } | null, countdownTimer: 0, shownBest: 0,
};

function numeric(selector: string, fallback: number): number {
  const value = Number($<HTMLInputElement>(selector).value);
  if (!Number.isFinite(value)) return fallback;
  const [min, max] = selector === '#body-height' ? [100, 230] : [120, 280];
  return Math.max(min, Math.min(max, value));
}
function scores(): number[] { try { const saved = JSON.parse(localStorage.getItem(SCORE_KEY) ?? '[]'); return Array.isArray(saved) ? saved.filter(Number.isFinite).slice(0, 10) : []; } catch { return []; } }
function saveScore(value: number) { localStorage.setItem(SCORE_KEY, JSON.stringify([...scores(), Math.round(value)].sort((a, b) => b - a).slice(0, 10))); }
const heightPercent = (cm: number) => `${Math.max(0, Math.min(100, cm / MAX_HEIGHT_CM * 100))}%`;
function renderRuler() {
  $('#ruler').innerHTML = Array.from({ length: 31 }, (_, index) => { const cm = index * 10; const major = cm % 50 === 0; return `<div class="ruler-mark ${major ? 'major' : ''}" style="bottom:${heightPercent(cm)}">${major ? `<b>${cm}</b>` : ''}</div>`; }).join('');
  $('#landmarks').innerHTML = LANDMARKS.map(({ cm, label }) => `<div class="landmark" style="bottom:${heightPercent(cm)}"><span>${label}</span><b>${cm}</b></div>`).join('');
}
function renderRanking() {
  const list = scores().slice(0, 5);
  $('#ranking').innerHTML = [0, 1, 2, 3, 4].map((index) => `<li class="${list[index] ? '' : 'empty'}"><span>${index + 1}</span><i style="--w:${list[index] ? list[index] / MAX_HEIGHT_CM * 100 : 0}%"></i><b>${list[index] ?? '---'}</b><small>cm</small></li>`).join('');
  $('#best-score').textContent = list[0] ? String(list[0]) : '---';
  $('#best-line').style.bottom = heightPercent(list[0] ?? 0); $('#best-line').classList.toggle('visible', !!list[0]); $('#best-line-value').textContent = list[0] ? `${list[0]}` : '--';
  $('#meter-best').style.left = heightPercent(list[0] ?? 0); $('#meter-best').hidden = !list[0];
}
function setPhase(phase: Phase, title: string, message: string) {
  state.phase = phase;
  $('#phase-chip').textContent = phase === 'ready' ? 'READY' : phase === 'calibrating' ? 'CALIBRATION' : phase === 'countdown' ? 'GET READY' : phase === 'playing' ? 'JUMP!' : 'FINISH';
  $('#main-title').textContent = title; $('#main-message').textContent = message;
  $('#system-state').textContent = phase === 'playing' ? '計測中' : phase === 'calibrating' ? '調整中' : '準備OK';
  document.body.dataset.phase = phase;
}
function setReachLine(value: number, visible: boolean) {
  $('#reach-line').style.bottom = heightPercent(Math.min(value, MAX_HEIGHT_CM * .985));
  $('#reach-line').classList.toggle('visible', visible); $('#reach-line-value').textContent = `${Math.round(value)} cm`;
}
function pop(selector: string) { const element = $(selector); element.classList.remove('pop'); void element.offsetWidth; element.classList.add('pop'); }
function render() {
  const visible = state.bestReach > 0 || state.phase === 'calibrating' || state.phase === 'playing';
  const displayReach = state.phase === 'playing' && state.liveReach > 0 ? state.liveReach : state.bestReach;
  $('#current-score').textContent = state.bestReach > 0 ? String(Math.round(state.bestReach)) : '---';
  if (Math.round(state.bestReach) > state.shownBest + 1 && state.phase === 'playing') pop('#score-card');
  state.shownBest = Math.round(state.bestReach);
  $('#score-meter').style.width = heightPercent(state.bestReach);
  $('#peak-line').style.bottom = heightPercent(state.bestReach); $('#peak-line').classList.toggle('visible', state.bestReach > 0 && (state.phase === 'playing' || state.phase === 'finished')); $('#peak-line-value').textContent = String(Math.round(state.bestReach));
  $('#jump-count').innerHTML = `ジャンプ <b>${state.jumps}</b>回`; setReachLine(displayReach || numeric('#standing-reach', 190), visible);
  if (state.phase === 'playing') {
    const remaining = Math.max(0, Math.ceil(ROUND_SECONDS - (performance.now() - state.roundStartedAt) / 1000));
    $('#timer').textContent = String(remaining); $('#timer-ring').style.setProperty('--progress', `${remaining / ROUND_SECONDS * 360}deg`);
    $('#timer-ring').classList.toggle('hurry', remaining <= 5);
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
  canvas.width = video.videoWidth; canvas.height = video.videoHeight; context.clearRect(0, 0, canvas.width, canvas.height); context.fillStyle = '#d4ff3a'; context.strokeStyle = '#070b1a'; context.lineWidth = 3;
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
function resetRound() { state.bestReach = 0; state.shownBest = 0; state.jumps = 0; state.airborne = false; $('#score-note').textContent = '最高到達点を測ります'; $('#timer').textContent = String(ROUND_SECONDS); $('#timer-ring').style.setProperty('--progress', '360deg'); $('#timer-ring').classList.remove('hurry'); }
async function startCamera() {
  resetRound(); state.liveReach = 0; $('#start-button').setAttribute('disabled', 'true'); setPhase('calibrating', '手を上げて、そのまま', '全身を映して3秒間止まってね'); $('#start-text').textContent = '調整中';
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('カメラを使うにはHTTPSまたはlocalhostで開いてください');
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } }); video.srcObject = state.stream; await video.play(); $('#camera-preview').classList.add('on');
    if (!state.pose) { const vision = await FilesetResolver.forVisionTasks(WASM_URL); try { state.pose = await PoseLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' }, runningMode: 'VIDEO', numPoses: 1 }); } catch { state.pose = await PoseLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' }, runningMode: 'VIDEO', numPoses: 1 }); } }
    state.calibration = []; state.startAt = performance.now(); requestAnimationFrame(cameraLoop);
  } catch (error) { resetAfterError('カメラを開始できません', error instanceof Error ? error.message : 'カメラの許可を確認してください'); }
}
function showCount(text: string) { $('#countdown-number').textContent = text; pop('#countdown-number'); }
function beginCountdown() {
  $('#start-button').setAttribute('disabled', 'true');
  setPhase('countdown', 'ジャンプの準備！', 'カウントが0になったらスタート'); let count = 3; $('#timer').textContent = String(ROUND_SECONDS); $('#timer-ring').style.setProperty('--progress', '360deg'); showCount(String(count)); window.clearInterval(state.countdownTimer);
  state.countdownTimer = window.setInterval(() => { count -= 1; showCount(count > 0 ? String(count) : 'GO!'); if (count <= 0) { window.clearInterval(state.countdownTimer); window.setTimeout(beginRound, 550); } }, 800);
}
function beginRound() { state.roundStartedAt = performance.now(); $('#start-text').textContent = 'プレイ中'; $('#score-note').textContent = 'もっと高く！'; $('#demo-jump-button').hidden = state.mode !== 'demo'; setPhase('playing', '手を上にのばそう', '15秒間、何度でもジャンプ！'); render(); }
function startDemo() { resetRound(); state.liveReach = numeric('#standing-reach', 190); beginCountdown(); }
function demoJump() {
  if (state.mode !== 'demo' || state.phase !== 'playing' || state.demoJump) return; state.jumps += 1; state.demoJump = { started: performance.now(), height: 25 + Math.random() * 35 };
  const animate = () => { if (!state.demoJump || state.phase !== 'playing') return; const progress = Math.min(1, (performance.now() - state.demoJump.started) / 720); state.liveReach = numeric('#standing-reach', 190) + state.demoJump.height * Math.sin(progress * Math.PI); state.bestReach = Math.max(state.bestReach, state.liveReach); $('#score-note').textContent = state.liveReach >= state.bestReach - 1 ? '記録更新！' : 'もっと高く！'; if (progress < 1) requestAnimationFrame(animate); else { state.liveReach = numeric('#standing-reach', 190); state.demoJump = null; } }; requestAnimationFrame(animate);
}
function stopCamera() { state.stream?.getTracks().forEach((track) => track.stop()); state.stream = null; video.srcObject = null; $('#camera-preview').classList.remove('on'); $('#camera-state').textContent = 'OFF'; }
function resetAfterError(title: string, message: string) { stopCamera(); window.clearInterval(state.countdownTimer); setPhase('ready', title, message); $('#start-button').removeAttribute('disabled'); $('#start-text').textContent = 'もう一度'; showToast(message); }
function showResult(score: number) {
  const previousBest = scores()[0] ?? 0;
  if (score > 0) saveScore(score);
  const rank = score > 0 ? scores().indexOf(score) + 1 : 0;
  const isRecord = score > 0 && score > previousBest;
  $('#result').classList.toggle('record', isRecord);
  $('#result-label').textContent = isRecord ? 'NEW RECORD!' : 'RESULT';
  $('#result-score').textContent = score > 0 ? String(score) : '---';
  $('#result-message').textContent = score <= 0 ? 'もう一度チャレンジしよう' : isRecord ? 'この端末の最高記録を更新！' : rank > 0 && rank <= 5 ? `ランキング ${rank}位！` : 'ナイスジャンプ！';
}
function finishRound() { if (state.phase !== 'playing') return; const score = Math.round(state.bestReach); setPhase('finished', 'チャレンジ終了！', score > 0 ? `記録は ${score} cm` : 'もう一度チャレンジしよう'); stopCamera(); $('#timer').textContent = '0'; $('#timer-ring').classList.remove('hurry'); $('#score-note').textContent = score > 0 ? 'ナイスジャンプ！' : 'もう一度挑戦しよう'; $('#demo-jump-button').hidden = true; $('#start-button').removeAttribute('disabled'); $('#start-text').textContent = 'もう一度'; showResult(score); renderRanking(); }
function openSettings(open: boolean) { $('#settings-drawer').classList.toggle('open', open); $('#settings-drawer').setAttribute('aria-hidden', String(!open)); $('#settings-backdrop').hidden = !open; }
function setLayout(portrait: boolean) {
  $('.app-shell').classList.toggle('portrait', portrait);
  document.body.classList.toggle('portrait-mode', portrait);
  $('#layout-button').textContent = portrait ? '横画面' : '縦画面';
  $('#layout-button').setAttribute('aria-pressed', String(portrait));
  localStorage.setItem('high-jump-layout-v1', portrait ? 'portrait' : 'landscape');
}

function setTheme(light: boolean) {
  document.body.classList.toggle('theme-light', light);
  $('#theme-button').textContent = light ? 'ダーク' : 'ライト';
  $('#theme-button').setAttribute('aria-pressed', String(light));
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', light ? '#ffffff' : '#060918');
}

document.querySelectorAll<HTMLButtonElement>('.mode').forEach((button) => button.addEventListener('click', () => { if (['playing', 'calibrating', 'countdown'].includes(state.phase)) return; state.mode = button.dataset.mode as Mode; document.querySelectorAll('.mode').forEach((element) => element.classList.toggle('active', element === button)); $('#camera-preview').classList.toggle('demo', state.mode === 'demo'); setPhase('ready', '手を上にのばそう', state.mode === 'camera' ? '立つ位置を合わせたら、スタートを押してね' : 'スタート後、ジャンプボタンかスペースキーを押してね'); $('#start-text').textContent = 'スタート'; }));
$('#start-button').addEventListener('click', () => { if (state.phase === 'playing') return; openSettings(false); if (state.mode === 'camera') void startCamera(); else startDemo(); });
$('#demo-jump-button').addEventListener('click', demoJump);
$('#result').addEventListener('click', () => { if (state.phase === 'finished') setPhase('ready', '手を上にのばそう', '次の人、どうぞ！'); });
document.addEventListener('keydown', (event) => { if (event.code === 'Space' && state.mode === 'demo' && state.phase === 'playing' && !(event.target instanceof HTMLInputElement)) { event.preventDefault(); demoJump(); } });
$('#settings-button').addEventListener('click', () => openSettings(true)); $('#settings-close').addEventListener('click', () => openSettings(false)); $('#settings-backdrop').addEventListener('click', () => openSettings(false));
$('#theme-button').addEventListener('click', () => { const light = !document.body.classList.contains('theme-light'); setTheme(light); localStorage.setItem('high-jump-theme-v2', light ? 'light' : 'dark'); });
$('#layout-button').addEventListener('click', () => setLayout(!$('.app-shell').classList.contains('portrait')));
$('#fullscreen-button').addEventListener('click', async () => { try { if (!document.fullscreenElement) await document.documentElement.requestFullscreen(); else await document.exitFullscreen(); } catch { showToast('全画面表示を開始できませんでした'); } });
$('#clear-scores').addEventListener('click', () => { localStorage.removeItem(SCORE_KEY); renderRanking(); showToast('ランキングを消去しました'); });
window.addEventListener('beforeunload', stopCamera);
setLayout(localStorage.getItem('high-jump-layout-v1') === 'portrait');
setTheme(localStorage.getItem('high-jump-theme-v2') !== 'dark');
renderRuler(); renderRanking(); setPhase('ready', '手を上にのばそう', '立つ位置を合わせたら、スタートを押してね'); setInterval(render, 80);
