import { FilesetResolver, PoseLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision';
import './style.css';

type Mode = 'camera' | 'demo';
type Phase = 'ready' | 'calibrating' | 'playing' | 'finished';

const ROUND_SECONDS = 20;
const CEILING_CM = 360;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const targets = [
  { cm: 250, name: 'ROOKIE', color: 'orange' },
  { cm: 300, name: 'ACE', color: 'cyan' },
  { cm: 334, name: 'LEGEND', color: 'lime' },
];

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <header class="topbar">
    <div class="brand"><span class="brand-mark">↗</span><span>HIGH JUMP<br><strong>ARENA</strong></span></div>
    <div class="top-right"><span class="live-dot"></span> JUMP REACH CHALLENGE <span class="top-divider"></span> 20 SEC ROUND</div>
  </header>
  <main class="layout">
    <section class="arena" aria-label="ジャンプ到達点表示">
      <div class="arena-grid"></div>
      <div class="arena-head"><div><span class="eyebrow">VERTICAL REACH // 001</span><h1>どこまで、<br><em>跳べる？</em></h1></div><div class="round-badge"><span>ROUND</span><strong id="timer">20</strong><small>SEC</small></div></div>
      <div class="height-stage" id="height-stage">
        <div class="height-top">MAX HEIGHT <strong>360 <small>CM</small></strong></div>
        <div class="ruler" id="ruler"></div>
        <div class="target-lines" id="target-lines"></div>
        <div class="player-line" id="player-line"><span class="player-line-tag">YOUR BEST <strong id="line-height">--</strong> cm</span></div>
        <div class="figure" aria-hidden="true"><div class="figure-head"></div><div class="figure-body"></div><div class="figure-arm left"></div><div class="figure-arm right"></div><div class="figure-leg left"></div><div class="figure-leg right"></div></div>
        <div class="floor"><span>00</span><span>JUMP ZONE</span><span>00</span></div>
      </div>
      <div class="arena-footer"><span>▲ STAND IN THE ZONE</span><span id="arena-status">READY TO JUMP</span></div>
    </section>

    <aside class="dashboard">
      <div class="panel-heading"><span>PLAYER CONSOLE</span><span class="panel-index">01 / 03</span></div>
      <div class="score-card"><div class="score-label">BEST REACH <span class="score-icon">↗</span></div><div class="score-number"><strong id="best-reach">--</strong><span>cm</span></div><div class="score-foot"><span id="result-label">挑戦を始めよう</span><span id="jump-count">0 JUMPS</span></div></div>
      <div class="mode-card"><div class="section-title">PLAY MODE <span>01</span></div><div class="mode-switch"><button class="mode active" data-mode="camera" type="button">◉ カメラ計測</button><button class="mode" data-mode="demo" type="button">◈ デモ</button></div><p id="mode-description">全身が映る位置にカメラを置き、実際にジャンプします。</p></div>
      <div class="camera-card"><div class="camera-head"><span>● LIVE TRACKING</span><span id="camera-state">OFFLINE</span></div><div class="camera-box" id="camera-box"><video id="camera-video" autoplay playsinline muted></video><canvas id="pose-canvas"></canvas><div class="camera-placeholder" id="camera-placeholder"><div class="scan-icon">⌗</div><strong>CAMERA PREVIEW</strong><span>開始すると映像を表示します</span></div><div class="camera-corners"></div></div><div class="camera-note" id="camera-note">カメラ映像は端末内で解析します</div></div>
      <div class="settings-card"><div class="section-title">PLAYER SETUP <span>02</span></div><div class="fields"><label>身長 <div class="number-input"><input id="body-height" type="number" min="100" max="230" value="165" inputmode="numeric"><span>cm</span></div></label><label>立位リーチ <div class="number-input"><input id="standing-reach" type="number" min="120" max="320" value="210" inputmode="numeric"><span>cm</span></div></label></div><p>立位リーチ＝床に立って腕を伸ばした指先の高さ。実測値を入力すると精度が上がります。</p></div>
      <div class="action-area"><button id="start-button" class="start-button" type="button"><span id="start-text">チャレンジ開始</span><span>↗</span></button><p id="helper-text">カメラの使用許可が必要です</p></div>
    </aside>
  </main>
  <section class="bottom-strip"><div><span class="strip-number">01</span><strong>セットアップ</strong><small>身長と立位リーチを入力</small></div><div><span class="strip-number">02</span><strong>カメラの前へ</strong><small>頭から足まで映す</small></div><div><span class="strip-number">03</span><strong>20秒ジャンプ</strong><small>最高到達点を更新</small></div></section>
  <footer>HIGH JUMP ARENA <span>CAMERA ESTIMATE · NOT AN OFFICIAL MEASUREMENT</span></footer>
`;

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const video = $<HTMLVideoElement>('#camera-video');
const canvas = $<HTMLCanvasElement>('#pose-canvas');
const ctx = canvas.getContext('2d')!;
const state = {
  mode: 'camera' as Mode,
  phase: 'ready' as Phase,
  pose: null as PoseLandmarker | null,
  stream: null as MediaStream | null,
  startAt: 0,
  calibration: [] as { foot: number; body: number }[],
  baselineFoot: 0,
  bodyPixels: 0,
  bestJump: 0,
  jumps: 0,
  airborne: false,
  lastVideoTime: -1,
  lastDetection: 0,
  demoJump: null as { started: number; height: number } | null,
};

function numeric(selector: string, fallback: number): number {
  const value = Number($<HTMLInputElement>(selector).value);
  if (!Number.isFinite(value) || value <= 0) return fallback;
  return selector === '#body-height' ? Math.min(230, Math.max(100, value)) : Math.min(320, Math.max(120, value));
}
function reach(): number { return numeric('#standing-reach', 210) + state.bestJump; }
function setStatus(message: string, helper?: string) {
  $('#arena-status').textContent = message;
  if (helper) $('#helper-text').textContent = helper;
}
function renderTargets() {
  $('#ruler').innerHTML = Array.from({ length: 8 }, (_, i) => {
    const cm = i * 50;
    return `<div class="ruler-mark" style="bottom:${cm / CEILING_CM * 100}%"><span>${cm}</span></div>`;
  }).join('');
  $('#target-lines').innerHTML = targets.map((t) => `<div class="target-line ${t.color}" style="bottom:${t.cm / CEILING_CM * 100}%"><span>${t.name}</span><strong>${t.cm} <small>cm</small></strong></div>`).join('');
}
function render() {
  const value = reach();
  const show = state.phase !== 'ready' || state.bestJump > 0;
  $('#best-reach').textContent = show ? value.toFixed(0) : '--';
  $('#line-height').textContent = show ? value.toFixed(0) : '--';
  $('#jump-count').textContent = `${state.jumps} JUMPS`;
  $('#player-line').style.bottom = `${Math.min(96, Math.max(0, value / CEILING_CM * 100))}%`;
  $('#player-line').classList.toggle('visible', show);
  const achieved = [...targets].reverse().find(t => value >= t.cm);
  $('#result-label').textContent = !show ? '挑戦を始めよう' : achieved ? `${achieved.name} 突破！` : `ジャンプ +${state.bestJump.toFixed(0)} cm`;
  if (state.phase === 'playing') {
    const remaining = Math.max(0, Math.ceil(ROUND_SECONDS - (performance.now() - state.startAt) / 1000));
    $('#timer').textContent = `${remaining}`;
    if (remaining === 0) finish();
  }
}

function validPoint(p?: NormalizedLandmark): p is NormalizedLandmark {
  return !!p && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1 && (p.visibility ?? 1) > 0.55;
}
function measure(points: NormalizedLandmark[]) {
  const nose = points[0], left = points[27], right = points[28];
  if (![nose, left, right].every(validPoint)) return null;
  const foot = Math.max(left.y, right.y);
  const body = (foot - nose.y) / 0.94;
  if (body < 0.35) return null;
  return { foot, body };
}
function drawPose(points: NormalizedLandmark[]) {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#c9fb59';
  for (const index of [0, 11, 12, 23, 24, 27, 28]) {
    const p = points[index];
    if (!validPoint(p)) continue;
    ctx.beginPath(); ctx.arc(p.x * canvas.width, p.y * canvas.height, 5, 0, Math.PI * 2); ctx.fill();
  }
}
function processPose(points: NormalizedLandmark[]) {
  drawPose(points);
  const m = measure(points);
  if (!m) { $('#camera-state').textContent = '全身を映してください'; return; }
  $('#camera-state').textContent = 'TRACKING';
  if (state.phase === 'calibrating') {
    state.calibration.push(m);
    if (state.calibration.length >= 35 && performance.now() - state.startAt >= 2500) {
      const sortedFeet = state.calibration.map(x => x.foot).sort((a, b) => a - b);
      const sortedBody = state.calibration.map(x => x.body).sort((a, b) => a - b);
      state.baselineFoot = sortedFeet[Math.floor(sortedFeet.length / 2)];
      state.bodyPixels = sortedBody[Math.floor(sortedBody.length / 2)];
      beginRound();
    }
  } else if (state.phase === 'playing') {
    const heightCm = numeric('#body-height', 165);
    const jump = Math.max(0, (state.baselineFoot - m.foot) / state.bodyPixels * heightCm);
    if (jump > 5 && jump < 125) {
      if (!state.airborne) { state.airborne = true; state.jumps += 1; }
      state.bestJump = Math.max(state.bestJump, jump);
      setStatus('JUMP DETECTED');
    } else if (jump < 3) {
      state.airborne = false;
      setStatus('READY TO JUMP');
    }
  }
  render();
}
async function cameraLoop() {
  if (!state.stream) return;
  requestAnimationFrame(cameraLoop);
  if (state.phase === 'calibrating' && performance.now() - state.startAt > 12000) {
    stopCamera();
    state.phase = 'ready';
    $('#start-button').removeAttribute('disabled');
    $('#start-text').textContent = 'もう一度試す';
    setStatus('CALIBRATION ERROR', '全身が映るようにカメラを置き、動かずに再試行してください');
    return;
  }
  if (!state.pose || video.readyState < 2) return;
  const now = performance.now();
  if (now - state.lastDetection < 65 || video.currentTime === state.lastVideoTime) return;
  state.lastDetection = now;
  state.lastVideoTime = video.currentTime;
  try {
    const result = state.pose.detectForVideo(video, now);
    if (result.landmarks[0]) processPose(result.landmarks[0]);
    else { ctx.clearRect(0, 0, canvas.width, canvas.height); $('#camera-state').textContent = '人を探しています'; }
  } catch (error) { console.error(error); setStatus('計測エラー', 'ページを再読み込みして試してください'); }
}
async function startCamera() {
  setStatus('LOADING CAMERA', 'カメラと姿勢モデルを準備しています…');
  $('#start-button').setAttribute('disabled', 'true');
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('このブラウザではカメラを使えません。HTTPS または localhost で開いてください。');
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } });
    video.srcObject = state.stream;
    await video.play();
    $('#camera-box').classList.add('on');
    if (!state.pose) {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      try {
        state.pose = await PoseLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' }, runningMode: 'VIDEO', numPoses: 1 });
      } catch {
        state.pose = await PoseLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' }, runningMode: 'VIDEO', numPoses: 1 });
      }
    }
    state.calibration = [];
    state.phase = 'calibrating';
    state.startAt = performance.now();
    $('#camera-state').textContent = 'CALIBRATING';
    $('#start-text').textContent = '計測中';
    setStatus('CALIBRATING', 'まっすぐ立って約3秒待ってください');
    requestAnimationFrame(cameraLoop);
  } catch (error) {
    state.stream?.getTracks().forEach(track => track.stop());
    state.stream = null;
    state.phase = 'ready';
    $('#start-button').removeAttribute('disabled');
    $('#camera-state').textContent = 'OFFLINE';
    setStatus('CAMERA ERROR', error instanceof Error ? error.message : 'カメラを開始できませんでした');
  }
}
function beginRound() {
  state.phase = 'playing';
  state.startAt = performance.now();
  $('#timer').textContent = String(ROUND_SECONDS);
  $('#start-text').textContent = 'プレイ中';
  setStatus('GO! JUMP!', state.mode === 'camera' ? '両足でジャンプ！ 最高到達点を記録します' : 'スペースキーまたは画面のボタンでジャンプ');
  render();
}
function startDemo() {
  state.bestJump = 0; state.jumps = 0; state.airborne = false;
  beginRound();
}
function demoJump() {
  if (state.mode !== 'demo' || state.phase !== 'playing' || state.demoJump) return;
  state.jumps += 1;
  const height = 32 + Math.random() * 38;
  state.demoJump = { started: performance.now(), height };
  state.bestJump = Math.max(state.bestJump, height);
  setStatus('NICE JUMP!');
  render();
  setTimeout(() => { state.demoJump = null; if (state.phase === 'playing') setStatus('READY TO JUMP'); }, 550);
}
function stopCamera() {
  state.stream?.getTracks().forEach(track => track.stop());
  state.stream = null;
  video.srcObject = null;
  $('#camera-box').classList.remove('on');
  $('#camera-state').textContent = 'OFFLINE';
}
function finish() {
  if (state.phase !== 'playing') return;
  state.phase = 'finished';
  stopCamera();
  $('#timer').textContent = '0';
  $('#start-button').removeAttribute('disabled');
  $('#start-text').textContent = 'もう一度挑戦';
  setStatus('FINISH!', `最高到達点 ${reach().toFixed(0)} cm ・ ${state.jumps} 回ジャンプ`);
  render();
}

document.querySelectorAll<HTMLButtonElement>('.mode').forEach(button => button.addEventListener('click', () => {
  if (state.phase === 'playing' || state.phase === 'calibrating') return;
  state.mode = button.dataset.mode as Mode;
  document.querySelectorAll('.mode').forEach(el => el.classList.toggle('active', el === button));
  $('#mode-description').textContent = state.mode === 'camera' ? '全身が映る位置にカメラを置き、実際にジャンプします。' : 'カメラなしで遊び方を試せます。数値はサンプルです。';
  $('#helper-text').textContent = state.mode === 'camera' ? 'カメラの使用許可が必要です' : 'スペースキーでもジャンプできます';
  $('#camera-note').textContent = state.mode === 'camera' ? 'カメラ映像は端末内で解析します' : 'デモのジャンプ値はランダムなサンプルです';
  $('#start-text').textContent = 'チャレンジ開始';
  state.phase = 'ready'; state.bestJump = 0; state.jumps = 0;
  $('#timer').textContent = String(ROUND_SECONDS);
  setStatus('READY TO JUMP'); render();
}));
$('#start-button').addEventListener('click', () => {
  if (state.mode === 'demo' && state.phase === 'playing') { demoJump(); return; }
  state.bestJump = 0; state.jumps = 0; state.airborne = false;
  if (state.mode === 'camera') void startCamera(); else startDemo();
});
document.addEventListener('keydown', (event) => {
  if (event.code === 'Space' && state.mode === 'demo' && state.phase === 'playing' && !(event.target instanceof HTMLInputElement)) { event.preventDefault(); demoJump(); }
});
['#body-height', '#standing-reach'].forEach(selector => $(selector).addEventListener('input', render));
window.addEventListener('beforeunload', stopCamera);
renderTargets(); render();
setInterval(render, 100);
