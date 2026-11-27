// Coach run what-if: real match vs the model playing the coach's instruction (scripts/coach_run.py -> data/coach_run.json).
// One deterministic timeline (window.__seek(t)) so it plays live or is recorded frame by frame (scripts/record_showcase.py --page coach_run.html).
import * as THREE from 'three';

const $ = (id) => document.getElementById(id);
const DATA = new URLSearchParams(location.search).get('data') || 'coach_run';      // coach_run.html?data=<file in data/, no .json>
const D = await (await fetch(`data/${DATA}.json`, { cache: 'no-store' })).json();
if (D.att === 'away') {                     // turn the pitch so the attacking team always plays left to right
  const fl = (q) => (q[0] < -50 ? q : [105 - q[0], 68 - q[1]]);
  for (const tr of [D.real, D.run, D.twin]) { tr.p = tr.p.map((pl) => pl.map(fl)); tr.b = tr.b.map(fl); }
  D.from = fl(D.from); D.to = fl(D.to);
}
const LABEL = D.label || 'run in behind';
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const ease = (v) => { v = clamp(v); return v * v * (3 - 2 * v); };
const last = (n) => n.split(' ').slice(-1)[0];
const S = D.slot, WHO = last(D.who), FPS = D.fps, NT = D.real.p[0].length, DUR_DATA = (NT - 1) / FPS;
const R = D.run, TW = D.twin, RS = D.run_stats;

// ------------------------------------------------------------------ data helpers
const at = (track, k, t) => {                       // position of player k (or the ball if k < 0) at data time t (s), linear
  const arr = k < 0 ? track.b : track.p[k];
  const f = clamp(t * FPS, 0, arr.length - 1), i = Math.floor(f), j = Math.min(i + 1, arr.length - 1), u = f - i;
  const a = arr[i], b = arr[j];
  if (a[0] < -50 || b[0] < -50) return null;
  return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];
};
const nearest = (track, t, from, to) => {
  const b = at(track, -1, t); let best = -1, bd = 1e9;
  for (let k = from; k < to; k++) { const p = at(track, k, t); if (!p) continue; const d = Math.hypot(p[0] - b[0], p[1] - b[1]); if (d < bd) { bd = d; best = k; } }
  return [best, bd];
};
const firstTouch = (track) => { for (let i = 0; i < NT; i++) { const [k, d] = nearest(track, i / FPS, 0, 22); if (k === S && d < 1.2) return i / FPS; } return null; };
const realTouch = firstTouch(D.real);
const realLost = D.real.lost_ball_s;
const passS = R.pass_s, touchS = R.touch_s ?? firstTouch(R);
const passer = passS != null ? last(D.names[nearest(R, passS, D.side === 'home' ? 0 : 11, D.side === 'home' ? 11 : 22)[0]]) : '';
const carrier0 = D.carrier ? last(D.carrier) : last(D.names[nearest(D.real, 0, 0, 22)[0]]);
const goalS = R.goal ? (R.goal_s ?? (R.p[0].length - 1) / FPS) : null;
const ATT0 = D.att === 'away' ? 11 : 0;
const scorer = (() => {
  if (goalS == null) return null;
  for (let u = goalS - 0.2; u > 0; u -= 0.1) { const [k, d] = nearest(R, u, 0, 22); if (d < 1.5) return k >= ATT0 && k < ATT0 + 11 ? k : null; }
  return null;
})();
const scorerName = scorer == null ? '' : (scorer === S ? WHO : last(D.names[scorer]));
document.getElementById('legend').innerHTML = `<i style="background:#6aa8ff"></i>${last(D.home)} <i style="background:#ff6f61"></i>${last(D.away)} <i style="background:#f5be46"></i>instructed player`;

// ------------------------------------------------------------------ scene
const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.autoClear = false;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a1222);
const cam = new THREE.PerspectiveCamera(36, 16 / 9, 0.5, 700);
scene.add(new THREE.HemisphereLight(0xe4ecff, 0x0b1a10, 1.15));
const sun = new THREE.DirectionalLight(0xffffff, 1.3); sun.position.set(-30, 90, 50); scene.add(sun);
let VW = 1920, VH = 1080;
function resize() { VW = innerWidth; VH = innerHeight; renderer.setSize(VW, VH, false); }
addEventListener('resize', resize); resize();

const P = (x, y, h = 0) => new THREE.Vector3(x - 52.5, h, -(y - 34));
const pitch = new THREE.Group(); scene.add(pitch);
for (let i = 0; i < 14; i++) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(105 / 14, 68), new THREE.MeshLambertMaterial({ color: i % 2 ? 0x1f6a3e : 0x1b5f37 }));
  m.rotation.x = -Math.PI / 2; m.position.set(-52.5 + (i + 0.5) * 105 / 14, 0, 0); pitch.add(m);
}
const surround = new THREE.Mesh(new THREE.PlaneGeometry(150, 110), new THREE.MeshLambertMaterial({ color: 0x163f28 }));
surround.rotation.x = -Math.PI / 2; surround.position.y = -0.02; pitch.add(surround);
const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 });
const seg = (pts) => pitch.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts.map(([x, y]) => P(x, y, 0.04))), lineMat));
seg([[0, 0], [105, 0], [105, 68], [0, 68], [0, 0]]); seg([[52.5, 0], [52.5, 68]]);
seg([[0, 13.84], [16.5, 13.84], [16.5, 54.16], [0, 54.16]]); seg([[105, 13.84], [88.5, 13.84], [88.5, 54.16], [105, 54.16]]);
seg([[0, 24.84], [5.5, 24.84], [5.5, 43.16], [0, 43.16]]); seg([[105, 24.84], [99.5, 24.84], [99.5, 43.16], [105, 43.16]]);
seg(Array.from({ length: 49 }, (_, i) => [52.5 + 9.15 * Math.cos(i / 48 * 2 * Math.PI), 34 + 9.15 * Math.sin(i / 48 * 2 * Math.PI)]));
seg(Array.from({ length: 25 }, (_, i) => { const a = -0.93 + 1.86 * i / 24; return [94 - 9.15 * Math.cos(a), 34 + 9.15 * Math.sin(a)]; }).filter((p) => p[0] < 88.5));
const goal = new THREE.Mesh(new THREE.BoxGeometry(1.2, 2.44, 7.32), new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 }));
goal.position.copy(P(105.6, 34, 1.22)); pitch.add(goal);

function label(text, color = '#f5be46', size = 44) {
  const c = document.createElement('canvas'); const g = c.getContext('2d');
  g.font = `700 ${size}px Segoe UI, Arial`; const w = Math.ceil(g.measureText(text).width) + 28;
  c.width = w; c.height = size + 22;
  g.font = `700 ${size}px Segoe UI, Arial`;
  g.fillStyle = 'rgba(10,18,34,.82)'; g.beginPath(); g.roundRect(0, 0, w, c.height, 12); g.fill();
  g.fillStyle = color; g.textBaseline = 'middle'; g.fillText(text, 14, c.height / 2 + 2);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sp.scale.set(w / c.height * 2.6, 2.6, 1); sp.renderOrder = 10; return sp;
}

// one "view" = 22 players + ball + trails, for the real match or a model run
function makeView(track, runnerCol, trailCol) {
  const g = new THREE.Group(); scene.add(g);
  const homeSide = D.side === 'home';
  const players = [];
  for (let k = 0; k < 22; k++) {
    const own = (k < 11) === homeSide;
    const col = k === S ? runnerCol : (k < 11 ? 0x6aa8ff : 0xff6f61);
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.85, 0.85, 1.9, 20), new THREE.MeshStandardMaterial({ color: col, roughness: 0.5 }));
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.55, 16, 12), new THREE.MeshStandardMaterial({ color: col, roughness: 0.5 }));
    head.position.y = 1.35; m.add(head);
    const pg = new THREE.Group(); pg.add(m); m.position.y = 0.95;
    if (k === S) {
      const ring = new THREE.Mesh(new THREE.RingGeometry(1.6, 2.2, 40), new THREE.MeshBasicMaterial({ color: runnerCol, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }));
      ring.rotation.x = -Math.PI / 2; ring.position.y = 0.06; pg.add(ring);
      const lab = label(WHO, '#' + runnerCol.toString(16).padStart(6, '0')); lab.position.y = 4.6; pg.add(lab);
    }
    g.add(pg); players.push({ pg, own });
  }
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.55, 18, 14), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x666666 }));
  g.add(ball);
  const mkLine = (col, n, w = 1) => {
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const l = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: w })); g.add(l); return l;
  };
  const trail = new THREE.InstancedMesh(new THREE.CircleGeometry(0.42, 12), new THREE.MeshBasicMaterial({ color: trailCol, transparent: true, opacity: 0.95, depthWrite: false }), NT * 2);
  const btrail = mkLine(0xffffff, 40, 0.55);
  g.add(trail);
  return { g, track, players, ball, trail, btrail };
}
const dummy = new THREE.Object3D();
function setLine(line, pts) {
  const a = line.geometry.attributes.position.array;
  pts.forEach((p, i) => { a[3 * i] = p.x; a[3 * i + 1] = p.y; a[3 * i + 2] = p.z; });
  line.geometry.attributes.position.needsUpdate = true; line.geometry.setDrawRange(0, pts.length);
  line.geometry.computeBoundingSphere();
}
function pose(v, t, trailOn = true) {
  for (let k = 0; k < 22; k++) {
    const p = at(v.track, k, t);
    v.players[k].pg.visible = !!p;
    if (p) v.players[k].pg.position.copy(P(p[0], p[1]));
  }
  const b = at(v.track, -1, t); if (b) v.ball.position.copy(P(b[0], b[1], 0.55));
  let n = 0;
  if (trailOn) for (let u = 0; u <= t + 1e-6; u += 0.1) {
    const p = at(v.track, S, u); if (!p) continue;
    dummy.position.copy(P(p[0], p[1], 0.1)); dummy.rotation.set(-Math.PI / 2, 0, 0); dummy.updateMatrix(); v.trail.setMatrixAt(n++, dummy.matrix);
  }
  v.trail.count = n; v.trail.instanceMatrix.needsUpdate = true;
  const bp = []; for (let u = Math.max(0, t - 1.5); u <= t + 1e-6; u += 0.05) { const p = at(v.track, -1, u); if (p) bp.push(P(p[0], p[1], 0.5)); }
  setLine(v.btrail, bp);
}
const vReal = makeView(D.real, 0xf5be46, 0xffffff);
const vRun = makeView(R, 0xf5be46, 0xf5be46);

// the coach's arrow and target zone
const coach = new THREE.Group(); scene.add(coach);
const A = D.from, B = D.to;
const dashGeo = new THREE.BufferGeometry().setFromPoints([P(A[0], A[1], 0.15), P(B[0], B[1], 0.15)]);
const dash = new THREE.Line(dashGeo, new THREE.LineDashedMaterial({ color: 0xf5be46, dashSize: 1.4, gapSize: 0.9 })); dash.computeLineDistances(); coach.add(dash);
const cone = new THREE.Mesh(new THREE.ConeGeometry(1.3, 3.2, 18), new THREE.MeshBasicMaterial({ color: 0xf5be46 }));
cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), P(B[0], B[1]).sub(P(A[0], A[1])).normalize()); coach.add(cone);
const ribbon = new THREE.Mesh(new THREE.PlaneGeometry(1, 1.1), new THREE.MeshBasicMaterial({ color: 0xf5be46, transparent: true, opacity: 0.75, side: THREE.DoubleSide, depthWrite: false }));
ribbon.rotation.order = 'YXZ'; ribbon.rotation.set(-Math.PI / 2, Math.atan2(B[1] - A[1], B[0] - A[0]), 0, 'YXZ'); coach.add(ribbon);
const zone = new THREE.Mesh(new THREE.CircleGeometry(5, 48), new THREE.MeshBasicMaterial({ color: 0xf5be46, transparent: true, opacity: 0.18, depthWrite: false }));
zone.rotation.x = -Math.PI / 2; zone.position.copy(P(B[0], B[1], 0.07)); coach.add(zone);
const zoneRing = new THREE.Mesh(new THREE.RingGeometry(4.8, 5.1, 64), new THREE.MeshBasicMaterial({ color: 0xf5be46, side: THREE.DoubleSide }));
zoneRing.rotation.x = -Math.PI / 2; zoneRing.position.copy(P(B[0], B[1], 0.08)); coach.add(zoneRing);
const coachLab = label(`COACH: ${LABEL}`, '#f5be46', 40); coachLab.position.copy(P((A[0] + B[0]) / 2, (A[1] + B[1]) / 2 + 6, 4)); coach.add(coachLab);
function setArrow(f) {
  const ax = A[0] + (B[0] - A[0]) * f, ay = A[1] + (B[1] - A[1]) * f;
  dash.geometry.setFromPoints([P(A[0], A[1], 0.15), P(ax, ay, 0.15)]); dash.computeLineDistances();
  cone.position.copy(P(ax, ay, 0.6)); cone.visible = f > 0.05;
  const len = Math.hypot(ax - A[0], ay - A[1]); ribbon.visible = len > 0.3;
  ribbon.scale.set(Math.max(len - 1.5, 0.01), 1, 1); ribbon.position.copy(P((A[0] + ax) / 2 - (ax - A[0]) / Math.max(len, 1e-6) * 0.75, (A[1] + ay) / 2 - (ay - A[1]) / Math.max(len, 1e-6) * 0.75, 0.1));
  zone.scale.setScalar(Math.max(0.01, ease((f - 0.6) / 0.4))); zoneRing.scale.copy(zone.scale);
  coachLab.material.opacity = ease((f - 0.3) / 0.4);
}

// final picture: the three paths of the instructed player
const paths = new THREE.Group(); scene.add(paths);
const pathOf = (track, col, op = 1) => {
  const pts = []; for (let i = 0; i < NT; i++) { const p = at(track, S, i / FPS); if (!p) continue; const v = P(p[0], p[1], 0.18); if (!pts.length || pts[pts.length - 1].distanceTo(v) > 0.15) pts.push(v); }
  const tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 200, 0.32, 8, false),
    new THREE.MeshBasicMaterial({ color: col, transparent: op < 1, opacity: op })); paths.add(tube);
  const end = new THREE.Mesh(new THREE.SphereGeometry(0.9, 16, 12), new THREE.MeshBasicMaterial({ color: col })); end.position.copy(pts[pts.length - 1]); paths.add(end);
  return tube;
};
pathOf(D.real, 0xffffff); pathOf(R, 0xf5be46); pathOf(TW, 0x8a96aa, 0.8);
const pl = [label('real', '#ffffff', 36), label('coach run', '#f5be46', 36), label('model, no change', '#aab4c6', 36)];
[[D.real, 0], [R, 1], [TW, 2]].forEach(([tr, i]) => { const p = at(tr, S, DUR_DATA); pl[i].position.copy(P(p[0], p[1], 3.2)); paths.add(pl[i]); });
const startLab = label('start', '#ffffff', 32); startLab.position.copy(P(A[0], A[1] - 4, 2.5)); paths.add(startLab);

// ------------------------------------------------------------------ timeline
const camPose = (pos, look) => ({ pos, look });
const C_WIDE = camPose(P(52, -52, 62), P(70, 30, 0));
const MY = (D.from[1] + D.to[1]) / 2, MX = (D.from[0] + D.to[0]) / 2, SPAN = Math.hypot(D.to[0] - D.from[0], D.to[1] - D.from[1]);
const C_ATT = camPose(P(Math.min(70, MX), MY - 54, 40), P(Math.min(84, MX + 14), MY + 2, 0));
const C_CLOSE = camPose(P(MX - 6, MY - 30 - SPAN * 0.5, 22 + SPAN * 0.45), P(MX + 2, MY, 0));
const C_SPLIT = camPose(P(72, -44, 54), P(82, 28, 0));
const C_PATHS = camPose(P(84, -42, 60), P(100, 30, 0));
let curLook = new THREE.Vector3();
function setCam(a, b, f, aspect) {
  cam.position.lerpVectors(a.pos, b.pos, ease(f)); curLook.lerpVectors(a.look, b.look, ease(f)); cam.lookAt(curLook);
  cam.aspect = aspect; cam.updateProjectionMatrix();
}
// what-if replay speed: 1x, slow motion (0.5x) from just before the pass to 1 s after he gets the ball
const slowA = Math.max(0, (passS ?? 2.5) - 0.4), slowB = Math.min(DUR_DATA, (touchS ?? (passS ?? 2.5) + 2) + 1.2);
const runDur = slowA + (slowB - slowA) * 2 + (DUR_DATA - slowB);
const runT = (s) => s < slowA ? s : s < slowA + (slowB - slowA) * 2 ? slowA + (s - slowA) / 2 : slowB + (s - slowA - (slowB - slowA) * 2);
const SC = [['title', 6], ['player', 11], ['freeze', 6], ['real', 14], ['coach', 8], ['run', runDur + 1.5], ['split', 15], ['paths', 14]];
let t0 = 0; const T = {}; for (const [n, d] of SC) { T[n] = [t0, t0 + d]; t0 += d; }
const DURATION = t0;
const ph = (n, t) => (t - T[n][0]) / (T[n][1] - T[n][0]);
const show = (id, on, html) => { const e = $(id); if (html != null && e.dataset.h !== html) { e.innerHTML = html; e.dataset.h = html; } e.style.opacity = on; };
const CLK0 = D.clock ? +D.clock.split(':')[0] * 60 + +D.clock.split(':')[1] : D.start_s;
const clock = (dt) => { const s = CLK0 + dt; return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`; };
const f1 = (v) => v == null ? '–' : v.toFixed(1);
const sg = (v, d = 3) => `${v >= 0 ? '+' : ''}${v.toFixed(d)}`;

function realCap(t) {
  if (realTouch != null && t >= realTouch - 0.2 && (realLost == null || t < realLost)) return `The ball reaches ${WHO} (${f1(realTouch)} s), well away from the coach's spot.`;
  if (realLost != null && t >= realLost) return `Ball lost after ${f1(realLost)} s. ${WHO} never got nearer than ${D.real.closest_to_spot_m} m to the coach's spot.`;
  return `Real match, ${clock(0)}. ${carrier0} has the ball; ${WHO} does not make the run.`;
}
function runCap(t) {
  if (goalS != null && t >= goalS - 0.4) return `GOAL after ${f1(goalS)} s${scorerName ? `, scored by ${scorerName}` : ''}. (Run shown: seed ${R.seed}, ${D.show_why || 'see the results'}.)`;
  if (passS == null && touchS == null) return goalS != null && t > goalS - 3
    ? `The space his run opened is used: the attack goes on without him on the ball.`
    : `${WHO} makes the coach's run (${LABEL}); defenders have to follow him. The team model plays the ball elsewhere.`;
  if (passS != null && t < passS) return `${WHO} makes the coach's run (${LABEL}). His teammates are played by the team model.`;
  if (touchS != null && t < touchS) return `${passer} plays him the ball (${f1(passS)} s).`;
  if (touchS != null && t < touchS + 2) return `${WHO} gets the ball (${f1(touchS)} s), ${D.run.closest_to_spot_m} m from the coach's spot.`;
  return `The model plays on: ${R.shots} shot${R.shots === 1 ? '' : 's'}, ${R.goal ? 'goal' : 'no goal'} in this run (threat peak ${(R.max_threat * 100).toFixed(0)}% vs ${(TW.max_threat * 100).toFixed(0)}% without the run).`;
}
function playerPanel() {
  const p = D.player || {}, mv = p.movement || {}, ph = mv.phases || {}, w = ph['with ball'] || {}, wo = ph['without ball'] || {};
  const mx = Math.max(p.behind90 || 0, p.wide_median90 || 0, ...((p.top_wide || []).map((q) => q[1]))) || 1;
  const bar = (lab, v, cls) => `<div class="row"><span>${lab}</span><span><span style="display:inline-block;width:${(220 * v / mx).toFixed(0)}px;height:14px;border-radius:7px;background:${cls};vertical-align:middle;margin-right:10px"></span><b>${v.toFixed(1)}</b></span></div>`;
  return `<h4>PLAYER MODEL · ${(D.who || '').toUpperCase()} · ${mv.position || ''} · ${p.matches} MATCHES, ${p.minutes} MIN</h4>
    <div class="row"><span>Runs in behind the defensive line</span><span class="gold">${p.behind_runs} runs</span></div>
    ${bar(`${WHO}, per 90`, p.behind90 || 0, '#f5be46')}${bar(`wide midfielders, median (${p.wide_n})`, p.wide_median90 || 0, '#8a96aa')}
    ${mv.phases ? `<div class="row"><span>Movement model, with the ball (per 2 s)</span><span>${(w.real_forward_m ?? 0) >= 0 ? '+' : ''}${(w.real_forward_m ?? 0).toFixed(2)} m forward</span></div>
    <div class="row"><span>Without the ball</span><span>${(wo.real_forward_m ?? 0).toFixed(2)} m (drops back)</span></div>
    <div class="row"><span>His own model vs role model</span><span class="mut">${mv.test ? mv.test.verdict : '–'}</span></div>` : ''}
    <div class="note">${WHO} makes ${(p.behind90 || 0).toFixed(1)} runs in behind per 90 (median of ${p.wide_n} wide midfielders: ${(p.wide_median90 || 0).toFixed(1)}), from real tracking. In this moment he did not make one.</div>`;
}
function resultPanel() {
  const rows = D.summary.filter((r) => ['goal', 'max_threat', 'max_xt', 'shots', 'xg', 'touched'].includes(r.metric)).map((r) => {
    const p = r.metric === 'shots' || r.metric === 'goal' ? 2 : 3;
    return `<div class="row"><span>${r.what}</span><span>${r.no_change.toFixed(p)} → <b class="${r.verdict === 'up' ? 'up' : ''}">${r.coach_run.toFixed(p)}</b>
      <span class="mut">(${sg(r.diff, p)} ± ${r.ci95.toFixed(p)}${r.verdict === 'no clear effect' ? ', unclear' : ''})</span></span></div>`;
  }).join('');
  return `<h4>${D.seeds} PAIRED RUNS · MODEL NO CHANGE → COACH RUN</h4>${rows}
    <div class="row"><span>Reached the spot</span><span class="gold">${(RS.arrived * 100).toFixed(0)}% <span class="mut">median ${RS.arrive_s.toFixed(1)} s</span></span></div>
    <div class="row"><span>Pass played to him</span><span class="gold">${(RS.passed * 100).toFixed(0)}%</span></div>
    <div class="row"><span>Real match: nearest to the spot</span><span>${D.real.closest_to_spot_m} m <span class="mut">(run shown: ${D.run.closest_to_spot_m} m)</span></span></div>
    <div class="note">Same seed for both runs of a pair, so the difference is the instruction. Run shown: seed ${R.seed}, ${D.show_why || 'the median one (by threat) of the runs where the pass reached him'}.
      Clear (95% interval above zero): ${D.summary.filter((r) => r.verdict === 'up').map((r) => r.what).join(', ') || 'none'}; within noise: ${D.summary.filter((r) => r.verdict !== 'up').map((r) => r.what).join(', ')}.</div>`;
}

function seek(t) {
  t = clamp(t, 0, DURATION - 1e-3);
  const n = SC.find(([k]) => t < T[k][1])[0], f = ph(n, t);
  let views = [[vReal, 0]], split = false, aspect = VW / VH;
  coach.visible = false; paths.visible = false; vReal.g.visible = vRun.g.visible = true;
  show('kick', 0); show('title', 0); show('sub', 0); show('panel', 0); show('cap', 0); show('tagL', 0); show('tagR', 0); show('split', 0); show('clock', 0); show('legend', 0);
  if (n === 'title') {
    setCam(C_WIDE, C_ATT, f * 0.6, aspect); pose(vReal, 0, false);
    const o = ease(f * 3) * (1 - ease((f - 0.85) / 0.15));
    show('kick', o, 'Player model + what-if'); show('title', o, `“${WHO}, ${LABEL}. We play you there.”`);
    show('sub', o, `${D.home} vs ${D.away}, ${clock(0)}. The real move, then the team models playing the coach's run in GRF, then both side by side.`);
  } else if (n === 'player') {
    setCam(C_ATT, C_CLOSE, f * 0.4, aspect); pose(vReal, 0, false);
    show('tagL', 1); show('clock', 1, clock(0)); show('legend', 1);
    show('panel', ease(f * 4), playerPanel());
    show('cap', ease(f * 4), `Player model first: does ${WHO} make runs in behind? Then the what-if: what happens if he makes this one.`);
  } else if (n === 'freeze') {
    setCam(C_ATT, C_ATT, 0, aspect); pose(vReal, 0, false);
    show('tagL', 1); show('clock', 1, clock(0)); show('legend', 1);
    show('cap', ease(f * 4), `The real moment: ${carrier0} has the ball${D.carrier_free_m ? `, ${D.carrier_free_m.toFixed(0)} m free` : ' 12 m free'}. ${WHO} (gold ring) is at (${A[0].toFixed(0)}, ${A[1].toFixed(0)}).`);
  } else if (n === 'real') {
    const dt = f * DUR_DATA; setCam(C_ATT, C_ATT, 0, aspect); pose(vReal, dt);
    show('tagL', 1); show('clock', 1, `${clock(dt)} · real`); show('cap', 1, realCap(dt)); show('legend', 1);
  } else if (n === 'coach') {
    views = [[vRun, 0]]; setCam(C_ATT, C_CLOSE, f * 0.5, aspect); pose(vRun, 0, false);
    coach.visible = true; setArrow(ease((f - 0.1) / 0.55));
    show('tagR', 1); show('clock', 1, clock(0)); show('legend', 1);
    show('cap', ease(f * 3), `Back to ${clock(0)}. The coach: “${WHO}, from (${A[0].toFixed(0)}, ${A[1].toFixed(0)}) ${LABEL} to (${B[0].toFixed(0)}, ${B[1].toFixed(0)}); we play you there.”`);
  } else if (n === 'run') {
    views = [[vRun, 0]]; const s = Math.min(f * (T.run[1] - T.run[0]), runDur); const dt = runT(s);
    setCam(C_CLOSE, C_ATT, ease(dt / DUR_DATA), aspect); pose(vRun, dt);
    coach.visible = true; setArrow(1); coachLab.visible = false; dash.material.opacity = 0.5;
    const slow = s >= slowA && s < slowA + (slowB - slowA) * 2;
    show('tagR', 1); show('clock', 1, `${clock(dt)} · model${slow ? ' · slow motion' : ''}`); show('cap', 1, runCap(dt)); show('legend', 1);
    if (goalS != null && dt >= goalS - 0.3) { show('kick', 1, 'GOAL'); show('title', 1, `GOAL${scorerName ? ` · ${scorerName}` : ''} · after ${WHO}'s run, ${f1(goalS)} s`); }
  } else if (n === 'split') {
    split = true; const dt = clamp(f * 15 / 13) * DUR_DATA; aspect = (VW / 2) / VH;
    setCam(C_SPLIT, C_SPLIT, 0, aspect); pose(vReal, dt); pose(vRun, dt);
    views = [[vReal, 0], [vRun, 1]];
    show('tagL', 1); show('tagR', 1); show('split', 1); show('clock', 1, `${clock(dt)}`);
    show('cap', 1, `Same start, same clock. Left: what ${WHO} really did. Right: the coach's run, played by the model.`);
  } else {
    views = [[null, 0]]; setCam(C_PATHS, C_PATHS, 0, aspect);
    vReal.g.visible = vRun.g.visible = false; paths.visible = true; coach.visible = true; setArrow(1); coachLab.visible = false;
    show('panel', ease(f * 3), resultPanel()); show('kick', ease(f * 3), `${WHO}'s path, 12 s`);
    show('cap', ease(f * 3), `White: real (${D.real.run_m} m in the first 4 s). Gold: the coach's run (${D.run.run_m} m). Grey: the model without the instruction.`);
  }
  coachLab.visible = n === 'coach';
  if (n !== 'run') dash.material.opacity = 1;
  dash.material.transparent = true;
  // draw
  renderer.setScissorTest(split);
  renderer.setViewport(0, 0, VW, VH); renderer.setScissor(0, 0, VW, VH); renderer.clear();
  for (const [v, side] of views) {
    if (split) {
      vReal.g.visible = v === vReal; vRun.g.visible = v === vRun; coach.visible = v === vRun;
      renderer.setViewport(side * VW / 2, 0, VW / 2, VH); renderer.setScissor(side * VW / 2, 0, VW / 2, VH);
    } else if (v) { vReal.g.visible = v === vReal; vRun.g.visible = v === vRun; }
    renderer.render(scene, cam);
  }
}

window.__seek = seek;
window.__duration = DURATION;
window.__scenes = SC.map(([k]) => (T[k][0] + T[k][1]) / 2);
seek(0);
window.__ready = true;
if (!location.search.includes('record')) {
  const start = performance.now();
  const loop = () => { seek(((performance.now() - start) / 1000) % DURATION); requestAnimationFrame(loop); };
  requestAnimationFrame(loop);
}
