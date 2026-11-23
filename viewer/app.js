// Match replay viewer: draws a real match exactly as recorded (22 players + ball from tracking data), and other runs of the
// same stretch (GRF replays, later what-if simulations) as ghost layers next to it. Data files come from scripts/export_viewer.py.
//
// Pitch coordinates in the data: metres x 0..105 (home attacks towards 105), y 0..68, stored as decimetres.
// Three.js scene: X = x - 52.5, Z = 34 - y, Y = up.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadThreat, threatOf } from './threat.js';
import { pressOf, pressSeries } from './press.js';

const L = 105, W = 68, DM = 10;
const HOME = 0xe0473c, AWAY = 0x3a7bd5, HOME_GK = 0xff9f1c, AWAY_GK = 0x8fd14f;
const LAYER_COLORS = [0xf2c14e, 0xc77dff, 0x5ce1e6, 0xff7aa2, 0x9be564, 0xff9f43, 0xffffff];

const $ = (id) => document.getElementById(id);
const pct = (p) => (100 * p).toFixed(p < 0.1 ? 1 : 0);
const S = { data: null, t: 0, t0: 0, t1: 1, playing: false, speed: 1, cam: 'broadcast', roster: null, rosterIdx: -1,
            showNumbers: true, showNames: false, showTrail: true, layers: [], threatState: {}, pressState: {}, press: null, cards: null, animDt: 0,
            view: { mode: 'overlay', layer: -1, inside: false },
            cb: { on: false, t: 0, team: 'home', board: 'defence', allowed: [], moves: {}, start: {}, drag: -1, threat0: null } };

// ---------- scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
$('view').appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1412);
scene.fog = new THREE.Fog(0x0f1412, 160, 320);
const camera = new THREE.PerspectiveCamera(42, 1, 0.5, 600);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enabled = false;
controls.maxPolarAngle = Math.PI * 0.49;
scene.add(new THREE.HemisphereLight(0xffffff, 0x2a3a2a, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(-40, 80, 30);
scene.add(sun);

const X = (x) => x - L / 2;
const Z = (y) => W / 2 - y;

function buildPitch() {
  const g = new THREE.Group();
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(L + 30, W + 24), new THREE.MeshLambertMaterial({ color: 0x1f5a2c }));
  ground.rotation.x = -Math.PI / 2;
  g.add(ground);
  const cv = document.createElement('canvas');            // mowing stripes
  cv.width = 1050; cv.height = 680;
  const c = cv.getContext('2d');
  for (let i = 0; i < 14; i++) { c.fillStyle = i % 2 ? '#2f7d3f' : '#2a7238'; c.fillRect(i * 75, 0, 75, 680); }
  const pitch = new THREE.Mesh(new THREE.PlaneGeometry(L, W), new THREE.MeshLambertMaterial({ map: new THREE.CanvasTexture(cv) }));
  pitch.rotation.x = -Math.PI / 2;
  pitch.position.y = 0.01;
  g.add(pitch);
  const mat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 });
  const line = (pts) => g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts.map(([x, y]) => new THREE.Vector3(X(x), 0.03, Z(y)))), mat));
  const rect = (x0, y0, x1, y1) => line([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]);
  const arc = (cx, cy, r, a0, a1, n = 48) => line(Array.from({ length: n + 1 }, (_, i) => {
    const a = a0 + (a1 - a0) * i / n; return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  }));
  rect(0, 0, L, W);
  line([[L / 2, 0], [L / 2, W]]);
  arc(L / 2, W / 2, 9.15, 0, 2 * Math.PI, 72);
  for (const [gx, s] of [[0, 1], [L, -1]]) {
    rect(gx, W / 2 - 20.16, gx + s * 16.5, W / 2 + 20.16);
    rect(gx, W / 2 - 9.16, gx + s * 5.5, W / 2 + 9.16);
    const a = Math.acos(5.5 / 9.15);                          // the arc outside the penalty area
    s > 0 ? arc(11, W / 2, 9.15, -a, a) : arc(L - 11, W / 2, 9.15, Math.PI - a, Math.PI + a);
    const spot = new THREE.Mesh(new THREE.CircleGeometry(0.2, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    spot.rotation.x = -Math.PI / 2; spot.position.set(X(gx + s * 11), 0.03, 0); g.add(spot);
    const post = new THREE.MeshLambertMaterial({ color: 0xffffff });
    for (const py of [W / 2 - 3.66, W / 2 + 3.66]) {
      const p = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 2.44, 8), post);
      p.position.set(X(gx), 1.22, Z(py)); g.add(p);
    }
    const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 7.32, 8), post);
    bar.rotation.x = Math.PI / 2; bar.position.set(X(gx), 2.44, 0); g.add(bar);
    const net = new THREE.Mesh(new THREE.BoxGeometry(2, 2.44, 7.32), new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.25 }));
    net.position.set(X(gx) - s * 1, 1.22, 0); g.add(net);
  }
  for (const [cx, cy, a0] of [[0, 0, 0], [L, 0, Math.PI / 2], [L, W, Math.PI], [0, W, 1.5 * Math.PI]]) arc(cx, cy, 1, a0, a0 + Math.PI / 2, 12);
  scene.add(g);
}
buildPitch();

// ---------- labels ----------
const texCache = new Map();
function labelTexture(text, big) {
  const key = text + (big ? '#b' : '');
  if (texCache.has(key)) return texCache.get(key);
  const cv = document.createElement('canvas');
  cv.width = big ? 512 : 128; cv.height = 64;
  const c = cv.getContext('2d');
  c.font = `bold ${big ? 34 : 40}px system-ui, sans-serif`;
  c.textAlign = 'center'; c.textBaseline = 'middle';
  c.lineWidth = 6; c.strokeStyle = 'rgba(0,0,0,0.75)'; c.strokeText(text, cv.width / 2, 34);
  c.fillStyle = '#fff'; c.fillText(text, cv.width / 2, 34);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  texCache.set(key, t);
  return t;
}
function makeSprite(big) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: labelTexture('', big), depthTest: false, transparent: true }));
  s.scale.set(big ? 8 : 2, 1, 1);
  s.renderOrder = 10;
  return s;
}

// ---------- players, ball, ghosts ----------
// Players are simple articulated people (no outside model files): shirt, shorts, socks, arms and legs that swing with
// the player's real speed, turned to face the way he runs. Ghost (layer) players are the same people, see-through.
const shadowGeo = new THREE.CircleGeometry(0.6, 16);
const shadowMat = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 });
const G = {
  torso: new THREE.CapsuleGeometry(0.2, 0.42, 4, 10), head: new THREE.SphereGeometry(0.13, 14, 10),
  hair: new THREE.SphereGeometry(0.135, 14, 8, 0, Math.PI * 2, 0, Math.PI / 2), shorts: new THREE.CylinderGeometry(0.22, 0.2, 0.22, 10),
  thigh: new THREE.CapsuleGeometry(0.075, 0.3, 4, 8), shin: new THREE.CapsuleGeometry(0.06, 0.32, 4, 8), shoe: new THREE.BoxGeometry(0.11, 0.07, 0.24),
  upper: new THREE.CapsuleGeometry(0.055, 0.24, 4, 8), fore: new THREE.CapsuleGeometry(0.045, 0.22, 4, 8),
};
const SKIN = 0xd9a77f, HAIR = 0x2b1d14, SHOE = 0x111111;
// the marker of a player the coach moved: a yellow arrow above his head and a ring at his feet
const MARK = { cone: new THREE.ConeGeometry(0.55, 1.2, 18), mat: new THREE.MeshBasicMaterial({ color: 0xffd84d, depthTest: false, transparent: true }),
               ring: new THREE.RingGeometry(1.1, 1.5, 32), ringMat: new THREE.MeshBasicMaterial({ color: 0xffd84d, side: THREE.DoubleSide, transparent: true, opacity: 0.95 }) };
function setMark(p, on) {
  p.userData.mark.visible = on; p.userData.ring.visible = on;
  if (on) p.userData.mark.position.y = 4.4 + 0.35 * Math.sin(performance.now() * 0.006);
}
function makePlayer(color, ghost) {
  const g = new THREE.Group();
  const M = (c) => ghost ? new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.42, depthWrite: false }) : new THREE.MeshLambertMaterial({ color: c });
  const shirt = M(color), skin = M(SKIN), dark = M(0x1a1a22), sock = M(color), shoe = M(SHOE), hair = M(HAIR);
  const body = new THREE.Group(); g.add(body);                 // body leans forward with speed; legs stay under it
  const mesh = (geo, mat, parent, x, y, z) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); parent.add(m); return m; };
  mesh(G.torso, shirt, body, 0, 1.27, 0);
  mesh(G.shorts, dark, body, 0, 0.98, 0);
  mesh(G.head, skin, body, 0, 1.72, 0);
  mesh(G.hair, hair, body, 0, 1.74, -0.01);
  const limb = (side, isArm) => {
    const top = new THREE.Group(); top.position.set(side * (isArm ? 0.27 : 0.1), isArm ? 1.5 : 0.92, 0); body.add(top);
    mesh(isArm ? G.upper : G.thigh, isArm ? shirt : skin, top, 0, isArm ? -0.15 : -0.19, 0);
    const joint = new THREE.Group(); joint.position.y = isArm ? -0.31 : -0.4; top.add(joint);
    mesh(isArm ? G.fore : G.shin, isArm ? skin : sock, joint, 0, isArm ? -0.14 : -0.2, 0);
    if (!isArm) mesh(G.shoe, shoe, joint, 0, -0.42, 0.05);
    return { top, joint };
  };
  g.userData.legs = [limb(-1, false), limb(1, false)];
  g.userData.arms = [limb(-1, true), limb(1, true)];
  g.userData.body = body; g.userData.shirt = shirt.color; g.userData.sock = sock.color;
  g.userData.phase = Math.random() * 6.28; g.userData.yaw = 0;
  if (!ghost) { const sh = new THREE.Mesh(shadowGeo, shadowMat); sh.rotation.x = -Math.PI / 2; sh.position.y = 0.02; g.add(sh); }
  g.userData.num = makeSprite(false); g.userData.num.position.y = 2.45; g.add(g.userData.num);
  g.userData.name = makeSprite(true); g.userData.name.position.y = 3.2; g.add(g.userData.name);
  const mark = new THREE.Mesh(MARK.cone, MARK.mat); mark.rotation.x = Math.PI; mark.position.y = 4.4; mark.visible = false; mark.renderOrder = 11; g.add(mark);
  const ring = new THREE.Mesh(MARK.ring, MARK.ringMat); ring.rotation.x = -Math.PI / 2; ring.position.y = 0.06; ring.visible = false; g.add(ring);
  g.userData.mark = mark; g.userData.ring = ring;
  body.scale.setScalar(1.3);                                   // a little larger than life, so players read well from the broadcast camera
  scene.add(g);
  return g;
}
// pose a person from his speed (m/s, on the pitch) and the time step: running legs, opposite arms, lean, facing direction
function animate(p, vx, vy, dt) {
  const u = p.userData, sp = Math.hypot(vx, vy);
  if (sp > 0.4) {
    const want = Math.atan2(X(vx) - X(0), Z(vy) - Z(0));
    let dy = want - u.yaw; dy = Math.atan2(Math.sin(dy), Math.cos(dy));
    u.yaw += dy * Math.min(1, dt * 8);
  }
  p.rotation.y = u.yaw;
  u.phase += dt * (2.2 + sp * 1.25) * (sp > 0.3 ? 1 : 0.15);
  const amp = Math.min(0.95, sp * 0.16), s = Math.sin(u.phase);
  u.legs[0].top.rotation.x = amp * s; u.legs[1].top.rotation.x = -amp * s;
  u.legs[0].joint.rotation.x = Math.max(0, amp * 1.3 * Math.sin(u.phase - 1.2)); u.legs[1].joint.rotation.x = Math.max(0, -amp * 1.3 * Math.sin(u.phase - 1.2));
  u.arms[0].top.rotation.x = -amp * 0.9 * s; u.arms[1].top.rotation.x = amp * 0.9 * s;
  u.arms[0].joint.rotation.x = u.arms[1].joint.rotation.x = -0.3 - amp * 0.7;
  u.body.rotation.x = Math.min(0.22, sp * 0.03);
  u.body.position.y = Math.abs(Math.cos(u.phase)) * amp * 0.06;
}
const players = Array.from({ length: 22 }, (_, k) => makePlayer(k < 11 ? HOME : AWAY, false));
// the chosen what-if run drawn as solid players (views 'What-if only' and 'Split'), so nobody appears twice
const simPlayers = Array.from({ length: 22 }, (_, k) => {
  const p = makePlayer(k === 0 ? HOME_GK : k === 11 ? AWAY_GK : k < 11 ? HOME : AWAY, false);
  p.visible = false; p.userData.valid = false; p.userData.lbl = null; return p;
});
const simBall = new THREE.Mesh(new THREE.SphereGeometry(0.35, 20, 14), new THREE.MeshLambertMaterial({ color: 0xffffff, emissive: 0x333333 }));
simBall.visible = false; scene.add(simBall);
const ball = new THREE.Mesh(new THREE.SphereGeometry(0.35, 20, 14), new THREE.MeshLambertMaterial({ color: 0xffffff, emissive: 0x333333 }));
scene.add(ball);
const ballShadow = new THREE.Mesh(shadowGeo, shadowMat); ballShadow.rotation.x = -Math.PI / 2; ballShadow.scale.setScalar(0.6); scene.add(ballShadow);
const TRAIL_N = 31;
const trailGeo = new THREE.BufferGeometry(); trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_N * 3), 3));
const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 }));
scene.add(trail);

// ---------- time helpers ----------
function bsearch(arr, t) {                       // last index with arr[i] <= t
  let lo = 0, hi = arr.length - 1;
  if (t <= arr[0]) return 0;
  if (t >= arr[hi]) return hi;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (arr[m] <= t) lo = m; else hi = m; }
  return lo;
}
function frameAt(tarr, t) {                     // [i, fraction to i+1]; no blending across holes in the data
  const i = bsearch(tarr, t);
  const j = Math.min(i + 1, tarr.length - 1);
  const gap = tarr[j] - tarr[i];
  return [i, gap > 0 && gap < 1.0 ? Math.min(Math.max((t - tarr[i]) / gap, 0), 1) : 0];
}
const lerp = (a, b, f) => (b < 0 || a < 0) ? a : a + (b - a) * f;
function matchClock(t) {
  const c = S.data.clock.filter((x) => x[0] <= t + 1e-6);
  const [start, at] = c.length ? c[c.length - 1] : S.data.clock[0];
  const s = Math.max(0, t - start + at);
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

// ---------- data ----------
async function loadIndex(pick) {
  const r = await fetch('data/index.json', { cache: 'no-store' });
  if (!r.ok) throw new Error('viewer/data/index.json not found: run scripts/export_viewer.py first');
  const idx = await r.json();
  $('file').innerHTML = idx.map((e) => `<option value="${e.file}">${e.title}</option>`).join('');
  const want = (typeof pick === 'string' && pick) || new URLSearchParams(location.search).get('file');
  if (want && idx.some((e) => e.file === want)) $('file').value = want;
  await loadFile($('file').value);
}

async function loadFile(name) {
  msg(`Loading ${name}…`);
  S.playing = false; setPlayIcon();
  const r = await fetch('data/' + name, { cache: 'no-store' });
  if (!r.ok) throw new Error('could not load data/' + name);
  const d = await r.json();
  S.data = d;
  S.t0 = d.t[0]; S.t1 = d.t[d.t.length - 1]; S.t = S.t0;
  S.rosterIdx = -1; S.threatState = {};
  $('home').textContent = d.home; $('away').textContent = d.away;
  $('seek').min = S.t0; $('seek').max = S.t1; $('seek').value = S.t0;
  for (const l of S.layers) { l.meshes.forEach((m) => scene.remove(m)); scene.remove(l.ball); scene.remove(l.links); }
  S.layers = d.layers.map((ld, li) => {
    const col = LAYER_COLORS[li % LAYER_COLORS.length];
    const meshes = Array.from({ length: 22 }, (_, k) => {
      const m = makePlayer(k < 11 ? HOME : AWAY, true);
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.55, 0.75, 20), new THREE.MeshBasicMaterial({ color: col, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2; ring.position.y = 0.04; m.add(ring);
      if (ld.jerseys && ld.jerseys.length === 22) m.userData.num.material.map = labelTexture(String(ld.jerseys[k]));
      return m;
    });
    const b = new THREE.Mesh(new THREE.SphereGeometry(0.35, 16, 12), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.8 }));
    scene.add(b);
    const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(23 * 2 * 3), 3));
    const links = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.7 }));
    scene.add(links);
    return { d: ld, col, meshes, ball: b, links, show: li < 2, showLinks: true, pair: null, threatState: {} };     // many layers: the first two shown
  });
  S.pressState = {};
  const pstep = Math.max(1, Math.round((d.fps || 10) / 2));          // pressing timeline at about 2 values per second
  S.press = pressSeries(d, pstep);
  S.layers.forEach((l) => { l.press = pressSeries(l.d, pstep); });
  buildLayerPanel();
  setupView(d);
  boardOnLoad(d);
  cardFor(-1);
  msg(null);
}

// ---------- compare views ----------
// Who the coach moved in a layer: the layer's own 'moved' list, else the coach board's moves (layer 'Your board'), else
// the names in a 'Coach: <name> ...' layer title. Returned as [{side, shirt}].
function movedFor(ld, d) {
  if (Array.isArray(ld.moved)) return ld.moved;
  const bm = d.card && d.card.board && d.card.board.moves;
  if (bm && ld.name !== 'No change' && /board/i.test(ld.name)) return bm.map((m) => ({ side: m.side, shirt: +m.shirt }));
  const out = [];
  if (/^coach:/i.test(ld.name || '')) {
    for (const side of ['home', 'away']) {
      for (const [shirt, name] of Object.entries((d.names && d.names[side]) || {})) {
        const sur = String(name).trim().split(/\s+/).pop();
        if (sur.length > 2 && ld.name.includes(sur)) out.push({ side, shirt: +shirt });
      }
    }
  }
  return out;
}
function setupView(d) {
  S.layers.forEach((l) => {
    // GRF puts everybody back in one step at a restart (ball out, goal): the run's open play ends there, so the views
    // stop at the frame before instead of showing players jumping tens of metres
    const ld = l.d; l.tStop = ld.t.length ? ld.t[ld.t.length - 1] : 0;
    for (let i = 0; i + 1 < ld.t.length; i++) {
      const dt = ld.t[i + 1] - ld.t[i]; let jumps = 0;
      if (dt <= 0 || dt > 1) continue;
      for (let k = 0; k < 22; k++) {
        const a = i * 22 + k, b = a + 22;
        if (ld.px[a] >= 0 && ld.px[b] >= 0 && Math.hypot(ld.px[b] - ld.px[a], ld.py[b] - ld.py[a]) / DM / dt > 15) jumps++;
      }
      if (jumps >= 6) { l.tStop = ld.t[i]; break; }
    }
    l.moved = movedFor(l.d, d);
    l.movedSlots = new Set();
    l.moved.forEach((m) => {
      const s0 = m.side === 'home' ? 0 : 11;
      for (let k = s0; k < s0 + 11; k++) if (l.d.jerseys && +l.d.jerseys[k] === m.shirt) l.movedSlots.add(k);
    });
  });
  const vl = $('viewLayer'), vm = $('viewMode');
  vl.innerHTML = S.layers.map((l, i) => `<option value="${i}">${l.d.name}</option>`).join('');
  vl.classList.toggle('hidden', S.layers.length < 2);
  vm.classList.toggle('hidden', !S.layers.length);
  let pick = S.layers.findIndex((l) => l.moved.length);
  if (pick < 0) pick = S.layers.length - 1;
  S.view.layer = pick;
  vl.value = String(Math.max(pick, 0));
  S.view.mode = !S.layers.length ? 'real' : (pick >= 0 && S.layers[pick].moved.length) || d.card ? 'split' : 'overlay';
  vm.value = S.view.mode === 'real' && !S.layers.length ? 'real' : S.view.mode;
  simPlayers.forEach((p) => { p.userData.lbl = null; });
}
// the real players the chosen what-if moved (slots in the real roster), so they can be marked too
function realMovedSlots() {
  const vl = S.layers[S.view.layer], out = new Set();
  if (!vl || !S.roster) return out;
  vl.moved.forEach((m) => {
    const s0 = m.side === 'home' ? 0 : 11;
    for (let k = s0; k < s0 + 11; k++) if (+S.roster[k] === m.shirt) out.add(k);
  });
  return out;
}

function buildLayerPanel() {
  const box = $('layers');
  if (!S.layers.length) { box.innerHTML = '<p class="hint">This file has no layers.</p>'; return; }
  box.innerHTML = S.layers.map((l, i) => `
    <div class="layer">
      <label><input type="checkbox" data-l="${i}" data-k="show" ${l.show ? 'checked' : ''}>
        <span class="swatch" style="background:#${l.col.toString(16).padStart(6, '0')}"></span>${l.d.name}</label>
      <label><input type="checkbox" data-l="${i}" data-k="showLinks" checked> lines to the real player</label>
      <div class="err" id="err${i}">&nbsp;</div>
    </div>`).join('');
  box.querySelectorAll('input').forEach((el) => el.addEventListener('change', () => { S.layers[+el.dataset.l][el.dataset.k] = el.checked; }));
}

// pair each ghost with the real player of the same team and shirt (the layer's slot order may differ)
function pairing(layer, roster) {
  const j = layer.d.jerseys;
  return Array.from({ length: 22 }, (_, k) => {
    if (!j || j.length !== 22) return k;
    const side = k < 11 ? 0 : 11;
    for (let s = side; s < side + 11; s++) if (roster[s] === j[k]) return s;
    return -1;
  });
}

// ---------- per-frame update ----------
const v3 = new THREE.Vector3();
let ballPos = new THREE.Vector3();
function update() {
  const d = S.data;
  if (!d) return;
  const [i, f] = frameAt(d.t, S.t);
  const j = Math.min(i + 1, d.t.length - 1);
  // roster (shirt numbers) for this frame
  let ri = 0;
  for (let r = 0; r < d.rosters.length; r++) if (d.rosters[r][0] <= i) ri = r;
  if (ri !== S.rosterIdx) {
    S.rosterIdx = ri; S.roster = d.rosters[ri][1];
    players.forEach((p, k) => {
      const n = S.roster[k];
      p.userData.num.material.map = labelTexture(n >= 0 ? String(n) : '');
      const side = k < 11 ? 'home' : 'away';
      p.userData.name.material.map = labelTexture((d.names[side] || {})[String(n)] || '', true);
      p.userData.num.material.needsUpdate = p.userData.name.material.needsUpdate = true;
      p.userData.shirt.setHex(k === 0 ? HOME_GK : k === 11 ? AWAY_GK : k < 11 ? HOME : AWAY);
    });
    S.layers.forEach((l) => {                       // each ghost carries the name of the real player it is paired with
      l.pair = pairing(l, S.roster);
      l.meshes.forEach((m, k) => {
        const s = l.pair[k], n = s >= 0 ? S.roster[s] : -1;
        m.userData.name.material.map = labelTexture(n >= 0 ? ((d.names[s < 11 ? 'home' : 'away'] || {})[String(n)] || '') : '', true);
        m.userData.name.material.needsUpdate = true;
      });
    });
  }
  const real = [], realVel = [];
  const rm = realMovedSlots();
  for (let k = 0; k < 22; k++) {
    const a = i * 22 + k, b = j * 22 + k;
    const x = d.px[a], y = d.py[a];
    const p = players[k];
    if (x < 0) { p.visible = false; p.userData.valid = false; real.push(null); realVel.push(null); continue; }
    p.visible = true; p.userData.valid = true;
    setMark(p, rm.has(k) && !S.cb.on);
    const xx = lerp(x, d.px[b], f) / DM, yy = lerp(y, d.py[b], f) / DM;
    p.position.set(X(xx), 0, Z(yy));
    const dtd = d.t[j] - d.t[i];
    animate(p, dtd > 0 && dtd < 1 && d.px[b] >= 0 ? (d.px[b] - x) / DM / dtd : 0, dtd > 0 && dtd < 1 && d.px[b] >= 0 ? (d.py[b] - y) / DM / dtd : 0, S.animDt);
    real.push([xx, yy]);
    realVel.push(dtd > 0 && dtd < 1 && d.px[b] >= 0 ? [(d.px[b] - x) / DM / dtd, (d.py[b] - y) / DM / dtd] : [0, 0]);
    p.userData.num.visible = S.showNumbers;
    p.userData.name.visible = S.showNames || (rm.has(k) && !S.cb.on);
  }
  if (S.cb.on) boardOverride(real);
  const bx = lerp(d.bx[i], d.bx[j], f) / DM, by = lerp(d.by[i], d.by[j], f) / DM, bz = Math.max(0, lerp(d.bz[i], d.bz[j], f) / DM);
  ball.position.set(X(bx), 0.35 + bz, Z(by));
  ballShadow.position.set(X(bx), 0.02, Z(by));
  ballPos.copy(ball.position);
  // trail: the last 3 s of the ball
  trail.visible = S.showTrail;
  if (S.showTrail) {
    const pos = trailGeo.attributes.position;
    for (let n = 0; n < TRAIL_N; n++) {
      const [ti, tf] = frameAt(d.t, S.t - (TRAIL_N - 1 - n) * 0.1);
      const tj = Math.min(ti + 1, d.t.length - 1);
      pos.setXYZ(n, X(lerp(d.bx[ti], d.bx[tj], tf) / DM), 0.3, Z(lerp(d.by[ti], d.by[tj], tf) / DM));
    }
    pos.needsUpdate = true;
  }
  // board
  $('score').textContent = `${d.sh[i]} - ${d.sa[i]}`;
  $('clock').textContent = matchClock(S.t);
  const live = d.live[i] === 1;
  $('live').textContent = live ? 'in play' : 'ball out';
  $('live').className = 'pill' + (live ? '' : ' dead');
  const th = threatOf(real, [bx, by], S.threatState);
  $('threat').textContent = th ? `threat ${pct(th.p)}% (${d[th.team] || th.team})` : 'threat –';
  $('threat').className = 'pill threat' + (th && th.p >= 0.2 ? ' hot' : '');
  const bdt = d.t[j] - d.t[i];
  const pr = pressOf(real, realVel, [bx, by], bdt > 0 && bdt < 1 ? [(d.bx[j] - d.bx[i]) / DM / bdt, (d.by[j] - d.by[i]) / DM / bdt] : [0, 0], S.pressState);
  $('press').textContent = pr ? `press ${Math.round(100 * pr.p)}% (${d[pr.team] || pr.team})` : 'press –';
  $('press').className = 'pill threat' + (pr && pr.p >= 0.5 ? ' press-hot' : '');
  drawPressLine();
  $('tlabel').textContent = `${fmt(S.t - S.t0)} / ${fmt(S.t1 - S.t0)}`;
  if (document.activeElement !== $('seek')) $('seek').value = S.t;
  // ghost layers
  S.layers.forEach((l, li) => {
    const ld = l.d;
    const inside = ld.t.length && S.t >= ld.t[0] - 0.05 && S.t <= (l.tStop ?? ld.t[ld.t.length - 1]) + 0.05;
    const on = l.show && inside && !S.cb.on && S.view.mode === 'overlay';   // the board shows the real players only; other views draw no ghosts
    l.meshes.forEach((m) => { m.visible = false; });
    l.ball.visible = false; l.links.visible = false;
    if (!inside) { $('err' + li).textContent = 'not in this part of the match'; return; }
    const [gi, gf] = frameAt(ld.t, S.t);
    const gj = Math.min(gi + 1, ld.t.length - 1);
    const lp = l.links.geometry.attributes.position;
    let sum = 0, cnt = 0, seg = 0;
    const gpts = Array(22).fill(null);
    for (let k = 0; k < 22; k++) {
      const a = gi * 22 + k, b = gj * 22 + k;
      if (ld.px[a] < 0) continue;
      const xx = lerp(ld.px[a], ld.px[b], gf) / DM, yy = lerp(ld.py[a], ld.py[b], gf) / DM;
      const m = l.meshes[k];
      const s0 = l.pair ? l.pair[k] : k;
      if (s0 >= 0) gpts[s0] = [xx, yy];
      m.visible = on; m.position.set(X(xx), 0, Z(yy));
      const gdt = ld.t[gj] - ld.t[gi];
      if (on) animate(m, gdt > 0 && gdt < 1 && ld.px[b] >= 0 ? (ld.px[b] - ld.px[a]) / DM / gdt : 0, gdt > 0 && gdt < 1 && ld.px[b] >= 0 ? (ld.py[b] - ld.py[a]) / DM / gdt : 0, S.animDt);
      setMark(m, on && l.movedSlots.has(k));
      m.userData.num.visible = on && S.showNumbers; m.userData.name.visible = on && (S.showNames || l.movedSlots.has(k));
      const s = l.pair ? l.pair[k] : k;
      if (s >= 0 && real[s]) {
        sum += Math.hypot(xx - real[s][0], yy - real[s][1]); cnt++;
        lp.setXYZ(seg++, X(xx), 0.1, Z(yy)); lp.setXYZ(seg++, X(real[s][0]), 0.1, Z(real[s][1]));
      }
    }
    const gbx = lerp(ld.bx[gi], ld.bx[gj], gf) / DM, gby = lerp(ld.by[gi], ld.by[gj], gf) / DM;
    l.ball.visible = on; l.ball.position.set(X(gbx), 0.35, Z(gby));
    lp.setXYZ(seg++, X(gbx), 0.35, Z(gby)); lp.setXYZ(seg++, X(bx), 0.35, Z(by));
    for (let n = seg; n < 46; n++) lp.setXYZ(n, 0, -10, 0);
    lp.needsUpdate = true;
    l.links.geometry.setDrawRange(0, seg);
    l.links.visible = on && l.showLinks;
    const gt = threatOf(gpts, [gbx, gby], l.threatState);
    const gp = seriesAt(l.press, S.t);
    $('err' + li).textContent = `now: players ${cnt ? (sum / cnt).toFixed(1) : '–'} m, ball ${Math.hypot(gbx - bx, gby - by).toFixed(1)} m from the real ones`
      + (gt ? ` · threat ${pct(gt.p)}% (${d[gt.team] || gt.team})` : '')
      + (gp ? ` · press ${Math.round(100 * gp.p)}% (${d[gp.team] || gp.team})` : '');
  });
  updateSim(d);
}

// ---------- pressing meter (press.js): timeline of the real match and of the shown what-if runs ----------
function seriesAt(s, t) {
  if (!s || !s.length || t < s[0].t - 1 || t > s[s.length - 1].t + 1) return null;
  let lo = 0, hi = s.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (s[m].t <= t) lo = m; else hi = m; }
  return s[lo];
}
const hex = (c) => '#' + c.toString(16).padStart(6, '0');
function drawPressLine() {
  const cv = $('pressLine'), d = S.data;
  if (!cv || !d || !S.press) return;
  const w = cv.clientWidth, h = cv.clientHeight, dpr = Math.min(window.devicePixelRatio, 2);
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const span = Math.max(S.t1 - S.t0, 1e-6), X_ = (t) => ((t - S.t0) / span) * w, Y_ = (p) => h - 2 - p * (h - 4);
  g.strokeStyle = 'rgba(255,255,255,0.12)'; g.beginPath(); g.moveTo(0, Y_(0.5)); g.lineTo(w, Y_(0.5)); g.stroke();
  // mean per 2-pixel column, so a full match stays readable; byTeam colours each piece by the pressing team (home red, away blue)
  const line = (s, col, width, byTeam) => {
    if (!s || !s.length) return;
    const nb = Math.max(1, Math.floor(w / 2)), sum = new Float32Array(nb), cnt = new Uint16Array(nb), home = new Uint16Array(nb);
    for (const q of s) {
      const b = Math.min(nb - 1, Math.max(0, Math.floor(((q.t - S.t0) / span) * nb)));
      sum[b] += q.p; cnt[b]++; if (q.team === 'home') home[b]++;
    }
    g.lineWidth = width;
    let prev = null, prevB = -1;
    const gapBins = Math.max(2, Math.ceil((3 / span) * nb));        // break the line only at holes longer than 3 s
    for (let b = 0; b < nb; b++) {
      if (!cnt[b]) continue;
      const pt = [(b + 0.5) * (w / nb), Y_(sum[b] / cnt[b])];
      if (prev && b - prevB > gapBins) prev = null;
      prevB = b;
      if (prev) {
        g.strokeStyle = byTeam ? (home[b] * 2 >= cnt[b] ? '#ff6b5e' : '#5d9bff') : col;
        g.beginPath(); g.moveTo(prev[0], prev[1]); g.lineTo(pt[0], pt[1]); g.stroke();
      }
      prev = pt;
    }
  };
  S.layers.forEach((l, li) => { if (l.show || li === S.view.layer) line(l.press, hex(l.col), 1.2); });
  line(S.press, null, 1.6, true);
  g.fillStyle = '#f2c14e'; g.fillRect(X_(S.t) - 1, 0, 2, h);
}

// the chosen what-if run as solid players with real shirt colours and numbers; its moved players carry the marker
function updateSim(d) {
  const vl = S.layers[S.view.layer];
  const ld = vl && vl.d;
  // from its start to the end of its data the run is shown; after a restart (tStop) it stays on its last open-play frame
  S.view.inside = !!ld && ld.t.length > 0 && S.t >= ld.t[0] - 0.05 && S.t <= ld.t[ld.t.length - 1] + 0.05;
  S.view.ended = S.view.inside && S.t > vl.tStop;
  simPlayers.forEach((p) => { p.userData.valid = false; });
  if (!S.view.inside || S.view.mode === 'overlay' || S.view.mode === 'real') return;
  const tt = Math.min(S.t, vl.tStop), still = S.view.ended;
  const [gi, gf] = frameAt(ld.t, tt);
  const gj = Math.min(gi + 1, ld.t.length - 1), gdt = ld.t[gj] - ld.t[gi];
  for (let k = 0; k < 22; k++) {
    const a = gi * 22 + k, b = gj * 22 + k, p = simPlayers[k];
    if (ld.px[a] < 0) continue;
    p.userData.valid = true;
    const xx = lerp(ld.px[a], ld.px[b], gf) / DM, yy = lerp(ld.py[a], ld.py[b], gf) / DM;
    p.position.set(X(xx), 0, Z(yy));
    const ok = !still && gdt > 0 && gdt < 1 && ld.px[b] >= 0;
    animate(p, ok ? (ld.px[b] - ld.px[a]) / DM / gdt : 0, ok ? (ld.py[b] - ld.py[a]) / DM / gdt : 0, S.animDt);
    const shirt = ld.jerseys && ld.jerseys.length === 22 ? ld.jerseys[k] : (S.roster ? S.roster[k] : -1);
    const side = k < 11 ? 'home' : 'away', moved = vl.movedSlots.has(k);
    const key = `${shirt}|${side}`;
    if (p.userData.lbl !== key) {
      p.userData.lbl = key;
      p.userData.num.material.map = labelTexture(shirt >= 0 ? String(shirt) : '');
      p.userData.name.material.map = labelTexture(((d.names && d.names[side]) || {})[String(shirt)] || '', true);
      p.userData.num.material.needsUpdate = p.userData.name.material.needsUpdate = true;
    }
    p.userData.num.visible = S.showNumbers;
    p.userData.name.visible = S.showNames || moved;
    setMark(p, moved);
  }
  simBall.position.set(X(lerp(ld.bx[gi], ld.bx[gj], gf) / DM), 0.35, Z(lerp(ld.by[gi], ld.by[gj], gf) / DM));
}

// which set of people a picture shows: 'real' (the data), 'sim' (the chosen what-if run) or 'overlay' (real + ghosts)
function showSet(which) {
  const sim = which === 'sim';
  players.forEach((p) => { p.visible = !sim && p.userData.valid; });
  ball.visible = ballShadow.visible = !sim;
  trail.visible = !sim && S.showTrail;
  simPlayers.forEach((p) => { p.visible = sim && p.userData.valid; });
  simBall.visible = sim;
}
function renderView() {
  const w = window.innerWidth, h = window.innerHeight;
  const mode = S.cb.on ? 'real' : S.view.mode;
  const split = mode === 'split' && S.layers.length > 0;
  const rightSim = S.view.inside;
  $('splitLine').classList.toggle('hidden', !split);
  $('capL').classList.toggle('hidden', !split);
  $('capR').classList.toggle('hidden', !split);
  if (split) {
    const vl = S.layers[S.view.layer];
    const names = vl ? vl.moved.map((m) => ((S.data.names && S.data.names[m.side]) || {})[String(m.shirt)] || `#${m.shirt}`) : [];
    const mv = names.length ? ` · <span class="mv">&#9660; moved: ${names.join(', ')}</span>` : '';
    const wait = vl && !rightSim && vl.d.t.length && S.t < vl.d.t[0] ? ` · starts in ${(vl.d.t[0] - S.t).toFixed(1)} s` : '';
    const end = S.view.ended ? ` · <span class="mv">ball out / restart: run ends here</span>` : '';
    $('capR').innerHTML = `WHAT-IF · ${vl ? vl.d.name : ''}${mv}${wait}${end}`;
    $('capL').innerHTML = `REAL MATCH${names.length ? ` · <span class="mv" style="color:#ffd84d">&#9660; ${names.join(', ')}</span>` : ''}`;
    renderer.setScissorTest(true);
    camera.aspect = (w / 2) / h; camera.updateProjectionMatrix();
    showSet('real'); renderer.setViewport(0, 0, w / 2, h); renderer.setScissor(0, 0, w / 2, h); renderer.render(scene, camera);
    const now = performance.now(), dt = Math.min((now - (S.cam2t || now)) / 1000, 0.1); S.cam2t = now;
    if (rightSim && S.cam !== 'free') {
      placeCameraOn(simBall.position.clone().setY(ballPos.y), camPos2, camLook2, S.cam2dt ?? dt, false);
      camera.position.copy(camPos2); camera.lookAt(camLook2);
    } else { camPos2.copy(camPos); camLook2.copy(camLook); }
    showSet(rightSim ? 'sim' : 'real'); renderer.setViewport(w / 2, 0, w / 2, h); renderer.setScissor(w / 2, 0, w / 2, h); renderer.render(scene, camera);
    camera.position.copy(camPos); camera.lookAt(camLook);
    renderer.setScissorTest(false);
    showSet('real');
  } else {
    camera.aspect = w / h; camera.updateProjectionMatrix();
    showSet(mode === 'sim' && rightSim ? 'sim' : 'real');
    renderer.setViewport(0, 0, w, h); renderer.render(scene, camera);
  }
}

// ---------- cameras ----------
const camPos = new THREE.Vector3(0, 40, 60), camLook = new THREE.Vector3();
// the right half of the split view has its own smooth camera that follows the what-if ball (its play can go elsewhere)
const camPos2 = new THREE.Vector3(0, 40, 60), camLook2 = new THREE.Vector3();
function placeCameraOn(bp, cp, cl, dt, snap) {
  const keep = ballPos.clone(), p0 = camPos.clone(), l0 = camLook.clone();
  ballPos.copy(bp); camPos.copy(cp); camLook.copy(cl);
  placeCamera(dt, snap);
  cp.copy(camPos); cl.copy(camLook);
  ballPos.copy(keep); camPos.copy(p0); camLook.copy(l0);
}
function placeCamera(dt, snap) {
  if (S.cam === 'free') { controls.update(); return; }
  const target = new THREE.Vector3(), pos = new THREE.Vector3();
  camera.up.set(0, 1, 0);
  if (S.cam === 'broadcast') {
    target.set(ballPos.x * 0.85, 0, ballPos.z * 0.35);
    pos.set(target.x * 0.9, 34, 58);
  } else if (S.cam === 'top') {
    const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const h = Math.max((W / 2 + 6) / tan, (L / 2 + 6) / (tan * camera.aspect));
    camera.up.set(0, 0, -1);
    target.set(0, 0, 0); pos.set(0, h, 0.001);
  } else if (S.cam === 'behind') {
    target.set(0, 0, 0); pos.set(-L / 2 - 30, 26, 0);
  } else if (S.cam === 'follow') {
    target.copy(ballPos).setY(0);
    pos.set(ballPos.x - 18, 16, ballPos.z + 22);
  }
  const k = snap ? 1 : 1 - Math.exp(-dt * 3);           // smooth camera, no jumps
  camPos.lerp(pos, k); camLook.lerp(target, k);
  camera.position.copy(camPos); camera.lookAt(camLook);
}
function setCam(c) {
  S.cam = c;
  document.querySelectorAll('#cams button').forEach((b) => b.classList.toggle('on', b.dataset.cam === c));
  controls.enabled = c === 'free';
  if (c === 'free') { controls.target.copy(camLook); camera.up.set(0, 1, 0); }
  else placeCamera(0, true);
}

// ---------- coach board: drag players, press Play, GRF plays it out (scripts/viewer_server.py + coach_board.py) ----------
// Boards (as in coach_board.py): defence = keeper + 4 deepest outfield players, midfield = next 3, forward = 3 most advanced.
const CB = S.cb;
const cbRings = players.map((p) => {
  const r = new THREE.Mesh(new THREE.RingGeometry(0.8, 1.05, 24), new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }));
  r.rotation.x = -Math.PI / 2; r.position.y = 0.05; r.visible = false; p.add(r);
  return r;
});
const cbLinesGeo = new THREE.BufferGeometry(); cbLinesGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(22 * 2 * 3), 3));
const cbLines = new THREE.LineSegments(cbLinesGeo, new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.8, gapSize: 0.5 }));
cbLines.visible = false; scene.add(cbLines);
const cbGhosts = players.map((p, k) => { const g = makePlayer(k < 11 ? HOME : AWAY, true); g.visible = false; return g; });   // where he really was

function realNow() {                                   // real positions (m) at the board's frame, from the data
  const d = S.data, [i] = frameAt(d.t, CB.t);
  return Array.from({ length: 22 }, (_, k) => (d.px[i * 22 + k] < 0 ? null : [d.px[i * 22 + k] / DM, d.py[i * 22 + k] / DM]));
}
function rosterNow() {                                 // shirt numbers per slot at the board's frame (no drawn frame needed)
  const d = S.data, [i] = frameAt(d.t, CB.t);
  let ro = d.rosters[0][1];
  for (const r of d.rosters) if (r[0] <= i) ro = r[1];
  return ro;
}
function boardUnits(pts, team) {
  const s0 = team === 'home' ? 0 : 11, keeper = pts[s0];
  const ownGoal = keeper && keeper[0] > L / 2 ? L : 0;
  const out = []; for (let s = s0 + 1; s < s0 + 11; s++) if (pts[s]) out.push(s);
  out.sort((a, b) => Math.abs(pts[a][0] - ownGoal) - Math.abs(pts[b][0] - ownGoal));
  return { defence: [s0, ...out.slice(0, 4)], midfield: out.slice(4, 7), forward: out.slice(7) };
}
function boardAllowed() {
  CB.allowed = boardUnits(realNow(), CB.team)[CB.board] || [];
  for (const k of Object.keys(CB.moves)) if (!CB.allowed.includes(+k)) delete CB.moves[k];
  cbRings.forEach((r, k) => { r.visible = CB.on && CB.allowed.includes(k); });
  boardStatus();
}
function boardOverride(real) {
  const lp = cbLinesGeo.attributes.position; let n = 0;
  cbGhosts.forEach((g) => { g.visible = false; });
  for (const [ks, xy] of Object.entries(CB.moves)) {
    const k = +ks, p = players[k];
    if (!real[k]) continue;
    const [x0, y0] = real[k];
    cbGhosts[k].visible = true; cbGhosts[k].position.set(X(x0), 0, Z(y0));
    p.position.set(X(xy[0]), 0, Z(xy[1]));
    real[k] = xy;
    lp.setXYZ(n++, X(x0), 0.08, Z(y0)); lp.setXYZ(n++, X(xy[0]), 0.08, Z(xy[1]));
  }
  for (let m = n; m < 44; m++) lp.setXYZ(m, 0, -10, 0);
  lp.needsUpdate = true; cbLinesGeo.setDrawRange(0, n); cbLines.computeLineDistances(); cbLines.visible = n > 0;
}
function boardStatus(extra) {
  const moved = Object.keys(CB.moves).length;
  const names = (S.data && S.data.names && S.data.names[CB.team]) || {};
  const ro = S.data && CB.on ? rosterNow() : [];
  const who = CB.allowed.map((k) => names[String(ro[k])] || `#${ro[k]}`).join(', ');
  let t = CB.on ? `Board at ${matchClock(CB.t)}: drag a ringed player (${who}). ${moved} moved.` : 'Pick a team and board, pause at the moment, then Start board here.';
  if (CB.on && moved) {
    const th = threatOf(realNow().map((p, k) => (CB.moves[k] || p)), ballNow(), { team: null });
    if (CB.threat0 && th) t += ` Threat now ${pct(th.p)}% (was ${pct(CB.threat0.p)}%).`;
  }
  $('cbStatus').textContent = extra || t;
  $('cbPlay').disabled = !(CB.on && moved);
}
function ballNow() { const d = S.data, [i] = frameAt(d.t, CB.t); return [d.bx[i] / DM, d.by[i] / DM]; }
function boardStart() {
  if (!S.data || !S.data.match) { boardStatus('The board needs a real match file (modes 1, 2, 4, 5 or a match).'); return; }
  S.playing = false; setPlayIcon();
  const [i] = frameAt(S.data.t, S.t);
  CB.t = S.data.t[i]; CB.on = true; CB.moves = {};
  CB.threat0 = threatOf(realNow(), ballNow(), { team: null });
  setCam('top'); boardAllowed();
}
function boardReset() {
  CB.on = false; CB.moves = {}; CB.drag = -1; cbLines.visible = false;
  cbGhosts.forEach((g) => { g.visible = false; }); cbRings.forEach((r) => { r.visible = false; });
  boardStatus();
}
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), ground = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), hit = new THREE.Vector3();
function pitchPoint(e) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  if (!ray.ray.intersectPlane(ground, hit)) return null;
  return [Math.min(Math.max(hit.x + L / 2, 0.5), L - 0.5), Math.min(Math.max(W / 2 - hit.z, 0.5), W - 0.5)];
}
renderer.domElement.addEventListener('pointerdown', (e) => {
  if (!CB.on) return;
  const p = pitchPoint(e); if (!p) return;
  const now = realNow();
  let best = -1, bd = 3.0;
  for (const k of CB.allowed) {
    const q = CB.moves[k] || now[k]; if (!q) continue;
    const dd = Math.hypot(q[0] - p[0], q[1] - p[1]); if (dd < bd) { bd = dd; best = k; }
  }
  if (best >= 0) { CB.drag = best; renderer.domElement.setPointerCapture(e.pointerId); e.preventDefault(); }
});
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!CB.on || CB.drag < 0) return;
  const p = pitchPoint(e); if (p) { CB.moves[CB.drag] = p; boardStatus(); }
});
renderer.domElement.addEventListener('pointerup', () => { CB.drag = -1; });

// ---------- player cards (FootballBERT, viewer/data/player_cards.json from scripts/export_fbert.py) ----------
const normName = (s) => String(s || '').normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/[^a-z ]/g, '').trim();
async function loadCards() {
  try {
    const r = await fetch('data/player_cards.json', { cache: 'no-store' });
    if (!r.ok) return;
    S.cards = await r.json();
    S.cards.byName = {};
    for (const [id, c] of Object.entries(S.cards.cards)) S.cards.byName[normName(c.name)] = id;
  } catch (e) { S.cards = null; }
}
function cardFor(k) {
  const box = $('pcard'); if (!box) return;
  const d = S.data;
  if (k < 0 || !d || !S.roster) { box.innerHTML = '<p class="hint">Click a player on the pitch.</p>'; return; }
  const side = k < 11 ? 'home' : 'away', shirt = S.roster[k];
  const name = ((d.names || {})[side] || {})[String(shirt)] || `#${shirt}`;
  const C = S.cards;
  const id = C && ((((C.matches || {})[d.match] || {})[side] || {})[String(shirt)] || C.byName[normName(name)]);
  const c = id && C.cards[id];
  if (!c) {
    box.innerHTML = `<div class="pname">${name} <span class="hint">#${shirt} · ${d[side] || side}</span></div>
      <p class="hint">FootballBERT knows World Cup 2022 players by name; this player is not matched (Bundesliga files only carry short names).</p>`;
    return;
  }
  box.innerHTML = `<div class="pname">${c.name} <span class="hint">#${shirt} · ${c.team}</span></div>
    <div class="prole">${c.role || '?'} · ${c.line || ''}</div>
    <div class="hint">Most similar World Cup 2022 players (FootballBERT):</div>
    <table>${c.similar.map((s) => `<tr><td>${s.name}</td><td class="hint">${s.team}</td><td>${s.sim.toFixed(2)}</td></tr>`).join('')}</table>`;
}
let downAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('click', (e) => {
  if (CB.on || !downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;   // a drag, not a click
  const p = pitchPoint(e); if (!p) return;
  let best = -1, bd = 2.5;
  players.forEach((q, k) => {
    if (!q.visible || !q.userData.valid) return;
    const dd = Math.hypot(q.position.x + L / 2 - p[0], W / 2 - q.position.z - p[1]); if (dd < bd) { bd = dd; best = k; }
  });
  if (best >= 0) { cardFor(best); if ($('panel').classList.contains('hidden')) togglePanel(); }
});
async function boardPlay() {
  const d = S.data;
  const ro = rosterNow();
  const moves = Object.entries(CB.moves).map(([k, xy]) => ({ side: +k < 11 ? 'home' : 'away', shirt: ro[+k], x: +xy[0].toFixed(2), y: +xy[1].toFixed(2) }));
  const board = { match: d.match, t: CB.t, board: CB.board, mode: $('cbMode').value, team: CB.team, moves };
  let r;
  try { r = await fetch('api/board', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(board) }); } catch (e) { r = null; }
  if (!r || !r.ok) {                                   // plain http.server: save the board, run it by hand
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(board, null, 1)], { type: 'application/json' }));
    a.download = 'coach_board.json'; a.click();
    boardStatus('Saved coach_board.json (Downloads). Run: .venv\\Scripts\\python.exe scripts\\coach_board.py --board <that file>. Or start scripts\\viewer_server.py so Play runs it for you.');
    return;
  }
  const { id } = await r.json();
  $('cbPlay').disabled = true;
  const poll = async () => {
    const s = await (await fetch(`api/board/${id}`, { cache: 'no-store' })).json();
    if (s.state === 'running') { boardStatus(`GRF is playing your board… ${s.line}`); setTimeout(poll, 2000); return; }
    if (s.state === 'failed') { boardStatus(`GRF run failed: ${s.line} (see outputs/coach_board/${id}.log)`); return; }
    boardReset();
    await loadIndex(s.file);
    setCam('broadcast');
  };
  poll();
}
function boardOnLoad(d) {
  boardReset();
  $('cbTeam').innerHTML = `<option value="home">${d.home || 'Home'}</option><option value="away">${d.away || 'Away'}</option>`;
  $('cbTeam').value = CB.team;
  const c = d.card;
  if (!c) { $('cbCard').innerHTML = ''; return; }
  const f = (r, v) => (r.what.startsWith('Threat') || r.what === 'Max threat' ? `${pct(v)}%` : (r.what === 'Ball lost within 12 s' || r.what === 'Time on the ball') ? `${Math.round(v * 100)}%` : v.toFixed(2));
  $('cbCard').innerHTML = `<div class="what">${c.what} · ${c.runs} runs each · ${c.team_on_board} on the board (${c.attacking ? 'attacking' : 'defending'})</div>
    <table><tr><th></th><th>No change</th><th>Your board</th><th>Diff</th></tr>${c.rows.map((r) => `<tr><td>${r.what}</td><td>${f(r, r.no_change)}</td>
    <td>${f(r, r.board)}</td><td class="${r.verdict === 'better' ? 'better' : r.verdict === 'worse' ? 'worse' : ''}">${r.diff >= 0 ? '+' : ''}${f(r, r.diff)} ${r.verdict === 'better' ? '✅' : r.verdict === 'worse' ? '❌' : '·'}</td></tr>`).join('')}</table>
    <div class="hint">✅/❌ = sure at 95%; · = no clear effect. Ghosts: No change, Your board.</div>`;
}
$('cbTeam').addEventListener('change', (e) => { CB.team = e.target.value; if (CB.on) { CB.moves = {}; boardAllowed(); } });
$('cbBoards').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
  CB.board = b.dataset.b;
  $('cbBoards').querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
  if (CB.on) boardAllowed();
}));
$('cbStart').addEventListener('click', boardStart);
$('cbReset').addEventListener('click', boardReset);
$('cbPlay').addEventListener('click', () => boardPlay().catch(fail));

// ---------- UI ----------
function msg(t) { $('msg').textContent = t || ''; $('msg').classList.toggle('hidden', !t); }
function setPlayIcon() { $('play').innerHTML = S.playing ? '&#10074;&#10074;' : '&#9654;'; }
function seek(t) { S.t = Math.min(Math.max(t, S.t0), S.t1); }
$('play').addEventListener('click', () => { if (S.t >= S.t1) S.t = S.t0; S.playing = !S.playing; setPlayIcon(); });
$('seek').addEventListener('input', (e) => seek(+e.target.value));
$('speed').addEventListener('change', (e) => { S.speed = +e.target.value; });
$('file').addEventListener('change', (e) => loadFile(e.target.value).catch(fail));
$('optNumbers').addEventListener('change', (e) => { S.showNumbers = e.target.checked; });
$('optNames').addEventListener('change', (e) => { S.showNames = e.target.checked; });
$('optTrail').addEventListener('change', (e) => { S.showTrail = e.target.checked; });
$('viewMode').addEventListener('change', (e) => { S.view.mode = e.target.value; });
$('viewLayer').addEventListener('change', (e) => { S.view.layer = +e.target.value; });
const VIEW_MODES = ['split', 'sim', 'real', 'overlay'];
document.querySelectorAll('#cams button').forEach((b) => b.addEventListener('click', () => setCam(b.dataset.cam)));
const togglePanel = () => { $('panel').classList.toggle('hidden'); $('togglePanel').classList.toggle('on', !$('panel').classList.contains('hidden')); };
$('togglePanel').addEventListener('click', togglePanel);
$('togglePanel').classList.add('on');
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT') return;
  const step = S.data ? 1 / S.data.fps : 0.1;
  if (e.code === 'Space') { e.preventDefault(); $('play').click(); }
  else if (e.key === 'ArrowRight') seek(S.t + (e.shiftKey ? 10 : 1));
  else if (e.key === 'ArrowLeft') seek(S.t - (e.shiftKey ? 10 : 1));
  else if (e.key === '.') { S.playing = false; setPlayIcon(); seek(S.t + step); }
  else if (e.key === ',') { S.playing = false; setPlayIcon(); seek(S.t - step); }
  else if (e.key === 'h' || e.key === 'H') togglePanel();
  else if ((e.key === 'v' || e.key === 'V') && S.layers.length) {
    S.view.mode = VIEW_MODES[(VIEW_MODES.indexOf(S.view.mode) + 1) % VIEW_MODES.length]; $('viewMode').value = S.view.mode;
  }
  else if ('12345'.includes(e.key) && e.key.length === 1) setCam(['broadcast', 'top', 'behind', 'follow', 'free'][+e.key - 1]);
});
function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();
function fail(err) { console.error(err); msg('Error: ' + err.message); }

let last = performance.now();
function loop(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (S.cb.on) { S.t = S.cb.t; S.playing = false; }          // the board is a still picture of one moment
  if (S.playing && S.data) {
    S.t += dt * S.speed;
    if (S.t >= S.t1) { S.t = S.t1; S.playing = false; setPlayIcon(); }
  }
  S.animDt = S.playing ? dt * S.speed : 0;                 // limbs move only while the match plays
  const w0 = performance.now();
  update();
  placeCamera(dt, false);
  renderView();
  S.frameMs = 0.95 * (S.frameMs || 0) + 0.05 * (performance.now() - w0);   // work per frame (ms), for checking speed
  requestAnimationFrame(loop);
}
window.__viewer = S;                                     // for debugging in the browser console
window.__viewerPlayers = players;
window.__viewerStep = () => { update(); placeCamera(0.016, false); renderView(); };
Promise.all([loadThreat(), loadCards()]).then(() => loadIndex()).then(() => setCam('broadcast')).catch(fail);
requestAnimationFrame(loop);
