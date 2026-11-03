// Match replay viewer: draws a real match exactly as recorded (22 players + ball from tracking data), and other runs of the
// same stretch (GRF replays, later what-if simulations) as ghost layers next to it. Data files come from scripts/export_viewer.py.
//
// Pitch coordinates in the data: metres x 0..105 (home attacks towards 105), y 0..68, stored as decimetres.
// Three.js scene: X = x - 52.5, Z = 34 - y, Y = up.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const L = 105, W = 68, DM = 10;
const HOME = 0xe0473c, AWAY = 0x3a7bd5, HOME_GK = 0xff9f1c, AWAY_GK = 0x8fd14f;
const LAYER_COLORS = [0xf2c14e, 0xc77dff, 0x5ce1e6, 0xff7aa2, 0x9be564, 0xff9f43, 0xffffff];

const $ = (id) => document.getElementById(id);
const S = { data: null, t: 0, t0: 0, t1: 1, playing: false, speed: 1, cam: 'broadcast', roster: null, rosterIdx: -1,
            showNumbers: true, showNames: false, showTrail: true, layers: [] };

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
const bodyGeo = new THREE.CylinderGeometry(0.42, 0.42, 1.5, 16);
const headGeo = new THREE.SphereGeometry(0.28, 16, 12);
const shadowGeo = new THREE.CircleGeometry(0.6, 16);
const shadowMat = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 });
function makePlayer(color, ghost) {
  const g = new THREE.Group();
  const mat = ghost ? new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.45 }) : new THREE.MeshLambertMaterial({ color });
  const body = new THREE.Mesh(bodyGeo, mat); body.position.y = 0.75; g.add(body);
  const head = new THREE.Mesh(headGeo, ghost ? mat : new THREE.MeshLambertMaterial({ color: 0xf1c7a1 })); head.position.y = 1.78; g.add(head);
  if (!ghost) { const sh = new THREE.Mesh(shadowGeo, shadowMat); sh.rotation.x = -Math.PI / 2; sh.position.y = 0.02; g.add(sh); }
  if (ghost) g.scale.setScalar(0.85);
  g.userData.num = makeSprite(false); g.userData.num.position.y = 2.7; g.add(g.userData.num);
  g.userData.name = makeSprite(true); g.userData.name.position.y = 3.6; g.add(g.userData.name);
  scene.add(g);
  return g;
}
const players = Array.from({ length: 22 }, (_, k) => makePlayer(k < 11 ? HOME : AWAY, false));
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
async function loadIndex() {
  const r = await fetch('data/index.json', { cache: 'no-store' });
  if (!r.ok) throw new Error('viewer/data/index.json not found: run scripts/export_viewer.py first');
  const idx = await r.json();
  $('file').innerHTML = idx.map((e) => `<option value="${e.file}">${e.title}</option>`).join('');
  const want = new URLSearchParams(location.search).get('file');
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
  S.rosterIdx = -1;
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
    return { d: ld, col, meshes, ball: b, links, show: li < 2, showLinks: true, pair: null };     // many layers: the first two shown
  });
  buildLayerPanel();
  msg(null);
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
      p.children[0].material.color.setHex(k === 0 ? HOME_GK : k === 11 ? AWAY_GK : k < 11 ? HOME : AWAY);
    });
    S.layers.forEach((l) => { l.pair = pairing(l, S.roster); });
  }
  const real = [];
  for (let k = 0; k < 22; k++) {
    const a = i * 22 + k, b = j * 22 + k;
    const x = d.px[a], y = d.py[a];
    const p = players[k];
    if (x < 0) { p.visible = false; real.push(null); continue; }
    p.visible = true;
    const xx = lerp(x, d.px[b], f) / DM, yy = lerp(y, d.py[b], f) / DM;
    p.position.set(X(xx), 0, Z(yy));
    real.push([xx, yy]);
    p.userData.num.visible = S.showNumbers;
    p.userData.name.visible = S.showNames;
  }
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
  $('tlabel').textContent = `${fmt(S.t - S.t0)} / ${fmt(S.t1 - S.t0)}`;
  if (document.activeElement !== $('seek')) $('seek').value = S.t;
  // ghost layers
  S.layers.forEach((l, li) => {
    const ld = l.d;
    const inside = ld.t.length && S.t >= ld.t[0] - 0.05 && S.t <= ld.t[ld.t.length - 1] + 0.05;
    const on = l.show && inside;
    l.meshes.forEach((m) => { m.visible = false; });
    l.ball.visible = false; l.links.visible = false;
    if (!inside) { $('err' + li).textContent = 'not in this part of the match'; return; }
    const [gi, gf] = frameAt(ld.t, S.t);
    const gj = Math.min(gi + 1, ld.t.length - 1);
    const lp = l.links.geometry.attributes.position;
    let sum = 0, cnt = 0, seg = 0;
    for (let k = 0; k < 22; k++) {
      const a = gi * 22 + k, b = gj * 22 + k;
      if (ld.px[a] < 0) continue;
      const xx = lerp(ld.px[a], ld.px[b], gf) / DM, yy = lerp(ld.py[a], ld.py[b], gf) / DM;
      const m = l.meshes[k];
      m.visible = on; m.position.set(X(xx), 0, Z(yy));
      m.userData.num.visible = on && S.showNumbers; m.userData.name.visible = false;
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
    $('err' + li).textContent = `now: players ${cnt ? (sum / cnt).toFixed(1) : '–'} m, ball ${Math.hypot(gbx - bx, gby - by).toFixed(1)} m from the real ones`;
  });
}

// ---------- cameras ----------
const camPos = new THREE.Vector3(0, 40, 60), camLook = new THREE.Vector3();
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
  if (S.playing && S.data) {
    S.t += dt * S.speed;
    if (S.t >= S.t1) { S.t = S.t1; S.playing = false; setPlayIcon(); }
  }
  const w0 = performance.now();
  update();
  placeCamera(dt, false);
  renderer.render(scene, camera);
  S.frameMs = 0.95 * (S.frameMs || 0) + 0.05 * (performance.now() - w0);   // work per frame (ms), for checking speed
  requestAnimationFrame(loop);
}
window.__viewer = S;                                     // for debugging in the browser console
window.__viewerStep = () => { update(); placeCamera(0.016, false); renderer.render(scene, camera); };
loadIndex().then(() => setCam('broadcast')).catch(fail);
requestAnimationFrame(loop);
