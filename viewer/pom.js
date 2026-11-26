// Player of the match (own_PlayerOfMatch, scripts/player_of_match.py export): a panel with the top 5 of the match, a gold
// ring on the chosen player (the Follow camera follows him), and his best moments. In an off-ball moment a see-through
// ghost shows where own_MovementGNN expected him to run, with a line from the ghost to where he really went.
// Data: viewer/data/<match>_pom.json, loaded when the open file is that match. Uses only the debug hooks of app.js.

import * as THREE from 'three';

const $ = (id) => document.getElementById(id);
const L = 105, W = 68, DM = 10;
const X = (x) => x - L / 2, Z = (y) => W / 2 - y;
const P = { data: null, match: null, pick: -1, moment: null };

// gold ring at the chosen player's feet, a ghost (see-through figure) and a line ghost -> real spot
const ring = new THREE.Mesh(new THREE.RingGeometry(1.6, 2.1, 40),
  new THREE.MeshBasicMaterial({ color: 0xffc83d, side: THREE.DoubleSide, transparent: true, opacity: 0.95 }));
ring.rotation.x = -Math.PI / 2; ring.visible = false;
const ghost = new THREE.Group();
const gm = new THREE.MeshBasicMaterial({ color: 0xbfe9ff, transparent: true, opacity: 0.35, depthWrite: false });
const body = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.3, 1.5, 16), gm); body.position.y = 1.0; ghost.add(body);
const head = new THREE.Mesh(new THREE.SphereGeometry(0.26, 16, 12), gm); head.position.y = 2.0; ghost.add(head);
const gring = new THREE.Mesh(new THREE.RingGeometry(1.2, 1.5, 32),
  new THREE.MeshBasicMaterial({ color: 0xbfe9ff, side: THREE.DoubleSide, transparent: true, opacity: 0.7 }));
gring.rotation.x = -Math.PI / 2; gring.position.y = 0.05; ghost.add(gring);
ghost.visible = false;
const lineGeo = new THREE.BufferGeometry(); lineGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
const line = new THREE.Line(lineGeo, new THREE.LineDashedMaterial({ color: 0xffc83d, dashSize: 0.7, gapSize: 0.4 }));
line.visible = false;
const realDot = new THREE.Mesh(new THREE.RingGeometry(1.2, 1.5, 32),
  new THREE.MeshBasicMaterial({ color: 0xffc83d, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }));
realDot.rotation.x = -Math.PI / 2; realDot.visible = false;

let added = false;
function addToScene() {
  if (added || !window.__scene) return;
  [ring, ghost, line, realDot].forEach((o) => window.__scene.add(o));
  added = true;
}

const sign = (v) => (v >= 0 ? '+' : '') + v.toFixed(2);
const pct = (v) => (v >= 0 ? '+' : '') + (100 * v).toFixed(1) + '%';
const clockOf = (t) => {                                   // match minute from the data clock [[t, match s], ...]
  const d = window.__viewer.data, c = (d && d.clock) || [[0, 0]];
  let base = c[0];
  for (const k of c) if (t >= k[0]) base = k;
  const s = base[1] + (t - base[0]);
  return `${Math.floor(s / 60)}'${String(Math.floor(s % 60)).padStart(2, '0')}`;
};

function slotOf(p) {                                       // viewer slot of this player at the current time (side + shirt)
  const S = window.__viewer, ro = S.roster || (S.data.rosters && S.data.rosters[0][1]);
  if (!ro) return -1;
  const lo = p.side === 'home' ? 0 : 11;
  for (let k = lo; k < lo + 11; k++) if (ro[k] === p.shirt) return k;
  return -1;
}

function render() {
  const d = P.data, box = $('pomBody');
  if (!d) { box.innerHTML = '<p class="hint">No player-of-the-match file for this match. Open "J03WOH full match".</p>'; return; }
  const rows = d.players.map((p, i) => `<tr data-i="${i}" class="${i === P.pick ? 'on' : ''}">
      <td>${p.rank}</td><td>${p.name}<span class="pteam ${p.side}">${p.team}</span></td>
      <td>${sign(p.onball)}</td><td>${sign(p.offball)}</td><td><b>${sign(p.pom)}</b></td>
      <td>${p.goals ? '&#9917;'.repeat(p.goals) : ''}${p.assists ? '<span class="ast">A</span>'.repeat(p.assists) : ''}</td></tr>`).join('');
  let html = `<div class="pomsub">${d.home} ${d.score} ${d.away}</div>
    <table class="pomt"><tr><th>#</th><th>Player</th><th title="goal chance added with the ball">on</th>
    <th title="goal chance of his runs minus the own_MovementGNN ghost">off</th><th>POM</th><th></th></tr>${rows}</table>`;
  if (P.pick >= 0) {
    const p = d.players[P.pick];
    html += `<div class="pomm"><div><b>${p.name}</b> &middot; best moments (click to watch)</div>` + p.moments.map((m, j) => `
      <button data-m="${j}" class="${P.moment === m ? 'on' : ''}">${clockOf(m.t1)} &middot; ${m.kind} <span>${pct(m.added)}</span></button>`).join('') + '</div>';
  }
  html += `<p class="hint">on = goal chance he added with the ball; off = goal chance of where he ran minus where the
    <span class="dbadge own">own_MovementGNN</span> ghost expected him (${d.ghost_error_m} m typical error).
    POM = both, scaled within the match. <span class="dbadge own">own_PlayerOfMatch</span></p>`;
  box.innerHTML = html;
  box.querySelectorAll('tr[data-i]').forEach((tr) => tr.addEventListener('click', () => pick(+tr.dataset.i)));
  box.querySelectorAll('button[data-m]').forEach((b) => b.addEventListener('click', () => watch(d.players[P.pick].moments[+b.dataset.m])));
}

export function pick(i) {
  P.pick = i; P.moment = null;
  const S = window.__viewer;
  document.querySelector('[data-cam="follow"]').click();
  render();
  const p = P.data.players[i];
  if (p.moments.length) watch(p.moments.find((m) => m.kind === 'off-ball run') || p.moments[0]);
  else S.followSlot = slotOf(p);
}

export function watch(m) {
  const S = window.__viewer;
  P.moment = m;
  S.t = Math.max(S.t0, (m.t0 ?? m.t1) - 3);
  S.playing = true;
  if (window.__viewerStep) window.__viewerStep();
  S.followSlot = slotOf(P.data.players[P.pick]);
  render();
}

async function loadFor(d) {
  P.data = null; P.pick = -1; P.moment = null;
  if (window.__viewer) window.__viewer.followSlot = -1;
  P.match = d && d.match;
  if (P.match) {
    try { const r = await fetch(`data/${P.match}_pom.json`); if (r.ok) P.data = await r.json(); } catch (e) { P.data = null; }
  }
  render();
}

function tick() {
  requestAnimationFrame(tick);
  step();
}

function step() {                                          // one update of ring, ghost and note (also called by tests / recorders)
  const S = window.__viewer;
  if (!S || !S.data) return;
  addToScene();
  if (S.data.match !== P.match) { loadFor(S.data); return; }
  const on = !$('pom').classList.contains('hidden') && P.data && P.pick >= 0;
  const k = on ? slotOf(P.data.players[P.pick]) : -1;
  if (on) S.followSlot = k;
  const pl = k >= 0 ? window.__viewerPlayers[k] : null;
  ring.visible = !!(pl && pl.visible);
  if (ring.visible) ring.position.set(pl.position.x, 0.07, pl.position.z);
  // the ghost: from his real spot at the start of the run, sliding to where the GNN expected him at its end; shown until 2 s after
  const m = on ? P.moment : null;
  const show = m && m.ghost && S.t >= m.t0 - 0.5 && S.t <= m.t1 + 2.5;
  ghost.visible = line.visible = realDot.visible = !!show;
  if (show) {
    const f = Math.min(Math.max((S.t - m.t0) / Math.max(m.t1 - m.t0, 0.1), 0), 1);
    const sx = pl ? pl.position.x : X(m.real[0]), sz = pl ? pl.position.z : Z(m.real[1]);
    const gx = X(m.ghost[0]), gz = Z(m.ghost[1]);
    if (!P.start || P.start.m !== m) P.start = { m, x: sx, z: sz };
    if (S.t <= m.t0) P.start = { m, x: sx, z: sz };
    ghost.position.set(P.start.x + (gx - P.start.x) * f, 0, P.start.z + (gz - P.start.z) * f);
    realDot.position.set(X(m.real[0]), 0.06, Z(m.real[1]));
    const a = lineGeo.attributes.position.array;
    a[0] = ghost.position.x; a[1] = 0.15; a[2] = ghost.position.z; a[3] = X(m.real[0]); a[4] = 0.15; a[5] = Z(m.real[1]);
    lineGeo.attributes.position.needsUpdate = true; line.computeLineDistances();
    $('pomNote').textContent = S.t < m.t1 ? `Ghost (${'own_MovementGNN'}) runs where an average player would go…`
      : `He ran to the gold ring, the ghost stopped at the pale one: ${pct(m.added)} goal chance if the ball comes to him.`;
    $('pomNote').classList.remove('hidden');
  } else {
    $('pomNote').classList.add('hidden');
  }
}

$('pomBtn').addEventListener('click', () => { $('pom').classList.toggle('hidden'); render(); });
$('pomClose').addEventListener('click', () => $('pom').classList.add('hidden'));
window.__pom = { pick, watch, step, state: P };
requestAnimationFrame(tick);
