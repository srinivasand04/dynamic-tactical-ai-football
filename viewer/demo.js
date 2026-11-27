// Demo mode: one real case, every model at work. A step panel drives the viewer (file, view, layer, camera, time) and shows
// which model is behind what you see. Badges: amber = open model (its own name), green = our model (own_ + name).
// Only DOM controls and the debug hooks of app.js (window.__viewer, window.__cardFor) are used, so the viewer is unchanged.

const OPEN = new Set(['RF-DETR-SoccerNet', 'unravelsports-PressingIntensity', 'FootballBERT', 'TiZero', 'offscreen-impute']);

const STEPS = [
  { title: '1 · A real goal, replayed', file: 'J03WOH_whatif_3610.json', view: 'real', cam: 'broadcast', t: 3606,
    story: 'Bundesliga, Fortuna Düsseldorf v Jahn Regensburg (J03WOH), 2.6 s before a real goal. The replay is the real tracking data.',
    result: 'Watch the threat pill rise as the attack builds, and the press meter show how hard the defenders close in.',
    models: ['own_Threat', 'own_xG', 'unravelsports-PressingIntensity'] },
  { title: '2 · Coach asks: “what if we pass earlier?”', file: 'J03WOH_whatif_3610.json', view: 'split', layer: 1, cam: 'broadcast', t: 3610,
    story: 'From the same real moment the simulator (GRF) continues the play with our team models; the carrier passes forward at once.',
    result: 'Left: real match. Right: what-if. Over 20 runs the goal rate goes 0.25 → 0.65 (+0.40 ± 0.26) against our own model defence.',
    models: ['own_TeamAgent', 'own_TeamPolicy', 'own_TeamDecisions', 'own_MovementGNN', 'own_WhatIfEmulator'] },
  { title: '3 · Does it still work against a top defence?', file: 'J03WOH_demo_tzdef.json', view: 'split', layer: 1, cam: 'top', t: 3610,
    story: 'Same early pass by our attack, but now TiZero, an open AI that beats GRF’s hardest built-in AI 75% of the time, defends.',
    result: 'Left: real match. Right: the same early pass with TiZero defending: 0 shots (with our model defending, step 2, this run scored after 10.3 s). Over 140 runs our attack scores 0: the gain only held against a weak defence.',
    models: ['TiZero', 'own_TeamAgent'] },
  { title: '4 · Only TV video? Fill in what the camera can’t see', file: 'video_SNGS-116_filled_learned.json', view: 'real', cam: 'top', t: null,
    story: 'A 30 s broadcast clip (SoccerNet). The open detector finds players and the ball; players off camera are added as see-through ghosts.',
    result: 'About 10 players seen → about 20 on the board. Our learned filler places hidden players 3.3 m from reality (simple guess: 15.9 m).',
    models: ['RF-DETR-SoccerNet', 'offscreen-impute', 'own_LearnedFiller'] },
  { title: '5 · Scouting: who could replace him?', file: '10517_full.json', view: 'real', cam: 'follow', t: null, player: 'Messi',
    story: 'World Cup 2022 final, Argentina v France (PFF tracking), a moment with Messi near the ball; his player card opens.',
    result: '', models: ['FootballBERT', 'unravelsports-PressingIntensity', 'own_Threat'] },
  { title: '6 · Who was the best player of the match?', file: 'J03WOH_full.json', view: 'real', cam: 'follow', t: null, pom: 0,
    story: 'Düsseldorf 4-0 Regensburg, full match. Every player is valued with the ball (goal chance added by passes, carries, shots) and without it: each run is compared with a ghost of where an average player would have gone.',
    result: 'Player of the match: D. Ginczek (1 goal, 1 assist), almost all from his runs. The pale ghost shows where own_MovementGNN expected him; he went to the gold ring instead.',
    models: ['own_PlayerOfMatch', 'own_MovementGNN', 'own_Threat', 'own_xG'] },
  { title: '7 · Coach: “Klaus, run in behind”', file: 'mode6_coach_run.json', view: 'split', layer: 1, cam: 'broadcast', t: 1619,
    story: 'Düsseldorf v Regensburg, 27:01. Hoffmann has the ball; Klaus stays wide (left, real). The coach tells him to sprint in behind the left-back and be played through; GRF plays it with our team models (right).',
    result: 'Left: in the real match the ball went wide to Klaus and was lost after 5 s. Right: he arrives in the box and is played through. Over 20 paired runs the attack’s threat goes 0.20 → 0.36 (±0.07); the pass reaches him in 25% of runs, so xG stays within noise. His player model: 3.3 runs in behind per 90 (wide midfielders 2.6), a run he can make.',
    models: ['own_PlayerModel', 'own_TeamAgent', 'own_MovementGNN', 'own_Threat'] },
];

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let cur = -1;

function badge(m) { return `<span class="dbadge ${OPEN.has(m) ? 'open' : 'own'}">${m}</span>`; }

async function loadFile(file) {
  const S = window.__viewer, before = S.data, sel = $('file');
  if (sel.value === file && S.data) return;
  sel.value = file; sel.dispatchEvent(new Event('change'));
  for (let i = 0; i < 100 && (S.data === before || !S.data); i++) await sleep(100);
  await sleep(200);
}

function findPlayer(name) {                       // slot and a time when this player is closest to the ball
  const S = window.__viewer, d = S.data;
  const side = Object.entries(d.names || {}).find(([, m]) => Object.values(m).some((n) => n.includes(name)));
  if (!side) return null;
  const shirt = +Object.entries(side[1]).find(([, n]) => n.includes(name))[0];
  let best = null;
  for (const [fi, ro] of d.rosters) {
    const k = ro.indexOf(shirt) >= 0 ? ro.indexOf(shirt) : -1;
    if (k < 0) continue;
    for (let i = fi; i < Math.min(fi + 6000, d.t.length); i += 25) {
      const a = i * 22 + k;
      if (d.px[a] < 0 || !d.live[i]) continue;
      const dist = Math.hypot(d.px[a] - d.bx[i], d.py[a] - d.by[i]);
      if (dist < 15 && d.t[i] - d.t[0] > 600) { best = { k, t: d.t[i] - 2 }; break; }
    }
    if (best) break;
  }
  return best;
}

async function go(n) {
  cur = Math.max(0, Math.min(STEPS.length - 1, n));
  const st = STEPS[cur], S = window.__viewer;
  $('demoTitle').textContent = st.title;
  $('demoStory').textContent = st.story;
  $('demoResult').textContent = st.result || '';
  $('demoBadges').innerHTML = st.models.map(badge).join('');
  $('demoCard').innerHTML = '';
  $('demoStep').textContent = `${cur + 1} / ${STEPS.length}`;
  await loadFile(st.file);
  $('viewMode').value = st.view; $('viewMode').dispatchEvent(new Event('change'));
  if (st.layer != null) { $('viewLayer').value = String(st.layer); $('viewLayer').dispatchEvent(new Event('change')); }
  const cam = document.querySelector(`[data-cam="${st.cam}"]`); if (cam) cam.click();
  let t = st.t;
  if (st.player) {
    const f = findPlayer(st.player);
    if (f) {
      t = f.t; S.t = t;
      if (window.__viewerStep) window.__viewerStep();          // the line-up of that moment, then the card
      if (window.__cardFor) { window.__cardFor(f.k); $('demoCard').innerHTML = $('pcard').innerHTML; }
      $('demoResult').textContent = 'FootballBERT compares players by the patterns it learned across many matches: these are the 5 most similar World Cup players, a shortlist for scouting.';
    }
  }
  S.t = t == null ? S.t0 : Math.min(Math.max(t, S.t0), S.t1);
  S.playing = true;
  if (st.pom == null) { $('pom').classList.add('hidden'); S.followSlot = -1; }
  else {                                                         // player of the match: open the panel, pick the #1, watch his best run
    $('pom').classList.remove('hidden');
    for (let i = 0; i < 50 && !(window.__pom && window.__pom.state.data); i++) { if (window.__pom) window.__pom.step(); await sleep(100); }
    if (window.__pom && window.__pom.state.data) window.__pom.pick(st.pom);
  }
}

export function startDemo() {
  $('demo').classList.remove('hidden');
  go(0);
}
$('demoBtn').addEventListener('click', () => ($('demo').classList.contains('hidden') ? startDemo() : $('demo').classList.add('hidden')));
$('demoNext').addEventListener('click', () => go(cur + 1));
$('demoPrev').addEventListener('click', () => go(cur - 1));
$('demoClose').addEventListener('click', () => $('demo').classList.add('hidden'));
if (location.hash === '#demo') setTimeout(startDemo, 1500);
window.__demoGo = go;
