// Player models showcase: one deterministic timeline (window.__seek(t)) so it can be played live or recorded frame by
// frame (scripts/record_showcase.py). Data: target_player.json, showcase_xhaka.json, backtest.json, reports.json.
import * as THREE from 'three';

const $ = (id) => document.getElementById(id);
const get = async (f) => { try { const r = await fetch(`data/${f}`, { cache: 'no-store' }); return r.ok ? r.json() : null; } catch (e) { return null; } };
const [T, X, B, RP] = await Promise.all([get('target_player.json'), get('showcase_xhaka.json'), get('backtest.json'), get('reports.json')]);
const NAME = T ? T.name : 'Target player';
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const ease = (v) => { v = clamp(v); return v * v * (3 - 2 * v); };
const pct = (v, d = 0) => `${(100 * v).toFixed(d)}%`;
const sgn = (v, d = 3) => `${v >= 0 ? '+' : ''}${v.toFixed(d)}`;

// ------------------------------------------------------------------ three.js scene
const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a1222);
scene.fog = new THREE.Fog(0x0a1222, 140, 260);
const camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.5, 600);
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x0b1a10, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(-40, 90, 60); scene.add(sun);
function resize() {
  renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight;
  camera.setViewOffset(innerWidth, innerHeight, innerWidth * 0.17, -innerHeight * 0.05, innerWidth, innerHeight);   // pitch left of the panel
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

const P = (x, y, h = 0) => new THREE.Vector3(x - 52.5, h, -(y - 34));
// pitch
const pitch = new THREE.Group(); scene.add(pitch);
for (let i = 0; i < 12; i++) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(105 / 12, 68), new THREE.MeshLambertMaterial({ color: i % 2 ? 0x1f6a3e : 0x1b5f37 }));
  m.rotation.x = -Math.PI / 2; m.position.set(-52.5 + (i + 0.5) * 105 / 12, 0, 0); pitch.add(m);
}
const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 });
const seg = (pts) => { const g = new THREE.BufferGeometry().setFromPoints(pts.map(([x, y]) => P(x, y, 0.05))); pitch.add(new THREE.Line(g, lineMat)); };
seg([[0, 0], [105, 0], [105, 68], [0, 68], [0, 0]]); seg([[52.5, 0], [52.5, 68]]);
seg([[0, 13.84], [16.5, 13.84], [16.5, 54.16], [0, 54.16]]); seg([[105, 13.84], [88.5, 13.84], [88.5, 54.16], [105, 54.16]]);
seg([[0, 24.84], [5.5, 24.84], [5.5, 43.16], [0, 43.16]]); seg([[105, 24.84], [99.5, 24.84], [99.5, 43.16], [105, 43.16]]);
seg(Array.from({ length: 49 }, (_, i) => [52.5 + 9.15 * Math.cos(i / 48 * 2 * Math.PI), 34 + 9.15 * Math.sin(i / 48 * 2 * Math.PI)]));
const zc = (z) => [(Math.floor(z / 4) + 0.5) * 17.5, (z % 4 + 0.5) * 17];
const groups = [];
const G = () => { const g = new THREE.Group(); g.visible = false; scene.add(g); groups.push(g); return g; };
const mat = (c, o = 1) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.55, metalness: 0.05, transparent: o < 1, opacity: o });

// S1: every action valued (instanced bars)
const gBars = G();
const acts = X ? X.actions : [];
const vmax = Math.max(0.02, ...acts.map((a) => Math.abs(a[4])));
const bars = new THREE.InstancedMesh(new THREE.BoxGeometry(0.55, 1, 0.55), new THREE.MeshStandardMaterial({ roughness: 0.5 }), Math.max(acts.length, 1));
acts.forEach((a, i) => bars.setColorAt(i, new THREE.Color(Math.abs(a[4]) < 0.002 ? 0x8090a8 : a[4] > 0 ? 0x46c878 : 0xff6f61)));
gBars.add(bars);
const dummy = new THREE.Object3D();
function setBars(f) {
  acts.forEach((a, i) => {
    const on = ease((f * acts.length - i * 0.85) / 30);
    const h = 0.15 + 16 * Math.sqrt(Math.abs(a[4]) / vmax);
    dummy.position.copy(P(a[0], a[1], (h * on) / 2)); dummy.scale.set(1, Math.max(h * on, 0.001), 1); dummy.updateMatrix();
    bars.setMatrixAt(i, dummy.matrix);
  });
  bars.instanceMatrix.needsUpdate = true;
}

// S2: lanes as arcs
const gLanes = G();
const lanes = (T && T.vaep ? T.vaep.lanes : T ? T.m3.lanes : []).slice(0, 6);
const lmax = Math.max(1e-6, ...lanes.map((l) => Math.abs(l.v90 ?? l.xt90)));
const laneMeshes = lanes.map((l, i) => {
  let [x0, y0] = zc(l.z0), [x1, y1] = zc(l.z1);
  if (l.z0 === l.z1) { x1 += 7; y1 += (i % 2 ? 5 : -5); }
  const w = Math.abs(l.v90 ?? l.xt90) / lmax;
  const curve = new THREE.QuadraticBezierCurve3(P(x0, y0, 0.3), P((x0 + x1) / 2, (y0 + y1) / 2, 6 + 10 * w), P(x1, y1, 0.3));
  const geo = new THREE.TubeGeometry(curve, 60, 0.35 + 1.1 * w, 10, false);
  const m = new THREE.Mesh(geo, mat(i === 0 ? 0xf5be46 : 0x6aa8ff, 0.92)); gLanes.add(m);
  const tip = new THREE.Mesh(new THREE.ConeGeometry(0.9 + 1.4 * w, 2.4, 14), mat(i === 0 ? 0xf5be46 : 0x6aa8ff));
  tip.position.copy(curve.getPoint(1)); tip.lookAt(curve.getPoint(0.97)); tip.rotateX(-Math.PI / 2); gLanes.add(tip);
  return { m, tip, n: geo.index.count };
});

// S3: pressing side
const gPress = G();
const body = new THREE.Mesh(new THREE.CapsuleGeometry(1.4, 3.2, 8, 16), mat(0xe03c5a)); body.position.copy(P(45, 38, 3)); gPress.add(body);
const sides = T ? Object.fromEntries(T.m4.sides.map((s) => [s.side, s])) : {};
const arrows = { front: [1, 0], left: [0, 1], right: [0, -1], behind: [-1, 0] };
const arrowObjs = Object.entries(arrows).map(([k, [dx, dy]]) => {
  const best = T && T.m4.best === k;
  const g = new THREE.Group();
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.45, 9, 12), mat(best ? 0xf5be46 : 0x6aa8ff));
  const head = new THREE.Mesh(new THREE.ConeGeometry(1.3, 3, 16), mat(best ? 0xf5be46 : 0x6aa8ff));
  shaft.position.y = 4.5; head.position.y = 10.5; head.rotation.x = Math.PI; g.add(shaft, head);
  const start = P(45 + dx * 20, 38 + dy * 20, 2.5), end = P(45 + dx * 3, 38 + dy * 3, 2.5);
  g.position.copy(start); g.lookAt(end); g.rotateX(Math.PI / 2); g.userData = { start, end }; gPress.add(g);
  return { k, g, best };
});

// S4: where to win the ball
const gWin = G();
const winZ = T ? T.m5.zones : [];
const wmax = Math.max(1e-6, ...winZ.map((z) => z.value90));
const winTiles = winZ.map((z, i) => {
  const [x, y] = zc(z.z);
  const m = new THREE.Mesh(new THREE.BoxGeometry(16.5, 1, 16), mat(i === 0 ? 0xf5be46 : 0x28c8be, 0.85));
  m.position.copy(P(x, y, 0)); gWin.add(m); return { m, h: 1 + 14 * z.value90 / wmax };
});

// S5: what-if plans as bars
const gPlan = G();
const plans = T ? (T.vaep ? T.vaep.plans.map((p) => ({ plan: p.plan, v: p.vaep100 })) : T.m7.plans.map((p) => ({ plan: p.plan, v: p.net100 }))) : [];
const bestPlan = T ? (T.vaep ? T.vaep.best : T.m7.best) : '';
const pmax = Math.max(1e-6, ...plans.map((p) => Math.abs(p.v)));
const planBars = plans.map((p, i) => {
  const m = new THREE.Mesh(new THREE.BoxGeometry(7, 1, 7), mat(p.plan === bestPlan ? 0xf5be46 : p.v > 0 ? 0xff6f61 : 0x46c878));
  m.position.copy(P(12 + i * 11.5, 34, 0)); gPlan.add(m); return { m, h: 1 + 20 * Math.abs(p.v) / pmax, neg: p.v < 0 };
});

// S6: on-ball style with bands
const gStyle = G();
const xkey = RP ? (RP.index.find((q) => q[1] === NAME && q[3].startsWith('Leverkusen')) || [null])[0] : null;
const xst = xkey ? RP.players[xkey].style : null;
const styleObjs = (xst ? xst.his : []).map((v, k) => {
  const x = 20 + k * 13;
  const bar = new THREE.Mesh(new THREE.BoxGeometry(6, 1, 6), mat(0x6aa8ff)); bar.position.copy(P(x, 34, 0)); gStyle.add(bar);
  const band = new THREE.Mesh(new THREE.BoxGeometry(7.5, 1, 7.5), mat(0xf5be46, 0.35)); gStyle.add(band);
  const typ = new THREE.Mesh(new THREE.TorusGeometry(4.6, 0.35, 8, 32), mat(0xffffff)); typ.rotation.x = Math.PI / 2; gStyle.add(typ);
  return { bar, band, typ, x, v, lo: xst.lo[k], hi: xst.hi[k], t: xst.typical[k] };
});

// S7: coach view grid (minute x score)
const gCoach = G();
const cw = T && T.coach ? T.coach.c2.whatif : [];
const cmax = Math.max(1e-6, ...cw.map((g) => g.xt90));
const coachBars = cw.map((g) => {
  const mi = [10, 30, 50, 70, 85].indexOf(g.minute), si = ['trailing', 'drawing', 'leading'].indexOf(g.state);
  const m = new THREE.Mesh(new THREE.BoxGeometry(9, 1, 9), mat(new THREE.Color().setHSL(0.12 - 0.12 * (1 - g.xt90 / cmax), 0.85, 0.55)));
  m.position.copy(P(22 + mi * 15, 16 + si * 18, 0)); gCoach.add(m); return { m, h: 1 + 18 * g.xt90 / cmax };
});

// S8: any player zones
const gAny = G();
const pick = [];
if (RP) {
  const byKey = (k) => RP.players[k] ? k : null;
  pick.push(xkey);
  const top = RP.index.map((q) => q[0]).filter((k) => RP.players[k].comp === 'Premier League' && RP.players[k].minutes >= 1800)
    .sort((a, b) => RP.players[b].vaep90 - RP.players[a].vaep90);
  pick.push(top[0]);
  const cb = RP.index.map((q) => q[0]).filter((k) => RP.players[k].comp === 'World Cup 2022' && RP.players[k].minutes >= 450)
    .sort((a, b) => RP.players[b].vaep90 - RP.players[a].vaep90);
  pick.push(cb[0]);
  pick.splice(0, pick.length, ...pick.filter(byKey));
}
const anyTiles = Array.from({ length: 24 }, (_, z) => {
  const [x, y] = zc(z);
  const m = new THREE.Mesh(new THREE.BoxGeometry(16.5, 1, 16), mat(0xf5be46, 0.85)); m.position.copy(P(x, y, 0)); gAny.add(m); return m;
});

// ------------------------------------------------------------------ text per scene
const m2 = T ? T.m2 : null, va = T ? T.vaep : null, bt = B || {};
const sbv = bt.value && bt.value.sb, wyv = bt.value && bt.value.wy, bay = bt.bayes && bt.bayes.season;
const row = (a, b) => `<div class="row"><span>${a}</span><b>${b}</b></div>`;
const barRow = (lab, v, mx, cls = '', txt = '') => `<div class="lab"><span>${lab}</span><span>${txt}</span></div><div class="bar ${cls}"><span style="width:${(100 * clamp(v / mx)).toFixed(1)}%"></span></div>`;
const S = [
  { d: 7, kick: 'Player models', title: `One player, every dataset`, sub: `${NAME}: Arsenal, Leverkusen, Switzerland. 3 data providers, 8 competitions, 2016 to 2024.`,
    panel: () => `<h4>DATA</h4><div class="chips">${(T ? T.contexts : []).map((c) => `<div class="chip">${c.label}<b>${c.matches}</b></div>`).join('')}</div>
      <div class="note">Events (StatsBomb 360, Wyscout) and broadcast tracking (PFF). Every model is tested on matches it never saw.</div>`,
    cap: 'Goal: turn raw match data into decisions a coach can act on, with honest accuracy.' },
  { d: 12, kick: 'Model 1 · Possession value (VAEP)', title: 'Every action gets a value', sub: 'How much did it change the chance to score, minus the chance to concede?',
    panel: () => `<h4>${NAME.toUpperCase()} · LEVERKUSEN + TOURNAMENTS</h4><div class="big">${va ? va.sb_vaep90.toFixed(3) : m2 ? m2.xt90.toFixed(3) : '-'}<small>value per 90</small></div>
      ${va ? barRow('Percentile among defensive / central midfielders', va.sb_pct, 100, '', `${va.sb_pct.toFixed(0)} / 100`) : ''}
      ${va ? barRow('Arsenal 2017/18 (Premier League)', va.pl_pct, 100, 'b', `${va.pl_pct.toFixed(0)} / 100`) : ''}
      ${sbv ? row('Model accuracy, unseen matches (AUC)', `${sbv.location.scores.auc.toFixed(3)} → ${sbv.full.scores.auc.toFixed(3)}`) : ''}
      <div class="note">Location-only model → full model with the 360 freeze frame. Green bars add value, red bars lose it.</div>`,
    cap: 'Each bar is one of his actions: its height is the value it added or lost.' },
  { d: 11, kick: 'Model 2 · Threat lanes', title: 'Where his value travels', sub: 'Zone-to-zone lanes of his completed passes and carries, ranked by value.',
    panel: () => `<h4>TOP LANES (VALUE PER 90)</h4>${lanes.map((l, i) => barRow(`${i + 1}. ${l.from} → ${l.to}`, Math.abs(l.v90 ?? l.xt90), lmax, i ? 'b' : '', sgn(l.v90 ?? l.xt90))).join('')}
      <div class="note">${T ? pct(T.m3.left_foot) : ''} of his completed passes are left-footed. Main receivers at Leverkusen: Wirtz, Grimaldo.</div>`,
    cap: 'Close the gold lane first: it carries the most value.' },
  { d: 11, kick: 'Model 3 · Pressing side', title: `Press him from ${T ? (T.m4.best === 'behind' ? 'behind' : 'his ' + T.m4.best) : ''}`, sub: 'Side of the nearest opponent in the 360 freeze frame, against all central midfielders.',
    panel: () => `<h4>FORWARD PASS SHARE BY PRESSURE SIDE</h4>${['none', 'front', 'left', 'right', 'behind'].filter((k) => sides[k]).map((k) => barRow(k === 'none' ? 'nobody within 5 m' : `from ${k === 'left' || k === 'right' ? 'his ' + k : k === 'front' ? 'the front' : 'behind'}`, sides[k].fwd, 0.7, k === T.m4.best ? '' : k === 'none' ? 'g' : 'b', pct(sides[k].fwd))).join('')}
      <div class="note">He keeps the ball under pressure (completion stays at ${pct(sides.behind ? sides.behind.completion : 0)}), but his forward play collapses. Pressure takes away his threat, not the ball.</div>`,
    cap: 'Pressure from behind, on the receive, costs him the most forward passes.' },
  { d: 10, kick: 'Model 4 · Ball loss', title: 'Where to win the ball from him', sub: 'Ball-loss model (his own correction) × what a regain there is worth to us.',
    panel: () => `<h4>BEST ZONES TO HUNT HIM</h4>${winZ.slice(0, 4).map((z, i) => row(`${i + 1}. ${z.zone}`, `${pct(z.risk_none)} → ${pct(z['risk_' + z.best_side])}`)).join('')}
      <div class="note">Loss risk when free → when pressed from the best side. Held out: well calibrated (${T ? pct(T.m5.calib[0], 1) : ''} predicted vs ${T ? pct(T.m5.calib[1], 1) : ''} real).</div>`,
    cap: 'Taller tiles = more value for us if we win it there.' },
  { d: 11, kick: 'Model 5 · What-if', title: 'Test the game plan before the match', sub: 'His real situations replayed under 8 defensive plans: his choices, his ball losses, his value.',
    panel: () => `<h4>HIS VALUE PER 100 ACTIONS (LOWER IS BETTER FOR US)</h4>${plans.map((p) => row(p.plan === bestPlan ? `<span class="hi">${p.plan}</span>` : p.plan, sgn(p.v, 2))).join('')}`,
    cap: `Best plan: ${bestPlan}.` },
  { d: 9, kick: 'Model 6 · Bayesian player model', title: 'What he will do, with honest uncertainty', sub: 'His choice shares with 90% bands (gold) against a typical player in his role (white ring).',
    panel: () => `<h4>WHY IT IS TRUSTWORTHY</h4>${bay ? row('Gain over role model, unseen matches', `${bay.gain_old_pct.toFixed(2)}% → ${bay.gain_bayes_pct.toFixed(2)}%`) + row('Players where it beats the role', `${pct(bay.share_beats_role.old)} → ${pct(bay.share_beats_role.bayes)}`) + row('90% bands contain the truth', Object.values(bay.coverage).map((v) => pct(v)).join(' · ')) : ''}
      <div class="note">Old player model → Bayesian model, on ${bay ? bay.players : ''} players over the last quarter of the season.</div>`,
    cap: 'From left to right: shot, dribble, carry, long pass, forward pass, safe pass.' },
  { d: 9, kick: 'Coach view', title: 'When does he matter most?', sub: 'His value per 90 by minute and score (71 club league matches).',
    panel: () => {
      if (!T || !T.coach) return '';
      const c1 = T.coach.c1, c2 = T.coach.c2, w = (m, s) => c2.whatif.find((g) => g.minute === m && g.state === s);
      const best = c1.styles.find((s) => s.label === c1.best), worst = c1.styles.find((s) => s.label === c1.worst);
      return `<h4>WHAT A COACH CAN TELL HIM</h4>
        ${row('Opponent: best vs ' + best.label.split(',')[0], sgn(best.rel.xt90) + ' / 90')}${row('Opponent: hardest vs ' + worst.label.split(',')[0], sgn(worst.rel.xt90) + ' / 90')}
        ${row('Minute 70, leading: forward passes', `${pct(w(70, 'leading').fwd)} vs ${pct(w(70, 'drawing').fwd)} drawing`)}
        ${row('First 15 min → last 15 min, value / 90', `${c2.by_bucket[0].xt90.toFixed(3)} → ${c2.by_bucket[5].xt90.toFixed(3)}`)}
        ${row('Teammates that clearly change him', `${T.coach.c3.mates.filter((m) => m.clear).length} of ${T.coach.c3.mates.length} (chance level)`)}
        <div class="note">"When we lead, keep breaking lines." · "Against a high press, drop and play the first pass safe."</div>`;
    },
    cap: 'Columns: minute 10 to 85. Rows: trailing, drawing, leading.' },
  { d: 10, kick: 'Product', title: 'Any player, one click', sub: `${RP ? RP.index.length.toLocaleString('en') : ''} player reports across 5 leagues and 5 tournaments.`,
    panel: () => `<h4>REPORT CARDS</h4><div class="cards">${pick.map((k) => { const p = RP.players[k]; return `<div class="rc"><div class="n">${p.name}</div>${p.team}<br>${p.comp} · ${p.role}<div class="v">${p.vaep90.toFixed(3)}</div>value / 90 · top ${Math.max(1, Math.round(100 - (p.pct || 0)))}%</div>`; }).join('')}</div>
      <div class="note">Each report: value per 90, value zones, most valuable actions, style with uncertainty bands, pressure response, similar players.</div>`,
    cap: 'The pitch shows where each player creates value.' },
  { d: 10, kick: 'Proof', title: 'Tested on matches it never saw', sub: 'Every number comes from held-out matches.',
    panel: () => `<h4>BACKTEST</h4>${sbv ? row('Value model AUC (StatsBomb 360)', `${sbv.full.scores.auc.toFixed(3)}`) : ''}${wyv ? row('Value model AUC (Wyscout, 1,941 matches)', `${wyv.full.scores.auc.toFixed(3)}`) : ''}
      ${bt.ratings && bt.ratings.wy && bt.ratings.wy.vaep.season_halves ? row('Rating: 1st half of season predicts 2nd', `r = ${bt.ratings.wy.vaep.season_halves.r.toFixed(2)}`) : ''}
      ${bay ? row('Bayesian player model vs role', `${bay.gain_bayes_pct.toFixed(2)}%`) : ''}
      ${bt.ball_loss ? row('Ball-loss model calibration', `${pct(bt.ball_loss.calib[0], 1)} vs ${pct(bt.ball_loss.calib[1], 1)}`) : ''}
      ${bt.movement ? row('Movement model error (World Cup)', `${bt.movement.bundesliga.toFixed(2)} → ${bt.movement.worldcup.toFixed(2)} m`) : ''}`,
    cap: 'Calibrated probabilities, honest limits: where a model does not help, we say so.' },
  { d: 7, kick: 'Next', title: 'From research to product', sub: '',
    panel: () => `<div class="flow"><div><b>1 Accuracy</b>VAEP + Bayesian player models</div><div><b>2 Coverage</b>any player, 5 leagues + tournaments</div><div><b>3 Product</b>one-click reports, what-if planner</div><div><b>4 Proof</b>backtest dashboard</div></div>`,
    cap: 'Dynamic Tactical AI: player models a club can trust.' },
];
let acc = 0; S.forEach((s) => { s.a = acc; acc += s.d; s.b = acc; });
const DUR = acc;
window.__duration = DUR;
const built = S.map((s) => s.panel());

// ------------------------------------------------------------------ timeline
const sceneGroups = [null, gBars, gLanes, gPress, gWin, gPlan, gStyle, gCoach, gAny, null, null];
let lastScene = -1;
function seek(t) {
  t = clamp(t, 0, DUR - 1e-3);
  const i = S.findIndex((s) => t >= s.a && t < s.b);
  const s = S[i], u = t - s.a, fin = ease(u / 0.7) * ease((s.d - u) / 0.6);
  if (i !== lastScene) {
    $('kick').textContent = s.kick; $('title').textContent = s.title; $('sub').textContent = s.sub;
    $('panel').innerHTML = built[i]; $('cap').textContent = s.cap; $('step').textContent = `${i + 1} / ${S.length}`;
    $('panel').style.display = built[i] ? 'block' : 'none';
    if (i === 10) { $('panel').style.width = '1300px'; $('panel').style.right = '310px'; $('panel').style.top = '420px'; }
    else { $('panel').style.width = '600px'; $('panel').style.right = '48px'; $('panel').style.top = '230px'; }
    lastScene = i;
  }
  for (const id of ['kick', 'title', 'sub', 'panel', 'cap', 'step']) $(id).style.opacity = fin.toFixed(3);
  groups.forEach((g) => { g.visible = false; });
  const g = sceneGroups[i]; if (g) g.visible = true;
  const k = ease((u - 0.3) / (s.d - 1.6));
  // camera
  let R = 104, h = 62, a = -0.32 + 0.022 * u, look = new THREE.Vector3(0, 0, 0);
  if (i === 0) { R = 130 - 10 * ease(u / s.d); h = 90; }
  if (i === 3) { R = 70; h = 42; look = P(45, 38, 2); a = -0.5 + 0.04 * u; }
  if (i === 5 || i === 6) { R = 92; h = 48; look = P(55, 34, 4); }
  if (i >= 9) { R = 140; h = 110; a = 0.0; }
  camera.position.set(look.x + R * Math.sin(a), h, look.z + R * Math.cos(a)); camera.lookAt(look);
  pitch.visible = true;
  pitch.children.forEach((c) => { if (c.material && c.material.color && c.material !== lineMat) c.material.emissive?.setScalar?.(0); });
  if (i === 1) setBars(k);
  if (i === 2) laneMeshes.forEach((l, j) => { const f = clamp(k * lanes.length - j * 0.6); l.m.geometry.setDrawRange(0, Math.floor(l.n * f / 6) * 6); l.tip.visible = f > 0.98; });
  if (i === 3) arrowObjs.forEach((o, j) => { const f = ease(k * 2.2 - j * 0.25); o.g.position.lerpVectors(o.g.userData.start, o.g.userData.end, 0.55 * f + (o.best ? 0.25 * ease(k * 2 - 1.2) : 0)); o.g.scale.setScalar(o.best ? 1 + 0.25 * ease(k * 2 - 1) : 1); });
  if (i === 4) winTiles.forEach((w, j) => { const f = ease(k * 2 - j * 0.15); w.m.scale.y = Math.max(0.001, w.h * f); w.m.position.y = w.h * f / 2; });
  if (i === 5) planBars.forEach((b, j) => { const f = ease(k * 2.2 - j * 0.18); b.m.scale.y = Math.max(0.001, b.h * f); b.m.position.y = b.h * f / 2; });
  if (i === 6) styleObjs.forEach((o, j) => {
    const f = ease(k * 2 - j * 0.15), H = (v) => 1 + 60 * v;
    o.bar.scale.y = Math.max(0.001, H(o.v) * f); o.bar.position.copy(P(o.x, 34, H(o.v) * f / 2));
    const lo = H(o.lo), hi = H(o.hi); o.band.scale.y = Math.max(0.3, hi - lo); o.band.position.copy(P(o.x, 34, (lo + hi) / 2)); o.band.visible = f > 0.9;
    o.typ.position.copy(P(o.x, 34, H(o.t))); o.typ.visible = f > 0.95;
  });
  if (i === 7) coachBars.forEach((b, j) => { const f = ease(k * 2 - j * 0.06); b.m.scale.y = Math.max(0.001, b.h * f); b.m.position.y = b.h * f / 2; });
  if (i === 8 && pick.length) {
    const pi = Math.min(pick.length - 1, Math.floor(u / (s.d / pick.length))), p = RP.players[pick[pi]];
    const mx = Math.max(1e-6, ...p.zones.map((v) => Math.max(v, 0))), lu = u - pi * s.d / pick.length;
    anyTiles.forEach((m, z) => { const hh = 0.4 + 16 * Math.max(p.zones[z], 0) / mx, f = ease(lu / 1.2); m.scale.y = Math.max(0.001, hh * f); m.position.y = hh * f / 2; });
    document.querySelectorAll('.rc').forEach((c, j) => { c.style.borderColor = j === pi ? '#f5be46' : ''; c.style.background = j === pi ? 'rgba(245,190,70,.12)' : ''; });
  }
  renderer.render(scene, camera);
  return true;
}
window.__seek = seek;
window.__ready = true;
if (!new URLSearchParams(location.search).has('record')) {
  const t0 = performance.now();
  const loop = () => { seek(((performance.now() - t0) / 1000) % DUR); requestAnimationFrame(loop); };
  loop();
}
