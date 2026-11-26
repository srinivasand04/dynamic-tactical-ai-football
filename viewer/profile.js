// Player report card (own_PlayerProfile, scripts/player_profile.py viewer): one player across a whole season.
// Radar of the six areas (percentile within his role), strengths and weaknesses with their 90% interval against the
// role median, and his best / worst moments on a small pitch (goal chance added or lost, own_Threat).
// Data: viewer/data/player_profiles.json. Standalone: it does not touch the 3D scene.

const $ = (id) => document.getElementById(id);
const R = { data: null, models: null, big: null, tp: null, rep: null, bt: null, rk: null, view: null, k: 0, mv: null, list: 'best', hover: -1, cls: 2 };
const SHORT = { 'Involvement': 'Involvement', 'Progression and creation': 'Creation', 'Passing': 'Passing',
  'Shooting': 'Shooting', 'Ball security': 'Ball security', 'Defending and pressing': 'Defending' };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const num = (s) => parseFloat(String(s).replace('%', ''));

async function load() {
  if (R.data) return R.data;
  try {
    const r = await fetch('data/player_profiles.json', { cache: 'no-store' });
    if (r.ok) R.data = await r.json();
  } catch (e) { R.data = null; }
  try {
    const r = await fetch('data/player_models.json', { cache: 'no-store' });
    if (r.ok) R.models = await r.json();
  } catch (e) { R.models = null; }
  try {
    const r = await fetch('data/player_models_big.json', { cache: 'no-store' });
    if (r.ok) R.big = await r.json();
  } catch (e) { R.big = null; }
  try {
    const r = await fetch('data/target_player.json', { cache: 'no-store' });
    if (r.ok) R.tp = await r.json();
  } catch (e) { R.tp = null; }
  return R.data;
}

function radar(c, areas) {
  const n = areas.length, cx = 150, cy = 118, rr = 82;
  const pt = (i, v) => { const a = -Math.PI / 2 + (2 * Math.PI * i) / n; return [cx + rr * v * Math.cos(a), cy + rr * v * Math.sin(a)]; };
  const poly = (v) => areas.map((_, i) => pt(i, typeof v === 'function' ? v(i) : v).map((q) => q.toFixed(1)).join(',')).join(' ');
  let g = '';
  [0.25, 0.5, 0.75, 1].forEach((v) => {
    g += `<polygon points="${poly(v)}" class="rgrid${v === 0.5 ? ' rmed' : ''}"/>`;
  });
  areas.forEach((a, i) => {
    const [x, y] = pt(i, 1), [lx, ly] = pt(i, 1.22);
    g += `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" class="rgrid"/>`;
    const v = c.areas[a];
    const anchor = lx < cx - 5 ? 'end' : lx > cx + 5 ? 'start' : 'middle';
    g += `<text x="${lx.toFixed(1)}" y="${(ly + 4).toFixed(1)}" text-anchor="${anchor}" class="rlab">${SHORT[a] || a}
      <tspan class="rval">${v == null ? '–' : Math.round(v)}</tspan></text>`;
  });
  g += `<polygon points="${poly((i) => Math.max(0.03, (c.areas[areas[i]] || 0) / 100))}" class="rshape"/>`;
  return `<svg viewBox="-45 0 390 236" class="pfradar" role="img" aria-label="Percentile per area">${g}</svg>`;
}

function row(i, unreliable) {
  const v = num(i.value), lo = num(i.interval_90[0]), hi = num(i.interval_90[1]), med = num(i.role_median);
  const vals = [v, lo, hi, med].filter(Number.isFinite);
  let lo0 = Math.min(...vals), hi0 = Math.max(...vals);
  const pad = (hi0 - lo0) * 0.15 || 1; lo0 -= pad; hi0 += pad;
  const X = (q) => (100 * (q - lo0) / (hi0 - lo0)).toFixed(1);
  const bar = vals.length === 4 ? `<svg viewBox="0 0 100 14" preserveAspectRatio="none" class="pfbar">
      <line x1="0" y1="7" x2="100" y2="7" class="bbase"/>
      <rect x="${X(lo)}" y="3" width="${(X(hi) - X(lo)).toFixed(1)}" height="8" rx="2" class="bint ${i.label}"/>
      <line x1="${X(med)}" y1="0" x2="${X(med)}" y2="14" class="bmed"/>
      <circle cx="${X(v)}" cy="7" r="2.6" class="bval"/></svg>` : '';
  return `<div class="pfrow${unreliable ? ' unrel' : ''}">
    <div class="pfwhat">${esc(i.what)}</div>
    <div class="pfnums"><b>${esc(i.value)}</b> <span class="hint">[${esc(i.interval_90[0])} – ${esc(i.interval_90[1])}] · role ${esc(i.role_median)}</span>
      <span class="pfpct ${i.label}">${i.percentile == null ? '–' : i.percentile}</span></div>
    ${bar}</div>`;
}

function pitch(ms, on) {
  // 105 x 68 m, his team attacking to the right
  let g = `<rect x="0" y="0" width="105" height="68" class="ppitch"/>
    <line x1="52.5" y1="0" x2="52.5" y2="68" class="pline"/><circle cx="52.5" cy="34" r="9.15" class="pline"/>
    <rect x="0" y="13.84" width="16.5" height="40.32" class="pline"/><rect x="88.5" y="13.84" width="16.5" height="40.32" class="pline"/>
    <rect x="0" y="24.84" width="5.5" height="18.32" class="pline"/><rect x="99.5" y="24.84" width="5.5" height="18.32" class="pline"/>`;
  ms.forEach((m, j) => {
    const cls = `pm ${R.list}${j === on ? ' on' : ''}`;
    const ex = m.ex == null ? m.x : m.ex, ey = m.ey == null ? m.y : m.ey;
    const dash = m.kind === 'Carry' ? ' stroke-dasharray="1.6 1.2"' : '';
    g += `<g class="${cls}" data-j="${j}"><line x1="${m.x}" y1="${m.y}" x2="${ex}" y2="${ey}"${dash} marker-end="url(#pfarr-${R.list})"/>
      <circle cx="${m.x}" cy="${m.y}" r="2.1"/><text x="${m.x}" y="${m.y + 1.1}" text-anchor="middle">${j + 1}</text></g>`;
  });
  return `<svg viewBox="-2 -2 109 72" class="pfpitch" role="img" aria-label="Moments on the pitch">
    <defs><marker id="pfarr-${R.list}" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto">
      <path d="M0,0 L6,3 L0,6 z" class="pmarr ${R.list}"/></marker></defs>${g}</svg>`;
}

function render() {
  const box = $('profBody'); if (!box) return;
  const D = R.data;
  if (!D) { box.innerHTML = '<p class="hint">No data/player_profiles.json. Run: scripts\\player_profile.py viewer</p>'; return; }
  const M = R.models;
  const mvOpts = M ? Object.entries(M.movement).map(([id, v]) => `<option value="m:${id}"${R.mv === id ? ' selected' : ''}>${esc(v.name)} · ${v.matches} matches</option>`).join('') : '';
  const opts = `<optgroup label="Leverkusen 2023/24: profile + on-ball model">${D.cards.map((q, k) => `<option value="c:${k}"${R.mv == null && R.view == null && k === R.k ? ' selected' : ''}>${esc(q.player)} · ${q.role} · ${q.minutes} min</option>`).join('')}</optgroup>`
    + (mvOpts ? `<optgroup label="Bundesliga 2 tracking: movement model">${mvOpts}</optgroup>` : '')
    + (R.big ? `<optgroup label="3 more datasets"><option value="f:0"${R.view === 'f' ? ' selected' : ''}>What we learned at scale</option></optgroup>` : '')
    + `<optgroup label="Any player · accuracy"><option value="r:0"${R.view === 'r' ? ' selected' : ''}>Player report (search any player)</option><option value="b:0"${R.view === 'b' ? ' selected' : ''}>Accuracy: backtest of all models</option><option value="s:0"${R.view === 's' ? ' selected' : ''}>Show 3, find all (moment search)</option></optgroup>`
    + (R.tp ? `<optgroup label="Target player: 7 opponent models + 3 coach models"><option value="t:0"${R.view === 't' ? ' selected' : ''}>${esc(R.tp.name)} · game plan</option></optgroup>` : '');
  if (R.view === 's') {
    loadMs().then(() => {
      box.innerHTML = `<select id="profPick" title="Player">${opts}</select>` + (R.ms ? msCard(R.ms) : '<div class="hint">Run scripts/moment_search.py all.</div>');
      wirePick();
      box.querySelectorAll('.mstabs button').forEach((el) => el.addEventListener('click', () => { R.msk = +el.dataset.k; R.msh = null; render(); }));
      box.querySelectorAll('.pfm, .msm.mshit').forEach((el) => el.addEventListener('click', () => { const j = +el.dataset.j; R.msh = R.msh === j ? null : j; render(); }));
    });
    return;
  }
  if (R.view === 'r' || R.view === 'b') {
    loadRep().then(() => {
      if (R.view === 'b') { box.innerHTML = `<select id="profPick" title="Player">${opts}</select>` + (R.bt ? backtestCard(R.bt) : '<div class="hint">Run scripts/player_report.py backtest.</div>'); wirePick(); return; }
      if (!R.rep) { box.innerHTML = `<select id="profPick" title="Player">${opts}</select><div class="hint">Run scripts/player_report.py reports.</div>`; wirePick(); return; }
      if (!R.rk) R.rk = (R.rep.index.find((q) => q[1] === 'Granit Xhaka' && q[3].startsWith('Leverkusen')) || R.rep.index[0])[0];
      const lab = (q) => `${q[1]} · ${q[2]} · ${q[3]} · ${q[4]}`;
      box.innerHTML = `<select id="profPick" title="Player">${opts}</select>
        <input id="repSearch" list="repList" placeholder="Search ${R.rep.index.length} players…" value="${esc(lab(R.rep.index.find((q) => q[0] === R.rk)))}">
        <datalist id="repList">${R.rep.index.map((q) => `<option value="${esc(lab(q))}">`).join('')}</datalist>` + reportCard(R.rep, R.rk);
      wirePick();
      const inp = document.getElementById('repSearch');
      if (inp) inp.addEventListener('change', () => { const q = R.rep.index.find((x) => lab(x) === inp.value); if (q) { R.rk = q[0]; render(); } });
    });
    return;
  }
  if (R.view === 't' && R.tp) { box.innerHTML = `<select id="profPick" title="Player">${opts}</select>` + targetCard(R.tp); wirePick(); return; }
  if (R.view === 'f' && R.big) { box.innerHTML = `<select id="profPick" title="Player">${opts}</select>` + findingsCard(R.big); wirePick(); return; }
  if (R.mv != null && M) { box.innerHTML = `<select id="profPick" title="Player">${opts}</select>` + movementCard(M.movement[R.mv]); wirePick(); return; }
  const c = D.cards[R.k], unrel = new Set(D.unreliable || []);
  const ms = c[R.list] || [];
  const list = (arr, empty) => arr.length ? arr.map((i) => row(i, unrel.has(i.metric))).join('') : `<p class="hint">${empty}</p>`;
  box.innerHTML = `<select id="profPick" title="Player">${opts}</select>
    <div class="pomsub">${esc(c.team)} · ${c.role} · ${c.matches} matches · ${c.minutes} min</div>
    ${radar(c, D.areas)}
    <div class="hint pfcap">Percentile among ${c.role} players (0–100). Dashed ring = role median (50).</div>
    <h3 class="pfh good">Strengths</h3>${list(c.strengths.slice(0, 6), 'None clearly above the role median.')}
    <h3 class="pfh bad">Weaknesses</h3>${list(c.weaknesses.slice(0, 4), 'None clearly below the role median.')}
    <div class="pflegend"><span class="lg int"></span> 90% interval <span class="lg med"></span> role median <span class="lg val"></span> his value</div>
    ${M && M.onball[c.player] ? onballCard(M.onball[c.player], M.classes) : ''}
    ${R.big && R.big.players[c.player] ? bigCard(R.big.players[c.player], R.big) : ''}
    <h3 class="pfh">Moments <span class="pftabs"><button data-l="best" class="${R.list === 'best' ? 'on' : ''}">Best</button><button data-l="worst" class="${R.list === 'worst' ? 'on' : ''}">Worst</button></span></h3>
    ${pitch(ms, R.hover)}
    <div class="pfms">${ms.map((m, j) => `<div class="pfm${j === R.hover ? ' on' : ''}" data-j="${j}"><span class="pfn">${j + 1}</span>
      ${m.minute}' ${m.home ? 'vs' : 'at'} ${esc(m.vs)} <span class="hint">${m.date}</span><br>
      <span class="hint">${m.kind}${m.result ? ' · ' + esc(m.result) : ''}${m.kind === 'Shot' ? ` · xG ${m.xg.toFixed(2)}` : ''}</span>
      <span class="pfadd ${m.added >= 0 ? 'good' : 'bad'}">${m.added >= 0 ? '+' : ''}${(100 * m.added).toFixed(1)}%</span></div>`).join('')}</div>
    <p class="hint">Goal chance added or lost = own_Threat after minus before the action. Arrows: his team attacks to the right; dashed = carry.
    Peers: ${esc(c.peers)}. Greyed metrics are too noisy across matches to trust. Off-ball runs are not in this data (positions are anonymous).</p>`;
  wirePick();
  box.querySelectorAll('.pfcls button').forEach((b) => b.addEventListener('click', () => { R.cls = +b.dataset.c; render(); }));
  box.querySelectorAll('.pftabs button').forEach((b) => b.addEventListener('click', () => { R.list = b.dataset.l; R.hover = -1; render(); }));
  box.querySelectorAll('.pfm, .pm').forEach((el) => el.addEventListener('click', () => {
    const j = +el.dataset.j; R.hover = R.hover === j ? -1 : j; render();
  }));
}

function wirePick() {
  $('profPick').addEventListener('change', (e) => {
    const [t, v] = e.target.value.split(':');
    R.view = null;
    if (t === 'f' || t === 't' || t === 'r' || t === 'b' || t === 's') { R.view = t; R.mv = null; } else if (t === 'm') R.mv = v; else { R.mv = null; R.k = +v; }
    R.hover = -1; render();
  });
}

function testLine(t, unit) {
  if (!t) return '';
  const cls = t.verdict === 'player model better' ? 'good' : t.verdict === 'role model better' ? 'bad' : '';
  return `<div class="pftest ${cls}"><b>${esc(t.verdict)}</b> on matches the model never saw: ${unit}
    · better in ${t.matches_better} of ${t.matches} matches</div>`;
}

const PITCH = `<rect x="0" y="0" width="105" height="68" class="ppitch"/><line x1="52.5" y1="0" x2="52.5" y2="68" class="pline"/>
  <rect x="0" y="13.84" width="16.5" height="40.32" class="pline"/><rect x="88.5" y="13.84" width="16.5" height="40.32" class="pline"/>`;

// on-ball choice model (Leverkusen): his choice shares vs a typical player of his role, in his own situations
function onballCard(o, classes) {
  const t = o.test;
  const unit = t ? `log-loss ${t.logloss_player.toFixed(3)} vs role ${t.logloss_role.toFixed(3)} (${t.gain_pct.toFixed(1)}% better)` : '';
  const mx = Math.max(...o.all.player, ...o.all.role);
  const rows = classes.map((c, k) => {
    const a = o.all.role[k], b = o.all.player[k], r = b / Math.max(a, 1e-6);
    const tag = r >= 1.25 ? 'up' : r <= 0.8 ? 'down' : '';
    return `<tr><td>${c}</td>
      <td><div class="cbar"><span class="cr" style="width:${(100 * a / mx).toFixed(1)}%"></span><span class="cp" style="width:${(100 * b / mx).toFixed(1)}%"></span></div></td>
      <td>${(100 * b).toFixed(1)}%</td><td class="hint">${(100 * a).toFixed(1)}%</td><td class="${tag}">×${r.toFixed(2)}</td></tr>`;
  }).join('');
  let zg = PITCH;
  for (const z of o.zones) {                                         // third 0..2 (own goal -> opponent goal), channel 0..2 (y low -> high)
    const a = z.role[R.cls], b = z.player[R.cls], r = b / Math.max(a, 1e-6);
    const x = [0, 35, 70][z.third], w = 35, h = 68 / 3, y = 68 - (z.channel + 1) * h;
    const col = r >= 1 ? `rgba(60,201,133,${Math.min(0.85, (r - 1) * 1.2).toFixed(2)})` : `rgba(255,138,122,${Math.min(0.85, (1 / r - 1) * 1.2).toFixed(2)})`;
    zg += `<rect x="${x}" y="${y.toFixed(1)}" width="${w}" height="${h.toFixed(1)}" fill="${col}" stroke="rgba(255,255,255,.15)" stroke-width=".3"/>
      <text x="${x + w / 2}" y="${(y + h / 2 + 0.5).toFixed(1)}" text-anchor="middle" class="ztxt">×${r.toFixed(2)}</text>
      <text x="${x + w / 2}" y="${(y + h / 2 + 4.5).toFixed(1)}" text-anchor="middle" class="ztxt2">${(100 * b).toFixed(0)}% vs ${(100 * a).toFixed(0)}%</text>`;
  }
  const pr = o.pressed, fr = o.free;
  const press = (pr && fr && pr.n >= 30) ? classes.map((c, k) => ({ c, d: (pr.player[k] - pr.role[k]) - (fr.player[k] - fr.role[k]) }))
    .sort((p, q) => Math.abs(q.d) - Math.abs(p.d)).slice(0, 2)
    .map((q) => `${q.c} ${q.d >= 0 ? 'more' : 'less'} than his role (${q.d >= 0 ? '+' : ''}${(100 * q.d).toFixed(1)} pts vs when free)`).join('; ') : '';
  return `<h3 class="pfh">Player model: on-ball choices <span class="hint">own_PlayerModel</span></h3>
    ${testLine(t, unit)}
    <table class="pfct"><tr><th>choice</th><th></th><th>his model</th><th>role</th><th></th></tr>${rows}</table>
    <div class="pflegend"><span class="lg cp"></span> his model <span class="lg cr"></span> typical ${o.role}, same situations (${o.decisions} decisions)</div>
    ${press ? `<div class="hint">Under pressure: ${esc(press)}.</div>` : ''}
    <div class="pfcls">Where: ${classes.map((c, k) => `<button data-c="${k}" class="${k === R.cls ? 'on' : ''}">${c}</button>`).join('')}</div>
    <svg viewBox="-2 -2 109 72" class="pfpitch" role="img" aria-label="Choice by zone">${zg}</svg>
    <div class="hint">Green: he picks <b>${classes[R.cls]}</b> more than a typical ${o.role} would in the same situations in that zone; red: less. Attacking to the right; zones with 20+ decisions.</div>`;
}

// movement model (DFL tracking): his extra movement beyond own_MovementGNN and his role
function movementCard(v) {
  const t = v.test;
  const unit = t ? `error ${t.err_player_m.toFixed(2)} m vs role ${t.err_role_m.toFixed(2)} m vs GNN ${t.err_gnn_m.toFixed(2)} m per step` : '';
  const sc = 40;                                                     // drawing scale: 1 m = 40 px
  const arrows = Object.entries(v.phases).map(([k, p], i) => {
    const cx = 85 + i * 170, cy = 84, dx = p.forward_m * sc, dy = -p.wide_m * sc;
    return `<g><circle cx="${cx}" cy="${cy}" r="5" class="mvdot"/>
      <line x1="${cx}" y1="${cy}" x2="${(cx + dx).toFixed(1)}" y2="${(cy + dy).toFixed(1)}" class="mvarr ${k === 'with ball' ? 'own' : 'opp'}" marker-end="url(#mvh)"/>
      <text x="${cx}" y="150" text-anchor="middle" class="rlab">${k}</text>
      <text x="${cx}" y="166" text-anchor="middle" class="hint2">${p.forward_m >= 0 ? '+' : ''}${p.forward_m.toFixed(2)} m forward · ${p.wide_m >= 0 ? '+' : ''}${p.wide_m.toFixed(2)} m wide</text></g>`;
  }).join('');
  return `<div class="pomsub">${esc(v.team)} · ${esc(v.position || '')} · ${v.matches} matches · ${v.steps} movement steps</div>
    <h3 class="pfh">Player model: off-ball movement <span class="hint">on own_MovementGNN</span></h3>
    ${testLine(t, unit)}
    <svg viewBox="0 0 340 176" class="pfmv" role="img" aria-label="Extra movement">
      <defs><marker id="mvh" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#F4F6F1"/></marker></defs>
      <text x="170" y="18" text-anchor="middle" class="hint2">attacking direction →  (1 m = 40 px)</text>${arrows}</svg>
    <div class="hint">Arrow: where he goes per Key State step beyond what own_MovementGNN and a typical player of his role expect
    (forward = towards the opponent goal, wide = towards his touchline). Learned from his other matches only, tested on the held-out one.</div>`;
}

// ---- 3 more datasets (scripts/player_models_big.py viewer)
const pct = (v, d = 1) => `${v >= 0 ? '+' : ''}${v.toFixed(d)}%`;

// this player in the bigger run: tournament test, club -> tournament, pressure profile
function bigCard(p, B) {
  let h = `<h3 class="pfh">At scale <span class="hint">3 more datasets</span></h3>`;
  if (p.test) h += `<div class="pftest ${p.test.verdict === 'player better' ? 'good' : p.test.verdict === 'role better' ? 'bad' : ''}">
    With tournaments in the pool: <b>${esc(p.test.verdict)}</b> · ${pct(p.test.gain_pct, 2)} vs role on ${p.test.matches} held-out matches</div>`;
  if (p.transfer) h += `<div class="pfsub">Club style 23/24 used at a national-team tournament:</div><table class="pfct">${p.transfer.map((t) =>
    `<tr><td>→ ${esc(t.to)}</td><td class="hint">${t.decisions} decisions</td><td class="${t.gain_pct > 0 ? 'up' : 'down'}">${pct(t.gain_pct, 2)}</td></tr>`).join('')}</table>`;
  if (p.pressure) {
    const d = p.pressure.delta, mx = Math.max(0.05, ...d.map((v) => Math.abs(v || 0)));
    h += `<div class="pfsub">Pressure profile: change in his choices when pressed, beyond a typical ${esc(p.pressure.role)}</div>
      <table class="pfct">${B.classes.map((c, k) => {
        const v = d[k] || 0, w = (50 * Math.abs(v) / mx).toFixed(1), r = B.pressure_reliability[c];
        return `<tr${r < 0.3 ? ' class="unrel"' : ''}><td>${c}</td><td><div class="dbar"><span class="${v >= 0 ? 'dpos' : 'dneg'}" style="width:${w}%;${v >= 0 ? 'left:50%' : 'right:50%'}"></span></div></td>
          <td>${v >= 0 ? '+' : ''}${(100 * v).toFixed(1)} pts</td><td class="hint">r ${r.toFixed(2)}</td></tr>`;
      }).join('')}</table>
      <div class="hint">Same situations with pressure switched on vs off. r = split-half reliability over all 121 players; greyed rows (r &lt; 0.3) are mostly noise.</div>`;
  }
  return h;
}

function curveSvg(B) {
  const w = 300, h = 150, x0 = 34, y0 = 120, nmax = 26, gmax = 4.5, gmin = -0.5;
  const X = (n) => x0 + (w - x0 - 10) * Math.log(n) / Math.log(nmax), Y = (g) => y0 - (y0 - 12) * (g - gmin) / (gmax - gmin);
  let g = `<line x1="${x0}" y1="${Y(0)}" x2="${w - 10}" y2="${Y(0)}" class="rgrid rmed"/>`;
  [0, 1, 2, 3, 4].forEach((v) => { g += `<text x="${x0 - 4}" y="${Y(v) + 3}" text-anchor="end" class="hint2">${v}%</text>`; });
  [1, 2, 4, 8, 16, 26].forEach((n) => { g += `<text x="${X(n)}" y="${y0 + 14}" text-anchor="middle" class="hint2">${n}</text>`; });
  const line = (c, cls) => {
    const band = c.map((q) => `${X(q.n).toFixed(1)},${Y(q.ci[1]).toFixed(1)}`)
      .concat(c.slice().reverse().map((q) => `${X(q.n).toFixed(1)},${Y(q.ci[0]).toFixed(1)}`)).join(' ');
    return `<polygon points="${band}" class="cband ${cls}"/><polyline points="${c.map((q) => `${X(q.n).toFixed(1)},${Y(q.gain).toFixed(1)}`).join(' ')}" class="cline ${cls}"/>`
      + c.map((q) => `<circle cx="${X(q.n).toFixed(1)}" cy="${Y(q.gain).toFixed(1)}" r="2.4" class="cdot ${cls}"/>`).join('');
  };
  g += line(B.curve_wyscout, 'wy') + line(B.curve_leverkusen, 'lv');
  g += `<text x="${(x0 + w) / 2}" y="${h - 2}" text-anchor="middle" class="hint2">matches the player model learned from</text>`;
  return `<svg viewBox="0 0 ${w} ${h}" class="pfcurve" role="img" aria-label="Learning curve">${g}</svg>
    <div class="pflegend"><span class="lg wy"></span> Wyscout, ${B.curve_wyscout[0].players} league players <span class="lg lv"></span> Leverkusen (360 data), ${B.curve_leverkusen[0].players} players · band = 90% interval</div>`;
}

function findingsCard(B) {
  const e = B.eval, n = B.national, s = B.stability, gn = B.gnn;
  const bar = (v, mx, cls, lab) => `<div class="gnrow"><span>${lab}</span><div class="cbar"><span class="${cls}" style="width:${(100 * v / mx).toFixed(1)}%"></span></div><b>${v.toFixed(2)} m</b></div>`;
  const vrow = (lab, q, unit) => `<tr><td>${lab}</td><td>${q.players}</td><td class="up">${q.better}</td><td>${q.unclear}</td><td class="down">${q.worse}</td><td>${unit}</td></tr>`;
  return `<div class="pomsub">${B.datasets.map((d) => `<b>${d.key}</b> ${esc(d.name)} (${d.matches} matches, ${d.what})`).join('<br>')}</div>
    <h3 class="pfh">Does a player model beat his role? <span class="hint">matches it never saw</span></h3>
    <table class="pfct"><tr><th></th><th>players</th><th>better</th><th>unclear</th><th>worse</th><th>median</th></tr>
      ${vrow('A season (Wyscout)', e.wyscout, pct(e.wyscout.median_gain, 2))}
      ${vrow('Tournaments (StatsBomb)', e.statsbomb, pct(e.statsbomb.median_gain, 2))}
      ${vrow('Movement (World Cup)', e.movement, `${e.movement.median_gain_cm.toFixed(2)} cm`)}</table>
    <h3 class="pfh">How many matches does it need?</h3>${curveSvg(B)}
    <div class="hint">Gain in log-loss over the role model on fixed later matches. Below ~5 matches it adds nothing; it keeps rising up to a full season.</div>
    <h3 class="pfh">Is style stable?</h3>
    <table class="pfct"><tr><th>choice</th><th>1st vs 2nd half of season</th><th>odd vs even matches</th></tr>
      ${B.classes.map((c) => `<tr><td>${c}</td><td>r ${s[c][0].toFixed(2)}</td><td class="hint">r ${s[c][1].toFixed(2)}</td></tr>`).join('')}</table>
    <div class="hint">Season halves are almost as alike as odd vs even matches (the noise ceiling): style hardly drifts within a season.</div>
    <h3 class="pfh">Club style at the national team</h3>
    <table class="pfct">${Object.entries(n).map(([t, q]) => `<tr><td>Club season 2017/18 → ${esc(t)}</td><td>${q.players} players</td><td class="up">${pct(q.gain, 2)}</td><td class="hint">better for ${Math.round(100 * q.share)}%</td></tr>`).join('')}</table>
    <h3 class="pfh">Movement model on World Cup tracking <span class="hint">error per player step</span></h3>
    ${bar(gn.still, gn.still, 'cr', 'nobody moves')}${bar(gn.bundesliga, gn.still, 'cr', 'GNN, Bundesliga-trained')}${bar(gn.worldcup, gn.still, 'cp', 'GNN, + World Cup training')}
    <div class="hint">More matching training data helps movement far more than a per-player correction (still no gain with up to 7 matches per player).</div>`;
}

// ---- target player: 7 opponent models (scripts/target_player.py viewer)
const PS = 300 / 105;
const pitchSvg = (inner, lab) => `<svg viewBox="0 0 300 194" class="tpitch" role="img" aria-label="${esc(lab)}">
  <rect x="0.5" y="0.5" width="299" height="193" class="tpl"/><line x1="150" y1="0" x2="150" y2="194" class="tpl"/>
  <rect x="0.5" y="${(68 / 2 - 20.16) * PS}" width="${16.5 * PS}" height="${40.32 * PS}" class="tpl"/>
  <rect x="${300 - 16.5 * PS}" y="${(68 / 2 - 20.16) * PS}" width="${16.5 * PS}" height="${40.32 * PS}" class="tpl"/>
  ${inner}<text x="296" y="190" text-anchor="end" class="hint2">attack →</text></svg>`;
const zrect = (z) => ({ x: Math.floor(z / 4) * 17.5 * PS, y: (68 - (z % 4 + 1) * 17) * PS, w: 17.5 * PS, h: 17 * PS });
const zc = (z) => { const r = zrect(z); return [r.x + r.w / 2, r.y + r.h / 2]; };

function zoneMap(vals, ref, lab) {
  const mx = Math.max(...vals, 1e-6);
  return pitchSvg(vals.map((v, z) => {
    const r = zrect(z), k = ref ? v / Math.max(ref[z], 1e-4) : null;
    return `<rect x="${r.x.toFixed(1)}" y="${r.y.toFixed(1)}" width="${r.w.toFixed(1)}" height="${r.h.toFixed(1)}" class="tpz" style="opacity:${(0.08 + 0.8 * v / mx).toFixed(2)}"/>`
      + (v / mx > 0.15 ? `<text x="${(r.x + r.w / 2).toFixed(1)}" y="${(r.y + r.h / 2 + 3).toFixed(1)}" text-anchor="middle" class="tpzt">${k != null ? k.toFixed(1) + 'x' : ''}</text>` : '');
  }).join(''), lab);
}

function laneMap(lanes) {
  const mx = Math.max(...lanes.map((l) => l.share));
  return pitchSvg(`<defs><marker id="tpah" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L6,3 L0,6 z" class="tpah"/></marker></defs>`
    + lanes.slice(0, 6).map((l, i) => {
      const [x0, y0] = zc(l.z0), [x1, y1] = zc(l.z1), same = l.z0 === l.z1;
      return same ? `<circle cx="${x0}" cy="${y0}" r="${(4 + 10 * l.share / mx).toFixed(1)}" class="tparr" style="stroke-width:2"/><text x="${x0}" y="${y0 + 3}" text-anchor="middle" class="tpzt">${i + 1}</text>`
        : `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" class="tparr" style="stroke-width:${(1.5 + 4 * l.share / mx).toFixed(1)}" marker-end="url(#tpah)"/><text x="${(x0 + x1) / 2}" y="${(y0 + y1) / 2 - 3}" text-anchor="middle" class="tpzt">${i + 1}</text>`;
    }).join(''), 'Threat lanes');
}

function targetCard(T) {
  const cls = T.m1.classes, p1 = (v) => `${v >= 0 ? '+' : ''}${(100 * v).toFixed(1)}`;
  const sideName = { none: 'nobody within 5 m', front: 'from the front', left: 'from his left', right: 'from his right', behind: 'from behind' };
  let h = `<div class="pomsub"><b>${esc(T.name)}</b> · in every dataset: ${T.contexts.map((c) => `${esc(c.label)} (${c.matches})`).join(' · ')}</div>
    <h3 class="pfh">Game plan <span class="hint">one line per model</span></h3>
    <ol class="tprec">${T.recommendations.map((r) => `<li><b>${esc(r.model)}</b> ${esc(r.text)}</li>`).join('')}</ol>`;
  // M1
  h += `<h3 class="pfh">M1 · What will he do with the ball? <span class="hint">his share minus a typical DM / CM in the same situations, pts</span></h3>
    <table class="pfct"><tr><th></th>${cls.map((c) => `<th>${c}</th>`).join('')}</tr>${T.m1.contexts.map((c) => `<tr><td>${esc(c.context)} <span class="hint">${c.decisions}</span></td>${c.real.map((v, k) => {
      const d = v - c.typical[k]; return `<td class="${d > 0.02 ? 'up' : d < -0.02 ? 'down' : ''}">${p1(d)}</td>`; }).join('')}</tr>`).join('')}</table>
    <div class="hint">Held out: ${Object.entries(T.m1.held_out).map(([k, v]) => `${esc(k)} ${v.gain_pct >= 0 ? '+' : ''}${v.gain_pct.toFixed(2)}% (${esc(v.verdict)})`).join(' · ')}. Style correlates across contexts r ${T.m1.style_r.toFixed(2)}. Most alike in 2017/18: ${T.m1.similar.slice(0, 3).map(esc).join(', ')}.</div>`;
  // M2
  const m2 = T.m2;
  h += `<h3 class="pfh">M2 · Where does he create threat? <span class="hint">expected threat (xT)</span></h3>
    <div class="pftest good">${m2.xt90.toFixed(3)} xT per 90 · percentile <b>${m2.pct.toFixed(0)}</b> of ${m2.pop_n} DM / CM (median ${m2.pop_median.toFixed(3)}) · Arsenal 17/18 passes: percentile ${m2.wy_pct.toFixed(0)} of ${m2.wy_pop_n}</div>
    ${zoneMap(m2.zones_his, m2.zones_pop, 'Where his threat starts')}
    <div class="hint">Shade = xT he adds per 90 from each zone; label = times a typical DM / CM.</div>
    <table class="pfct"><tr><th></th><th>xT pass /90</th><th>progressive /90</th><th>into final third /90</th><th>completion</th></tr>
    ${m2.contexts.map((c) => `<tr><td>${esc(c.context)} <span class="hint">${c.data}</span></td><td>${c.xt_pass90.toFixed(3)}</td><td>${c.prog90.toFixed(1)}</td><td>${c.final_third90.toFixed(1)}</td><td>${(100 * c.completion).toFixed(0)}%</td></tr>`).join('')}</table>`;
  // M3
  h += `<h3 class="pfh">M3 · Which lanes carry the threat?</h3>${laneMap(T.m3.lanes)}
    <table class="pfct"><tr><th>#</th><th>lane</th><th>/90</th><th>share</th><th>left foot</th><th>receivers</th></tr>
    ${T.m3.lanes.slice(0, 6).map((l, i) => `<tr><td>${i + 1}</td><td>${esc(l.from)} → ${esc(l.to)}</td><td>${l.per90.toFixed(1)}</td><td>${(100 * l.share).toFixed(0)}%</td><td>${(100 * l.left_foot).toFixed(0)}%</td><td class="hint">${l.receivers.slice(0, 2).map((r) => esc(r[0])).join(', ')}</td></tr>`).join('')}</table>
    <div class="hint">Top 8 lanes = ${(100 * T.m3.top_share).toFixed(0)}% of the xT he adds · left foot ${(100 * T.m3.left_foot).toFixed(0)}% · switches ${T.m3.switch90.toFixed(1)} per 90.</div>`;
  // M4
  h += `<h3 class="pfh">M4 · From which side do we press him? <span class="hint">nearest opponent in the 360 frame</span></h3>
    <table class="pfct"><tr><th>pressure</th><th>passes</th><th>completion</th><th>xT / pass</th><th>vs typical</th></tr>
    ${T.m4.sides.map((s) => `<tr${s.side === T.m4.best ? ' class="tpbest"' : ''}><td>${sideName[s.side]}</td><td>${s.n}</td><td>${(100 * s.completion).toFixed(0)}% <span class="hint">(${(100 * s.pop_completion).toFixed(0)}%)</span></td><td>${(1000 * s.xt).toFixed(1)}e-3</td><td class="${s.side === 'none' ? '' : s.rel_completion < 0 ? 'down' : 'up'}">${s.side === 'none' ? '' : p1(s.rel_completion) + ' pts'}</td></tr>`).join('')}</table>
    <div class="hint">"vs typical": his drop from free minus a typical DM / CM's drop. Highlighted = the side that costs him most threat.</div>`;
  // M5
  const m5 = T.m5;
  h += `<h3 class="pfh">M5 · Where can we win the ball from him?</h3>
    <div class="pftest">Ball-loss model on his held-out matches: ${m5.gain_pct >= 0 ? '+' : ''}${m5.gain_pct.toFixed(1)}% log-loss vs typical, AUC ${m5.auc[0].toFixed(2)} → ${m5.auc[1].toFixed(2)} · predicted ${(100 * m5.calib[0]).toFixed(1)}% vs real ${(100 * m5.calib[1]).toFixed(1)}%</div>
    <table class="pfct"><tr><th>zone</th><th>touches /90</th><th>risk free</th><th>risk pressed</th><th>regain worth</th></tr>
    ${m5.zones.slice(0, 5).map((z) => `<tr><td>${esc(z.zone)}</td><td>${z.per90.toFixed(1)}</td><td>${(100 * z.risk_none).toFixed(0)}%</td><td>${(100 * z['risk_' + z.best_side]).toFixed(0)}% <span class="hint">${sideName[z.best_side]}</span></td><td>${z.ours_xt.toFixed(3)} xT</td></tr>`).join('')}</table>`;
  // M6
  const m6 = T.m6;
  h += `<h3 class="pfh">M6 · Where is he without the ball? <span class="hint">World Cup 2022 tracking</span></h3>
    <div class="hint">Mean position: in possession (${m6.mean_in[0].toFixed(0)}, ${m6.mean_in[1].toFixed(0)}) m, out of possession (${m6.mean_out[0].toFixed(0)}, ${m6.mean_out[1].toFixed(0)}) m. His position model vs typical, held out: ${m6.err_typical.toFixed(2)} → ${m6.err_his.toFixed(2)} m.</div>
    <table class="pfct"><tr><th>phase</th><th>ball at</th><th>him vs typical DM / CM</th></tr>
    ${m6.grid.filter((g) => g.ball[1] === 34 || Math.hypot(g.dx, g.dy) > 4).map((g) => `<tr><td>${esc(g.phase)}</td><td>(${g.ball[0].toFixed(0)}, ${g.ball[1].toFixed(0)})</td><td>${g.dx >= 0 ? '+' : ''}${g.dx.toFixed(1)} m forward, ${g.dy >= 0 ? '+' : ''}${g.dy.toFixed(1)} m across</td></tr>`).join('')}</table>`;
  // M7
  const mx7 = Math.max(...T.m7.plans.map((p) => Math.abs(p.net100)));
  h += `<h3 class="pfh">M7 · What-if counter plan <span class="hint">net threat per 100 of his actions</span></h3>
    ${T.m7.plans.map((p) => `<div class="gnrow"><span${p.plan === T.m7.best ? ' class="tpbestt"' : ''}>${esc(p.plan)}</span><div class="cbar"><span class="${p.plan === T.m7.best ? 'cp' : 'cr'}" style="width:${(100 * Math.max(p.net100, 0) / mx7).toFixed(1)}%"></span></div><b>${p.net100.toFixed(2)}</b></div>`).join('')}
    <div class="hint">Net = threat he still creates (choice mix × value × no loss) minus the counter value of balls we win. Model answers, not an experiment.</div>`;
  if (T.vaep) {
    const v = T.vaep;
    h += `<h3 class="pfh">Upgrade · threat, lanes and what-if with VAEP <span class="hint">possession value model instead of the static xT grid</span></h3>
      <div class="pftest good">VAEP per 90 ${v.sb_vaep90.toFixed(3)} · percentile ${v.sb_pct.toFixed(0)} of ${v.sb_n} DM / CM (StatsBomb) · Arsenal 17/18 percentile ${v.pl_pct.toFixed(0)} of ${v.pl_n}</div>
      <table class="pfct"><tr><th>context</th><th>VAEP /90</th><th>attack</th><th>defence</th></tr>${v.contexts.map((c) => `<tr><td>${esc(c.context)}</td><td>${c.vaep90.toFixed(3)}</td><td>${c.off90.toFixed(3)}</td><td>${c.def90.toFixed(3)}</td></tr>`).join('')}</table>
      ${laneMap(v.lanes.map((l) => ({ ...l, share: Math.abs(l.share) })))}
      <table class="pfct"><tr><th>plan</th><th>his value /100 actions</th><th>vs letting him play</th></tr>${v.plans.map((p) => `<tr${p.plan === v.best ? ' class="tpbest"' : ''}><td>${esc(p.plan)}</td><td>${p.vaep100 >= 0 ? '+' : ''}${p.vaep100.toFixed(3)}</td><td>${p.vs_free >= 0 ? '+' : ''}${p.vs_free.toFixed(0)}%</td></tr>`).join('')}</table>`;
  }
  if (T.coach) h += coachCard(T.coach);
  return h;
}

// ---- target player, coach view: 3 models over his club league matches (scripts/target_coach.py viewer)
function coachCard(C) {
  const s3 = (v) => `${v >= 0 ? '+' : ''}${v.toFixed(3)}`, pt = (v) => `${v >= 0 ? '+' : ''}${(100 * v).toFixed(1)}`;
  let h = `<h3 class="pfh">Coach view <span class="hint">his club league matches: Arsenal 17/18 + Leverkusen 23/24</span></h3>
    <ol class="tprec">${C.plan.map((r) => `<li><b>${esc(r.model)}</b> ${esc(r.text)}</li>`).join('')}</ol>`;
  // C1
  const c1 = C.c1;
  h += `<h3 class="pfh">C1 · Against which opponents? <span class="hint">${c1.n} matches, opponent style from its other matches</span></h3>
    <table class="pfct"><tr><th>opponent style</th><th>matches</th><th>threat /90</th><th>forward</th><th>loss</th><th>height</th><th>best zone</th></tr>
    ${c1.styles.map((s) => `<tr${s.label === c1.best ? ' class="tpbest"' : ''}><td>${esc(s.label)}<div class="hint">${s.opponents.slice(0, 5).map(esc).join(', ')}</div></td><td>${s.matches}</td>
      <td class="${s.rel.xt90 > 0 ? 'up' : 'down'}">${s3(s.rel.xt90)}</td><td>${pt(s.rel.fwd)}</td><td>${pt(s.rel.loss)}</td><td>${s.rel.height >= 0 ? '+' : ''}${s.rel.height.toFixed(1)} m</td><td class="hint">${esc(s.zones[0].zone)}</td></tr>`).join('')}</table>
    <div class="hint">Differences from his own league average. Held out, one match at a time: opponent style changes the error of his threat per 90 by ${c1.tests.xt90.gain_pct.toFixed(1)}% (a tendency, not a forecast).</div>`;
  // C2
  const c2 = C.c2, mx = Math.max(...c2.whatif.map((g) => g.xt90));
  const cellc = (g) => `<td style="background:rgba(242,193,78,${(0.08 + 0.5 * g.xt90 / mx).toFixed(2)})">${g.xt90.toFixed(3)}<div class="hint">fwd ${(100 * g.fwd).toFixed(0)}% · loss ${(100 * g.loss).toFixed(0)}%</div></td>`;
  h += `<h3 class="pfh">C2 · At which moment of the match? <span class="hint">${c2.minutes} minutes: leading ${(100 * c2.share.leading).toFixed(0)}%, drawing ${(100 * c2.share.drawing).toFixed(0)}%, trailing ${(100 * c2.share.trailing).toFixed(0)}%</span></h3>
    <table class="pfct"><tr><th>minute</th><th>trailing</th><th>drawing</th><th>leading</th></tr>
    ${[10, 30, 50, 70, 85].map((m) => `<tr><td>${m}'</td>${['trailing', 'drawing', 'leading'].map((s) => cellc(c2.whatif.find((g) => g.minute === m && g.state === s))).join('')}</tr>`).join('')}</table>
    <div class="hint">What-if: his threat per 90 (shade), forward pass share and pass loss at that minute and score. Held out by match, minute + score improve forward-pass prediction by ${c2.tests['forward pass'].gain_pct.toFixed(2)}% and pass loss by ${c2.tests['pass loss'].gain_pct.toFixed(2)}%: real but small.</div>
    <table class="pfct"><tr><th>period</th><th>threat /90</th><th>actions /90</th><th>forward</th><th>loss</th></tr>
    ${c2.by_bucket.map((b) => `<tr><td>${b.bucket}</td><td>${b.xt90.toFixed(3)}</td><td>${b.touches90.toFixed(0)}</td><td>${(100 * b.fwd).toFixed(0)}%</td><td>${(100 * b.loss).toFixed(1)}%</td></tr>`).join('')}</table>`;
  // C3
  const c3 = C.c3;
  h += `<h3 class="pfh">C3 · With which teammates? <span class="hint">270+ min with and without, ${c3.mates.length} teammates</span></h3>
    <table class="pfct"><tr><th>teammate</th><th>min with / without</th><th>threat /90 with vs without</th><th>90%</th></tr>
    ${c3.mates.filter((m, i, a) => m.clear || i < 3 || i >= a.length - 3).map((m) => `<tr${m.clear ? ' class="tpbest"' : ''}><td>${esc(m.name)} <span class="hint">${m.league === 'Premier League' ? 'Arsenal' : 'Leverkusen'}</span></td><td>${m.with_min} / ${m.without_min}</td>
      <td class="${m.diff_xt90 > 0 ? 'up' : 'down'}">${m.with_xt90.toFixed(3)} vs ${m.without_xt90.toFixed(3)}</td><td class="hint">${s3(m.ci[0])} .. ${s3(m.ci[1])}</td></tr>`).join('')}</table>
    <div class="hint">Highlighted = 90% interval excludes 0 (${c3.mates.filter((m) => m.clear).length} of ${c3.mates.length}; about ${Math.round(0.1 * c3.mates.length)} expected by chance). Joint model of all teammates, held out by match: ${c3.tests.gain_pct.toFixed(1)}% vs his average. With / without also splits the season, so check on video.</div>`;
  return h;
}

// ---- any-player report and backtest (scripts/player_report.py)
async function loadRep() {
  if (!R.rep) { try { const r = await fetch('data/reports.json', { cache: 'no-store' }); if (r.ok) R.rep = await r.json(); } catch (e) { R.rep = null; } }
  if (!R.bt) { try { const r = await fetch('data/backtest.json', { cache: 'no-store' }); if (r.ok) R.bt = await r.json(); } catch (e) { R.bt = null; } }
}

function styleBars(st, classes) {
  if (!st) return '<div class="hint">Not enough on-ball decisions for a style model (300+ needed).</div>';
  return `<table class="pfct"><tr><th>choice</th><th>his share (90% band)</th><th>typical ${esc(st.role)}</th><th>when pressed</th></tr>${classes.map((c, k) => {
    const d = st.his[k] - st.typical[k], sure = st.lo[k] > st.typical[k] || st.hi[k] < st.typical[k];
    return `<tr><td>${c}</td><td class="${sure ? (d > 0 ? 'up' : 'down') : ''}">${(100 * st.his[k]).toFixed(1)}% <span class="hint">${(100 * st.lo[k]).toFixed(1)}-${(100 * st.hi[k]).toFixed(1)}</span></td>
      <td>${(100 * st.typical[k]).toFixed(1)}%</td><td>${st.pressed_delta[k] >= 0 ? '+' : ''}${(100 * st.pressed_delta[k]).toFixed(1)} pts</td></tr>`; }).join('')}</table>
    <div class="hint">Coloured = his 90% band does not include the typical value: a real habit, not noise. ${st.decisions} decisions.</div>`;
}

function reportCard(D, key) {
  const p = D.players[key];
  if (!p) return '<div class="hint">No report.</div>';
  const pos = p.zones.map((v) => Math.max(v, 0));
  let h = `<div class="pftest ${p.pct >= 75 ? 'good' : ''}"><b>${esc(p.name)}</b> · ${esc(p.team)} · ${esc(p.comp)} · ${esc(p.role)} · ${p.minutes} min · ${esc(p.data)}</div>
    <h3 class="pfh">Value <span class="hint">VAEP: change in the chance to score minus the chance to concede, per action</span></h3>
    <div class="gnrow"><span>VAEP per 90</span><div class="cbar"><span class="cp" style="width:${p.pct == null ? 0 : p.pct.toFixed(0)}%"></span></div><b>${p.vaep90.toFixed(3)}</b></div>
    <div class="hint">${p.pct == null ? 'too few minutes for a percentile' : `percentile ${p.pct.toFixed(0)} of ${p.pct_n} ${esc(p.role)} (${esc(p.pct_group)})`} · attack ${p.off90.toFixed(3)} · defence ${p.def90.toFixed(3)} · location-only model ${p.loc90.toFixed(3)}</div>
    ${zoneMap(pos, null, 'Where his value comes from')}
    <div class="hint">Shade = VAEP per 90 from actions starting in each zone. Most valuable actions: ${p.kinds.map((k) => `${esc(k[0])} ${k[1] >= 0 ? '+' : ''}${k[1].toFixed(3)}`).join(' · ')}</div>
    <h3 class="pfh">On-ball style <span class="hint">Bayesian player model</span></h3>${styleBars(p.style, D.classes)}`;
  if (p.model) h += `<div class="hint">Player model vs role on matches it never saw: ${p.model.gain_pct >= 0 ? '+' : ''}${p.model.gain_pct.toFixed(2)}% (${esc(p.model.verdict)}, ${p.model.matches} matches).</div>`;
  if (p.similar && p.similar.length) h += `<h3 class="pfh">Plays most like</h3><div class="hint">${p.similar.map(esc).join(' · ')}</div>`;
  return h;
}

function calSvg(cal1, cal0) {
  const w = 300, hh = 170, x0 = 34, y0 = 146, mx = Math.max(...cal1.map((c) => Math.max(c[0], c[1])), ...cal0.map((c) => Math.max(c[0], c[1])));
  const X = (v) => x0 + (w - x0 - 10) * v / mx, Y = (v) => y0 - (y0 - 10) * v / mx;
  const pl = (c, cls) => `<polyline points="${c.map((q) => `${X(q[0]).toFixed(1)},${Y(q[1]).toFixed(1)}`).join(' ')}" class="cline ${cls}"/>` + c.map((q) => `<circle cx="${X(q[0]).toFixed(1)}" cy="${Y(q[1]).toFixed(1)}" r="2.4" class="cdot ${cls}"/>`).join('');
  return `<svg viewBox="0 0 ${w} ${hh}" class="pfcurve" role="img" aria-label="Calibration"><line x1="${X(0)}" y1="${Y(0)}" x2="${X(mx)}" y2="${Y(mx)}" class="rgrid rmed"/>${pl(cal0, 'wy')}${pl(cal1, 'lv')}
    <text x="${(x0 + w) / 2}" y="${hh - 2}" text-anchor="middle" class="hint2">predicted chance to score (next 10 actions)</text><text x="${x0 - 4}" y="${Y(mx) + 4}" text-anchor="end" class="hint2">${(100 * mx).toFixed(0)}%</text></svg>
    <div class="pflegend"><span class="lg lv"></span> full model <span class="lg wy"></span> location only · diagonal = perfect</div>`;
}

function backtestCard(B) {
  const v = B.value, f = (x, d = 3) => x == null ? '-' : x.toFixed(d);
  let h = `<div class="pomsub">Every number here comes from matches the model never saw while learning.</div>
    <h3 class="pfh">1 · Possession value (VAEP) <span class="hint">does the team score within 10 actions?</span></h3>
    <table class="pfct"><tr><th>data</th><th>model</th><th>AUC score</th><th>Brier skill</th><th>AUC concede</th></tr>
    ${Object.entries(v).map(([t, q]) => ['location', 'full'].map((k) => `<tr${k === 'full' ? ' class="tpbest"' : ''}><td>${t === 'sb' ? 'StatsBomb 360' : 'Wyscout'}</td><td>${k === 'full' ? 'full (context' + (t === 'sb' ? ' + 360' : '') + ')' : 'location only (xT-like)'}</td><td>${f(q[k].scores.auc)}</td><td>${f(q[k].scores.brier_skill)}</td><td>${f(q[k].concedes.auc)}</td></tr>`).join('')).join('')}</table>
    ${v.sb ? calSvg(v.sb.full.scores.calibration, v.sb.location.scores.calibration) : ''}`;
  if (B.ratings) h += `<table class="pfct"><tr><th>rating reliability</th><th>VAEP</th><th>location only</th></tr>
    ${Object.entries(B.ratings).map(([t, q]) => Object.keys(q.vaep || {}).map((s) => `<tr><td>${t === 'sb' ? 'StatsBomb' : 'Wyscout'} · ${s === 'odd_even' ? 'odd vs even matches' : '1st half of season → 2nd'}</td><td>r ${f(q.vaep[s].r, 2)} <span class="hint">${q.vaep[s].players}</span></td><td>r ${f((q.vaep_loc[s] || {}).r, 2)}</td></tr>`).join('')).join('')}</table>`;
  const s = B.bayes.season, c = B.bayes.curve;
  if (s) h += `<h3 class="pfh">2 · Bayesian player model <span class="hint">${s.players} players, last quarter of the season</span></h3>
    <table class="pfct"><tr><th>model</th><th>log-loss</th><th>gain vs role</th><th>beats role for</th></tr>
      <tr><td>role only</td><td>${f(s.logloss.role, 4)}</td><td></td><td></td></tr>
      <tr><td>old player model</td><td>${f(s.logloss.old, 4)}</td><td>${s.gain_old_pct.toFixed(2)}%</td><td>${(100 * s.share_beats_role.old).toFixed(0)}%</td></tr>
      <tr class="tpbest"><td>Bayesian player model</td><td>${f(s.logloss.bayes, 4)}</td><td>${s.gain_bayes_pct.toFixed(2)}%</td><td>${(100 * s.share_beats_role.bayes).toFixed(0)}%</td></tr></table>
    <div class="hint">90% bands contain the real share: ${Object.entries(s.coverage).map(([k, x]) => `${k} ${(100 * x).toFixed(0)}%`).join(' · ')}</div>`;
  if (c) h += `<table class="pfct"><tr><th>matches seen</th><th>old</th><th>Bayes</th><th>Bayes ≥ old</th></tr>${c.map((q) => `<tr><td>${q.n}</td><td>${q.old >= 0 ? '+' : ''}${q.old.toFixed(2)}%</td><td class="${q.bayes > q.old ? 'up' : ''}">${q.bayes >= 0 ? '+' : ''}${q.bayes.toFixed(2)}%</td><td>${(100 * q.share).toFixed(0)}%</td></tr>`).join('')}</table>`;
  const bl = B.ball_loss;
  h += `<h3 class="pfh">3 · Other models</h3><table class="pfct">
    <tr><td>Ball-loss model (target player)</td><td>${bl.gain_pct.toFixed(1)}% log-loss · AUC ${bl.auc[1].toFixed(2)} · predicted ${(100 * bl.calib[0]).toFixed(1)}% vs real ${(100 * bl.calib[1]).toFixed(1)}%</td></tr>
    ${B.movement ? `<tr><td>Movement GNN, World Cup 2022</td><td>${B.movement.still.toFixed(2)} m (nobody moves) → ${B.movement.bundesliga.toFixed(2)} → ${B.movement.worldcup.toFixed(2)} m</td></tr>` : ''}
    <tr><td>Player vs role at scale (Wyscout)</td><td>${B.decision_eval.wyscout.better} better · ${B.decision_eval.wyscout.unclear} unclear · ${B.decision_eval.wyscout.worse} worse of ${B.decision_eval.wyscout.players}</td></tr></table>`;
  return h;
}

// ---- show 3, find all: example-based moment search (scripts/moment_search.py presets)
async function loadMs() {
  if (R.ms === undefined) { try { const r = await fetch('data/moment_search.json', { cache: 'no-store' }); R.ms = r.ok ? await r.json() : null; } catch (e) { R.ms = null; } }
}
const MSW = { pressed: 'under pressure', dy_abs: 'goes across the pitch', len: 'long', dx: 'goes forward', ok: 'succeeds', ey: 'ends on one side',
  ex: 'ends high up', x: 'starts high up', end_box: 'ends in the box', in_box: 'starts in the box', goal_d_end: 'ends near goal', gain3: 'team gains ground right after',
  shot5: 'leads to a shot', lost3: 'ball lost right after', near_opp: 'space around him', opp5: 'opponents close', goal_side: 'opponents goal-side', mates_ahead: 'mates ahead' };
function msWord(f) {
  const n = f.feature, hi = f.dir === 'high';
  if (n.startsWith('kind=')) return (hi ? '' : 'not a ') + n.slice(5).toLowerCase();
  const m = n.match(/^(prev|next)(\d)_(\w+)$/);
  if (m) {
    const when = m[1] === 'prev' ? `${m[2]} before` : `${m[2]} after`, w = m[3];
    const what = { same: 'own team on the ball', other: 'opponent on the ball', pass: 'a pass', carry: 'a carry', shot: 'a shot', win: 'a ball win', lose: 'a ball loss', set: 'a set piece', x: 'ball high up', dt: 'slow' }[w] || w;
    return `${hi ? what : 'no ' + what.replace(/^an? /, '')} (${when})`;
  }
  return (hi ? '' : 'not ') + (MSW[n] || n);
}
function msPitch(list, on, ex) {
  const X = (p) => (p[0] * PS).toFixed(1), Y = (p) => (p[1] * PS).toFixed(1);
  let g = '';
  list.forEach((m, j) => {
    const sel = j === on, cls = ex ? (m.neg ? 'msneg' : 'mspos') : (sel ? 'mspos' : 'mshit');
    if (sel || ex) g += m.ctx.map((c) => `<line x1="${X(c.a)}" y1="${Y(c.a)}" x2="${X(c.b)}" y2="${Y(c.b)}" class="msctx ${c.same ? 'own' : 'opp'}"${c.kind === 'Carry' ? ' stroke-dasharray="3 2"' : ''}/>`).join('');
    g += `<g class="msm ${cls}${on >= 0 && !sel && !ex ? ' dim' : ''}" data-j="${j}"><line x1="${X(m.a)}" y1="${Y(m.a)}" x2="${X(m.b)}" y2="${Y(m.b)}"${m.kind === 'Carry' ? ' stroke-dasharray="4 3"' : ''}/>
      <circle cx="${X(m.a)}" cy="${Y(m.a)}" r="${ex || sel ? 5 : 3}"/>${ex || sel ? `<text x="${X(m.a)}" y="${(+Y(m.a) + 2.6).toFixed(1)}" text-anchor="middle">${m.neg ? '✕' : j + 1}</text>` : ''}</g>`;
  });
  return pitchSvg(g, 'Moments');
}
function msCard(D) {
  const pr = D.presets, k = Math.min(R.msk || 0, pr.length - 1), p = pr[k], t = p.test, pc = (x) => `${(100 * x).toFixed(0)}%`;
  const exs = p.examples.concat(p.counter.map((m) => ({ ...m, neg: 1 })));
  const on = R.msh == null ? -1 : R.msh;
  let h = `<div class="pomsub">Show 3, find all: a coach marks 3 moments he likes (here 3 of Xhaka's own) and 1 he does not.
      The search finds what they have in common and looks through all 472k StatsBomb actions (200 matches, 360 frames).</div>
    <div class="mstabs">${pr.map((q, j) => `<button data-k="${j}" class="${j === k ? 'on' : ''}">${esc(q.name)}</button>`).join('')}</div>
    <h3 class="pfh">The examples <span class="hint">gold = liked, red ✕ = not liked · faint = 2 actions before and 3 after</span></h3>
    ${msPitch(exs, -1, true)}
    <div class="hint">${exs.map((m, j) => `${m.neg ? '✕' : j + 1}: ${esc(m.player)} ${m.min}' · ${esc(m.source)} · ${m.kind}${m.pressed ? ' · pressed' : ''} · VAEP ${m.vaep >= 0 ? '+' : ''}${m.vaep.toFixed(3)}`).join('<br>')}</div>
    <h3 class="pfh">What they have in common</h3><div class="mschips">${p.common.map((c) => `<span>${esc(msWord(c))}</span>`).join('')}</div>
    <div class="pftest good"><b>${p.found.toLocaleString()}</b> moments found · ${pc(p.fit200)} of the top 200 fit the idea${t ? ` · on unseen matches ${pc(t.p50.full)} of the top 50 were right (plain distance ${pc(t.p50.proto)}, random ${pc(t.base)})` : ''}</div>
    <h3 class="pfh">Top moments found <span class="hint">click one to see what happened before and after</span></h3>
    ${msPitch(p.moments, on, false)}
    <div class="pfms">${p.moments.slice(0, 10).map((m, j) => `<div class="pfm${j === on ? ' on' : ''}" data-j="${j}"><span class="pfn">${j + 1}</span>
      ${esc(m.player)} <span class="hint">${esc(m.team)} · ${esc(m.source)} · ${m.min}'</span><br><span class="hint">${m.kind}${m.pressed ? ' · pressed' : ''}${m.ok ? '' : ' · failed'}</span>
      <span class="pfadd ${m.vaep >= 0 ? 'good' : 'bad'}">${m.vaep >= 0 ? '+' : ''}${m.vaep.toFixed(3)}</span></div>`).join('')}</div>
    <h3 class="pfh">Who does this best <span class="hint">270+ min, 5+ moments · VAEP per 90 from these moments</span></h3>
    <table class="pfct"><tr><th>player</th><th>team</th><th>per 90</th><th>success</th><th>VAEP/90</th></tr>${p.players.slice(0, 10).map((q) => `<tr><td>${esc(q.player)}</td><td class="hint">${esc(q.team)}</td><td>${q.per90.toFixed(2)}</td><td>${pc(q.ok)}</td><td class="${q.vaep90 >= 0 ? 'up' : 'down'}">${q.vaep90 >= 0 ? '+' : ''}${q.vaep90.toFixed(3)}</td></tr>`).join('')}</table>
    <div class="hint">Granit Xhaka: ${p.xhaka_rank ? `rank ${p.xhaka_rank} of ${p.ranked}` : 'not ranked (too few moments)'} on this kind of moment.</div>`;
  const ev = D.eval;
  if (ev && ev.rows) h += `<h3 class="pfh">Accuracy <span class="hint">examples from ${ev.matches_examples} matches, search on the other ${ev.matches_search}; share of the top 50 that are right</span></h3>
    <table class="pfct"><tr><th>pattern</th><th>random</th><th>plain distance</th><th>our search</th></tr>${ev.rows.map((r) => `<tr><td>${esc(r.pattern)}</td><td>${pc(r.base)}</td><td>${pc(r.p50.proto)}</td><td class="${r.p50.full > r.p50.proto ? 'up' : ''}">${pc(r.p50.full)}</td></tr>`).join('')}
    <tr class="tpbest"><td>average of 12</td><td>${pc(ev.avg_p50.random)}</td><td>${pc(ev.avg_p50.proto)}</td><td>${pc(ev.avg_p50.full)}</td></tr></table>
    <div class="hint">Test patterns are written as rules so the answer is known; a coach's real idea is fuzzier. Your own search: python scripts/moment_search.py query --pos ID,ID,ID --neg ID</div>`;
  return h;
}

async function open() {
  await load();
  $('prof').classList.toggle('hidden');
  if (!$('prof').classList.contains('hidden')) $('pom')?.classList.add('hidden');
  render();
}
$('profBtn').addEventListener('click', open);
$('profClose').addEventListener('click', () => $('prof').classList.add('hidden'));
$('pomBtn')?.addEventListener('click', () => $('prof').classList.add('hidden'));
window.__profile = { R, load, render, open };
