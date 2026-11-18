// Live threat for the viewer: the state logistic model of scripts/value_models.py (chance that the team with the ball
// shoots within 10 s, seeing where the ball and all 22 players are), ported to JavaScript. Weights: data/threat_model.json
// (a copy of outputs/value_models/threat_model.json). Calibrated on held-out World Cup matches (ECE 0.003).
//
// Who has the ball: the team of the player nearest to it (within 3 m; otherwise the last team that had it).
// Direction: each team attacks away from its own keeper (slot 0 = home keeper, slot 11 = away keeper).

const L = 105, W = 68;
const GOAL = [L, W / 2], P0 = [L, W / 2 - 3.66], P1 = [L, W / 2 + 3.66];
let M = null;

export async function loadThreat() {
  try {
    const r = await fetch('data/threat_model.json', { cache: 'no-store' });
    if (r.ok) M = await r.json();
  } catch (e) { M = null; }
  return !!M;
}

function inTriangle(p, a, b, c) {
  const s = (p1, p2, p3) => (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
  const d1 = s(p, a, b), d2 = s(p, b, c), d3 = s(p, c, a);
  return !(((d1 < 0) || (d2 < 0) || (d3 < 0)) && ((d1 > 0) || (d2 > 0) || (d3 > 0)));
}

function features(b, att, dfs) {
  const dist = Math.max(Math.hypot(GOAL[0] - b[0], GOAL[1] - b[1]), 0.5);
  const a = [P0[0] - b[0], P0[1] - b[1]], c = [P1[0] - b[0], P1[1] - b[1]];
  const angle = Math.abs(Math.atan2(a[0] * c[1] - a[1] * c[0], a[0] * c[0] + a[1] * c[1]));
  const dd = dfs.length ? dfs.map((d) => Math.hypot(d[0] - b[0], d[1] - b[1])) : [15];
  let free = 0;
  for (const p of att) {
    const inBox = p[0] > L - 16.5 && Math.abs(p[1] - W / 2) < 20.16;
    const near = dfs.length ? Math.min(...dfs.map((d) => Math.hypot(p[0] - d[0], p[1] - d[1]))) : 99;
    if (inBox && near > 3) free++;
  }
  return {
    s_dist: dist, s_log_dist: Math.log(dist), s_angle: angle, s_bx: b[0], s_by_c: Math.abs(b[1] - W / 2),
    s_def_goal_side: dfs.filter((d) => d[0] > b[0]).length,
    s_def_in_cone: dfs.filter((d) => inTriangle(d, b, P0, P1)).length,
    s_def_5m: dd.filter((x) => x < 5).length, s_def_10m: dd.filter((x) => x < 10).length,
    s_near_def: Math.min(Math.min(...dd), 15),
    s_att_ahead: att.filter((p) => p[0] > b[0] + 1).length,
    s_best_mate_goal: att.length ? Math.min(Math.min(...att.map((p) => Math.hypot(p[0] - GOAL[0], p[1] - GOAL[1]))), 60) : 60,
    s_free_in_box: free,
  };
}

// pts: 22 entries [x, y] in metres or null (0-10 home, 11-21 away); ball [x, y]; state: {team} remembered between calls.
// Returns {team: 'home'|'away', p: 0..1} or null.
export function threatOf(pts, ball, state) {
  if (!M || !ball || !isFinite(ball[0])) return null;
  let best = 99, near = null;
  pts.forEach((p, k) => { if (p) { const d = Math.hypot(p[0] - ball[0], p[1] - ball[1]); if (d < best) { best = d; near = k; } } });
  if (near !== null && (best < 3 || !state.team)) state.team = near < 11 ? 'home' : 'away';   // nobody close yet: nearest team
  if (!state.team) return null;
  const a0 = state.team === 'home' ? 0 : 11, d0 = 11 - a0;
  const keeper = pts[a0];
  const flip = keeper ? keeper[0] > L / 2 : false;
  const f = (p) => (flip ? [L - p[0], W - p[1]] : p);
  const att = pts.slice(a0, a0 + 11).filter(Boolean).map(f);
  const dfs = pts.slice(d0, d0 + 11).filter(Boolean).map(f);
  const x = features(f(ball), att, dfs);
  let z = M.weights[0];
  M.features.forEach((name, i) => { z += M.weights[i + 1] * (x[name] - M.mean[i]) / M.std[i]; });
  return { team: state.team, p: 1 / (1 + Math.exp(-z)) };
}
