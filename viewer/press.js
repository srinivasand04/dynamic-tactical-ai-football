// Live pressing for the viewer: Pressing Intensity of unravelsports (Bekkers 2024, BSD-3-Clause; time_to_intercept and
// probability_to_intercept in unravel/soccer/models/utils.py), ported to JavaScript with the package's defaults:
// reaction 0.7 s, max speed 12 m/s, threshold 1.5 s, sigma 0.45, method "teams", ball_method "max".
// Team score per frame = mean over the pressing team's players of the highest probability against any opponent, the same
// number as scripts/unravel_trial.py and scripts/pressing_from_csv.py (checked there against the package: r = 0.997).
//
// Who has the ball: the team of the player nearest to it (within 3 m; otherwise the last team that had it), as threat.js.
// The tracking files carry no owner, so this is the one difference from the package (which reads the data provider's owner).

const REACT = 0.7, VMAX = 12, THR = 1.5, SIG = 0.45, K = Math.PI / Math.sqrt(3) / SIG;

function tti(p1, v1, p2, v2) {          // time for p1 (speed v1) to reach where p2 will be in 1 s; 2-D, z = 0
  const um = Math.hypot(v1[0], v1[1]);
  const d2 = [p2[0] + v2[0], p2[1] + v2[1]];
  const v = [d2[0] - p1[0], d2[1] - p1[1]];
  const vm = Math.hypot(v[0], v[1]);
  const ang = Math.acos(Math.max(-1, Math.min(1, (v1[0] * v[0] + v1[1] * v[1]) / (um * vm + 1e-10))));
  const r = [p1[0] + v1[0] * REACT, p1[1] + v1[1] * REACT];
  return um * ang / Math.PI + REACT + Math.hypot(d2[0] - r[0], d2[1] - r[1]) / VMAX;
}
const pti = (t) => 1 / (1 + Math.exp(Math.max(-700, Math.min(700, -K * (THR - t)))));

// pts / vel: 22 entries of [x, y] in metres / m/s (null = nobody); ball [x, y], bvel [vx, vy]; state keeps the last owner.
// Returns { p: team score 0..1, team: 'home' | 'away' (the pressing team), owner } or null.
export function pressOf(pts, vel, ball, bvel, state) {
  let best = -1, bd = 1e9;
  pts.forEach((q, k) => { if (q) { const d = Math.hypot(q[0] - ball[0], q[1] - ball[1]); if (d < bd) { bd = d; best = k; } } });
  if (best < 0) return null;
  if (bd < 3) state.owner = best < 11 ? 'home' : 'away';
  const owner = state.owner || (best < 11 ? 'home' : 'away');
  const own = [], opp = [];
  for (let k = 0; k < 22; k++) if (pts[k]) ((k < 11) === (owner === 'home') ? own : opp).push(k);
  if (!own.length || !opp.length) return null;
  let ci = -1, cd = 25;                                   // ball carrier: nearest player of the owning team within 25 m
  own.forEach((k, i) => { const d = Math.hypot(pts[k][0] - ball[0], pts[k][1] - ball[1]); if (d < cd) { cd = d; ci = i; } });
  if (ci < 0) return null;
  const V = (k) => vel[k] || [0, 0];
  let sum = 0;
  for (const j of opp) {                                  // rows of the package's "pressing" matrix are columns here
    let mx = 0;
    own.forEach((k, i) => {
      let t = tti(pts[k], V(k), pts[j], V(j));
      if (i === ci) t = Math.min(t, tti(ball, bvel || [0, 0], pts[j], V(j)));
      mx = Math.max(mx, pti(t));
    });
    sum += mx;
  }
  return { p: sum / opp.length, team: owner === 'home' ? 'away' : 'home', owner };
}

// one series over a whole file (real data or a layer), every `step` frames: [{t, p, team}]
export function pressSeries(d, step = 5) {
  const out = [], st = {}, n = d.t.length;
  for (let i = 0; i < n; i += step) {
    const j = Math.min(i + 1, n - 1), dt = d.t[j] - d.t[i];
    const pts = [], vel = [];
    for (let k = 0; k < 22; k++) {
      const a = i * 22 + k, b = j * 22 + k;
      if (d.px[a] < 0) { pts.push(null); vel.push(null); continue; }
      pts.push([d.px[a] / 10, d.py[a] / 10]);
      vel.push(dt > 0 && dt < 1 && d.px[b] >= 0 ? [(d.px[b] - d.px[a]) / 10 / dt, (d.py[b] - d.py[a]) / 10 / dt] : [0, 0]);
    }
    const ok = dt > 0 && dt < 1;
    const r = pressOf(pts, vel, [d.bx[i] / 10, d.by[i] / 10], ok ? [(d.bx[j] - d.bx[i]) / 10 / dt, (d.by[j] - d.by[i]) / 10 / dt] : [0, 0], st);
    if (r) out.push({ t: d.t[i], p: r.p, team: r.team });
  }
  return out;
}
