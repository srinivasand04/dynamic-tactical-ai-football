"""Off-screen player filling (Plan 2) and a learned correction on top of it (Plan 3).

Broadcast video shows only part of the pitch, so off-camera players are missing; our video board and PFF data have holes.
Open method: "centroid voting" (B4) of offscreen-impute (Kim 2026, arXiv 2607.11548, MIT, github nowayfootball/offscreen-impute
@bddca37, kept in models/offscreen_impute/). Training-free and online: each visible player votes for the team centre as
(position - its running role offset); a hidden player = voted centre + its own role offset. B1 (last seen, decaying to the
team mean) and B2 (formation offset from the visible mean) are the paper's simpler baselines. This file re-implements the
paper's loop as a reusable class (OffscreenImputer) so the viewer exports and the tests share one implementation.

Plan 3 ("our model learns to fill gaps", the masking idea of UniTraj / Sports-Traj, ICLR 2025): a small network trained on
DFL with a simulated camera learns a correction to the B4 guess from the last-seen spot, time hidden, last speed, the ball
and the visible players. Leave-one-match-out over the 7 fully tracked DFL matches.

Test (DFL, all 22 players known): a virtual broadcast camera 44 m wide follows the ball with lag (the paper's protocol);
players outside it are hidden and filled in; the error against the real positions is measured, plus what the holes do to
our pressing number (unravelsports formula) and to pitch control (the paper's model).

Usage : .venv-analysis\\Scripts\\python.exe scripts\\impute_offscreen.py bench [--matches ...] [--width 44] [--minutes 90]
        .venv-analysis\\Scripts\\python.exe scripts\\impute_offscreen.py learn        (needs bench output; Plan 3)
Output: outputs/impute/bench_<match>.npz, bench_summary.csv, learn_summary.csv, impute_model.pt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "impute")
sys.path.insert(0, os.path.join(ROOT, "models", "offscreen_impute", "src"))
L, W = 105.0, 68.0
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


class OffscreenImputer:
    """Online filler for one match. Call update(t, vis) every evaluation step, then guess(team, pid, t, vis_team)."""

    def __init__(self, tau=8.0):
        self.tau = tau
        self.last_seen, self.offsets, self.offs_ema, self.vels, self.prev = {}, {}, {}, {}, {}

    def _vote(self, team, vis_t):
        v = [(p[0] - self.offs_ema[(team, k)][0], p[1] - self.offs_ema[(team, k)][1]) for k, p in vis_t.items() if (team, k) in self.offs_ema]
        return np.mean(v, axis=0) if len(v) >= 3 else None

    def update(self, t, vis):
        """vis: {0: {pid: (x, y)}, 1: {...}} players the camera sees at time t (seconds)."""
        for team in (0, 1):
            vt = vis[team]
            for k, p in vt.items():
                key = (team, k)
                if key in self.prev and 0 < t - self.prev[key][1] <= 0.5:
                    q, tq = self.prev[key]
                    self.vels[key] = ((p[0] - q[0]) / (t - tq), (p[1] - q[1]) / (t - tq))
                self.prev[key] = (p, t)
                self.last_seen[key] = (p, t)
            if len(vt) >= 3:
                cen = np.mean(list(vt.values()), axis=0)
                for k, p in vt.items():
                    self.offsets[(team, k)] = (p[0] - cen[0], p[1] - cen[1])
                cv = self._vote(team, vt)
                ref = cen if cv is None else cv
                for k, p in vt.items():
                    off = (p[0] - ref[0], p[1] - ref[1])
                    o = self.offs_ema.get((team, k), off)
                    self.offs_ema[(team, k)] = (0.9 * o[0] + 0.1 * off[0], 0.9 * o[1] + 0.1 * off[1])

    def guess(self, team, k, t, vis_t, mode="b4"):
        """Estimated (x, y) of hidden player k of team at time t, or None if never seen. mode: b1 | b2 | b4."""
        key = (team, k)
        cen = np.mean(list(vis_t.values()), axis=0) if vis_t else np.array([L / 2, W / 2])
        b1 = None
        if key in self.last_seen:
            (lx, ly), lt = self.last_seen[key]
            w = np.exp(-(t - lt) / self.tau)
            b1 = (w * lx + (1 - w) * cen[0], w * ly + (1 - w) * cen[1])
        if mode == "b1":
            return b1
        b2 = (cen[0] + self.offsets[key][0], cen[1] + self.offsets[key][1]) if key in self.offsets else b1
        if mode == "b2":
            return b2
        cv = self._vote(team, vis_t)
        if cv is not None and key in self.offs_ema:
            return (cv[0] + self.offs_ema[key][0], cv[1] + self.offs_ema[key][1])
        return b2


def camera_x(ball_x, dt, alpha25=0.06):
    """Virtual broadcast camera: lagging average of the ball's x (the paper's 0.06 per 1/25 s, rescaled to our frame rate)."""
    a = 1 - (1 - alpha25) ** (dt * 25)
    cx, c = np.empty(len(ball_x)), L / 2
    for i, b in enumerate(ball_x):
        if np.isfinite(b):
            c += a * (b - c)
        cx[i] = c
    return cx


def press_value(att, att_v, dfd, dfd_v, ball):
    """Pressing intensity of the defending team (unravelsports formula, as viewer/press.js; ball merged into the carrier)."""
    REACT, VMAX, THR, SIG = 0.7, 12.0, 1.5, 0.45
    K = np.pi / np.sqrt(3) / SIG
    if len(att) == 0 or len(dfd) == 0:
        return np.nan

    def tti(p1, v1, p2, v2):
        um = np.linalg.norm(v1, axis=-1)
        d2 = p2 + v2
        v = d2 - p1
        vm = np.linalg.norm(v, axis=-1)
        ang = np.arccos(np.clip((v1 * v).sum(-1) / (um * vm + 1e-10), -1, 1))
        r = p1 + v1 * REACT
        return um * ang / np.pi + REACT + np.linalg.norm(d2 - r, axis=-1) / VMAX

    t = tti(att[:, None], att_v[:, None], dfd[None], dfd_v[None])            # (A, D): attacker reaches defender
    p = 1 / (1 + np.exp(np.clip(-K * (THR - t), -700, 700)))
    ci = np.argmin(np.linalg.norm(att - ball, axis=1))
    p[ci] = np.maximum(p[ci], 1 / (1 + np.exp(np.clip(-K * (THR - tti(ball[None], np.zeros((1, 2)), dfd, dfd_v)), -700, 700))))
    return float(p.max(0).mean())


def bench_match(m, width=44.0, fps=5, minutes=None):
    """Simulated camera on one DFL match. Returns per-hidden-player rows (features for Plan 3) and per-frame metrics."""
    from pitch_control import pitch_control
    d = pd.read_csv(os.path.join(ROOT, "data", "real", "processed", f"{m}.csv.gz"),
                    usecols=["frame", "time_s", "team", "dfl_id", "x_m", "y_m", "vx_ms", "vy_ms", "ball_x_m", "ball_y_m",
                             "ball_owner_team", "game_mode", "period"])
    d = d[d["game_mode"] == "normal"].dropna(subset=["x_m", "y_m"])
    if minutes:
        d = d[d["time_s"] < d["time_s"].min() + minutes * 60]
    fr = d.drop_duplicates("frame").set_index("frame")
    frames = fr.index.to_numpy()
    dt = float(np.median(np.diff(fr["time_s"].to_numpy())))
    step = max(1, int(round(1 / (fps * dt))))
    cx = pd.Series(camera_x(fr["ball_x_m"].to_numpy(), dt), index=frames)
    gx, gy = np.meshgrid(np.arange(1, L, 3.0), np.arange(1, W, 3.0))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    G = {f: g for f, g in d.groupby("frame")}
    imp = OffscreenImputer()
    rows, frows, prev_period = [], [], None
    for f in frames[::step]:
        g = G[f]
        t = float(fr.at[f, "time_s"])
        per = fr.at[f, "period"]
        if per != prev_period:                                            # new half: teams change ends, start fresh
            imp, prev_period = OffscreenImputer(), per
        vis, hid, vel = {0: {}, 1: {}}, {0: {}, 1: {}}, {}
        for r in g.itertuples(index=False):
            team = 0 if r.team == "home" else 1
            p = (r.x_m, r.y_m)
            vel[r.dfl_id] = (0.0 if np.isnan(r.vx_ms) else r.vx_ms, 0.0 if np.isnan(r.vy_ms) else r.vy_ms)
            (vis if abs(r.x_m - cx[f]) <= width / 2 else hid)[team][r.dfl_id] = p
        last = {k: v for k, v in imp.last_seen.items()}                   # state before this frame (for the features)
        imp.update(t, vis)
        ball = np.array([fr.at[f, "ball_x_m"], fr.at[f, "ball_y_m"]])
        est = {}
        for team in (0, 1):
            cen = np.mean(list(vis[team].values()), axis=0) if vis[team] else np.array([L / 2, W / 2])
            for k, gt in hid[team].items():
                e = {mo: imp.guess(team, k, t, vis[team], mo) for mo in ("b1", "b2", "b4")}
                if e["b4"] is None:
                    continue
                est[(team, k)] = e
                ls = last.get((team, k))
                gap = t - ls[1] if ls else 99.0
                lv = imp.vels.get((team, k), (0.0, 0.0))
                side = 1.0 if (team == 0) == (cx[f] < L / 2) else -1.0     # crude: which way the team faces doesn't matter for the net
                rows.append([f, team, gt[0], gt[1], *e["b4"], *(e["b1"]), *(e["b2"]),
                             *(ls[0] if ls else e["b4"]), min(gap, 30.0), *lv, *cen, *ball, cx[f], len(vis[team]), side])
        # what the holes do to pressing and pitch control (ground truth vs visible only vs B4-filled)
        own = fr.at[f, "ball_owner_team"]
        ft = {}
        for name in ("full", "visible", "b4"):
            P = {0: [], 1: []}
            for team in (0, 1):
                for k, p in vis[team].items():
                    P[team].append((p, vel[k]))
                for k, p in hid[team].items():
                    if name == "full":
                        P[team].append((p, vel[k]))
                    elif name == "b4" and (team, k) in est:
                        P[team].append((est[(team, k)]["b4"], (0.0, 0.0)))
            if own in ("home", "away"):
                a, b_ = (0, 1) if own == "home" else (1, 0)
                A = np.array([p for p, _ in P[a]]).reshape(-1, 2); Av = np.array([v for _, v in P[a]]).reshape(-1, 2)
                D = np.array([p for p, _ in P[b_]]).reshape(-1, 2); Dv = np.array([v for _, v in P[b_]]).reshape(-1, 2)
                ft[f"press_{name}"] = press_value(A, Av, D, Dv, ball)
            pc = pitch_control([(p[0], p[1], 0.0, 0.0, team) for team in (0, 1) for p, _ in P[team]], grid)
            ft[f"pc_{name}"] = pc
        hz = np.abs(grid[:, 0] - cx[f]) > width / 2
        fm = {"frame": f, "n_visible": len(vis[0]) + len(vis[1])}
        for k in ("visible", "b4"):
            if ft.get("press_full") is not None and f"press_{k}" in ft:
                fm[f"press_err_{k}"] = abs(ft[f"press_{k}"] - ft["press_full"])
            if ft["pc_full"] is not None and ft[f"pc_{k}"] is not None:
                fm[f"share_err_{k}"] = abs(ft[f"pc_{k}"].mean() - ft["pc_full"].mean()) * 100
                fm[f"hidden_pc_mae_{k}"] = np.abs(ft[f"pc_{k}"][hz] - ft["pc_full"][hz]).mean() * 100 if hz.any() else np.nan
        frows.append(fm)
    cols = ["frame", "team", "gx", "gy", "b4x", "b4y", "b1x", "b1y", "b2x", "b2y", "lsx", "lsy", "gap", "lvx", "lvy",
            "cenx", "ceny", "ballx", "bally", "camx", "nvis", "side"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(frows)


def bench(a):
    os.makedirs(OUT, exist_ok=True)
    summ = []
    for m in a.matches:
        P, F = bench_match(m, a.width, a.fps, a.minutes)
        P.to_csv(os.path.join(OUT, f"bench_{m}_players.csv.gz"), index=False)
        F.to_csv(os.path.join(OUT, f"bench_{m}_frames.csv"), index=False)
        r = {"match": m, "frames": len(F), "visible_mean": F["n_visible"].mean(), "hidden_rows": len(P)}
        for mo in ("b1", "b2", "b4"):
            e = np.hypot(P[f"{mo}x"] - P["gx"], P[f"{mo}y"] - P["gy"])
            r[f"{mo}_median_m"], r[f"{mo}_mean_m"] = e.median(), e.mean()
        for k in ("press_err", "share_err", "hidden_pc_mae"):
            for s in ("visible", "b4"):
                c = f"{k}_{s}"
                if c in F:
                    r[c] = F[c].mean()
        summ.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    S = pd.DataFrame(summ)
    p = os.path.join(OUT, f"bench_summary_w{int(a.width)}.csv")
    S.round(4).to_csv(p, index=False)
    print(S.drop(columns=["match"]).mean(numeric_only=True).round(3).to_string())
    print("wrote", os.path.relpath(p, ROOT))


# ---------------- Plan 3: learned correction to the B4 guess ----------------
FEAT = ["b4x", "b4y", "b1x", "b1y", "b2x", "b2y", "lsx", "lsy", "gap", "lvx", "lvy", "cenx", "ceny", "ballx", "bally", "camx", "nvis"]


def features(P):
    X = P[FEAT].to_numpy(np.float32).copy()
    X[:, [2, 4, 6, 11, 13]] -= X[:, [0]]; X[:, [3, 5, 7, 12, 14]] -= X[:, [1]]     # everything relative to the B4 guess
    X[:, 15] -= X[:, 0]
    X[:, 8] = np.log1p(X[:, 8])
    return np.c_[X, X[:, 0] / L, X[:, 1] / W]


def learn(a):
    import torch
    torch.manual_seed(0)
    data = {m: pd.read_csv(os.path.join(OUT, f"bench_{m}_players.csv.gz")) for m in a.matches}
    rows = []
    for test in a.matches:
        tr = pd.concat([data[m] for m in a.matches if m != test])
        X, Y = features(tr), (tr[["gx", "gy"]].to_numpy(np.float32) - tr[["b4x", "b4y"]].to_numpy(np.float32))
        mu, sd = X.mean(0), X.std(0) + 1e-6
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
        opt = torch.optim.Adam(net.parameters(), 2e-3)
        Xt, Yt = torch.tensor((X - mu) / sd), torch.tensor(Y)
        for ep in range(a.epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 1024):
                b = perm[i:i + 1024]
                loss = torch.nn.functional.smooth_l1_loss(net(Xt[b]), Yt[b], beta=2.0)
                opt.zero_grad(); loss.backward(); opt.step()
        te = data[test]
        with torch.no_grad():
            corr = net(torch.tensor((features(te) - mu) / sd)).numpy()
        g = te[["gx", "gy"]].to_numpy()
        e_b4 = np.hypot(*(te[["b4x", "b4y"]].to_numpy() - g).T)
        e_nn = np.hypot(*(te[["b4x", "b4y"]].to_numpy() + corr - g).T)
        short = te["gap"].to_numpy() <= 9.6
        r = {"held_out": test, "rows": len(te), "b4_median_m": np.median(e_b4), "learned_median_m": np.median(e_nn),
             "b4_mean_m": e_b4.mean(), "learned_mean_m": e_nn.mean(),
             "b4_median_short_m": np.median(e_b4[short]), "learned_median_short_m": np.median(e_nn[short]),
             "better_share": float((e_nn < e_b4).mean())}
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if test == a.matches[-1]:
            torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "feat": FEAT}, os.path.join(OUT, "impute_model.pt"))
    S = pd.DataFrame(rows)
    S.round(4).to_csv(os.path.join(OUT, "learn_summary.csv"), index=False)
    print(S.drop(columns=["held_out"]).mean(numeric_only=True).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "learn"])
    ap.add_argument("--matches", nargs="+", default=MATCHES)
    ap.add_argument("--width", type=float, default=44.0, help="camera width in metres (the paper's main setting)")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--minutes", type=float, default=None, help="only the first N minutes (quick test)")
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    bench(a) if a.cmd == "bench" else learn(a)


if __name__ == "__main__":
    main()
