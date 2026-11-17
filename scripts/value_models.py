"""Phase 2: value models. xG (how likely a shot is a goal) and xT (how much a ball position is worth), learned from real data.

xG
    Data: every shot of the 64 World Cup 2022 matches (PFF), with the freeze frame of all 22 players at the shot.
    Features (all computable from positions, so the same model scores shots in GRF and in the DFL tracking):
        dist      distance to the goal centre (m), also as log
        angle     angle the goal mouth opens from the shot spot (radians)
        header    shot with the head
        free_kick direct free kick
        n_block   outfield defenders inside the triangle shot spot - post - post
        near_def  nearest outfield defender to the shooter (m, capped at 10)
        gk_goal   keeper's distance from the goal centre (m)
        gk_line   keeper's distance from the straight line shot spot - goal centre (m)
    Penalties are not modelled: they get the training penalty conversion rate.
    Models compared, leave-matches-out (8 groups of 8 matches): the old distance-only proxy (team_agent.xg_proxy),
    distance + angle, + header and free kick (what event data alone gives), all features (ours), 5 small nets averaged.
    Outside check: the shots of the 7 DFL Bundesliga matches (other league, other data provider), positions from tracking.

xT (expected threat, Karun Singh 2018)
    The pitch is cut into 16 x 12 cells. For each cell: how often the team shoots from it, how often it moves the ball
    (pass, cross, carry) and where the ball goes when the move succeeds. xT(cell) = P(shoot) * xG(cell) +
    P(move) * sum over target cells of P(target) * xT(target), solved by iteration. xG(cell) comes from the distance +
    angle model, so cells with few shots still get a smooth value.
    Check, leave-matches-out: does xT of where the ball is predict a shot / goal by that team in the next 10 / 15 s
    better than distance to goal alone?

Outputs (outputs/value_models/):
    pff_shots.csv, pff_moves.csv.gz        extracted tables (cache; delete to re-extract)
    xg_cv.csv, xg_folds.csv, xg_dfl.csv    scores per model
    xg_model.json                          the chosen xG model (all features), fitted on all shots
    xt_grid.json, xt_check.csv             the xT grid and its check
    value_models_chart.png

Use from other scripts:
    import value_models as V
    V.xg_shot(shooter_xy, defenders_xy, keeper_index=None)   # metres, attacking towards x = 105
    V.xt(ball_xy)                                            # metres, attacking towards x = 105

Usage:  .venv-analysis\\Scripts\\python.exe scripts\\value_models.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

L, W = 105.0, 68.0
GOAL = np.array([L, W / 2])
POSTS = np.array([[L, W / 2 - 3.66], [L, W / 2 + 3.66]])
NX, NY = 16, 12
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFF = os.path.join(ROOT, "data", "external", "pff_worldcup", "raw", "FIFA World Cup 2022")
OUT = os.path.join(ROOT, "outputs", "value_models")
FEATS = ["dist", "log_dist", "angle", "header", "free_kick", "n_block", "near_def", "gk_goal", "gk_line"]
SETS = {"distance + angle": ["dist", "log_dist", "angle"],
        "+ header, free kick (event data only)": ["dist", "log_dist", "angle", "header", "free_kick"],
        "all features (ours)": FEATS}
N_GROUPS = 8


# ---------------------------------------------------------------- geometry (shared by PFF, DFL and GRF)

def in_triangle(p, a, b, c):
    def s(p1, p2, p3):
        return (p1[..., 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[..., 1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def shot_features(shooter, defenders, keeper=None, header=False, free_kick=False):
    """shooter (2,), defenders (n, 2) in metres, attacking towards x = L. keeper: index into defenders, or None to take
    the defender nearest the goal centre."""
    shooter = np.asarray(shooter, float)
    dfs = np.asarray(defenders, float).reshape(-1, 2)
    dfs = dfs[~np.isnan(dfs[:, 0])]
    v = GOAL - shooter
    dist = max(float(np.hypot(*v)), 0.5)
    a, b = POSTS[0] - shooter, POSTS[1] - shooter
    angle = float(abs(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)))
    if len(dfs) == 0:
        gk, out = None, dfs
    else:
        k = int(np.argmin(np.hypot(*(dfs - GOAL).T))) if keeper is None else keeper
        gk, out = dfs[k], np.delete(dfs, k, axis=0)
    n_block = int(in_triangle(out, shooter, POSTS[0], POSTS[1]).sum()) if len(out) else 0
    near = float(np.min(np.hypot(*(out - shooter).T))) if len(out) else 10.0
    if gk is None:
        gk_goal, gk_line = 5.0, 5.0
    else:
        gk_goal = float(np.hypot(*(gk - GOAL)))
        u = v / dist
        r = gk - shooter
        gk_line = float(abs(r[0] * u[1] - r[1] * u[0]))
    return {"dist": dist, "log_dist": float(np.log(dist)), "angle": angle, "header": float(header),
            "free_kick": float(free_kick), "n_block": float(n_block), "near_def": min(near, 10.0),
            "gk_goal": min(gk_goal, 20.0), "gk_line": min(gk_line, 10.0)}


STATE = ["dist", "log_dist", "angle", "bx", "by_c", "def_goal_side", "def_in_cone", "def_5m", "def_10m", "near_def",
         "att_ahead", "best_mate_goal", "free_in_box"]


def state_features(ball, attackers, defenders):
    """The value of a ball position given where everybody is (attacking towards x = L). Used by the threat models."""
    b = np.asarray(ball, float)
    att = np.asarray(attackers, float).reshape(-1, 2)
    att = att[~np.isnan(att[:, 0])]
    dfs = np.asarray(defenders, float).reshape(-1, 2)
    dfs = dfs[~np.isnan(dfs[:, 0])]
    dist = max(float(np.hypot(*(GOAL - b))), 0.5)
    a, c = POSTS[0] - b, POSTS[1] - b
    angle = float(abs(np.arctan2(a[0] * c[1] - a[1] * c[0], a @ c)))
    dd = np.hypot(*(dfs - b).T) if len(dfs) else np.array([15.0])
    in_box = (att[:, 0] > L - 16.5) & (abs(att[:, 1] - W / 2) < 20.16) if len(att) else np.zeros(0, bool)
    if len(att) and len(dfs):
        free = np.min(np.hypot(*(att[:, None] - dfs[None]).transpose(2, 0, 1)), axis=1) > 3.0
    else:
        free = np.ones(len(att), bool)
    return {"dist": dist, "log_dist": float(np.log(dist)), "angle": angle, "bx": float(b[0]), "by_c": float(abs(b[1] - W / 2)),
            "def_goal_side": float((dfs[:, 0] > b[0]).sum()) if len(dfs) else 0.0,
            "def_in_cone": float(in_triangle(dfs, b, POSTS[0], POSTS[1]).sum()) if len(dfs) else 0.0,
            "def_5m": float((dd < 5).sum()), "def_10m": float((dd < 10).sum()), "near_def": float(min(dd.min(), 15.0)),
            "att_ahead": float((att[:, 0] > b[0] + 1).sum()) if len(att) else 0.0,
            "best_mate_goal": float(min(np.hypot(*(att - GOAL).T).min(), 60.0)) if len(att) else 60.0,
            "free_in_box": float((in_box & free).sum())}


def proxy(dist):
    """The distance-only proxy GRF runs used so far (team_agent.xg_proxy)."""
    return 0.8 * np.exp(-0.15 * np.asarray(dist))


# ---------------------------------------------------------------- PFF extraction

def attack_sign(events, meta):
    """(home?, period) -> +1 if that team attacks +x in PFF coordinates. Taken from where its shots are; the
    metadata (home starts left, ends swap each period) fills periods without shots."""
    start = 1.0 if meta.get("homeTeamStartLeft", True) else -1.0
    votes = {}
    for x in events:
        pe, ge = x.get("possessionEvents") or {}, x.get("gameEvents") or {}
        if pe.get("possessionEventType") == "SH" and x.get("ball") and ge.get("period"):
            votes.setdefault((bool(ge["homeTeam"]), ge["period"]), []).append(np.sign(x["ball"][0]["x"]))
    out = {}
    for home in (True, False):
        for period in range(1, 6):
            v = votes.get((home, period))
            if v and abs(np.mean(v)) > 0.3:
                out[(home, period)] = float(np.sign(np.mean(v)))
            else:
                v2 = votes.get((not home, period))
                if v2 and abs(np.mean(v2)) > 0.3:
                    out[(home, period)] = -float(np.sign(np.mean(v2)))
                else:
                    out[(home, period)] = (start if home else -start) * (1 if period % 2 == 1 else -1)
    return out


def to_pitch(x, y, s):
    """PFF centre-origin metres -> 0..L, 0..W with the team attacking +x."""
    return np.array([L / 2 + s * x, W / 2 + s * y])


def extract_pff():
    shots, moves = [], []
    files = sorted(glob.glob(os.path.join(PFF, "Event Data", "*.json")))
    for n, f in enumerate(files):
        gid = os.path.basename(f)[:-5]
        ev = json.load(open(f, encoding="utf-8"))
        meta = json.load(open(os.path.join(PFF, "Metadata", gid + ".json"), encoding="utf-8"))[0]
        sign = attack_sign(ev, meta)
        rows = []
        for x in ev:
            pe, ge = x.get("possessionEvents") or {}, x.get("gameEvents") or {}
            if not pe.get("possessionEventType") or not x.get("ball") or ge.get("period") is None:
                continue
            b = x["ball"][0]
            if b.get("x") is None:
                continue
            rows.append((x, pe, ge, b))
        for i, (x, pe, ge, b) in enumerate(rows):
            home, period, kind = bool(ge["homeTeam"]), ge["period"], pe["possessionEventType"]
            s = sign[(home, period)]
            p = to_pitch(b["x"], b["y"], s)
            t = float(x["eventTime"])
            if kind == "SH":
                dfs = x["awayPlayers"] if home else x["homePlayers"]
                dxy = np.array([to_pitch(d["x"], d["y"], s) for d in dfs]) if dfs else np.zeros((0, 2))
                gk = next((k for k, d in enumerate(dfs or []) if d.get("positionGroupType") == "GK"), None)
                feat = shot_features(p, dxy, gk, header=pe.get("bodyType") == "HE", free_kick=ge.get("setpieceType") == "F")
                shots.append({"match": gid, "time": t, "period": period, "home": home, "x": p[0], "y": p[1],
                              "penalty": ge.get("setpieceType") == "P", "goal": int(pe.get("shotOutcomeType") == "G"),
                              "pressure": float(pe.get("pressureType") not in (None, "N")), **feat})
                moves.append({"match": gid, "time": t, "home": home, "kind": "shot", "x": p[0], "y": p[1],
                              "ex": np.nan, "ey": np.nan, "ok": 0, "goal": int(pe.get("shotOutcomeType") == "G"),
                              "set_piece": ge.get("setpieceType") not in (None, "O")})
            elif kind in ("PA", "CR", "BC"):
                nxt = rows[i + 1] if i + 1 < len(rows) else None
                same = nxt is not None and bool(nxt[2]["homeTeam"]) == home and nxt[2]["period"] == period
                if kind == "PA":
                    ok = pe.get("passOutcomeType") == "C" and same
                elif kind == "CR":
                    ok = pe.get("crossOutcomeType") == "C" and same
                else:
                    ok = same
                e = to_pitch(nxt[3]["x"], nxt[3]["y"], s) if same else np.array([np.nan, np.nan])
                own_l, opp_l = (x["homePlayers"], x["awayPlayers"]) if home else (x["awayPlayers"], x["homePlayers"])
                axy = np.array([to_pitch(q["x"], q["y"], s) for q in own_l or []]).reshape(-1, 2)
                dxy = np.array([to_pitch(q["x"], q["y"], s) for q in opp_l or []]).reshape(-1, 2)
                st = state_features(p, axy, dxy) if len(axy) >= 8 and len(dxy) >= 8 else {k: np.nan for k in STATE}
                moves.append({"match": gid, "time": t, "home": home, "kind": kind, "x": p[0], "y": p[1],
                              "ex": e[0], "ey": e[1], "ok": int(ok), "goal": 0,
                              "set_piece": ge.get("setpieceType") not in (None, "O"), **{"s_" + k: v for k, v in st.items()}})
        print(f"  {n + 1:2d}/{len(files)}  {gid}  shots so far {len(shots)}", flush=True)
    return pd.DataFrame(shots), pd.DataFrame(moves)


# ---------------------------------------------------------------- DFL shots (outside check)

def extract_dfl():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import settings
    from extract_passes import load_tracking, FPS, WINDOW_S, MAX_SYNC_M, BALL
    out = []
    for m in settings.get("real_matches", default=None) or ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]:
        e = pd.read_csv(os.path.join(ROOT, "data", "real", "raw", f"{m}_events.csv.gz"))
        sh = e[e["event_type"] == "SHOT"].copy()
        d0 = pd.read_csv(settings.csv(m), usecols=["frame", "time_s", "period"]).drop_duplicates("frame")
        p2 = float(d0.loc[d0["period"] == 2, "time_s"].min())
        sh["time_s"] = pd.to_timedelta(sh["timestamp"]).dt.total_seconds() + np.where(sh["period_id"] == 2, p2, 0.0)
        last = int(d0["frame"].max())
        centre = np.clip(np.round(sh["time_s"].to_numpy() * FPS).astype(int), 0, last)
        k = int(WINDOW_S * FPS)
        frames = np.unique(np.concatenate([np.clip(centre + j, 0, last) for j in range(-k, k + 1)]))
        xy, _, ids, _ = load_tracking(m, frames)
        row = {f: i for i, f in enumerate(frames)}
        for c, ev in zip(centre, sh.itertuples()):
            best = None
            for f in range(max(c - k, 0), min(c + k, last) + 1):
                i = row[f]
                where = np.flatnonzero(ids[i] == ev.player_id)
                if len(where) == 0 or np.isnan(xy[i, BALL, 0]):
                    continue
                dd = float(np.linalg.norm(xy[i, where[0]] - xy[i, BALL]))
                if best is None or dd < best[0]:
                    best = (dd, i, int(where[0]))
            if best is None or best[0] > MAX_SYNC_M:
                continue
            _, i, who = best
            pos = xy[i].copy()
            shooter = pos[who]
            if shooter[0] < L / 2:                      # attacking the goal at x = 0: turn the pitch
                pos[:, 0], pos[:, 1] = L - pos[:, 0], W - pos[:, 1]
                shooter = pos[who]
            dfs = pos[11:22] if who < 11 else pos[:11]
            feat = shot_features(shooter, dfs, None, header=ev.body_part_type == "HEAD",
                                 free_kick=ev.set_piece_type == "FREE_KICK")
            out.append({"match": m, "x": shooter[0], "y": shooter[1], "penalty": ev.set_piece_type == "PENALTY",
                        "goal": int(ev.result == "GOAL"), **feat})
        print(f"  DFL {m}: {len(sh)} shots, {sum(r['match'] == m for r in out)} matched to tracking", flush=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- models

def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def scores(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"auc": auc(y, p), "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "predicted": float(p.sum()), "real": int(np.sum(y))}


def add_derived(d):
    d = d.copy()
    d["dist_x_angle"] = d["dist"] * d["angle"]
    d["header_x_dist"] = d["header"] * d["dist"]
    d["block_x_dist"] = d["n_block"] * d["dist"]
    d["gk_line_x_dist"] = d["gk_line"] / d["dist"]
    return d


INTER = FEATS + ["dist_x_angle", "header_x_dist", "block_x_dist", "gk_line_x_dist"]


class Logit:
    def __init__(self, cols, l2=1.0, y="goal"):
        self.cols, self.l2, self.y = cols, l2, y

    def fit(self, d):
        X = d[self.cols].to_numpy(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        X = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        y = d[self.y].to_numpy(float)
        w = np.zeros(X.shape[1])
        reg = np.eye(len(w)) * self.l2
        reg[0, 0] = 0
        for _ in range(50):                              # Newton steps (IRLS) with a small ridge
            p = 1 / (1 + np.exp(-X @ w))
            g = X.T @ (p - y) + reg @ w
            H = (X * (p * (1 - p))[:, None]).T @ X + reg
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def predict(self, d):
        X = d[self.cols].to_numpy(float) if isinstance(d, pd.DataFrame) else np.atleast_2d(d)
        X = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        return 1 / (1 + np.exp(-X @ self.w))

    def to_json(self):
        return {"kind": "logistic", "features": self.cols, "mean": self.mu.tolist(), "std": self.sd.tolist(),
                "weights": self.w.tolist()}


class NetEnsemble:
    def __init__(self, cols, n=5, hidden=(16,), epochs=300, y="goal", wd=1e-3):
        self.cols, self.n, self.hidden, self.epochs, self.y, self.wd = cols, n, hidden, epochs, y, wd

    def fit(self, d):
        import torch
        X = d[self.cols].to_numpy(np.float32)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        X = torch.tensor((X - self.mu) / self.sd)
        y = torch.tensor(d[self.y].to_numpy(np.float32))
        self.nets = []
        for s in range(self.n):
            torch.manual_seed(s)
            layers, k = [], X.shape[1]
            for h in self.hidden:
                layers += [torch.nn.Linear(k, h), torch.nn.ReLU()]
                k = h
            net = torch.nn.Sequential(*layers, torch.nn.Linear(k, 1))
            opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=self.wd)
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X).squeeze(1), y)
                loss.backward()
                opt.step()
            self.nets.append(net)
        return self

    def predict(self, d):
        import torch
        X = torch.tensor((d[self.cols].to_numpy(np.float32) - self.mu) / self.sd)
        with torch.no_grad():
            return np.mean([torch.sigmoid(n(X).squeeze(1)).numpy() for n in self.nets], axis=0)


class ZoneTable:
    """Goal share per zone (distance band x angle band x header), pulled towards the overall share when a zone has few shots."""
    D_BINS = [0, 6, 9, 12, 16, 20, 25, 30, 200]

    def __init__(self, prior=10.0):
        self.prior = prior

    def _key(self, d):
        db = np.digitize(d["dist"], self.D_BINS)
        ab = np.digitize(d["angle"], self.a_bins)
        return db * 100 + ab * 10 + d["header"].to_numpy().astype(int)

    def fit(self, d):
        self.a_bins = np.quantile(d["angle"], [1 / 3, 2 / 3])
        k = self._key(d)
        self.base = float(d["goal"].mean())
        g = pd.DataFrame({"k": k, "y": d["goal"].to_numpy()}).groupby("k")["y"].agg(["sum", "count"])
        self.rate = ((g["sum"] + self.prior * self.base) / (g["count"] + self.prior)).to_dict()
        return self

    def predict(self, d):
        return np.array([self.rate.get(k, self.base) for k in self._key(d)])


class KNN:
    """Goal share of the k most similar shots in the training set (all features, scaled)."""
    def __init__(self, cols, k=75):
        self.cols, self.k = cols, k

    def fit(self, d):
        X = d[self.cols].to_numpy(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.X, self.y = (X - self.mu) / self.sd, d["goal"].to_numpy(float)
        return self

    def predict(self, d):
        Z = (d[self.cols].to_numpy(float) - self.mu) / self.sd
        out = np.empty(len(Z))
        for i in range(0, len(Z), 256):
            dd = ((Z[i:i + 256, None] - self.X[None]) ** 2).sum(-1)
            nn = np.argpartition(dd, self.k, axis=1)[:, :self.k]
            out[i:i + 256] = self.y[nn].mean(1)
        return np.clip(out, 0.005, 0.995)


def groups(matches):
    u = sorted(matches.unique())
    rng = np.random.default_rng(0)
    rng.shuffle(u)
    return {m: i % N_GROUPS for i, m in enumerate(u)}


# (name, maker, uses the PFF-only pressure label)
XG_MODELS = [
    ("zone table (distance x angle)", lambda: ZoneTable(), False),
    ("distance + angle", lambda: Logit(SETS["distance + angle"]), False),
    ("+ header, free kick (event data only)", lambda: Logit(SETS["+ header, free kick (event data only)"]), False),
    ("all features (ours)", lambda: Logit(FEATS), False),
    ("ours + interactions", lambda: Logit(INTER), False),
    ("ours + pressure label (PFF only)", lambda: Logit(FEATS + ["pressure"]), True),
    ("k nearest shots (k=75)", lambda: KNN(FEATS), False),
    ("1 small net", lambda: NetEnsemble(FEATS, n=1), False),
    ("5 small nets averaged", lambda: NetEnsemble(FEATS), False),
    ("5 deeper nets (32-32) averaged", lambda: NetEnsemble(FEATS, hidden=(32, 32), wd=3e-3), False),
]
BLEND = ("blend: ours + 5 nets", ["all features (ours)", "5 small nets averaged"])


def xg_cv(shots, dfl):
    open_ = add_derived(shots[~shots["penalty"]].reset_index(drop=True))
    dfl_open = add_derived(dfl[~dfl["penalty"]].reset_index(drop=True))
    g = open_["match"].map(groups(open_["match"])).to_numpy()
    preds = {"distance proxy (old)": proxy(open_["dist"].to_numpy())}
    dpreds = {"distance proxy (old)": proxy(dfl_open["dist"].to_numpy())}
    fitted = {}
    for name, mk, pff_only in XG_MODELS:
        p = np.zeros(len(open_))
        for k in range(N_GROUPS):
            p[g == k] = mk().fit(open_[g != k]).predict(open_[g == k])
        preds[name] = p
        fitted[name] = mk().fit(open_)
        dpreds[name] = np.full(len(dfl_open), np.nan) if pff_only else fitted[name].predict(dfl_open)
        print(f"  xG: {name} done", flush=True)
    preds[BLEND[0]] = np.mean([preds[n] for n in BLEND[1]], axis=0)
    dpreds[BLEND[0]] = np.mean([dpreds[n] for n in BLEND[1]], axis=0)
    y, dy = open_["goal"].to_numpy(), dfl_open["goal"].to_numpy()
    rows, folds = [], []
    for name, p in preds.items():
        r = {"model": name, "shots": len(y), **scores(y, p)}
        dp = dpreds[name]
        if not np.isnan(dp).all():
            r.update({f"bundesliga_{k}": v for k, v in scores(dy, dp).items()})
        rows.append(r)
        for k in range(N_GROUPS):
            folds.append({"model": name, "group": k, **scores(y[g == k], p[g == k])})
    cv, folds = pd.DataFrame(rows), pd.DataFrame(folds)
    ref = folds[folds["model"] == "all features (ours)"].set_index("group")["log_loss"]
    for name in preds:
        f = folds[folds["model"] == name].set_index("group")["log_loss"]
        d = f - ref
        cv.loc[cv["model"] == name, "logloss_vs_ours"] = d.mean()
        cv.loc[cv["model"] == name, "logloss_vs_ours_ci95"] = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        cv.loc[cv["model"] == name, "beats_ours_in_groups"] = f"{int((d < 0).sum())}/{N_GROUPS}"
    pen_rate = float(shots.loc[shots["penalty"], "goal"].mean())
    return cv, folds, preds, open_, fitted, pen_rate


# ---------------------------------------------------------------- xT and threat models

def cell(x, y, nx=NX, ny=NY):
    cx = np.clip((np.asarray(x) / L * nx).astype(int), 0, nx - 1)
    cy = np.clip((np.asarray(y) / W * ny).astype(int), 0, ny - 1)
    return cx * ny + cy


def fit_xt(moves, xg_geom, nx=NX, ny=NY):
    m = moves[~moves["set_piece"]]
    n = nx * ny
    s_from = cell(m["x"], m["y"], nx, ny)
    is_shot = (m["kind"] == "shot").to_numpy()
    shots = np.bincount(s_from[is_shot], minlength=n).astype(float)
    mv = np.bincount(s_from[~is_shot], minlength=n).astype(float)
    ok = (~is_shot) & (m["ok"] == 1).to_numpy() & ~np.isnan(m["ex"].to_numpy())
    T = np.zeros((n, n))
    np.add.at(T, (s_from[ok], cell(m["ex"].to_numpy()[ok], m["ey"].to_numpy()[ok], nx, ny)), 1.0)
    tot = shots + mv
    p_shoot = np.divide(shots, tot, out=np.zeros(n), where=tot > 0)
    p_move = np.divide(mv, tot, out=np.zeros(n), where=tot > 0)
    T = np.divide(T, mv[:, None], out=np.zeros_like(T), where=mv[:, None] > 0)
    cx, cy = np.meshgrid((np.arange(nx) + 0.5) * L / nx, (np.arange(ny) + 0.5) * W / ny, indexing="ij")
    g = np.array([xg_geom(c) for c in np.c_[cx.ravel(), cy.ravel()]])
    xt = np.zeros(n)
    for _ in range(200):
        new = p_shoot * g + p_move * (T @ xt)
        if np.abs(new - xt).max() < 1e-8:
            xt = new
            break
        xt = new
    return xt.reshape(nx, ny)


SCOLS = ["s_" + k for k in STATE]


def threat_labels(moves):
    """Open-play passes / crosses / carries with a full freeze frame, labelled: does this team shoot within 10 s, score within 15 s?"""
    ev = moves[(moves["kind"] != "shot") & ~moves["set_piece"]].dropna(subset=SCOLS).reset_index(drop=True)
    sh = moves[moves["kind"] == "shot"]
    by = {k: v for k, v in sh.groupby(["match", "home"])}
    ev["shot10"], ev["goal15"] = 0, 0
    for (mt, hm), idx in ev.groupby(["match", "home"]).groups.items():
        s = by.get((mt, hm))
        if s is None:
            continue
        dt = s["time"].to_numpy()[None, :] - ev.loc[idx, "time"].to_numpy()[:, None]
        ev.loc[idx, "shot10"] = ((dt > 0) & (dt <= 10)).any(1).astype(int)
        ev.loc[idx, "goal15"] = ((dt > 0) & (dt <= 15) & (s["goal"].to_numpy()[None, :] == 1)).any(1).astype(int)
    return ev


def threat_cv(moves, shots_all, make_geom):
    ev = threat_labels(moves)
    gmap = groups(moves["match"])
    gev = ev["match"].map(gmap).to_numpy()
    gmv = moves["match"].map(gmap).to_numpy()
    names = ["distance to goal", "xT grid 8x6", "xT grid 16x12 (Phase 2)", "xT grid 24x16",
             "state logistic (sees defenders)", "state, 5 nets averaged"]
    preds = {t: {n: np.zeros(len(ev)) for n in names} for t in ("shot10", "goal15")}
    dist = -ev["s_dist"].to_numpy()
    for t in preds:
        preds[t]["distance to goal"] = dist
    for k in range(N_GROUPS):
        tr_m = moves[gmv != k]
        geom = make_geom(shots_all[shots_all["match"].isin(tr_m["match"].unique()) & ~shots_all["penalty"]])
        te = gev == k
        for (nx, ny), nm in (((8, 6), "xT grid 8x6"), ((16, 12), "xT grid 16x12 (Phase 2)"), ((24, 16), "xT grid 24x16")):
            grid = fit_xt(tr_m, geom, nx, ny)
            v = grid.ravel()[cell(ev["x"].to_numpy()[te], ev["y"].to_numpy()[te], nx, ny)]
            for t in preds:
                preds[t][nm][te] = v
        for t in preds:
            tr = ev[gev != k]
            preds[t]["state logistic (sees defenders)"][te] = Logit(SCOLS, y=t).fit(tr).predict(ev[te])
            preds[t]["state, 5 nets averaged"][te] = NetEnsemble(SCOLS, y=t, epochs=200, hidden=(32,)).fit(tr).predict(ev[te])
        print(f"  threat: group {k + 1}/{N_GROUPS} done", flush=True)
    rows = []
    for t, label in (("shot10", "shot within 10 s"), ("goal15", "goal within 15 s")):
        y = ev[t].to_numpy()
        ref = [auc(y[gev == k], preds[t]["distance to goal"][gev == k]) for k in range(N_GROUPS)]
        for n in names:
            f = np.array([auc(y[gev == k], preds[t][n][gev == k]) for k in range(N_GROUPS)])
            d = f - np.array(ref)
            rows.append({"target": label, "model": n, "positions": len(y), "positives": int(y.sum()), "auc": auc(y, preds[t][n]),
                         "vs_distance": d.mean(), "vs_distance_ci95": 1.96 * d.std(ddof=1) / np.sqrt(len(d)),
                         "beats_distance_in_groups": f"{int((d > 0).sum())}/{N_GROUPS}"})
    final = {t: Logit(SCOLS, y=t).fit(ev) for t in ("shot10", "goal15")}
    return pd.DataFrame(rows), final


# ---------------------------------------------------------------- chart

def chart(cv, xt, thr, preds, open_, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(19, 12))
    a = ax[0, 0]
    names = cv["model"].tolist()
    y = np.arange(len(names))
    col = ["#b5b3ab"] + ["#8fb1e0"] * (len(names) - 1)
    col[names.index("all features (ours)")] = "#2a7ad5"
    best = names[int(cv["log_loss"].to_numpy().argmin())]
    col[names.index(best)] = "#e0483b"
    a.barh(y - 0.2, cv["auc"], 0.4, color=col, label="World Cup, held-out matches")
    bl = cv.get("bundesliga_auc", pd.Series(np.nan, index=cv.index)).to_numpy()
    a.barh(y + 0.2, np.nan_to_num(bl), 0.4, color=col, alpha=0.4, hatch="//", label="Bundesliga (never seen)")
    for i, (u, v) in enumerate(zip(cv["auc"], bl)):
        a.text(u + 0.002, i - 0.2, f"{u:.3f}", va="center", fontsize=8)
        a.text((v if v == v else 0.6) + 0.002, i + 0.2, f"{v:.3f}" if v == v else "n/a", va="center", fontsize=8, color="#555")
    a.set_yticks(y, names)
    a.invert_yaxis()
    a.set_xlim(0.6, 0.86)
    a.legend(fontsize=8, loc="lower right")
    a.set_title(f"xG model types: telling goals from misses (AUC)\nblue = ours, red = lowest log loss ({best})")
    a = ax[0, 1]
    yv = open_["goal"].to_numpy()
    for name, c in (("distance proxy (old)", "#b5b3ab"), ("zone table (distance x angle)", "#f2c14e"),
                    ("k nearest shots (k=75)", "#8fb1e0"), ("all features (ours)", "#2a7ad5"), (best, "#e0483b")):
        p = preds[name]
        bins = np.quantile(p, np.linspace(0, 1, 11))
        k = np.clip(np.searchsorted(bins, p, side="right") - 1, 0, 9)
        a.plot([p[k == i].mean() for i in range(10)], [yv[k == i].mean() for i in range(10)], "o-", color=c, label=name, ms=4)
    a.plot([0, 0.6], [0, 0.6], "k--", lw=0.8)
    a.set_xlabel("predicted xG (10 groups of shots)")
    a.set_ylabel("real goal share")
    a.legend(fontsize=8)
    a.set_title("xG calibration: on the dashed line = the numbers can be trusted as chances")
    a = ax[1, 0]
    im = a.imshow(xt.T, origin="lower", extent=[0, L, 0, W], cmap="YlOrRd", aspect="auto")
    for i in range(xt.shape[0]):
        for j in range(xt.shape[1]):
            if xt[i, j] > 0.04:
                a.text((i + 0.5) * L / xt.shape[0], (j + 0.5) * W / xt.shape[1], f"{xt[i, j]:.2f}", ha="center", va="center", fontsize=7)
    a.set_title("xT grid 16x12: value of having the ball in each zone (attacking to the right)")
    plt.colorbar(im, ax=a, fraction=0.03)
    a = ax[1, 1]
    tn = thr["model"].unique().tolist()
    cols = ["#b5b3ab", "#f2c14e", "#8fb1e0", "#5d8fd0", "#2a7ad5", "#e0483b"]
    for k, n in enumerate(tn):
        vals = [thr[(thr["model"] == n) & (thr["target"] == t)]["auc"].iloc[0] for t in ("shot within 10 s", "goal within 15 s")]
        xs = np.arange(2) + (k - (len(tn) - 1) / 2) * 0.13
        a.bar(xs, vals, 0.13, color=cols[k % len(cols)], label=n)
        for xx, v in zip(xs, vals):
            a.text(xx, v + 0.003, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    a.set_xticks(range(2), ["shot by this team in next 10 s", "goal by this team in next 15 s"])
    a.set_ylim(0.6, 0.95)
    a.legend(fontsize=8, loc="upper right")
    a.set_title("Threat model types, held-out matches (AUC): how dangerous is this moment?")
    fig.suptitle("Phase 2, value models: model types compared (World Cup 2022 = 64 matches, Bundesliga = 7 matches, never seen)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


# ---------------------------------------------------------------- use from GRF / what-if

_cache = {}


def _load():
    if not _cache:
        _cache["xg"] = json.load(open(os.path.join(OUT, "xg_model.json")))
        _cache["xt"] = np.array(json.load(open(os.path.join(OUT, "xt_grid.json")))["grid"])
        p = os.path.join(OUT, "threat_model.json")
        _cache["threat"] = json.load(open(p)) if os.path.exists(p) else None
    return _cache


def _logit(m, f):
    x = (np.array([f[c] for c in m["features"]]) - np.array(m["mean"])) / np.array(m["std"])
    w = np.array(m["weights"])
    return float(1 / (1 + np.exp(-(w[0] + x @ w[1:]))))


def xg_shot(shooter, defenders, keeper=None, header=False):
    return _logit(_load()["xg"], shot_features(shooter, defenders, keeper, header=header))


def xt(ball):
    b = np.asarray(ball, float)
    g = _load()["xt"]
    v = g.ravel()[cell(b[..., 0], b[..., 1], g.shape[0], g.shape[1])]
    return float(v) if b.ndim == 1 else v


def threat(ball, attackers, defenders):
    """Chance this team shoots within 10 s, from where the ball and all players are (the state logistic model)."""
    m = _load()["threat"]
    f = {"s_" + k: v for k, v in state_features(ball, attackers, defenders).items()}
    return _logit(m, f)


# ---------------------------------------------------------------- main

def main():
    os.makedirs(OUT, exist_ok=True)
    sp, mp = os.path.join(OUT, "pff_shots.csv"), os.path.join(OUT, "pff_moves.csv.gz")
    if os.path.exists(sp) and os.path.exists(mp) and "s_dist" in pd.read_csv(mp, nrows=2).columns:
        shots, moves = pd.read_csv(sp), pd.read_csv(mp)
    else:
        print("extracting World Cup shots and moves with freeze frames (64 matches) ...", flush=True)
        shots, moves = extract_pff()
        shots.to_csv(sp, index=False)
        moves.to_csv(mp, index=False)
    dp = os.path.join(OUT, "dfl_shots.csv")
    if os.path.exists(dp):
        dfl = pd.read_csv(dp)
    else:
        dfl = extract_dfl()
        dfl.to_csv(dp, index=False)
    print(f"\nWorld Cup: {len(shots)} shots ({int(shots['goal'].sum())} goals, {int(shots['penalty'].sum())} penalties), "
          f"{int((moves['kind'] != 'shot').sum())} passes / crosses / carries;  Bundesliga: {len(dfl)} shots ({int(dfl['goal'].sum())} goals)")

    cv, folds, preds, open_, fitted, pen_rate = xg_cv(shots, dfl)
    cv.to_csv(os.path.join(OUT, "xg_cv.csv"), index=False)
    folds.to_csv(os.path.join(OUT, "xg_folds.csv"), index=False)
    ours = fitted["all features (ours)"]
    json.dump({**ours.to_json(), "penalty": pen_rate, "trained_on": f"{len(open_)} open-play and free-kick shots, World Cup 2022"},
              open(os.path.join(OUT, "xg_model.json"), "w"), indent=1)

    def make_geom(train_shots):
        g = Logit(SETS["distance + angle"]).fit(train_shots)
        return lambda c: float(g.predict(np.array([[shot_features(c, np.zeros((0, 2)))[k] for k in g.cols]]))[0])
    grid = fit_xt(moves, make_geom(shots[~shots["penalty"]]))
    json.dump({"nx": NX, "ny": NY, "length": L, "width": W, "attacking": "towards x = 105", "grid": grid.tolist()},
              open(os.path.join(OUT, "xt_grid.json"), "w"))
    thr, final = threat_cv(moves, shots, make_geom)
    thr.to_csv(os.path.join(OUT, "threat_cv.csv"), index=False)
    json.dump({**final["shot10"].to_json(), "target": "this team shoots within 10 s", "trained_on": "World Cup 2022 open-play moves"},
              open(os.path.join(OUT, "threat_model.json"), "w"), indent=1)
    for old in ("xt_check.csv", "xg_dfl.csv"):
        if os.path.exists(os.path.join(OUT, old)):
            os.remove(os.path.join(OUT, old))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["model", "auc", "brier", "log_loss", "predicted", "real", "logloss_vs_ours", "logloss_vs_ours_ci95",
            "beats_ours_in_groups", "bundesliga_auc", "bundesliga_predicted", "bundesliga_real"]
    print("\nxG model types (World Cup, leave-matches-out, 8 groups; Bundesliga = fitted on all World Cup shots):")
    print(cv[show].round(3).to_string(index=False))
    print(f"\npenalties: {pen_rate:.2f} (training conversion rate)")
    print("\nThreat model types (leave-matches-out):")
    print(thr.round(3).to_string(index=False))
    chart(cv, grid, thr, preds, open_, os.path.join(OUT, "value_models_chart.png"))
    print("wrote outputs/value_models/")


if __name__ == "__main__":
    main()
