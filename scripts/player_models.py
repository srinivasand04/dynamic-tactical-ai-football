"""own_PlayerModel (phase 5 of the player profile plan): models of ONE player, learned from his own matches, on top of the
shared models. Each player model is a small correction to the shared model, pulled towards zero (it only moves away
from the shared model as far as his own data supports), and it is judged only on matches it never saw.

A. On-ball choices (Bayer Leverkusen 2023/24, StatsBomb events + 360 frames, 34 matches, both teams)
   Every open-play decision on the ball: shot, dribble, carry (5 m+), long pass (30 m+), forward pass (5 m+ forward),
   safe pass (sideways / back). Situation: where he is, pressure, nearest opponent, opponents goal-side, team-mates ahead.
     shared model  : multinomial logistic regression on all players
     role model    : + a correction per role (CB, FB, DM, CM, W, AM, ST)
     player model  : + a correction for this player (intercepts, by pitch depth and pressure); its strength (L2) is
                     chosen by cross-validation inside his training matches
   Test: leave one match out (34 folds). For each Leverkusen player with 900+ min: held-out log-loss of role vs player
   model, matches where the player model is better, bootstrap interval over matches.
   Style: on his own situations, how often the player model expects each choice vs a typical player of his role.

B. Off-ball movement (Fortuna Düsseldorf, DFL tracking, 7 Bundesliga 2 matches)
   own_MovementGNN predicts where each player moves to the next Key State (fold model that never saw the match).
   The player model learns his systematic difference from it: a ridge regression of (real - predicted) movement on the
   situation (own team / opponent / nobody on the ball, where he is, where the ball is, time to the next Key State),
   in his team's attacking direction. Role correction first, then the player's, strength chosen inside his training
   matches. Test: leave one match out, players with 3+ matches of 30+ minutes: error in metres, GNN vs + role vs + player.
   Style: where he goes beyond what the GNN expects (forward / wide), with and without the ball.

Output: outputs/player_models/  onball_eval.csv, onball_style.json, movement_eval.csv, movement_style.json, summary.txt
        viewer/data/player_models.json (for the viewer's Player profile card)
Usage : set PYTHONPATH=scripts, then from the project root
        .venv-analysis\\Scripts\\python.exe scripts\\player_models.py onball | movement | viewer | all
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import player_of_match as PM  # noqa: E402

ROOT = PM.ROOT
OUT = os.path.join(ROOT, "outputs", "player_models")
PROF = os.path.join(ROOT, "outputs", "player_profile")
L, W = 105.0, 68.0
RNG = np.random.default_rng(0)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CLASSES = ["shot", "dribble", "carry", "long pass", "forward pass", "safe pass"]
LAMS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
MIN_PLAYER_MIN = 900


# ------------------------------------------------------------------ small fitting tools

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_mn(X, y, K, lam, offset=None, free_first=False):
    """Multinomial logistic regression with L2 (strength lam on the summed log-likelihood) and an optional offset
    (the logits of the model underneath). Returns B (p, K)."""
    n, p = X.shape
    off = np.zeros((n, K)) if offset is None else offset
    Y = np.eye(K)[y]
    pen = np.full((p, K), lam)
    if free_first:
        pen[0] = 1e-3                                                       # intercepts of the shared model: hardly pulled

    def f(b):
        B = b.reshape(p, K)
        Z = off + X @ B
        Z = Z - Z.max(1, keepdims=True)
        lse = np.log(np.exp(Z).sum(1))
        nll = -(Z[np.arange(n), y] - lse).sum() + (pen * B * B).sum()
        P = np.exp(Z - lse[:, None])
        g = X.T @ (P - Y) + 2 * pen * B
        return nll, g.ravel()

    r = minimize(f, np.zeros(p * K), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return r.x.reshape(p, K)


def nll_rows(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))


def ridge(X, Y, lam):
    p = X.shape[1]
    R = lam * np.eye(p)
    return np.linalg.solve(X.T @ X + R, X.T @ Y)


def boot_ci(v, n=2000):
    v = np.asarray(v, float)
    m = RNG.integers(0, len(v), (n, len(v)))
    b = v[m].mean(1)
    return float(np.percentile(b, 5)), float(np.percentile(b, 95))


def pick_lam(groups, fit_pred, lams=LAMS, folds=5):
    """Cross-validation inside the training matches only: groups = match id per row; fit_pred(train_mask, test_mask,
    lam) -> loss per test row. Returns the lam with the lowest mean loss."""
    u = np.unique(groups)
    if len(u) < 2:
        return lams[-1]
    fold_of = {g: i % folds for i, g in enumerate(RNG.permutation(u))}
    f = np.array([fold_of[g] for g in groups])
    best, bl = None, np.inf
    for lam in lams:
        tot, cnt = 0.0, 0
        for k in np.unique(f):
            te = f == k
            loss = fit_pred(~te, te, lam)
            tot += loss.sum()
            cnt += len(loss)
        if tot / cnt < bl:
            bl, best = tot / cnt, lam
    return best


# ------------------------------------------------------------------ A. on-ball choices (Leverkusen)

def onball_data():
    path = os.path.join(OUT, "onball_decisions.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path)
    A = pd.read_csv(os.path.join(PROF, "actions_leverkusen.csv.gz"))
    roles = pd.read_csv(os.path.join(PROF, "roles.csv"))[["match", "player_id", "role"]]
    A = A[A["kind"].isin(["Pass", "Carry", "Shot", "Dribble"])].copy()
    A = A[~((A["kind"] == "Pass") & A["sub_type"].isin(PM.SET_PIECE))]
    A = A[~((A["kind"] == "Shot") & (A["sub_type"] != "Open Play"))]
    A["dist"] = np.hypot(A["ex"] - A["x"], A["ey"] - A["y"])
    A = A[~((A["kind"] == "Carry") & ~(A["dist"] >= 5))]
    A = A.dropna(subset=["x", "y"])
    fwd = A["ex"] - A["x"]
    A["choice"] = np.select(
        [A["kind"] == "Shot", A["kind"] == "Dribble", A["kind"] == "Carry", (A["kind"] == "Pass") & (A["dist"] >= 30),
         (A["kind"] == "Pass") & (fwd >= 5)], [0, 1, 2, 3, 4], default=5)
    A = A.merge(roles, on=["match", "player_id"], how="left")
    A = A[A["role"].notna() & (A["role"] != "GK")]
    rows = []
    for mid, a in A.groupby("match"):
        tp = os.path.join(PM.SB, "three-sixty", f"{mid}.json")
        if not os.path.exists(tp):
            continue
        ffs = {f["event_uuid"]: f["freeze_frame"] for f in json.load(open(tp, encoding="utf-8"))}
        for r in a.itertuples():
            ff = ffs.get(r.id)
            if not ff:
                continue
            me = np.array([r.x, r.y])
            opp = np.array([PM.to_m(p["location"]) for p in ff if not p["teammate"]]).reshape(-1, 2)
            mates = np.array([PM.to_m(p["location"]) for p in ff if p["teammate"] and not p.get("actor")]).reshape(-1, 2)
            d = np.linalg.norm(opp - me, axis=1) if len(opp) else np.array([30.0])
            gs = ((opp[:, 0] > me[0]) & (np.abs(opp[:, 1] - me[1]) < 15)).sum() if len(opp) else 0
            rows.append({"match": mid, "id": r.id, "player_id": r.player_id, "player": r.player, "team": r.team,
                         "role": r.role, "x": r.x, "y": r.y, "pressed": r.pressed, "choice": r.choice,
                         "near_opp": min(d.min(), 15.0), "opp5": int((d < 5).sum()), "goal_side": int(gs),
                         "mates_ahead": int((mates[:, 0] > me[0] + 5).sum()) if len(mates) else 0, "added": r.added})
        print(f"  {mid}: {len(a)} decisions", flush=True)
    D = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    D.to_csv(path, index=False)
    return D


def onball_X(D):
    xn = D["x"].to_numpy() / L
    yc = np.abs(D["y"].to_numpy() - W / 2) / (W / 2)
    dg = np.hypot(L - D["x"].to_numpy(), W / 2 - D["y"].to_numpy()) / 50
    ang = np.abs(np.arctan2(W / 2 - D["y"].to_numpy(), L - D["x"].to_numpy()))
    box = ((D["x"] > 88.5) & (np.abs(D["y"] - W / 2) < 20.16)).to_numpy(float)
    pr = D["pressed"].to_numpy(float)
    no = D["near_opp"].to_numpy() / 15
    X = np.column_stack([np.ones(len(D)), xn, xn ** 2, yc, dg, ang, box, pr, no, D["opp5"].clip(0, 4) / 4,
                         D["goal_side"].clip(0, 8) / 8, D["mates_ahead"].clip(0, 8) / 8, pr * xn, box * no])
    Z = np.column_stack([np.ones(len(D)), xn - 0.5, pr])                    # what a role / player correction may depend on
    return X, Z


class OnballBase:
    """Shared + role model fitted on a set of rows."""
    def __init__(self, D, X, Z, mask):
        self.K = len(CLASSES)
        self.B = fit_mn(X[mask], D["choice"].to_numpy()[mask], self.K, 1.0, free_first=True)
        self.R = {}
        for role in D["role"].unique():
            m = mask & (D["role"] == role).to_numpy()
            self.R[role] = fit_mn(Z[m], D["choice"].to_numpy()[m], self.K, 10.0, offset=X[m] @ self.B)

    def logits(self, D, X, Z, idx):
        out = X[idx] @ self.B
        roles = D["role"].to_numpy()[idx]
        for role, Rb in self.R.items():
            m = roles == role
            out[m] += Z[idx][m] @ Rb
        return out


def onball():
    D = onball_data()
    X, Z = onball_X(D)
    y = D["choice"].to_numpy()
    mins = pd.read_csv(os.path.join(PROF, "player_match_leverkusen.csv")).groupby("player_id")["minutes"].sum()
    focus = [p for p in D.loc[D["team"] == PM.CLUB, "player_id"].unique() if mins.get(p, 0) >= MIN_PLAYER_MIN]
    print(f"{len(D)} decisions with a 360 frame, {D['match'].nunique()} matches; {len(focus)} Leverkusen players with "
          f"{MIN_PLAYER_MIN}+ min", flush=True)
    res = []
    matches = sorted(D["match"].unique())
    done = os.path.join(OUT, "onball_folds.csv")                            # saved folds are reused; delete the file to redo
    for t, mid in enumerate([] if os.path.exists(done) else matches):
        tr = (D["match"] != mid).to_numpy()
        base = OnballBase(D, X, Z, tr)
        for pid in focus:
            te_i = np.flatnonzero(((D["match"] == mid) & (D["player_id"] == pid)).to_numpy())
            if len(te_i) < 5:
                continue
            tr_i = np.flatnonzero(tr & (D["player_id"] == pid).to_numpy())
            off_tr, off_te = base.logits(D, X, Z, tr_i), base.logits(D, X, Z, te_i)
            shared_te = X[te_i] @ base.B
            g = D["match"].to_numpy()[tr_i]

            def fp(a, b, lam):
                Bp = fit_mn(Z[tr_i][a], y[tr_i][a], len(CLASSES), lam, offset=off_tr[a])
                return nll_rows(softmax(off_tr[b] + Z[tr_i][b] @ Bp), y[tr_i][b])

            lam = pick_lam(g, fp)
            Bp = fit_mn(Z[tr_i], y[tr_i], len(CLASSES), lam, offset=off_tr)
            freq = (np.bincount(y[tr_i], minlength=len(CLASSES)) + 1) / (len(tr_i) + len(CLASSES))
            yt = y[te_i]
            res.append({"match": mid, "player_id": pid, "player": D.loc[te_i[0], "player"], "role": D.loc[te_i[0], "role"],
                        "n": len(te_i), "lam": lam,
                        "nll_freq": nll_rows(np.tile(freq, (len(te_i), 1)), yt).mean(),
                        "nll_shared": nll_rows(softmax(shared_te), yt).mean(),
                        "nll_role": nll_rows(softmax(off_te), yt).mean(),
                        "nll_player": nll_rows(softmax(off_te + Z[te_i] @ Bp), yt).mean()})
        print(f"  fold {t + 1}/{len(matches)} {mid}", flush=True)
    R = pd.read_csv(done) if not res else pd.DataFrame(res)
    R.to_csv(done, index=False)
    rows = []
    for (pid, nm), g in R.groupby(["player_id", "player"]):
        role = g.groupby("role")["n"].sum().idxmax()                        # main role; the folds use each match's role
        gain = (g["nll_role"] - g["nll_player"]).to_numpy()                 # per match, weighted below by decisions
        w = g["n"].to_numpy()
        lo, hi = boot_ci(gain * w / w.mean())
        rows.append({"player_id": pid, "player": nm, "role": role, "matches": len(g), "decisions": int(w.sum()),
                     "logloss_own_freq": np.average(g["nll_freq"], weights=w), "logloss_shared": np.average(g["nll_shared"], weights=w),
                     "logloss_role": np.average(g["nll_role"], weights=w), "logloss_player": np.average(g["nll_player"], weights=w),
                     "gain_ci90_lo": lo, "gain_ci90_hi": hi, "matches_better": int((gain > 0).sum())})
    E = pd.DataFrame(rows)
    E["gain_pct"] = 100 * (E["logloss_role"] - E["logloss_player"]) / E["logloss_role"]
    E["verdict"] = np.where(E["gain_ci90_lo"] > 0, "player model better", np.where(E["gain_ci90_hi"] < 0, "role model better", "no clear difference"))
    E = E.sort_values("gain_pct", ascending=False)
    E.to_csv(os.path.join(OUT, "onball_eval.csv"), index=False)
    allw = R["n"]
    print("\nHeld-out log-loss (lower is better), all players pooled:")
    for c in ("nll_freq", "nll_shared", "nll_role", "nll_player"):
        print(f"  {c:12s} {np.average(R[c], weights=allw):.4f}")
    print(E[["player", "role", "matches", "decisions", "logloss_role", "logloss_player", "gain_pct", "matches_better", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    onball_style(D, X, Z, focus)


def onball_style(D, X, Z, focus):
    """Player models fitted on all his matches: on his own situations, the choice shares a typical player of his role
    would make vs what his model expects; by pressure and by pitch third; plus a zone map for the viewer."""
    y = D["choice"].to_numpy()
    E = pd.read_csv(os.path.join(OUT, "onball_eval.csv")).set_index("player_id")
    out = {}
    for pid in focus:
        i = np.flatnonzero((D["player_id"] == pid).to_numpy())
        main = E.loc[pid, "role"] if pid in E.index else D.loc[i, "role"].mode()[0]
        base = OnballBase(D, X, Z, (D["player_id"] != pid).to_numpy())     # the typical player of his role, without him
        off = base.logits(D.assign(role=main), X, Z, i)                     # judged as his main role throughout
        g = D["match"].to_numpy()[i]

        def fp(a, b, lam):
            Bp = fit_mn(Z[i][a], y[i][a], len(CLASSES), lam, offset=off[a])
            return nll_rows(softmax(off[b] + Z[i][b] @ Bp), y[i][b])

        lam = pick_lam(g, fp)
        Bp = fit_mn(Z[i], y[i], len(CLASSES), lam, offset=off)
        Pr, Pp = softmax(off), softmax(off + Z[i] @ Bp)
        sub = D.iloc[i]
        def shares(m):
            return {"n": int(m.sum()), "role": [round(float(v), 3) for v in Pr[m].mean(0)],
                    "player": [round(float(v), 3) for v in Pp[m].mean(0)],
                    "real": [round(float(v), 3) for v in np.bincount(y[i][m], minlength=len(CLASSES)) / max(m.sum(), 1)]}
        third = np.digitize(sub["x"].to_numpy(), [35, 70])
        chan = np.digitize(sub["y"].to_numpy(), [W / 3, 2 * W / 3])
        zones = []
        for a in range(3):
            for b in range(3):
                m = (third == a) & (chan == b)
                if m.sum() >= 20:
                    zones.append({"third": a, "channel": b, **shares(m)})
        r = E.loc[pid] if pid in E.index else None
        lines = []
        s = shares(np.ones(len(i), bool))
        for k, c in enumerate(CLASSES):
            ratio = s["player"][k] / max(s["role"][k], 1e-6)
            if s["player"][k] >= 0.01 and (ratio >= 1.25 or ratio <= 0.8):
                lines.append({"choice": c, "player": s["player"][k], "role": s["role"][k], "ratio": round(ratio, 2)})
        lines.sort(key=lambda q: -abs(np.log(q["ratio"])))
        out[str(pid)] = {"player": sub["player"].iloc[0], "role": main, "decisions": len(i), "lam": lam,
                         "all": s, "pressed": shares(sub["pressed"].to_numpy() == 1),
                         "free": shares(sub["pressed"].to_numpy() == 0), "zones": zones, "tendencies": lines,
                         "test": None if r is None else {k: (round(float(r[k]), 4) if isinstance(r[k], (float, np.floating)) else
                                                            (int(r[k]) if isinstance(r[k], (int, np.integer)) else r[k]))
                                                        for k in ("matches", "decisions", "logloss_role", "logloss_player",
                                                                  "gain_pct", "gain_ci90_lo", "gain_ci90_hi", "matches_better", "verdict")}}
    json.dump({"classes": CLASSES, "players": out}, open(os.path.join(OUT, "onball_style.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nTendencies (player model vs typical player of his role, his own situations):")
    for v in out.values():
        t = ", ".join(f"{q['choice']} x{q['ratio']}" for q in v["tendencies"][:4]) or "close to his role"
        print(f"  {v['player'][:26]:26s} {v['role']:3s} {t}")


# ------------------------------------------------------------------ B. off-ball movement (Düsseldorf)

def movement_data():
    path = os.path.join(OUT, "movement_rows.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path)
    import torch
    import train_gnn as T
    matches = pd.read_csv(os.path.join(PM.DFL, "matches.csv"))["match_id"].tolist()
    rows = []
    for m in matches:
        ids, fr, home_dir = PM.dfl_tracking(m)
        d = T.load(m)
        z = np.load(T.PATH[m] + f"_graphs{T.SUFFIX}.npz")
        g = {k: z[k] for k in ("frame", "time_s", "xy", "role")}               # read once (an npz re-reads on every access)
        net = T.Net(d["feat"].shape[2], 48, "ours")
        net.load_state_dict(torch.load(os.path.join(ROOT, "outputs", f"gnn_model_fold_{m}.pt"), map_location="cpu"))
        keep = []
        T.evaluate(net, d, keep=keep)
        pred = np.concatenate([k[0] for k in keep])
        targ = np.concatenate([k[1] for k in keep])
        ok = np.concatenate([k[2] for k in keep])
        roles = g["role"]
        frd = fr.to_dict("index")
        idd = {f: r for f, r in zip(ids.index, ids.to_numpy())}
        for j, k in enumerate(d["idx"]):
            f0 = int(g["frame"][k])
            if f0 not in frd or frd[f0]["game_mode"] != "normal":
                continue
            per, own = int(frd[f0]["period"]), frd[f0]["ball_owner_team"]
            gap = float(g["time_s"][k + 1] - g["time_s"][k])
            ball = g["xy"][k, 22]
            if np.isnan(ball[0]):
                continue
            for s in range(22):
                if not ok[j, s] or np.isnan(g["xy"][k, s, 0]):
                    continue
                home = s < 11
                sg = home_dir.get(per, 1) if home else -home_dir.get(per, 1)       # +1: his team attacks to x = 105
                p = g["xy"][k, s]
                px, py = (p[0], p[1]) if sg > 0 else (L - p[0], W - p[1])
                bx, by = (ball[0], ball[1]) if sg > 0 else (L - ball[0], W - ball[1])
                side = "home" if home else "away"
                rows.append({"match": m, "j": j, "dfl_id": idd[f0][s] if f0 in idd else None, "slot": s,
                             "role": int(roles[k, s]), "own": int(own == side), "opp": int(own in ("home", "away") and own != side),
                             "px": px, "py": py, "bx": bx, "by": by, "gap": gap,
                             "predx": sg * pred[j, s, 0], "predy": sg * pred[j, s, 1],
                             "realx": sg * targ[j, s, 0], "realy": sg * targ[j, s, 1]})
        print(f"  {m}: {len(d['idx'])} Key State steps", flush=True)
    M = pd.DataFrame(rows).dropna(subset=["dfl_id"])
    M.to_csv(path, index=False)
    return M


def movement_X(M):
    o, q = M["own"].to_numpy(float), M["opp"].to_numpy(float)
    xn, yn = M["px"].to_numpy() / L - 0.5, M["py"].to_numpy() / W - 0.5
    bx, by = (M["bx"] - M["px"]).to_numpy() / 50, (M["by"] - M["py"]).to_numpy() / 50
    gp = np.clip(M["gap"].to_numpy(), 0, 10) / 5
    return np.column_stack([np.ones(len(M)), o, q, gp, o * gp, q * gp, xn, yn, o * xn, q * xn, bx, by, o * bx, o * by])


def movement():
    M = movement_data()
    X = movement_X(M)
    Rres = (M[["realx", "realy"]].to_numpy() - M[["predx", "predy"]].to_numpy())
    names = pd.concat([pd.read_csv(os.path.join(PM.DFL, "raw", f"{m}_players.csv")).assign(match=m)
                       for m in M["match"].unique()])
    nm = names.drop_duplicates("player_id").set_index("player_id")
    per = M.groupby(["dfl_id", "match"]).size().unstack(fill_value=0)
    steps30 = (per >= 150).sum(1)                                            # ~30+ min on the pitch in that match
    focus = steps30[steps30 >= 3].index.tolist()
    print(f"{len(M)} player steps, {M['match'].nunique()} matches; {len(focus)} players with 3+ matches", flush=True)
    err = lambda r: np.linalg.norm(r, axis=1)
    res = []
    done = os.path.join(OUT, "movement_folds.csv")                          # saved folds are reused; delete the file to redo
    for mid in [] if os.path.exists(done) else sorted(M["match"].unique()):
        tr = (M["match"] != mid).to_numpy()
        roleB = {}
        for role in M["role"].unique():
            m = tr & (M["role"] == role).to_numpy()
            roleB[role] = ridge(X[m], Rres[m], 100.0)
        rc = np.zeros_like(Rres)
        for role, B in roleB.items():
            m = (M["role"] == role).to_numpy()
            rc[m] = X[m] @ B
        for pid in focus:
            te = np.flatnonzero(((M["match"] == mid) & (M["dfl_id"] == pid)).to_numpy())
            if len(te) < 50:
                continue
            ti = np.flatnonzero(tr & (M["dfl_id"] == pid).to_numpy())
            Y = Rres[ti] - rc[ti]
            g = M["match"].to_numpy()[ti]

            def fp(a, b, lam):
                B = ridge(X[ti][a], Y[a], lam)
                return err(Y[b] - X[ti][b] @ B)

            lam = pick_lam(g, fp, lams=[10, 30, 100, 300, 1000, 3000, 10000], folds=10)
            B = ridge(X[ti], Y, lam)
            r0 = Rres[te]
            res.append({"match": mid, "dfl_id": pid, "name": nm.loc[pid, "name"] if pid in nm.index else pid,
                        "team": nm.loc[pid, "team_name"] if pid in nm.index else "", "n": len(te), "lam": lam,
                        "err_gnn": err(r0).mean(), "err_role": err(r0 - rc[te]).mean(),
                        "err_player": err(r0 - rc[te] - X[te] @ B).mean()})
        print(f"  fold {mid}", flush=True)
    R = pd.read_csv(done) if not res else pd.DataFrame(res)
    R.to_csv(done, index=False)
    rows = []
    for (pid, name, team), g in R.groupby(["dfl_id", "name", "team"]):
        w = g["n"].to_numpy()
        gain = (g["err_role"] - g["err_player"]).to_numpy()
        lo, hi = boot_ci(gain * w / w.mean())
        rows.append({"dfl_id": pid, "name": name, "team": team, "matches": len(g), "steps": int(w.sum()),
                     "err_gnn_m": np.average(g["err_gnn"], weights=w), "err_role_m": np.average(g["err_role"], weights=w),
                     "err_player_m": np.average(g["err_player"], weights=w), "gain_ci90_lo_m": lo, "gain_ci90_hi_m": hi,
                     "matches_better": int((gain > 0).sum())})
    E = pd.DataFrame(rows)
    E["gain_pct"] = 100 * (E["err_role_m"] - E["err_player_m"]) / E["err_role_m"]
    E["verdict"] = np.where(E["gain_ci90_lo_m"] > 0, "player model better",
                            np.where(E["gain_ci90_hi_m"] < 0, "role model better", "no clear difference"))
    E = E.sort_values("gain_pct", ascending=False)
    E.to_csv(os.path.join(OUT, "movement_eval.csv"), index=False)
    w = R["n"]
    print("\nHeld-out movement error per player step (m), pooled:")
    for c in ("err_gnn", "err_role", "err_player"):
        print(f"  {c:11s} {np.average(R[c], weights=w):.3f}")
    print(E[["name", "matches", "steps", "err_gnn_m", "err_role_m", "err_player_m", "gain_pct", "matches_better", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    movement_style(M, X, Rres, focus, nm)


def movement_style(M, X, Rres, focus, nm):
    """Fitted on all his matches: the average extra movement (m per Key State step, his team attacking to the right)
    beyond the GNN and beyond his role's correction, with the ball (own team), against it, and in loose moments."""
    rc = np.zeros_like(Rres)
    for role in M["role"].unique():
        m = (M["role"] == role).to_numpy()
        rc[m] = X[m] @ ridge(X[m], Rres[m], 100.0)
    E = pd.read_csv(os.path.join(OUT, "movement_eval.csv")).set_index("dfl_id")
    out = {}
    for pid in focus:
        i = np.flatnonzero((M["dfl_id"] == pid).to_numpy())
        Y = Rres[i] - rc[i]
        g = M["match"].to_numpy()[i]
        lam = pick_lam(g, lambda a, b, lam: np.linalg.norm(Y[b] - X[i][b] @ ridge(X[i][a], Y[a], lam), axis=1),
                       lams=[10, 30, 100, 300, 1000, 3000, 10000], folds=10)
        B = ridge(X[i], Y, lam)
        corr = X[i] @ B
        sub = M.iloc[i]
        wide = np.sign(sub["py"].to_numpy() - W / 2)                          # + = towards his own touchline
        phases = {}
        for ph, m in (("with ball", sub["own"].to_numpy() == 1), ("without ball", sub["opp"].to_numpy() == 1)):
            if m.sum() < 30:
                continue
            phases[ph] = {"n": int(m.sum()), "forward_m": round(float(corr[m, 0].mean()), 2),
                          "wide_m": round(float((corr[m, 1] * wide[m]).mean()), 2),
                          "gnn_forward_m": round(float(sub["predx"].to_numpy()[m].mean()), 2),
                          "real_forward_m": round(float(sub["realx"].to_numpy()[m].mean()), 2)}
        r = E.loc[pid] if pid in E.index else None
        out[pid] = {"name": nm.loc[pid, "name"] if pid in nm.index else pid,
                    "team": nm.loc[pid, "team_name"] if pid in nm.index else "",
                    "position": nm.loc[pid, "position"] if pid in nm.index and "position" in nm.columns else "",
                    "matches": int(M.loc[M["dfl_id"] == pid, "match"].nunique()), "steps": len(i), "lam": lam, "phases": phases,
                    "test": None if r is None else {
                        "matches": int(r["matches"]), "err_gnn_m": round(float(r["err_gnn_m"]), 3),
                        "err_role_m": round(float(r["err_role_m"]), 3), "err_player_m": round(float(r["err_player_m"]), 3),
                        "gain_pct": round(float(r["gain_pct"]), 2), "gain_ci90_m": [round(float(r["gain_ci90_lo_m"]), 3),
                                                                                    round(float(r["gain_ci90_hi_m"]), 3)],
                        "matches_better": int(r["matches_better"]), "verdict": r["verdict"]}}
    json.dump({"players": out}, open(os.path.join(OUT, "movement_style.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nMovement style (beyond GNN and role, m per Key State step; + forward, + wide):")
    for v in out.values():
        ph = "  ".join(f"{k}: fwd {p['forward_m']:+.2f} wide {p['wide_m']:+.2f}" for k, p in v["phases"].items())
        print(f"  {v['name'][:22]:22s} {ph}")


# ------------------------------------------------------------------ viewer export

def viewer():
    ob = json.load(open(os.path.join(OUT, "onball_style.json"), encoding="utf-8"))
    mv = json.load(open(os.path.join(OUT, "movement_style.json"), encoding="utf-8"))
    by_name = {}
    for pid, v in ob["players"].items():
        by_name[v["player"]] = v
    json.dump({"model": "own_PlayerModel", "classes": ob["classes"], "onball": by_name, "movement": mv["players"],
               "onball_data": "Bayer Leverkusen 2023/24, StatsBomb events + 360, 34 matches, leave one match out",
               "movement_data": "7 Bundesliga matches (DFL tracking), own_MovementGNN fold models, leave one match out"},
              open(os.path.join(ROOT, "viewer", "data", "player_models.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print("wrote viewer/data/player_models.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["onball", "movement", "viewer", "all"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.step in ("onball", "all"):
        onball()
    if a.step in ("movement", "all"):
        movement()
    if a.step in ("viewer", "all"):
        viewer()
