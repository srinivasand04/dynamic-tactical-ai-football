"""own_TargetPlayer: seven opponent models built against ONE player, Granit Xhaka. He is the one well-known player who
appears in every dataset we have:

    Wyscout public     Arsenal, Premier League 2017/18 (38 of 38 matches), World Cup 2018, Euro 2016
    StatsBomb 360      Leverkusen 2023/24 (30 of 31 matches with 360 data), Euro 2020, World Cup 2022, Euro 2024
    PFF FC tracking    World Cup 2022 (4 matches, off-ball movement)

The question is the one a coach asks before a match: where does he hurt us, and what do we do about it?

    m1 decisions   What will he do with the ball? The hierarchical on-ball choice model (shared -> role -> player) from
                   player_models.py / player_models_big.py, his choice mix in every context against a typical DM / CM.
                   The held-out tests are reused from player_models_big (sb_eval, wy_eval, wy_national).
    m2 threat      Where does he create threat? An expected-threat (xT) grid (16 x 12, value iteration) from all 197
                   StatsBomb matches; his xT per 90 from passes and carries, percentile among DM / CM, by pitch zone.
                   The same grid scores his Wyscout passes (Arsenal 17/18, World Cup 2018, Euro 2016).
    m3 lanes       Which passing lanes carry that threat? Zone -> zone lanes of his successful passes / carries, ranked
                   by xT; left-foot share, length, main receivers.
    m4 pressure    From which side should we press him? The nearest opponent in the 360 freeze frame (within 5 m)
                   gives the side of the pressure (front / his left / his right / behind); completion, forward share
                   and xT per pass for each side, against all DM / CM in the same situation.
    m5 ball loss   Where can we win the ball from him? A ball-loss model (logistic, population of DM / CM plus his own
                   correction), tested on his held-out matches; risk by zone and pressing side, and the value of a regain
                   (our xT at the place we win it).
    m6 movement    Where is he when he does not have the ball? From World Cup 2022 tracking: his position as a function
                   of the ball (in and out of possession) against a typical DM / CM, tested leave one match out, and
                   where his own movement surprises the movement GNN.
    m7 counter     What-if counter plan: his real situations replayed under 7 defensive plans (let him play, press from
                   the front / left / right / behind, double him, screen + press). For each plan: his predicted choice
                   mix (m1), ball-loss risk (m5), threat he still creates (m2) and the counter value for us; the plan
                   with the lowest net threat.

The what-if scenarios change the situation the models see; they are model answers ("what the data says he does in
such situations"), not a causal experiment.

Outputs: outputs/target_player/  (csv per step, summary_<step>.txt, target.json); viewer/data/target_player.json
Usage  : python scripts/target_player.py data | m1 .. m7 | viewer | all      (run from the project root)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import player_models as P  # noqa: E402
import player_models_big as BG  # noqa: E402
import player_of_match as PM  # noqa: E402

ROOT = PM.ROOT
OUT = os.path.join(ROOT, "outputs", "target_player")
VIEW = os.path.join(ROOT, "viewer", "data", "target_player.json")
L, W = 105.0, 68.0
NAME, T_SB, T_WY, T_PFF = "Granit Xhaka", 3500, 49876, 157
MIDS = ("DM", "CM")
LEV = BG.LEV
GX, GY = 16, 12                                   # xT grid
ZX, ZY = 6, 4                                     # coarse zones for maps and lanes
SIDES = ["none", "front", "left", "right", "behind"]
WY_KEYS = (("England", "Premier League"), ("World_Cup", "World Cup 2018"), ("European_Championship", "Euro 2016"))
CONTEXTS = ["Premier League", "Euro 2016", "World Cup 2018", "Euro 2020", "World Cup 2022", LEV, "Euro 2024"]
CTX_LABEL = {"Premier League": "Arsenal 17/18", "Euro 2016": "Euro 2016", "World Cup 2018": "World Cup 2018",
             "Euro 2020": "Euro 2020", "World Cup 2022": "World Cup 2022", LEV: "Leverkusen 23/24", "Euro 2024": "Euro 2024"}
RNG = np.random.default_rng(7)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def save(step, obj=None):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, f"summary_{step}.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    LOG.clear()
    if obj is not None:
        path = os.path.join(OUT, "target.json")
        T = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        T[step] = obj
        json.dump(T, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


def load(step):
    return json.load(open(os.path.join(OUT, "target.json"), encoding="utf-8"))[step]


# ------------------------------------------------------------------ pitch tools

def cell(x, y, nx=GX, ny=GY):
    cx = np.clip((np.asarray(x, float) / L * nx).astype(int), 0, nx - 1)
    cy = np.clip((np.asarray(y, float) / W * ny).astype(int), 0, ny - 1)
    return cx * ny + cy


def zone(x, y):
    return cell(x, y, ZX, ZY)


ZXN = ["own box", "own third", "own half", "opp. half", "final third", "opp. box"]
ZYN = ["right wing", "right-centre", "left-centre", "left wing"]        # metres: y up = the attacking team's left


def zone_name(z):
    return f"{ZXN[z // ZY]}, {ZYN[z % ZY]}"


def goal_dist(x, y):
    return np.hypot(L - np.asarray(x, float), W / 2 - np.asarray(y, float))


def side_of(ff, me):
    """Side of the nearest opponent within 5 m, seen from the player facing the opponent goal (+x; +y is his left)."""
    opp = np.array([PM.to_m(p["location"]) for p in ff if not p["teammate"]]).reshape(-1, 2)
    if not len(opp):
        return "none"
    v = opp - me
    d = np.hypot(v[:, 0], v[:, 1])
    k = d.argmin()
    if d[k] > 5:
        return "none"
    a = np.degrees(np.arctan2(v[k, 1], v[k, 0]))
    return "front" if abs(a) <= 45 else "behind" if abs(a) >= 135 else "left" if a > 0 else "right"


# ------------------------------------------------------------------ data: all on-ball actions, StatsBomb + Wyscout

KINDS = ("Pass", "Carry", "Shot", "Dribble", "Miscontrol", "Dispossessed")


def sb_actions():
    path = os.path.join(OUT, "sb_actions.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path)
    mins = pd.read_csv(os.path.join(BG.OUT, "sb_minutes.csv"))
    role = {(m, p): r for m, p, r in zip(mins["match"], mins["player_id"], mins["role"])}
    jobs = [(PM.SB, m["match_id"], LEV, m["match_date"])
            for m in json.load(open(os.path.join(PM.SB, "matches_9_281.json"), encoding="utf-8"))]
    for (c, s), src in BG.TOURN.items():
        jobs += [(BG.SBT, m["match_id"], src, m["match_date"])
                 for m in json.load(open(os.path.join(BG.SBT, f"matches_{c}_{s}.json"), encoding="utf-8"))]
    rows = []
    for i, (base, mid, src, date) in enumerate(jobs, 1):
        evp = os.path.join(base, "events", f"{mid}.json")
        if not os.path.exists(evp):
            continue
        ev = json.load(open(evp, encoding="utf-8"))
        tp = os.path.join(base, "three-sixty", f"{mid}.json")
        ffs = {f["event_uuid"]: f["freeze_frame"] for f in json.load(open(tp, encoding="utf-8"))} if os.path.exists(tp) else {}
        for e in ev:
            kind = e["type"]["name"]
            if kind not in KINDS or "location" not in e or "player" not in e:
                continue
            sub = e.get(kind.lower()) or {}
            st = (sub.get("type") or {}).get("name", "")
            me = PM.to_m(e["location"])
            end = PM.to_m(sub["end_location"][:2]) if "end_location" in sub else np.array([np.nan, np.nan])
            out = (sub.get("outcome") or {}).get("name", "")
            ok = {"Pass": out == "", "Carry": True, "Dribble": out == "Complete", "Shot": out == "Goal"}.get(kind, False)
            pid = e["player"]["id"]
            ff = ffs.get(e["id"])
            r = {"source": src, "date": date, "match": mid, "id": e["id"], "poss": e.get("possession"), "team": e["team"]["name"],
                 "player_id": pid, "player": e["player"]["name"], "role": role.get((mid, pid), "CM"), "kind": kind,
                 "setp": int((kind == "Pass" and st in PM.SET_PIECE) or (kind == "Shot" and st != "Open Play")),
                 "x": me[0], "y": me[1], "ex": end[0], "ey": end[1], "ok": int(ok),
                 "pressed": int(bool(e.get("under_pressure"))), "body": (sub.get("body_part") or {}).get("name", ""),
                 "recipient": (sub.get("recipient") or {}).get("name", ""), "xg": float(sub.get("statsbomb_xg") or 0),
                 "minute": e["minute"], "has_ff": int(ff is not None)}
            if ff:
                r["near_opp"], r["opp5"], r["goal_side"], r["mates_ahead"] = BG.ff_feats(ff, me)
                r["side"] = side_of(ff, me)
            rows.append(r)
        if i % 20 == 0:
            print(f"  StatsBomb {i}/{len(jobs)}: {len(rows)} actions", flush=True)
    A = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    A.to_csv(path, index=False)
    return A


def wy_actions():
    """Passes, shots and carries (Accelerations) of every player in the three Wyscout competitions he played.
    Coordinates in metres with y up = the attacking team's left (as the StatsBomb rows)."""
    path = os.path.join(OUT, "wy_actions.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path)
    parts = []
    per_order = {"1H": 0, "2H": 1, "E1": 2, "E2": 3, "P": 4}
    for key, comp in WY_KEYS:
        t0 = time.time()
        ev = json.load(open(os.path.join(BG.WY, "events", f"events_{key}.json"), encoding="utf-8"))
        c = {k: [] for k in ("match", "team_id", "player_id", "ev", "sub", "per", "t", "x0", "y0", "x1", "y1", "acc", "goal", "left")}
        for e in ev:
            pos = e.get("positions") or [{"x": np.nan, "y": np.nan}]
            p1 = pos[1] if len(pos) > 1 else {"x": np.nan, "y": np.nan}
            tg = {t["id"] for t in e.get("tags") or []}
            c["match"].append(e["matchId"]); c["team_id"].append(e["teamId"]); c["player_id"].append(e["playerId"])
            c["ev"].append(e["eventName"]); c["sub"].append(e["subEventName"]); c["per"].append(per_order.get(e["matchPeriod"], 5))
            c["t"].append(e["eventSec"]); c["x0"].append(pos[0]["x"]); c["y0"].append(pos[0]["y"])
            c["x1"].append(p1["x"]); c["y1"].append(p1["y"]); c["acc"].append(int(1801 in tg)); c["goal"].append(int(101 in tg))
            c["left"].append(int(401 in tg))
        del ev
        E = pd.DataFrame(c)
        same = (E["match"] == E["match"].shift(1)) & (E["per"] == E["per"].shift(1))
        same2 = (E["match"] == E["match"].shift(2)) & (E["per"] == E["per"].shift(2))
        dt1, dt2 = (E["t"] - E["t"].shift(1)).where(same), (E["t"] - E["t"].shift(2)).where(same2)
        opp1 = same & (E["team_id"] != E["team_id"].shift(1))
        opp2 = same2 & (E["team_id"] != E["team_id"].shift(2))
        E["pressed"] = ((opp1 & (dt1 < 2)) | (opp2 & (dt2 < 2))).astype(int)
        keep = ((E["ev"] == "Pass") & (E["sub"] != "Hand pass")) | (E["ev"] == "Shot") | (E["sub"] == "Acceleration")
        E = E[keep & (E["player_id"] > 0)].copy()
        E["source"] = comp
        E["x"], E["y"] = E["x0"] * L / 100, W - E["y0"] * W / 100
        E["ex"], E["ey"] = E["x1"] * L / 100, W - E["y1"] * W / 100
        E["setp"] = 0
        E["kind"] = np.where(E["ev"] == "Shot", "Shot", np.where(E["sub"] == "Acceleration", "Carry", "Pass"))
        E["ok"] = np.where(E["kind"] == "Shot", E["goal"], np.where(E["kind"] == "Carry", 1, E["acc"]))
        parts.append(E[["source", "match", "team_id", "player_id", "kind", "sub", "setp", "x", "y", "ex", "ey", "ok", "pressed", "left"]])
        say(f"  Wyscout {comp}: {len(E)} passes / shots / carries ({time.time() - t0:.0f}s)")
    A = pd.concat(parts, ignore_index=True)
    A.to_csv(path, index=False)
    return A


def decisions_of(A):
    """The on-ball decision rows of player_models_big.sb_match, from the action table (same rules)."""
    d = A[A["kind"].isin(["Pass", "Carry", "Shot", "Dribble"]) & (A["setp"] == 0) & (A["has_ff"] == 1) & (A["role"] != "GK")].copy()
    dist = np.hypot(d["ex"] - d["x"], d["ey"] - d["y"])
    d = d[~((d["kind"] == "Carry") & ~(dist >= 5))]
    dist = np.hypot(d["ex"] - d["x"], d["ey"] - d["y"])
    d["choice"] = np.select([d["kind"] == "Shot", d["kind"] == "Dribble", d["kind"] == "Carry", dist >= 30,
                             d["ex"] - d["x"] >= 5], [0, 1, 2, 3, 4], default=5)
    return d


def minutes():
    sb = pd.read_csv(os.path.join(BG.OUT, "sb_minutes.csv"))
    wy = pd.read_csv(os.path.join(BG.OUT, "wy_minutes.csv")).rename(columns={"comp": "source"})
    return sb, wy


# ------------------------------------------------------------------ m2 tools: expected threat

def xt_grid(A):
    op = A[A["setp"] == 0]
    mv = op[op["kind"].isin(["Pass", "Carry"]) & op["ex"].notna()]
    sh = op[op["kind"] == "Shot"]
    n = GX * GY
    nm = np.bincount(cell(mv["x"], mv["y"]), minlength=n).astype(float)
    cs = cell(sh["x"], sh["y"])
    ns = np.bincount(cs, minlength=n).astype(float)
    ng = np.bincount(cs, weights=sh["ok"].to_numpy(float), minlength=n)
    tot = np.maximum(ns + nm, 1)
    s, m = ns / tot, nm / tot
    g = (ng + 0.5 * ng.sum() / ns.sum()) / (ns + 0.5)                    # light shrink for cells with few shots
    su = mv[mv["ok"] == 1]
    T = np.zeros((n, n))
    np.add.at(T, (cell(su["x"], su["y"]), cell(su["ex"], su["ey"])), 1.0)
    T /= np.maximum(nm, 1)[:, None]
    xt = np.zeros(n)
    for _ in range(60):
        xt = s * g + m * (T @ xt)
    return xt


def add_gain(A, xt):
    mv = A["kind"].isin(["Pass", "Carry"]) & A["ex"].notna() & (A["ok"] == 1) & (A["setp"] == 0)
    g = np.zeros(len(A))
    g[mv.to_numpy()] = xt[cell(A.loc[mv, "ex"], A.loc[mv, "ey"])] - xt[cell(A.loc[mv, "x"], A.loc[mv, "y"])]
    A["gain"] = g
    return A


def main_role(df, key="player_id"):
    return df.groupby(key)["role"].agg(lambda s: s.value_counts().index[0])


# ------------------------------------------------------------------ data step

def step_data():
    t0 = time.time()
    A = sb_actions()
    Wa = wy_actions()
    h = A[A["player_id"] == T_SB]
    say(f"StatsBomb: {len(A)} on-ball actions, {A['match'].nunique()} matches; {NAME}: {len(h)} actions in "
        f"{h['match'].nunique()} matches, {int(h['has_ff'].sum())} with a 360 frame")
    for s, g in h.groupby("source"):
        say(f"  {s:18s} {g['match'].nunique():2d} matches {len(g):5d} actions")
    hw = Wa[Wa["player_id"] == T_WY]
    say(f"Wyscout: {len(Wa)} passes / shots / carries; {NAME}: {len(hw)} in {hw['match'].nunique()} matches")
    for s, g in hw.groupby("source"):
        say(f"  {s:18s} {g['match'].nunique():2d} matches {len(g):5d} actions")
    M, names = BG.pff_rows()
    k = M["pid"] == T_PFF
    say(f"PFF World Cup 2022: {int(k.sum())} visible tracking steps of {NAME} in {M.loc[k, 'match'].nunique()} matches")
    say(f"({time.time() - t0:.0f}s)")
    save("data", {"name": NAME, "contexts": [
        {"key": s, "label": CTX_LABEL[s], "matches": int(g["match"].nunique()), "actions": int(len(g)),
         "data": "Wyscout events" if s in dict(WY_KEYS).values() else "StatsBomb 360 events"}
        for s, g in pd.concat([h[["source", "match"]], hw[["source", "match"]]]).groupby("source")]
        + [{"key": "PFF", "label": "World Cup 2022 tracking", "matches": int(M.loc[k, "match"].nunique()),
            "actions": int(k.sum()), "data": "PFF tracking (off-ball)"}]})


# ------------------------------------------------------------------ m1 decisions

def sb_base():
    D, _ = BG.sb_data()
    X, Z = P.onball_X(D)
    return D, X, Z, P.OnballBase(D, X, Z, np.ones(len(D), bool))


def his_fit(D, X, Z, base):
    i = np.flatnonzero((D["player_id"] == T_SB).to_numpy())
    Dm = D.iloc[i].assign(role="DM")
    off = base.logits(Dm, X[i], Z[i], np.arange(len(i)))
    Bp, lam = BG.fit_player(Z[i], D["choice"].to_numpy()[i], off, D["match"].to_numpy()[i])
    return Bp, lam


def step_m1():
    D, X, Z, base = sb_base()
    Bp, lam = his_fit(D, X, Z, base)
    np.save(os.path.join(OUT, "m1_player_offset.npy"), Bp)
    cls = P.CLASSES
    rows = []
    # StatsBomb contexts: real share vs what a typical DM does in the same situations (shared + role model)
    for s in CONTEXTS:
        k = ((D["player_id"] == T_SB) & (D["source"] == s)).to_numpy()
        if not k.any():
            continue
        i = np.flatnonzero(k)
        Dm = D.iloc[i].assign(role="DM")
        role_p = P.softmax(base.logits(Dm, X[i], Z[i], np.arange(len(i)))).mean(0)
        real = np.bincount(D["choice"].to_numpy()[i], minlength=len(cls)) / len(i)
        rows.append({"context": s, "data": "StatsBomb", "decisions": len(i), "role": "DM",
                     **{f"real_{c}": real[j] for j, c in enumerate(cls)}, **{f"role_{c}": role_p[j] for j, c in enumerate(cls)}})
    # Wyscout contexts: same, with the Wyscout model (no freeze frames) fitted on a sample of all five leagues + tournaments
    Dw = BG.wy_data()
    samp = np.zeros(len(Dw), bool)
    samp[RNG.choice(len(Dw), 400000, replace=False)] = True
    samp |= (Dw["player_id"] == T_WY).to_numpy()
    Dw = Dw[samp].reset_index(drop=True)
    Xw, Zw = BG.wy_X(Dw)
    bw = P.OnballBase(Dw, Xw, Zw, np.ones(len(Dw), bool))
    for s in CONTEXTS:
        k = ((Dw["player_id"] == T_WY) & (Dw["source"] == s)).to_numpy()
        if not k.any():
            continue
        i = np.flatnonzero(k)
        rl = Dw["role"].iloc[i].mode()[0]
        role_p = P.softmax(bw.logits(Dw.iloc[i], Xw[i], Zw[i], np.arange(len(i)))).mean(0)
        real = np.bincount(Dw["choice"].to_numpy()[i], minlength=len(cls)) / len(i)
        rows.append({"context": s, "data": "Wyscout", "decisions": len(i), "role": rl,
                     **{f"real_{c}": real[j] for j, c in enumerate(cls)}, **{f"role_{c}": role_p[j] for j, c in enumerate(cls)}})
    C = pd.DataFrame(rows).set_index("context").loc[[c for c in CONTEXTS if c in {r["context"] for r in rows}]].reset_index()
    C.to_csv(os.path.join(OUT, "m1_contexts.csv"), index=False)
    say(f"M1. {NAME}: choice mix per context (real share minus a typical {'/'.join(MIDS)} in the same situations, pts)")
    say("  " + f"{'context':18s} {'n':>5s} " + " ".join(f"{c:>12s}" for c in cls))
    for _, r in C.iterrows():
        say("  " + f"{CTX_LABEL[r['context']]:18s} {r['decisions']:5d} " + " ".join(f"{100 * (r[f'real_{c}'] - r[f'role_{c}']):+12.1f}" for c in cls))
    # stability: correlation of the difference vectors between contexts
    dif = np.array([[r[f"real_{c}"] - r[f"role_{c}"] for c in cls] for _, r in C.iterrows()])
    cc = np.corrcoef(dif)
    mean_r = float(cc[np.triu_indices(len(C), 1)].mean())
    say(f"  style vectors (real - role) correlate across his {len(C)} contexts: mean r = {mean_r:.2f}")
    # held-out tests already run in player_models_big
    ev = {}
    for f, key, lab in (("sb_eval.csv", T_SB, "StatsBomb 5 folds (43 matches)"), ("wy_eval.csv", T_WY, "Wyscout Arsenal, last quarter of the season")):
        e = pd.read_csv(os.path.join(BG.OUT, f))
        e = e[e["player_id"] == key].iloc[0]
        ev[lab] = {"matches": int(e.matches), "decisions": int(e.decisions), "gain_pct": float(e.gain_pct), "verdict": e.verdict}
    nat = pd.read_csv(os.path.join(BG.OUT, "wy_national.csv"))
    for _, e in nat[nat["player_id"] == T_WY].iterrows():
        ev[f"Arsenal style used at {e.tournament}"] = {"decisions": int(e.test_decisions), "gain_pct": float(e.gain_pct),
                                                       "verdict": "player better" if e.gain_pct > 0 else "role better"}
    for k2, v in ev.items():
        say(f"  held-out: {k2:45s} gain over role model {v['gain_pct']:+.2f}%  ({v['verdict']})")
    sim = pd.read_csv(os.path.join(BG.OUT, "wy_similar.csv"))
    srow = sim[sim.iloc[:, 0] == "G. Xhaka"]
    similar = [str(v) for v in srow.iloc[0, 4:9]] if len(srow) else []
    say("  most alike in 2017/18 (Wyscout, same role): " + "; ".join(similar))
    # what-if: zone third x pressure -> his choice mix vs typical DM (his own situations, pressure switched)
    i = np.flatnonzero((D["player_id"] == T_SB).to_numpy())
    Dh = D.iloc[i].assign(role="DM").reset_index(drop=True)
    grid = []
    third = np.digitize(Dh["x"], [35, 70])
    for t, tn in enumerate(["own third", "middle third", "final third"]):
        for pr in (0, 1):
            Dq = Dh[third == t].assign(pressed=pr)
            if len(Dq) < 30:
                continue
            Xq, Zq = P.onball_X(Dq)
            o = base.logits(Dq, Xq, Zq, np.arange(len(Dq)))
            ph, pr_ = P.softmax(o + Zq @ Bp).mean(0), P.softmax(o).mean(0)
            grid.append({"third": tn, "pressed": pr, "n": len(Dq), "his": [round(float(v), 4) for v in ph],
                         "role": [round(float(v), 4) for v in pr_]})
            say(f"  what-if {tn:12s} pressed={pr}: his " + " ".join(f"{c} {100 * v:.0f}%" for c, v in zip(cls, ph))
                + " | DM " + " ".join(f"{100 * v:.0f}%" for v in pr_))
    pq = pd.read_csv(os.path.join(BG.OUT, "sb_pressure.csv"))
    pq = pq[pq["player_id"] == T_SB].iloc[0]
    save("m1", {"classes": cls, "contexts": [{"context": CTX_LABEL[r.context], "data": r.data, "decisions": int(r.decisions),
                                              "role": r.role, "real": [float(r[f"real_{c}"]) for c in cls],
                                              "typical": [float(r[f"role_{c}"]) for c in cls]} for _, r in C.iterrows()],
                "style_r": mean_r, "held_out": ev, "similar": similar, "whatif": grid,
                "pressure": [float(pq[f"all_{c}"]) for c in cls], "lam": lam})


# ------------------------------------------------------------------ m2 threat

def per90(df, mins_by, key="player_id", weight=None):
    v = df.groupby(key)[weight].sum() if weight else df.groupby(key).size()
    m = mins_by.reindex(v.index)
    return (90 * v / m).where(m >= 1)


def step_m2():
    A = sb_actions()
    xt = xt_grid(A)
    np.save(os.path.join(OUT, "xt_grid.npy"), xt)
    A = add_gain(A, xt)
    Wa = add_gain(wy_actions(), xt)
    sbm, wym = minutes()
    say(f"M2. xT grid {GX} x {GY} from {A['match'].nunique()} StatsBomb matches; max cell value {xt.max():.3f}")
    # population: DM / CM with 900+ minutes in the StatsBomb pool
    role = main_role(sbm[sbm["role"] != "GK"])
    mtot = sbm.groupby("player_id")["minutes"].sum()
    pop_ids = [p for p in role.index if role[p] in MIDS and mtot.get(p, 0) >= 360]
    op = A[(A["setp"] == 0) & A["kind"].isin(["Pass", "Carry"])]
    pas = op[op["kind"] == "Pass"]
    tot = per90(op, mtot, weight="gain").reindex(pop_ids).dropna()
    pt = per90(pas, mtot, weight="gain").reindex(pop_ids).dropna()
    him_tot, him_p = float(tot.get(T_SB)), float(pt.get(T_SB))
    pct_tot, pct_p = float((tot < him_tot).mean() * 100), float((pt < him_p).mean() * 100)
    say(f"  {NAME}, all StatsBomb matches: xT per 90 {him_tot:.3f} (passes {him_p:.3f}) -> percentile {pct_tot:.0f} "
        f"(passes {pct_p:.0f}) among {len(pop_ids)} DM / CM with 360+ min (4+ full matches); population median {tot.median():.3f}")
    top = tot.sort_values(ascending=False).head(10)
    nm = A.drop_duplicates("player_id").set_index("player_id")["player"]
    say("  top 10 DM / CM by xT per 90: " + "; ".join(f"{nm.get(p, p)} {v:.3f}" for p, v in top.items()))
    ctx = []

    def prof(df, mins, label, data):
        dp = df[(df["kind"] == "Pass") & (df["setp"] == 0)]
        okp = dp[dp["ok"] == 1]
        prog = okp[goal_dist(okp["ex"], okp["ey"]) <= 0.75 * goal_dist(okp["x"], okp["y"])]
        f3 = okp[(okp["x"] < 70) & (okp["ex"] >= 70)]
        box = okp[(okp["ex"] > 88.5) & (np.abs(okp["ey"] - W / 2) < 20.16)]
        sh = df[(df["kind"] == "Shot") & (df["setp"] == 0)]
        k = 90 / mins
        r = {"context": label, "data": data, "minutes": round(float(mins)), "passes90": len(dp) * k,
             "completion": float(dp["ok"].mean()), "xt_pass90": float(dp["gain"].sum() * k),
             "xt_carry90": float(df.loc[df["kind"] == "Carry", "gain"].sum() * k), "prog90": len(prog) * k,
             "final_third90": len(f3) * k, "box90": len(box) * k, "shots90": len(sh) * k,
             "long_share": float((np.hypot(dp["ex"] - dp["x"], dp["ey"] - dp["y"]) >= 30).mean())}
        ctx.append(r)
    for s in CONTEXTS:
        if s in dict(WY_KEYS).values():
            h = Wa[(Wa["player_id"] == T_WY) & (Wa["source"] == s)]
            mm = wym[(wym["player_id"] == T_WY) & (wym["source"] == s)]["minutes"].sum()
            if len(h):
                prof(h, mm, CTX_LABEL[s], "Wyscout")
        else:
            h = A[(A["player_id"] == T_SB) & (A["source"] == s)]
            mm = sbm[(sbm["player_id"] == T_SB) & (sbm["source"] == s)]["minutes"].sum()
            if len(h):
                prof(h, mm, CTX_LABEL[s], "StatsBomb")
    C = pd.DataFrame(ctx)
    C.to_csv(os.path.join(OUT, "m2_contexts.csv"), index=False)
    say("  per context (per 90): " + C.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    # Wyscout population: Premier League DM / CM, passes only (comparable with StatsBomb passes)
    wr = BG.wy_data()
    wr = wr[wr["source"] == "Premier League"]
    wrole = main_role(wr)
    wmt = wym[wym["source"] == "Premier League"].groupby("player_id")["minutes"].sum()
    wpop = [p for p in wrole.index if wrole[p] in MIDS and wmt.get(p, 0) >= 900]
    wp = Wa[(Wa["source"] == "Premier League") & (Wa["kind"] == "Pass")]
    wv = per90(wp, wmt, weight="gain").reindex(wpop).dropna()
    pct_w = float((wv < wv.get(T_WY)).mean() * 100)
    say(f"  Arsenal 17/18 (Wyscout): xT per 90 from passes {wv.get(T_WY):.3f} -> percentile {pct_w:.0f} among {len(wpop)} "
        f"Premier League DM / CM with 900+ min (median {wv.median():.3f})")
    # zone map: his xT per 90 by start zone vs the DM / CM population (per 90 of their minutes)
    hz = op[op["player_id"] == T_SB]
    hm = mtot[T_SB]
    pz = op[op["player_id"].isin(pop_ids)]
    pm_ = mtot.reindex(pop_ids).sum()
    his_z = np.bincount(zone(hz["x"], hz["y"]), weights=hz["gain"], minlength=ZX * ZY) * 90 / hm
    pop_z = np.bincount(zone(pz["x"], pz["y"]), weights=pz["gain"], minlength=ZX * ZY) * 90 / pm_
    cnt_z = np.bincount(zone(hz["x"], hz["y"]), minlength=ZX * ZY) * 90 / hm
    order = np.argsort(-his_z)
    say("  where his threat starts (xT per 90 by zone, vs a typical DM / CM):")
    for z in order[:5]:
        say(f"    {zone_name(z):32s} {his_z[z]:.3f}  (typical {pop_z[z]:.3f}, x{his_z[z] / max(pop_z[z], 1e-4):.1f})  "
            f"{cnt_z[z]:.1f} actions / 90")
    share_pass = float(hz.loc[hz["kind"] == "Pass", "gain"].sum() / hz["gain"].sum())
    say(f"  share of his xT from passes: {100 * share_pass:.0f}% (carries {100 * (1 - share_pass):.0f}%)")
    save("m2", {"xt90": him_tot, "xt_pass90": him_p, "pct": pct_tot, "pct_pass": pct_p, "pop_n": len(pop_ids),
                "pop_median": float(tot.median()), "wy_xt_pass90": float(wv.get(T_WY)), "wy_pct": pct_w, "wy_pop_n": len(wpop),
                "top10": [[str(nm.get(p, p)), float(v)] for p, v in top.items()], "contexts": C.to_dict("records"),
                "zones_his": his_z.tolist(), "zones_pop": pop_z.tolist(), "zones_count": cnt_z.tolist(),
                "top_zones": [{"zone": zone_name(z), "z": int(z), "his": float(his_z[z]), "typical": float(pop_z[z])} for z in order[:5]],
                "share_pass": share_pass, "grid": xt.reshape(GX, GY).tolist()})


# ------------------------------------------------------------------ m3 lanes

def step_m3():
    A = add_gain(sb_actions(), np.load(os.path.join(OUT, "xt_grid.npy")))
    sbm, _ = minutes()
    hm = sbm[sbm["player_id"] == T_SB]["minutes"].sum()
    h = A[(A["player_id"] == T_SB) & (A["setp"] == 0) & A["kind"].isin(["Pass", "Carry"]) & (A["ok"] == 1)].copy()
    h["z0"], h["z1"] = zone(h["x"], h["y"]), zone(h["ex"], h["ey"])
    h["len"] = np.hypot(h["ex"] - h["x"], h["ey"] - h["y"])
    pos = h[h["gain"] > 0]
    G = pos.groupby(["z0", "z1"]).agg(n=("gain", "size"), xt=("gain", "sum"), length=("len", "mean"),
                                       left=("body", lambda b: (b == "Left Foot").mean()),
                                       carry=("kind", lambda k: (k == "Carry").mean())).reset_index()
    G = G.sort_values("xt", ascending=False)
    tot = pos["gain"].sum()
    lanes = []
    say(f"M3. {NAME}: threat lanes (successful passes / carries that add xT, zone -> zone), {len(pos)} actions")
    for r in G.head(8).itertuples():
        rec = pos[(pos["z0"] == r.z0) & (pos["z1"] == r.z1) & (pos["recipient"] != "")]["recipient"].value_counts().head(3)
        lanes.append({"from": zone_name(r.z0), "to": zone_name(r.z1), "z0": int(r.z0), "z1": int(r.z1), "per90": r.n * 90 / hm,
                      "xt90": r.xt * 90 / hm, "share": r.xt / tot, "length": r.length, "left_foot": r.left, "carry": r.carry,
                      "receivers": [[k, int(v)] for k, v in rec.items()]})
        say(f"  {zone_name(r.z0):30s} -> {zone_name(r.z1):30s} {r.n * 90 / hm:4.1f}/90  xT {r.xt * 90 / hm:.3f}/90 "
            f"({100 * r.xt / tot:4.1f}%)  {r.length:4.1f} m  left foot {100 * r.left:3.0f}%  carries {100 * r.carry:3.0f}%  "
            + ", ".join(f"{k} {v}" for k, v in rec.items()))
    top_share = float(G.head(8)["xt"].sum() / tot)
    p = h[h["kind"] == "Pass"]
    sw = p[(np.abs(p["ey"] - p["y"]) >= 25)]
    say(f"  top 8 lanes = {100 * top_share:.0f}% of his added xT; left foot on {100 * (p['body'] == 'Left Foot').mean():.0f}% of his "
        f"completed passes; switches (25+ m across) {len(sw) * 90 / hm:.1f} / 90")
    rec = pos[pos["recipient"] != ""].groupby(["source", "recipient"])["gain"].agg(["size", "sum"]).reset_index()
    rec = rec.sort_values("sum", ascending=False).head(8)
    say("  who he feeds (xT added): " + "; ".join(f"{r.recipient} ({CTX_LABEL[r.source]}) {r['sum']:.2f}" for _, r in rec.iterrows()))
    # direction of his completed passes: which side of the pitch he plays to
    to_left = float((p["ey"] - p["y"] > 5).mean())
    to_right = float((p["ey"] - p["y"] < -5).mean())
    say(f"  completed passes to his left {100 * to_left:.0f}%, to his right {100 * to_right:.0f}%")
    save("m3", {"lanes": lanes, "top_share": top_share, "left_foot": float((p["body"] == "Left Foot").mean()),
                "switch90": len(sw) * 90 / hm, "to_left": to_left, "to_right": to_right,
                "receivers": [[r.recipient, CTX_LABEL[r.source], float(r["sum"])] for _, r in rec.iterrows()]})


# ------------------------------------------------------------------ m4 pressure side

def step_m4():
    A = add_gain(sb_actions(), np.load(os.path.join(OUT, "xt_grid.npy")))
    role = main_role(A[A["role"] != "GK"])
    p = A[(A["kind"] == "Pass") & (A["setp"] == 0) & (A["has_ff"] == 1)].copy()
    p["fwd"] = (p["ex"] - p["x"] >= 5).astype(int)
    p["lf"] = (p["body"] == "Left Foot").astype(int)
    h = p[p["player_id"] == T_SB]
    pop = p[p["player_id"].map(role).isin(MIDS) & (p["player_id"] != T_SB)]
    rows = []
    for s in SIDES:
        a, b = h[h["side"] == s], pop[pop["side"] == s]
        rows.append({"side": s, "n": len(a), "completion": a["ok"].mean(), "fwd": a["fwd"].mean(), "xt": a["gain"].mean(),
                     "left_foot": a["lf"].mean(), "pop_n": len(b), "pop_completion": b["ok"].mean(), "pop_fwd": b["fwd"].mean(),
                     "pop_xt": b["gain"].mean()})
    S = pd.DataFrame(rows).set_index("side")
    # drop from 'nobody within 5 m', his drop minus the typical DM / CM drop (negative = pressure hurts him more)
    for c in ("completion", "fwd", "xt"):
        S[f"rel_{c}"] = (S[c] - S.loc["none", c]) - (S[f"pop_{c}"] - S.loc["none", f"pop_{c}"])
    # bootstrap over his matches for the completion difference per side
    ci = {}
    ms = h["match"].unique()
    for s in SIDES[1:]:
        v = []
        for _ in range(500):
            mm = RNG.choice(ms, len(ms))
            hb = pd.concat([h[h["match"] == m] for m in mm])
            a, n0 = hb[hb["side"] == s], hb[hb["side"] == "none"]
            if len(a) and len(n0):
                v.append(a["ok"].mean() - n0["ok"].mean() - (S.loc[s, "pop_completion"] - S.loc["none", "pop_completion"]))
        ci[s] = (float(np.percentile(v, 5)), float(np.percentile(v, 95)))
    S.to_csv(os.path.join(OUT, "m4_pressure_side.csv"))
    say(f"M4. {NAME}: passes by the side of the nearest opponent (within 5 m) in the 360 frame; {len(h)} passes, "
        f"population {len(pop)} passes of DM / CM")
    for s, r in S.iterrows():
        extra = "" if s == "none" else (f"  vs typical: completion {100 * r.rel_completion:+.1f} pts (90%: {100 * ci[s][0]:+.1f} .. "
                                        f"{100 * ci[s][1]:+.1f}), forward {100 * r.rel_fwd:+.1f} pts, xT/pass {1000 * r.rel_xt:+.2f} e-3")
        say(f"  {s:7s} n={r.n:5.0f} completion {100 * r.completion:5.1f}% (typ {100 * r.pop_completion:5.1f}%)  forward "
            f"{100 * r.fwd:4.1f}%  xT/pass {1000 * r.xt:5.2f}e-3  left foot {100 * r.left_foot:3.0f}%{extra}")
    cand = S.loc[["front", "left", "right", "behind"]]
    cand = cand[cand["n"] >= 40]
    best = cand["rel_xt"].idxmin() if len(cand) else "front"
    say(f"  -> pressing side that costs him most threat beyond a typical DM / CM: {best}")
    pr = A[(A["player_id"] == T_SB) & (A["kind"] == "Pass") & (A["setp"] == 0)]
    say(f"  StatsBomb 'under pressure' flag: completion {100 * pr.loc[pr.pressed == 1, 'ok'].mean():.1f}% pressed vs "
        f"{100 * pr.loc[pr.pressed == 0, 'ok'].mean():.1f}% not ({int(pr.pressed.sum())} pressed passes, {100 * pr.pressed.mean():.0f}%)")
    save("m4", {"sides": [{"side": s, **{k: (None if pd.isna(v) else float(v)) for k, v in r.items()}, "ci": ci.get(s)}
                          for s, r in S.iterrows()], "best": best,
                "pressed_completion": float(pr.loc[pr.pressed == 1, "ok"].mean()),
                "unpressed_completion": float(pr.loc[pr.pressed == 0, "ok"].mean()), "pressed_share": float(pr.pressed.mean())})


# ------------------------------------------------------------------ m5 ball loss

def loss_rows(A):
    k = (A["has_ff"] == 1) & (((A["kind"] == "Pass") & (A["setp"] == 0)) | A["kind"].isin(["Dribble", "Miscontrol", "Dispossessed"]))
    d = A[k].copy()
    d["loss"] = 1 - d["ok"]
    return d


def loss_X(d):
    xn = d["x"].to_numpy() / L
    yc = np.abs(d["y"].to_numpy() - W / 2) / (W / 2)
    pr = d["pressed"].to_numpy(float)
    sd = {s: (d["side"] == s).to_numpy(float) for s in SIDES[1:]}
    F = np.column_stack([np.ones(len(d)), xn, xn ** 2, yc, pr, d["near_opp"].to_numpy() / 15, d["opp5"].clip(0, 4) / 4,
                         d["goal_side"].clip(0, 8) / 8, sd["front"], sd["left"], sd["right"], sd["behind"], pr * xn])
    Zl = np.column_stack([np.ones(len(d)), pr, sd["front"], sd["left"], sd["right"], xn - 0.5])
    return F, Zl


def fit_loss(F, Zl, y, groups, mask_pop, mask_his):
    B0 = P.fit_mn(F[mask_pop], y[mask_pop], 2, 1.0, free_first=True)
    off = F[mask_his] @ B0
    Zh, yh, gh = Zl[mask_his], y[mask_his], groups[mask_his]

    def fp(a, b, lam):
        Bp = P.fit_mn(Zh[a], yh[a], 2, lam, offset=off[a])
        return P.nll_rows(P.softmax(off[b] + Zh[b] @ Bp), yh[b])
    lam = P.pick_lam(gh, fp, folds=5)
    return B0, P.fit_mn(Zh, yh, 2, lam, offset=off), lam


def step_m5():
    from sklearn.metrics import roc_auc_score
    A = sb_actions()
    xt = np.load(os.path.join(OUT, "xt_grid.npy"))
    role = main_role(A[A["role"] != "GK"])
    d = loss_rows(A).reset_index(drop=True)
    d = d[d["player_id"].map(role).isin(MIDS)].reset_index(drop=True)
    F, Zl = loss_X(d)
    y = d["loss"].to_numpy().astype(int)
    his = (d["player_id"] == T_SB).to_numpy()
    g = d["match"].to_numpy()
    say(f"M5. Ball-loss model: {len(d)} passes / dribbles / miscontrols of DM / CM with a 360 frame; {NAME}: {int(his.sum())} "
        f"(loss rate {100 * y[his].mean():.1f}%, typical {100 * y[~his].mean():.1f}%)")
    ms = np.unique(g[his])
    fold = {m: i % 5 for i, m in enumerate(RNG.permutation(ms))}
    res, auc = [], []
    for f in range(5):
        te_m = [m for m in ms if fold[m] == f]
        te = his & np.isin(g, te_m)
        tr = ~np.isin(g, te_m)
        B0, Bp, lam = fit_loss(F, Zl, y, g, tr & ~his, tr & his)
        o = F[te] @ B0
        p0, p1 = P.softmax(o)[:, 1], P.softmax(o + Zl[te] @ Bp)[:, 1]
        for m in te_m:
            k = g[te] == m
            res.append({"match": m, "n": int(k.sum()), "nll_pop": float(P.nll_rows(P.softmax(o[k]), y[te][k]).mean()),
                        "nll_his": float(P.nll_rows(P.softmax(o[k] + Zl[te][k] @ Bp), y[te][k]).mean()),
                        "p_pop": float(p0[k].mean()), "p_his": float(p1[k].mean()), "real": float(y[te][k].mean())})
        auc.append((roc_auc_score(y[te], p0), roc_auc_score(y[te], p1), int(te.sum())))
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "m5_folds.csv"), index=False)
    w = R["n"]
    gain = 100 * (1 - np.average(R.nll_his, weights=w) / np.average(R.nll_pop, weights=w))
    lo, hi = P.boot_ci(((R.nll_pop - R.nll_his) * w / w.mean()).to_numpy())
    auc0 = np.average([a[0] for a in auc], weights=[a[2] for a in auc])
    auc1 = np.average([a[1] for a in auc], weights=[a[2] for a in auc])
    say(f"  held-out ({len(ms)} matches, 5 folds): log-loss typical {np.average(R.nll_pop, weights=w):.4f}  his model "
        f"{np.average(R.nll_his, weights=w):.4f} ({gain:+.2f}%, 90% per match {lo:+.4f} .. {hi:+.4f}); AUC {auc0:.3f} -> {auc1:.3f}")
    say(f"  calibration on his held-out matches: predicted loss {100 * np.average(R.p_his, weights=w):.1f}%, real "
        f"{100 * np.average(R.real, weights=w):.1f}%")
    B0, Bp, lam = fit_loss(F, Zl, y, g, ~his, his)
    np.save(os.path.join(OUT, "m5_pop.npy"), B0)
    np.save(os.path.join(OUT, "m5_his.npy"), Bp)
    dh = d[his].reset_index(drop=True)
    sbm, _ = minutes()
    hm = sbm[sbm["player_id"] == T_SB]["minutes"].sum()

    def risk(dq, sc):
        dq = dq.copy()
        if sc == "none":
            dq["pressed"], dq["near_opp"], dq["opp5"], dq["side"] = 0, np.maximum(dq["near_opp"], 8), 0, "none"
        else:
            dq["pressed"], dq["near_opp"], dq["opp5"], dq["side"] = 1, np.minimum(dq["near_opp"], 1.5), np.maximum(dq["opp5"], 1), sc
        Fq, Zq = loss_X(dq)
        return P.softmax(Fq @ B0 + Zq @ Bp)[:, 1], P.softmax(Fq @ B0)[:, 1]
    zh = zone(dh["x"], dh["y"])
    ours = xt[cell(L - dh["x"], W - dh["y"])]                                   # our xT if we win it there
    rows = []
    for z in range(ZX * ZY):
        k = zh == z
        if k.sum() < 25:
            continue
        r = {"z": z, "zone": zone_name(z), "n": int(k.sum()), "per90": float(k.sum() * 90 / hm), "real": float(dh["loss"][k].mean()),
             "ours_xt": float(ours[k].mean())}
        for sc in ["none"] + SIDES[1:]:
            ph, pp = risk(dh[k], sc)
            r[f"risk_{sc}"], r[f"typ_{sc}"] = float(ph.mean()), float(pp.mean())
        best = min(SIDES[1:], key=lambda s: -r[f"risk_{s}"])
        r["best_side"] = best
        r["win90"] = r["per90"] * r[f"risk_{best}"]                            # balls we could win per 90 if every touch here is pressed
        r["value90"] = r["win90"] * r["ours_xt"]
        rows.append(r)
    Zt = pd.DataFrame(rows).sort_values("value90", ascending=False)
    Zt.to_csv(os.path.join(OUT, "m5_zones.csv"), index=False)
    say("  where to win the ball (his touches pressed from the best side; value = our xT where we win it):")
    for r in Zt.head(6).itertuples():
        say(f"    {r.zone:32s} {r.per90:4.1f} touches/90  risk free {100 * r.risk_none:4.1f}%  pressed ({r.best_side}) "
            f"{100 * getattr(r, 'risk_' + r.best_side):4.1f}% (typical {100 * getattr(r, 'typ_' + r.best_side):4.1f}%)  "
            f"regain value {r.value90:.3f} xT/90")
    allr = {sc: float(risk(dh, sc)[0].mean()) for sc in ["none"] + SIDES[1:]}
    typr = {sc: float(risk(dh, sc)[1].mean()) for sc in ["none"] + SIDES[1:]}
    say("  his loss risk on all his touches: " + "  ".join(f"{sc} {100 * v:.1f}% (typ {100 * typr[sc]:.1f}%)" for sc, v in allr.items()))
    save("m5", {"n": int(his.sum()), "loss_rate": float(y[his].mean()), "typ_rate": float(y[~his].mean()), "gain_pct": gain,
                "ci": [lo, hi], "auc": [auc0, auc1], "calib": [float(np.average(R.p_his, weights=w)), float(np.average(R.real, weights=w))],
                "risk_all": allr, "risk_typ": typr, "zones": Zt.head(8).to_dict("records"), "lam": lam})


# ------------------------------------------------------------------ m6 movement (PFF tracking)

def mv_X(bx, by, own, opp):
    a, b = np.asarray(bx) / L - 0.5, np.asarray(by) / W - 0.5
    base = np.column_stack([np.ones(len(a)), a, b, a * a, b * b, a * b])
    gates = [np.asarray(own, float), np.asarray(opp, float), 1 - np.asarray(own, float) - np.asarray(opp, float)]
    return np.column_stack([base * gt[:, None] for gt in gates])


def step_m6():
    M, names = BG.pff_rows()
    M = M[M["role"].isin([4, 5])].reset_index(drop=True)                     # DM / CM (build_graphs roles)
    his = (M["pid"] == T_PFF).to_numpy()
    X = mv_X(M["bx"], M["by"], M["own"], M["opp"])
    Y = M[["px", "py"]].to_numpy()
    m = M["match"].to_numpy()
    say(f"M6. Off-ball position model (PFF World Cup 2022): {len(M)} visible steps of DM / CM, {NAME}: {int(his.sum())} "
        f"in {len(np.unique(m[his]))} matches")
    say(f"  his mean position (metres, own team attacking +x): in possession ({M.loc[his & (M.own == 1), 'px'].mean():.1f}, "
        f"{M.loc[his & (M.own == 1), 'py'].mean():.1f}), out of possession ({M.loc[his & (M.opp == 1), 'px'].mean():.1f}, "
        f"{M.loc[his & (M.opp == 1), 'py'].mean():.1f}); typical DM / CM ({M.loc[~his, 'px'].mean():.1f}, {M.loc[~his, 'py'].mean():.1f})")
    res = []
    for mid in np.unique(m[his]):
        tr, te = ~(m == mid), his & (m == mid)
        Bpop = P.ridge(X[tr & ~his], Y[tr & ~his], 10.0)
        k = tr & his
        Bh = P.ridge(X[k], Y[k] - X[k] @ Bpop, 30.0)
        e0 = np.linalg.norm(Y[te] - X[te] @ Bpop, axis=1).mean()
        e1 = np.linalg.norm(Y[te] - X[te] @ (Bpop + Bh), axis=1).mean()
        res.append({"match": mid, "n": int(te.sum()), "err_typical": e0, "err_his": e1})
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "m6_folds.csv"), index=False)
    say("  leave one match out, mean position error (m): " + "  ".join(f"{r.match}: typical {r.err_typical:.2f} -> his {r.err_his:.2f}"
                                                                      for r in R.itertuples()))
    Bpop = P.ridge(X[~his], Y[~his], 10.0)
    Bh = P.ridge(X[his], Y[his] - X[his] @ Bpop, 30.0)
    grid = []
    for phase, own, opp in (("we have the ball", 1, 0), ("they have the ball", 0, 1)):
        for bx in (25.0, 52.5, 80.0):
            for by in (12.0, 34.0, 56.0):
                Xq = mv_X([bx], [by], [own], [opp])
                t, hh = (Xq @ Bpop)[0], (Xq @ (Bpop + Bh))[0]
                grid.append({"phase": phase, "ball": [bx, by], "typical": t.tolist(), "his": hh.tolist(),
                             "dx": float(hh[0] - t[0]), "dy": float(hh[1] - t[1])})
    say("  what-if ball position -> his position minus a typical DM / CM (m, +x = nearer the opponent goal, +y = the left in "
        "this frame); 'his team' = the team of the player:")
    for gq in grid:
        say(f"    {gq['phase']:18s} ball ({gq['ball'][0]:4.1f}, {gq['ball'][1]:4.1f}): his ({gq['his'][0]:5.1f}, {gq['his'][1]:5.1f}) "
            f"typical ({gq['typical'][0]:5.1f}, {gq['typical'][1]:5.1f})  diff ({gq['dx']:+.1f}, {gq['dy']:+.1f})")
    # where his movement surprises the movement GNN (residual = real - GNN prediction per Key State step)
    rr = M[["realx", "realy"]].to_numpy() - M[["predx", "predy"]].to_numpy()
    gap = np.clip(M["gap"].to_numpy(), 0.2, 5)
    sur = {}
    for phase, k in (("in possession", M["own"] == 1), ("out of possession", M["opp"] == 1)):
        k = k.to_numpy()
        a = rr[his & k] / gap[his & k, None]
        b = rr[~his & k] / gap[~his & k, None]
        sur[phase] = {"his": a.mean(0).tolist(), "typical": b.mean(0).tolist(),
                      "err_his": float(np.linalg.norm(rr[his & k], axis=1).mean()), "err_typ": float(np.linalg.norm(rr[~his & k], axis=1).mean())}
        say(f"  GNN surprise {phase:18s}: his mean residual ({a.mean(0)[0]:+.2f}, {a.mean(0)[1]:+.2f}) m/s vs typical "
            f"({b.mean(0)[0]:+.2f}, {b.mean(0)[1]:+.2f}); GNN error {sur[phase]['err_his']:.2f} m vs {sur[phase]['err_typ']:.2f} m")
    save("m6", {"folds": R.to_dict("records"), "err_typical": float(np.average(R.err_typical, weights=R.n)),
                "err_his": float(np.average(R.err_his, weights=R.n)), "grid": grid, "surprise": sur,
                "mean_in": [float(M.loc[his & (M.own == 1), "px"].mean()), float(M.loc[his & (M.own == 1), "py"].mean())],
                "mean_out": [float(M.loc[his & (M.opp == 1), "px"].mean()), float(M.loc[his & (M.opp == 1), "py"].mean())],
                "steps": int(his.sum()), "matches": int(len(R))})


# ------------------------------------------------------------------ m7 what-if counter plan

SCEN = [
    ("as played", {}),
    ("let him play: nobody within 8 m", {"pressed": 0, "near_opp": ("max", 8.0), "opp5": 0, "side": "none"}),
    ("press from the front", {"pressed": 1, "near_opp": ("min", 1.5), "opp5": ("max", 1), "side": "front"}),
    ("press from his left", {"pressed": 1, "near_opp": ("min", 1.5), "opp5": ("max", 1), "side": "left"}),
    ("press from his right", {"pressed": 1, "near_opp": ("min", 1.5), "opp5": ("max", 1), "side": "right"}),
    ("press from behind (on the receive)", {"pressed": 1, "near_opp": ("min", 1.5), "opp5": ("max", 1), "side": "behind"}),
    ("double him: 2 within 5 m", {"pressed": 1, "near_opp": ("min", 1.2), "opp5": ("max", 2), "side": "front"}),
    ("screen + press: 2 more goal-side", {"pressed": 1, "near_opp": ("min", 2.5), "opp5": ("max", 1), "side": "front",
                                          "goal_side": ("add", 2)}),
]


def apply(d, sc):
    d = d.copy()
    for k, v in sc.items():
        if isinstance(v, tuple):
            f, a = v
            d[k] = np.maximum(d[k], a) if f == "max" else np.minimum(d[k], a) if f == "min" else d[k] + a
        else:
            d[k] = v
    return d


def step_m7():
    A = add_gain(sb_actions(), np.load(os.path.join(OUT, "xt_grid.npy")))
    xt = np.load(os.path.join(OUT, "xt_grid.npy"))
    D, X, Z, base = sb_base()
    Bp = np.load(os.path.join(OUT, "m1_player_offset.npy"))
    B0, Bl = np.load(os.path.join(OUT, "m5_pop.npy")), np.load(os.path.join(OUT, "m5_his.npy"))
    role = main_role(A[A["role"] != "GK"])
    dec = decisions_of(A)
    dec["value"] = np.where(dec["choice"] == 0, dec["xg"], dec["gain"])
    dec["third"] = np.digitize(dec["x"], [35, 70])
    h = dec[dec["player_id"] == T_SB].assign(role="DM").reset_index(drop=True)
    pop = dec[dec["player_id"].map(role).isin(MIDS)]
    # value of a choice when it comes off, per third: his mean, shrunk to the DM / CM mean (k = 30 actions)
    V = np.zeros((3, 6))
    for t in range(3):
        for c in range(6):
            a = h[(h["third"] == t) & (h["choice"] == c)]
            b = pop[(pop["third"] == t) & (pop["choice"] == c)]
            if c == 1:                                                          # dribble: value of his carries there
                a = h[(h["third"] == t) & (h["choice"] == 2)]
                b = pop[(pop["third"] == t) & (pop["choice"] == 2)]
            ok_a, ok_b = a[a["ok"] == 1] if c != 0 else a, b[b["ok"] == 1] if c != 0 else b
            mb = ok_b["value"].mean() if len(ok_b) else 0.0
            V[t, c] = (ok_a["value"].sum() + 30 * mb) / (len(ok_a) + 30)
    say(f"M7. What-if counter plan on {NAME}'s {len(h)} real on-ball situations (StatsBomb 360)")
    say("  value of a choice that comes off (xT, shots = xG), own / middle / final third:")
    for c, nm in enumerate(P.CLASSES):
        say(f"    {nm:12s} " + "  ".join(f"{V[t, c]:.4f}" for t in range(3)))
    ours = xt[cell(L - h["x"], W - h["y"])]
    rows = []
    for name, sc in SCEN:
        q = apply(h, sc)
        Xq, Zq = P.onball_X(q)
        o = base.logits(q, Xq, Zq, np.arange(len(q)))
        pc = P.softmax(o + Zq @ Bp)
        Fq, Zl = loss_X(q.assign(side=q["side"].fillna("none")))
        risk = P.softmax(Fq @ B0 + Zl @ Bl)[:, 1]
        val = (pc * V[q["third"].to_numpy()]).sum(1)
        threat = val * (1 - risk)
        counter = risk * ours
        rows.append({"plan": name, "mix": pc.mean(0).tolist(), "risk": float(risk.mean()), "threat100": float(100 * threat.mean()),
                     "counter100": float(100 * counter.mean()), "net100": float(100 * (threat - counter).mean())})
    S = pd.DataFrame(rows)
    ref = S.loc[S.plan == "let him play: nobody within 8 m", "threat100"].iloc[0]
    S["vs_free"] = 100 * (S["threat100"] - ref) / ref                        # change in the threat he creates
    S.to_csv(os.path.join(OUT, "m7_plans.csv"), index=False)
    say("  per 100 of his on-ball actions (xT): threat he creates, our counter value from balls we win, net = threat - counter")
    for r in S.itertuples():
        say(f"    {r.plan:36s} risk {100 * r.risk:4.1f}%  threat {r.threat100:.3f}  counter {r.counter100:.3f}  net "
            f"{r.net100:+.3f} (his threat {r.vs_free:+.0f}% vs letting him play)  mix " + " ".join(f"{100 * v:.0f}" for v in r.mix))
    best = S[S.plan != "as played"].sort_values("net100").iloc[0]
    say(f"  -> best plan: {best.plan} (net {best.net100:+.3f} per 100 actions, his threat {best.vs_free:+.0f}% vs letting him play)")
    save("m7", {"plans": S.to_dict("records"), "best": best.plan, "values": V.tolist(), "n": len(h)})


# ------------------------------------------------------------------ viewer export + recommendations

def step_viewer():
    T = json.load(open(os.path.join(OUT, "target.json"), encoding="utf-8"))
    m1, m2, m3, m4, m5, m6, m7 = (T[k] for k in ("m1", "m2", "m3", "m4", "m5", "m6", "m7"))
    side = {s["side"]: s for s in m4["sides"]}
    b4 = side[m4["best"]]
    lane = m3["lanes"][0]
    z5 = m5["zones"][0]
    best7 = next(p for p in m7["plans"] if p["plan"] == m7["best"])
    free7 = next(p for p in m7["plans"] if p["plan"].startswith("let him play"))
    g_big = max(m6["grid"], key=lambda g: np.hypot(g["dx"], g["dy"]))
    sb_ho = next(v for k, v in m1["held_out"].items() if k.startswith("StatsBomb"))
    sideword = {"front": "the front", "left": "his left", "right": "his right", "behind": "behind"}
    rec = [
        {"model": "M2 threat", "text": f"He is a real threat: {m2['xt90']:.3f} xT per 90 (percentile "
         f"{m2['pct']:.0f} of {m2['pop_n']} DM / CM; Arsenal 17/18 percentile {m2['wy_pct']:.0f}); most of it starts in {m2['top_zones'][0]['zone']} "
         f"({m2['top_zones'][0]['his'] / max(m2['top_zones'][0]['typical'], 1e-4):.1f}x a typical DM / CM there)."},
        {"model": "M3 lanes", "text": f"Close the lane {lane['from']} -> {lane['to']}: alone {100 * lane['share']:.0f}% of the threat he adds "
         f"(top 8 lanes {100 * m3['top_share']:.0f}%); {100 * m3['left_foot']:.0f}% of his completed passes are left-footed"
         + (f"; main receivers {', '.join(r[0] for r in lane['receivers'][:2])}." if lane["receivers"] else ".")},
        {"model": "M4 pressure", "text": f"Pressure does not make him lose the ball, it makes him play safe: completion holds "
         f"({100 * b4['completion']:.0f}% pressed from {sideword[m4['best']]} vs {100 * side['none']['completion']:.0f}% free, better than a "
         f"typical DM / CM) but forward passes fall {100 * (side['none']['fwd'] - b4['fwd']):.0f} pts and his threat per pass "
         f"{1000 * side['none']['xt']:.1f} -> {1000 * b4['xt']:.1f} e-3. Press from {sideword[m4['best']]} costs him most."},
        {"model": "M5 ball loss", "text": f"Hunt the ball in {z5['zone']} (his own half = our high press): {z5['per90']:.1f} touches per 90, loss risk "
         f"{100 * z5['risk_none']:.0f}% free -> {100 * z5['risk_' + z5['best_side']]:.0f}% pressed from {sideword[z5['best_side']]}; "
         f"a regain there is worth {z5['ours_xt']:.3f} xT to us."},
        {"model": "M6 movement", "text": f"Off the ball he holds his side: wherever the ball is, he stands about "
         f"{np.mean([g['dy'] for g in m6['grid']]):.1f} m wider to his side and {-np.mean([g['dx'] for g in m6['grid']]):.1f} m deeper than a "
         f"typical DM / CM (World Cup 2022 tracking), so the space is on his far side and in front of him."},
        {"model": "M7 counter plan", "text": f"Best plan: {best7['plan']}: his threat {best7['vs_free']:+.0f}% vs letting him play, "
         f"loss risk {100 * best7['risk']:.0f}% (free {100 * free7['risk']:.0f}%), net {best7['net100']:+.2f} xT per 100 actions in our favour."},
        {"model": "M1 decisions", "text": f"His choices are close to a typical DM (player model {sb_ho['gain_pct']:+.1f}%, {sb_ho['verdict']}) "
         f"and his small deviations change from context to context (style r {m1['style_r']:.2f} across his 7 contexts): prepare for a "
         f"typical DM; the danger is where and how well he does it (M2, M3), not surprise choices."},
    ]
    out = {"name": NAME, "contexts": T["data"]["contexts"], "recommendations": rec, **T}
    json.dump(out, open(VIEW, "w", encoding="utf-8"), ensure_ascii=False, default=float)
    for r in rec:
        say(f"  {r['model']:16s} {r['text']}")
    say(f"viewer data: {VIEW} ({os.path.getsize(VIEW) // 1024} KB)")
    save("viewer")


STEPS = {"data": step_data, "m1": step_m1, "m2": step_m2, "m3": step_m3, "m4": step_m4, "m5": step_m5, "m6": step_m6,
         "m7": step_m7, "viewer": step_viewer}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for s in (list(STEPS) if a.step == "all" else [a.step]):
        t0 = time.time()
        STEPS[s]()
        print(f"[{s} done in {time.time() - t0:.0f}s]", flush=True)
