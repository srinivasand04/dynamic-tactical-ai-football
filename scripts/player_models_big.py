"""own_PlayerModel at scale: the player models of player_models.py on three more datasets, with the questions a club
would ask before trusting a per-player model.

B. StatsBomb 360 tournaments (Euro 2020, World Cup 2022, Euro 2024; 166 matches) + Leverkusen 2023/24 (34 matches).
   Same on-ball choice model and the same 360 situation features as player_models.py.
     sb_eval      5 folds by match: role vs player model for every player with 3+ matches and 150+ decisions
     sb_curve     learning curve: Leverkusen players, test = the last 8 matches, player model from his first n matches
     sb_transfer  does style carry across contexts? player model from one dataset (club or a tournament), tested on
                  another (e.g. Leverkusen 23/24 -> Euro 2024); style vectors compared
     sb_pressure  pressure profile: how each player changes his choices when pressed (same situations, pressure on/off),
                  beyond his role; split-half reliability
A. Wyscout public events (Pappalardo et al. 2019, CC BY 4.0): 5 leagues 2017/18 + World Cup 2018 + Euro 2016.
   No freeze frames: the situation is location + context (an opponent action in the last 2 s, the previous action was
   the opponent's, tempo). Roles from Wyscout's role + where he plays.
     wy_eval      season split: first ~3/4 of each league's match days train, last ~1/4 test
     wy_curve     learning curve: player model from his first n matches (1 .. 20), same test matches
     wy_stability style vectors from the first vs second half of the season, against odd vs even matches (the noise
                  ceiling): does style drift?
     wy_similar   the 5 players who choose most alike (same role, all leagues); how often they share a team / league
     wy_league    league style per role (correction per league x role over the shared model)
     wy_national  club season 2017/18 -> World Cup 2018 (and -> Euro 2016)
C. PFF FC World Cup 2022 tracking (64 matches, broadcast-derived): off-ball movement.
     pff_gnn      own_MovementGNN fine-tuned on World Cup matches, 4 folds of 16 held-out matches (starting from a
                  Bundesliga fold model)
     pff          movement rows (visible players only), GNN: Bundesliga model vs World Cup-trained; player model vs role
                  (leave one match out, players with 3+ matches); learning curve (players with 6+ matches)

Outputs: outputs/player_models_big/  (csv / json per step, summary_<step>.txt)
Usage  : .venv-analysis\\Scripts\\python.exe scripts\\player_models_big.py <step> | B | A | C
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import player_models as P  # noqa: E402
import player_of_match as PM  # noqa: E402
import player_profile as PP  # noqa: E402

ROOT = PM.ROOT
OUT = os.path.join(ROOT, "outputs", "player_models_big")
SBT = os.path.join(ROOT, "data", "external", "statsbomb_360_tournaments")
WY = os.path.join(ROOT, "data", "external", "wyscout_public")
TOURN = {(55, 43): "Euro 2020", (43, 106): "World Cup 2022", (55, 282): "Euro 2024"}
LEV = "Leverkusen 23/24"
WY_COMP = {"England": "Premier League", "France": "Ligue 1", "Germany": "Bundesliga", "Italy": "Serie A",
           "Spain": "La Liga", "World_Cup": "World Cup 2018", "European_Championship": "Euro 2016"}
LEAGUES = ["Premier League", "Ligue 1", "Bundesliga", "Serie A", "La Liga"]
K = len(P.CLASSES)
L, W = 105.0, 68.0
RNG = np.random.default_rng(1)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def save_log(step):
    open(os.path.join(OUT, f"summary_{step}.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    LOG.clear()


# ------------------------------------------------------------------ shared tools

def fit_player(Zt, yt, off, groups, lams=P.LAMS):
    """Player correction (Z -> logits) on top of 'off', strength chosen inside his own training matches."""
    def fp(a, b, lam):
        Bp = P.fit_mn(Zt[a], yt[a], K, lam, offset=off[a])
        return P.nll_rows(P.softmax(off[b] + Zt[b] @ Bp), yt[b])
    lam = P.pick_lam(groups, fp, lams, folds=min(5, len(np.unique(groups))))
    return P.fit_mn(Zt, yt, K, lam, offset=off), lam


def per_match(rows_match, nll_a, nll_b, extra):
    """Per test match: decisions, mean loss of two models."""
    out = []
    for m in np.unique(rows_match):
        k = rows_match == m
        out.append({**extra, "match": m, "n": int(k.sum()), "nll_role": float(nll_a[k].mean()),
                    "nll_player": float(nll_b[k].mean())})
    return out


def summarise(R, key, label):
    """Per player: gain (role - player log-loss) per match, bootstrap 90% interval over his matches, verdict."""
    rows = []
    for pid, g in R.groupby(key):
        w = g["n"].to_numpy()
        gain = (g["nll_role"] - g["nll_player"]).to_numpy()
        lo, hi = P.boot_ci(gain * w / w.mean()) if len(g) > 1 else (gain[0], gain[0])
        first = g.iloc[0]
        rows.append({key: pid, **{c: first[c] for c in label if c in g}, "matches": len(g), "decisions": int(w.sum()),
                     "logloss_role": np.average(g["nll_role"], weights=w),
                     "logloss_player": np.average(g["nll_player"], weights=w),
                     "gain_ci90_lo": lo, "gain_ci90_hi": hi, "matches_better": int((gain > 0).sum())})
    E = pd.DataFrame(rows)
    E["gain_pct"] = 100 * (E["logloss_role"] - E["logloss_player"]) / E["logloss_role"]
    E["verdict"] = np.where(E["gain_ci90_lo"] > 0, "player better", np.where(E["gain_ci90_hi"] < 0, "role better", "unclear"))
    return E.sort_values("gain_pct", ascending=False)


def report_eval(E, R, by=None):
    w = R["n"]
    say(f"  pooled held-out log-loss: role {np.average(R['nll_role'], weights=w):.4f}  player "
        f"{np.average(R['nll_player'], weights=w):.4f}  ({100 * (1 - np.average(R['nll_player'], weights=w) / np.average(R['nll_role'], weights=w)):+.2f}% )")
    say(f"  players: {len(E)}  player model better {int((E.verdict == 'player better').sum())}  unclear "
        f"{int((E.verdict == 'unclear').sum())}  role better {int((E.verdict == 'role better').sum())}  "
        f"| gain > 0 on average: {100 * (E.gain_pct > 0).mean():.0f}%  median gain {E.gain_pct.median():+.2f}%")
    if by:
        for k, g in E.groupby(by):
            say(f"    {k:22s} players {len(g):4d}  better {int((g.verdict == 'player better').sum()):3d}  "
                f"median gain {g.gain_pct.median():+.2f}%  mean {g.gain_pct.mean():+.2f}%")


def style_vec(off, Zi, Bp):
    """log(player share / role share) per choice, on his own situations."""
    pr, pp = P.softmax(off).mean(0), P.softmax(off + Zi @ Bp).mean(0)
    return np.log(np.clip(pp, 1e-4, 1) / np.clip(pr, 1e-4, 1)), pp, pr


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan


# ------------------------------------------------------------------ B. StatsBomb 360 tournaments + Leverkusen

def ff_feats(ff, me):
    opp = np.array([PM.to_m(p["location"]) for p in ff if not p["teammate"]]).reshape(-1, 2)
    mates = np.array([PM.to_m(p["location"]) for p in ff if p["teammate"] and not p.get("actor")]).reshape(-1, 2)
    d = np.linalg.norm(opp - me, axis=1) if len(opp) else np.array([30.0])
    gs = ((opp[:, 0] > me[0]) & (np.abs(opp[:, 1] - me[1]) < 15)).sum() if len(opp) else 0
    return min(d.min(), 15.0), int((d < 5).sum()), int(gs), int((mates[:, 0] > me[0] + 5).sum()) if len(mates) else 0


def sb_match(mid, src, date):
    ev = json.load(open(os.path.join(SBT, "events", f"{mid}.json"), encoding="utf-8"))
    tp = os.path.join(SBT, "three-sixty", f"{mid}.json")
    if not os.path.exists(tp):
        return [], []
    ffs = {f["event_uuid"]: f["freeze_frame"] for f in json.load(open(tp, encoding="utf-8"))}
    lu = PP.lineups(ev, mid)
    role = {r["player_id"]: r["role"] for r in lu}
    rows = []
    for e in ev:
        kind = e["type"]["name"]
        if kind not in ("Pass", "Carry", "Shot", "Dribble") or "location" not in e or "player" not in e:
            continue
        sub = e.get(kind.lower(), {}) or {}
        st = (sub.get("type") or {}).get("name", "")
        if (kind == "Pass" and st in PM.SET_PIECE) or (kind == "Shot" and st != "Open Play"):
            continue
        pid = e["player"]["id"]
        if role.get(pid, "GK") == "GK" or e["id"] not in ffs:
            continue
        me = PM.to_m(e["location"])
        end = PM.to_m(sub["end_location"]) if "end_location" in sub else np.array([np.nan, np.nan])
        dist = float(np.hypot(*(end - me)))
        if kind == "Carry" and not dist >= 5:
            continue
        ch = (0 if kind == "Shot" else 1 if kind == "Dribble" else 2 if kind == "Carry" else
              3 if dist >= 30 else 4 if end[0] - me[0] >= 5 else 5)
        no, o5, gs, ma = ff_feats(ffs[e["id"]], me)
        rows.append({"source": src, "date": date, "match": mid, "player_id": pid, "player": e["player"]["name"],
                     "team": e["team"]["name"], "role": role[pid], "x": me[0], "y": me[1],
                     "pressed": int(bool(e.get("under_pressure"))), "choice": ch, "near_opp": no, "opp5": o5,
                     "goal_side": gs, "mates_ahead": ma})
    return rows, [dict(r, source=src) for r in lu]


def sb_data():
    path = os.path.join(OUT, "sb_decisions.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path), pd.read_csv(os.path.join(OUT, "sb_minutes.csv"))
    rows, mins = [], []
    for (c, s), src in TOURN.items():
        ms = json.load(open(os.path.join(SBT, f"matches_{c}_{s}.json"), encoding="utf-8"))
        for i, m in enumerate(ms, 1):
            r, lu = sb_match(m["match_id"], src, m["match_date"])
            rows += r
            mins += lu
            print(f"  {src} {i}/{len(ms)}: {len(r)} decisions", flush=True)
    D = pd.DataFrame(rows)
    lev = pd.read_csv(os.path.join(P.OUT, "onball_decisions.csv.gz"))
    dates = {m["match_id"]: m["match_date"] for m in json.load(open(os.path.join(PM.SB, "matches_9_281.json"), encoding="utf-8"))}
    lev["source"], lev["date"] = LEV, lev["match"].map(dates)
    D = pd.concat([lev[D.columns], D], ignore_index=True)
    M = pd.concat([pd.read_csv(os.path.join(P.PROF, "roles.csv")).assign(source=LEV), pd.DataFrame(mins)], ignore_index=True)
    D.to_csv(path, index=False)
    M.to_csv(os.path.join(OUT, "sb_minutes.csv"), index=False)
    return D, M


def sb_eval(D, X, Z):
    y = D["choice"].to_numpy()
    matches = D["match"].unique()
    fold_of = {m: i % 5 for i, m in enumerate(RNG.permutation(matches))}
    fold = D["match"].map(fold_of).to_numpy()
    pm = D.groupby(["player_id", "match"]).size()
    nm = (pm >= 10).groupby(level=0).sum()
    elig = [p for p in nm[nm >= 3].index if pm[p].sum() >= 150]
    say(f"B eval: {len(D)} decisions, {D['match'].nunique()} matches, {D['player_id'].nunique()} players; "
        f"{len(elig)} players with 3+ matches and 150+ decisions")
    for s, g in D.groupby("source"):
        say(f"  {s:18s} {g['match'].nunique():3d} matches {len(g):7d} decisions {g['player_id'].nunique():4d} players")
    pid_arr, m_arr = D["player_id"].to_numpy(), D["match"].to_numpy()
    src_of = D.groupby("player_id")["source"].agg(lambda s: s.value_counts().index[0])
    res = []
    for f in range(5):
        t0 = time.time()
        tr = fold != f
        base = P.OnballBase(D, X, Z, tr)
        for pid in elig:
            te_i = np.flatnonzero((fold == f) & (pid_arr == pid))
            tr_i = np.flatnonzero(tr & (pid_arr == pid))
            if len(te_i) < 5 or len(tr_i) < 50:
                continue
            off_tr, off_te = base.logits(D, X, Z, tr_i), base.logits(D, X, Z, te_i)
            Bp, lam = fit_player(Z[tr_i], y[tr_i], off_tr, m_arr[tr_i])
            res += per_match(m_arr[te_i], P.nll_rows(P.softmax(off_te), y[te_i]),
                             P.nll_rows(P.softmax(off_te + Z[te_i] @ Bp), y[te_i]),
                             {"player_id": pid, "player": D["player"].to_numpy()[te_i[0]], "main_source": src_of[pid],
                              "role": D["role"].to_numpy()[te_i[0]], "lam": lam})
        print(f"  fold {f + 1}/5 ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "sb_eval_matches.csv"), index=False)
    E = summarise(R, "player_id", ["player", "main_source", "role"])
    E.to_csv(os.path.join(OUT, "sb_eval.csv"), index=False)
    say("B1. Player model vs role model, held-out matches (5 folds by match)")
    report_eval(E, R, by="main_source")
    for k, g in E.groupby("role"):
        say(f"    role {k:3s} players {len(g):4d}  better {int((g.verdict == 'player better').sum()):3d}  median gain {g.gain_pct.median():+.2f}%")
    say("  top 8:", "; ".join(f"{r.player} ({r.role}, {r.main_source}) {r.gain_pct:+.1f}%" for r in E.head(8).itertuples()))


def learning_curve(D, X, Z, test_mask_of_player, train_order_of_player, players, ns, tag):
    """Generic learning curve: base fitted without any player's test rows; the player model from his first n training
    matches; scored on his fixed test matches."""
    y = D["choice"].to_numpy()
    pid_arr, m_arr = D["player_id"].to_numpy(), D["match"].to_numpy()
    test = np.zeros(len(D), bool)
    for p in players:
        test |= test_mask_of_player(p)
    base = P.OnballBase(D, X, Z, ~test)
    rows = []
    for j, p in enumerate(players):
        te = np.flatnonzero(test_mask_of_player(p))
        off_te = base.logits(D, X, Z, te)
        nll_r = P.nll_rows(P.softmax(off_te), y[te]).mean()
        order = train_order_of_player(p)
        for n in ns:
            if n > len(order):
                continue
            tr = np.flatnonzero(np.isin(m_arr, order[:n]) & (pid_arr == p))
            if len(tr) < 5:
                continue
            Bp, lam = fit_player(Z[tr], y[tr], base.logits(D, X, Z, tr), m_arr[tr])
            nll_p = P.nll_rows(P.softmax(off_te + Z[te] @ Bp), y[te]).mean()
            rows.append({"player_id": p, "n_matches": n, "train_decisions": len(tr), "test_decisions": len(te),
                         "lam": lam, "nll_role": nll_r, "nll_player": nll_p, "gain_pct": 100 * (nll_r - nll_p) / nll_r})
        if (j + 1) % 50 == 0:
            print(f"  {tag} curve {j + 1}/{len(players)}", flush=True)
    C = pd.DataFrame(rows)
    say(f"  matches  players  mean gain %  (90% CI)        share of players with gain > 0")
    for n, g in C.groupby("n_matches"):
        lo, hi = P.boot_ci(g["gain_pct"])
        say(f"  {n:7d}  {len(g):7d}  {g.gain_pct.mean():+8.2f}    ({lo:+.2f} .. {hi:+.2f})   {100 * (g.gain_pct > 0).mean():5.0f}%")
    return C


def sb_curve(D, X, Z):
    lev = D[D["source"] == LEV]
    order = lev.drop_duplicates("match").sort_values("date")["match"].tolist()
    test_m = order[-8:]
    pm = lev[lev["team"] == PM.CLUB].groupby("player_id")["match"].nunique()
    players = [p for p in pm.index if lev[(lev.player_id == p) & lev.match.isin(order[:-8])]["match"].nunique() >= 16
               and lev[(lev.player_id == p) & lev.match.isin(test_m)].shape[0] >= 50]
    say(f"B2. Learning curve, Leverkusen players (360 features): {len(players)} players, test = last 8 matches")
    pid_arr, m_arr = D["player_id"].to_numpy(), D["match"].to_numpy()
    played = {p: [m for m in order[:-8] if ((pid_arr == p) & (m_arr == m)).sum() >= 10] for p in players}
    C = learning_curve(D, X, Z, lambda p: (pid_arr == p) & np.isin(m_arr, test_m), lambda p: played[p], players,
                       [1, 2, 4, 8, 16, 26], "B")
    C.to_csv(os.path.join(OUT, "sb_curve.csv"), index=False)


def sb_transfer(D, X, Z):
    """Player model learned in one dataset, tested in another."""
    y = D["choice"].to_numpy()
    pid_arr, m_arr, src_arr = D["player_id"].to_numpy(), D["match"].to_numpy(), D["source"].to_numpy()
    cnt = D.groupby(["player_id", "source"]).size().unstack(fill_value=0)
    rows = []
    for test_src in cnt.columns:
        base = P.OnballBase(D, X, Z, src_arr != test_src)
        for pid in cnt.index[cnt[test_src] >= 50]:
            te = np.flatnonzero((pid_arr == pid) & (src_arr == test_src))
            off_te = base.logits(D, X, Z, te)
            # in-context model (fitted on the test dataset itself) only for the style comparison, never scored
            Bin, _ = fit_player(Z[te], y[te], off_te, m_arr[te])
            v_in, _, _ = style_vec(off_te, Z[te], Bin)
            for train_src in cnt.columns:
                if train_src == test_src or cnt.loc[pid, train_src] < 150:
                    continue
                tr = np.flatnonzero((pid_arr == pid) & (src_arr == train_src))
                Bp, lam = fit_player(Z[tr], y[tr], base.logits(D, X, Z, tr), m_arr[tr])
                nr = P.nll_rows(P.softmax(off_te), y[te]).mean()
                npl = P.nll_rows(P.softmax(off_te + Z[te] @ Bp), y[te]).mean()
                v_tr, _, _ = style_vec(off_te, Z[te], Bp)
                rows.append({"player_id": pid, "player": D["player"].to_numpy()[te[0]], "team_test": D["team"].to_numpy()[te[0]],
                             "team_train": D["team"].to_numpy()[tr[0]], "train": train_src, "test": test_src,
                             "train_decisions": len(tr), "test_decisions": len(te), "nll_role": nr, "nll_player": npl,
                             "gain_pct": 100 * (nr - npl) / nr,
                             **{f"style_train_{c}": float(v) for c, v in zip(P.CLASSES, v_tr)},
                             **{f"style_test_{c}": float(v) for c, v in zip(P.CLASSES, v_in)}})
        print(f"  transfer -> {test_src}: {len(rows)} pairs so far", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "sb_transfer.csv"), index=False)
    say("B3. Style learned in one context, tested in another (gain of player model over role model on the new context)")
    say("  train -> test                          players  mean gain %  (90% CI)       gain > 0")
    for (a, b), g in T.groupby(["train", "test"]):
        if len(g) < 5:
            continue
        lo, hi = P.boot_ci(g["gain_pct"])
        say(f"  {a:18s} -> {b:18s} {len(g):5d}  {g.gain_pct.mean():+8.2f}   ({lo:+.2f} .. {hi:+.2f})   {100 * (g.gain_pct > 0).mean():4.0f}%")
    lo, hi = P.boot_ci(T["gain_pct"])
    say(f"  all pairs: {len(T)}  mean gain {T.gain_pct.mean():+.2f}% ({lo:+.2f} .. {hi:+.2f}), gain > 0 in {100 * (T.gain_pct > 0).mean():.0f}%")
    say("  style carries? correlation of style (log ratio vs role) learned in context 1 vs context 2, per choice:")
    say("   " + "  ".join(f"{c}: r={corr(T[f'style_train_{c}'], T[f'style_test_{c}']):.2f}" for c in P.CLASSES))
    lv = T[(T["train"] == LEV) & (T["team_train"] == PM.CLUB)]
    if len(lv):
        say("  Leverkusen players, club 23/24 -> tournament:")
        for r in lv.sort_values("gain_pct", ascending=False).itertuples():
            say(f"    {r.player[:28]:28s} -> {r.test:15s} ({r.test_decisions:4d} decisions)  gain {r.gain_pct:+.2f}%")


def sb_pressure(D, X, Z):
    """Pressure response = how the player's predicted choice shares change when pressure is switched on, on his own
    situations, minus the same change for his role. Fitted on all his matches, and on odd / even matches (reliability)."""
    y = D["choice"].to_numpy()
    pid_arr, m_arr = D["player_id"].to_numpy(), D["match"].to_numpy()
    X1, Z1 = P.onball_X(D.assign(pressed=1))
    X0, Z0 = P.onball_X(D.assign(pressed=0))
    base = P.OnballBase(D, X, Z, np.ones(len(D), bool))
    cnt = D.groupby("player_id").size()
    players = cnt.index[cnt >= 400]
    rows = []
    for j, pid in enumerate(players):
        i = np.flatnonzero(pid_arr == pid)
        main = D["role"].to_numpy()[i]
        main = pd.Series(main).mode()[0]
        Dm = D.iloc[i].assign(role=main)
        ii = np.arange(len(i))
        off, off1, off0 = (base.logits(Dm, A[i], B[i], ii) for A, B in ((X, Z), (X1, Z1), (X0, Z0)))
        out = {"player_id": pid, "player": Dm["player"].iloc[0], "team": Dm["team"].iloc[0], "role": main,
               "decisions": len(i), "pressed_share": float(Dm["pressed"].mean())}
        ms = np.unique(m_arr[i])
        halves = {"all": np.ones(len(i), bool), "odd": np.isin(m_arr[i], ms[::2]), "even": np.isin(m_arr[i], ms[1::2])}
        for h, k in halves.items():
            if k.sum() < 100:
                continue
            Bp, _ = fit_player(Z[i][k], y[i][k], off[k], m_arr[i][k])
            d_pl = P.softmax(off1 + Z1[i] @ Bp).mean(0) - P.softmax(off0 + Z0[i] @ Bp).mean(0)
            d_ro = P.softmax(off1).mean(0) - P.softmax(off0).mean(0)
            for c, v in zip(P.CLASSES, d_pl - d_ro):
                out[f"{h}_{c}"] = float(v)
        rows.append(out)
        if (j + 1) % 100 == 0:
            print(f"  pressure {j + 1}/{len(players)}", flush=True)
    Q = pd.DataFrame(rows)
    Q.to_csv(os.path.join(OUT, "sb_pressure.csv"), index=False)
    say(f"B4. Pressure profile: {len(Q)} players with 400+ decisions (change in choice share when pressed, beyond his role)")
    say("  split-half reliability (odd vs even matches) per choice: " +
        "  ".join(f"{c}: r={corr(Q[f'odd_{c}'], Q[f'even_{c}']):.2f}" for c in P.CLASSES))
    for c, lab in (("safe pass", "goes safe when pressed (more sideways / back passes than his role)"),
                   ("dribble", "takes players on when pressed")):
        top = Q.sort_values(f"all_{c}", ascending=False).head(5)
        say(f"  most {lab}: " + "; ".join(f"{r.player} ({r.role}) {100 * r[f'all_{c}']:+.1f} pts" for _, r in top.iterrows()))
    low = Q.sort_values("all_safe pass").head(5)
    say("  least (keeps playing forward when pressed): " + "; ".join(f"{r.player} ({r.role}) {100 * r['all_safe pass']:+.1f} pts"
                                                               for _, r in low.iterrows()))


def run_B(steps):
    D, M = sb_data()
    X, Z = P.onball_X(D)
    for s in steps:
        {"sb_eval": sb_eval, "sb_curve": sb_curve, "sb_transfer": sb_transfer, "sb_pressure": sb_pressure}[s](D, X, Z)
        save_log(s)


# ------------------------------------------------------------------ A. Wyscout public events

def wy_minutes(matches):
    rows = []
    for m in matches:
        for tid, td in m["teamsData"].items():
            f = td.get("formation") or {}
            subs = f.get("substitutions") or []
            if subs == "null":
                subs = []
            off = {s["playerOut"]: s["minute"] for s in subs}
            on = {s["playerIn"]: s["minute"] for s in subs}
            for p in f.get("lineup") or []:
                rows.append((m["wyId"], p["playerId"], int(tid), min(off.get(p["playerId"], 90), 90)))
            for p, t in on.items():
                rows.append((m["wyId"], p, int(tid), max(min(off.get(p, 90), 90) - t, 1)))
    return pd.DataFrame(rows, columns=["match", "player_id", "team_id", "minutes"])


def wy_data():
    path = os.path.join(OUT, "wy_decisions.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path)
    players = {p["wyId"]: p for p in json.load(open(os.path.join(WY, "players.json"), encoding="utf-8"))}
    teams = {t["wyId"]: t["name"] for t in json.load(open(os.path.join(WY, "teams.json"), encoding="utf-8"))}
    parts, mins = [], []
    per_order = {"1H": 0, "2H": 1, "E1": 2, "E2": 3, "P": 4}
    for key, comp in WY_COMP.items():
        t0 = time.time()
        ms = json.load(open(os.path.join(WY, "matches", f"matches_{key}.json"), encoding="utf-8"))
        date = {m["wyId"]: m["dateutc"][:10] for m in ms}
        mins.append(wy_minutes(ms).assign(comp=comp))
        ev = json.load(open(os.path.join(WY, "events", f"events_{key}.json"), encoding="utf-8"))
        cols = {"match": [], "team_id": [], "player_id": [], "ev": [], "sub": [], "per": [], "t": [], "x0": [], "y0": [],
                "x1": [], "y1": [], "takeon": []}
        for e in ev:
            pos = e.get("positions") or [{"x": np.nan, "y": np.nan}]
            p1 = pos[1] if len(pos) > 1 else {"x": np.nan, "y": np.nan}
            tg = {t["id"] for t in e.get("tags") or []}
            cols["match"].append(e["matchId"]); cols["team_id"].append(e["teamId"]); cols["player_id"].append(e["playerId"])
            cols["ev"].append(e["eventName"]); cols["sub"].append(e["subEventName"]); cols["per"].append(per_order.get(e["matchPeriod"], 5))
            cols["t"].append(e["eventSec"]); cols["x0"].append(pos[0]["x"]); cols["y0"].append(pos[0]["y"])
            cols["x1"].append(p1["x"]); cols["y1"].append(p1["y"]); cols["takeon"].append(int(bool(tg & {503, 504})))
        del ev
        E = pd.DataFrame(cols)
        del cols
        # context from the event stream (order as in the file, per match and period)
        same = (E["match"] == E["match"].shift(1)) & (E["per"] == E["per"].shift(1))
        same2 = (E["match"] == E["match"].shift(2)) & (E["per"] == E["per"].shift(2))
        dt1 = (E["t"] - E["t"].shift(1)).where(same)
        dt2 = (E["t"] - E["t"].shift(2)).where(same2)
        opp1 = same & (E["team_id"] != E["team_id"].shift(1))
        opp2 = same2 & (E["team_id"] != E["team_id"].shift(2))
        E["pressed"] = ((opp1 & (dt1 < 2)) | (opp2 & (dt2 < 2))).astype(int)
        E["won"] = opp1.astype(int)
        E["tempo"] = dt1.fillna(10).clip(0, 10)
        x, yy = E["x0"] * 1.05, E["y0"] * 0.68
        ex, ey = E["x1"] * 1.05, E["y1"] * 0.68
        dist = np.hypot(ex - x, ey - yy)
        isp = (E["ev"] == "Pass") & (E["sub"] != "Hand pass")
        sel = ((E["ev"] == "Shot") | ((E["sub"] == "Ground attacking duel") & (E["takeon"] == 1)) |
               ((E["sub"] == "Acceleration") & (dist >= 5)) | isp)
        ch = np.select([E["ev"] == "Shot", E["sub"] == "Ground attacking duel", E["sub"] == "Acceleration",
                        isp & (dist >= 30), isp & (ex - x >= 5)], [0, 1, 2, 3, 4], default=5)
        A = pd.DataFrame({"source": comp, "match": E["match"], "date": E["match"].map(date), "team_id": E["team_id"],
                          "player_id": E["player_id"], "x": x, "y": yy, "pressed": E["pressed"], "won": E["won"],
                          "tempo": E["tempo"], "choice": ch})[sel.to_numpy() & x.notna().to_numpy() & (E["player_id"] > 0).to_numpy()]
        parts.append(A)
        say(f"  {comp}: {E['match'].nunique()} matches, {len(E)} events -> {len(A)} decisions ({time.time() - t0:.0f}s)")
        del E
    D = pd.concat(parts, ignore_index=True)
    D["team"] = D["team_id"].map(teams)
    D["player"] = D["player_id"].map(lambda p: players.get(p, {}).get("shortName", str(p)))
    pos = D["player_id"].map(lambda p: (players.get(p, {}).get("role") or {}).get("code2", ""))
    # role: Wyscout's role + where he plays in that competition (mean location of his decisions)
    # (Wyscout turns every action to the acting team's direction, so a left-sided player keeps one side; thresholds from
    #  the spread of mean positions per Wyscout role: centre-backs ~10 m off the middle, full-backs ~20-27 m)
    g = D.groupby(["player_id", "source"]).agg(mx=("x", "mean"), my=("y", "mean"))
    D = D.join(g, on=["player_id", "source"])
    side = (D["my"] - W / 2).abs()
    wide = np.where(pos == "DF", side > 15, np.where(pos == "MD", side > 13, side > 12))
    D["role"] = np.select([pos == "GK", (pos == "DF") & wide, pos == "DF", (pos == "MD") & wide, (pos == "MD") & (D["mx"] < 52),
                           (pos == "MD") & (D["mx"] > 63), pos == "MD", (pos == "FW") & wide, pos == "FW"],
                          ["GK", "FB", "CB", "W", "DM", "AM", "CM", "W", "ST"], default="CM")
    D = D[D["role"] != "GK"].drop(columns=["mx", "my"])
    Mn = pd.concat(mins, ignore_index=True)
    Mn.to_csv(os.path.join(OUT, "wy_minutes.csv"), index=False)
    D.to_csv(path, index=False)
    return D


def wy_X(D):
    xn = D["x"].to_numpy() / L
    yc = np.abs(D["y"].to_numpy() - W / 2) / (W / 2)
    dg = np.hypot(L - D["x"].to_numpy(), W / 2 - D["y"].to_numpy()) / 50
    ang = np.abs(np.arctan2(W / 2 - D["y"].to_numpy(), L - D["x"].to_numpy()))
    box = ((D["x"] > 88.5) & (np.abs(D["y"] - W / 2) < 20.16)).to_numpy(float)
    pr, won, tp = D["pressed"].to_numpy(float), D["won"].to_numpy(float), D["tempo"].to_numpy() / 10
    X = np.column_stack([np.ones(len(D)), xn, xn ** 2, yc, dg, ang, box, pr, won, tp, pr * xn, box * pr, won * xn])
    Z = np.column_stack([np.ones(len(D)), xn - 0.5, pr])
    return X, Z


def wy_split(D):
    """Per league: the first ~3/4 of its match dates train, the rest test."""
    tr = np.zeros(len(D), bool)
    for lg in LEAGUES:
        k = (D["source"] == lg).to_numpy()
        days = np.sort(D.loc[k, "date"].unique())
        cut = days[int(len(days) * 0.74)]
        tr |= k & (D["date"] < cut).to_numpy()
    te = D["source"].isin(LEAGUES).to_numpy() & ~tr
    return tr, te


def wy_eval(D, X, Z):
    y = D["choice"].to_numpy()
    pid_arr, m_arr = D["player_id"].to_numpy(), D["match"].to_numpy()
    tr, te = wy_split(D)
    say(f"A data: {len(D)} decisions, {D['match'].nunique()} matches, {D['player_id'].nunique()} players")
    for s, g in D.groupby("source"):
        say(f"  {s:16s} {g['match'].nunique():4d} matches {len(g):8d} decisions {g['player_id'].nunique():5d} players")
    t0 = time.time()
    base = P.OnballBase(D, X, Z, tr)
    say(f"  base model fitted on {tr.sum()} decisions ({time.time() - t0:.0f}s)")
    pm_tr = D[tr].groupby(["player_id", "match"]).size()
    pm_te = D[te].groupby(["player_id", "match"]).size()
    ntr = (pm_tr >= 15).groupby(level=0).sum()
    nte = (pm_te >= 15).groupby(level=0).sum()
    elig = [p for p in ntr[ntr >= 10].index if nte.get(p, 0) >= 3]
    say(f"A1. Season split (train first ~3/4 of match days, test the rest): {len(elig)} players with 10+ train and 3+ test matches")
    res = []
    for j, pid in enumerate(elig):
        tr_i = np.flatnonzero(tr & (pid_arr == pid))
        te_i = np.flatnonzero(te & (pid_arr == pid))
        off_tr, off_te = base.logits(D, X, Z, tr_i), base.logits(D, X, Z, te_i)
        Bp, lam = fit_player(Z[tr_i], y[tr_i], off_tr, m_arr[tr_i])
        res += per_match(m_arr[te_i], P.nll_rows(P.softmax(off_te), y[te_i]), P.nll_rows(P.softmax(off_te + Z[te_i] @ Bp), y[te_i]),
                         {"player_id": pid, "player": D["player"].to_numpy()[te_i[0]], "team": D["team"].to_numpy()[te_i[0]],
                          "source": D["source"].to_numpy()[te_i[0]], "role": pd.Series(D["role"].to_numpy()[tr_i]).mode()[0], "lam": lam})
        if (j + 1) % 200 == 0:
            print(f"  eval {j + 1}/{len(elig)} ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "wy_eval_matches.csv"), index=False)
    E = summarise(R, "player_id", ["player", "team", "source", "role"])
    E.to_csv(os.path.join(OUT, "wy_eval.csv"), index=False)
    report_eval(E, R, by="source")
    for k, g in E.groupby("role"):
        say(f"    role {k:3s} players {len(g):4d}  better {int((g.verdict == 'player better').sum()):4d}  median gain {g.gain_pct.median():+.2f}%")
    say("  lambda chosen (how strongly the player model is pulled to his role): " +
        ", ".join(f"{k:g}: {v}" for k, v in E.merge(R.drop_duplicates('player_id')[['player_id', 'lam']], on='player_id')['lam'].value_counts().sort_index().items()))
    say("  top 10:", "; ".join(f"{r.player} ({r.role}, {r.team}) {r.gain_pct:+.1f}%" for r in E.head(10).itertuples()))


def wy_curve(D, X, Z):
    tr, te = wy_split(D)
    pid_arr, m_arr = D["player_id"].to_numpy(), D["match"].to_numpy()
    pm_tr = D[tr].groupby(["player_id", "match"]).size()
    pm_te = D[te].groupby(["player_id", "match"]).size()
    ntr = (pm_tr >= 15).groupby(level=0).sum()
    nte = (pm_te >= 15).groupby(level=0).sum()
    elig = [p for p in ntr[ntr >= 20].index if nte.get(p, 0) >= 5]
    players = list(RNG.choice(elig, min(400, len(elig)), replace=False))
    dates = D.drop_duplicates("match").set_index("match")["date"]
    order = {}
    for p in players:
        ms = pm_tr.loc[p]
        ms = ms[ms >= 15].index
        order[p] = sorted(ms, key=lambda m: dates[m])
    say(f"A2. Learning curve (Wyscout): {len(players)} players (of {len(elig)} with 20+ train and 5+ test matches)")
    C = learning_curve(D, X, Z, lambda p: te & (pid_arr == p), lambda p: order[p], players, [1, 2, 3, 5, 8, 12, 20], "A")
    C.to_csv(os.path.join(OUT, "wy_curve.csv"), index=False)


def season_fits(D, X, Z, base, players, split_fn):
    """Style vectors for each player and each part returned by split_fn(match list sorted by date) -> {name: matches}."""
    y = D["choice"].to_numpy()
    pid_arr, m_arr = D["player_id"].to_numpy(), D["match"].to_numpy()
    dates = D.drop_duplicates("match").set_index("match")["date"]
    rows = []
    for j, p in enumerate(players):
        i = np.flatnonzero(pid_arr == p)
        role = pd.Series(D["role"].to_numpy()[i]).mode()[0]
        Dm = D.iloc[i].assign(role=role)
        off = base.logits(Dm, X[i], Z[i], np.arange(len(i)))
        ms = sorted(np.unique(m_arr[i]), key=lambda m: dates[m])
        out = {"player_id": p, "player": Dm["player"].iloc[0], "team": Dm["team"].iloc[0], "source": Dm["source"].iloc[0],
               "role": role, "decisions": len(i), "matches": len(ms)}
        for name, part in split_fn(ms).items():
            k = np.isin(m_arr[i], part)
            Bp, _ = fit_player(Z[i][k], y[i][k], off[k], m_arr[i][k])
            v, pp, pr = style_vec(off, Z[i], Bp)                          # on ALL his situations: only the model differs
            for c, a in zip(P.CLASSES, v):
                out[f"{name}_{c}"] = float(a)
            if name == "all":
                for c, a, b in zip(P.CLASSES, pp, pr):
                    out[f"share_{c}"], out[f"role_share_{c}"] = float(a), float(b)
        rows.append(out)
        if (j + 1) % 200 == 0:
            print(f"  style fits {j + 1}/{len(players)}", flush=True)
    return pd.DataFrame(rows)


def wy_style_table(D, X, Z):
    path = os.path.join(OUT, "wy_style.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    lg = D["source"].isin(LEAGUES).to_numpy()
    base = P.OnballBase(D, X, Z, lg)
    Dl = D[lg]
    pm = Dl.groupby(["player_id", "match"]).size()
    nm = (pm >= 15).groupby(level=0).sum()
    players = nm[nm >= 20].index.tolist()
    split = lambda ms: {"all": ms, "first": ms[:len(ms) // 2], "second": ms[len(ms) // 2:], "odd": ms[::2], "even": ms[1::2]}
    S = season_fits(Dl.reset_index(drop=True), X[lg], Z[lg], base, players, split)
    S.to_csv(path, index=False)
    return S


def wy_stability(D, X, Z):
    S = wy_style_table(D, X, Z)
    say(f"A3. Style stability, {len(S)} league players with 20+ matches: correlation of style vectors (log ratio vs role)")
    say("  choice          first vs second half of season   odd vs even matches (noise ceiling)")
    for c in P.CLASSES:
        say(f"  {c:14s}  r = {corr(S[f'first_{c}'], S[f'second_{c}']):.2f}                         r = {corr(S[f'odd_{c}'], S[f'even_{c}']):.2f}")


def wy_similar(D, X, Z):
    S = wy_style_table(D, X, Z)
    cols = [f"all_{c}" for c in P.CLASSES]
    V = S[cols].to_numpy()
    V = (V - V.mean(0)) / V.std(0)
    rows, same_team, same_lg, chance_team, chance_lg = [], [], [], [], []
    for role, g in S.groupby("role"):
        idx = g.index.to_numpy()
        A = V[idx]
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        sim = A @ A.T
        np.fill_diagonal(sim, -9)
        teams, lgs = g["team"].to_numpy(), g["source"].to_numpy()
        for a in range(len(idx)):
            nn = np.argsort(-sim[a])[:5]
            same_team.append((teams[nn] == teams[a]).mean())
            same_lg.append((lgs[nn] == lgs[a]).mean())
            chance_team.append(((teams == teams[a]).sum() - 1) / (len(idx) - 1))
            chance_lg.append(((lgs == lgs[a]).sum() - 1) / (len(idx) - 1))
            rows.append({"player": g["player"].iloc[a], "team": teams[a], "league": lgs[a], "role": role,
                         **{f"nn{k + 1}": f"{g['player'].iloc[n]} ({teams[n]}) {sim[a, n]:.2f}" for k, n in enumerate(nn)}})
    N = pd.DataFrame(rows)
    N.to_csv(os.path.join(OUT, "wy_similar.csv"), index=False)
    say(f"A4. Similar players: 5 nearest by style (same role, all 5 leagues), {len(N)} players")
    say(f"  share of the 5 nearest from the SAME TEAM  : {100 * np.mean(same_team):.1f}%  (chance {100 * np.mean(chance_team):.1f}%)")
    say(f"  share of the 5 nearest from the SAME LEAGUE: {100 * np.mean(same_lg):.1f}%  (chance {100 * np.mean(chance_lg):.1f}%)")
    top = S.sort_values("decisions", ascending=False).groupby("role").head(1)
    for p in top["player"]:
        r = N[N["player"] == p].iloc[0]
        say(f"  {r.player} ({r.role}, {r.team}): {r.nn1}; {r.nn2}; {r.nn3}")


def wy_league(D, X, Z):
    y = D["choice"].to_numpy()
    lg = D["source"].isin(LEAGUES).to_numpy()
    base = P.OnballBase(D, X, Z, lg)
    say("A5. League style per role: choice share predicted with the league x role correction / without (same situations)")
    say("  role league            " + "  ".join(f"{c[:9]:>9s}" for c in P.CLASSES))
    rows = []
    for role in ["CB", "FB", "DM", "CM", "AM", "W", "ST"]:
        for L_ in LEAGUES:
            i = np.flatnonzero(lg & (D["role"] == role).to_numpy() & (D["source"] == L_).to_numpy())
            if len(i) < 2000:
                continue
            off = base.logits(D, X, Z, i)
            Bl = P.fit_mn(Z[i], y[i], K, 30.0, offset=off)
            ratio = P.softmax(off + Z[i] @ Bl).mean(0) / P.softmax(off).mean(0)
            rows.append({"role": role, "league": L_, **{c: float(r) for c, r in zip(P.CLASSES, ratio)}})
            say(f"  {role:4s} {L_:16s} " + "  ".join(f"{r:9.2f}" for r in ratio))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "wy_league.csv"), index=False)


def wy_national(D, X, Z):
    y = D["choice"].to_numpy()
    pid_arr, m_arr, src = D["player_id"].to_numpy(), D["match"].to_numpy(), D["source"].to_numpy()
    lg = np.isin(src, LEAGUES)
    base = P.OnballBase(D, X, Z, lg)
    cnt_lg = D[lg].groupby(["player_id", "match"]).size()
    nlg = (cnt_lg >= 15).groupby(level=0).sum()
    rows = []
    for nat in ("World Cup 2018", "Euro 2016"):
        nat_m = src == nat
        cnt = D[nat_m].groupby("player_id").size()
        for pid in [p for p in cnt.index[cnt >= 50] if nlg.get(p, 0) >= 15]:
            tr = np.flatnonzero(lg & (pid_arr == pid))
            te = np.flatnonzero(nat_m & (pid_arr == pid))
            role = pd.Series(D["role"].to_numpy()[tr]).mode()[0]
            Dte = D.iloc[te].assign(role=role)
            off_te = base.logits(Dte, X[te], Z[te], np.arange(len(te)))
            Bp, lam = fit_player(Z[tr], y[tr], base.logits(D, X, Z, tr), m_arr[tr])
            nr = P.nll_rows(P.softmax(off_te), y[te]).mean()
            npl = P.nll_rows(P.softmax(off_te + Z[te] @ Bp), y[te]).mean()
            rows.append({"player_id": pid, "player": D["player"].to_numpy()[te[0]], "club": D["team"].to_numpy()[tr[0]],
                         "national": D["team"].to_numpy()[te[0]], "tournament": nat, "role": role,
                         "club_decisions": len(tr), "test_decisions": len(te), "nll_role": nr, "nll_player": npl,
                         "gain_pct": 100 * (nr - npl) / nr})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "wy_national.csv"), index=False)
    say("A6. Club season 2017/18 style -> national team tournament (player model from his league season only)")
    for nat, g in T.groupby("tournament"):
        lo, hi = P.boot_ci(g["gain_pct"])
        say(f"  -> {nat:14s} players {len(g):4d}  mean gain {g.gain_pct.mean():+.2f}% ({lo:+.2f} .. {hi:+.2f})  gain > 0 in {100 * (g.gain_pct > 0).mean():.0f}%")
    say("  biggest gains:", "; ".join(f"{r.player} ({r.national}) {r.gain_pct:+.1f}%" for r in T.sort_values('gain_pct', ascending=False).head(6).itertuples()))


def run_A(steps):
    D = wy_data()
    X, Z = wy_X(D)
    for s in steps:
        {"wy_eval": wy_eval, "wy_curve": wy_curve, "wy_stability": wy_stability, "wy_similar": wy_similar,
         "wy_league": wy_league, "wy_national": wy_national}[s](D, X, Z)
        save_log(s)


# ------------------------------------------------------------------ C. PFF World Cup 2022 tracking

def pff_matches():
    import settings
    return sorted(settings.dataset_matches("pff"), key=str)


def pff_gnn(fold, epochs=10, patience=2):
    import torch
    import torch.nn as nn
    import settings
    import train_gnn as T
    out = os.path.join(OUT, f"pff_gnn_fold{fold}.pt")
    if os.path.exists(out):
        print("exists", out)
        return
    torch.set_num_threads(8)
    torch.manual_seed(fold)
    np.random.seed(fold)
    ms = pff_matches()
    for m in ms:
        T.PATH[m] = settings.base(m)
    train = [m for i, m in enumerate(ms) if i % 4 != fold]
    t0 = time.time()
    data = {m: T.load(m) for m in train}
    print(f"fold {fold}: {len(train)} training matches loaded ({time.time() - t0:.0f}s)", flush=True)
    tr_idx, va_idx = {}, {}
    for m in train:
        idx = data[m]["idx"]
        cut = int(len(idx) * 0.9)
        tr_idx[m], va_idx[m] = idx[:cut], idx[cut:]
    model = T.Net(data[train[0]]["feat"].shape[2], 48, "ours")
    model.load_state_dict(torch.load(os.path.join(ROOT, "outputs", "gnn_model_fold_J03WOH.pt"), map_location="cpu"))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pool = [(m, k) for m in train for k in tr_idx[m]]
    v0 = np.mean([T.evaluate(model, data[m], va_idx[m])[0][0] for m in train])
    print(f"  Bundesliga start: validation error {v0:.3f} m", flush=True)
    best, best_state, stale = v0, copy.deepcopy(model.state_dict()), 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(pool))
        t1 = time.time()
        for i in range(0, len(perm), 128):
            chunk = [pool[j] for j in perm[i:i + 128]]
            by = {}
            for m, k in chunk:
                by.setdefault(m, []).append(k)
            parts = [T.batch(data[m], np.array(ks)) for m, ks in by.items()]
            b = T.mirror({key: torch.cat([p[key] for p in parts]) for key in parts[0]})
            pred, _ = model(b)
            ok = b["ok"].float()[..., None]
            loss = (nn.functional.smooth_l1_loss(pred, b["target"], reduction="none") * ok).sum() / (ok.sum() * 2 + 1e-6)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = np.mean([T.evaluate(model, data[m], va_idx[m])[0][0] for m in train])
        improved = v < best - 0.003
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
        stale = 0 if improved else stale + 1
        print(f"  epoch {ep + 1}: validation error {v:.3f} m ({time.time() - t1:.0f}s)", flush=True)
        if ep >= 2 and stale >= patience:
            break
    torch.save(best_state, out)
    print(f"saved {out}; best validation {best:.3f} m (start {v0:.3f}); {(time.time() - t0) / 60:.0f} min", flush=True)


def pff_ids(m, home, away):
    """slot -> (player id, name) per frame, from the jersey in the processed file and the PFF roster."""
    import settings
    d = pd.read_csv(settings.base(m) + ".csv.gz", usecols=["frame", "team", "player_id", "dfl_id", "visible",
                                                           "ball_owner_team", "game_mode"])
    ro = json.load(open(os.path.join(ROOT, "data", "external", "pff_worldcup", "raw", "FIFA World Cup 2022", "Rosters",
                                     f"{m}.json"), encoding="utf-8"))
    side_of = {home: "home", away: "away"}
    jer = {(side_of.get(r["team"]["name"], "?"), int(r["shirtNumber"])): (int(r["player"]["id"]), r["player"]["nickname"])
           for r in ro}
    d["slot"] = np.where(d["team"] == "home", d["player_id"], 11 + d["player_id"])
    d["jersey"] = d["dfl_id"].str.rsplit("-", n=1).str[-1].astype(int)
    d["pid"] = [jer.get((t, j), (-1, ""))[0] for t, j in zip(d["team"], d["jersey"])]
    names = {v[0]: v[1] for v in jer.values()}
    pid = d.pivot_table(index="frame", columns="slot", values="pid", aggfunc="first").reindex(columns=range(22))
    vis = d.pivot_table(index="frame", columns="slot", values="visible", aggfunc="first").reindex(columns=range(22))
    fr = d.drop_duplicates("frame").set_index("frame")[["ball_owner_team", "game_mode"]]
    return pid, vis, fr, names


def pff_rows():
    path = os.path.join(OUT, "pff_rows.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path), json.load(open(os.path.join(OUT, "pff_names.json"), encoding="utf-8"))
    import torch
    import settings
    import train_gnn as T
    ms = pff_matches()
    reg = pd.read_csv(os.path.join(ROOT, "data", "pff", "matches.csv"), dtype={"match_id": str}).set_index("match_id")
    for m in ms:
        T.PATH[m] = settings.base(m)
    parts, allnames = [], {}
    for i, m in enumerate(ms):
        d = T.load(m)
        z = np.load(T.PATH[m] + f"_graphs{T.SUFFIX}.npz")
        g = {k: z[k] for k in ("frame", "time_s", "xy", "role")}
        preds = {}
        for tag, w in (("wc", os.path.join(OUT, f"pff_gnn_fold{i % 4}.pt")), ("bl", os.path.join(ROOT, "outputs", "gnn_model_fold_J03WOH.pt"))):
            net = T.Net(d["feat"].shape[2], 48, "ours")
            net.load_state_dict(torch.load(w, map_location="cpu"))
            keep = []
            T.evaluate(net, d, keep=keep)
            preds[tag] = np.concatenate([k[0] for k in keep])
            targ = np.concatenate([k[1] for k in keep])
            ok = np.concatenate([k[2] for k in keep])
        pid, vis, fr, names = pff_ids(m, reg.loc[str(m), "home"], reg.loc[str(m), "away"])
        allnames.update({str(k): v for k, v in names.items()})
        ks = d["idx"]
        f0, f1 = g["frame"][ks].astype(int), g["frame"][ks + 1].astype(int)
        P0, V0 = pid.reindex(f0).to_numpy(), vis.reindex(f0).to_numpy()
        V1 = vis.reindex(f1).to_numpy()
        own_t = fr["ball_owner_team"].reindex(f0).to_numpy()
        mode = fr["game_mode"].reindex(f0).to_numpy()
        gap = (g["time_s"][ks + 1] - g["time_s"][ks]).astype(float)
        n = len(ks)
        S = np.tile(np.arange(22), (n, 1))
        home = S < 11
        sg = np.where(home, 1.0, -1.0)                                         # PFF files: home attacks to x = 105
        side = np.where(home, "home", "away")
        p = g["xy"][ks, :22]
        ball = g["xy"][ks, 22]
        px = np.where(sg > 0, p[..., 0], L - p[..., 0]); py = np.where(sg > 0, p[..., 1], W - p[..., 1])
        bx = np.where(sg > 0, ball[:, None, 0], L - ball[:, None, 0]); by = np.where(sg > 0, ball[:, None, 1], W - ball[:, None, 1])
        keep = (ok[:, :22] & ~np.isnan(p[..., 0]) & ~np.isnan(ball[:, None, 0]) & (mode[:, None] == "normal")
                & (np.nan_to_num(P0, nan=-1) > 0) & (np.nan_to_num(V0) == 1) & (np.nan_to_num(V1) == 1))
        own = (own_t[:, None] == side)
        opp = np.isin(own_t, ["home", "away"])[:, None] & ~own
        J = np.tile(np.arange(n)[:, None], (1, 22))
        R = pd.DataFrame({"match": m, "j": J[keep], "pid": P0[keep].astype(int), "slot": S[keep], "role": g["role"][ks, :22][keep].astype(int),
                          "own": own[keep].astype(int), "opp": opp[keep].astype(int), "px": px[keep], "py": py[keep],
                          "bx": bx[keep], "by": by[keep], "gap": np.repeat(gap[:, None], 22, 1)[keep],
                          "predx": (sg * preds["wc"][:, :22, 0])[keep], "predy": (sg * preds["wc"][:, :22, 1])[keep],
                          "blx": (sg * preds["bl"][:, :22, 0])[keep], "bly": (sg * preds["bl"][:, :22, 1])[keep],
                          "realx": (sg * targ[:, :22, 0])[keep], "realy": (sg * targ[:, :22, 1])[keep]})
        parts.append(R)
        print(f"  {i + 1}/{len(ms)} {m}: {n} Key States, {len(R)} visible player steps", flush=True)
    M = pd.concat(parts, ignore_index=True)
    M.to_csv(path, index=False)
    json.dump(allnames, open(os.path.join(OUT, "pff_names.json"), "w", encoding="utf-8"), ensure_ascii=False)
    return M, allnames


def pff():
    M, names = pff_rows()
    err = lambda r: np.linalg.norm(r, axis=1)
    real = M[["realx", "realy"]].to_numpy()
    say(f"C. PFF World Cup 2022: {len(M)} visible player steps, {M['match'].nunique()} matches, {M['pid'].nunique()} players")
    still = err(real).mean()
    say("C1. own_MovementGNN on held-out World Cup matches (error per player step, m):")
    say(f"  nobody moves {still:.3f}   Bundesliga-trained GNN {err(real - M[['blx', 'bly']].to_numpy()).mean():.3f}   "
        f"World Cup fine-tuned GNN (4 folds) {err(real - M[['predx', 'predy']].to_numpy()).mean():.3f}")
    per = M.groupby("match").apply(lambda g: pd.Series({
        "bl": err(g[["realx", "realy"]].to_numpy() - g[["blx", "bly"]].to_numpy()).mean(),
        "wc": err(g[["realx", "realy"]].to_numpy() - g[["predx", "predy"]].to_numpy()).mean()}))
    say(f"  World Cup GNN better in {int((per.wc < per.bl).sum())} of {len(per)} matches")
    X = P.movement_X(M)
    Rres = real - M[["predx", "predy"]].to_numpy()
    pm = M.groupby(["pid", "match"]).size().unstack(fill_value=0)
    focus = pm.index[(pm >= 150).sum(1) >= 3].tolist()
    say(f"C2. Player movement model vs role (leave one match out): {len(focus)} players with 3+ matches of 150+ steps "
        f"(max matches per player {int((pm >= 150).sum(1).max())})")
    res = []
    pid_arr, m_arr, role_arr = M["pid"].to_numpy(), M["match"].to_numpy(), M["role"].to_numpy()
    for t, mid in enumerate(sorted(M["match"].unique())):
        tr = m_arr != mid
        rc = np.zeros_like(Rres)
        for role in np.unique(role_arr):
            k = tr & (role_arr == role)
            B = P.ridge(X[k], Rres[k], 100.0)
            rc[role_arr == role] = X[role_arr == role] @ B
        for pid in focus:
            te = np.flatnonzero((m_arr == mid) & (pid_arr == pid))
            if len(te) < 50:
                continue
            ti = np.flatnonzero(tr & (pid_arr == pid))
            Y = Rres[ti] - rc[ti]
            lam = P.pick_lam(m_arr[ti], lambda a, b, lam: err(Y[b] - X[ti][b] @ P.ridge(X[ti][a], Y[a], lam)),
                             lams=[10, 30, 100, 300, 1000, 3000, 10000], folds=10)
            B = P.ridge(X[ti], Y, lam)
            r0 = Rres[te]
            res.append({"match": mid, "pid": pid, "name": names.get(str(pid), str(pid)), "n": len(te), "lam": lam,
                        "train_matches": len(np.unique(m_arr[ti])), "err_gnn": err(r0).mean(),
                        "err_role": err(r0 - rc[te]).mean(), "err_player": err(r0 - rc[te] - X[te] @ B).mean()})
        if (t + 1) % 16 == 0:
            print(f"  fold {t + 1}/64", flush=True)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "pff_movement_matches.csv"), index=False)
    w = R["n"]
    say(f"  pooled error (m): GNN {np.average(R.err_gnn, weights=w):.4f}  + role {np.average(R.err_role, weights=w):.4f}  "
        f"+ player {np.average(R.err_player, weights=w):.4f}")
    rows = []
    for pid, g in R.groupby("pid"):
        gain = (g["err_role"] - g["err_player"]).to_numpy()
        lo, hi = P.boot_ci(gain)
        rows.append({"pid": pid, "name": g["name"].iloc[0], "matches": len(g), "gain_cm": 100 * gain.mean(),
                     "lo_cm": 100 * lo, "hi_cm": 100 * hi, "matches_better": int((gain > 0).sum())})
    E = pd.DataFrame(rows).sort_values("gain_cm", ascending=False)
    E["verdict"] = np.where(E.lo_cm > 0, "player better", np.where(E.hi_cm < 0, "role better", "unclear"))
    E.to_csv(os.path.join(OUT, "pff_movement_eval.csv"), index=False)
    say(f"  players: player model better {int((E.verdict == 'player better').sum())}, unclear {int((E.verdict == 'unclear').sum())}, "
        f"role better {int((E.verdict == 'role better').sum())}; median gain {E.gain_cm.median():+.2f} cm")
    say("  by number of training matches: " + "  ".join(
        f"{n}: {100 * ((g.err_role - g.err_player) * g.n).sum() / g.n.sum():+.2f} cm" for n, g in R.groupby("train_matches")))
    say("  top 5:", "; ".join(f"{r.name} {r.gain_cm:+.1f} cm ({r.matches_better}/{r.matches})" for r in E.head(5).itertuples()))


# ------------------------------------------------------------------ viewer export

def viewer():
    """viewer/data/player_models_big.json: the findings table, learning curves, GNN numbers, and per player (by
    StatsBomb name, as on the Leverkusen cards): tournament test, club -> tournament transfer, pressure profile."""
    rd = lambda f: pd.read_csv(os.path.join(OUT, f))
    r3 = lambda v: None if pd.isna(v) else round(float(v), 3)
    curve = lambda C: [{"n": int(n), "players": len(g), "gain": r3(g.gain_pct.mean()), "ci": [r3(v) for v in P.boot_ci(g.gain_pct)],
                        "share": r3((g.gain_pct > 0).mean())} for n, g in C.groupby("n_matches")]
    WE, SE, ST, SP, PE = rd("wy_eval.csv"), rd("sb_eval.csv"), rd("sb_transfer.csv"), rd("sb_pressure.csv"), rd("pff_movement_eval.csv")
    S, NAT, MR = rd("wy_style.csv"), rd("wy_national.csv"), rd("pff_rows.csv.gz")
    err = lambda a, b: float(np.linalg.norm(MR[a].to_numpy() - MR[b].to_numpy(), axis=1).mean())
    still = float(np.linalg.norm(MR[["realx", "realy"]].to_numpy(), axis=1).mean())
    stab = {c: [r3(corr(S[f"first_{c}"], S[f"second_{c}"])), r3(corr(S[f"odd_{c}"], S[f"even_{c}"]))] for c in P.CLASSES}
    rel = {c: r3(corr(SP[f"odd_{c}"], SP[f"even_{c}"])) for c in P.CLASSES}
    v = lambda E, k: int((E.verdict == k).sum())
    out = {"model": "own_PlayerModel at scale", "classes": P.CLASSES,
           "datasets": [
               {"key": "B", "name": "StatsBomb 360: Euro 2020, World Cup 2022, Euro 2024 + Leverkusen 23/24", "matches": 197, "what": "on-ball choices"},
               {"key": "A", "name": "Wyscout public: 5 leagues 2017/18, World Cup 2018, Euro 2016", "matches": 1941, "what": "on-ball choices"},
               {"key": "C", "name": "PFF FC World Cup 2022 tracking (camera-visible steps)", "matches": 64, "what": "off-ball movement"}],
           "eval": {"wyscout": {"players": len(WE), "better": v(WE, "player better"), "unclear": v(WE, "unclear"), "worse": v(WE, "role better"),
                                "median_gain": r3(WE.gain_pct.median())},
                    "statsbomb": {"players": len(SE), "better": v(SE, "player better"), "unclear": v(SE, "unclear"), "worse": v(SE, "role better"),
                                  "median_gain": r3(SE.gain_pct.median())},
                    "movement": {"players": len(PE), "better": v(PE, "player better"), "unclear": v(PE, "unclear"), "worse": v(PE, "role better"),
                                 "median_gain_cm": r3(PE.gain_cm.median())}},
           "curve_wyscout": curve(rd("wy_curve.csv")), "curve_leverkusen": curve(rd("sb_curve.csv")),
           "stability": stab, "pressure_reliability": rel,
           "national": {t: {"players": len(g), "gain": r3(g.gain_pct.mean()), "share": r3((g.gain_pct > 0).mean())} for t, g in NAT.groupby("tournament")},
           "gnn": {"still": r3(still), "bundesliga": r3(err(["realx", "realy"], ["blx", "bly"])), "worldcup": r3(err(["realx", "realy"], ["predx", "predy"]))},
           "players": {}}
    for _, r in SP.iterrows():
        out["players"].setdefault(r["player"], {})["pressure"] = {"role": r["role"], "decisions": int(r["decisions"]),
                                                                  "delta": [r3(r[f"all_{c}"]) for c in P.CLASSES]}
    for _, r in SE.iterrows():
        out["players"].setdefault(r["player"], {})["test"] = {"source": r["main_source"], "matches": int(r["matches"]),
                                                              "gain_pct": r3(r["gain_pct"]), "verdict": r["verdict"]}
    for _, r in ST[ST["train"] == LEV].iterrows():
        out["players"].setdefault(r["player"], {}).setdefault("transfer", []).append(
            {"to": r["test"], "decisions": int(r["test_decisions"]), "gain_pct": r3(r["gain_pct"])})
    json.dump(out, open(os.path.join(ROOT, "viewer", "data", "player_models_big.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print("wrote viewer/data/player_models_big.json;", len(out["players"]), "players")


# ------------------------------------------------------------------ main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    B_STEPS = ["sb_eval", "sb_curve", "sb_transfer", "sb_pressure"]
    A_STEPS = ["wy_eval", "wy_curve", "wy_stability", "wy_similar", "wy_league", "wy_national"]
    if a.step == "B" or a.step in B_STEPS:
        run_B(B_STEPS if a.step == "B" else [a.step])
    elif a.step == "A" or a.step in A_STEPS:
        run_A(A_STEPS if a.step == "A" else [a.step])
    elif a.step == "pff_gnn":
        pff_gnn(a.fold)
    elif a.step in ("pff", "C"):
        pff()
        save_log("pff")
    elif a.step == "viewer":
        viewer()
