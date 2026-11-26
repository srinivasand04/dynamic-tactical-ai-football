"""own_ValueModel: a possession value model (VAEP, Decroos et al. 2019) for every on-ball and defensive action.

For each action: P(the acting team scores within the next 10 actions) and P(it concedes within the next 10 actions),
from gradient-boosted trees on the action (type, start / end, result, body part), the two previous actions, and
-- for StatsBomb -- the 360 freeze frame (nearest opponent, opponents within 5 m, opponents goal-side, team-mates
ahead, side of the pressure). Value of an action = change in P(score) minus change in P(concede).

Two models, each tested on matches it never saw (5 folds by match):
    sb   StatsBomb 360: Leverkusen 23/24 + Euro 2020 + World Cup 2022 + Euro 2024 (200 matches)
    wy   Wyscout public: 5 leagues 2017/18 + World Cup 2018 + Euro 2016 (1,941 matches; no freeze frames)
Baseline for each: the same trees on location only (start / end, action type) -- an expected-threat-like model.

Outputs (outputs/value_model/):
    {sb,wy}_values.csv.gz   per action, out-of-fold probabilities and values (rebuilt, not committed)
    {sb,wy}_ratings.csv     per player and competition: minutes, VAEP per 90 (attack / defence), percentile in role
    eval.json               AUC, Brier, log-loss, calibration (full vs location-only), rating reliability and
                            predictive validity (first half of the season -> second half)
Usage : python scripts/value_model.py sb | wy | ratings | all       (run from the project root)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import player_models_big as BG  # noqa: E402
import player_of_match as PM  # noqa: E402
import target_player as TP  # noqa: E402

ROOT = PM.ROOT
OUT = os.path.join(ROOT, "outputs", "value_model")
L, W = 105.0, 68.0
NXT = 10
RNG = np.random.default_rng(3)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def save_log(step):
    open(os.path.join(OUT, f"summary_{step}.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    LOG.clear()


def put_eval(key, obj):
    p = os.path.join(OUT, "eval.json")
    E = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    E[key] = obj
    json.dump(E, open(p, "w", encoding="utf-8"), indent=1, default=float)


# ------------------------------------------------------------------ StatsBomb: every action, with the 360 frame

SB_KINDS = {"Pass", "Carry", "Shot", "Dribble", "Miscontrol", "Dispossessed", "Duel", "Interception", "Ball Recovery",
            "Clearance", "Block", "Foul Committed", "Goal Keeper", "50/50", "Own Goal Against"}


def sb_parse():
    path = os.path.join(OUT, "sb_actions_all.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    mins = pd.read_csv(os.path.join(BG.OUT, "sb_minutes.csv"))
    role = {(m, p): r for m, p, r in zip(mins["match"], mins["player_id"], mins["role"])}
    jobs = [(PM.SB, m["match_id"], TP.LEV, m["match_date"]) for m in json.load(open(os.path.join(PM.SB, "matches_9_281.json"), encoding="utf-8"))]
    for (c, s), src in BG.TOURN.items():
        jobs += [(BG.SBT, m["match_id"], src, m["match_date"]) for m in json.load(open(os.path.join(BG.SBT, f"matches_{c}_{s}.json"), encoding="utf-8"))]
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
            if kind not in SB_KINDS or "location" not in e or "player" not in e or e["period"] > 4:
                continue
            sub = e.get(kind.lower().replace(" ", "_")) or {}
            st = (sub.get("type") or {}).get("name", "")
            me = PM.to_m(e["location"])
            end = PM.to_m(sub["end_location"][:2]) if "end_location" in sub else me
            out = (sub.get("outcome") or {}).get("name", "")
            ok = {"Pass": out == "", "Carry": True, "Dribble": out == "Complete", "Shot": out == "Goal",
                  "Duel": out in ("Won", "Success In Play", "Success Out", "Success"), "Interception": out in ("Won", "Success In Play", "Success Out", "Success"),
                  "Ball Recovery": not sub.get("recovery_failure", False), "Clearance": True, "Block": True,
                  "Goal Keeper": out in ("Success", "In Play Safe", "Saved Twice", "Claim", "Clear", "Won", "Success In Play", "Success Out"),
                  "50/50": "Won" in str((sub.get("outcome") or {}).get("name", ""))}.get(kind, False)
            sk = kind
            if kind == "Pass":
                sk = ("Cross" if sub.get("cross") else st if st in ("Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off") else "Pass")
            if kind == "Shot" and st == "Penalty":
                sk = "Penalty"
            if kind == "Shot" and st == "Free Kick":
                sk = "Free Kick Shot"
            ff = ffs.get(e["id"])
            r = {"source": src, "date": date, "match": mid, "id": e["id"], "period": e["period"], "t": 60 * e["minute"] + e["second"],
                 "team": e["team"]["name"], "player_id": e["player"]["id"], "player": e["player"]["name"], "role": role.get((mid, e["player"]["id"]), "CM"),
                 "kind": sk, "x": me[0], "y": me[1], "ex": end[0], "ey": end[1], "ok": int(ok),
                 "goal": int(kind == "Shot" and out == "Goal"), "owngoal": int(kind == "Own Goal Against"),
                 "body": (sub.get("body_part") or {}).get("name", ""), "pressed": int(bool(e.get("under_pressure"))),
                 "xg": float(sub.get("statsbomb_xg") or 0), "has_ff": int(ff is not None)}
            if ff:
                r["near_opp"], r["opp5"], r["goal_side"], r["mates_ahead"] = BG.ff_feats(ff, me)
                r["side"] = TP.side_of(ff, me)
            rows.append(r)
        if i % 25 == 0:
            print(f"  StatsBomb {i}/{len(jobs)}: {len(rows)} actions", flush=True)
    A = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    A.to_csv(path, index=False)
    return A


# ------------------------------------------------------------------ Wyscout: every event with a position

def wy_parse():
    path = os.path.join(OUT, "wy_actions_all.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    names = {p["wyId"]: p["shortName"] for p in json.load(open(os.path.join(BG.WY, "players.json"), encoding="utf-8"))}
    teams = {t["wyId"]: t["name"] for t in json.load(open(os.path.join(BG.WY, "teams.json"), encoding="utf-8"))}
    parts = []
    per = {"1H": 1, "2H": 2, "E1": 3, "E2": 4}
    for key, comp in BG.WY_COMP.items():
        t0 = time.time()
        ms = json.load(open(os.path.join(BG.WY, "matches", f"matches_{key}.json"), encoding="utf-8"))
        date = {m["wyId"]: m["dateutc"][:10] for m in ms}
        ev = json.load(open(os.path.join(BG.WY, "events", f"events_{key}.json"), encoding="utf-8"))
        c = {k: [] for k in ("match", "period", "t", "team_id", "player_id", "kind", "x", "y", "ex", "ey", "ok", "goal", "owngoal", "body")}
        for e in ev:
            if e["matchPeriod"] not in per or e["eventName"] in ("Interruption", "Offside") or not e.get("positions") or e["playerId"] <= 0:
                continue
            tg = {t["id"] for t in e.get("tags") or []}
            sub = e["subEventName"] or e["eventName"]
            if e["eventName"] == "Duel" and sub == "Ground attacking duel" and tg & {503, 504}:
                sub = "Dribble"
            pos = e["positions"]
            p1 = pos[1] if len(pos) > 1 else pos[0]
            c["match"].append(e["matchId"]); c["period"].append(per[e["matchPeriod"]]); c["t"].append(e["eventSec"])
            c["team_id"].append(e["teamId"]); c["player_id"].append(e["playerId"]); c["kind"].append(sub)
            c["x"].append(pos[0]["x"] * L / 100); c["y"].append(W - pos[0]["y"] * W / 100)
            c["ex"].append(p1["x"] * L / 100); c["ey"].append(W - p1["y"] * W / 100)
            c["ok"].append(int(1801 in tg or 703 in tg)); c["goal"].append(int(101 in tg and e["eventName"] in ("Shot", "Free Kick")))
            c["owngoal"].append(int(102 in tg)); c["body"].append("Left Foot" if 401 in tg else "Right Foot" if 402 in tg else "Head" if 403 in tg else "")
        del ev
        E = pd.DataFrame(c)
        E["source"], E["date"] = comp, E["match"].map(date)
        E["team"], E["player"] = E["team_id"].map(teams), E["player_id"].map(lambda p: names.get(p, str(p)))
        parts.append(E)
        say(f"  Wyscout {comp}: {len(E)} actions ({time.time() - t0:.0f}s)")
    A = pd.concat(parts, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    A.to_csv(path, index=False)
    return A


# ------------------------------------------------------------------ labels, features, model

def labels(A):
    """scores / concedes within the next NXT actions (same match), from the acting team's view."""
    n = len(A)
    m = A["match"].to_numpy()
    team = pd.factorize(A["team"].astype(str))[0]
    g = A["goal"].to_numpy().astype(bool)
    og = A["owngoal"].to_numpy().astype(bool)
    sc, co = np.zeros(n, bool), np.zeros(n, bool)
    idx = np.arange(n)
    for k in range(NXT):
        j = np.minimum(idx + k, n - 1)
        same_m = (idx + k < n) & (m[j] == m)
        same_t = team[j] == team
        sc |= same_m & ((same_t & g[j]) | (~same_t & og[j]))
        co |= same_m & ((~same_t & g[j]) | (same_t & og[j]))
    return sc.astype(int), co.astype(int)


def features(A, full=True):
    x, y, ex, ey = (A[c].to_numpy(float) for c in ("x", "y", "ex", "ey"))
    dg = np.hypot(L - x, W / 2 - y)
    dge = np.hypot(L - ex, W / 2 - ey)
    ang = np.abs(np.arctan2(W / 2 - y, L - x))
    kind = A["kind"].astype("category")
    F = pd.DataFrame({"kind": kind.cat.codes, "x": x, "y": y, "ex": ex, "ey": ey, "dist_goal": dg, "end_dist_goal": dge,
                      "angle": ang, "dx": ex - x, "dy": ey - y})
    if not full:
        return F, ["kind"]
    F["ok"] = A["ok"].to_numpy()
    F["body"] = A["body"].astype("category").cat.codes
    m = A["match"].to_numpy()
    team = A["team"].astype(str).to_numpy()
    t = A["t"].to_numpy(float) + 3600 * A["period"].to_numpy()
    for k in (1, 2):
        sh = np.roll(np.arange(len(A)), k)
        valid = (np.arange(len(A)) >= k) & (m[sh] == m)
        same = valid & (team[sh] == team)
        px, py = np.where(same, ex[sh], L - ex[sh]), np.where(same, ey[sh], W - ey[sh])
        F[f"p{k}_kind"] = np.where(valid, kind.cat.codes.to_numpy()[sh], -1)
        F[f"p{k}_same"] = np.where(valid, same.astype(int), -1)
        F[f"p{k}_ex"], F[f"p{k}_ey"] = np.where(valid, px, np.nan), np.where(valid, py, np.nan)
        F[f"p{k}_ok"] = np.where(valid, A["ok"].to_numpy()[sh], -1)
        F[f"p{k}_dt"] = np.where(valid, np.clip(t - t[sh], 0, 60), np.nan)
    F["gap_move"] = np.hypot(x - F["p1_ex"], y - F["p1_ey"])
    if "has_ff" in A:
        F["pressed"] = A["pressed"].to_numpy()
        for c in ("near_opp", "opp5", "goal_side", "mates_ahead"):
            F[c] = A[c].to_numpy(float)
        F["side"] = A["side"].astype("category").cat.codes
    cats = [c for c in ("kind", "body", "p1_kind", "p2_kind", "side") if c in F]
    return F, cats


def fit_oof(F, cats, y, groups, tag):
    from sklearn.ensemble import HistGradientBoostingClassifier
    u = np.unique(groups)
    fold = dict(zip(RNG.permutation(u), np.arange(len(u)) % 5))
    f = np.array([fold[g] for g in groups])
    p = np.zeros(len(y))
    for k in range(5):
        tr, te = f != k, f == k
        Fc = F.copy()
        for c in cats:
            Fc[c] = Fc[c].astype(int) + 1                                     # HistGB categories must be >= 0
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=80,
                                             l2_regularization=1.0, categorical_features=[Fc.columns.get_loc(c) for c in cats],
                                             early_stopping=True, validation_fraction=0.1, random_state=0)
        clf.fit(Fc[tr], y[tr])
        p[te] = clf.predict_proba(Fc[te])[:, 1]
        print(f"    {tag} fold {k + 1}/5", flush=True)
    return p


def scores(y, p):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    q = np.clip(p, 1e-6, 1 - 1e-6)
    bins = np.quantile(q, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(bins, q, side="right") - 1, 0, 9)
    cal = [[float(q[idx == b].mean()), float(y[idx == b].mean())] for b in range(10) if (idx == b).any()]
    base = y.mean()
    return {"auc": float(roc_auc_score(y, q)), "brier": float(brier_score_loss(y, q)), "logloss": float(log_loss(y, q)),
            "brier_skill": float(1 - brier_score_loss(y, q) / (base * (1 - base))), "rate": float(base), "calibration": cal}


def values(A, ps, pc):
    """VAEP: value = (Ps_i - Ps_prev) - (Pc_i - Pc_prev), previous from the acting team's view; reset after a goal."""
    m = A["match"].to_numpy()
    team = A["team"].astype(str).to_numpy()
    n = len(A)
    prev = np.roll(np.arange(n), 1)
    valid = (np.arange(n) > 0) & (m[prev] == m)
    same = team[prev] == team
    ps0 = np.where(same, ps[prev], pc[prev])
    pc0 = np.where(same, pc[prev], ps[prev])
    after_goal = (A["goal"].to_numpy() + A["owngoal"].to_numpy())[prev] > 0
    reset = ~valid | after_goal
    ps0, pc0 = np.where(reset, 0.0, ps0), np.where(reset, 0.0, pc0)
    off, dfn = ps - ps0, -(pc - pc0)
    return off, dfn


def run_model(tag, A):
    t0 = time.time()
    A = A.sort_values(["match", "period", "t"], kind="stable").reset_index(drop=True) if tag == "wy" else A
    sc, co = labels(A)
    say(f"{tag}: {len(A)} actions, {A['match'].nunique()} matches; scores within {NXT} actions {100 * sc.mean():.2f}%, concedes {100 * co.mean():.2f}%")
    res = {}
    for full in (False, True):
        F, cats = features(A, full)
        name = "full" if full else "location"
        ps = fit_oof(F, cats, sc, A["match"].to_numpy(), f"{tag} {name} scores")
        pc = fit_oof(F, cats, co, A["match"].to_numpy(), f"{tag} {name} concedes")
        res[name] = {"scores": scores(sc, ps), "concedes": scores(co, pc)}
        say(f"  {name:9s} scores: AUC {res[name]['scores']['auc']:.3f}  Brier skill {res[name]['scores']['brier_skill']:+.3f}  | "
            f"concedes: AUC {res[name]['concedes']['auc']:.3f}  Brier skill {res[name]['concedes']['brier_skill']:+.3f}")
        if full:
            off, dfn = values(A, ps, pc)
            A["p_score"], A["p_concede"], A["v_off"], A["v_def"] = ps, pc, off, dfn
            A["vaep"] = off + dfn
        else:
            off0, dfn0 = values(A, ps, pc)
            A["vaep_loc"] = off0 + dfn0
    # one-match check: total value per team vs goals
    say(f"  ({time.time() - t0:.0f}s)")
    keep = ["source", "date", "match", "period", "t", "team", "player_id", "player", "kind", "x", "y", "ex", "ey", "ok",
            "goal", "p_score", "p_concede", "v_off", "v_def", "vaep", "vaep_loc"] + (["id", "role", "body"] if tag == "sb" else ["team_id"])
    A[keep].to_csv(os.path.join(OUT, f"{tag}_values.csv.gz"), index=False)
    put_eval(tag, res)
    return A


def step_sb():
    run_model("sb", sb_parse())
    save_log("sb")


def step_wy():
    run_model("wy", wy_parse())
    save_log("wy")


# ------------------------------------------------------------------ ratings, reliability, predictive validity

def per90(A, mins, keys, col):
    v = A.groupby(keys)[col].sum()
    m = mins.groupby(keys)["minutes"].sum()
    return (90 * v / m.reindex(v.index)), m


def step_ratings():
    out = {}
    sbm = pd.read_csv(os.path.join(BG.OUT, "sb_minutes.csv"))
    wym = pd.read_csv(os.path.join(BG.OUT, "wy_minutes.csv")).rename(columns={"comp": "source"})
    wrole = pd.read_csv(os.path.join(BG.OUT, "wy_decisions.csv.gz"), usecols=["player_id", "source", "role"])
    wrole = wrole.groupby(["source", "player_id"])["role"].agg(lambda s: s.value_counts().index[0])
    for tag, mins in (("sb", sbm), ("wy", wym)):
        A = pd.read_csv(os.path.join(OUT, f"{tag}_values.csv.gz"), low_memory=False)
        g = A.groupby(["source", "player_id"])
        R = g.agg(player=("player", "first"), team=("team", lambda s: s.value_counts().index[0]), actions=("vaep", "size"),
                  vaep=("vaep", "sum"), v_off=("v_off", "sum"), v_def=("v_def", "sum"), vaep_loc=("vaep_loc", "sum")).reset_index()
        m = mins.groupby(["source", "player_id"])["minutes"].sum()
        R["minutes"] = [m.get((s, p), np.nan) for s, p in zip(R["source"], R["player_id"])]
        if tag == "sb":
            rl = mins.groupby(["source", "player_id"])["role"].agg(lambda s: s.value_counts().index[0])
        else:
            rl = wrole
        R["role"] = [rl.get((s, p), "") for s, p in zip(R["source"], R["player_id"])]
        R = R[R["minutes"] >= 90].copy()
        for c in ("vaep", "v_off", "v_def", "vaep_loc"):
            R[f"{c}90"] = 90 * R[c] / R["minutes"]
        q = R["minutes"] >= (900 if tag == "wy" else 270)
        R["pct_role"], R["pct_n"] = np.nan, np.nan
        # Wyscout: within the competition and role; StatsBomb: within the role over all its competitions (one club
        # season + tournaments, too few players per competition)
        for _, gg in R[q].groupby(["source", "role"] if tag == "wy" else ["role"]):
            R.loc[gg.index, "pct_role"] = 100 * gg["vaep90"].rank(pct=True)
            R.loc[gg.index, "pct_n"] = len(gg)
        R = R[~R["role"].isin(["GK", ""]) & R["role"].notna()].sort_values("vaep90", ascending=False)       # outfield only (no role = keeper)
        R.to_csv(os.path.join(OUT, f"{tag}_ratings.csv"), index=False)
        say(f"{tag}: {len(R)} player-competitions with 90+ min; top 10 by VAEP per 90 (min {'900' if tag == 'wy' else '270'}):")
        for r in R[R["minutes"] >= (900 if tag == "wy" else 270)].head(10).itertuples():
            say(f"    {r.player:28s} {r.team:22s} {r.source:16s} {r.role:3s} {r.minutes:5.0f} min  VAEP/90 {r.vaep90:.3f} "
                f"(attack {r.v_off90:.3f}, defence {r.v_def90:.3f})")
        # reliability: odd vs even matches; predictive validity: first half of the season -> second half
        rel = {}
        mm = mins.copy()
        for col in ("vaep", "vaep_loc"):
            Am = A.groupby(["source", "player_id", "match"])[col].sum().rename("v").reset_index()
            Am = Am.merge(mm[["source", "player_id", "match", "minutes"]], on=["source", "player_id", "match"], how="inner")
            Am = Am[Am["minutes"] >= 30]
            Am["order"] = Am.groupby(["source", "player_id"])["match"].rank(method="first")
            dates = A.drop_duplicates("match").set_index("match")["date"]
            Am["date"] = Am["match"].map(dates)
            res = {}
            for split in ("odd_even", "season_halves"):
                if split == "odd_even":
                    a = Am[Am["order"] % 2 == 1]; b = Am[Am["order"] % 2 == 0]
                else:
                    if tag == "sb":
                        S = Am[Am["source"] == TP.LEV]
                    else:
                        S = Am[Am["source"].isin(BG.LEAGUES)]
                    cut = S.groupby("source")["date"].transform(lambda d: d.sort_values().iloc[len(d) // 2])
                    a, b = S[S["date"] < cut], S[S["date"] >= cut]
                pa = a.groupby(["source", "player_id"]).agg(v=("v", "sum"), m=("minutes", "sum"))
                pb = b.groupby(["source", "player_id"]).agg(v=("v", "sum"), m=("minutes", "sum"))
                j = pa.join(pb, lsuffix="_a", rsuffix="_b", how="inner")
                j = j[(j.m_a >= 450) & (j.m_b >= 450)]
                if len(j) > 10:
                    res[split] = {"r": float(np.corrcoef(90 * j.v_a / j.m_a, 90 * j.v_b / j.m_b)[0, 1]), "players": int(len(j))}
            rel[col] = res
        out[tag] = rel
        say(f"  rating reliability ({tag}): " + "  ".join(f"{'VAEP' if c == 'vaep' else 'location-only'} " + ", ".join(
            f"{k} r={v['r']:.2f} ({v['players']} players)" for k, v in d.items()) for c, d in rel.items()))
    put_eval("ratings", out)
    save_log("ratings")


STEPS = {"sb": step_sb, "wy": step_wy, "ratings": step_ratings}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for s in (list(STEPS) if a.step == "all" else [a.step]):
        t0 = time.time()
        STEPS[s]()
        print(f"[{s} done in {time.time() - t0:.0f}s]", flush=True)
