"""own_PlayerOfMatch: who was the best player of a club match, from what he did WITH the ball and WITHOUT it.

Part 1  onball   On-ball threat added (StatsBomb events + 360 freeze frames, Bayer Leverkusen 2023/24, 34 matches).
                 For every open-play pass, carry and shot of either team, own_Threat in goal units (the state model of
                 value_models.threat, target 'this team scores within 15 s', from where the ball and every visible player
                 are) is taken before and after the action:
                   pass / carry completed : value(ball at the end) - value(ball at the start)   (same freeze frame)
                   pass lost              : 0 - value(start)                                    (the chance is gone)
                   shot                   : own_xG(shot) - value(start)                         (penalties left out)
                 3 of the 34 matches have 360 files whose ids do not match their events (StatsBomb open data): left out.
                 The value goes to the player who made the action. Summed per player and match, and per 90 minutes.
                 Check: Impect's packing KPIs for the same 34 matches (bypassed opponents, packing xG; Impect open data,
                 credit Impect, research use only), correlated with ours per player-match and per player over the season.

Part 2  dfl      The full model on the 7 DFL matches (full tracking of all 22, Bundesliga / 2. Bundesliga, 2022/23):
                 on-ball   the same rule as Part 1, from the DFL events (passes, shots) and the tracking frame of each event
                           (extract_passes.py snapshots); the run with the ball between receiving and the next pass counts
                           as a carry for the receiver.
                 off-ball  own_MovementGNN (train_gnn.py, Delaunay graphs) predicts where every player goes from one Key State
                           to the next: the ghost. For every attacker without the ball while his team keeps it:
                               value(he gets the ball where he really went) - value(he gets the ball where the ghost went)
                           with everybody else where they really are. Positive = his run made him more dangerous than an
                           average player would have been. Each match is scored by the GNN fold that never saw it.
                 own_PlayerOfMatch = z(on-ball) + z(off-ball), z within the match, players with 30+ minutes.
                 Check: goals and assists (DFL events) of the top-ranked players.

Outputs: outputs/player_of_match/
    onball_actions.csv.gz      one row per valued action (Leverkusen)
    onball_player_match.csv    per player and match: threat added, actions, minutes
    onball_season.csv          Leverkusen players over the season, per 90
    onball_vs_impect.csv       the check against Impect
    goal15_model.json          own_Threat in goal units
    dfl_player_match.csv       on-ball, off-ball, own_PlayerOfMatch per player and DFL match
    dfl_check.csv              goals / assists among the top-ranked players

Usage:  .venv-analysis\\Scripts\\python.exe scripts\\player_of_match.py onball
        .venv-analysis\\Scripts\\python.exe scripts\\player_of_match.py dfl
"""
import argparse
import glob
import json
import os
import sys
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import value_models as V

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SB = os.path.join(ROOT, "data", "external", "statsbomb_leverkusen_2324")
IMP = os.path.join(ROOT, "data", "external", "impect_bundesliga_2324", "data")
OUT = os.path.join(ROOT, "outputs", "player_of_match")
L, W = 105.0, 68.0
CLUB = "Bayer Leverkusen"
SET_PIECE = {"Corner", "Free Kick", "Kick Off", "Goal Kick", "Throw-in"}


def to_m(loc):
    """StatsBomb 120 x 80, y down, acting team attacks to x = 120  ->  metres 105 x 68, y up, attacking to x = 105."""
    return np.array([loc[0] * L / 120.0, (80.0 - loc[1]) * W / 80.0])


def frame_sides(ff):
    att = np.array([to_m(p["location"]) for p in ff if p["teammate"]]).reshape(-1, 2)
    dfs = np.array([to_m(p["location"]) for p in ff if not p["teammate"]]).reshape(-1, 2)
    kp = [i for i, p in enumerate([p for p in ff if not p["teammate"]]) if p["keeper"]]
    return att, dfs, (kp[0] if kp else None)


def minutes_played(ev):
    """Minutes on the pitch per player id, from the Starting XI, substitutions and red cards."""
    end = max(e["minute"] + e["second"] / 60 for e in ev)
    on, off = {}, {}
    for e in ev:
        t = e["minute"] + e["second"] / 60
        if e["type"]["name"] == "Starting XI":
            for p in e["tactics"]["lineup"]:
                on[p["player"]["id"]] = 0.0
        elif e["type"]["name"] == "Substitution":
            off[e["player"]["id"]] = t
            on[e["substitution"]["replacement"]["id"]] = t
        else:
            card = (e.get("foul_committed") or {}).get("card") or (e.get("bad_behaviour") or {}).get("card")
            if card and card["name"] in ("Red Card", "Second Yellow") and "player" in e:
                off[e["player"]["id"]] = t
    return {p: max(off.get(p, end) - t0, 1.0) for p, t0 in on.items()}


_GOAL = {}


def goal_model():
    """own_Threat in goal units: the same state logistic as value_models.threat, but for 'this team scores within 15 s'
    (value_models fits it in its check and does not save it). Fitted once on the same World Cup moves, cached here.
    Needed because a shot is valued with own_xG (a goal chance): before and after must both be goal chances."""
    if "m" not in _GOAL:
        p = os.path.join(OUT, "goal15_model.json")
        if not os.path.exists(p):
            moves = pd.read_csv(os.path.join(V.OUT, "pff_moves.csv.gz"))
            ev = V.threat_labels(moves)
            m = V.Logit(V.SCOLS, y="goal15").fit(ev)
            json.dump({**m.to_json(), "target": "this team scores within 15 s",
                       "trained_on": f"{len(ev)} World Cup 2022 open-play moves ({int(ev['goal15'].sum())} followed by a goal)"},
                      open(p, "w"), indent=1)
        _GOAL["m"] = json.load(open(p))
    return _GOAL["m"]


def state_value(ball, att, dfs):
    """Chance this team scores within 15 s from here (goal units, like own_xG)."""
    f = {"s_" + k: v for k, v in V.state_features(ball, att, dfs).items()}
    return V._logit(goal_model(), f)


def match_actions(mid):
    ev = json.load(open(os.path.join(SB, "events", f"{mid}.json"), encoding="utf-8"))
    tp = os.path.join(SB, "three-sixty", f"{mid}.json")
    ffs = {f["event_uuid"]: f["freeze_frame"] for f in json.load(open(tp, encoding="utf-8"))} if os.path.exists(tp) else {}
    rows, seen = [], 0
    for e in ev:
        kind = e["type"]["name"]
        if kind not in ("Pass", "Carry", "Shot") or "location" not in e:
            continue
        sub = e.get(kind.lower(), {})
        if kind == "Pass" and (sub.get("type") or {}).get("name") in SET_PIECE:
            continue
        if kind == "Shot" and (sub.get("type") or {}).get("name") != "Open Play":
            continue
        seen += 1
        ff = ffs.get(e["id"])
        if not ff:
            continue
        att, dfs, kp = frame_sides(ff)
        start = to_m(e["location"])
        before = state_value(start, att, dfs)
        if kind == "Shot":
            after = V.xg_shot(start, dfs, kp, header=(sub.get("body_part") or {}).get("name") == "Head")
            ok = True
        else:
            ok = kind == "Carry" or "outcome" not in sub           # StatsBomb: a completed pass has no outcome
            after = state_value(to_m(sub["end_location"]), att, dfs) if ok else 0.0
        rows.append({"match": mid, "team": e["team"]["name"], "player_id": e["player"]["id"], "player": e["player"]["name"],
                     "kind": kind, "minute": e["minute"], "ok": int(ok), "before": round(before, 5), "after": round(after, 5),
                     "added": round(after - before, 5), "goal": int(kind == "Shot" and sub.get("outcome", {}).get("name") == "Goal")})
    mins = minutes_played(ev)
    names = {}
    for e in ev:
        if e["type"]["name"] == "Starting XI":
            for p in e["tactics"]["lineup"]:
                names[p["player"]["id"]] = (p["player"]["name"], e["team"]["name"], p["position"]["name"])
        elif e["type"]["name"] == "Substitution":
            r = e["substitution"]["replacement"]
            names.setdefault(r["id"], (r["name"], e["team"]["name"], e["position"]["name"]))
    return rows, seen, mins, names


def ascii_name(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def impect_table():
    """Leverkusen players per Impect match: bypassed opponents, packing xG, expected goal assists, minutes (Impect ids)."""
    defs = {d["id"]: d["name"] for d in json.load(open(os.path.join(IMP, "kpi_definitions.json"), encoding="utf-8"))}
    want = {"BYPASSED_OPPONENTS": "imp_bypassed", "BYPASSED_OPPONENTS_RECEIVING": "imp_bypassed_recv", "PACKING_XG": "imp_packing_xg",
            "SHOT_XG": "imp_shot_xg", "EXPECTED_GOAL_ASSISTS": "imp_xga"}
    players = {}
    for f in glob.glob(os.path.join(IMP, "players", "*.json")):
        for p in json.load(open(f, encoding="utf-8")):
            players[p["id"]] = p["commonname"]
    matches = {}
    for f in glob.glob(os.path.join(IMP, "matches", "*.json")):
        for m in json.load(open(f, encoding="utf-8")):
            matches[m["id"]] = m
    rows = []
    for f in glob.glob(os.path.join(IMP, "player_kpis", "player_kpis_*.json")):
        d = json.load(open(f, encoding="utf-8"))
        m = matches.get(d["matchId"], {})
        for side in ("squadHome", "squadAway"):
            if d[side]["id"] != 41:                                    # 41 = Bayer Leverkusen
                continue
            for p in d[side]["players"]:
                k = {defs.get(x["kpiId"]): x["value"] for x in p["kpis"]}
                rows.append({"imp_match": d["matchId"], "date": str(m.get("scheduledDate", ""))[:10],
                             "name": players.get(p["id"], str(p["id"])), "imp_minutes": p["playDuration"] / 60,
                             **{c: k.get(n, 0.0) for n, c in want.items()}})
    return pd.DataFrame(rows)


def name_key(s):
    """Last word of the ascii name, plus first letter: 'Florian Wirtz' -> 'wirtz f'; handles 'Alejandro Grimaldo García'."""
    parts = ascii_name(s).replace("-", " ").split()
    return parts


def link_names(sb_names, imp_names):
    """StatsBomb full names -> Impect common names: same surname word and first name initial, else any shared long word."""
    out = {}
    imp = {n: set(name_key(n)) for n in imp_names}
    for s in sb_names:
        w = name_key(s)
        best = [n for n, ws in imp.items() if w[-1] in ws and (w[0][0] == list(name_key(n))[0][0])]
        if not best:
            best = [n for n, ws in imp.items() if len(set(x for x in w if len(x) > 3) & ws) > 0]
        if len(best) == 1:
            out[s] = best[0]
    return out


def onball():
    os.makedirs(OUT, exist_ok=True)
    mids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(SB, "events", "*.json")))
    meta = {str(m["match_id"]): m for m in json.load(open(os.path.join(SB, "matches_9_281.json"), encoding="utf-8"))}
    acts, pm, seen_all = [], [], 0
    for i, mid in enumerate(mids):
        rows, seen, mins, names = match_actions(mid)
        seen_all += seen
        acts += rows
        a = pd.DataFrame(rows, columns=["player_id", "added", "goal"]) if not rows else pd.DataFrame(rows)
        g = a.groupby("player_id").agg(added=("added", "sum"), gained=("added", lambda x: x[x > 0].sum()),
                                       lost=("added", lambda x: x[x < 0].sum()), actions=("added", "size"), goals=("goal", "sum"))
        m = meta.get(mid, {})
        for pid, (nm, team, pos) in names.items():
            r = g.loc[pid] if pid in g.index else pd.Series({"added": 0.0, "gained": 0.0, "lost": 0.0, "actions": 0, "goals": 0})
            pm.append({"match": mid, "date": m.get("match_date"), "home": m.get("home_team", {}).get("home_team_name"),
                       "away": m.get("away_team", {}).get("away_team_name"), "team": team, "player_id": pid, "player": nm,
                       "position": pos, "minutes": round(mins.get(pid, 0.0), 1), **r.to_dict()})
        print(f"  {i + 1}/{len(mids)} {mid}: {len(rows)} actions valued of {seen}", flush=True)
    A = pd.DataFrame(acts)
    A.to_csv(os.path.join(OUT, "onball_actions.csv.gz"), index=False)
    P = pd.DataFrame(pm)
    P["per90"] = P["added"] / P["minutes"].clip(lower=1) * 90
    P["rank_in_match"] = P.groupby("match")["added"].rank(ascending=False, method="first").astype(int)
    P.to_csv(os.path.join(OUT, "onball_player_match.csv"), index=False)
    S = (P[P["team"] == CLUB].groupby(["player_id", "player"])
         .agg(matches=("match", "nunique"), minutes=("minutes", "sum"), added=("added", "sum"), gained=("gained", "sum"),
              lost=("lost", "sum"), actions=("actions", "sum"), goals=("goals", "sum"),
              best_in_match=("rank_in_match", lambda r: int((r == 1).sum()))).reset_index())
    S["per90"] = S["added"] / S["minutes"] * 90
    S = S.sort_values("per90", ascending=False)
    S.to_csv(os.path.join(OUT, "onball_season.csv"), index=False)
    print(f"\nvalued {len(A)} of {seen_all} open-play passes / carries / shots ({len(A) / max(seen_all, 1):.0%} had a 360 frame)")
    print("\nLeverkusen, season, players with 900+ minutes, by on-ball threat added per 90:")
    print(S[S["minutes"] >= 900][["player", "matches", "minutes", "per90", "added", "best_in_match"]].round(3).to_string(index=False))
    check(P)


def check(P):
    """Correlate with Impect per player-match (Leverkusen) and per player over the season (per 90)."""
    try:
        I = impect_table()
    except Exception as e:
        print("Impect check skipped:", repr(e))
        return
    lev = P[P["team"] == CLUB].copy()
    link = link_names(lev["player"].unique(), I["name"].unique())
    lev["name"] = lev["player"].map(link)
    J = lev.dropna(subset=["name"]).merge(I, on=["date", "name"], how="inner")
    J = J[J["minutes"] >= 20]
    cols = ["imp_bypassed", "imp_bypassed_recv", "imp_packing_xg", "imp_shot_xg", "imp_xga"]
    rows = []
    for c in cols:
        rows.append({"level": "player-match", "impect_kpi": c, "n": len(J), "spearman_added": J["added"].corr(J[c], method="spearman"),
                     "spearman_gained": J["gained"].corr(J[c], method="spearman")})
    S = J.groupby("name").agg(minutes=("minutes", "sum"), added=("added", "sum"), gained=("gained", "sum"),
                              **{c: (c, "sum") for c in cols}).reset_index()
    S = S[S["minutes"] >= 900]
    for c in cols:
        rows.append({"level": "season per 90 (900+ min)", "impect_kpi": c, "n": len(S),
                     "spearman_added": (S["added"] / S["minutes"]).corr(S[c] / S["minutes"], method="spearman"),
                     "spearman_gained": (S["gained"] / S["minutes"]).corr(S[c] / S["minutes"], method="spearman")})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "onball_vs_impect.csv"), index=False)
    print(f"\nCheck against Impect ({len(link)} of {lev['player'].nunique()} Leverkusen names linked, "
          f"{J['date'].nunique()} matches joined):")
    print(R.round(3).to_string(index=False))


# ---------------------------------------------------------------- Part 2: DFL, on-ball + off-ball (ghost)

DFL = os.path.join(ROOT, "data", "real")
FPS = 10


def dfl_tracking(m):
    """Per frame: dfl id of every slot (0-10 home, 11-21 away), period, owning team; and the attacking direction of home
    per period (+1 = towards x = 105), from where the home keeper stands."""
    d = pd.read_csv(os.path.join(DFL, "processed", f"{m}.csv.gz"),
                    usecols=["frame", "team", "player_id", "role", "x_m", "period", "ball_owner_team", "game_mode", "dfl_id"])
    d["slot"] = np.where(d["team"] == "home", d["player_id"], 11 + d["player_id"])
    ids = d.pivot_table(index="frame", columns="slot", values="dfl_id", aggfunc="first")
    fr = d.drop_duplicates("frame").set_index("frame")[["period", "ball_owner_team", "game_mode"]]
    gk = d[(d["team"] == "home") & (d["role"] == "GK")].groupby("period")["x_m"].mean()
    home_dir = {int(p): (1 if x < L / 2 else -1) for p, x in gk.items()}
    return ids.reindex(columns=range(22)), fr, home_dir


def oriented(xy, attack_home, home_dir_p):
    """Copy of positions (23, 2) turned so the attacking team plays towards x = 105; attackers and defenders slots."""
    p = xy.copy()
    sgn = home_dir_p if attack_home else -home_dir_p
    if sgn < 0:
        p[:, 0], p[:, 1] = L - p[:, 0], W - p[:, 1]
    att = list(range(11)) if attack_home else list(range(11, 22))
    dfs = list(range(11, 22)) if attack_home else list(range(11))
    return p, att, dfs


def clean(a):
    a = np.asarray(a, float).reshape(-1, 2)
    return a[~np.isnan(a[:, 0])]


def dfl_onball(m, ids, fr, home_dir):
    """Passes (and the carry after receiving) and open-play shots, valued like Part 1. Returns rows per action."""
    z = np.load(os.path.join(DFL, "processed", f"{m}_passes.npz"))
    rows = []
    n = len(z["frame"])
    for i in range(n):
        f = int(z["frame"][i])
        if f not in fr.index:
            continue
        per, who = int(fr.loc[f, "period"]), int(z["passer"][i])
        home = who < 11
        p, att, dfs = oriented(z["xy"][i], home, home_dir.get(per, 1))
        A, D = clean(p[att]), clean(p[dfs])
        start = p[22]
        if np.isnan(start[0]):
            continue
        pid = ids.loc[f, who] if f in ids.index else None
        if z["set_piece"][i]:
            continue
        before = state_value(start, A, D)
        if z["complete"][i]:
            end = z["end_xy"][i].copy()
            if (home_dir.get(per, 1) if home else -home_dir.get(per, 1)) < 0:
                end = np.array([L - end[0], W - end[1]])
            after = state_value(end, A, D)
        else:
            after = 0.0
        rows.append({"match": m, "frame": f, "dfl_id": pid, "kind": "pass", "added": after - before})
        # carry: the receiver runs with the ball until his own next pass (within 15 s, nothing else in between)
        rec = int(z["receiver"][i])
        if z["complete"][i] and rec >= 0 and i + 1 < n and int(z["passer"][i + 1]) == rec and not z["set_piece"][i + 1]:
            f2 = int(z["frame"][i + 1])
            if 0 < f2 - f <= 15 * FPS and f2 in fr.index:
                p2, att2, dfs2 = oriented(z["xy"][i + 1], home, home_dir.get(int(fr.loc[f2, "period"]), 1))
                A2, D2 = clean(p2[att2]), clean(p2[dfs2])
                end2 = z["end_xy"][i].copy()
                if (home_dir.get(per, 1) if home else -home_dir.get(per, 1)) < 0:
                    end2 = np.array([L - end2[0], W - end2[1]])
                if not np.isnan(p2[22, 0]):
                    rows.append({"match": m, "frame": f2, "dfl_id": ids.loc[f2, rec] if f2 in ids.index else None, "kind": "carry",
                                 "added": state_value(p2[22], A2, D2) - state_value(end2, A2, D2)})
    # shots: matched to the tracking frame where the shooter is nearest the ball (as value_models.extract_dfl)
    from extract_passes import load_tracking, WINDOW_S, MAX_SYNC_M, BALL
    e = pd.read_csv(os.path.join(DFL, "raw", f"{m}_events.csv.gz"))
    sh = e[(e["event_type"] == "SHOT") & (e["set_piece_type"] != "PENALTY")].copy()
    p2s = float(fr.index[fr["period"] == 2].min()) / FPS
    sh["time_s"] = pd.to_timedelta(sh["timestamp"]).dt.total_seconds() + np.where(sh["period_id"] == 2, p2s, 0.0)
    last = int(fr.index.max())
    centre = np.clip(np.round(sh["time_s"].to_numpy() * FPS).astype(int), 0, last)
    k = int(WINDOW_S * FPS)
    frames = np.unique(np.concatenate([np.clip(centre + j, 0, last) for j in range(-k, k + 1)])) if len(sh) else np.array([], int)
    goals = []
    if len(frames):
        xy, _, tid, _ = load_tracking(m, frames)
        row = {f: i for i, f in enumerate(frames)}
        for c, ev in zip(centre, sh.itertuples()):
            best = None
            for f in range(max(c - k, 0), min(c + k, last) + 1):
                i = row[f]
                where = np.flatnonzero(tid[i] == ev.player_id)
                if len(where) == 0 or np.isnan(xy[i, BALL, 0]):
                    continue
                dd = float(np.linalg.norm(xy[i, where[0]] - xy[i, BALL]))
                if best is None or dd < best[0]:
                    best = (dd, f, i, int(where[0]))
            if ev.result == "GOAL":
                goals.append({"match": m, "dfl_id": ev.player_id, "time_s": ev.time_s})
            if best is None or best[0] > MAX_SYNC_M or best[1] not in fr.index:
                continue
            _, f, i, who = best
            p, att, dfs = oriented(xy[i], who < 11, home_dir.get(int(fr.loc[f, "period"]), 1))
            A, D = clean(p[att]), clean(p[dfs])
            xg = V.xg_shot(p[who], D, None, header=ev.body_part_type == "HEAD")
            rows.append({"match": m, "frame": f, "dfl_id": ev.player_id, "kind": "shot", "added": xg - state_value(p[who], A, D)})
    # assists: the last completed open-play pass to the scorer within 10 s before the goal
    for g in goals:
        ok = (z["complete"]) & (z["time_s"] <= g["time_s"]) & (z["time_s"] >= g["time_s"] - 10)
        for i in np.flatnonzero(ok)[::-1]:
            f = int(z["frame"][i])
            rec = int(z["receiver"][i])
            if rec >= 0 and f in ids.index and ids.loc[f, rec] == g["dfl_id"]:
                g["assist"] = ids.loc[f, int(z["passer"][i])]
                break
    return rows, goals


def dfl_offball(m, ids, fr, home_dir):
    """Off-ball value of every attacker against the own_MovementGNN ghost, from the fold model that never saw match m."""
    import torch
    import train_gnn as T
    d = T.load(m)
    g = np.load(T.PATH[m] + f"_graphs{T.SUFFIX}.npz")
    net = T.Net(d["feat"].shape[2], 48, "ours")
    net.load_state_dict(torch.load(os.path.join(ROOT, "outputs", f"gnn_model_fold_{m}.pt"), map_location="cpu"))
    net.eval()
    keep = []
    (pl_err, _), _, (pl_kv, _) = T.evaluate(net, d, keep=keep)
    pred = np.concatenate([k[0] for k in keep])                            # (n, 23, 2) metres, movement to the next Key State
    ks = d["idx"]
    rows = []
    for j, k in enumerate(ks):
        f0, f1 = int(g["frame"][k]), int(g["frame"][k + 1])
        if f0 not in fr.index or f1 not in fr.index:
            continue
        own0, own1 = fr.loc[f0, "ball_owner_team"], fr.loc[f1, "ball_owner_team"]
        if own0 not in ("home", "away") or own1 != own0 or fr.loc[f1, "game_mode"] != "normal":
            continue
        home = own0 == "home"
        per = int(fr.loc[f1, "period"])
        real1 = g["xy"][k + 1].copy()
        ghost1 = g["xy"][k] + pred[j]
        p, att, dfs = oriented(real1, home, home_dir.get(per, 1))
        q, _, _ = oriented(np.where(np.isnan(ghost1), real1, ghost1), home, home_dir.get(per, 1))
        D = clean(p[dfs])
        if np.isnan(p[22, 0]):
            continue
        dball = np.linalg.norm(p[att] - p[22], axis=1)
        carrier = att[int(np.nanargmin(dball))] if not np.all(np.isnan(dball)) else -1
        for s in att:
            if s == carrier or not (g["present"][k, s] and g["present"][k + 1, s]) or np.isnan(p[s, 0]):
                continue
            others = clean(p[[a for a in att if a != s]])
            v_real = state_value(p[s], others, D)
            v_ghost = state_value(q[s], others, D)
            rows.append({"match": m, "frame": f1, "frame0": f0, "dfl_id": ids.loc[f1, s] if f1 in ids.index else None,
                         "added": v_real - v_ghost, "real": v_real, "ghost": v_ghost,
                         "rx": real1[s, 0], "ry": real1[s, 1], "gx": ghost1[s, 0], "gy": ghost1[s, 1]})   # pitch metres, as tracked
    return rows, {"match": m, "ghost_error_m": round(pl_err, 3), "keep_velocity_m": round(pl_kv, 3), "transitions": len(ks)}


def dfl():
    os.makedirs(OUT, exist_ok=True)
    matches = pd.read_csv(os.path.join(DFL, "matches.csv"))
    allp, goals_all, gerr = [], [], []
    for m in matches["match_id"]:
        ids, fr, home_dir = dfl_tracking(m)
        on, goals = dfl_onball(m, ids, fr, home_dir)
        off, ge = dfl_offball(m, ids, fr, home_dir)
        gerr.append(ge)
        goals_all += goals
        pl = pd.read_csv(os.path.join(DFL, "raw", f"{m}_players.csv"))
        mins = ids.stack().value_counts() / FPS / 60                         # minutes on the pitch from the tracking
        On = pd.DataFrame(on).groupby("dfl_id")["added"].agg(["sum", "size"]).rename(columns={"sum": "onball", "size": "actions"})
        Off = pd.DataFrame(off).groupby("dfl_id")["added"].sum().rename("offball")
        P = pl.rename(columns={"player_id": "dfl_id"})[["dfl_id", "name", "team_name", "position"]].set_index("dfl_id")
        P = P.join(mins.rename("minutes")).join(On).join(Off).fillna({"onball": 0.0, "actions": 0, "offball": 0.0})
        P = P[P["minutes"] >= 1].reset_index()
        P["match"] = m
        P["goals"] = P["dfl_id"].map(pd.Series([g["dfl_id"] for g in goals]).value_counts()).fillna(0).astype(int)
        P["assists"] = P["dfl_id"].map(pd.Series([g.get("assist") for g in goals if g.get("assist")]).value_counts()).fillna(0).astype(int)
        q = P["minutes"] >= 30
        for c in ("onball", "offball"):
            mu, sd = P.loc[q, c].mean(), P.loc[q, c].std()
            P[f"z_{c}"] = (P[c] - mu) / sd
        P["pom"] = P["z_onball"] + P["z_offball"]
        P.loc[~q, "pom"] = np.nan
        P["pom_rank"] = P["pom"].rank(ascending=False, method="first")
        allp.append(P)
        h = matches.set_index("match_id").loc[m]
        top = P.sort_values("pom", ascending=False).head(3)
        print(f"{m} {h['home']} {h['score']} {h['away']}  (ghost {ge['ghost_error_m']} m, keep-velocity {ge['keep_velocity_m']} m)")
        for r in top.itertuples():
            print(f"    {int(r.pom_rank)}. {r.name:22s} {r.team_name[:22]:22s} on {r.onball:+.3f}  off {r.offball:+.3f}  "
                  f"POM {r.pom:+.2f}  goals {r.goals} assists {r.assists}", flush=True)
    A = pd.concat(allp, ignore_index=True)
    A.to_csv(os.path.join(OUT, "dfl_player_match.csv"), index=False)
    ga = A[(A["goals"] + A["assists"]) > 0]
    rows = []
    for col, nm in (("pom", "own_PlayerOfMatch"), ("onball", "on-ball only"), ("offball", "off-ball only")):
        r = A[A["minutes"] >= 30].groupby("match")[col].rank(ascending=False)
        A[f"r_{nm}"] = r
        g = A.loc[ga.index, f"r_{nm}"].dropna()
        rows.append({"ranking": nm, "goal_or_assist_players": len(g), "median_rank": g.median(),
                     "in_top3": f"{int((g <= 3).sum())}/{len(g)}", "chance_top3": f"{3 / A[A['minutes'] >= 30].groupby('match').size().mean():.0%}"})
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(OUT, "dfl_check.csv"), index=False)
    print("\nPlayers with a goal or an assist, where each ranking puts them (of ~25 per match with 30+ min):")
    print(C.to_string(index=False))


def export(m, top=5, moments=3):
    """viewer/data/<m>_pom.json for the Three.js viewer: the ranking of match m (from dfl_player_match.csv) and the best
    moments of the top players: their biggest off-ball runs (real spot, ghost spot) and their biggest on-ball action."""
    A = pd.read_csv(os.path.join(OUT, "dfl_player_match.csv"))
    A = A[(A["match"] == m) & A["pom"].notna()].sort_values("pom", ascending=False)
    ids, fr, home_dir = dfl_tracking(m)
    on, goals = dfl_onball(m, ids, fr, home_dir)
    off, ge = dfl_offball(m, ids, fr, home_dir)
    On, Off = pd.DataFrame(on), pd.DataFrame(off)
    pl = pd.read_csv(os.path.join(DFL, "raw", f"{m}_players.csv")).set_index("player_id")
    meta = pd.read_csv(os.path.join(DFL, "matches.csv")).set_index("match_id").loc[m]
    rnd = lambda v, n=3: round(float(v), n)
    out = {"match": m, "home": meta["home"], "away": meta["away"], "score": meta["score"], "model": "own_PlayerOfMatch",
           "ghost_model": "own_MovementGNN", "ghost_error_m": ge["ghost_error_m"], "keep_velocity_m": ge["keep_velocity_m"],
           "how": "on-ball: goal chance added with the ball (passes, carries, shots); off-ball: goal chance of where he ran "
                  "minus where the own_MovementGNN ghost expected him; POM = z(on-ball) + z(off-ball) within the match",
           "players": []}
    for rank, r in enumerate(A.head(top).itertuples(), 1):
        side = "home" if pl.loc[r.dfl_id, "ground"] == "home" else "away"
        mo = []
        o = Off[Off["dfl_id"] == r.dfl_id].nlargest(moments - 1, "added")
        for x in o.itertuples():
            mo.append({"kind": "off-ball run", "t0": rnd(x.frame0 / FPS, 1), "t1": rnd(x.frame / FPS, 1), "added": rnd(x.added, 4),
                       "real": [rnd(x.rx, 2), rnd(x.ry, 2)], "ghost": [rnd(x.gx, 2), rnd(x.gy, 2)]})
        b = On[On["dfl_id"] == r.dfl_id].nlargest(1, "added")
        for x in b.itertuples():
            mo.append({"kind": f"on-ball {x.kind}", "t1": rnd(x.frame / FPS, 1), "added": rnd(x.added, 4)})
        out["players"].append({"rank": rank, "name": r.name, "team": r.team_name, "side": side,
                               "shirt": int(pl.loc[r.dfl_id, "jersey_no"]), "position": r.position, "minutes": rnd(r.minutes, 0),
                               "onball": rnd(r.onball), "offball": rnd(r.offball), "pom": rnd(r.pom, 2),
                               "goals": int(r.goals), "assists": int(r.assists), "moments": sorted(mo, key=lambda q: q["t1"])})
    p = os.path.join(ROOT, "viewer", "data", f"{m}_pom.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", p)
    for q in out["players"]:
        print(f"  {q['rank']}. {q['name']:20s} #{q['shirt']:<3d} POM {q['pom']:+.2f}  moments at "
              + ", ".join(f"{z['t1']:.0f}s ({z['kind']})" for z in q["moments"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["onball", "dfl", "export"])
    ap.add_argument("--match", default="J03WOH", help="export: the DFL match for the viewer")
    a = ap.parse_args()
    {"onball": onball, "dfl": dfl, "export": lambda: export(a.match)}[a.part]()
