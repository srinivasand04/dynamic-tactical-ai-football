"""own_TargetCoach: three coaching models for the target player (Granit Xhaka) over his CLUB league matches:
Arsenal, Premier League 2017/18 (Wyscout, 38 matches) and Leverkusen, Bundesliga 2023/24 (StatsBomb, 33 matches).
Seen from his own coach: how does he play against different opponents, at different moments of a match, and with
different teammates, and what can the coach tell him?

    c0 data      per match: his on-pitch minutes (line-ups, substitutions, red cards), the score at every minute, who
                 was on the pitch with him, and each team's style in that match (possession, PPDA, defending height)
    c1 opponent  his output per match (threat per 90, forward passes, pass losses, height) explained by the opponent's
                 style; the opponent's style comes from its OTHER matches (Premier League: its other 37; Bundesliga:
                 its other match against Leverkusen), z-scored within each league. Ridge, tested leave one match out
                 against his league average. Opponents grouped into 3 styles; his output and best zones per style.
    c2 phase     his actions by match period (15-minute blocks) x score (leading / drawing / trailing), per minute on
                 the pitch; per-action models for forward passing, pass loss and threat with minute and score vs
                 without (5 folds by match); what-if grid minute x score.
    c3 mates     with / without each teammate who was on and off the pitch with him often enough (270+ min each);
                 bootstrap over matches; a joint ridge over all of them (presences are correlated), tested leave one
                 match out; what-if line-ups (remove one teammate) and who he passes to instead.

Outputs: outputs/target_player/coach_*.csv, summary_c*.txt, coach.json; adds "coach" to viewer/data/target_player.json
Usage  : python scripts/target_coach.py c0 | c1 | c2 | c3 | viewer | all      (run from the project root)
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
import target_player as TP  # noqa: E402

OUT = TP.OUT
L, W = TP.L, TP.W
LEV, PL = TP.LEV, "Premier League"
CLUB = {LEV: "Bayer Leverkusen", PL: "Arsenal"}
HIS = {LEV: TP.T_SB, PL: TP.T_WY}
RNG = np.random.default_rng(11)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LOG = []
BUCKETS = [0, 15, 30, 45, 60, 75, 999]
BLAB = ["0-15", "15-30", "30-45+", "45-60", "60-75", "75-90+"]
STATES = ["trailing", "drawing", "leading"]


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def save(step, obj=None):
    open(os.path.join(OUT, f"summary_{step}.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    LOG.clear()
    if obj is not None:
        path = os.path.join(OUT, "coach.json")
        T = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        T[step] = obj
        json.dump(T, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


# ------------------------------------------------------------------ c0: match timelines, line-ups, style

def style_of(passes, dfa, team, opp):
    """possession = pass share; PPDA = opponent passes in its own 60% per defensive action of 'team' in that zone;
    height = mean x of the team's defensive actions (own direction)."""
    pt, po = len(passes[team]), len(passes[opp])
    own60 = sum(1 for x in passes[opp] if x < 0.6 * L)
    da = sum(1 for x in dfa[team] if x > 0.4 * L)
    return {"possession": pt / max(pt + po, 1), "ppda": own60 / max(da, 1), "height": float(np.mean(dfa[team])) if dfa[team] else np.nan}


def minute_grid(mid, league, team, his, on, team_of, goals, end):
    """One row per minute he was on the pitch: score state (his team's view) and the teammates on the pitch."""
    s, e = on[his]
    rows = []
    for k in range(int(np.floor(s)), int(np.ceil(e if e is not None else end))):
        t = k + 0.5
        if t < s or t > (e if e is not None else end):
            continue
        gf = sum(1 for tg, tm in goals if tg < t and tm == team)
        ga = sum(1 for tg, tm in goals if tg < t and tm != team)
        mates = sorted(p for p, (a, b) in on.items() if p != his and team_of[p] == team and a <= t and (b is None or b > t))
        rows.append({"league": league, "match": mid, "minute": k, "gd": gf - ga, "mates": "|".join(map(str, mates))})
    return rows


def sb_club():
    DEF = {"Duel", "Interception", "Ball Recovery", "Foul Committed", "Block"}
    ms = json.load(open(os.path.join(PM.SB, "matches_9_281.json"), encoding="utf-8"))
    tmap, info, sty, grid, names = {}, [], [], [], {}
    for m in ms:
        mid = m["match_id"]
        ev = json.load(open(os.path.join(PM.SB, "events", f"{mid}.json"), encoding="utf-8"))
        p1 = max(e["minute"] + e["second"] / 60 for e in ev if e["period"] == 1)
        T = lambda e: (e["minute"] + e["second"] / 60) if e["period"] == 1 else (p1 + e["minute"] + e["second"] / 60 - 45 if e["period"] == 2 else None)
        on, team_of, goals = {}, {}, []
        passes, dfa = {}, {}
        for e in ev:
            t, typ, tm = T(e), e["type"]["name"], e["team"]["name"]
            passes.setdefault(tm, []); dfa.setdefault(tm, [])
            if typ == "Starting XI":
                for p in e["tactics"]["lineup"]:
                    on[p["player"]["id"]] = [0.0, None]; team_of[p["player"]["id"]] = tm; names[p["player"]["id"]] = p["player"]["name"]
            if t is None:
                continue
            if typ == "Substitution":
                if e["player"]["id"] in on:
                    on[e["player"]["id"]][1] = t
                r = e["substitution"]["replacement"]
                on[r["id"]] = [t, None]; team_of[r["id"]] = tm; names[r["id"]] = r["name"]
            card = ((e.get("foul_committed") or {}).get("card") or (e.get("bad_behaviour") or {}).get("card") or {}).get("name", "")
            if card in ("Red Card", "Second Yellow") and "player" in e and e["player"]["id"] in on:
                on[e["player"]["id"]][1] = t
            if typ == "Shot" and (e["shot"].get("outcome") or {}).get("name") == "Goal":
                goals.append((t, tm))
            if typ == "Own Goal For":
                goals.append((t, tm))
            if "location" in e and typ == "Pass":
                passes[tm].append(PM.to_m(e["location"])[0])
            if "location" in e and typ in DEF:
                dfa[tm].append(PM.to_m(e["location"])[0])
            tmap[e["id"]] = t
        end = max(T(e) for e in ev if T(e) is not None)
        teams = list(passes)
        home, away = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
        for tm in teams:
            opp = [x for x in teams if x != tm][0]
            sty.append({"league": LEV, "match": mid, "team": tm, "opponent": opp, **style_of(passes, dfa, tm, opp)})
        club = CLUB[LEV]
        opp = away if home == club else home
        gf = sum(1 for _, t in goals if t == club)
        info.append({"league": LEV, "match": mid, "date": m["match_date"], "opponent": opp, "home": int(home == club),
                     "gf": gf, "ga": len(goals) - gf, "score_check": (f"{m['home_score']}-{m['away_score']}" if home == club else f"{m['away_score']}-{m['home_score']}"),
                     "played": int(TP.T_SB in on)})
        if TP.T_SB in on:
            grid += minute_grid(mid, LEV, club, TP.T_SB, on, team_of, goals, end)
    return tmap, info, sty, grid, names


def wy_club():
    teams = {t["wyId"]: t["name"] for t in json.load(open(os.path.join(BG.WY, "teams.json"), encoding="utf-8"))}
    names = {p["wyId"]: p["shortName"] for p in json.load(open(os.path.join(BG.WY, "players.json"), encoding="utf-8"))}
    ms = {m["wyId"]: m for m in json.load(open(os.path.join(BG.WY, "matches", "matches_England.json"), encoding="utf-8"))}
    ev = json.load(open(os.path.join(BG.WY, "events", "events_England.json"), encoding="utf-8"))
    by = {}
    for e in ev:
        by.setdefault(e["matchId"], []).append(e)
    del ev
    arsenal = [k for k, v in teams.items() if v == "Arsenal"][0]
    info, sty, grid, acts = [], [], [], []
    for mid, E in by.items():
        p1 = max(e["eventSec"] / 60 for e in E if e["matchPeriod"] == "1H")
        T = lambda e: e["eventSec"] / 60 if e["matchPeriod"] == "1H" else (p1 + e["eventSec"] / 60 if e["matchPeriod"] == "2H" else None)
        tids = [int(k) for k in ms[mid]["teamsData"]]
        passes, dfa, goals = {t: [] for t in tids}, {t: [] for t in tids}, []
        reds = {}
        for e in E:
            t = T(e)
            if t is None:
                continue
            tg = {x["id"] for x in e.get("tags") or []}
            x = e["positions"][0]["x"] * L / 100 if e.get("positions") else np.nan
            if e["eventName"] == "Pass":
                passes[e["teamId"]].append(x)
            if e["subEventName"] in ("Ground defending duel", "Ground loose ball duel") or 1401 in tg or e["eventName"] == "Foul":
                dfa[e["teamId"]].append(x)
            if 101 in tg and e["eventName"] in ("Shot", "Free Kick"):
                goals.append((t, e["teamId"]))
            if 102 in tg:
                goals.append((t, [z for z in tids if z != e["teamId"]][0]))
            if tg & {1701, 1703}:
                reds[e["playerId"]] = t
        for tm in tids:
            opp = [z for z in tids if z != tm][0]
            sty.append({"league": PL, "match": mid, "team": teams[tm], "opponent": teams[opp], **style_of(passes, dfa, tm, opp)})
        if arsenal not in tids:
            continue
        td = ms[mid]["teamsData"]
        on, team_of = {}, {}
        for tk, d in td.items():
            for p in d["formation"]["lineup"]:
                on[p["playerId"]] = [0.0, None]; team_of[p["playerId"]] = int(tk)
            for s in d["formation"]["substitutions"] or []:
                if not isinstance(s, dict):
                    continue
                t = s["minute"] if s["minute"] <= 45 else p1 + s["minute"] - 45
                if s["playerOut"] in on:
                    on[s["playerOut"]][1] = t
                on[s["playerIn"]] = [t, None]; team_of[s["playerIn"]] = int(tk)
        for p, t in reds.items():
            if p in on and (on[p][1] is None or on[p][1] > t):
                on[p][1] = t
        end = max(T(e) for e in E if T(e) is not None)
        opp = [z for z in tids if z != arsenal][0]
        gf = sum(1 for _, t in goals if t == arsenal)
        sc = {int(k): v["score"] for k, v in td.items()}
        info.append({"league": PL, "match": mid, "date": ms[mid]["dateutc"][:10], "opponent": teams[opp],
                     "home": int(td[str(arsenal)]["side"] == "home"), "gf": gf, "ga": len(goals) - gf,
                     "score_check": f"{sc[arsenal]}-{sc[opp]}", "played": int(TP.T_WY in on)})
        if TP.T_WY not in on:
            continue
        grid += minute_grid(mid, PL, arsenal, TP.T_WY, on, team_of, goals, end)
        # his actions, with the next team-mate who touches the ball as receiver
        for i, e in enumerate(E):
            if e["playerId"] != TP.T_WY:
                continue
            t = T(e)
            if t is None:
                continue
            tg = {x["id"] for x in e.get("tags") or []}
            ev_, sub = e["eventName"], e["subEventName"]
            pos = e.get("positions") or []
            if len(pos) < 1:
                continue
            p1_ = pos[1] if len(pos) > 1 else {"x": np.nan, "y": np.nan}
            kind = ("Pass" if ev_ == "Pass" and sub != "Hand pass" else "Shot" if ev_ == "Shot" else "Carry" if sub == "Acceleration" else None)
            if kind is None:
                continue
            rec = ""
            if kind == "Pass" and 1801 in tg:
                for e2 in E[i + 1:i + 4]:
                    if e2["teamId"] == e["teamId"] and e2["playerId"] != TP.T_WY:
                        rec = names.get(e2["playerId"], str(e2["playerId"]))
                        break
            acts.append({"league": PL, "match": mid, "t": t, "kind": kind, "setp": 0, "x": pos[0]["x"] * L / 100,
                         "y": W - pos[0]["y"] * W / 100, "ex": p1_["x"] * L / 100, "ey": W - p1_["y"] * W / 100,
                         "ok": int(101 in tg) if kind == "Shot" else int(1801 in tg) if kind == "Pass" else 1,
                         "left": int(401 in tg), "recipient": rec})
    return info, sty, grid, acts, names


def step_c0():
    t0 = time.time()
    tmap, i1, s1, g1, n1 = sb_club()
    say(f"Bundesliga 2023/24 (StatsBomb): {len(i1)} Leverkusen matches, {NAME_HE()} on the pitch in {sum(r['played'] for r in i1)}, "
        f"{len(g1)} minutes ({time.time() - t0:.0f}s)")
    i2, s2, g2, a2, n2 = wy_club()
    say(f"Premier League 2017/18 (Wyscout): {len(i2)} Arsenal matches, on the pitch in {sum(r['played'] for r in i2)}, "
        f"{len(g2)} minutes; style of all {len(s2) // 2} league matches ({time.time() - t0:.0f}s)")
    I = pd.DataFrame(i1 + i2)
    bad = I[I.apply(lambda r: f"{r.gf}-{r.ga}" != r.score_check, axis=1)]
    say(f"  goal timelines vs official scores: {len(I) - len(bad)} of {len(I)} match"
        + ("" if bad.empty else "; differ: " + ", ".join(f"{r.opponent} {r.score_check}" for r in bad.itertuples())))
    A = TP.sb_actions()
    A = A[(A["player_id"] == TP.T_SB) & (A["source"] == LEV)].copy()
    A["t"] = A["id"].map(tmap)
    A["league"], A["left"] = LEV, (A["body"] == "Left Foot").astype(int)
    A = A[A["kind"].isin(["Pass", "Carry", "Shot"])][["league", "match", "t", "kind", "setp", "x", "y", "ex", "ey", "ok", "left", "recipient"]]
    Ac = pd.concat([A, pd.DataFrame(a2)], ignore_index=True)
    Ac = TP.add_gain(Ac, np.load(os.path.join(OUT, "xt_grid.npy")))
    Ac["minute"] = np.floor(Ac["t"]).astype(int)
    I.to_csv(os.path.join(OUT, "coach_matches.csv"), index=False)
    pd.DataFrame(s1 + s2).to_csv(os.path.join(OUT, "coach_style.csv"), index=False)
    pd.DataFrame(g1 + g2).to_csv(os.path.join(OUT, "coach_minutes.csv.gz"), index=False)
    Ac.to_csv(os.path.join(OUT, "coach_actions.csv.gz"), index=False)
    nm = {**{str(k): v for k, v in n1.items()}, **{str(k): v for k, v in n2.items()}}
    json.dump(nm, open(os.path.join(OUT, "coach_names.json"), "w", encoding="utf-8"), ensure_ascii=False)
    say(f"  his club actions: {len(Ac)} (passes {int((Ac.kind == 'Pass').sum())}, carries {int((Ac.kind == 'Carry').sum())})")
    save("c0")


def NAME_HE():
    return TP.NAME


def load_all():
    I = pd.read_csv(os.path.join(OUT, "coach_matches.csv"))
    S = pd.read_csv(os.path.join(OUT, "coach_style.csv"))
    G = pd.read_csv(os.path.join(OUT, "coach_minutes.csv.gz"), keep_default_na=False)
    A = pd.read_csv(os.path.join(OUT, "coach_actions.csv.gz"), keep_default_na=False, na_values=[""])
    A["recipient"] = A["recipient"].fillna("")
    nm = json.load(open(os.path.join(OUT, "coach_names.json"), encoding="utf-8"))
    return I, S, G, A, nm


def metrics(a, minutes):
    """His output over a set of actions and minutes on the pitch."""
    p = a[(a["kind"] == "Pass") & (a["setp"] == 0)]
    k = 90 / max(minutes, 1e-9)
    return {"minutes": float(minutes), "xt90": float(a["gain"].sum() * k), "touches90": float(len(a) * k),
            "fwd": float((p["ex"] - p["x"] >= 5).mean()) if len(p) else np.nan,
            "loss": float(1 - p["ok"].mean()) if len(p) else np.nan,
            "long": float((np.hypot(p["ex"] - p["x"], p["ey"] - p["y"]) >= 30).mean()) if len(p) else np.nan,
            "height": float(a["x"].mean()) if len(a) else np.nan}


MET = ["xt90", "fwd", "loss", "height"]
MLAB = {"xt90": "threat per 90 (xT)", "fwd": "forward pass share", "loss": "pass loss rate", "height": "mean height (m)",
        "touches90": "actions per 90", "long": "long pass share"}


# ------------------------------------------------------------------ c1 opponent matchup

def step_c1():
    from sklearn.cluster import KMeans
    I, S, G, A, nm = load_all()
    # opponent style from its OTHER matches
    rows = []
    for r in I[I["played"] == 1].itertuples():
        s = S[(S["league"] == r.league) & (S["team"] == r.opponent) & (S["match"] != r.match)]
        if r.league == LEV:
            s = s[s["opponent"] == CLUB[LEV]]                                  # only Leverkusen matches in this data
        if s.empty:
            s = S[(S["league"] == r.league) & (S["team"] == r.opponent) & (S["match"] == r.match)]
        rows.append({"league": r.league, "match": r.match, "opponent": r.opponent, "home": r.home,
                     **{f"opp_{c}": s[c].mean() for c in ("possession", "ppda", "height")}})
    O = pd.DataFrame(rows)
    for c in ("possession", "ppda", "height"):
        z = O.groupby("league")[f"opp_{c}"].transform(lambda v: (v - v.mean()) / v.std())
        O[f"z_{c}"] = z
    O["z_press"] = -O["z_ppda"]                                                # low PPDA = intense press
    mins = G.groupby("match").size()
    M = []
    for r in O.itertuples():
        a = A[A["match"] == r.match]
        M.append({**r._asdict(), **metrics(a, mins.get(r.match, 0))})
    M = pd.DataFrame(M).drop(columns=["Index"])
    M = M[M["minutes"] >= 30].reset_index(drop=True)
    F = ["z_possession", "z_press", "z_height", "home"]
    say(f"C1. Opponent matchup: {len(M)} club matches with 30+ min ({(M.league == PL).sum()} Arsenal, {(M.league == LEV).sum()} Leverkusen), "
        f"{M['opponent'].nunique()} opponents; opponent style from its other matches, z-scored within the league")
    res, coefs = {}, {}
    w = M["minutes"].to_numpy()
    for y in MET:
        Y = M[y].to_numpy(float)
        X = M[F].to_numpy(float)
        lg = (M["league"] == PL).to_numpy()
        pred, base = np.zeros(len(M)), np.zeros(len(M))
        for i in range(len(M)):
            tr = np.arange(len(M)) != i
            mu = {g: np.average(Y[tr & (lg == g)], weights=w[tr & (lg == g)]) for g in (True, False)}
            yc = Y[tr] - np.array([mu[g] for g in lg[tr]])
            Xc = X[tr] - X[tr].mean(0)
            B = np.linalg.solve(Xc.T @ (Xc * w[tr, None]) + 5.0 * np.eye(len(F)), Xc.T @ (yc * w[tr]))
            base[i] = mu[lg[i]]
            pred[i] = mu[lg[i]] + (X[i] - X[tr].mean(0)) @ B
        e0 = np.sqrt(np.average((Y - base) ** 2, weights=w))
        e1 = np.sqrt(np.average((Y - pred) ** 2, weights=w))
        mu = {g: np.average(Y[lg == g], weights=w[lg == g]) for g in (True, False)}
        yc = Y - np.array([mu[g] for g in lg])
        Xc = X - X.mean(0)
        B = np.linalg.solve(Xc.T @ (Xc * w[:, None]) + 5.0 * np.eye(len(F)), Xc.T @ (yc * w))
        bs = []
        for _ in range(500):
            k = RNG.integers(0, len(M), len(M))
            bs.append(np.linalg.solve(Xc[k].T @ (Xc[k] * w[k, None]) + 5.0 * np.eye(len(F)), Xc[k].T @ (yc[k] * w[k])))
        bs = np.array(bs)
        coefs[y] = {f: {"coef": float(B[j]), "lo": float(np.percentile(bs[:, j], 5)), "hi": float(np.percentile(bs[:, j], 95))}
                    for j, f in enumerate(F)}
        res[y] = {"rmse_base": float(e0), "rmse_model": float(e1), "gain_pct": float(100 * (1 - e1 / e0))}
        say(f"  {MLAB[y]:20s} leave-one-match-out error: his average {e0:.4f} -> with opponent style {e1:.4f} "
            f"({100 * (1 - e1 / e0):+.1f}%) | per 1 SD: " + "  ".join(
                f"{f[2:] if f.startswith('z_') else f} {B[j]:+.4f} [{np.percentile(bs[:, j], 5):+.4f}, {np.percentile(bs[:, j], 95):+.4f}]"
                for j, f in enumerate(F)))
    # three opponent styles
    Z = M[["z_possession", "z_press", "z_height"]].to_numpy()
    km = KMeans(3, n_init=20, random_state=0).fit(Z)
    M["style"] = km.labels_
    styles = []
    lev_mu = {g: {y: np.average(M.loc[M.league == g, y], weights=M.loc[M.league == g, "minutes"]) for y in MET} for g in (PL, LEV)}
    for c in range(3):
        cz = km.cluster_centers_[c]
        lab = (("high press" if cz[1] > 0.4 else "low press" if cz[1] < -0.4 else "medium press") + ", " +
               ("more of the ball" if cz[0] > 0.4 else "less of the ball" if cz[0] < -0.4 else "even ball") + ", " +
               ("high line" if cz[2] > 0.4 else "deep line" if cz[2] < -0.4 else "mid line"))
        g = M[M["style"] == c]
        a = A[A["match"].isin(g["match"])]
        mt = metrics(a, g["minutes"].sum())
        rel = {y: float(np.average([r[y] - lev_mu[r.league][y] for _, r in g.iterrows()], weights=g["minutes"])) for y in MET}
        zh = np.bincount(TP.zone(a["x"], a["y"]), weights=a["gain"], minlength=24) * 90 / g["minutes"].sum()
        top = np.argsort(-zh)[:3]
        styles.append({"style": c, "label": lab, "centre": cz.tolist(), "matches": len(g), "opponents": sorted(g["opponent"].unique().tolist()),
                       "metrics": mt, "rel": rel, "zones": [{"zone": TP.zone_name(z), "xt90": float(zh[z])} for z in top],
                       "zones_all": zh.tolist()})
        say(f"  style {c}: {lab:45s} {len(g):2d} matches  threat/90 {rel['xt90']:+.3f}  forward {100 * rel['fwd']:+.1f} pts  loss "
            f"{100 * rel['loss']:+.1f} pts  height {rel['height']:+.1f} m (vs his league average)  best zone {TP.zone_name(top[0])}")
        say(f"      e.g. {', '.join(g['opponent'].value_counts().index[:6])}")
    M.to_csv(os.path.join(OUT, "coach_c1_matches.csv"), index=False)
    best = max(styles, key=lambda s: s["rel"]["xt90"])
    worst = min(styles, key=lambda s: s["rel"]["xt90"])
    say(f"  -> best against: {best['label']}; hardest: {worst['label']}")
    save("c1", {"n": len(M), "tests": res, "coefs": coefs, "styles": styles, "best": best["label"], "worst": worst["label"],
                "matches": M[["league", "opponent", "home", "z_possession", "z_press", "z_height", "minutes", "xt90", "fwd", "loss", "style"]].to_dict("records")})


# ------------------------------------------------------------------ c2 match phase x score

def state_of(gd):
    return np.where(gd > 0, "leading", np.where(gd < 0, "trailing", "drawing"))


def phase_X(t, gd, xn, lg, full=True):
    tn = np.clip(np.asarray(t, float), 0, 100) / 90
    ld, tr = (np.asarray(gd) > 0).astype(float), (np.asarray(gd) < 0).astype(float)
    cols = [np.ones(len(tn)), xn, lg]
    if full:
        cols += [tn, tn ** 2, ld, tr, ld * tn, tr * tn]
    return np.column_stack(cols)


def step_c2():
    I, S, G, A, nm = load_all()
    G["bucket"] = pd.cut(G["minute"], BUCKETS, right=False, labels=BLAB)
    G["state"] = state_of(G["gd"])
    gd_at = {(m, k): d for m, k, d in zip(G["match"], G["minute"], G["gd"])}
    A["gd"] = [gd_at.get((m, k), np.nan) for m, k in zip(A["match"], A["minute"])]
    A = A[A["gd"].notna()].copy()
    A["bucket"] = pd.cut(A["minute"], BUCKETS, right=False, labels=BLAB)
    A["state"] = state_of(A["gd"])
    say(f"C2. Match phase x score: {len(G)} minutes on the pitch, {len(A)} actions, {G['match'].nunique()} club matches; "
        f"minutes leading {(G.state == 'leading').mean() * 100:.0f}%, drawing {(G.state == 'drawing').mean() * 100:.0f}%, "
        f"trailing {(G.state == 'trailing').mean() * 100:.0f}%")
    tab = {}
    for key in ("bucket", "state"):
        rows = []
        for v, g in G.groupby(key, observed=True):
            rows.append({key: str(v), **metrics(A[A[key] == v], len(g))})
        tab[key] = rows
        say(f"  by {key}:")
        for r in rows:
            say(f"    {r[key]:9s} {r['minutes']:6.0f} min  threat/90 {r['xt90']:.3f}  actions/90 {r['touches90']:5.1f}  forward "
                f"{100 * r['fwd']:4.1f}%  loss {100 * r['loss']:4.1f}%  long {100 * r['long']:4.1f}%  height {r['height']:4.1f} m")
    cross = []
    for (b, s), g in G.groupby(["bucket", "state"], observed=True):
        if len(g) >= 60:
            cross.append({"bucket": str(b), "state": s, **metrics(A[(A.bucket == b) & (A.state == s)], len(g))})
    # per-action models, 5 folds by match: with vs without minute + score
    p = A[(A["kind"] == "Pass") & (A["setp"] == 0)].reset_index(drop=True)
    xn = p["x"].to_numpy() / TP.L
    lg = (p["league"] == PL).to_numpy(float)
    ms = p["match"].unique()
    fold = dict(zip(RNG.permutation(ms), np.arange(len(ms)) % 5))
    f = p["match"].map(fold).to_numpy()
    tests = {}
    for target, yv in (("forward pass", (p["ex"] - p["x"] >= 5).astype(int).to_numpy()), ("pass loss", (1 - p["ok"]).astype(int).to_numpy())):
        l0, l1 = [], []
        for k in range(5):
            tr, te = f != k, f == k
            for full, out in ((False, l0), (True, l1)):
                Xa = phase_X(p["minute"], p["gd"], xn, lg, full)
                B = P.fit_mn(Xa[tr], yv[tr], 2, 1.0)
                out.append(P.nll_rows(P.softmax(Xa[te] @ B), yv[te]))
        a0, a1 = np.concatenate(l0).mean(), np.concatenate(l1).mean()
        tests[target] = {"base": float(a0), "model": float(a1), "gain_pct": float(100 * (1 - a1 / a0))}
        say(f"  held out (5 folds by match), {target:12s}: log-loss location only {a0:.4f} -> + minute and score {a1:.4f} "
            f"({100 * (1 - a1 / a0):+.2f}%)")
    # what-if grid: minute x score -> forward share, loss, threat per 90 (exposure from minutes in that state/period)
    yf = (p["ex"] - p["x"] >= 5).astype(int).to_numpy()
    yl = (1 - p["ok"]).astype(int).to_numpy()
    Xa = phase_X(p["minute"], p["gd"], xn, lg)
    Bf, Bl = P.fit_mn(Xa, yf, 2, 1.0), P.fit_mn(Xa, yl, 2, 1.0)
    # threat per 90 as a smooth function: Poisson-style ridge on per-minute threat
    gm = G.copy()
    gm["xt"] = gm.set_index(["match", "minute"]).index.map(A.groupby(["match", "minute"])["gain"].sum()).fillna(0).to_numpy()
    Xg = phase_X(gm["minute"], gm["gd"], np.full(len(gm), 0.5), (gm["league"] == PL).to_numpy(float))
    Bg = P.ridge(Xg, gm["xt"].to_numpy(), 30.0)
    grid = []
    xm = float(xn.mean())
    for t in (10, 30, 50, 70, 85):
        for s, gd in (("trailing", -1), ("drawing", 0), ("leading", 1)):
            Xq = phase_X([t], [gd], [xm], [0.5])
            Xq2 = phase_X([t], [gd], [0.5], [0.5])
            grid.append({"minute": t, "state": s, "fwd": float(P.softmax(Xq @ Bf)[0, 1]), "loss": float(P.softmax(Xq @ Bl)[0, 1]),
                         "xt90": float(90 * (Xq2 @ Bg)[0])})
    say("  what-if (his model, average club context):  minute  state     forward  loss   threat/90")
    for g_ in grid:
        say(f"    {g_['minute']:3d}  {g_['state']:9s} {100 * g_['fwd']:5.1f}%  {100 * g_['loss']:4.1f}%  {g_['xt90']:.3f}")
    early = [r for r in tab["bucket"] if r["bucket"] == "0-15"][0]
    late = [r for r in tab["bucket"] if r["bucket"] == "75-90+"][0]
    pick = lambda m, s: [g_ for g_ in grid if g_["minute"] == m and g_["state"] == s][0]
    tips = [
        f"Last 15 minutes vs first 15: threat per 90 {early['xt90']:.3f} -> {late['xt90']:.3f}, pass loss {100 * early['loss']:.1f}% -> "
        f"{100 * late['loss']:.1f}%, height {early['height']:.0f} -> {late['height']:.0f} m.",
        f"Minute 70, leading: forward passes {100 * pick(70, 'leading')['fwd']:.0f}% vs drawing {100 * pick(70, 'drawing')['fwd']:.0f}% "
        f"and trailing {100 * pick(70, 'trailing')['fwd']:.0f}%.",
        f"Minute 85, trailing: pass loss {100 * pick(85, 'trailing')['loss']:.1f}% vs {100 * pick(85, 'drawing')['loss']:.1f}% drawing; "
        f"threat per 90 {pick(85, 'trailing')['xt90']:.3f} vs {pick(85, 'drawing')['xt90']:.3f}."]
    for t_ in tips:
        say("  -> " + t_)
    pd.DataFrame(cross).to_csv(os.path.join(OUT, "coach_c2_cross.csv"), index=False)
    save("c2", {"minutes": int(len(G)), "share": {s: float((G.state == s).mean()) for s in STATES}, "by_bucket": tab["bucket"],
                "by_state": tab["state"], "cross": cross, "tests": tests, "whatif": grid, "tips": tips})


# ------------------------------------------------------------------ c3 teammates with / without

def step_c3():
    I, S, G, A, nm = load_all()
    G["mset"] = G["mates"].str.split("|").apply(lambda v: set(x for x in v if x))
    mates_at = {(m, k): s for m, k, s in zip(G["match"], G["minute"], G["mset"])}
    A["mset"] = [mates_at.get((m, k), set()) for m, k in zip(A["match"], A["minute"])]
    A = A[A["mset"].apply(len) > 0].copy()
    key = []
    for lg in (PL, LEV):
        g = G[G["league"] == lg]
        tot = len(g)
        cnt = {}
        for s in g["mset"]:
            for x in s:
                cnt[x] = cnt.get(x, 0) + 1
        for x, c in cnt.items():
            if c >= 270 and tot - c >= 270:
                key.append((lg, x, c, tot - c))
    say(f"C3. Teammates: {len(key)} teammates with 270+ min both with and without him on the pitch together")
    rows = []
    for lg, x, wmin, omin in key:
        g = G[G["league"] == lg]
        a = A[A["league"] == lg]
        w_ = g["mset"].apply(lambda s: x in s).to_numpy()
        aw = a["mset"].apply(lambda s: x in s).to_numpy()
        mw, mo = metrics(a[aw], w_.sum()), metrics(a[~aw], (~w_).sum())
        ms = g["match"].unique()
        bs = []
        gm = {m: (g["match"] == m).to_numpy() for m in ms}
        am = {m: (a["match"] == m).to_numpy() for m in ms}
        for _ in range(400):
            kk = RNG.choice(ms, len(ms))
            gw = np.concatenate([w_[gm[m]] for m in kk]); ga_ = np.concatenate([aw[am[m]] for m in kk])
            gain = np.concatenate([a["gain"].to_numpy()[am[m]] for m in kk])
            if gw.sum() and (~gw).sum():
                bs.append(90 * gain[ga_].sum() / gw.sum() - 90 * gain[~ga_].sum() / (~gw).sum())
        rec_w = a[aw & (a["recipient"] != "")]["recipient"].value_counts(normalize=True).head(3)
        rec_o = a[~aw & (a["recipient"] != "")]["recipient"].value_counts(normalize=True).head(3)
        zw = np.bincount(TP.zone(a[aw]["x"], a[aw]["y"]), weights=a[aw]["gain"], minlength=24) * 90 / max(w_.sum(), 1)
        zo = np.bincount(TP.zone(a[~aw]["x"], a[~aw]["y"]), weights=a[~aw]["gain"], minlength=24) * 90 / max((~w_).sum(), 1)
        rows.append({"league": lg, "id": x, "name": nm.get(x, x), "with_min": int(wmin), "without_min": int(omin),
                     **{f"with_{k}": v for k, v in mw.items()}, **{f"without_{k}": v for k, v in mo.items()},
                     "diff_xt90": mw["xt90"] - mo["xt90"], "ci": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
                     "rec_with": [[k, float(v)] for k, v in rec_w.items()], "rec_without": [[k, float(v)] for k, v in rec_o.items()],
                     "zone_with": TP.zone_name(int(np.argmax(zw))), "zone_without": TP.zone_name(int(np.argmax(zo)))})
    R = pd.DataFrame(rows).sort_values("diff_xt90", ascending=False)
    for r in R.itertuples():
        say(f"  {r.name:24s} ({'Arsenal' if r.league == PL else 'Leverkusen'}) with {r.with_min:4d} / without {r.without_min:4d} min: "
            f"threat/90 {r.with_xt90:.3f} vs {r.without_xt90:.3f} ({r.diff_xt90:+.3f}, 90% {r.ci[0]:+.3f} .. {r.ci[1]:+.3f})  "
            f"forward {100 * r.with_fwd:.0f}% vs {100 * r.without_fwd:.0f}%  loss {100 * r.with_loss:.1f}% vs {100 * r.without_loss:.1f}%  "
            f"height {r.with_height:.0f} vs {r.without_height:.0f} m")
    # joint ridge on segments (match x set of key teammates on the pitch): his threat per minute
    kid = [(lg, x) for lg, x, _, _ in key]
    seg = []
    for (lg, m), g in G.groupby(["league", "match"]):
        sig = g["mset"].apply(lambda s: tuple(int((l2 == lg) and (x in s)) for l2, x in kid))
        for sg, gg in g.groupby(sig):
            a = A[(A["match"] == m) & A["minute"].isin(gg["minute"])]
            seg.append({"league": lg, "match": m, "minutes": len(gg), "xt": float(a["gain"].sum()), **{f"k{j}": v for j, v in enumerate(sg)}})
    Sg = pd.DataFrame(seg)
    Sg = Sg[Sg["minutes"] >= 5].reset_index(drop=True)
    Xk = Sg[[f"k{j}" for j in range(len(kid))]].to_numpy(float)
    lg_ = (Sg["league"] == PL).to_numpy(float)
    for g_ in (0.0, 1.0):                                                       # centre presences within each league
        k_ = lg_ == g_
        Xk[k_] -= np.average(Xk[k_], axis=0, weights=Sg["minutes"].to_numpy(float)[k_])
    y = 90 * Sg["xt"].to_numpy() / Sg["minutes"].to_numpy()
    w = Sg["minutes"].to_numpy(float)

    def fit(idx, lam):
        Xd = np.column_stack([lg_[idx], 1 - lg_[idx], Xk[idx]])
        pen = np.diag([1e-6, 1e-6] + [lam * w[idx].sum()] * len(kid))
        return np.linalg.solve(Xd.T @ (Xd * w[idx, None]) + pen, Xd.T @ (y[idx] * w[idx]))

    def loo(lam):
        pr, bs_ = np.zeros(len(Sg)), np.zeros(len(Sg))
        for m in Sg["match"].unique():
            te = (Sg["match"] == m).to_numpy()
            B = fit(np.flatnonzero(~te), lam)
            pr[te] = np.column_stack([lg_[te], 1 - lg_[te], Xk[te]]) @ B
            tr = ~te
            bs_[te] = np.where(lg_[te] == 1, np.average(y[tr & (lg_ == 1)], weights=w[tr & (lg_ == 1)]),
                               np.average(y[tr & (lg_ == 0)], weights=w[tr & (lg_ == 0)]))
        return np.sqrt(np.average((y - bs_) ** 2, weights=w)), np.sqrt(np.average((y - pr) ** 2, weights=w))
    lams = [0.01, 0.1, 1.0, 10.0]
    errs = {lam: loo(lam) for lam in lams}
    say("  joint model penalty -> leave-one-match-out error: " + "  ".join(f"{lam}: {e[1]:.4f}" for lam, e in errs.items())
        + f"  (his average {errs[lams[0]][0]:.4f}); the penalty is chosen on the same test, so read the best one as an upper bound")
    lam = min(lams, key=lambda l_: errs[l_][1])
    e0, e1 = errs[lam]
    B = fit(np.arange(len(Sg)), lam)
    say(f"  joint model on {len(Sg)} segments: leave-one-match-out error of threat/90 {e0:.4f} (his average) -> {e1:.4f} "
        f"({100 * (1 - e1 / e0):+.1f}%)")
    joint = {f"{nm.get(x, x)}": float(B[2 + j]) for j, (lg, x) in enumerate(kid)}
    say("  joint effect on his threat/90 (all teammates together): " + "; ".join(f"{k} {v:+.3f}" for k, v in sorted(joint.items(), key=lambda kv: -kv[1])))
    R["joint"] = R["name"].map(joint)
    R.drop(columns=["rec_with", "rec_without"]).to_csv(os.path.join(OUT, "coach_c3_mates.csv"), index=False)
    R["clear"] = R["ci"].apply(lambda c: c[0] > 0 or c[1] < 0)
    up, dn = R[R.clear & (R.diff_xt90 > 0)], R[R.clear & (R.diff_xt90 < 0)]
    say(f"  clear with/without differences (90% interval excludes 0): {len(up)} up, {len(dn)} down of {len(R)}; "
        f"{0.1 * len(R):.0f} expected by chance alone")
    tips = []
    for _, r in up.iterrows():
        tips.append(f"With {r['name']} on the pitch his threat per 90 is {r['with_xt90']:.3f} vs {r['without_xt90']:.3f} without; "
                    f"without him his best zone moves from {r['zone_with']} to {r['zone_without']}.")
    for _, r in dn.iterrows():
        tips.append(f"With {r['name']} his threat per 90 is lower ({r['with_xt90']:.3f} vs {r['without_xt90']:.3f}); forward passes "
                    f"{100 * r['with_fwd']:.0f}% vs {100 * r['without_fwd']:.0f}%, height {r['with_height']:.0f} vs {r['without_height']:.0f} m.")
    tips.append("Careful: with / without also splits the season (arrivals, injuries, rotation against weaker opponents), "
                "so these are leads to check on video, not proof.")
    for t_ in tips:
        say("  -> " + t_)
    save("c3", {"mates": R.to_dict("records"), "tests": {"rmse_base": float(e0), "rmse_model": float(e1), "gain_pct": float(100 * (1 - e1 / e0)), "lam": lam},
                "joint": joint, "tips": tips, "segments": len(Sg)})


# ------------------------------------------------------------------ viewer

def step_viewer():
    C = json.load(open(os.path.join(OUT, "coach.json"), encoding="utf-8"))
    c1, c2, c3 = C["c1"], C["c2"], C["c3"]
    best = max(c1["styles"], key=lambda s: s["rel"]["xt90"])
    worst = min(c1["styles"], key=lambda s: s["rel"]["xt90"])
    t1 = c1["tests"]["xt90"]["gain_pct"]
    clear = [m for m in c3["mates"] if m.get("clear")]
    plan = [
        {"model": "C1 opponent", "text": f"Against {best['label']} sides he is at his best (threat {best['rel']['xt90']:+.3f} per 90 vs his "
         f"average, mostly from {best['zones'][0]['zone']}); against {worst['label']} sides he drops ({worst['rel']['xt90']:+.3f}, "
         f"plays {abs(worst['rel']['height']):.0f} m deeper, loses {100 * worst['rel']['loss']:+.1f} pts more passes). Coach: against a "
         f"high press let him drop and play the first pass safe, the threat must come from elsewhere. Opponent style does not predict a "
         f"single match better than his average ({t1:+.0f}%), so this is a tendency, not a forecast."},
        {"model": "C2 phase", "text": c2["tips"][0] + " " + c2["tips"][1] + " Coach: when leading he stops playing forward "
         "(his threat per 90 drops by about a third); if you need a second goal, tell him to keep breaking lines; when trailing he already goes forward."},
        {"model": "C3 teammates", "text": f"Only {len(clear)} of {len(c3['mates'])} teammates change his output clearly (about "
         f"{round(0.1 * len(c3['mates']))} expected by chance) and a joint model predicts no better than his average: "
         f"his game does not depend on who plays next to him. Leads to check: " + "; ".join(t.rstrip(".") for t in c3["tips"][:-1]) + "."},
    ]
    C["plan"] = plan
    path = TP.VIEW
    T = json.load(open(path, encoding="utf-8"))
    T["coach"] = C
    json.dump(T, open(path, "w", encoding="utf-8"), ensure_ascii=False, default=float)
    for p in plan:
        say(f"  {p['model']:14s} {p['text']}")
    say(f"viewer data: {path} ({os.path.getsize(path) // 1024} KB)")
    save("cviewer")


STEPS = {"c0": step_c0, "c1": step_c1, "c2": step_c2, "c3": step_c3, "viewer": step_viewer}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    a = ap.parse_args()
    for s in (list(STEPS) if a.step == "all" else [a.step]):
        t0 = time.time()
        STEPS[s]()
        print(f"[{s} done in {time.time() - t0:.0f}s]", flush=True)
