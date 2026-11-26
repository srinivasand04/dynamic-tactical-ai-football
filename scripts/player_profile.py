"""own_PlayerProfile (Track A): the profile of one player learned from all his matches: role and zones, on-ball value,
passing, shooting, ball security, defending and pressing, with strengths and weaknesses backed by numbers and a
90% interval. Plan: the "Player profile model: build plan" doc (phases 0-4 and 6, existing models only).

Data: StatsBomb open data, Bayer Leverkusen 2023/24 (34 matches, events + 360 freeze frames). Every player of both
teams is profiled, so a Leverkusen player is compared with all players of his role who faced Leverkusen.
Impect open data (same 34 matches, research use only, credit Impect) is the external check.

Steps (one command each, or `all`):
  actions   one row per action of every player: type, where, under pressure, success, own_Threat (goal units) before
            and after (player_of_match.state_value, own_xG for shots), StatsBomb xG / xA links
  metrics   ~30 metrics per player-match (players with 30+ minutes), per 90; defensive counts adjusted for the
            opponent's share of the ball
  profile   pooled across all of a player's matches with empirical-Bayes shrinkage towards his role (normal-normal for
            per-90 numbers, beta-binomial for rates), a 90% interval from a bootstrap over his matches, percentile in
            his role; strengths (percentile >= 80 and interval above the role median) and weaknesses (<= 20 and interval
            below it), everything else 'unclear'
  check     split-half reliability of every metric (odd vs even matches), leave-one-match-out stability, Spearman
            against Impect KPIs, and the eye test (top players of each metric)

Outputs: outputs/player_profile/
    actions_leverkusen.csv.gz, player_match_leverkusen.csv, roles.csv
    profiles_leverkusen.csv          every player x metric: value, interval, percentile, label
    profile_<player>.json            one file per Leverkusen player with 900+ minutes
    reliability.csv, stability.csv, impect_check.csv

Usage:  .venv-analysis\\Scripts\\python.exe scripts\\player_profile.py all
        .venv-analysis\\Scripts\\python.exe scripts\\player_profile.py show --player Wirtz
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
import player_of_match as PM
import value_models as V

ROOT = PM.ROOT
SB, IMP = PM.SB, PM.IMP
OUT = os.path.join(ROOT, "outputs", "player_profile")
CLUB = "Bayer Leverkusen"
L, W = 105.0, 68.0
RNG = np.random.default_rng(0)

# StatsBomb position -> 8 role groups
ROLE = {"Goalkeeper": "GK", "Right Center Back": "CB", "Left Center Back": "CB", "Center Back": "CB",
        "Right Back": "FB", "Left Back": "FB", "Right Wing Back": "FB", "Left Wing Back": "FB",
        "Right Defensive Midfield": "DM", "Left Defensive Midfield": "DM", "Center Defensive Midfield": "DM",
        "Right Center Midfield": "CM", "Left Center Midfield": "CM", "Center Midfield": "CM",
        "Right Midfield": "W", "Left Midfield": "W", "Right Wing": "W", "Left Wing": "W",
        "Right Attacking Midfield": "AM", "Left Attacking Midfield": "AM", "Center Attacking Midfield": "AM",
        "Right Center Forward": "ST", "Left Center Forward": "ST", "Center Forward": "ST", "Secondary Striker": "ST"}

# metric -> (area, how it is pooled, higher is better, label for the report)
#   'p90' = count per 90 minutes;  ('rate', num, den) = num / den with beta-binomial shrinkage
METRICS = {
    "actions_p90": ("Involvement", "p90", True, "on-ball actions per 90"),
    "receipts_p90": ("Involvement", "p90", True, "balls received per 90"),
    "final_third_share": ("Involvement", ("rate", "final_third_n", "actions_n"), True, "share of actions in the final third"),
    "threat_added_p90": ("Progression and creation", "p90", True, "goal chance added per 90 (own_Threat)"),
    "pass_threat_p90": ("Progression and creation", "p90", True, "goal chance added by passes per 90"),
    "carry_threat_p90": ("Progression and creation", "p90", True, "goal chance added by carries per 90"),
    "progressive_p90": ("Progression and creation", "p90", True, "progressive passes and carries per 90"),
    "box_entries_p90": ("Progression and creation", "p90", True, "passes and carries into the box per 90"),
    "key_passes_p90": ("Progression and creation", "p90", True, "passes that lead to a shot per 90"),
    "xa_p90": ("Progression and creation", "p90", True, "expected assists per 90"),
    "pass_completion": ("Passing", ("rate", "passes_ok", "passes_n"), True, "pass completion"),
    "pressed_completion": ("Passing", ("rate", "pressed_ok", "pressed_n"), True, "pass completion under pressure"),
    "long_completion": ("Passing", ("rate", "long_ok", "long_n"), True, "long pass completion (30 m+)"),
    "shots_p90": ("Shooting", "p90", True, "shots per 90"),
    "xg_p90": ("Shooting", "p90", True, "expected goals per 90"),
    "xg_per_shot": ("Shooting", ("ratio", "xg_n", "shots_n"), True, "expected goals per shot (chance quality)"),
    "goals_minus_xg_p90": ("Shooting", "p90", True, "goals minus expected goals per 90 (finishing)"),
    "loss_rate": ("Ball security", ("rate", "losses_n", "actions_n"), False, "share of on-ball actions that lose the ball"),
    "threat_lost_p90": ("Ball security", "p90", False, "goal chance lost by failed actions per 90"),
    "dribble_success": ("Ball security", ("rate", "dribbles_ok", "dribbles_n"), True, "dribble success"),
    "pressures_p90": ("Defending and pressing", "p90", True, "pressures per 90 (possession-adjusted)"),
    "counterpress_p90": ("Defending and pressing", "p90", True, "counter-pressures per 90 (possession-adjusted)"),
    "high_press_share": ("Defending and pressing", ("rate", "press_high_n", "press_n"), True, "share of pressures in the opponent half"),
    "recoveries_p90": ("Defending and pressing", "p90", True, "ball recoveries and interceptions per 90 (possession-adjusted)"),
    "tackle_success": ("Defending and pressing", ("rate", "tackles_ok", "tackles_n"), True, "tackles won"),
    "aerial_success": ("Defending and pressing", ("rate", "aerials_ok", "aerials_n"), True, "aerial duels won"),
    "dribbled_past_p90": ("Defending and pressing", "p90", False, "times dribbled past per 90"),
}
DEF_ADJ = {"pressures_p90", "counterpress_p90", "recoveries_p90"}       # scaled by the opponent's share of the ball
MIN_MATCH_MIN, MIN_PEER_MIN, MIN_PROFILE_MIN = 30, 150, 900


def ascii_name(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


# ------------------------------------------------------------------ actions

def to_m(loc):
    return PM.to_m(loc)


def match_actions(mid):
    """Every action of every player in one match, with own_Threat before/after where a 360 frame exists."""
    ev = json.load(open(os.path.join(SB, "events", f"{mid}.json"), encoding="utf-8"))
    tp = os.path.join(SB, "three-sixty", f"{mid}.json")
    ffs = {f["event_uuid"]: f["freeze_frame"] for f in json.load(open(tp, encoding="utf-8"))} if os.path.exists(tp) else {}
    xg_of = {e["id"]: e["shot"].get("statsbomb_xg", 0.0) for e in ev if e["type"]["name"] == "Shot"}
    rows = []
    for e in ev:
        kind = e["type"]["name"]
        if "player" not in e or kind in ("Starting XI", "Substitution", "Player On", "Player Off", "Tactical Shift",
                                         "Half Start", "Half End", "Injury Stoppage", "Referee Ball-Drop"):
            continue
        sub = e.get(kind.lower().replace(" ", "_").replace("*", ""), {}) or {}
        loc = to_m(e["location"]) if "location" in e else (np.nan, np.nan)
        end = to_m(sub["end_location"]) if "end_location" in sub else (np.nan, np.nan)
        r = {"match": mid, "id": e["id"], "team": e["team"]["name"], "possession_team": e["possession_team"]["name"],
             "player_id": e["player"]["id"], "player": e["player"]["name"], "kind": kind, "minute": e["minute"],
             "second": e["second"], "duration": e.get("duration", 0.0) or 0.0, "x": loc[0], "y": loc[1],
             "ex": end[0], "ey": end[1], "pressed": int(bool(e.get("under_pressure"))),
             "outcome": (sub.get("outcome") or {}).get("name", ""), "sub_type": (sub.get("type") or {}).get("name", ""),
             "counterpress": int(bool(e.get("counterpress"))), "length": sub.get("length", np.nan),
             "key_pass": int(bool(sub.get("shot_assist") or sub.get("goal_assist"))),
             "xa": xg_of.get(sub.get("assisted_shot_id"), 0.0), "xg": sub.get("statsbomb_xg", 0.0) if kind == "Shot" else 0.0,
             "goal": int(kind == "Shot" and (sub.get("outcome") or {}).get("name") == "Goal"),
             "recovery_failure": int(bool(sub.get("recovery_failure"))), "before": np.nan, "after": np.nan}
        if kind in ("Pass", "Carry", "Shot") and e["id"] in ffs and "location" in e:
            att, dfs, kp = PM.frame_sides(ffs[e["id"]])
            r["before"] = PM.state_value(np.array(loc), att, dfs)
            if kind == "Shot":
                r["after"] = V.xg_shot(np.array(loc), dfs, kp, header=(sub.get("body_part") or {}).get("name") == "Head")
            elif kind == "Carry" or not r["outcome"]:
                r["after"] = PM.state_value(np.array(end), att, dfs)
            else:
                r["after"] = 0.0
        rows.append(r)
    return pd.DataFrame(rows), ev


def lineups(ev, mid):
    """Per player in the match: team, minutes (PM.minutes_played) and the position he spent most time in."""
    mins = PM.minutes_played(ev)
    end = max(e["minute"] + e["second"] / 60 for e in ev)
    pos, t_in, team, name = {}, {}, {}, {}
    def start(pid, p, t, tm, nm):
        pos.setdefault(pid, {}); t_in[pid] = (p, t); team[pid] = tm; name[pid] = nm
    def stop(pid, t):
        if pid in t_in:
            p, t0 = t_in.pop(pid); pos[pid][p] = pos[pid].get(p, 0) + max(t - t0, 0)
    for e in ev:
        t = e["minute"] + e["second"] / 60
        if e["type"]["name"] == "Starting XI":
            for p in e["tactics"]["lineup"]:
                start(p["player"]["id"], p["position"]["name"], 0.0, e["team"]["name"], p["player"]["name"])
        elif e["type"]["name"] == "Tactical Shift":
            for p in e["tactics"]["lineup"]:
                pid = p["player"]["id"]
                if pid in t_in and t_in[pid][0] != p["position"]["name"]:
                    stop(pid, t); start(pid, p["position"]["name"], t, e["team"]["name"], p["player"]["name"])
        elif e["type"]["name"] == "Substitution":
            stop(e["player"]["id"], t)
            r = e["substitution"]["replacement"]
            start(r["id"], e["position"]["name"], t, e["team"]["name"], r["name"])
    for pid in list(t_in):
        stop(pid, end)
    rows = []
    for pid, d in pos.items():
        main = max(d, key=d.get) if d else ""
        rows.append({"match": mid, "player_id": pid, "player": name[pid], "team": team[pid], "position": main,
                     "role": ROLE.get(main, "CM"), "minutes": round(mins.get(pid, 0.0), 1)})
    return rows


def actions():
    os.makedirs(OUT, exist_ok=True)
    mids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(SB, "events", "*.json")))
    A, R = [], []
    for i, mid in enumerate(mids, 1):
        a, ev = match_actions(mid)
        A.append(a)
        R += lineups(ev, mid)
        print(f"  {i}/{len(mids)} {mid}: {len(a)} actions, {a['before'].notna().sum()} valued", flush=True)
    A = pd.concat(A, ignore_index=True)
    A["added"] = A["after"] - A["before"]
    A.to_csv(os.path.join(OUT, "actions_leverkusen.csv.gz"), index=False)
    pd.DataFrame(R).to_csv(os.path.join(OUT, "roles.csv"), index=False)
    print(f"{len(A)} actions, {len(R)} player-matches")


# ------------------------------------------------------------------ metrics per player-match

def match_metrics():
    A = pd.read_csv(os.path.join(OUT, "actions_leverkusen.csv.gz"))
    R = pd.read_csv(os.path.join(OUT, "roles.csv"))
    # share of the ball per team and match, from the time of the events in each possession
    poss = A.groupby(["match", "possession_team"])["duration"].sum()
    share = poss / poss.groupby(level=0).transform("sum")
    rows = []
    for (mid, pid), g in A.groupby(["match", "player_id"]):
        r = R[(R["match"] == mid) & (R["player_id"] == pid)]
        if r.empty or r["minutes"].iloc[0] < MIN_MATCH_MIN:
            continue
        r = r.iloc[0]
        teams = A.loc[A["match"] == mid, "team"].unique()
        opp = [t for t in teams if t != r["team"]]
        opp_share = float(share.get((mid, opp[0]), 0.5)) if opp else 0.5
        adj = 0.5 / max(opp_share, 0.2)                           # as if the opponent had the ball half the time
        k = g["kind"]
        passes = g[(k == "Pass") & (~g["sub_type"].isin(list(PM.SET_PIECE)))]
        carries = g[k == "Carry"]
        shots = g[k == "Shot"]
        onball = g[k.isin(["Pass", "Carry", "Shot", "Dribble"])]
        valued = g[k.isin(["Pass", "Carry", "Shot"]) & g["added"].notna()]
        prog = lambda d: ((d["ex"] - d["x"]) >= 10) & (d["ex"] >= 0.4 * L)            # 10 m+ towards goal, ending past 40%
        in_box = lambda d: (d["ex"] >= L - 16.5) & ((d["ey"] - W / 2).abs() <= 20.16) & ~((d["x"] >= L - 16.5) & ((d["y"] - W / 2).abs() <= 20.16))
        failed = valued["after"].eq(0) & valued["kind"].eq("Pass")
        duels = g[k == "Duel"]
        tackles = duels[duels["sub_type"] == "Tackle"]
        press = g[k == "Pressure"]
        m = {"match": mid, "player_id": pid, "player": r["player"], "team": r["team"], "role": r["role"],
             "position": r["position"], "minutes": r["minutes"], "opp_share": round(opp_share, 3),
             "actions_n": len(onball), "receipts_n": int((k == "Ball Receipt*").sum()),
             "final_third_n": int((onball["x"] >= 2 * L / 3).sum()),
             "threat_added_n": valued["added"].sum(), "pass_threat_n": valued.loc[valued["kind"] == "Pass", "added"].sum(),
             "carry_threat_n": valued.loc[valued["kind"] == "Carry", "added"].sum(),
             "progressive_n": int(prog(passes[passes["outcome"].isna() | (passes["outcome"] == "")]).sum() + prog(carries).sum()),
             "box_entries_n": int(in_box(passes[passes["outcome"].isna() | (passes["outcome"] == "")]).sum() + in_box(carries).sum()),
             "key_passes_n": int(passes["key_pass"].sum()), "xa_n": passes["xa"].sum(),
             "passes_n": len(passes), "passes_ok": int((passes["outcome"].isna() | (passes["outcome"] == "")).sum()),
             "pressed_n": int(passes["pressed"].sum()),
             "pressed_ok": int((passes["pressed"].eq(1) & (passes["outcome"].isna() | (passes["outcome"] == ""))).sum()),
             "long_n": int((passes["length"] * L / 120 >= 30).sum()),
             "long_ok": int(((passes["length"] * L / 120 >= 30) & (passes["outcome"].isna() | (passes["outcome"] == ""))).sum()),
             "shots_n": len(shots), "xg_n": shots["xg"].sum(), "goals_minus_xg_n": shots["goal"].sum() - shots["xg"].sum(),
             "goals": int(shots["goal"].sum()),
             "losses_n": int(len(passes) - (passes["outcome"].isna() | (passes["outcome"] == "")).sum()
                             + (k == "Miscontrol").sum() + (k == "Dispossessed").sum()
                             + ((k == "Dribble") & (g["outcome"] == "Incomplete")).sum()),
             "threat_lost_n": -valued.loc[valued["added"] < 0, "added"].sum(),
             "dribbles_n": int((k == "Dribble").sum()), "dribbles_ok": int(((k == "Dribble") & (g["outcome"] == "Complete")).sum()),
             "pressures_n": len(press) * adj, "counterpress_n": press["counterpress"].sum() * adj,
             "press_n": len(press), "press_high_n": int((press["x"] >= L / 2).sum()),
             "recoveries_n": (((k == "Ball Recovery") & (g["recovery_failure"] == 0)).sum()
                              + ((k == "Interception") & g["outcome"].isin(["Won", "Success In Play", "Success Out"])).sum()) * adj,
             "tackles_n": len(tackles), "tackles_ok": int(tackles["outcome"].isin(["Won", "Success In Play", "Success Out"]).sum()),
             "dribbled_past_n": int((k == "Dribbled Past").sum())}                    # aerials: add_aerials()
        rows.append(m)
    M = pd.DataFrame(rows)
    # aerials won are flags on passes, shots and clearances; lost ones are duels: count both from the raw events
    M = add_aerials(M)
    for c, (area, how, hib, lab) in METRICS.items():
        if how == "p90":
            M[c] = M[c.replace("_p90", "_n")] / M["minutes"] * 90
        elif how[0] == "rate":
            M[c] = M[how[1]] / M[how[2]].replace(0, np.nan)
        else:
            M[c] = M[how[1]] / M[how[2]].replace(0, np.nan)
    M.to_csv(os.path.join(OUT, "player_match_leverkusen.csv"), index=False)
    print(f"{len(M)} player-matches with {MIN_MATCH_MIN}+ minutes, {M['player_id'].nunique()} players, {len(METRICS)} metrics")


def add_aerials(M):
    won, lost = {}, {}
    for f in glob.glob(os.path.join(SB, "events", "*.json")):
        mid = int(os.path.basename(f)[:-5])
        for e in json.load(open(f, encoding="utf-8")):
            if "player" not in e:
                continue
            key = (mid, e["player"]["id"])
            t = e["type"]["name"]
            sub = e.get(t.lower(), {}) or {}
            if sub.get("aerial_won"):
                won[key] = won.get(key, 0) + 1
            if t == "Duel" and (sub.get("type") or {}).get("name") == "Aerial Lost":
                lost[key] = lost.get(key, 0) + 1
    keys = list(zip(M["match"].astype(int), M["player_id"]))
    M["aerials_ok"] = [won.get(k, 0) for k in keys]
    M["aerials_n"] = [won.get(k, 0) + lost.get(k, 0) for k in keys]
    return M


# ------------------------------------------------------------------ pooled profile

def pooled(g, c):
    """Raw pooled value of metric c over the player-match rows g (sums, not means of per-match values)."""
    how = METRICS[c][1]
    if how == "p90":
        return g[c.replace("_p90", "_n")].sum() / g["minutes"].sum() * 90
    num, den = g[how[1]].sum(), g[how[2]].sum()
    return num / den if den > 0 else np.nan


def role_priors(M, c):
    """Per role: prior mean and prior variance of a player's true value (method of moments), and the per-90 sampling
    variance of one minute of play, for the normal-normal shrinkage of metric c."""
    out = {}
    how = METRICS[c][1]
    for role, gr in M.groupby("role"):
        P = gr.groupby("player_id")
        if how == "p90":
            n = P[c.replace("_p90", "_n")].sum()
            mins = P["minutes"].sum()
            raw = n / mins * 90
            ok = mins >= MIN_PEER_MIN
            mu = float(n.sum() / mins.sum() * 90)
            # within-player spread of per-match values, weighted by minutes -> sampling variance per 90 minutes
            dev = (gr[c] - gr["player_id"].map(raw)) ** 2 * gr["minutes"] / 90
            s2 = float(dev.sum() / max(len(gr) - gr["player_id"].nunique(), 1))
            samp = s2 / (mins / 90)
            # minutes-weighted spread between players = true spread (tau2) + the same-weighted sampling noise
            if ok.sum() > 2:
                obs = float(np.average((raw[ok] - mu) ** 2, weights=mins[ok]))
                noise = float(np.average(samp[ok], weights=mins[ok]))
            else:
                obs, noise = float(raw.var()), 0.0
            tau2 = max(obs - noise, 0.05 * obs, 1e-9)                # never shrink a metric flat onto the role mean
            out[role] = ("normal", mu, tau2, s2)
        else:
            num, den = P[how[1]].sum(), P[how[2]].sum()
            mu = float(num.sum() / max(den.sum(), 1e-9))
            p = (num / den.replace(0, np.nan)).dropna()
            dd = den[p.index]
            okp = dd >= 10
            if how[0] == "ratio":                                  # xG per shot: normal shrinkage with shots as weight
                var_b = float(np.average((p[okp] - mu) ** 2, weights=dd[okp])) if okp.sum() > 2 else 0.01
                out[role] = ("ratio", mu, max(var_b, 1e-6), None)
                continue
            # beta-binomial prior strength m from the spread between players beyond binomial noise
            obs = float(np.average((p[okp] - mu) ** 2, weights=dd[okp])) if okp.sum() > 2 else 0.0
            binom = float(np.mean(mu * (1 - mu) / dd[okp])) if okp.sum() else 0.0
            between = max(obs - binom, 1e-5)
            m = max(min(mu * (1 - mu) / between - 1, 2000), 2)
            out[role] = ("beta", mu, m, None)
    return out


def shrink(g, c, prior):
    kind, mu, a, b = prior
    how = METRICS[c][1]
    raw = pooled(g, c)
    if kind == "normal":
        mins = g["minutes"].sum()
        samp = b / (mins / 90)
        w = a / (a + samp)
        return raw, w * raw + (1 - w) * mu
    num, den = g[how[1]].sum(), g[how[2]].sum()
    if kind == "beta":
        return raw, (num + a * mu) / (den + a)
    k = 8.0                                                       # ratio: pseudo-shots at the role mean
    return raw, (num + k * mu) / (den + k)


def profile():
    M = pd.read_csv(os.path.join(OUT, "player_match_leverkusen.csv"))
    main_role = M.groupby("player_id").apply(lambda g: g.groupby("role")["minutes"].sum().idxmax())
    M["role"] = M["player_id"].map(main_role)                   # one role per player: where he played most
    priors = {c: role_priors(M, c) for c in METRICS}
    rows = []
    players = M.groupby("player_id")
    tot_min = players["minutes"].sum()
    for pid, g in players:
        role = main_role[pid]
        for c in METRICS:
            raw, val = shrink(g, c, priors[c][role])
            rows.append({"player_id": pid, "player": g["player"].iloc[0], "team": g["team"].mode().iloc[0], "role": role,
                         "matches": g["match"].nunique(), "minutes": round(tot_min[pid]), "metric": c, "raw": raw, "value": val})
    P = pd.DataFrame(rows)
    # percentiles within the role, among peers with enough minutes (shrunk values)
    P["percentile"] = np.nan
    for (role, c), gr in P.groupby(["role", "metric"]):
        peers = gr[gr["minutes"] >= MIN_PEER_MIN]["value"].dropna().to_numpy()
        hib = METRICS[c][2]
        if len(peers) < 5:
            continue
        pct = gr["value"].apply(lambda v: 100 * (np.mean(peers < v) + 0.5 * np.mean(peers == v)) if pd.notna(v) else np.nan)
        P.loc[gr.index, "percentile"] = pct if hib else 100 - pct
        P.loc[gr.index, "role_median"] = np.median(peers)
        P.loc[gr.index, "peers"] = len(peers)
    # 90% interval by bootstrap over the player's matches (players with enough minutes for a profile)
    P["lo"], P["hi"] = np.nan, np.nan
    for pid, g in players:
        if tot_min[pid] < MIN_PROFILE_MIN:
            continue
        role = main_role[pid]
        mids = g["match"].unique()
        boots = {c: [] for c in METRICS}
        for _ in range(400):
            gb = pd.concat([g[g["match"] == m] for m in RNG.choice(mids, len(mids))])
            for c in METRICS:
                boots[c].append(shrink(gb, c, priors[c][role])[1])
        for c in METRICS:
            v = np.array(boots[c], float)
            v = v[~np.isnan(v)]
            if len(v):
                i = P.index[(P["player_id"] == pid) & (P["metric"] == c)]
                P.loc[i, "lo"], P.loc[i, "hi"] = np.percentile(v, 5), np.percentile(v, 95)
    P["area"] = P["metric"].map(lambda c: METRICS[c][0])
    P["higher_is_better"] = P["metric"].map(lambda c: METRICS[c][2])
    P["label"] = P.apply(label_of, axis=1)
    rp = os.path.join(OUT, "reliability.csv")                   # from `check`: noise-level metrics are never a label
    if os.path.exists(rp):
        bad = set(pd.read_csv(rp).query("~usable")["metric"])
        P.loc[P["metric"].isin(bad), "label"] = "unclear (unreliable metric)"
    P.to_csv(os.path.join(OUT, "profiles_leverkusen.csv"), index=False)
    for pid in tot_min[(tot_min >= MIN_PROFILE_MIN)].index:
        q = P[P["player_id"] == pid]
        if q["team"].iloc[0] != CLUB:
            continue
        write_card(q)
    print(f"profiles: {P['player_id'].nunique()} players; cards for {CLUB} players with {MIN_PROFILE_MIN}+ minutes")


def label_of(r):
    if pd.isna(r["percentile"]) or pd.isna(r["lo"]):
        return "unclear"
    med = r["role_median"]
    eps = 1e-6 * max(abs(med), 1e-6)                             # equal to the median (to rounding) is not above it
    above, below = ((r["lo"] > med + eps, r["hi"] < med - eps) if r["higher_is_better"]
                    else (r["hi"] < med - eps, r["lo"] > med + eps))
    if r["percentile"] >= 80 and above:
        return "strength"
    if r["percentile"] <= 20 and below:
        return "weakness"
    return "unclear"


def fmt(c, v):
    if pd.isna(v):
        return "–"
    how = METRICS[c][1]
    if how != "p90" and how[0] == "rate":
        return f"{100 * v:.0f}%"
    return f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}"


def write_card(q):
    r0 = q.iloc[0]
    slug = ascii_name(r0["player"]).lower().replace(" ", "_")
    items = []
    for r in q.itertuples():
        items.append({"metric": r.metric, "area": r.area, "what": METRICS[r.metric][3], "value": fmt(r.metric, r.value),
                      "interval_90": [fmt(r.metric, r.lo), fmt(r.metric, r.hi)], "role_median": fmt(r.metric, r.role_median),
                      "percentile": None if pd.isna(r.percentile) else round(float(r.percentile)), "label": r.label})
    rel = 0.0
    sw = lambda lab: sorted([i for i in items if i["label"] == lab], key=lambda i: -abs(i["percentile"] - 50))
    card = {"model": "own_PlayerProfile", "player": r0["player"], "team": r0["team"], "role": r0["role"],
            "matches": int(r0["matches"]), "minutes": int(r0["minutes"]),
            "peers": f"players in role {r0['role']} with {MIN_PEER_MIN}+ minutes in the 34 Leverkusen matches 2023/24 (both teams)",
            "strengths": sw("strength"), "weaknesses": sw("weakness"), "metrics": items,
            "areas": {a: round(float(np.nanmean([i["percentile"] for i in items if i["area"] == a and i["percentile"] is not None])), 0)
                      for a in dict.fromkeys(i["area"] for i in items)},
            "source": "StatsBomb open data (Bayer Leverkusen 2023/24); own_Threat and own_xG from value_models.py"}
    json.dump(card, open(os.path.join(OUT, f"profile_{slug}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ checks

def spearman(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else np.nan


def check():
    M = pd.read_csv(os.path.join(OUT, "player_match_leverkusen.csv"))
    P = pd.read_csv(os.path.join(OUT, "profiles_leverkusen.csv"))
    lev = M[M["team"] == CLUB]
    tot = lev.groupby("player_id")["minutes"].sum()
    big = tot[tot >= MIN_PROFILE_MIN].index
    # 1) split-half reliability: odd vs even matches (in date order), Spearman-Brown corrected
    order = {m: i for i, m in enumerate(sorted(lev["match"].unique()))}
    lev = lev.assign(half=lev["match"].map(order) % 2)
    rows = []
    for c in METRICS:
        a = lev[lev["half"] == 0].groupby("player_id").apply(lambda g: pooled(g, c))
        b = lev[lev["half"] == 1].groupby("player_id").apply(lambda g: pooled(g, c))
        idx = [p for p in big if p in a.index and p in b.index and pd.notna(a[p]) and pd.notna(b[p])]
        r = spearman(a[idx], b[idx])
        rows.append({"metric": c, "area": METRICS[c][0], "players": len(idx), "split_half": r,
                     "reliability": 2 * r / (1 + r) if pd.notna(r) and r > -1 else np.nan})
    R = pd.DataFrame(rows)
    R["usable"] = R["reliability"] >= 0.5
    R.to_csv(os.path.join(OUT, "reliability.csv"), index=False)
    # 2) leave-one-match-out: does the player's profile from his other matches predict the held-out match better than
    #    the role mean? (per-90 metrics, minutes-weighted absolute error)
    rows = []
    priors_all = {c: role_priors(M, c) for c in METRICS if METRICS[c][1] == "p90"}
    for c in priors_all:
        e_prof, e_role, w = [], [], []
        for pid in big:
            g = lev[lev["player_id"] == pid]
            role = P.loc[P["player_id"] == pid, "role"].iloc[0]
            for mid in g["match"].unique():
                tr, te = g[g["match"] != mid], g[g["match"] == mid]
                pred = shrink(tr, c, priors_all[c][role])[1]
                truth = pooled(te, c)
                e_prof.append(abs(pred - truth)); e_role.append(abs(priors_all[c][role][1] - truth)); w.append(te["minutes"].sum())
        e_prof, e_role, w = map(np.array, (e_prof, e_role, w))
        rows.append({"metric": c, "profile_error": np.average(e_prof, weights=w), "role_mean_error": np.average(e_role, weights=w),
                     "profile_better_%": 100 * float(np.average(e_prof < e_role, weights=w))})
    S = pd.DataFrame(rows)
    S["gain_%"] = 100 * (1 - S["profile_error"] / S["role_mean_error"])
    S.to_csv(os.path.join(OUT, "stability.csv"), index=False)
    # 3) Impect: per player over the season, our metric vs the closest Impect KPI
    I = PM.impect_table()
    link = PM.link_names(lev["player"].unique(), I["name"].unique())
    pairs = [("progressive_p90", "imp_bypassed"), ("threat_added_p90", "imp_bypassed"), ("xa_p90", "imp_xga"),
             ("xg_p90", "imp_shot_xg"), ("box_entries_p90", "imp_bypassed_recv")]
    lev_season = lev.groupby("player").agg(minutes=("minutes", "sum"))
    I2 = I.assign(player=I["name"].map({v: k for k, v in link.items()})).dropna(subset=["player"])
    I2 = I2.groupby("player").agg(imp_minutes=("imp_minutes", "sum"),
                                  **{k: (k, "sum") for k in ["imp_bypassed", "imp_bypassed_recv", "imp_xga", "imp_shot_xg"]})
    rows = []
    for c, k in pairs:
        ours = lev.groupby("player").apply(lambda g: pooled(g, c))
        j = pd.concat([ours.rename("ours"), lev_season, I2], axis=1, join="inner")
        j = j[j["minutes"] >= MIN_PROFILE_MIN]
        rows.append({"ours": c, "impect": k, "players": len(j), "spearman": spearman(j["ours"], j[k] / j["imp_minutes"])})
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(OUT, "impect_check.csv"), index=False)
    pd.set_option("display.width", 200)
    print("Reliability (split-half, Leverkusen players 900+ min):")
    print(R.round(2).to_string(index=False))
    print("\nLeave-one-match-out: profile vs role mean (per-90 metrics):")
    print(S.round(3).to_string(index=False))
    print("\nAgainst Impect (season per 90):")
    print(C.round(2).to_string(index=False))


def export_viewer():
    """All report cards in one file for the 3D viewer (viewer/profile.js): the profile, plus each player's best and
    worst moments (own_Threat goal chance added or lost by a pass, carry or shot), with match, minute and the
    action's start and end (pitch metres, his team attacking to the right, as in StatsBomb)."""
    meta = {int(m["match_id"]): m for m in json.load(open(os.path.join(SB, "matches_9_281.json"), encoding="utf-8"))}
    A = pd.read_csv(os.path.join(OUT, "actions_leverkusen.csv.gz"))
    A = A[A["kind"].isin(["Pass", "Carry", "Shot"]) & A["added"].notna()]
    def moment(r):
        m = meta.get(int(r.match), {})
        home, away = m.get("home_team", {}).get("home_team_name", "?"), m.get("away_team", {}).get("away_team_name", "?")
        ok = r.kind == "Carry" or (r.kind == "Pass" and not isinstance(r.outcome, str)) or (r.kind == "Shot")
        res = "goal" if r.goal else (r.outcome if isinstance(r.outcome, str) and r.outcome else ("complete" if ok else ""))
        return {"match": int(r.match), "date": m.get("match_date"), "vs": away if r.team == home else home,
                "home": r.team == home, "minute": int(r.minute), "second": int(r.second), "kind": r.kind, "result": res,
                "xg": round(float(r.xg), 2), "added": round(float(r.added), 3),
                "x": round(float(r.x), 1), "y": round(float(r.y), 1),
                "ex": None if pd.isna(r.ex) else round(float(r.ex), 1), "ey": None if pd.isna(r.ey) else round(float(r.ey), 1)}
    cards = []
    for f in sorted(glob.glob(os.path.join(OUT, "profile_*.json"))):
        c = json.load(open(f, encoding="utf-8"))
        a = A[A["player"] == c["player"]]
        c["best"] = [moment(r) for r in a.nlargest(6, "added").itertuples()]
        c["worst"] = [moment(r) for r in a.nsmallest(3, "added").itertuples()]
        cards.append(c)
    cards.sort(key=lambda c: -c["minutes"])
    out = os.path.join(ROOT, "viewer", "data", "player_profiles.json")
    rel = pd.read_csv(os.path.join(OUT, "reliability.csv")) if os.path.exists(os.path.join(OUT, "reliability.csv")) else None
    json.dump({"model": "own_PlayerProfile", "season": "Bayer Leverkusen 2023/24 (StatsBomb open data, 34 matches)",
               "areas": list(dict.fromkeys(v[0] for v in METRICS.values())),
               "unreliable": [] if rel is None else [str(m) for m in rel.loc[~rel["usable"].astype(bool), "metric"]]
               if "usable" in rel else [], "cards": cards},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print("wrote", out, len(cards), "cards")


def show(name):
    for f in glob.glob(os.path.join(OUT, "profile_*.json")):
        c = json.load(open(f, encoding="utf-8"))
        if name.lower() in ascii_name(c["player"]).lower():
            print(f"{c['player']} ({c['role']}, {c['matches']} matches, {c['minutes']} min)  peers: {c['peers']}")
            print("areas (mean percentile):", c["areas"])
            for lab in ("strengths", "weaknesses"):
                print(f"\n{lab.upper()}:")
                for i in c[lab]:
                    print(f"  {i['what']:55s} {i['value']:>7s}  [{i['interval_90'][0]} – {i['interval_90'][1]}]  "
                          f"role median {i['role_median']:>7s}  pct {i['percentile']}")
            return
    print("no profile for", name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["actions", "metrics", "profile", "check", "all", "show", "viewer"])
    ap.add_argument("--player", default="Wirtz")
    a = ap.parse_args()
    if a.step in ("actions", "all"):
        actions()
    if a.step in ("metrics", "all"):
        match_metrics()
    if a.step in ("profile", "all"):
        profile()
    if a.step in ("check", "all"):
        check()
        if a.step == "all":
            profile()                                             # labels again, now knowing which metrics are reliable
    if a.step in ("viewer", "all"):
        export_viewer()
    if a.step == "show":
        show(a.player)
