"""Layer 4, first version: what do teams do in each tactical state, and do the tactical states mean anything?

Joins three things that already exist:
    tactical states   the 4 states of tactical_states.py (k-means on the network's vectors; numbered from slowest to fastest players)
    pass options      for every pass, every teammate with arrival chance, danger of the arrival point, expected gain
                      (pass_features.py, fit_pass_model.py; real matches are scored with models that never saw that match)
    the pass played   who the player chose
A pass belongs to the tactical state of the last Key State before it.

Four outputs
    1  PLAYBOOK     per tactical state: share of passes, pass length, forward gain, how often the best-gain option was chosen
                    (real pooled, and every simulator setting) -> outputs/tactics_playbook.csv
    2  TEAM STYLE   per real team in a match: the same numbers, one row per team and match -> outputs/tactics_team_style.csv
    3  UNUSED BEST  when the best-gain option was not chosen, why (the best option was riskier / much further / closely marked /
                    its lane could be cut / the best option overall was offside) -> outputs/tactics_unused.csv
    4  VALIDITY     do the tactical states tell us anything about what happens next? For every Key State with a ball owner: AUC for
                    'the team with the ball shoots within 10 s' (and 'keeps the ball 10 s') from (a) the place of the ball alone,
                    (b) the place plus the tactical state; each real match is judged with models that never saw it
                    -> outputs/tactics_validity.csv (read by scoreboard.py)

Reasons are rules on model numbers, not facts about intentions; read them as 'what the best option looked like', not 'why the
player decided'.

Output : the tables, the files named above, outputs/tactics_profile.png
Usage  :  python scripts\\tactics_profile.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit_pass_model as F
import settings
import validate_gain as V

REASONS = ["best option riskier", "best option much further", "best option closely marked", "lane could be cut", "best option offside"]


def state_of(m, times):
    t = pd.read_csv(settings.base(m) + "_tactical_state.csv")
    idx = np.searchsorted(t["time_s"].to_numpy(), times, side="right") - 1
    return np.where(idx >= 0, t["tactical_state"].to_numpy()[np.maximum(idx, 0)], 0)


def pass_table(m, d, models, group, names):
    w, comp, val = models
    I, M, T = d["I"], d["mask"], d["target"]
    n, ar = len(T), np.arange(len(T))
    s = F.choice_scores(w, I, M, F.FULL)
    rank = 1 + (s > s[ar, T][:, None]).sum(1)
    arrives = F.completion_prob(comp, I)
    danger = F.value_at(val, d["dest"][..., 0], d["dest"][..., 1])
    gain = np.where(M, arrives * danger, -1.0)
    on = M & ~d["offside"]
    gain_on = np.where(on, gain, -1.0)
    best, best_all = gain_on.argmax(1), gain.argmax(1)
    dist, opn, margin = I[..., F.COL["dist"]] * 10, I[..., F.COL["open_space"]] * 10, I[..., F.COL["margin"]]
    g_best, g_ch = gain_on[ar, best], gain[ar, T]
    unused = T != best
    team = np.where(d["passer"] < 11, names[0], names[1])
    return pd.DataFrame({
        "match": m, "group": group, "team": team, "state": state_of(m, d["time_s"]), "complete": d["complete"],
        "length_m": dist[ar, T], "forward_gain_m": I[ar, T, F.COL["adv"]] * 20, "picked_top1": rank <= 1,
        "arrives_chosen": arrives[ar, T], "gain_chosen": g_ch, "gain_best": g_best, "chose_best": ~unused,
        "regret": np.maximum(g_best - g_ch, 0) / np.maximum(g_best, 1e-9), "unused": unused,
        "r_risky": unused & (arrives[ar, best] < arrives[ar, T] - 0.10), "r_far": unused & (dist[ar, best] > dist[ar, T] + 10),
        "r_marked": unused & (opn[ar, best] < 2.0), "r_cut": unused & (margin[ar, best] < 0), "r_offside": d["offside"][ar, best_all],
    })


def playbook(df):
    rows = []
    for grp in ["real"] + sorted(set(df["group"]) - {"real"}):
        g = df[df["group"] == grp]
        for s in sorted(g["state"].unique()):
            x = g[(g["state"] == s)]
            if s == 0 or len(x) < 5:
                continue
            c = x[x["complete"]]
            rows.append({"group": grp, "state": int(s), "passes": len(x), "share_%": 100 * len(x) / len(g[g["state"] > 0]),
                         "length_m": c["length_m"].mean(), "forward_gain_m": c["forward_gain_m"].mean(), "chose_best_%": 100 * c["chose_best"].mean(),
                         "regret_%": 100 * c["regret"].mean(), "arrive_%": 100 * x["complete"].mean() if grp == "real" else np.nan,
                         "model_picks_top1_%": 100 * c["picked_top1"].mean()})
    return pd.DataFrame(rows)


def unused_table(df):
    rows = []
    for grp in ["real"] + sorted(set(df["group"]) - {"real"}):
        u = df[(df["group"] == grp) & df["complete"] & df["unused"]]
        allc = df[(df["group"] == grp) & df["complete"]]
        row = {"group": grp, "passes": len(allc), "best_unused_%": 100 * len(u) / max(len(allc), 1)}
        for name, col in zip(REASONS[:4], ["r_risky", "r_far", "r_marked", "r_cut"]):
            row[name + " (%)"] = 100 * u[col].mean()
        row["none of these (%)"] = 100 * (~(u[["r_risky", "r_far", "r_marked", "r_cut"]].any(axis=1))).mean()
        row["best overall offside (% of passes)"] = 100 * allc["r_offside"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def validity(real, models_by_match):
    """Per held-out real match: AUC for 'shot within 10 s' and 'ball kept 10 s' with the place of the ball alone, and with the tactical state added."""
    val = F.from_json()[2]
    samples = {}
    for m in real:
        g = np.load(settings.graphs(m))
        c = pd.read_csv(settings.csv(m), usecols=["frame", "ball_owner_team", "game_mode"]).drop_duplicates("frame").set_index("frame").loc[g["frame"]]
        st = state_of(m, g["time_s"])
        ok = (c["ball_owner_team"].isin(["home", "away"]).to_numpy() & (c["game_mode"] == "normal").to_numpy() & ~np.isnan(g["xy"][:, 22, 0]) & (st > 0))
        side = np.where(c["ball_owner_team"].to_numpy()[ok] == "home", 0, 1)
        bx, by = g["xy"][ok, 22, 0], g["xy"][ok, 22, 1]
        loc = F.value_at(val, np.where(side == 0, bx, 105.0 - bx), by)
        shot, kept = V.outcomes(m, g["frame"][ok], g["time_s"][ok], np.where(side == 0, 0, 11))
        samples[m] = {"loc": loc, "state": st[ok], "shot": shot, "kept": kept}
    k = max(max(s["state"]) for s in samples.values())
    rows = []
    for m in real:
        tr = [x for x in real if x != m]
        row = {"match": m, "key_states": len(samples[m]["loc"])}
        for target in ("shot", "kept"):
            y_tr = np.concatenate([samples[x][target] for x in tr]).astype(float)
            loc_tr = np.concatenate([samples[x]["loc"] for x in tr])[:, None] * 10
            st_tr = np.eye(k + 1)[np.concatenate([samples[x]["state"] for x in tr])][:, 1:]
            te_loc, te_st, y = samples[m]["loc"][:, None] * 10, np.eye(k + 1)[samples[m]["state"]][:, 1:], samples[m][target].astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            w1, b1 = F.fit_logistic(loc_tr, y_tr)
            w2, b2 = F.fit_logistic(np.hstack([loc_tr, st_tr]), y_tr)
            suffix = "" if target == "shot" else "_kept"
            row["auc_location" + suffix] = F.auc(y, te_loc @ w1 + b1)
            row["auc_location_and_state" + suffix] = F.auc(y, np.hstack([te_loc, te_st]) @ w2 + b2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("match")


def main():
    real = [m for m in settings.available(settings.REAL) if os.path.exists(settings.base(m) + "_passfeat.npz") and os.path.exists(settings.base(m) + "_tactical_state.csv")]
    sim = [m for m in settings.available(settings.SIM) if os.path.exists(settings.base(m) + "_passfeat.npz") and os.path.exists(settings.base(m) + "_tactical_state.csv")]
    if len(real) < 2:
        raise SystemExit("run scripts/pass_features.py, scripts/fit_pass_model.py and scripts/tactical_states.py first")
    reg = pd.read_csv("data/real/matches.csv").set_index("match_id")
    data = F.load(real + sim)
    parts = []
    for m in real:
        tr = [x for x in real if x != m]
        models = (F.fit_choice(F.cat(data, tr, "I", True), F.cat(data, tr, "mask", True), F.cat(data, tr, "target", True), F.FULL),
                  F.fit_completion(data, tr), F.from_json()[2])
        parts.append(pass_table(m, data[m], models, "real", (reg.loc[m, "home"], reg.loc[m, "away"])))
        print(f"  {m} done", flush=True)
    models = F.from_json()
    for m in sim:
        parts.append(pass_table(m, data[m], models, settings.variant(m), ("sim home", "sim away")))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["state"] > 0]
    os.makedirs("outputs", exist_ok=True)

    pb = playbook(df)
    pb.round(2).to_csv("outputs/tactics_playbook.csv", index=False)
    print("\n1  PLAYBOOK: what is played in each tactical state (states numbered from slowest to fastest players; pooled over matches)\n")
    for grp, g in pb.groupby("group", sort=False):
        print(f"-- {grp}")
        print(g.drop(columns="group").set_index("state").round(1).to_string())

    ts = df[(df["group"] == "real") & df["complete"]].groupby(["match", "team"]).agg(
        passes=("length_m", "size"), length_m=("length_m", "mean"), forward_gain_m=("forward_gain_m", "mean"), chose_best_pct=("chose_best", lambda s: 100 * s.mean()),
        regret_pct=("regret", lambda s: 100 * s.mean()), arrives_chosen=("arrives_chosen", "mean"))
    ts.round(2).to_csv("outputs/tactics_team_style.csv")
    print("\n2  TEAM STYLE (real teams, one row per team and match; completed passes)\n")
    print(ts.round(2).to_string())
    print(f"\n   spread between team-matches: length {ts['length_m'].min():.1f}-{ts['length_m'].max():.1f} m, forward gain {ts['forward_gain_m'].min():.1f}-{ts['forward_gain_m'].max():.1f} m")

    un = unused_table(df)
    un.round(1).to_csv("outputs/tactics_unused.csv", index=False)
    print("\n3  WHEN THE BEST-GAIN OPTION IS NOT CHOSEN: what did the best option look like? (% of those passes; one pass can have several)\n")
    print(un.set_index("group").round(1).T.to_string())

    print("\n4  DO THE TACTICAL STATES MEAN ANYTHING? AUC for what happens in the next 10 s, Key States with a ball owner, held-out matches\n")
    v = validity(real, None)
    v.round(4).to_csv("outputs/tactics_validity.csv")
    print(v.round(3).to_string())
    print("\n   mean:", v.drop(columns="key_states").mean().round(3).to_dict())
    gain = (v["auc_location_and_state"] - v["auc_location"]).mean()
    print(f"   the tactical state adds {gain:+.3f} AUC for 'shot within 10 s' on top of the place of the ball")

    groups = list(pb["group"].unique())
    cols = ["#2a78d6"] + [plt.cm.Oranges(x) for x in np.linspace(0.9, 0.45, max(len(groups) - 1, 1))]
    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    states = sorted(pb["state"].unique())
    for a, (k, title) in zip(ax.ravel()[:3], [("share_%", "share of passes (%)"), ("length_m", "pass length (m)"), ("chose_best_%", "best-gain option chosen (%)")]):
        for i, grp in enumerate(groups):
            sub = pb[pb["group"] == grp].set_index("state")[k].reindex(states)
            a.bar(np.arange(len(states)) + (i - len(groups) / 2 + 0.5) * 0.15, sub, width=0.15, color=cols[i], label=grp)
        a.set_xticks(range(len(states)), [f"state {s}" for s in states])
        a.set_title(title, loc="left", fontsize=9)
    ax[0, 0].legend(frameon=False, fontsize=7)
    names = [c for c in un.columns if c.endswith("(%)") and "offside" not in c]
    for i, grp in enumerate(groups):
        r = un[un["group"] == grp].iloc[0]
        ax[1, 1].bar(np.arange(len(names)) + (i - len(groups) / 2 + 0.5) * 0.15, [r[c] for c in names], width=0.15, color=cols[i])
    ax[1, 1].set_xticks(range(len(names)), [n.replace(" (%)", "").replace(" ", "\n", 2) for n in names], fontsize=7)
    ax[1, 1].set_title("what the best option looked like when it was not chosen (%)", loc="left", fontsize=9)
    for a in ax.ravel():
        for side in ("right", "top"):
            a.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/tactics_profile.png", dpi=150)
    print("\nsaved outputs/tactics_playbook.csv, tactics_team_style.csv, tactics_unused.csv, tactics_validity.csv, tactics_profile.png")


if __name__ == "__main__":
    main()
