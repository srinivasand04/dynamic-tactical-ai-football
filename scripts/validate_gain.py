"""Does the 'expected gain' of a pass mean anything? Check it against what really happens afterwards.

For every real pass (all open-play passes, completed or failed) we know the gain of the option the player chose:
    gain = chance the pass arrives (model B, learned on the other 6 matches) x danger of the arrival point (model C)
We then look at the 10 seconds after the pass and ask two questions:
    shot   does the passing team shoot within 10 s?
    kept   does the passing team keep the ball for 10 s (the other team never has it in live play)?
and how well each number predicts them (AUC: 1 = perfect, 0.5 = guessing; each match is judged with models that never saw it):
    danger of where the ball was       (the place before the pass)
    danger of the arrival point        (where the chosen pass goes)
    chance it arrives                  (model B)
    gain of the chosen option          (B x C)
    gain of the best option, regret    (best gain - chosen gain)
and whether the gain tells us MORE than the place the ball was in (a logistic regression with the danger of the starting place
alone, and then with the gain added: AUC of both). Also the shot rate for the four quarters of the gain.

How to read it
    gain predicts shots better than the starting place   -> the number carries information about the pass itself
    arrival chance predicts 'kept'                       -> it measures risk
    regret does NOT predict anything                     -> 'regret' is not a measure of a bad decision (do not use it as one)
Caveat: the danger map was fitted on ball positions and shots of the same matches, so 'danger' predicting 'shot' is partly by
construction; the useful questions are the extra information of the gain and the behaviour of the arrival chance.

Output : the tables, outputs/validate_gain.csv, outputs/validate_gain.png
Usage  :  python scripts\\validate_gain.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit_pass_model as F
import pass_features as P
import settings

HORIZON_S = 10.0


def outcomes(m, frames, times, passer):
    """Per pass: shot (the passing team shoots within the next 10 s) and kept (it keeps the ball for 10 s of live play)."""
    f = pd.read_csv(settings.csv(m), usecols=["frame", "time_s", "ball_owner_team", "game_mode", "period"]).drop_duplicates("frame").sort_values("frame")
    period2 = float(f.loc[f["period"] == 2, "time_s"].min())
    pl = pd.read_csv(f"data/real/raw/{m}_players.csv")
    side = pl.drop_duplicates("team_id").set_index("team_id")["ground"]
    e = pd.read_csv(f"data/real/raw/{m}_events.csv.gz")
    e = e[e["event_type"] == "SHOT"].copy()
    e["time_s"] = pd.to_timedelta(e["timestamp"]).dt.total_seconds() + np.where(e["period_id"] == 2, period2, 0.0)
    shots = {s: np.sort(e.loc[e["team_id"].map(side) == s, "time_s"].to_numpy()) for s in ("home", "away")}
    t, own, live = f["time_s"].to_numpy(), f["ball_owner_team"].to_numpy(), (f["game_mode"] == "normal").to_numpy()
    shot, kept = np.zeros(len(times), bool), np.zeros(len(times), bool)
    for i, (tm, p) in enumerate(zip(times, passer)):
        team = "home" if p < 11 else "away"
        j = np.searchsorted(shots[team], tm, side="right")
        shot[i] = j < len(shots[team]) and shots[team][j] - tm <= HORIZON_S
        lo, hi = np.searchsorted(t, tm, side="right"), np.searchsorted(t, tm + HORIZON_S, side="right")
        other = "away" if team == "home" else "home"
        kept[i] = not ((own[lo:hi] == other) & live[lo:hi]).any()
    return shot, kept


def main():
    matches = [m for m in settings.available(settings.REAL) if os.path.exists(settings.base(m) + "_passfeat.npz")]
    if len(matches) < 2:
        raise SystemExit("run scripts/pass_features.py and scripts/fit_pass_model.py first")
    data = F.load(matches)
    val = F.from_json()[2]
    rows = []
    for m in matches:
        tr = [x for x in matches if x != m]
        comp = F.fit_completion(data, tr)
        d = data[m]
        n, ar = len(d["target"]), np.arange(len(d["target"]))
        arrive = F.completion_prob(comp, d["I"])
        danger = F.value_at(val, d["dest"][..., 0], d["dest"][..., 1])
        on = d["mask"] & ~d["offside"]
        gain = np.where(d["mask"], arrive * danger, -1.0)
        best = np.where(on, gain, -1.0).max(1)
        x0 = np.where(d["passer"] < 11, 0, 1)
        start = np.zeros(n)
        z = np.load(settings.base(m) + "_passes.npz", allow_pickle=True)
        fr = {int(f): i for i, f in enumerate(z["frame"])}
        for i, f0 in enumerate(d["frame"]):
            k = fr[int(f0)]
            a = z["xy"][k, int(d["passer"][i])]
            start[i] = F.value_at(val, np.array(a[0] if x0[i] == 0 else P.L - a[0]), np.array(a[1]))
        shot, kept = outcomes(m, d["frame"], d["time_s"], d["passer"])
        for i in range(n):
            rows.append({"match": m, "shot": shot[i], "kept": kept[i], "start": start[i], "arrival_point": danger[i, d["target"][i]],
                         "arrives": arrive[i, d["target"][i]], "gain": gain[i, d["target"][i]], "best": best[i],
                         "regret": max(best[i] - gain[i, d["target"][i]], 0.0), "complete": bool(d["complete"][i])})
    df = pd.DataFrame(rows)
    print(f"{len(df)} real passes, {len(matches)} matches; a shot follows within 10 s in {100 * df['shot'].mean():.1f}% of passes, "
          f"the ball is kept for 10 s in {100 * df['kept'].mean():.1f}%\n")

    names = {"start": "danger of where the ball was", "arrival_point": "danger of the arrival point", "arrives": "chance it arrives",
             "gain": "gain of the chosen option", "best": "gain of the best option", "regret": "regret (best - chosen)"}
    per = []
    for k, label in names.items():
        for target in ("shot", "kept"):
            for m_, g in df.groupby("match"):
                if 0 < g[target].sum() < len(g):
                    per.append({"match": m_, "predictor": label, "outcome": target, "AUC": F.auc(g[target].to_numpy().astype(int), g[k].to_numpy())})
    pm = pd.DataFrame(per)
    os.makedirs("outputs", exist_ok=True)
    pm.round(4).to_csv("outputs/validate_gain_matches.csv", index=False)
    r = pm.groupby(["predictor", "outcome"])["AUC"].mean().unstack().loc[list(names.values())]
    print("HOW WELL EACH NUMBER PREDICTS WHAT HAPPENS NEXT (AUC, average over matches)\n")
    print(r.round(3).to_string())

    extra = []
    for m in matches:
        tr, te = df[df["match"] != m], df[df["match"] == m]
        if te["shot"].sum() == 0:
            continue
        base = F.fit_logistic(np.stack([tr["start"]], 1) * 10, tr["shot"].to_numpy().astype(float))
        both = F.fit_logistic(np.stack([tr["start"], tr["gain"]], 1) * 10, tr["shot"].to_numpy().astype(float))
        y = te["shot"].to_numpy().astype(int)
        extra.append((F.auc(y, np.stack([te["start"]], 1) @ base[0] * 10 + base[1]),
                      F.auc(y, np.stack([te["start"], te["gain"]], 1) @ both[0] * 10 + both[1])))
        pm = pd.concat([pm, pd.DataFrame([{"match": m, "predictor": "location alone", "outcome": "shot", "AUC": extra[-1][0]},
                                          {"match": m, "predictor": "location + gain", "outcome": "shot", "AUC": extra[-1][1]}])])
    pm.round(4).to_csv("outputs/validate_gain_matches.csv", index=False)
    e = np.mean(extra, axis=0)
    print(f"\nshot within 10 s, AUC with the place of the ball alone {e[0]:.3f}, with the gain of the chosen option added {e[1]:.3f}")

    df["quarter"] = pd.qcut(df["gain"].rank(method="first"), 4, labels=["lowest", "low", "high", "highest"])
    q = df.groupby("quarter", observed=True).agg(passes=("shot", "size"), shot_within_10s_pct=("shot", lambda s: 100 * s.mean()),
                                                   ball_kept_pct=("kept", lambda s: 100 * s.mean()), arrives_pct=("complete", lambda s: 100 * s.mean()))
    print("\nBY QUARTER OF THE GAIN OF THE CHOSEN OPTION\n")
    print(q.round(1).to_string())
    df["regret_quarter"] = pd.qcut(df["regret"].rank(method="first"), 4, labels=["smallest", "small", "large", "largest"])
    qr = df.groupby("regret_quarter", observed=True).agg(shot_within_10s_pct=("shot", lambda s: 100 * s.mean()), ball_kept_pct=("kept", lambda s: 100 * s.mean()))
    print("\nBY QUARTER OF THE REGRET\n")
    print(qr.round(1).to_string())

    os.makedirs("outputs", exist_ok=True)
    r.round(3).to_csv("outputs/validate_gain.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    y = np.arange(len(r))[::-1]
    ax[0].barh(y + 0.18, r["shot"], height=0.34, color="#2a78d6", label="shot within 10 s")
    ax[0].barh(y - 0.18, r["kept"], height=0.34, color="#b4b2a9", label="ball kept for 10 s")
    ax[0].axvline(0.5, color="#73726c", lw=1, ls=(0, (4, 3)))
    ax[0].set_yticks(y, r.index)
    ax[0].set_xlim(0.3, 0.9)
    ax[0].set_xlabel("AUC (dashed = guessing)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("What predicts the next 10 seconds?", loc="left", fontsize=10)
    x = np.arange(4)
    ax[1].bar(x - 0.18, q["shot_within_10s_pct"], width=0.34, color="#2a78d6", label="shot within 10 s")
    ax[1].bar(x + 0.18, q["ball_kept_pct"], width=0.34, color="#b4b2a9", label="ball kept for 10 s")
    ax[1].set_xticks(x, q.index)
    ax[1].set_ylabel("% of passes")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("By quarter of the gain of the chosen pass", loc="left", fontsize=10)
    for a in ax:
        for side in ("right", "top"):
            a.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/validate_gain.png", dpi=150)
    print("\nsaved outputs/validate_gain.csv, outputs/validate_gain.png")


if __name__ == "__main__":
    main()
