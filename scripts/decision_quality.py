"""Steps 7 and 8 of the pass work: how GOOD are the real players' passing decisions, and do the simulated players decide like them?

For every pass the three models of fit_pass_model.py give every teammate three numbers:
    likely   the chance the passer picks him                       (model A)
    arrives  the chance the pass arrives                           (model B)
    danger   how dangerous the arrival point is                    (model C)
    gain     arrives x danger                                      the expected gain of the option
Then, for the option the player really chose:
    regret          the gain of the best option (offside ones excluded) minus the gain of the chosen one, as % of the best
    chose_best_%    the chosen teammate is the best option
    safe_vs_best    the chance of arriving of the chosen pass vs of the best-gain pass: real players choose safer passes than the
                    ones with the best expected gain?  (a low-risk habit shows here)
    offside_cost_%  the best option overall was offside (the signed weight idea of the professor: how much the offside rule costs)
Models are applied to REAL matches in a leave-one-match-out way (never to a match they learned from), and to the SIMULATED games
as learned on all real matches. 'predicted_%' says how often the real-player model guesses the simulator's choice: if simulated
players decide like real ones it is as high as for real matches.

Output : outputs/decision_quality.csv (per group), outputs/decision_quality_matches.csv, outputs/decision_quality.png, run_log.csv
Usage  :  python scripts\\decision_quality.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit_pass_model as F
import pass_features as P
import runlog
import settings

MEASURES = ["passes", "predicted_%", "predicted_top3_%", "completed_%", "pass_length_m", "arrives_chosen", "arrives_best", "danger_chosen",
            "danger_best", "chose_best_%", "regret_%", "offside_cost_%", "forward_gain_m"]


def softmax_rows(s):
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def measure(m, data, models):
    d = data[m]
    I, M, T, C = d["I"], d["mask"], d["target"], d["complete"]
    keep = C
    I, M, T, dest, off = I[keep], M[keep], T[keep], d["dest"][keep], d["offside"][keep]
    n = len(T)
    ar = np.arange(n)
    s = F.choice_scores(models[0], I, M, F.FULL)
    rank = 1 + (s > s[ar, T][:, None]).sum(1)
    arrives = F.completion_prob(models[1], I)
    danger = F.value_at(models[2], dest[..., 0], dest[..., 1])
    gain = np.where(M, arrives * danger, -1.0)
    on = M & ~off
    gain_on = np.where(on, gain, -1.0)
    best = gain_on.argmax(1)
    best_all = gain.argmax(1)
    g_ch, g_best = gain[ar, T], gain_on[ar, best]
    regret = np.maximum(g_best - g_ch, 0) / np.maximum(g_best, 1e-9)
    dist = I[..., F.COL["dist"]] * 10
    return {"passes": n, "predicted_%": 100 * float((rank <= 1).mean()), "predicted_top3_%": 100 * float((rank <= 3).mean()),
            "pass_length_m": float(dist[ar, T].mean()), "arrives_chosen": float(arrives[ar, T].mean()), "arrives_best": float(arrives[ar, best].mean()),
            "danger_chosen": float(danger[ar, T].mean()), "danger_best": float(danger[ar, best].mean()),
            "chose_best_%": 100 * float((T == best).mean()), "regret_%": 100 * float(regret.mean()),
            "offside_cost_%": 100 * float(off[ar, best_all].mean()),
            "forward_gain_m": float((I[..., F.COL["adv"]] * 20)[ar, T].mean())}


def main():
    real = [m for m in settings.available(settings.REAL) if os.path.exists(settings.base(m) + "_passfeat.npz")]
    sim = [m for m in settings.available(settings.SIM) if os.path.exists(settings.base(m) + "_passfeat.npz")]
    if len(real) < 2:
        raise SystemExit("run scripts/pass_features.py and scripts/fit_pass_model.py first")
    data = F.load(real + sim)
    rows = []
    print(f"{len(real)} real matches, {len(sim)} simulated games\n")
    for m in real:
        tr = [x for x in real if x != m]
        models = (F.fit_choice(F.cat(data, tr, "I", True), F.cat(data, tr, "mask", True), F.cat(data, tr, "target", True), F.FULL),
                  F.fit_completion(data, tr), F.from_json()[2])
        z = np.load(settings.base(m) + "_passes.npz", allow_pickle=True)       # every open-play pass, not only the ones with an inferred target
        rows.append({"match": m, "group": "real", **measure(m, data, models),
                     "completed_%": 100 * float(z["complete"][~z["set_piece"]].mean())})
        print(f"  {m} done", flush=True)
    models = F.from_json()
    for m in sim:
        z = np.load(settings.base(m) + "_passfeat.npz")
        n_cut = int(z["n_intercept"])
        n_ok = len(z["target"])
        rows.append({"match": m, "group": settings.variant(m), **measure(m, data, models),
                     "completed_%": 100 * n_ok / max(n_ok + n_cut, 1)})
    df = pd.DataFrame(rows).set_index("match")
    df.round(3).to_csv("outputs/decision_quality_matches.csv")
    g = df.groupby("group")[MEASURES].mean()
    g["passes"] = df.groupby("group")["passes"].sum()
    g["games"] = df.groupby("group").size()
    order = ["real"] + [x for x in g.index if x != "real"]
    g = g.loc[order]
    print("\nDECISION QUALITY (group averages over matches)\n")
    print(g.round(2).T.to_string())
    g.round(3).to_csv("outputs/decision_quality.csv")

    print("\nREAD THE TABLE")
    r = g.loc["real"]
    print(f"  real players pick the option with the best expected gain in {r['chose_best_%']:.0f}% of passes (they lose {r['regret_%']:.0f}% of the best gain on average)")
    print(f"  their pass arrives with chance {r['arrives_chosen']:.2f}, the best-gain pass would arrive with {r['arrives_best']:.2f}: "
          f"{'they play safer than the best-gain option' if r['arrives_chosen'] > r['arrives_best'] else 'they take more risk than the best-gain option'}")
    print(f"  the best option overall was offside in {r['offside_cost_%']:.1f}% of passes")
    print("  (choice measures use completed passes only, for real and simulated alike; 'completed_%' uses all passes)")

    groups = list(g.index)
    colors = ["#2a78d6"] + ["#eb6834"] * (len(groups) - 1)
    panels = [("predicted_%", "real-player model guesses the choice (%)"), ("completed_%", "passes that arrive (%)"),
              ("arrives_chosen", "chance of arriving, chosen pass"), ("pass_length_m", "pass length (m)"),
              ("regret_%", "regret vs best-gain option (%)"), ("forward_gain_m", "forward gain of the pass (m)")]
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.4))
    for a, (k, title) in zip(ax.ravel(), panels):
        a.bar(groups, g[k], color=colors)
        a.axhline(g.loc["real", k], color="#2a78d6", lw=1.5)
        a.set_title(f"{title}   (blue line = real: {g.loc['real', k]:.2f})", loc="left", fontsize=9)
        a.tick_params(axis="x", labelsize=8)
        for side in ("right", "top"):
            a.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/decision_quality.png", dpi=150)
    runlog.log({"script": "decision_quality", **{f"{x}_predicted_%": round(float(g.loc[x, 'predicted_%']), 1) for x in groups}})
    print("\nsaved outputs/decision_quality.csv, outputs/decision_quality_matches.csv, outputs/decision_quality.png")


if __name__ == "__main__":
    main()
