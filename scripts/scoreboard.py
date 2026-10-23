"""The scoreboard: every quantifier of the project in one table, each with a 95% confidence interval over matches.

Why: "how do we know we are doing well in prediction and tactics?" needs numbers that (1) are measured on matches the model
never saw, (2) are compared with an honest baseline, and (3) say how sure we are. Every row below has the three.

How the interval is made: each quantifier is measured once per held-out real match (leave-one-match-out). The 7 per-match numbers
are resampled with replacement 5000 times and the 2.5% / 97.5% points of the resampled average are the interval (a bootstrap over
matches, not over frames, because frames of one match are not independent). With 7 matches the interval is honest but wide; with
fewer than 5 matches it is printed but marked 'few'.

Reads whatever exists and skips what does not (it says which script to run):
    outputs/run_log.csv                     movement of the network, per held-out match (run_folds.py)
    outputs/pass_model.csv                  who the passer picks (fit_pass_model.py)
    outputs/pass_model_completion.csv       will the pass arrive (fit_pass_model.py)
    outputs/validate_gain_matches.csv       does the gain predict what happens next (validate_gain.py)
    outputs/tactics_validity.csv            do tactical states predict what happens next (tactics_profile.py)
    outputs/realism_scorecard.csv           gap of each simulator setting to real football (realism_scorecard.py)

Output : the table, outputs/scoreboard.csv, outputs/scoreboard.png
Usage  :  python scripts\\scoreboard.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
BLUE, GRAY = "#2a78d6", "#b4b2a9"
ROWS, PANELS = [], []


def ci(x, n=5000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    means = RNG.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(x)


def add(layer, quantifier, ours_label, ours, baselines, better="higher", note=""):
    """ours: per-match values; baselines: {label: per-match values}. Adds table rows."""
    m, lo, hi, n = ci(ours)
    row = {"layer": layer, "quantifier": quantifier, "who": ours_label, "mean": m, "ci_low": lo, "ci_high": hi, "matches": n, "note": note + (" (few matches)" if n < 5 else "")}
    ROWS.append(row)
    for lab, v in baselines.items():
        bm, blo, bhi, bn = ci(v)
        ROWS.append({"layer": layer, "quantifier": quantifier, "who": lab, "mean": bm, "ci_low": blo, "ci_high": bhi, "matches": bn, "note": "baseline"})
    PANELS.append((f"{layer}: {quantifier}", [(ours_label, ours)] + list(baselines.items())))


def need(path, script):
    if os.path.exists(path):
        return True
    print(f"  (skipped: {path} not found; run {script})")
    return False


def movement():
    path = "outputs/run_log.csv"
    if not need(path, "scripts/run_folds.py"):
        return
    log = pd.read_csv(path)
    log = log[(log["script"] == "train_gnn") & log["holdout"].notna() & (log["holdout"] != "none")].copy()
    log = log[~log["tag"].astype(str).str.contains("test|smoke", case=False)]
    log["arch"] = log["arch"].fillna("ours") if "arch" in log.columns else "ours"
    log = log.sort_values("time").drop_duplicates(["arch", "holdout", "seed"], keep="last")
    per = {a: g.groupby("holdout")[["real_player_model_m", "real_player_keepvel_m", "real_ball_model_m", "real_ball_keepvel_m"]].mean() for a, g in log.groupby("arch")}
    if "ours" not in per:
        print("  (skipped movement: no 'ours' folds in the run log)")
        return
    o = per["ours"]
    add("Movement", "player error on an unseen match (m, lower is better)", "network (ours)", o["real_player_model_m"],
        {"keep your velocity": o["real_player_keepvel_m"]}, "lower", "baseline: everybody keeps his velocity")
    add("Movement", "gain over keep-velocity (%)", "network (ours)", 100 * (1 - o["real_player_model_m"] / o["real_player_keepvel_m"]), {}, "higher")
    add("Movement", "ball error on an unseen match (m, lower is better)", "network (ours)", o["real_ball_model_m"], {"keep your velocity": o["real_ball_keepvel_m"]}, "lower")
    for arch, label in [("mlp", "no graph (control)"), ("gcn", "GCN"), ("gat", "GAT"), ("pass_raw", "ours + pass edges"), ("pass_shuffle", "ours + shuffled pass edges")]:
        if arch not in per:
            continue
        common = o.index.intersection(per[arch].index)
        if len(common) < 2:
            continue
        d = per[arch].loc[common, "real_player_model_m"] - o.loc[common, "real_player_model_m"]
        m, lo, hi, n = ci(d)
        ROWS.append({"layer": "Movement", "quantifier": f"player error: {label} minus ours (m; negative = {label} better)", "who": f"{label} - ours",
                     "mean": m, "ci_low": lo, "ci_high": hi, "matches": n, "note": "paired over the same held-out matches" + (" (few matches)" if n < 5 else "")})


def passes():
    if need("outputs/pass_model.csv", "scripts/fit_pass_model.py"):
        d = pd.read_csv("outputs/pass_model.csv")
        g = lambda model, col: d[d["model"] == model].set_index("held_out")[col]
        for col, name, better in [("top1_%", "who the passer picks: top-1 (%)", "higher"), ("top3_%", "who the passer picks: top-3 (%)", "higher"),
                                  ("log_loss", "who the passer picks: log-loss (lower is better)", "lower")]:
            add("Passes", name, "pass model (with movement)", g("+ running (all)", col),
                {"nearest teammate": g("nearest teammate", col), "chance": g("chance", col), "still picture only": g("still picture", col)}, better)
    if need("outputs/pass_model_completion.csv", "scripts/fit_pass_model.py"):
        c = pd.read_csv("outputs/pass_model_completion.csv").set_index("held_out")
        add("Passes", "will the pass arrive: AUC (0.5 = guessing)", "pass model (all numbers)", c["AUC all numbers"], {"pass length only": c["AUC pass length only"]})
        add("Passes", "will the pass arrive: Brier score (lower is better)", "pass model (all numbers)", c["brier"], {"always the average": c["brier_always_mean"]}, "lower")
    if need("outputs/validate_gain_matches.csv", "scripts/validate_gain.py"):
        v = pd.read_csv("outputs/validate_gain_matches.csv")
        g = lambda pred, out: v[(v["predictor"] == pred) & (v["outcome"] == out)].set_index("match")["AUC"]
        add("Passes", "does a pass lead to a shot within 10 s: AUC", "location + gain of the chosen pass", g("location + gain", "shot"), {"location alone": g("location alone", "shot")})
        add("Passes", "is the ball kept for 10 s: AUC of the chance it arrives", "chance it arrives", g("chance it arrives", "kept"), {"guessing": pd.Series([0.5] * 7)})


def tactics():
    if need("outputs/tactics_validity.csv", "scripts/tactics_profile.py"):
        t = pd.read_csv("outputs/tactics_validity.csv").set_index("match")
        add("Tactics", "does the tactical state predict a shot within 10 s: AUC", "location + tactical state", t["auc_location_and_state"],
            {"location alone": t["auc_location"]}, note="Key States with a ball owner")


def realism():
    if need("outputs/realism_scorecard.csv", "scripts/realism_scorecard.py"):
        r = pd.read_csv("outputs/realism_scorecard.csv", index_col=0)
        if "gap_to_real_%" in r:
            for grp, gap in r["gap_to_real_%"].items():
                if grp != "real":
                    ROWS.append({"layer": "Realism", "quantifier": "average gap to real football (%, lower is better)", "who": f"simulator: {grp}", "mean": float(gap),
                                 "ci_low": np.nan, "ci_high": np.nan, "matches": int(r.loc[grp, "matches"]) if "matches" in r else 0, "note": "descriptive, no interval"})


def panel(ax, title, items):
    dots = any(k in title for k in ("AUC", "Brier", "log-loss"))               # scores that do not start at zero: dots, so no truncated bars
    if not dots:
        ax.bar(range(len(items)), [ci(v)[0] for _, v in items], color=[BLUE] + [GRAY] * (len(items) - 1), width=0.6)
    for i, (_, v) in enumerate(items):
        m, lo, hi, _ = ci(v)
        ax.plot([i, i], [lo, hi], color=BLUE if (dots and i == 0) else "#3d3d3a", lw=2 if dots else 1.5)
        if dots:
            ax.scatter([i], [m], s=60, color=BLUE if i == 0 else GRAY, zorder=3, edgecolor="#3d3d3a", linewidth=0.8)
        ax.text(i + (0.08 if dots else 0), hi if not dots else m, f"{m:.2f}" if abs(m) < 10 else f"{m:.0f}", ha="left" if dots else "center",
                va="center" if dots else "bottom", fontsize=8, color="#3d3d3a")
    if dots:
        ax.set_xlim(-0.5, len(items) - 0.5)
        if "AUC" in title:
            ax.axhline(0.5, color="#73726c", lw=1, ls=(0, (4, 3)))
    ax.set_xticks(range(len(items)), [l.replace(" ", "\n", 1) if len(l) > 12 else l for l, _ in items], fontsize=7)
    ax.set_title(title, loc="left", fontsize=8)
    lo_all = min(np.nanmin(ci(v)[1]) for _, v in items)
    hi_all = max(np.nanmax(ci(v)[2]) for _, v in items)
    pad = (hi_all - lo_all) * 0.15 + 1e-9
    ax.set_ylim(lo_all - 3 * pad if dots else min(0, lo_all - pad), hi_all + 3 * pad)           # bars always start at zero
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)


def main():
    print("collecting quantifiers (each is measured on matches the model never saw)\n")
    movement()
    passes()
    tactics()
    realism()
    if not ROWS:
        raise SystemExit("nothing to report yet: run the scripts named above first")
    df = pd.DataFrame(ROWS)
    os.makedirs("outputs", exist_ok=True)
    df.round(4).to_csv("outputs/scoreboard.csv", index=False)
    last = None
    for r in df.itertuples():
        key = (r.layer, r.quantifier)
        if key != last:
            print(f"\n[{r.layer}] {r.quantifier}")
            last = key
        interval = "" if np.isnan(r.ci_low) else f"  95% CI [{r.ci_low:.3f}, {r.ci_high:.3f}]"
        print(f"    {r.who:38s} {r.mean:8.3f}{interval}   n={r.matches}  {r.note if r.note != 'baseline' else ''}")

    panels = [p for p in PANELS if len(p[1]) > 1]
    if panels:
        cols = 3
        rows_n = -(-len(panels) // cols)
        fig, ax = plt.subplots(rows_n, cols, figsize=(13, 3.1 * rows_n), squeeze=False)
        for a, (title, items) in zip(ax.ravel(), panels):
            panel(a, title, items)
        for a in ax.ravel()[len(panels):]:
            a.axis("off")
        plt.tight_layout()
        plt.savefig("outputs/scoreboard.png", dpi=150)
    print("\nblue = ours, gray = baseline, black line = 95% interval over held-out matches")
    print("saved outputs/scoreboard.csv and outputs/scoreboard.png")


if __name__ == "__main__":
    main()
