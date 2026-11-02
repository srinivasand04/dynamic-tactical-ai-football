"""Realism scorecard: how close is each simulator setting to real football?

For real matches and for every simulator setting (the groups in settings.json: stochastic, easy, hard, competition, ...)
the same six measures are computed per match and averaged per group:

    stoppage_scenes_%   share of scenes that are stoppages (the type found in every real match)
    ball_dead_%         share of frames with the ball out of play
    player_speed_m/s    median speed of the outfield players
    key_states_%        share of frames that become Key States (busy play gives more)
    novelty             how far a match's Key States are from real Key States, in the network's vector space
    network_gain_%      how much better the network predicts movement than 'keep your velocity'
                        (real matches: from the 7 hold-out runs; simulator: the network trained on real matches only)
  and, when outputs/decision_quality_matches.csv exists (decision_quality.py), three measures of the passing:
    pass_length_m       average length of a completed pass
    forward_gain_m      how far a pass takes the ball towards the opponent goal
    predicted_%         how often a model learned on real players guesses the receiver (low = decides unlike real players)

'gap to real' for a measure = |group - real| / real (capped at 100%); the scorecard ranks groups by the average gap.
A group with a small gap behaves like real football on these measures.

Needs  : the pipeline steps for every match (run_pipeline.py), and outputs/gnn_model.pt from
         train_gnn.py --holdout none --save-embeddings  (the network trained on all real matches)
Output : the table, outputs/realism_scorecard.csv, outputs/realism_scorecard.png, a line in outputs/run_log.csv

Usage :  python scripts\\realism_scorecard.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import runlog
import settings
import train_gnn as T

REAL = [m for m in settings.available(settings.REAL) if os.path.exists(settings.graphs(m))]
SIM = [m for m in settings.available(settings.SIM) if os.path.exists(settings.graphs(m))
       and os.path.exists(settings.base(m) + "_sequence_types.csv")]
MEASURES = ["stoppage_scenes_%", "ball_dead_%", "player_speed_m/s", "key_states_%", "novelty", "network_gain_%"]
PASS_MEASURES = ["pass_length_m", "forward_gain_m", "predicted_%"]
PASS_FILE = "outputs/decision_quality_matches.csv"


def load_model():
    d = T.load(REAL[0])
    model = T.Net(d["feat"].shape[2], settings.get("gnn", "dim", default=48), "ours")
    model.load_state_dict(torch.load("outputs/gnn_model.pt"))
    model.eval()
    return model


def embeddings(model, m):
    """One 48-number vector per Key State, from the network trained on all real matches."""
    d = T.load(m)
    K = len(d["feat"])
    ks = np.arange(T.L - 1, K)
    out = []
    with torch.no_grad():
        for i in range(0, len(ks), 256):
            kk = np.minimum(ks[i:i + 256], K - 2)
            _, z = model(T.batch(d, kk))
            pres = torch.from_numpy(d["present"][kk].astype(np.float32))[..., None]
            out.append(((z * pres).sum(1) / pres.sum(1)).numpy())
    return np.concatenate(out), d


def nearest(query, ref, n_q=1500, n_r=6000, seed=0):
    rng = np.random.default_rng(seed)
    q = torch.from_numpy(query[rng.choice(len(query), min(n_q, len(query)), replace=False)]).float()
    r = torch.from_numpy(ref[rng.choice(len(ref), min(n_r, len(ref)), replace=False)]).float()
    return float(torch.cdist(q, r).min(dim=1).values.mean())


def basics(m):
    c = pd.read_csv(settings.csv(m), usecols=["frame", "role", "vx_ms", "vy_ms", "game_mode"])
    frames = c.drop_duplicates("frame")
    dead = 100 * float((frames["game_mode"] != "normal").mean())
    out = c[c["role"] != "GK"]
    sp = np.hypot(out["vx_ms"], out["vy_ms"])
    speed = float(sp[(sp < 12)].median())
    ks = pd.read_csv(settings.base(m) + "_adaptive_ks.csv")
    types = pd.read_csv(settings.base(m) + "_sequence_types.csv")["type"]
    return {"stoppage_scenes_%": 100 * float((types == 2).mean()), "ball_dead_%": dead, "player_speed_m/s": speed,
            "key_states_%": 100 * len(ks) / len(frames), "longest_gap_s": float(ks["gap_s"].max())}


def main():
    model = load_model()
    print(f"{len(REAL)} real matches, {len(SIM)} simulated matches in {len(set(settings.variant(m) for m in SIM))} settings\n")
    rows, emb = [], {}
    for m in REAL + SIM:
        row = {"match": m, "group": settings.variant(m)}
        row.update(basics(m))
        z, d = embeddings(model, m)
        emb[m] = z
        (p, _), _, (pv, _) = T.evaluate(model, d)
        row["network_gain_%"] = 100 * (1 - p / pv)
        rows.append(row)
        print(f"  {m:16s} done", flush=True)

    real_z = np.concatenate([emb[m] for m in REAL])
    mu, sd = real_z.mean(axis=0), real_z.std(axis=0) + 1e-6
    for r in rows:
        m = r["match"]
        ref = np.concatenate([emb[x] for x in REAL if x != m])
        r["novelty"] = nearest((emb[m] - mu) / sd, (ref - mu) / sd)

    df = pd.DataFrame(rows)
    measures = list(MEASURES)
    if os.path.exists(PASS_FILE):
        dq = pd.read_csv(PASS_FILE, index_col="match")
        for k in PASS_MEASURES:
            df[k] = df["match"].map(dq[k])
        measures += PASS_MEASURES
    else:
        print(f"\n({PASS_FILE} not found: the three passing measures are left out; run decision_quality.py to add them)")
    log = pd.read_csv("outputs/run_log.csv")
    folds = log[(log["script"] == "train_gnn") & log["tag"].astype(str).str.startswith("fold_") & log["real_player_model_m"].notna()]
    folds = folds.sort_values("time").drop_duplicates("holdout", keep="last")
    real_gain = float((100 * (1 - folds["real_player_model_m"] / folds["real_player_keepvel_m"])).mean())
    df.loc[df["group"] == "real", "network_gain_%"] = real_gain      # held-out value for real matches (the all-real network has seen them)

    g = df.groupby("group")[measures + ["longest_gap_s"]].mean()
    g["matches"] = df.groupby("group").size()
    ref = g.loc["real", measures]
    gaps = pd.DataFrame({k: (100 * (g[k] - ref[k]).abs() / abs(ref[k])).clip(upper=100) for k in measures})
    g["gap_to_real_%"] = gaps.mean(axis=1)
    g = g.sort_values("gap_to_real_%")
    print("\nREALISM SCORECARD (group averages; the last column is the average gap to real football, lower = more realistic)\n")
    print(g.round(2).to_string())
    print("\nGAP PER MEASURE (% away from real)\n")
    print(gaps.loc[g.index].round(0).to_string())

    g.round(3).to_csv("outputs/realism_scorecard.csv")
    rows_n = -(-len(measures) // 3)
    fig, ax = plt.subplots(rows_n, 3, figsize=(13, 3.2 * rows_n), squeeze=False)
    groups = [x for x in g.index if x != "real"]
    for a, k in zip(ax.ravel(), measures):
        vals = [g.loc[x, k] for x in groups]
        a.bar(groups, vals, color="#eb6834")
        a.axhline(g.loc["real", k], color="#2a78d6", lw=2)
        a.set_title(f"{k}   (blue line = real: {g.loc['real', k]:.1f})", loc="left", fontsize=9)
        for side in ("right", "top"):
            a.spines[side].set_visible(False)
        a.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    plt.savefig("outputs/realism_scorecard.png", dpi=150)
    runlog.log({"script": "realism_scorecard", **{f"gap_{k}": round(float(v), 1) for k, v in g["gap_to_real_%"].items() if k != "real"}})
    print("\nsaved outputs/realism_scorecard.csv and outputs/realism_scorecard.png")


if __name__ == "__main__":
    main()
