"""Stage D: tactical states from the GNN vectors (embeddings), and how far each match is from real football.

Every Key State has a vector (made by train_gnn.py) that summarises the arrangement of the 22 players and the ball
and how it has been moving. Here we
  1. group the vectors of the REAL matches into k tactical states (k-means) and check how well the groups agree with
     things the model never saw: the ball being dead, and the sequence type (open play / stoppage);
  2. put the simulated Key States into the nearest real tactical state, to compare the shares;
  3. measure novelty: for each match, the distance from each Key State to the closest Key State in the OTHER real
     matches. A simulated match that is much further away than a real one is playing a different kind of football.

Input : <name>_embeddings.npz (train_gnn.py --save-embeddings), <name>_graphs.npz, <name>_sequence_types.csv
Output: <name>_tactical_state.csv, outputs/tactical_states.png, printed tables

Usage :  python scripts\\tactical_states.py
         python scripts\\tactical_states.py --k 5
"""
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import settings
from scipy.cluster.vq import kmeans2

REALS = [m for m in settings.available(settings.REAL) if __import__("os").path.exists(settings.base(m) + "_embeddings.npz")]
SIMS = [m for m in settings.available(settings.SIM) if __import__("os").path.exists(settings.base(m) + "_embeddings.npz")]
PATH = {m: settings.base(m) for m in REALS + SIMS}
CSV = {m: settings.csv(m) for m in REALS + SIMS}


def load(m):
    g = np.load(settings.graphs(m))
    emb = np.load(PATH[m] + "_embeddings.npz")["emb"]
    ok = ~np.isnan(emb).any(axis=1)                                         # the first Key States have no full window
    mode = pd.read_csv(CSV[m], usecols=["frame", "game_mode"]).drop_duplicates("frame").set_index("frame").loc[g["frame"], "game_mode"].to_numpy()
    seq_type = pd.read_csv(PATH[m] + "_sequence_types.csv").set_index("seq_id")["type"]
    present = g["present"][:, :22]
    speed = np.linalg.norm(g["vel"][:, :22], axis=2)
    speed = np.where(present, speed, np.nan)
    ys = np.where(present, g["xy"][:, :22, 1], np.nan)
    return dict(match=m, emb=emb[ok], time_s=g["time_s"][ok], ks_id=g["ks_id"][ok],
                dead=(mode != "normal")[ok], seq_type=pd.Series(g["seq_id"]).map(seq_type).to_numpy()[ok],
                speed=np.nanmean(speed, axis=1)[ok], width=(np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1))[ok],
                ball_x=g["xy"][:, 22, 0][ok])


def adjusted_rand(x, y):
    table = pd.crosstab(np.asarray(x), np.asarray(y)).to_numpy()
    comb = lambda n: n * (n - 1) / 2
    sij, sa, sb = comb(table).sum(), comb(table.sum(axis=1)).sum(), comb(table.sum(axis=0)).sum()
    exp, mx = sa * sb / comb(table.sum()), (sa + sb) / 2
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


def silhouette(Z, lab, n=2500, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(n, len(Z)), replace=False)
    Z, lab = Z[idx], lab[idx]
    tz = torch.from_numpy(Z).float()
    D = torch.cdist(tz, tz).numpy()                                         # pairwise distances without a huge intermediate array
    s = []
    for i in range(len(Z)):
        same = lab == lab[i]
        if same.sum() < 2 or len(set(lab)) < 2:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, lab == c].mean() for c in set(lab) if c != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def fit(Z, k, tries=8):
    best = None
    for seed in range(tries):
        cent, lab = kmeans2(Z, k, minit="++", seed=seed)
        if len(set(lab)) < k:
            continue
        inertia = ((Z - cent[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, cent, lab)
    return best[1], best[2]


def nearest_dist(query, ref, n_q=1500, n_r=4000, seed=0):
    rng = np.random.default_rng(seed)
    q = torch.from_numpy(query[rng.choice(len(query), min(n_q, len(query)), replace=False)]).float()
    r = torch.from_numpy(ref[rng.choice(len(ref), min(n_r, len(ref)), replace=False)]).float()
    return float(torch.cdist(q, r).min(dim=1).values.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=8)
    a = ap.parse_args()

    data = {m: load(m) for m in REALS + SIMS}
    real_emb = np.concatenate([data[m]["emb"] for m in REALS])
    mu, sd = real_emb.mean(axis=0), real_emb.std(axis=0) + 1e-6
    for d in data.values():
        d["z"] = (d["emb"] - mu) / sd
    Zr = np.concatenate([data[m]["z"] for m in REALS])
    dead_r = np.concatenate([data[m]["dead"] for m in REALS])
    type_r = np.concatenate([data[m]["seq_type"] for m in REALS])
    print(f"{len(Zr)} real Key States, vector size {Zr.shape[1]}\n")

    print("1. How many tactical states? (silhouette: how clearly they separate; agreement with what the model never saw)")
    rows = []
    for k in range(2, a.kmax + 1):
        _, lab = fit(Zr, k)
        rows.append({"k": k, "silhouette": round(silhouette(Zr, lab), 3),
                     "agrees_with_dead_ball": round(adjusted_rand(lab, dead_r), 2),
                     "agrees_with_sequence_type": round(adjusted_rand(lab, type_r), 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    cent, lab = fit(Zr, a.k)
    order = pd.Series(np.concatenate([data[m]["speed"] for m in REALS])).groupby(lab).mean().sort_values().index.tolist()
    remap = {old: new + 1 for new, old in enumerate(order)}
    cent = cent[order]
    off = 0
    for m in REALS:
        n = len(data[m]["z"])
        data[m]["state"] = np.array([remap[x] for x in lab[off:off + n]])
        off += n
    for m in SIMS:
        data[m]["state"] = np.linalg.norm(data[m]["z"][:, None] - cent[None], axis=2).argmin(axis=1) + 1

    print(f"\n2. The {a.k} tactical states (numbered from slowest to fastest players)")
    allreal = pd.DataFrame({"state": np.concatenate([data[m]["state"] for m in REALS]), "speed": np.concatenate([data[m]["speed"] for m in REALS]),
                            "dead": dead_r, "width": np.concatenate([data[m]["width"] for m in REALS]),
                            "ball_x": np.concatenate([data[m]["ball_x"] for m in REALS]), "type": type_r})
    prof = []
    for s in range(1, a.k + 1):
        r = allreal[allreal.state == s]
        row = {"state": s, "real_%": round(100 * len(r) / len(allreal), 1)}
        for m in REALS + SIMS:
            row[m] = round(100 * (data[m]["state"] == s).mean(), 1)
        row.update({"player_m/s": round(r.speed.mean(), 1), "ball_dead_%": round(100 * r.dead.mean()), "team_width_m": round(r.width.mean()),
                    "seq_type": f"{int(r['type'].mode().iloc[0])} ({round(100 * (r['type'] == r['type'].mode().iloc[0]).mean())}%)"})
        prof.append(row)
    print(pd.DataFrame(prof).drop(columns=["real_%"]).to_string(index=False))

    print("\n3. Novelty: average distance from a Key State to the closest Key State in the OTHER real matches (larger = more unusual)")
    nov = []
    for m in REALS + SIMS:
        others = np.concatenate([data[x]["z"] for x in REALS if x != m])
        nov.append({"match": m, "kind": "real" if m in REALS else "simulated", "distance": round(nearest_dist(data[m]["z"], others), 2)})
    print(pd.DataFrame(nov).to_string(index=False))

    for m in REALS + SIMS:
        pd.DataFrame({"ks_id": data[m]["ks_id"], "time_s": data[m]["time_s"], "tactical_state": data[m]["state"]}).to_csv(PATH[m] + "_tactical_state.csv", index=False)

    # picture: the real and simulated Key States on the first two directions of variation, and the shares per state
    U, S, Vt = np.linalg.svd(Zr - Zr.mean(axis=0), full_matrices=False)
    P = lambda z: (z - Zr.mean(axis=0)) @ Vt[:2].T
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    rng = np.random.default_rng(0)
    pr = P(Zr)
    sel = rng.choice(len(pr), min(5000, len(pr)), replace=False)
    ps = P(np.concatenate([data[m]["z"] for m in SIMS]))
    ax[0].scatter(pr[sel, 0], pr[sel, 1], s=4, color="#1B7F4B", alpha=0.35, label="real")
    ax[0].scatter(ps[:, 0], ps[:, 1], s=6, color="#E8833A", alpha=0.6, label="simulated")
    ax[0].set_title("Every Key State as a point (2 main directions of the vectors)", loc="left", fontsize=10)
    ax[0].legend(frameon=False)
    shares = pd.DataFrame({m: [100 * (data[m]["state"] == s).mean() for s in range(1, a.k + 1)] for m in REALS + SIMS}, index=range(1, a.k + 1))
    shares.plot(kind="bar", ax=ax[1], width=0.85, color=settings.palette(len(REALS), len(SIMS)))
    ax[1].set_title("Share of Key States in each tactical state (%)", loc="left", fontsize=10)
    ax[1].set_xlabel("tactical state (1 = slowest players)")
    ax[1].legend(frameon=False, fontsize=7)
    plt.xticks(rotation=0)
    for x in ax:
        for side in ("right", "top"):
            x.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/tactical_states.png", dpi=150)
    print("\nsaved outputs/tactical_states.png and *_tactical_state.csv")


if __name__ == "__main__":
    main()
