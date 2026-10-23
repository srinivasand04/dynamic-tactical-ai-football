"""Group sequences into data-derived TYPES from several real matches, test whether the types are stable,
and compare the simulator against them.

Features come from geometry only (no possession, no dead-ball flag, no events):
    duration, Key-State rate, ball speed, ball displacement, distance of the ball from the halfway line,
    mean player speed, team width / depth, distance between the two team centres.
Types are found by k-means on ALL real matches pooled. Simulated sequences are then assigned to the nearest real type.

Stability checks
    * per-match agreement : cluster each real match on its own and compare with the pooled types (adjusted Rand index)
    * leave-one-match-out : cluster two real matches, assign the third, compare with the pooled types (adjusted Rand index)
      (1 = identical grouping, 0 = no better than chance)

Football descriptions (possession, dead-ball share, what starts / ends a sequence) are added AFTERWARDS,
only to read the types.

Input : <name>_sequences.csv, <name>_ks_features.csv, <name>_sequence_desc.csv for each match
Output: <name>_sequence_types.csv for each match, outputs/sequence_types.png, a printed summary

Usage :  python scripts\\cluster_sequences.py
         python scripts\\cluster_sequences.py --k 4
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import settings
from scipy.cluster.vq import kmeans2

REALS = [settings.base(m) for m in settings.available(settings.REAL)]
SIMS = [settings.base(m) for m in settings.available(settings.SIM)]
CLIP = settings.get("types", "clip", default=3.0)
FEATS = ["log_duration", "log_ks_rate", "log_ball_speed", "log_ball_dx", "ball_off_centre",
         "mean_speed", "width", "depth", "centre_gap"]


def short(name):
    return name.split("/")[-1]


def sequence_features(name):
    ks = pd.read_csv(name + "_ks_features.csv")
    seq = pd.read_csv(name + "_sequences.csv")
    t = ks["time_s"].to_numpy()
    t_next = np.r_[t[1:], t[-1] + ks["gap_s"].median()]
    rows = []
    for r in seq.itertuples():
        a, b = r.start_time_s, r.end_time_s
        i0 = max(0, np.searchsorted(t, a, side="right") - 1)
        i1 = max(i0, np.searchsorted(t, b, side="left") - 1)
        idx = np.arange(i0, i1 + 1)
        w = np.clip(np.minimum(b, t_next[idx]) - np.maximum(a, t[idx]), 0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(idx)) / len(idx)
        k = ks.iloc[idx]
        wm = lambda s: float(np.sum(w * s.to_numpy()))
        n_ks = int(((t >= a) & (t < b)).sum())
        rows.append({
            "match": short(name), "seq_id": r.seq_id, "start_s": a, "end_s": b, "duration_s": r.duration_s, "n_ks": n_ks,
            "log_duration": np.log1p(r.duration_s),
            "log_ks_rate": np.log1p(n_ks / r.duration_s),
            "log_ball_speed": np.log1p(wm(k["ball_speed"])),
            "log_ball_dx": np.log1p(abs(k["ball_x_m"].iloc[-1] - k["ball_x_m"].iloc[0])),
            "ball_off_centre": wm((k["ball_x_m"] - 52.5).abs()),
            "mean_speed": wm(k["mean_speed"].fillna(0)),
            "width": wm((k["home_width"] + k["away_width"]) / 2),
            "depth": wm((k["home_depth"] + k["away_depth"]) / 2),
            "centre_gap": wm((k["home_cx"] - k["away_cx"]).abs()),
        })
    return pd.DataFrame(rows)


def silhouette(Z, labels):
    D = np.linalg.norm(Z[:, None] - Z[None], axis=2)
    s = []
    for i in range(len(Z)):
        same = labels == labels[i]
        if same.sum() < 2:
            s.append(0)
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, labels == c].mean() for c in set(labels) if c != labels[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def adjusted_rand(x, y):
    """Agreement of two groupings of the same items; 1 = identical, about 0 = chance."""
    x, y = np.asarray(x), np.asarray(y)
    table = pd.crosstab(x, y).to_numpy()
    comb = lambda n: n * (n - 1) / 2
    sum_ij = comb(table).sum()
    sum_a, sum_b = comb(table.sum(axis=1)).sum(), comb(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb(len(x))
    maximum = (sum_a + sum_b) / 2
    return float((sum_ij - expected) / (maximum - expected)) if maximum != expected else 1.0


def fit(Z, k, tries=15):
    best = None
    for seed in range(tries):
        cent, lab = kmeans2(Z, k, minit="++", seed=seed)
        if len(set(lab)) < k:
            continue
        inertia = ((Z - cent[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, cent, lab)
    return best[1], best[2]


def assign(Z, cent):
    D = np.linalg.norm(Z[:, None] - cent[None], axis=2)
    return D.argmin(axis=1), D.min(axis=1)


def stability(Zr, real, k):
    """Per real match: agreement (adjusted Rand index) of its own clustering and of a leave-one-match-out clustering
    with the pooled clustering. Returns a table."""
    _, pooled = fit(Zr, k)
    rows = []
    for n in REALS:
        m = short(n)
        sel = (real["match"] == m).to_numpy()
        _, own = fit(Zr[sel], k)                                   # this match clustered on its own
        c_other, _ = fit(Zr[~sel], k)                              # the other matches clustered, then assign this one
        loo, _ = assign(Zr[sel], c_other)
        rows.append({"match": m, "own_clustering": round(adjusted_rand(own, pooled[sel]), 2),
                     "leave_one_out": round(adjusted_rand(loo, pooled[sel]), 2)})
    return pd.DataFrame(rows)


def first_labels(series, n=2):
    items = series.fillna("").str.split("; ").explode().replace("", "no event nearby")
    items = items[~items.str.startswith("possession")]
    return ", ".join(f"{k} ({v})" for k, v in items.value_counts().head(n).items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--kmax", type=int, default=8)
    a = ap.parse_args()

    real = pd.concat([sequence_features(n) for n in REALS], ignore_index=True)
    mu, sd = real[FEATS].mean(), real[FEATS].std()
    Zr = ((real[FEATS] - mu) / sd).clip(-CLIP, CLIP).to_numpy()      # clip: a few extreme sequences must not form their own type
    print("real matches:", ", ".join(f"{short(n)} ({(real['match'] == short(n)).sum()} sequences)" for n in REALS), "\n")

    print("How many types? silhouette = how clearly they separate; stability = how well they repeat across matches")
    print("(adjusted Rand index, mean over the real matches; 1 = identical grouping, about 0 = chance)")
    table = []
    for kk in range(2, a.kmax + 1):
        _, lab = fit(Zr, kk)
        st = stability(Zr, real, kk)
        table.append({"k": kk, "silhouette": round(silhouette(Zr, lab), 3),
                      "own_clustering": round(st["own_clustering"].mean(), 2), "leave_one_out": round(st["leave_one_out"].mean(), 2)})
    table = pd.DataFrame(table)
    print(table.to_string(index=False))
    stable = table[(table["own_clustering"] >= 0.8) & (table["leave_one_out"] >= 0.8)]
    k = a.k or (int(stable["k"].max()) if len(stable) else 2)
    print(f"-> using k = {k}  (the most types that still repeat across matches, unless --k is given)\n")

    cent, lab = fit(Zr, k)
    order = real.assign(t=lab).groupby("t")["duration_s"].median().sort_values().index.tolist()
    remap = {old: new + 1 for new, old in enumerate(order)}
    pooled = np.array([remap[x] for x in lab])
    real["type"] = pooled
    real["dist_to_type"] = np.linalg.norm(Zr - cent[lab], axis=1)
    cent = cent[order]

    # ---------- stability ----------
    print(f"STABILITY of the k = {k} types, per match (adjusted Rand index)")
    print(stability(Zr, real, k).to_string(index=False), "\n")

    allseq = {short(n): real[real["match"] == short(n)].copy() for n in REALS}
    for s in SIMS:
        df = sequence_features(s)
        Z = ((df[FEATS] - mu) / sd).clip(-CLIP, CLIP).to_numpy()
        idx, dist = assign(Z, cent)
        df["type"] = idx + 1
        df["dist_to_type"] = dist
        allseq[short(s)] = df

    for key, df in allseq.items():
        path = next(p for p in REALS + SIMS if short(p) == key)
        desc = pd.read_csv(path + "_sequence_desc.csv")
        full = df.merge(desc[["seq_id", "ball_dead_%", "possession", "possession_%", "ball_advance_m",
                              "starts_after", "ends_with"]], on="seq_id")
        allseq[key] = full
        full.round(3).to_csv(path + "_sequence_types.csv", index=False)

    # ---------- the types ----------
    realdf = pd.concat([allseq[short(n)] for n in REALS])
    rows = []
    for ty in range(1, k + 1):
        r = realdf[realdf["type"] == ty]
        row = {"type": ty}
        for key, df in allseq.items():
            row[f"{key}_%"] = round(100 * (df["type"] == ty).mean(), 1)
        row.update({"dur_s": round(r["duration_s"].median(), 1), "ball_m/s": round(np.expm1(r["log_ball_speed"]).median(), 1),
                    "player_m/s": round(r["mean_speed"].median(), 1), "width_m": round(r["width"].median()),
                    "ball_dead_%": round(r["ball_dead_%"].median()), "advance_m": round(r["ball_advance_m"].median())})
        rows.append(row)
    print("Share of each match's sequences in each type (%), and what the type looks like (real, pooled):")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nWhat starts and ends each type (real, football description):")
    for ty in range(1, k + 1):
        r = realdf[realdf["type"] == ty]
        print(f"  type {ty}: starts after {first_labels(r['starts_after'])}  |  ends with {first_labels(r['ends_with'])}")

    far = np.percentile(real["dist_to_type"], 90)
    for s in SIMS:
        x = allseq[short(s)]
        print(f"{short(s)}: {100 * (x['dist_to_type'] > far).mean():.0f}% of sequences are further from their type than 90% of real ones")

    shares = pd.DataFrame({key: df["type"].value_counts(normalize=True).reindex(range(1, k + 1)).fillna(0) * 100
                           for key, df in allseq.items()})
    colors = ["#14532D", "#2E7D4F", "#4F9D69", "#9CC5A1", "#E8833A", "#EFA566", "#F2C14E", "#2F6DB5"]
    ax = shares.plot(kind="bar", figsize=(11, 4.4), color=settings.palette(len(REALS), len(SIMS)), width=0.85)
    ax.set_xlabel("sequence type (numbered by median duration, shortest first)")
    ax.set_ylabel("share of sequences (%)")
    ax.set_title("Which kinds of sequences occur: real vs simulated matches", loc="left")
    ax.legend(frameon=False, fontsize=8)
    plt.xticks(rotation=0)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/sequence_types_8matches.png", dpi=150)
    print("\nsaved outputs/sequence_types_8matches.png and *_sequence_types.csv")


if __name__ == "__main__":
    main()

