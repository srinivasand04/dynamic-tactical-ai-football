"""Blame measured inside GRF, and a fast blame model learned from it.

Still-picture blame (who_is_to_blame.py) was not confirmed by GRF (blame_grf_check.py: fixing the most-blamed defender
helped no more than fixing a random one). Here blame is MEASURED by simulation instead:
    1. 40 dangerous real moments (threat >= 20%, at least 20 s apart, at most 6 per match).
    2. Candidates: the keeper + the 6 outfield players of the defending team nearest the ball.
    3. GRF (team models, 12 s, hold): no change, and for each candidate alone 'he starts at his best reachable spot and
       keeps that offset all run' (fix_<slot>). Same 8 seeds for all.
    4. GRF blame of a defender = threat drop when HE is fixed (threat from 1 s while the attack has the ball, pp),
       mean over seeds, with a 95% interval.
    5. Does still-picture blame agree? (same player on top, rank correlation per moment)
    6. Fast blame model: ridge regression from simple features of the still picture (distance to ball, in the shooting
       cone, goal-side, nearest attacker, keeper, distance to own goal, still-picture blame), leave-one-match-out.
       Then every defender in all 999 moments gets a fast blame -> new player table.

Output (outputs/blame/, new files only): true_moments.csv, true_runs.csv, true_blame.csv (one row per candidate),
true_summary.csv, true_model.json, true_players.csv, true_chart.png
Usage : .venv\\Scripts\\python.exe scripts\\blame_grf_true.py --seeds 8 --procs 4        (GRF, about 45 min, resumable)
        .venv-analysis\\Scripts\\python.exe scripts\\blame_grf_true.py --report         (steps 4-6)
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blame_grf_check as BC  # noqa: E402

ROOT, OUT = BC.ROOT, BC.OUT
L, W = 105.0, 68.0
N_NEAR = 6


# ---------------------------------------------------------------- 1-2: moments and candidates

def same_attack(seg, t, defending):
    """GRF scores the team the match data says owns the ball (segment 'owner'); blame used the team nearest the ball.
    Only moments where both agree are fair (otherwise GRF measures the other team's threat)."""
    z = np.load(seg if os.path.isabs(seg) else os.path.join(ROOT, seg), allow_pickle=True)
    i = int(np.argmin(abs(z["time_s"] - t)))
    att = "home" if str(z["owner"][i]) != "away" else "away"            # as whatif.moment decides it
    return att != defending


_OWN = {}


def owner_at(match, t):
    """The match data's ball owner ('home' if not 'away', as whatif.moment reads it) at time t."""
    import settings
    if match not in _OWN:
        d = pd.read_csv(settings.csv(match), usecols=["time_s", "ball_owner_team"]).drop_duplicates("time_s").sort_values("time_s")
        _OWN[match] = (d["time_s"].to_numpy(), d["ball_owner_team"].astype(str).to_numpy())
    ts, ow = _OWN[match]
    return "away" if ow[min(int(np.searchsorted(ts, t)), len(ts) - 1)] == "away" else "home"


def pick(n, per_match):
    import settings
    from football_scorecard import SUFFIX
    df = pd.read_csv(os.path.join(OUT, "blame_moments.csv")).replace([np.inf, -np.inf], np.nan).dropna(subset=["blame_best"])
    mom = df.drop_duplicates(["match", "ks"]).sort_values("threat", ascending=False)
    mom = mom[mom.threat >= 0.20]
    rows, used, kept, fails = [], {}, [], {}
    for r in mom.itertuples():
        if len(kept) >= n:
            break
        if used.get(r.match, 0) >= per_match or any(m == r.match and abs(t - r.time_s) < 20 for m, t in kept):
            continue
        if owner_at(r.match, r.time_s) == r.defending or fails.get(r.match, 0) >= 3:   # check first: cutting is slow
            continue
        seg = BC.ensure_segment(r.match, r.time_s)
        if seg is None:                                                     # e.g. a team down to 10 after a red card
            fails[r.match] = fails.get(r.match, 0) + 1
            continue
        if not same_attack(seg, r.time_s, r.defending):
            continue
        g = np.load(settings.base(r.match) + f"_graphs{SUFFIX}.npz")
        pos = g["xy"][int(r.ks)]
        a0, d0 = (0, 11) if r.defending == "away" else (11, 0)
        flip = np.isfinite(pos[a0, 0]) and pos[a0, 0] > L / 2
        dd = df[(df.match == r.match) & (df.ks == r.ks)]
        cand = pd.concat([dd[dd.role == "keeper"], dd[dd.role != "keeper"].nsmallest(N_NEAR, "dist_ball_m")])
        cs = []
        for c in cand.itertuples():
            best = np.array([c.best_x, c.best_y])
            if flip:
                best = np.array([L, W]) - best
            off = best - pos[d0 + int(c.slot)]
            if int(c.shirt) < 0 or not np.isfinite(off).all() or np.hypot(*off) > 12:
                continue
            cs.append({"moment": f"{r.match}_{int(r.ks)}", "match": r.match, "ks": int(r.ks), "time_s": float(r.time_s),
                       "segment": os.path.relpath(seg, ROOT), "side": r.defending, "threat": r.threat, "slot": int(c.slot),
                       "shirt": int(c.shirt), "name": c.name, "role": c.role, "blame_still": c.blame_best,
                       "dx": round(float(off[0]), 2), "dy": round(float(off[1]), 2)})
        if len(cs) < 4:
            continue
        rows += cs
        kept.append((r.match, r.time_s))
        used[r.match] = used.get(r.match, 0) + 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3: GRF

def jobs_for(cands, seeds, seconds, done):
    import whatif as WI
    out = []
    for m, g in cands.groupby("moment", sort=False):
        r0 = g.iloc[0]
        seg = os.path.join(ROOT, r0.segment)
        z = np.load(seg, allow_pickle=True)
        scen = [("baseline", None)]
        for c in g.itertuples():
            try:
                k, _ = WI.find_player(z, f"{c.side}:{c.shirt}")
            except AssertionError:                                          # came on as a substitute after the stretch starts
                continue
            scen.append((f"fix_{c.slot}", (k, (c.dx, c.dy))))
        for s, mv in scen:
            for seed in range(seeds):
                if (m, s, seed) not in done:
                    out.append((m, seg, r0.time_s, s, 3.0, seed, seconds, mv))
    return out


def run(a):
    cp = os.path.join(OUT, "true_moments.csv")
    if os.path.exists(cp):
        cands = pd.read_csv(cp)
    else:
        print("picking moments and candidates...", flush=True)
        cands = pick(a.n, a.per_match)
        cands.to_csv(cp, index=False)
    if a.only:
        cands = cands[cands.moment == a.only]
    print(f"{cands.moment.nunique()} moments, {len(cands)} candidate defenders, {a.seeds} seeds", flush=True)
    BC._patch()
    runs_p = os.path.join(OUT, "true_runs.csv")
    done_df = pd.read_csv(runs_p) if os.path.exists(runs_p) else pd.DataFrame(columns=["moment", "scenario", "seed"])
    done = set(zip(done_df["moment"], done_df["scenario"], done_df["seed"].astype(int)))
    jobs = jobs_for(cands, a.seeds, a.seconds, done)
    print(f"{len(jobs)} runs to play ({len(done)} already done), {a.procs} at a time", flush=True)
    t0, rows = time.time(), []
    with Pool(a.procs, initializer=BC.init, initargs=(a.dataset, True)) as pool:
        for k, r in enumerate(pool.imap_unordered(BC.job, jobs, chunksize=2)):
            rows.append(r)
            if (k + 1) % 48 == 0 or k + 1 == len(jobs):
                pd.concat([done_df, pd.DataFrame(rows)], ignore_index=True).to_csv(runs_p, index=False)
                el = (time.time() - t0) / 60
                print(f"  {k + 1}/{len(jobs)} runs ({el:.1f} min, about {el / (k + 1) * (len(jobs) - k - 1):.0f} min left)", flush=True)


# ---------------------------------------------------------------- 4-6: report

def features(match, ks, side, slots=None):
    """Still-picture features of the defenders of one moment (attack towards x = 105)."""
    import settings
    import value_models as V
    from football_scorecard import SUFFIX
    from who_is_to_blame import oriented
    g = np.load(settings.base(match) + f"_graphs{SUFFIX}.npz")
    pos = g["xy"][int(ks)]
    team = 0 if side == "away" else 1
    b, att, dfs, _ = oriented(pos, team)
    out = {}
    for j in range(11) if slots is None else slots:
        p = dfs[j]
        if not np.isfinite(p[0]):
            continue
        a_ok = att[np.isfinite(att[:, 0])]
        out[j] = {"dist_ball": float(np.hypot(*(p - b))),
                  "in_cone": float(V.in_triangle(p[None], b, V.POSTS[0], V.POSTS[1])[0]),
                  "goal_side": float(p[0] > b[0]),
                  "near_att": float(np.min(np.hypot(*(a_ok - p).T))) if len(a_ok) else 30.0,
                  "dist_goal": float(np.hypot(*(p - V.GOAL))),
                  "keeper": float(j == 0)}
    return out


FEATS = ["dist_ball", "in_cone", "goal_side", "near_att", "dist_goal", "keeper", "blame_still"]


def ridge_fit(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    I = np.eye(Z.shape[1])
    I[0, 0] = 0
    w = np.linalg.solve(Z.T @ Z + lam * len(X) * I * 0.01, Z.T @ y)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}


def ridge_pred(m, X):
    return np.c_[np.ones(len(X)), (X - np.array(m["mu"])) / np.array(m["sd"])] @ np.array(m["w"])


def spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    ok = a.notna() & b.notna()
    return a[ok].rank().corr(b[ok].rank()) if ok.sum() > 2 else np.nan


def report():
    cands = pd.read_csv(os.path.join(OUT, "true_moments.csv"))
    runs = pd.read_csv(os.path.join(OUT, "true_runs.csv"))
    runs = runs[runs["error"].isna()] if "error" in runs else runs
    base = runs[runs.scenario == "baseline"].set_index(["moment", "seed"])["threat_play"]
    rows = []
    for c in cands.itertuples():
        f = runs[(runs.moment == c.moment) & (runs.scenario == f"fix_{c.slot}")].set_index(["moment", "seed"])["threat_play"]
        common = f.index.intersection(base.index)
        if len(common) < 3:
            continue
        drop = (base.loc[common] - f.loc[common]) * 100                     # pp of threat removed by fixing him
        rows.append({**c._asdict(), "blame_grf": drop.mean(), "blame_grf_ci95": BC.ci(drop), "seeds": len(common)})
    tb = pd.DataFrame(rows).drop(columns=["Index"])
    for (m, ks, side), g in tb.groupby(["match", "ks", "side"]):
        fe = features(m, ks, side, list(g.slot))
        for i, r in g.iterrows():
            for k, v in fe.get(r.slot, {}).items():
                tb.loc[i, k] = v
    tb.round(4).to_csv(os.path.join(OUT, "true_blame.csv"), index=False)

    # 4-5: agreement of still-picture blame with GRF blame
    per = []
    for m, g in tb.groupby("moment"):
        per.append({"moment": m, "same_top": g.blame_grf.idxmax() == g.blame_still.idxmax(), "rho": spearman(g.blame_still, g.blame_grf),
                    "top_grf_role": g.loc[g.blame_grf.idxmax(), "role"], "top_grf_sure": g.loc[g.blame_grf.idxmax(), "blame_grf"] > g.loc[g.blame_grf.idxmax(), "blame_grf_ci95"]})
    per = pd.DataFrame(per)
    sure = (tb.blame_grf.abs() > tb.blame_grf_ci95)
    # 6: fast blame model, leave-one-match-out, against 'still-picture blame alone' and 'nearest to the ball'
    X, y = tb[FEATS].to_numpy(float), tb.blame_grf.to_numpy(float)
    pred = np.full(len(tb), np.nan)
    for m in tb.match.unique():
        tr, te = (tb.match != m).to_numpy(), (tb.match == m).to_numpy()
        pred[te] = ridge_pred(ridge_fit(X[tr], y[tr]), X[te])
    tb["blame_fast_cv"] = pred

    def within(col, sign=1):                   # mean per-moment rank corr. with GRF blame, and top-1 hit rate
        r = [spearman(sign * g[col], g.blame_grf) for _, g in tb.groupby("moment")]
        hit = [(sign * g[col]).idxmax() == g.blame_grf.idxmax() for _, g in tb.groupby("moment")]
        return np.nanmean(r), np.mean(hit)
    comp = []
    for lab, col, sg in (("fast blame model (CV)", "blame_fast_cv", 1), ("still-picture blame", "blame_still", 1),
                         ("nearest to the ball", "dist_ball", -1), ("random guess", None, 1)):
        if col is None:
            k = tb.groupby("moment").size()
            comp.append({"predictor": lab, "rank_corr_in_moment": 0.0, "picks_true_top": float((1 / k).mean())})
            continue
        rr, hh = within(col, sg)
        comp.append({"predictor": lab, "rank_corr_in_moment": rr, "picks_true_top": hh})
    comp = pd.DataFrame(comp)
    model = ridge_fit(X, y)
    model["features"] = FEATS
    json.dump(model, open(os.path.join(OUT, "true_model.json"), "w"), indent=1)
    coef = pd.Series(model["w"][1:], index=FEATS).sort_values()

    # re-rank all 999 moments with the fast model
    allm = pd.read_csv(os.path.join(OUT, "blame_moments.csv")).replace([np.inf, -np.inf], np.nan).dropna(subset=["blame_best"])
    allm = allm.rename(columns={"blame_best": "blame_still"})
    fe_rows = []
    for (m, ks, side), g in allm.groupby(["match", "ks", "defending"]):
        fe = features(m, ks, side, list(g.slot))
        for i, r in g.iterrows():
            fe_rows.append({"_i": i, **fe.get(r.slot, {})})
    fdf = pd.DataFrame(fe_rows).set_index("_i")
    allm = allm.join(fdf)
    ok = allm[FEATS].notna().all(1)
    allm.loc[ok, "blame_fast"] = ridge_pred(model, allm.loc[ok, FEATS].to_numpy(float))
    pos = allm["blame_fast"].clip(lower=0)
    allm["share_fast"] = pos / pos.groupby([allm.match, allm.ks]).transform("sum").replace(0, np.nan) * 100
    pl = allm.groupby(["name", "team"]).agg(chances=("ks", "size"), mean_blame_fast=("blame_fast", "mean"), mean_share_fast=("share_fast", "mean"),
                                           mean_blame_still=("blame_still", "mean"), main_role=("role", lambda s: s.mode().iat[0])).reset_index()
    pl = pl[pl.chances >= 20].sort_values("mean_share_fast", ascending=False)
    pl.round(2).to_csv(os.path.join(OUT, "true_players.csv"), index=False)
    roles = allm.groupby("role")["share_fast"].mean().sort_values(ascending=False)

    summ = {"moments": int(tb.moment.nunique()), "candidates": len(tb), "runs": len(runs),
            "mean_grf_blame_pp": tb.blame_grf.mean(), "sure_nonzero_pct": sure.mean() * 100,
            "still_top_is_grf_top_pct": per.same_top.mean() * 100, "still_vs_grf_rank_corr": per.rho.mean(),
            "top_grf_is_sure_pct": per.top_grf_sure.mean() * 100}
    pd.DataFrame([summ]).round(3).to_csv(os.path.join(OUT, "true_summary.csv"), index=False)
    comp.round(3).to_csv(os.path.join(OUT, "true_predictors.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"\n{summ['moments']} moments, {summ['candidates']} defenders fixed one at a time in GRF ({summ['runs']} runs)")
    print(f"GRF blame: mean {summ['mean_grf_blame_pp']:.2f} pp; clearly non-zero for {summ['sure_nonzero_pct']:.0f}% of defenders; "
          f"the GRF top defender is clearly non-zero in {summ['top_grf_is_sure_pct']:.0f}% of moments")
    print(f"still-picture top = GRF top in {summ['still_top_is_grf_top_pct']:.0f}% of moments; rank corr. {summ['still_vs_grf_rank_corr']:.2f}")
    print("\nwho finds the defender GRF blames most? (per moment)\n" + comp.round(3).to_string(index=False))
    print("\nGRF top defender by role:\n" + per.top_grf_role.value_counts().to_string())
    print("\nfast blame model weights (standardised):\n" + coef.round(3).to_string())
    print("\nfast blame share by role (999 moments):\n" + roles.round(1).to_string())
    print("\nmost to blame by the fast model (20+ chances):\n" + pl.head(10).round(2).to_string(index=False))
    try:
        chart(tb, comp, coef, os.path.join(OUT, "true_chart.png"))
        print("\nwrote outputs/blame/true_blame.csv, true_summary.csv, true_predictors.csv, true_model.json, true_players.csv, true_chart.png")
    except ImportError:
        print("chart needs matplotlib (.venv-analysis)")


def chart(tb, comp, coef, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].scatter(tb.blame_still, tb.blame_grf, c=np.where(tb.role == "keeper", "#f29f1c", "#e0483b"), s=18)
    ax[0].axhline(0, c="k", lw=0.8)
    ax[0].set_xlabel("still-picture blame (pp)")
    ax[0].set_ylabel("GRF blame: threat drop when he is fixed (pp)")
    ax[0].set_title("Still picture vs GRF (orange = keeper)")
    c = comp.iloc[::-1]
    ax[1].barh(c.predictor, c.picks_true_top * 100, color=["#b5b3ab", "#8a8a8a", "#e0483b", "#2a78d6"][::-1][:len(c)])
    ax[1].set_xlabel("% of moments where it names the defender GRF blames most")
    ax[1].set_title("Who finds the real culprit?")
    ax[2].barh(coef.index, coef.values, color=np.where(coef.values > 0, "#e0483b", "#2a78d6"))
    ax[2].axvline(0, c="k", lw=0.8)
    ax[2].set_title("Fast blame model: what makes a defender to blame")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--per-match", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--dataset", default="pff")
    ap.add_argument("--only", help="play only this moment (smoke test)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run(a)


if __name__ == "__main__":
    main()
