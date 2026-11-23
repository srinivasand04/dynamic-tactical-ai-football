"""Tactics Tournament: attack plans against defence plans, played out in GRF on real moments.

Each side picks one of 6 plans; GRF (team models, 12 s) plays every pair on the 28 real moments of the pairs batch, same
seeds for all pairs, so every pair is compared with 'none + none' run with the same seed.
    attack  : none, wide_attack, overload_ball_side, extra_runner, quick_play, long_ball
    defence : none, double_team, team_press, low_block, compact, keeper_off_line
Shape plans (who stands where) HOLD for the whole run: each moved player keeps his offset on top of the team model's
target (coach_board 'hold'), so a plan does not fade. quick_play / long_ball change the attacking team's decisions
for the whole run.

Scores per run (coach_board): threat_auc (danger over time: mean threat of the attack from 1 s, 0 off the ball),
threat_play (max threat from 1 s), box_ctrl (share of the defending box the defence controls), t_regain (s until the
defence holds the ball for 1 s), shots, xg.

Report (--report): payoff table (mean danger per pair, -none+none, with control-variate intervals), best counter to
each plan, robust plans (best worst case for each side), the mixed equilibrium (fictitious play), and how much the
control variate tightens the intervals.

Output: outputs/tournament/runs.csv, payoff.csv, best_responses.csv, robust.csv, chart.png (new files only)
Usage : .venv\\Scripts\\python.exe scripts\\tactics_tournament.py --seeds 6 --procs 4       (about 100 min, resumable)
        .venv-analysis\\Scripts\\python.exe scripts\\tactics_tournament.py --report
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("TOURNAMENT_OUT") or os.path.join(ROOT, "outputs", "tournament")   # v2 (hold fix + GRF_PACE=1): outputs/tournament_v2
ATTACK = ["none", "wide_attack", "overload_ball_side", "extra_runner", "quick_play", "long_ball"]
DEFENCE = ["none", "double_team", "team_press", "low_block", "compact", "keeper_off_line"]
BRAIN = ("quick_play", "long_ball")                                   # decision plans (the rest are shape plans)
NAME = {"none": "no change", "wide_attack": "wide attack", "overload_ball_side": "overload ball side", "extra_runner": "extra runner",
        "quick_play": "quick play", "long_ball": "long balls", "double_team": "double team", "team_press": "team press",
        "low_block": "low block", "compact": "compact", "keeper_off_line": "keeper off line"}
_T = {"att_side": None, "brain": ()}


class _AllPairs(set):
    def __contains__(self, s):
        return s == "board" or "+" in str(s)


def init(dataset):
    import coach_board as CB  # noqa: F401  (threat_play, threat_auc, box_ctrl, t_regain on every run; 'hold')
    import team_agent
    import whatif as WI
    CB.HOLD_SCENS = _AllPairs()
    if not getattr(team_agent.Brain, "_tour_patched", False):
        init0 = team_agent.Brain.__init__

        def init_t(self, *a, **k):
            init0(self, *a, **k)
            if self.side == _T["att_side"]:
                if "quick_play" in _T["brain"]:
                    self.hold_steps = max(3, self.hold_steps // 2)
                if "long_ball" in _T["brain"]:
                    self.long_share = 10.0
        team_agent.Brain.__init__ = init_t
        team_agent.Brain._tour_patched = True
    WI._init(dataset)


def job(j):
    import coach_board as CB
    import whatif as WI
    mid, seg, start_s, scen, seed, seconds = j
    a, d = scen.split("+")
    base = WI.moment(seg, start_s, "baseline", 3.0)
    pos = WI.moment(seg, start_s, scen, 3.0)[0]
    delta = pos[:, -1] - base[0][:, -1]                                # the plan's moves (GRF frame), held all run
    CB._B["delta"] = {str(k): delta[k].tolist() for k in range(22) if np.hypot(*delta[k]) > 0.3}
    CB._B["hold"] = bool(CB._B["delta"])
    _T["att_side"] = "left" if base[3] == "home" else "right"
    _T["brain"] = tuple(p for p in (a, d) if p in BRAIN)
    r = WI._job((seg, start_s, scen, 3.0, seed, seconds, None))
    r.update({"moment": mid, "attack": a, "defence": d})
    return r


def run(a):
    os.makedirs(OUT, exist_ok=True)
    mo = pd.read_csv(os.path.join(ROOT, "outputs", "whatif", "pairs_moments.csv"))
    if a.only:
        mo = mo[mo.moment == a.only]
    scen = [f"{x}+{y}" for x in ATTACK for y in DEFENCE]
    runs_p = os.path.join(OUT, "runs.csv")
    done_df = pd.read_csv(runs_p) if os.path.exists(runs_p) else pd.DataFrame(columns=["moment", "scenario", "seed"])
    done = set(zip(done_df["moment"], done_df["scenario"], done_df["seed"].astype(int)))
    jobs = [(r.moment, os.path.join(ROOT, r.segment), float(r.start_s), s, seed, a.seconds)
            for seed in range(a.seeds) for r in mo.itertuples() for s in scen if (r.moment, s, seed) not in done]
    print(f"{len(mo)} moments x {len(scen)} pairs x {a.seeds} seeds: {len(jobs)} runs to play ({len(done)} done), {a.procs} at a time", flush=True)
    t0, rows = time.time(), []
    with Pool(a.procs, initializer=init, initargs=(a.dataset,)) as pool:
        for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
            rows.append(r)
            if (k + 1) % 72 == 0 or k + 1 == len(jobs):
                pd.concat([done_df, pd.DataFrame(rows)], ignore_index=True).to_csv(runs_p, index=False)
                el = (time.time() - t0) / 60
                print(f"  {k + 1}/{len(jobs)} runs ({el:.1f} min, about {el / (k + 1) * (len(jobs) - k - 1):.0f} min left)", flush=True)


# ---------------------------------------------------------------- report

def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def fictitious_play(M, iters=20000):
    """Mixed equilibrium of the zero-sum game M[attack, defence] (attack maximises danger, defence minimises)."""
    na, nd = M.shape
    ca, cd = np.zeros(na), np.zeros(nd)
    ca[0], cd[0] = 1, 1
    for _ in range(iters):
        ca[np.argmax(M @ (cd / cd.sum()))] += 1
        cd[np.argmin((ca / ca.sum()) @ M)] += 1
    pa, pd_ = ca / ca.sum(), cd / cd.sum()
    return pa, pd_, float(pa @ M @ pd_)


def report(metric="threat_auc"):
    runs = pd.read_csv(os.path.join(OUT, "runs.csv"))
    runs = runs[runs["error"].isna()] if "error" in runs else runs
    sc = 100.0 if metric in ("threat_auc", "threat_play", "box_ctrl") else 1.0
    base = runs[runs.scenario == "none+none"].set_index(["moment", "seed"])[metric] * sc
    rows, widths = [], []
    for (a, d), g in runs.groupby(["attack", "defence"]):
        y = g.set_index(["moment", "seed"])[metric] * sc
        common = y.index.intersection(base.index)
        y, b = y.loc[common], base.loc[common]
        per_m = pd.DataFrame({"y": y, "b": b}).groupby(level=0).mean()     # per moment (mean over seeds)
        raw = per_m.y                                                      # level, no pairing
        diff = per_m.y - per_m.b                                           # paired with 'none + none'
        theta = np.cov(per_m.y, per_m.b)[0, 1] / max(np.var(per_m.b, ddof=1), 1e-9)
        cv = per_m.y - theta * (per_m.b - per_m.b.mean())                  # control variate: best use of the twin
        rows.append({"attack": a, "defence": d, "level": raw.mean(), "level_ci95": ci(raw), "vs_none": diff.mean(), "vs_none_ci95": ci(diff),
                     "cv_level": cv.mean(), "cv_ci95": ci(cv), "moments": len(per_m), "runs": len(y)})
        if (a, d) != ("none", "none"):
            widths.append((ci(raw), ci(diff), ci(cv)))
    pay = pd.DataFrame(rows)
    pay.round(3).to_csv(os.path.join(OUT, f"payoff_{metric}.csv"), index=False)
    M = pay.pivot(index="attack", columns="defence", values="cv_level").reindex(index=ATTACK, columns=DEFENCE)
    E = pay.pivot(index="attack", columns="defence", values="cv_ci95").reindex(index=ATTACK, columns=DEFENCE)
    V = pay.pivot(index="attack", columns="defence", values="vs_none").reindex(index=ATTACK, columns=DEFENCE)
    VE = pay.pivot(index="attack", columns="defence", values="vs_none_ci95").reindex(index=ATTACK, columns=DEFENCE)
    lower_is_def = metric not in ("box_ctrl", "t_regain")                # danger: defence wants it low
    sgn = 1 if lower_is_def else -1
    br = []
    for a in ATTACK:
        d = (sgn * M.loc[a]).idxmin()
        br.append({"if attack plays": NAME[a], "best defence": NAME[d], "danger": M.loc[a, d], "vs no change": M.loc[a, d] - M.loc[a, "none"]})
    for d in DEFENCE:
        a = (sgn * M[d]).idxmax()
        br.append({"if defence plays": NAME[d], "best attack": NAME[a], "danger": M.loc[a, d], "vs no change": M.loc[a, d] - M.loc["none", d]})
    br = pd.DataFrame(br)
    br.round(3).to_csv(os.path.join(OUT, f"best_responses_{metric}.csv"), index=False)
    worst_def = (sgn * M).max(axis=0) * sgn                             # each defence plan's worst case over attacks
    worst_att = (sgn * M).min(axis=1) * sgn
    rob_d, rob_a = (sgn * worst_def).idxmin(), (sgn * worst_att).idxmax()
    pa, pdf, val = fictitious_play(sgn * M.to_numpy())
    rob = pd.DataFrame([{"side": "defence", "robust plan": NAME[rob_d], "worst-case danger": worst_def[rob_d],
                         "mix": ", ".join(f"{NAME[d]} {p:.0%}" for d, p in zip(DEFENCE, pdf) if p >= 0.05)},
                        {"side": "attack", "robust plan": NAME[rob_a], "worst-case danger": worst_att[rob_a],
                         "mix": ", ".join(f"{NAME[a]} {p:.0%}" for a, p in zip(ATTACK, pa) if p >= 0.05)}])
    rob.round(3).to_csv(os.path.join(OUT, f"robust_{metric}.csv"), index=False)
    w = np.array(widths)
    pd.set_option("display.width", 220)
    print(f"\n{runs.moment.nunique()} moments, {len(runs)} runs; metric {metric} (x{sc:g}); payoff = control-variate level per pair\n")
    print(M.rename(index=NAME, columns=NAME).round(2).to_string())
    print("\nchange vs 'no change + no change' (paired), with 95% interval:")
    print((V.round(2).astype(str) + " ±" + VE.round(2).astype(str)).rename(index=NAME, columns=NAME).to_string())
    print("\nbest counters:\n" + br.round(2).to_string(index=False))
    print(f"\nrobust plans (equilibrium value {sgn * val:.2f}):\n" + rob.round(2).to_string(index=False))
    print(f"\ninterval width, median over pairs: unpaired {np.nanmedian(w[:, 0]):.2f}, paired {np.nanmedian(w[:, 1]):.2f}, "
          f"control variate {np.nanmedian(w[:, 2]):.2f}  -> {np.nanmedian(w[:, 0]) / np.nanmedian(w[:, 2]):.1f}x tighter than unpaired")
    return M, E, V, VE


def chart(M, V, VE, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
    for a_, X, title, cmap in ((ax[0], M, f"Danger per pair ({metric}, x100)", "Reds"),
                               (ax[1], V, "Change vs no change + no change (blue = safer)", "RdBu_r")):
        vmax = np.nanmax(np.abs(X.to_numpy())) if cmap == "RdBu_r" else None
        im = a_.imshow(X.to_numpy(float), cmap=cmap, vmin=-vmax if vmax else None, vmax=vmax, aspect="auto")
        a_.set_xticks(range(len(DEFENCE)), [NAME[d] for d in DEFENCE], rotation=25, ha="right")
        a_.set_yticks(range(len(ATTACK)), [NAME[a] for a in ATTACK])
        a_.set_xlabel("defence plan")
        a_.set_ylabel("attack plan")
        for i in range(len(ATTACK)):
            for j in range(len(DEFENCE)):
                v = X.iat[i, j]
                sure = "" if X is M else ("*" if abs(v) > VE.iat[i, j] else "")
                a_.text(j, i, f"{v:.2f}{sure}", ha="center", va="center", fontsize=10)
        a_.set_title(title + ("  (* = sure at 95%)" if X is V else ""))
        fig.colorbar(im, ax=a_, fraction=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"chart_{metric}.png"), dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--dataset", default="pff")
    ap.add_argument("--only", help="one moment (smoke test)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--metric", default="threat_auc")
    a = ap.parse_args()
    if a.report:
        M, E, V, VE = report(a.metric)
        chart(M, V, VE, a.metric)
        print(f"wrote {os.path.relpath(OUT, ROOT)}/payoff_{a.metric}.csv, best_responses_{a.metric}.csv, robust_{a.metric}.csv, chart_{a.metric}.png")
    else:
        run(a)


if __name__ == "__main__":
    main()
