"""own_BayesPlayer: the on-ball player model of player_models_big, made Bayesian.

Before: a player's correction on top of his role (3 x 6 numbers) was pulled to 0 by one ridge strength picked by
cross-validation inside his own matches -- noisy for players with few matches, and no uncertainty.
Now (empirical Bayes):
    prior        b ~ N(0, tau^2) per number and role; tau^2 = spread of the corrections across all players minus their
                 own estimation noise (method of moments over hundreds of players)
    posterior    MAP under that prior + Laplace approximation (Gaussian around the MAP) -> 90% bands for every
                 predicted choice share ("he plays a forward pass 31% of the time, 90%: 28-34%")
Tests (Wyscout, 5 leagues 2017/18; same base model and season split as player_models_big):
    season       train first ~3/4 of match days, test the rest: log-loss role / old player model / Bayes
    curve        player model from his first n matches (1 .. 12), same test matches
    coverage     do the 90% bands of his test-period choice shares contain the real shares 90% of the time?
Export: every player with 300+ decisions (Wyscout and StatsBomb): his choice shares vs his role with 90% bands.

Outputs: outputs/player_bayes/ (eval.json, summary_*.txt, styles.json)
Usage  : python scripts/player_bayes.py prior | season | curve | export | all       (run from the project root)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import player_models as P  # noqa: E402
import player_models_big as BG  # noqa: E402

ROOT = BG.ROOT
OUT = os.path.join(ROOT, "outputs", "player_bayes")
K = len(P.CLASSES)
Q = 3
RNG = np.random.default_rng(5)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def save(step, obj=None):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, f"summary_{step}.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    LOG.clear()
    if obj is not None:
        p = os.path.join(OUT, "eval.json")
        E = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        E[step] = obj
        json.dump(E, open(p, "w", encoding="utf-8"), indent=1, default=float)


def hessian(Zi, Pi):
    A = Pi[:, :, None] * (np.eye(K)[None] - Pi[:, None, :])                 # diag(p) - p p^T
    return np.einsum("ia,ib,ikl->akbl", Zi, Zi, A).reshape(Q * K, Q * K)


def posterior(Zi, yi, off, pen):
    """MAP with per-number penalty pen (prior precision 2 * pen) and its Laplace covariance."""
    B = P.fit_mn(Zi, yi, K, pen, offset=off)
    Pi = P.softmax(off + Zi @ B)
    cov = np.linalg.inv(hessian(Zi, Pi) + np.diag(2 * pen.ravel()) + 1e-9 * np.eye(Q * K))
    return B, cov


_CACHE = {}


def data():
    if "d" in _CACHE:
        return _CACHE["d"]
    D = BG.wy_data()
    X, Z = BG.wy_X(D)
    tr, te = BG.wy_split(D)
    _CACHE["d"] = (D, X, Z, tr, te)
    return D, X, Z, tr, te


def season_base(D, X, Z, tr):
    if "base" not in _CACHE:
        _CACHE["base"] = P.OnballBase(D, X, Z, tr)
    return _CACHE["base"]


def eligible(D, tr, te, ntrain=10, ntest=3):
    pm_tr = D[tr].groupby(["player_id", "match"]).size()
    pm_te = D[te].groupby(["player_id", "match"]).size()
    a = (pm_tr >= 15).groupby(level=0).sum()
    b = (pm_te >= 15).groupby(level=0).sum()
    return [p for p in a[a >= ntrain].index if b.get(p, 0) >= ntest], pm_tr


def prior_from(D, X, Z, base, mask, players):
    """tau^2 per role and number: spread of weakly-penalised corrections minus their estimation noise."""
    y = D["choice"].to_numpy()
    pid = D["player_id"].to_numpy()
    roles = D["role"].to_numpy()
    est = {}
    for p in players:
        i = np.flatnonzero(mask & (pid == p))
        if len(i) < 200:
            continue
        off = base.logits(D, X, Z, i)
        B, cov = posterior(Z[i], y[i], off, np.full((Q, K), 0.05))
        # softmax ignores a shift common to all choices: measure spread and noise on choice-centred numbers only
        Tm = np.kron(np.eye(Q), np.eye(K) - 1.0 / K)
        est.setdefault(pd.Series(roles[i]).mode()[0], []).append((Tm @ B.ravel(), np.diag(Tm @ cov @ Tm.T)))
    allb = np.array([b for v in est.values() for b, _ in v])
    alln = np.array([c for v in est.values() for _, c in v])
    pooled = np.maximum(allb.var(0) - alln.mean(0), 1e-3)
    tau = {}
    for r, v in est.items():
        b, c = np.array([x for x, _ in v]), np.array([x for _, x in v])
        t = np.maximum(b.var(0) - c.mean(0), 1e-3) if len(v) >= 40 else pooled
        tau[r] = t.reshape(Q, K)
    return tau, pooled.reshape(Q, K), {r: len(v) for r, v in est.items()}


def pen_of(tau, role, pooled):
    return 1.0 / (2.0 * tau.get(role, pooled))


# ------------------------------------------------------------------ steps

def step_prior():
    D, X, Z, tr, te = data()
    t0 = time.time()
    base = season_base(D, X, Z, tr)
    elig, _ = eligible(D, tr, te)
    tau, pooled, n = prior_from(D, X, Z, base, tr, elig)
    say(f"Prior from {sum(n.values())} players' training matches ({time.time() - t0:.0f}s): players per role {n}")
    lab = ["intercept", "height", "pressed"]
    for r, t in sorted(tau.items()):
        say(f"  {r:3s} prior sd of the player correction: " + "  ".join(
            f"{lab[q]}: " + " ".join(f"{np.sqrt(t[q, k]):.2f}" for k in range(K)) for q in range(Q)))
    say("  (one number per choice: " + ", ".join(P.CLASSES) + "; the old model used one strength for all, chosen per player)")
    c, f, grid = tune(D, X, Z, base, tr, elig, tau, pooled)
    np.savez(os.path.join(OUT, "prior_season.npz"), pooled=pooled, scale=c, inflate=f, **{f"tau_{r}": t for r, t in tau.items()})
    save("prior", {"players": n, "tau": {r: np.sqrt(t).tolist() for r, t in tau.items()}, "scale": c, "inflate": f, "grid": grid})


def tune(D, X, Z, base, mask, players, tau, pooled):
    """Inside the TRAINING matches only: each player's first 75% of training match days fit, last 25% validate.
    Picks the prior scale c (tau^2 * c) by validation log-loss, then the band inflation f (covariance * f) so the
    90% bands of validation choice shares contain the truth 90% of the time."""
    y = D["choice"].to_numpy()
    pid, mid = D["player_id"].to_numpy(), D["match"].to_numpy()
    dates = D["date"].to_numpy()
    cands = [1, 3, 10, 30, 100, 1000]
    tot = {c: 0.0 for c in cands}
    nrow = 0
    keep = []
    for p in players:
        i = np.flatnonzero(mask & (pid == p))
        ds = np.sort(np.unique(dates[i]))
        cut = ds[int(len(ds) * 0.75)]
        fi, vi = i[dates[i] < cut], i[dates[i] >= cut]
        if len(fi) < 100 or len(vi) < 50:
            continue
        role = pd.Series(D["role"].to_numpy()[fi]).mode()[0]
        of, ov = base.logits(D, X, Z, fi), base.logits(D, X, Z, vi)
        fits = {}
        for c in cands:
            B, cov = posterior(Z[fi], y[fi], of, pen_of(tau, role, pooled) / c)
            tot[c] += P.nll_rows(P.softmax(ov + Z[vi] @ B), y[vi]).sum()
            fits[c] = (B, cov)
        nrow += len(vi)
        keep.append((vi, ov, fits))
    grid = {str(c): tot[c] / nrow for c in cands}
    c = min(cands, key=lambda q: tot[q])
    say(f"  tuning inside the training matches (validation = last 25% of each player's training days, {len(keep)} players):")
    say("    prior scale -> validation log-loss: " + "  ".join(f"{q}: {tot[q] / nrow:.4f}" for q in cands) + f"  -> {c}")
    cover = {}
    for f in (1, 2, 4, 8, 16):
        hits = []
        for vi, ov, fits in keep[:400]:
            B, cov = fits[c]
            S = RNG.multivariate_normal(B.ravel(), cov * f, 120).reshape(120, Q, K)
            sh = np.array([P.softmax(ov + Z[vi] @ s_) for s_ in S])
            cat = (np.cumsum(sh, 2) < RNG.random((sh.shape[0], sh.shape[1], 1))).sum(2).clip(0, K - 1)
            dr = np.eye(K)[cat].mean(1)
            lo, hi = np.percentile(dr, 5, 0), np.percentile(dr, 95, 0)
            real = np.bincount(y[vi], minlength=K) / len(vi)
            hits.append((real >= lo - 1e-9) & (real <= hi + 1e-9))
        cover[f] = float(np.mean(hits))
        if cover[f] >= 0.9:
            break
    f = max(cover) if max(cover.values()) < 0.9 else min(k for k, v in cover.items() if v >= 0.9)
    say("    band inflation -> validation coverage of 90% bands: " + "  ".join(f"x{k}: {100 * v:.0f}%" for k, v in cover.items()) + f"  -> x{f}")
    return c, f, {"scale": grid, "coverage": {str(k): v for k, v in cover.items()}}


def load_prior(name="prior_season.npz"):
    z = np.load(os.path.join(OUT, name))
    c, f = float(z["scale"]), float(z["inflate"])
    return {k[4:]: z[k] * c for k in z.files if k.startswith("tau_")}, z["pooled"] * c, f


def step_season():
    D, X, Z, tr, te = data()
    y = D["choice"].to_numpy()
    pid, mid = D["player_id"].to_numpy(), D["match"].to_numpy()
    base = season_base(D, X, Z, tr)
    tau, pooled, infl = load_prior()
    elig, _ = eligible(D, tr, te)
    old = pd.read_csv(os.path.join(BG.OUT, "wy_eval_matches.csv"))
    res, cover = [], []
    t0 = time.time()
    for j, p in enumerate(elig):
        tri, tei = np.flatnonzero(tr & (pid == p)), np.flatnonzero(te & (pid == p))
        role = pd.Series(D["role"].to_numpy()[tri]).mode()[0]
        off_tr, off_te = base.logits(D, X, Z, tri), base.logits(D, X, Z, tei)
        B, cov = posterior(Z[tri], y[tri], off_tr, pen_of(tau, role, pooled))
        nll_r = P.nll_rows(P.softmax(off_te), y[tei])
        nll_b = P.nll_rows(P.softmax(off_te + Z[tei] @ B), y[tei])
        for m in np.unique(mid[tei]):
            k = mid[tei] == m
            res.append({"player_id": p, "match": m, "n": int(k.sum()), "role": role, "nll_role": float(nll_r[k].mean()), "nll_bayes": float(nll_b[k].mean())})
        # coverage of 90% predictive bands for his test-period choice shares
        S = RNG.multivariate_normal(B.ravel(), cov * infl, 200).reshape(200, Q, K)
        sh = np.array([P.softmax(off_te + Z[tei] @ s) for s in S])          # (200, n, K)
        cs = np.cumsum(sh, 2)
        cat = (cs < RNG.random((cs.shape[0], cs.shape[1], 1))).sum(2).clip(0, K - 1)    # one multinomial draw per row and sample
        draws = np.eye(K)[cat].mean(1)
        lo, hi = np.percentile(draws, 5, 0), np.percentile(draws, 95, 0)
        real = np.bincount(y[tei], minlength=K) / len(tei)
        cover.append(((real >= lo - 1e-9) & (real <= hi + 1e-9)).tolist())
        if (j + 1) % 200 == 0:
            print(f"  season {j + 1}/{len(elig)} ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(res).merge(old[["player_id", "match", "nll_player"]], on=["player_id", "match"], how="inner")
    R.to_csv(os.path.join(OUT, "season_matches.csv"), index=False)
    w = R["n"]
    a, b, c = (np.average(R[k], weights=w) for k in ("nll_role", "nll_player", "nll_bayes"))
    pp = R.groupby("player_id").apply(lambda g: pd.Series({"old": np.average(g.nll_role - g.nll_player, weights=g.n),
                                                           "bayes": np.average(g.nll_role - g.nll_bayes, weights=g.n)}))
    lo, hi = P.boot_ci((pp["bayes"] - pp["old"]).to_numpy())
    cov_ = np.array(cover).mean(0)
    say(f"Season split, {R['player_id'].nunique()} players, {len(R)} test player-matches:")
    say(f"  log-loss role {a:.4f}   old player model {b:.4f} ({100 * (a - b) / a:+.2f}%)   Bayes {c:.4f} ({100 * (a - c) / a:+.2f}%)")
    say(f"  Bayes vs old per player: mean {(pp['bayes'] - pp['old']).mean():+.5f} (90% {lo:+.5f} .. {hi:+.5f}); Bayes better for "
        f"{100 * (pp['bayes'] > pp['old']).mean():.0f}% of players; beats role for {100 * (pp['bayes'] > 0).mean():.0f}% (old {100 * (pp['old'] > 0).mean():.0f}%)")
    say("  90% bands of test-period choice shares contain the real share: " + "  ".join(f"{c_} {100 * v:.0f}%" for c_, v in zip(P.CLASSES, cov_)))
    save("season", {"players": int(R["player_id"].nunique()), "logloss": {"role": a, "old": b, "bayes": c},
                    "gain_old_pct": 100 * (a - b) / a, "gain_bayes_pct": 100 * (a - c) / a,
                    "bayes_vs_old": [float((pp["bayes"] - pp["old"]).mean()), lo, hi], "share_bayes_better": float((pp["bayes"] > pp["old"]).mean()),
                    "share_beats_role": {"old": float((pp["old"] > 0).mean()), "bayes": float((pp["bayes"] > 0).mean())},
                    "coverage": dict(zip(P.CLASSES, cov_.tolist()))})


def step_curve():
    D, X, Z, tr, te = data()
    y = D["choice"].to_numpy()
    pid, mid = D["player_id"].to_numpy(), D["match"].to_numpy()
    elig, pm_tr = eligible(D, tr, te, 12, 5)
    players = list(RNG.choice(elig, min(250, len(elig)), replace=False))
    test = te & np.isin(pid, players)
    base = P.OnballBase(D, X, Z, ~test)
    tau, pooled, _ = load_prior()
    dates = D.drop_duplicates("match").set_index("match")["date"]
    rows = []
    t0 = time.time()
    for j, p in enumerate(players):
        tei = np.flatnonzero(test & (pid == p))
        off_te = base.logits(D, X, Z, tei)
        nr = P.nll_rows(P.softmax(off_te), y[tei]).mean()
        ms = pm_tr.loc[p]
        order = sorted(ms[ms >= 15].index, key=lambda m: dates[m])
        role = pd.Series(D["role"].to_numpy()[tei]).mode()[0]
        for n in (1, 2, 3, 5, 8, 12):
            tri = np.flatnonzero(np.isin(mid, order[:n]) & (pid == p))
            off_tr = base.logits(D, X, Z, tri)
            Bo, _ = BG.fit_player(Z[tri], y[tri], off_tr, mid[tri])
            Bb, _ = posterior(Z[tri], y[tri], off_tr, pen_of(tau, role, pooled))
            rows.append({"player_id": p, "n": n, "role": nr,
                         "old": P.nll_rows(P.softmax(off_te + Z[tei] @ Bo), y[tei]).mean(),
                         "bayes": P.nll_rows(P.softmax(off_te + Z[tei] @ Bb), y[tei]).mean()})
        if (j + 1) % 50 == 0:
            print(f"  curve {j + 1}/{len(players)} ({time.time() - t0:.0f}s)", flush=True)
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(OUT, "curve.csv"), index=False)
    out = []
    say(f"Learning curve, {len(players)} players (prior from the season-split training matches; these players' test matches unseen):")
    say("  matches   old player model gain   Bayes gain     share of players Bayes >= old")
    for n, g in C.groupby("n"):
        go, gb = 100 * (g.role - g.old) / g.role, 100 * (g.role - g.bayes) / g.role
        lo, hi = P.boot_ci(gb)
        out.append({"n": int(n), "old": float(go.mean()), "bayes": float(gb.mean()), "ci": [lo, hi], "share": float((g.bayes <= g.old).mean())})
        say(f"  {n:7d}   {go.mean():+8.2f}%            {gb.mean():+7.2f}% ({lo:+.2f} .. {hi:+.2f})   {100 * (g.bayes <= g.old).mean():4.0f}%")
    save("curve", out)


def bands(Zi, off, B, cov, n=300):
    S = RNG.multivariate_normal(B.ravel(), cov, n).reshape(n, Q, K)
    sh = np.array([P.softmax(off + Zi @ s).mean(0) for s in S])
    return P.softmax(off + Zi @ B).mean(0), np.percentile(sh, 5, 0), np.percentile(sh, 95, 0), P.softmax(off).mean(0)


def step_export():
    """Every player with 300+ decisions: his shares vs role with 90% bands (all data; prior re-estimated on all data)."""
    out = {}
    for tag in ("wy", "sb"):
        t0 = time.time()
        if tag == "wy":
            D = BG.wy_data()
            X, Z = BG.wy_X(D)
        else:
            D, _ = BG.sb_data()
            X, Z = P.onball_X(D)
        y = D["choice"].to_numpy()
        pid = D["player_id"].to_numpy()
        allm = np.ones(len(D), bool)
        base = P.OnballBase(D, X, Z, allm)
        cnt = D.groupby("player_id").size()
        players = cnt.index[cnt >= 300].tolist()
        tau, pooled, _ = prior_from(D, X, Z, base, allm, players)
        _, _, infl = load_prior()
        c = float(np.load(os.path.join(OUT, "prior_season.npz"))["scale"])
        tau, pooled = {r: t * c for r, t in tau.items()}, pooled * c
        for j, p in enumerate(players):
            i = np.flatnonzero(pid == p)
            role = pd.Series(D["role"].to_numpy()[i]).mode()[0]
            off = base.logits(D, X, Z, i)
            B, cov = posterior(Z[i], y[i], off, pen_of(tau, role, pooled))
            m, lo, hi, r = bands(Z[i], off, B, cov * infl)
            # pressure what-if: his shares with pressure on vs off on his own situations
            Dq1, Dq0 = D.iloc[i].assign(pressed=1), D.iloc[i].assign(pressed=0)
            (X1, Z1), (X0, Z0) = (BG.wy_X(Dq1), BG.wy_X(Dq0)) if tag == "wy" else (P.onball_X(Dq1), P.onball_X(Dq0))
            o1, o0 = base.logits(Dq1, X1, Z1, np.arange(len(i))), base.logits(Dq0, X0, Z0, np.arange(len(i)))
            dp = P.softmax(o1 + Z1 @ B).mean(0) - P.softmax(o0 + Z0 @ B).mean(0)
            out[f"{tag}:{p}"] = {"decisions": len(i), "role": role, "his": m.round(4).tolist(), "lo": lo.round(4).tolist(),
                                 "hi": hi.round(4).tolist(), "typical": r.round(4).tolist(), "pressed_delta": dp.round(4).tolist()}
            if (j + 1) % 300 == 0:
                print(f"  {tag} export {j + 1}/{len(players)} ({time.time() - t0:.0f}s)", flush=True)
        say(f"{tag}: {len(players)} players exported ({time.time() - t0:.0f}s)")
    json.dump({"classes": P.CLASSES, "players": out}, open(os.path.join(OUT, "styles.json"), "w", encoding="utf-8"))
    save("export")


STEPS = {"prior": step_prior, "season": step_season, "curve": step_curve, "export": step_export}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for s in (list(STEPS) if a.step == "all" else [a.step]):
        t0 = time.time()
        STEPS[s]()
        print(f"[{s} done in {time.time() - t0:.0f}s]", flush=True)
