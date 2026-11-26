"""Show 3, find all: example-based moment search over every StatsBomb action (472k, 200 matches, 360 context).

A coach marks 3 moments he likes (and optionally 1 he does not). The search finds the "common denominator" of the
examples (features where the examples agree get more weight, features where the counter-example differs too),
ranks every moment, refines once with its own top hits, then says which players do this most and best (VAEP).

Moment = one action + the 2 actions before + the 3 after (who, what, where), pressure from the 360 frame.

Steps
  eval    : held-out test. 12 patterns defined by a rule (the rule is only used to pick examples and to score);
            examples come from half of the matches, the search runs on the other half. Precision vs simple baselines.
  presets : 6 searches seeded with 3 of Granit Xhaka's own moments -> viewer/data/moment_search.json
  query   : your own search.  python scripts/moment_search.py query --pos ID,ID,ID [--neg ID] [--top 20]
            (IDs are StatsBomb event ids, column "id" in outputs/value_model/sb_values.csv.gz)
Usage : python scripts/moment_search.py all        (eval + presets, about 2 minutes)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VM = os.path.join(ROOT, "outputs", "value_model")
OUT = os.path.join(ROOT, "outputs", "moment_search")
VIEW = os.path.join(ROOT, "viewer", "data", "moment_search.json")
L, W = 105.0, 68.0
LOG = []

GROUP = {"Pass": "pass", "Cross": "pass", "Free Kick": "set", "Corner": "set", "Throw-in": "set", "Goal Kick": "set",
         "Kick Off": "set", "Carry": "carry", "Dribble": "carry", "Shot": "shot", "Free Kick Shot": "shot", "Penalty": "shot",
         "Ball Recovery": "win", "Interception": "win", "Duel": "win", "Block": "win", "Clearance": "win", "50/50": "win",
         "Miscontrol": "lose", "Dispossessed": "lose", "Foul Committed": "other", "Goal Keeper": "other", "Own Goal Against": "other"}
GROUPS = ["pass", "set", "carry", "shot", "win", "lose", "other"]
KINDS = ["Pass", "Cross", "Carry", "Dribble", "Shot", "Ball Recovery", "Interception", "Duel", "Clearance", "Block",
         "Miscontrol", "Dispossessed", "Throw-in", "Free Kick", "Corner", "Goal Kick", "Foul Committed", "Goal Keeper"]


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ------------------------------------------------------------------ data and moment features

_D = {}


def load():
    if "A" in _D:
        return _D["A"], _D["F"], _D["names"]
    t0 = time.time()
    A = pd.read_csv(os.path.join(VM, "sb_actions_all.csv.gz"), low_memory=False)
    V = pd.read_csv(os.path.join(VM, "sb_values.csv.gz"), usecols=["id", "vaep", "v_off", "v_def"])
    A = A.merge(V.drop_duplicates("id"), on="id", how="left").fillna({"vaep": 0, "v_off": 0, "v_def": 0})
    A = A.sort_values(["match", "period", "t"], kind="stable").reset_index(drop=True)
    F, names = features(A)
    _D.update(A=A, F=F, names=names)
    say(f"loaded {len(A)} actions, {A.match.nunique()} matches, {F.shape[1]} moment features ({time.time() - t0:.0f}s)")
    return A, F, names


def features(A):
    n = len(A)
    m, team = A["match"].to_numpy(), A["team"].astype(str).to_numpy()
    x, y, ex, ey = (A[c].to_numpy(float) for c in ("x", "y", "ex", "ey"))
    t = (A["period"].to_numpy() * 10000 + A["t"].to_numpy()).astype(float)
    grp = A["kind"].map(GROUP).fillna("other").to_numpy()
    cols, names = [], []

    def add(name, v):
        cols.append(np.asarray(v, float))
        names.append(name)

    for k in KINDS:                                                            # what
        add(f"kind={k}", A["kind"].to_numpy() == k)
    add("ok", A["ok"])
    body = A["body"].fillna("").astype(str)
    add("head", body.str.contains("Head"))
    add("weak_body", ~body.str.contains("Foot|Head") & (body != ""))
    gx, gy = L, W / 2                                                          # where
    add("x", x / L); add("y_wide", np.abs(y - W / 2) / (W / 2)); add("y", y / W)
    add("ex", ex / L); add("ey_wide", np.abs(ey - W / 2) / (W / 2)); add("ey", ey / W)
    add("dx", (ex - x) / 30); add("dy_abs", np.abs(ey - y) / 30); add("len", np.hypot(ex - x, ey - y) / 30)
    add("goal_d", np.hypot(gx - x, gy - y) / 50); add("goal_d_end", np.hypot(gx - ex, gy - ey) / 50)
    inbox = lambda a, b: (a >= L - 16.5) & (np.abs(b - W / 2) <= 20.16)
    add("in_box", inbox(x, y)); add("end_box", inbox(ex, ey))
    add("pressed", A["pressed"])                                               # pressure (360 frame)
    has = A["has_ff"].to_numpy(float)
    add("has_ff", has)
    for c, cap in (("near_opp", 15), ("opp5", 6), ("goal_side", 11), ("mates_ahead", 10)):
        v = np.clip(A[c].to_numpy(float), 0, cap) / cap
        add(c, np.where(np.isnan(v), 0, v) * has)
    idx = np.arange(n)
    for k, tag in ((-1, "prev1"), (-2, "prev2"), (1, "next1"), (2, "next2"), (3, "next3")):   # before and after
        j = np.clip(idx + k, 0, n - 1)
        valid = (m[j] == m) & (idx + k >= 0) & (idx + k < n)
        same = valid & (team[j] == team)
        jx = np.where(same, x[j], L - x[j]) if k < 0 else np.where(same, ex[j], L - ex[j])
        add(f"{tag}_same", same); add(f"{tag}_other", valid & ~same)
        for g in ("pass", "carry", "shot", "win", "lose", "set"):
            add(f"{tag}_{g}", valid & (grp[j] == g))
        add(f"{tag}_x", np.where(valid, jx, x) / L)
        add(f"{tag}_dt", np.where(valid, np.clip(np.abs(t[j] - t), 0, 20), 20) / 20)
    gain = np.zeros(n); shot5 = np.zeros(n); lost3 = np.zeros(n)               # what it led to
    for k in range(1, 6):
        j = np.clip(idx + k, 0, n - 1)
        valid = (m[j] == m) & (idx + k < n) & (np.abs(t[j] - t) <= 15)
        same = valid & (team[j] == team)
        if k <= 3:
            gain = np.maximum(gain, np.where(same, ex[j] - x, 0))
            lost3 = np.maximum(lost3, valid & ~same & np.isin(grp[j], ["win", "pass", "carry"]))
        shot5 = np.maximum(shot5, same & (grp[j] == "shot"))
    add("gain3", np.clip(gain, 0, 60) / 30); add("shot5", shot5); add("lost3", lost3)
    F = np.vstack(cols).T.astype(np.float32)
    mu, sd = F.mean(0), F.std(0) + 1e-6
    return (F - mu) / sd, names


# ------------------------------------------------------------------ the search

def search(F, pos, neg=(), feedback=30, bg=4000, seed=0, method="full"):
    """Score every row of F given row indices of positive (and negative) examples. Higher = more like the examples."""
    P = F[list(pos)]
    if method == "proto":                                                      # baseline: plain distance to the mean example
        return -((F - P.mean(0)) ** 2).sum(1)
    w = 1.0 / (P.var(0) + 0.15)                                                # common denominator: where the examples agree
    w *= 1 + np.abs(P.mean(0)) / 2                                             # ... and where they differ from a normal action
    if len(neg):
        N = F[list(neg)]
        w *= 1 + ((N.mean(0) - P.mean(0)) ** 2) / 2                            # ... and where they differ from the counter-example
    w /= w.mean()
    Z = F * np.sqrt(w)
    Pz = Z[list(pos)]
    d = np.stack([((Z - z) ** 2).sum(1) for z in Pz], 1)
    s = -d.min(1) - 0.5 * d.mean(1)                                           # close to one example, and to the group
    if len(neg):
        s += 0.5 * ((Z - Z[list(neg)].mean(0)) ** 2).sum(1) ** 0.5 * np.sqrt(d.min(1))
    if method == "weighted":
        return s
    rng = np.random.default_rng(seed)                                          # refine once with the top hits
    top = np.argsort(-s)[:feedback]
    b = rng.choice(len(F), min(bg, len(F)), replace=False)
    ypos = np.unique(np.concatenate([np.asarray(pos), top]))
    X = np.vstack([Z[ypos], Z[b]] + ([Z[list(neg)]] * 5 if len(neg) else []))
    yy = np.concatenate([np.ones(len(ypos)), np.zeros(len(b))] + ([np.zeros(5 * len(neg))] if len(neg) else []))
    lr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500).fit(X, yy)
    return lr.decision_function(Z)


# ------------------------------------------------------------------ patterns used only for the held-out test

def patterns(A, F, names):
    c = {k: A[k].to_numpy() for k in ("kind", "x", "y", "ex", "ey", "ok", "pressed")}
    k, x, y, ex, ey, ok, pr = c["kind"], c["x"], c["y"], c["ex"], c["ey"], c["ok"] == 1, c["pressed"] == 1
    box = (ex >= L - 16.5) & (np.abs(ey - W / 2) <= 20.16)
    inb = (x >= L - 16.5) & (np.abs(y - W / 2) <= 20.16)
    m, team = A["match"].to_numpy(), A["team"].astype(str).to_numpy()
    t = (A["period"].to_numpy() * 10000 + A["t"].to_numpy()).astype(float)
    n, idx = len(A), np.arange(len(A))
    fast = np.zeros(n, bool)
    for j0 in range(1, 4):
        j = np.clip(idx + j0, 0, n - 1)
        fast |= (m[j] == m) & (team[j] == team) & (t[j] - t <= 10) & (ex[j] - x >= 30)
    P = {
        "Switch of play": (k == "Pass") & ok & (np.abs(ey - y) >= 35) & (ex > x - 10),
        "Switch under pressure": (k == "Pass") & ok & (np.abs(ey - y) >= 30) & pr,
        "Carry into the box": (k == "Carry") & ~inb & box,
        "Cut-back": np.isin(k, ["Pass", "Cross"]) & (x >= 94) & ((y < 20) | (y > 48)) & (ex >= 83) & (ex < x) & (ey >= 22) & (ey <= 46),
        "Line-breaking pass under pressure": (k == "Pass") & ok & pr & (ex - x >= 15) & (x < 70),
        "High ball win": np.isin(k, ["Ball Recovery", "Interception"]) & (x >= 65),
        "Long shot": (k == "Shot") & (x <= L - 18),
        "Early cross": (k == "Cross") & (x >= 62) & (x <= 88),
        "Counter start": np.isin(k, ["Ball Recovery", "Interception"]) & (x < 60) & fast,
        "Dribble in final third": (k == "Dribble") & ok & (x >= 70),
        "Back pass under pressure": (k == "Pass") & pr & (ex <= x - 8),
        "Progressive carry from own half": (k == "Carry") & (x < 52.5) & (ex - x >= 15),
    }
    near = {"Switch of play": k == "Pass", "Switch under pressure": (k == "Pass") & pr, "Carry into the box": k == "Carry",
            "Cut-back": np.isin(k, ["Pass", "Cross"]) & (x >= 88), "Line-breaking pass under pressure": (k == "Pass") & pr,
            "High ball win": np.isin(k, ["Ball Recovery", "Interception"]), "Long shot": k == "Shot", "Early cross": k == "Cross",
            "Counter start": np.isin(k, ["Ball Recovery", "Interception"]), "Dribble in final third": k == "Dribble",
            "Back pass under pressure": (k == "Pass") & pr, "Progressive carry from own half": k == "Carry"}
    return P, {q: near[q] & ~P[q] for q in P}


# ------------------------------------------------------------------ step: held-out test

def step_eval():
    A, F, names = load()
    P, NEAR = patterns(A, F, names)
    ms = np.array(sorted(A.match.unique()))
    rng = np.random.default_rng(7)
    half = set(rng.choice(ms, len(ms) // 2, replace=False))
    qa = A["match"].isin(half).to_numpy()                                      # examples come from here
    tb = ~qa                                                                   # the search runs here (matches never used for examples)
    Fb, ib = F[tb], np.where(tb)[0]
    rows = []
    say(f"\nHeld-out test: examples from {len(half)} matches, search over the other {len(ms) - len(half)} ({tb.sum()} actions); 20 draws per pattern")
    say(f"  {'pattern':34s} {'base':>6s} | P@50: {'random':>6s} {'proto':>6s} {'weighted':>8s} {'full':>6s} | P@200 full | P@200 / recall at found size")
    for q, lab in P.items():
        yb = lab[tb]
        pa, na = np.where(qa & lab)[0], np.where(qa & NEAR[q])[0]
        res = {mth: [] for mth in ("proto", "weighted", "full")}
        res200, rec = [], []
        for d in range(20):
            pos = rng.choice(pa, 3, replace=False)
            neg = rng.choice(na, 1) if len(na) else []
            Fq = np.vstack([F[pos], F[neg]] if len(neg) else [F[pos]])
            Fall = np.vstack([Fb, Fq])
            pi, ni = np.arange(len(Fb), len(Fb) + 3), np.arange(len(Fb) + 3, len(Fall))
            for mth in res:
                s = search(Fall, pi, ni, method=mth, seed=d)[: len(Fb)]
                o = np.argsort(-s)
                res[mth].append(yb[o[:50]].mean())
                if mth == "full":
                    res200.append(yb[o[:200]].mean())
                    found = s > 0
                    rec.append((yb[found].mean() if found.any() else 0, yb[found].sum() / max(yb.sum(), 1)))
        r = {"pattern": q, "base": float(yb.mean()), "n_true": int(yb.sum()),
             "p50": {m_: float(np.mean(v)) for m_, v in res.items()}, "p200_full": float(np.mean(res200)),
             "found_precision": float(np.mean([a for a, _ in rec])), "found_recall": float(np.mean([b for _, b in rec]))}
        r["p50"]["random"] = r["base"]
        rows.append(r)
        say(f"  {q:34s} {r['base']:6.1%} |       {r['base']:6.1%} {r['p50']['proto']:6.1%} {r['p50']['weighted']:8.1%} {r['p50']['full']:6.1%} |"
            f"   {r['p200_full']:6.1%}   | {r['found_precision']:5.1%} / {r['found_recall']:5.1%}")
    avg = {m_: float(np.mean([r["p50"][m_] for r in rows])) for m_ in ("random", "proto", "weighted", "full")}
    say(f"  {'average':34s}        |       {avg['random']:6.1%} {avg['proto']:6.1%} {avg['weighted']:8.1%} {avg['full']:6.1%} |"
        f"   {np.mean([r['p200_full'] for r in rows]):6.1%}   | {np.mean([r['found_precision'] for r in rows]):5.1%} / {np.mean([r['found_recall'] for r in rows]):5.1%}")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"rows": rows, "avg_p50": avg, "matches_examples": len(half), "matches_search": len(ms) - len(half)},
              open(os.path.join(OUT, "eval.json"), "w"), indent=1)
    return rows, avg


# ------------------------------------------------------------------ results for one search

def minutes():
    R = pd.read_csv(os.path.join(VM, "sb_ratings.csv"))
    g = R.groupby("player_id").agg(minutes=("minutes", "sum"), team=("team", "last"), role=("role", "last"))
    return g


def moment(A, i):
    r = A.iloc[i]
    ctx = []
    for k in (-2, -1, 1, 2, 3):
        j = i + k
        if 0 <= j < len(A) and A.at[j, "match"] == r["match"]:
            s = A.iloc[j]
            same = s["team"] == r["team"]
            f = (lambda a, b: (round(a, 1), round(b, 1))) if same else (lambda a, b: (round(L - a, 1), round(W - b, 1)))
            ctx.append({"k": k, "same": int(same), "kind": s["kind"], "a": f(s["x"], s["y"]), "b": f(s["ex"], s["ey"])})
    return {"id": r["id"], "player": r["player"], "team": r["team"], "source": r["source"], "date": r["date"],
            "min": int(r["t"] // 60), "period": int(r["period"]), "kind": r["kind"], "ok": int(r["ok"]), "pressed": int(r["pressed"]),
            "a": (round(r["x"], 1), round(r["y"], 1)), "b": (round(r["ex"], 1), round(r["ey"], 1)),
            "vaep": round(float(r["vaep"]), 4), "ctx": ctx}


def players(A, s, mins, min_minutes=270, min_n=5):
    found = s > 0
    M = A.loc[found, ["player_id", "player", "ok", "vaep"]].copy()
    g = M.groupby("player_id").agg(player=("player", "first"), n=("ok", "size"), ok=("ok", "mean"), vaep=("vaep", "sum"), mean=("vaep", "mean"))
    g = g.join(mins, how="inner")
    g = g[(g.minutes >= min_minutes) & (g.n >= min_n)].copy()
    g["per90"] = g.n / g.minutes * 90
    g["vaep90"] = g.vaep / g.minutes * 90
    return g.sort_values("vaep90", ascending=False), int(found.sum())


def describe(names, F, pos, neg):
    P = F[list(pos)]
    w = 1.0 / (P.var(0) + 0.15) * (1 + np.abs(P.mean(0)) / 2)
    o = np.argsort(-w * np.abs(P.mean(0)))[:6]
    return [{"feature": names[j], "dir": "high" if P[:, j].mean() > 0 else "low"} for j in o]


def run_query(pos_ids, neg_ids=(), top=20, quiet=False):
    A, F, names = load()
    where = {v: i for i, v in enumerate(A["id"].to_numpy())}
    pos = [where[i] for i in pos_ids]
    neg = [where[i] for i in neg_ids]
    s = search(F, pos, neg)
    s[pos] = -np.inf                                                           # do not return the examples themselves
    o = np.argsort(-s)
    P, nfound = players(A, s, minutes())
    if not quiet:
        say(f"{nfound} matching moments. Top {top}:")
        for i in o[:top]:
            r = A.iloc[i]
            say(f"  {r['player'][:26]:26s} {r['team'][:18]:18s} {r['source']:16s} {int(r['t'] // 60):3d}' {r['kind']:13s} "
                f"({r['x']:.0f},{r['y']:.0f})->({r['ex']:.0f},{r['ey']:.0f}) VAEP {r['vaep']:+.3f}  id {r['id']}")
        say("Players (270+ min, 5+ moments), by VAEP per 90 from this kind of moment:")
        for pid, r in P.head(15).iterrows():
            say(f"  {r['player'][:26]:26s} {r['team'][:18]:18s} {r['n']:4d} moments  {r['per90']:.2f}/90  ok {r['ok']:.0%}  VAEP/90 {r['vaep90']:+.3f}")
    return A, F, names, s, o, P, nfound


# ------------------------------------------------------------------ step: Xhaka-seeded presets for the viewer

def step_presets():
    A, F, names = load()
    P, NEAR = patterns(A, F, names)
    ev = json.load(open(os.path.join(OUT, "eval.json"))) if os.path.exists(os.path.join(OUT, "eval.json")) else {"rows": []}
    evr = {r["pattern"]: r for r in ev["rows"]}
    xh = (A["player"] == "Granit Xhaka").to_numpy()
    mins = minutes()
    pick = ["Switch under pressure", "Line-breaking pass under pressure", "Progressive carry from own half", "Counter start",
            "Switch of play", "Long shot", "Back pass under pressure", "High ball win"]
    rng = np.random.default_rng(11)
    out = []
    say("\nPresets seeded with 3 of Granit Xhaka's own moments (and 1 he did not like):")
    for q in pick:
        xs, xn = np.where(xh & P[q])[0], np.where(xh & NEAR[q])[0]
        if len(xs) < 3:
            say(f"  {q}: Xhaka has only {len(xs)}; skipped")
            continue
        xs = xs[np.argsort(-A["vaep"].to_numpy()[xs])][:12]                    # three of his better ones, as a coach would pick
        pos = list(rng.choice(xs, 3, replace=False))
        neg = list(rng.choice(xn, 1)) if len(xn) else []
        s = search(F, pos, neg)
        s[pos] = -np.inf
        o = np.argsort(-s)
        pl, nfound = players(A, s, mins)
        hits = o[:24]
        fit = float(P[q][o[:200]].mean())
        xr = pl.index.get_loc(A["player_id"].iloc[pos[0]]) + 1 if A["player_id"].iloc[pos[0]] in pl.index else None
        out.append({"name": q, "examples": [moment(A, i) for i in pos], "counter": [moment(A, i) for i in neg],
                    "common": describe(names, F, pos, neg), "found": nfound, "fit200": round(fit, 3),
                    "test": evr.get(q), "moments": [moment(A, i) for i in hits],
                    "players": [{"player": r["player"], "team": r["team"], "role": r["role"], "n": int(r["n"]), "per90": round(r["per90"], 2),
                                 "ok": round(r["ok"], 2), "vaep90": round(r["vaep90"], 4), "mean": round(r["mean"], 4)} for _, r in pl.head(15).iterrows()],
                    "xhaka_rank": xr, "ranked": int(len(pl))})
        say(f"  {q:34s} found {nfound:6d}  fits the idea (top 200) {fit:5.0%}  top player {pl.iloc[0]['player']} "
            f"({pl.iloc[0]['vaep90']:+.3f}/90)  Xhaka rank {xr} of {len(pl)}")
    os.makedirs(os.path.dirname(VIEW), exist_ok=True)
    json.dump({"presets": out, "eval": ev}, open(VIEW, "w", encoding="utf-8"), separators=(",", ":"))
    say(f"wrote {VIEW} ({os.path.getsize(VIEW) // 1024} KB)")


def step_query(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", default="")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(argv)
    run_query([s for s in a.pos.split(",") if s], [s for s in a.neg.split(",") if s], a.top)


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if step == "query":
        return step_query(sys.argv[2:])
    if step in ("eval", "all"):
        step_eval()
    if step in ("presets", "all"):
        step_presets()
    say(f"[{step} done in {time.time() - t0:.0f}s]")
    open(os.path.join(OUT, f"run_log_{step}.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
