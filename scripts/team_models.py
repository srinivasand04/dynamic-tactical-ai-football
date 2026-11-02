"""Team models: does a model of ONE team predict that team's movement better than a model of everybody?

For a dataset with several matches per team (settings.json dataset + its registry csv with home/away names):

  1. Pick the teams with at least --min-matches matches. A FOLD holds out one match per team (fold 0: each team's latest,
     fold 1: the one before, ...) as that team's test match; --folds 3 gives every team three test matches.
  2. POOLED model: the graph network of train_gnn.py (same input, same Delaunay graphs), trained on every other match.
     The learning rate is halved whenever the validation error gets worse (the first test showed an unstable pooled model).
  3. TEAM model of T: the pooled model, trained a little further on the other matches of T only, with the loss on T's own
     eleven players only (the opponent in those matches is not what we want to learn).
  4. CONTROL of T: the pooled model trained a little further on the same NUMBER of windows from matches of OTHER teams,
     loss on the same side. If the team model beats the pooled model only as much as the control does, the gain is just 'more
     training', not 'this team's style'.
  5. Score every model on every team's test match, on that team's eleven players, in metres of error for the movement to the next
     Key State (same measure as train_gnn.py), next to 'nobody moves' and 'keep velocity'.

The summary reports, per team and fold, the STYLE EFFECT: (team model - control) on the team's own match minus the same on the
other teams' matches. Negative = the team model's edge is bigger on its own team, i.e. the model learned something specific to it.

Input : <name>_graphs_delaunay.npz of the dataset's matches (run_pipeline.py --dataset NAME)
Output: outputs/team_models_<dataset>.csv (one row per fold x model x test team, saved after every fold),
        outputs/team_models_<dataset>_effects.csv (per team and fold) and a printed summary
Needs : torch, so run it with .venv-analysis

Usage :  .venv-analysis\\Scripts\\python.exe scripts\\team_models.py --dataset pff --folds 3
         .venv-analysis\\Scripts\\python.exe scripts\\team_models.py --dataset pff --teams Argentina France --epochs 3     (a smaller test)
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import settings
import train_gnn as G

MAX_GAP_S = 10.0          # a target further than this from its Key State crosses a hole in the data: not scored, not learned


def prepare(m):
    G.PATH[m] = settings.base(m)
    d = G.load(m)
    d["disp_ok"] = d["disp_ok"] & (d["gap_next"][:, None] <= MAX_GAP_S)
    return d


def side_mask(side):
    mk = np.zeros(G.N, np.float32)
    mk[:11] = side == "home"
    mk[11:22] = side == "away"
    return torch.from_numpy(mk)


def masked(b, mask):
    if mask is not None:
        b["ok"] = b["ok"] * mask.to(b["ok"].dtype)
    return b


def score(model, d, mask, bs=256):
    """Player error (m) of the model, of 'nobody moves' and of 'keep velocity', over the nodes in mask."""
    model.eval()
    tot, base0, basev = np.zeros(4), np.zeros(4), np.zeros(4)
    with torch.no_grad():
        for i in range(0, len(d["idx"]), bs):
            b = masked(G.batch(d, d["idx"][i:i + bs]), mask)
            pred, _ = model(b)
            tot += G.errors(pred, b)
            base0 += G.errors(torch.zeros_like(pred), b)
            basev += G.errors(b["vel"] * b["gap_next"][:, None, None] / G.D_MOVE, b)
    f = lambda a: a[0] / max(a[1], 1)
    return f(tot), f(base0), f(basev), int(tot[1])


def train(model, data, windows, masks, epochs, lr, patience, tag, seed, save=None):
    """windows: list of (match, ks array); masks: match -> node mask or None. Early stopping on the last 10% of every match."""
    rng = np.random.default_rng(seed)
    tr, va = {}, {}
    for m, idx in windows:
        cut = int(len(idx) * 0.9)
        tr[m], va[m] = idx[:cut], idx[cut:]
    pool = [(m, k) for m in tr for k in tr[m]]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=0)     # halve the step when validation gets worse
    bs = settings.get("gnn", "batch", default=128)
    best, best_state, stale = 1e9, copy.deepcopy(model.state_dict()), 0
    for ep in range(epochs):
        model.train()
        t0, perm, tot, cnt = time.time(), rng.permutation(len(pool)), 0.0, 0
        for i in range(0, len(perm), bs):
            chunk = [pool[j] for j in perm[i:i + bs]]
            ms = sorted({m for m, _ in chunk})
            parts = [masked(G.batch(data[m], np.array([k for mm, k in chunk if mm == m])), masks.get(m)) for m in ms]
            b = {key: torch.cat([p[key] for p in parts]) for key in parts[0]}
            b = G.mirror(b)
            pred, _ = model(b)
            ok = b["ok"].float()[..., None]
            loss = (nn.functional.smooth_l1_loss(pred, b["target"], reduction="none") * ok).sum() / (ok.sum() * 2 + 1e-6)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(chunk)
            cnt += len(chunk)
        v = np.mean([score(model, {**data[m], "idx": va[m]}, masks.get(m))[0] for m in va if len(va[m])])
        stale = 0 if v < best - 0.005 else stale + 1
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
            if save:                                             # keep the best weights on disk: a crash then loses nothing
                torch.save(best_state, save)
        print(f"  [{tag}] epoch {ep + 1:2d}  loss {tot / cnt:.4f}  validation {v:.3f} m  ({time.time() - t0:.0f}s)", flush=True)
        sched.step(v)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model


def run_fold(a, fold, games, teams, data):
    """Hold out one match per team, train the pooled model, then every team model and control, score the matrix."""
    test = {}                                                           # team -> (match, side); one match is never the test of two teams
    for t in teams:
        for m, s in list(reversed(games[t]))[fold:]:
            if m not in {x[0] for x in test.values()}:
                test[t] = (m, s)
                break
    test_ids = {m for m, _ in test.values()}
    train_ids = [m for m in data if m not in test_ids]
    print(f"\n===== FOLD {fold}: test matches {test} | training matches {len(train_ids)}", flush=True)

    pooled = G.Net(data[train_ids[0]]["feat"].shape[2], a.dim, "ours")
    windows = [(m, data[m]["idx"]) for m in train_ids]
    print(f"POOLED model on {sum(len(w[1]) for w in windows)} windows", flush=True)
    pooled = train(pooled, data, windows, {}, a.epochs, a.lr, 2, f"f{fold} pooled", a.seed + fold)

    models = {"pooled": pooled}
    rng = np.random.default_rng(a.seed + fold)
    for t in teams:
        own = [(m, s) for m, s in games[t] if m in train_ids]
        masks = {m: side_mask(s) for m, s in own}
        w_own = [(m, data[m]["idx"]) for m, _ in own]
        n_win = sum(len(w[1]) for w in w_own)
        others = [m for m in train_ids if m not in {x for x, _ in own}]      # control: the same number of windows from other teams
        rng.shuffle(others)
        w_ctl, c_masks, got = [], {}, 0
        for m in others:
            if got >= n_win:
                break
            take = min(len(data[m]["idx"]), n_win - got)
            w_ctl.append((m, data[m]["idx"][:take]))
            c_masks[m] = side_mask(["home", "away"][rng.integers(2)])
            got += take
        print(f"TEAM model {t}: {len(own)} matches, {n_win} windows | control: {len(w_ctl)} matches of other teams", flush=True)
        models[t] = train(copy.deepcopy(pooled), data, w_own, masks, a.tune_epochs, a.lr / 4, 2, f"f{fold} {t}", a.seed + fold)
        models["control:" + t] = train(copy.deepcopy(pooled), data, w_ctl, c_masks, a.tune_epochs, a.lr / 4, 2, f"f{fold} control {t}", a.seed + fold)

    rows = []
    for u, (m, s) in test.items():
        mk = side_mask(s)
        for name, mod in models.items():
            e, e0, ev, n = score(mod, data[m], mk)
            rows.append({"fold": fold, "model": name, "test_team": u, "test_match": m, "error_m": round(e, 4), "still_m": round(e0, 4),
                         "keepvel_m": round(ev, 4), "nodes": n})
    return pd.DataFrame(rows)


def summarise(out, teams):
    """Per (fold, team): own-match edge of the team model and of its control over the pooled model, the same on the other teams' matches."""
    rec = []
    for f, g in out.groupby("fold"):
        P = g.pivot(index="model", columns="test_team", values="error_m")
        for t in teams:
            if t not in P.columns:
                continue
            oth = [u for u in P.columns if u != t]
            rec.append({"fold": f, "team": t,
                        "own_team": P.loc[t, t] - P.loc["pooled", t], "own_ctl": P.loc["control:" + t, t] - P.loc["pooled", t],
                        "other_team": np.mean([P.loc[t, u] - P.loc["pooled", u] for u in oth]),
                        "other_ctl": np.mean([P.loc["control:" + t, u] - P.loc["pooled", u] for u in oth])})
    r = pd.DataFrame(rec)
    r["team_vs_ctl_own"] = r["own_team"] - r["own_ctl"]                  # negative = better than a model given the same extra training
    r["team_vs_ctl_other"] = r["other_team"] - r["other_ctl"]
    r["style_effect"] = r["team_vs_ctl_own"] - r["team_vs_ctl_other"]     # negative = the team model's edge is bigger on its OWN team
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pff")
    ap.add_argument("--teams", nargs="+", default=None, help="default: every team with at least --min-matches matches")
    ap.add_argument("--min-matches", type=int, default=5)
    ap.add_argument("--folds", type=int, default=1, help="test matches per team (each fold trains its own pooled model)")
    ap.add_argument("--epochs", type=int, default=6, help="maximum for the pooled model")
    ap.add_argument("--tune-epochs", type=int, default=4, help="maximum for the team model and its control")
    ap.add_argument("--lr", type=float, default=1e-3, help="pooled model; the fine-tunes use a quarter of it")
    ap.add_argument("--dim", type=int, default=settings.get("gnn", "dim", default=48))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t_start = time.time()

    reg = settings.registry(a.dataset)
    have = set(settings.dataset_matches(a.dataset)) & {m for m in reg["match_id"] if os.path.exists(settings.graphs(m))}
    reg = reg[reg["match_id"].isin(have)].sort_values("date")
    games = {}                                                          # team -> [(match, side)] oldest first
    for r in reg.itertuples():
        games.setdefault(r.home, []).append((r.match_id, "home"))
        games.setdefault(r.away, []).append((r.match_id, "away"))
    teams = a.teams or [t for t, g in games.items() if len(g) >= a.min_matches]
    teams = sorted(teams, key=lambda t: -len(games[t]))
    print(f"{len(have)} matches with graphs; teams: " + ", ".join(f"{t} ({len(games[t])})" for t in teams), flush=True)
    data = {m: prepare(m) for m in sorted(have)}

    path = f"outputs/team_models_{a.dataset}.csv"
    parts = []
    for fold in range(a.folds):
        parts.append(run_fold(a, fold, games, teams, data))
        out = pd.concat(parts)
        out.to_csv(path, index=False)                                   # saved after every fold, so a stop loses at most one fold
        print(f"fold {fold} done after {(time.time() - t_start) / 60:.0f} min; saved {path}", flush=True)

    r = summarise(out, teams)
    r.round(4).to_csv(f"outputs/team_models_{a.dataset}_effects.csv", index=False)
    print("\nPER TEAM (mean over folds, metres; negative = better than the pooled model)")
    print(r.groupby("team")[["own_team", "own_ctl", "other_team", "other_ctl", "style_effect"]].mean().round(3).to_string())
    se = r["style_effect"]
    print(f"\nSTYLE EFFECT = (team model - control) on its own team minus the same on other teams: mean {se.mean():+.4f} m "
          f"(standard error {se.std() / np.sqrt(max(len(se), 1)):.4f}), negative in {(se < 0).sum()} of {len(se)} team-folds")
    print(f"finished in {(time.time() - t_start) / 60:.0f} min; wrote {path} and the _effects file")


if __name__ == "__main__":
    main()
