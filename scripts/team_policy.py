"""Team movement model (Part A2/A4): ONE graph network for every team, with the team's identity as an input on its players.

Every player node gets, besides the usual inputs of train_gnn.py, a one-hot 'which team is this' (the chosen teams, plus 'other'
for everybody else). One training run on all matches of the dataset therefore gives a model of every chosen team at once, and
teams with few matches borrow what is common from the rest. Asking the model about Argentina's players with France's identity is
how the teams are compared.

Held out: the latest match of every chosen team (never trained on). The check (A4) scores each held-out match, on that team's
eleven players, with the correct identity, with every other chosen identity and with 'other'. If the identity carries a team's
style, the correct one gives the lowest error in its column.

Input : <match>_graphs_delaunay.npz of the dataset (run_pipeline.py --dataset NAME) and its registry (team names)
Output: outputs/team_policy/<dataset>_model.pt and _config.json (for team_agent.py), outputs/team_policy/<dataset>_identity_check.csv
Needs : torch (.venv-analysis or .venv)

Usage :  .venv-analysis\\Scripts\\python.exe scripts\\team_policy.py --dataset pff --teams Argentina France Croatia
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import settings
import train_gnn as G
from team_models import prepare, side_mask, score, train


def add_identity(d, home_id, away_id, n_ids):
    """Append the one-hot team identity to every player node (ball node: zeros)."""
    K = d["feat"].shape[0]
    idf = np.zeros((K, G.N, n_ids), np.float32)
    idf[:, :11, home_id] = 1
    idf[:, 11:22, away_id] = 1
    idf *= d["present"][..., None]
    d = dict(d)
    d["feat"] = np.concatenate([d["feat"], idf], axis=2)
    return d


def with_identity(d, side, new_id, n_ids):
    """The same match, with one side's identity replaced (for the check)."""
    d = dict(d)
    f = d["feat"].copy()
    sl = slice(0, 11) if side == "home" else slice(11, 22)
    f[:, sl, -n_ids:] = 0
    f[:, sl, -n_ids + new_id] = d["present"][:, sl]
    d["feat"] = f
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pff")
    ap.add_argument("--teams", nargs="+", default=["Argentina", "France", "Croatia"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=settings.get("gnn", "dim", default=48))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t0 = time.time()
    names = a.teams + ["other"]
    n_ids = len(names)
    tid = lambda team: a.teams.index(team) if team in a.teams else n_ids - 1

    reg = settings.registry(a.dataset)
    have = set(settings.dataset_matches(a.dataset)) & {m for m in reg["match_id"] if os.path.exists(settings.graphs(m))}
    reg = reg[reg["match_id"].isin(have)].sort_values("date")
    test = {}
    for t in a.teams:                                        # each team's latest match, never the test match of two teams
        rows = reg[(reg["home"] == t) | (reg["away"] == t)]
        for r in rows.iloc[::-1].itertuples():
            if r.match_id not in {m for m, _ in test.values()}:
                test[t] = (r.match_id, "home" if r.home == t else "away")
                break
    print("test matches:", test, flush=True)

    data, sides = {}, {}
    for r in reg.itertuples():
        data[r.match_id] = add_identity(prepare(r.match_id), tid(r.home), tid(r.away), n_ids)
        sides[r.match_id] = (r.home, r.away)
    train_ids = [m for m in data if m not in {m for m, _ in test.values()}]
    f_in = data[train_ids[0]]["feat"].shape[2]
    model = G.Net(f_in, a.dim, "ours")
    windows = [(m, data[m]["idx"]) for m in train_ids]
    print(f"training on {len(train_ids)} matches, {sum(len(w[1]) for w in windows)} windows, {f_in} inputs per node", flush=True)
    os.makedirs("outputs/team_policy", exist_ok=True)
    model = train(model, data, windows, {}, a.epochs, a.lr, 2, "team policy", a.seed, save=f"outputs/team_policy/{a.dataset}_model.pt")

    torch.save(model.state_dict(), f"outputs/team_policy/{a.dataset}_model.pt")
    cfg = {"dataset": a.dataset, "teams": names, "f_in": f_in, "dim": a.dim, "window": G.L, "d_move": G.D_MOVE,
           "graph_mode": settings.get("graph", "mode"), "ball_k": settings.get("graph", "ball_k", default=3),
           "test_matches": {t: list(v) for t, v in test.items()}}
    json.dump(cfg, open(f"outputs/team_policy/{a.dataset}_config.json", "w"), indent=1)

    # A4: does the identity matter? Each held-out match, the team's own players, scored under every identity
    rows = []
    for t, (m, side) in test.items():
        mk = side_mask(side)
        for j, nm in enumerate(names):
            e, e0, ev, n = score(model, with_identity(data[m], side, j, n_ids), mk)
            rows.append({"test_team": t, "identity_given": nm, "error_m": round(e, 4), "still_m": round(e0, 4), "keepvel_m": round(ev, 4)})
    r = pd.DataFrame(rows)
    r.to_csv(f"outputs/team_policy/{a.dataset}_identity_check.csv", index=False)
    P = r.pivot(index="identity_given", columns="test_team", values="error_m").reindex(names)
    print("\nERROR (m) on each team's held-out match (column) when the model is told the team is ... (row); lower is better\n")
    print(P.round(4).to_string())
    wins = sum(P[t].idxmin() == t for t in a.teams)
    gain = np.mean([P.loc["other", t] - P.loc[t, t] for t in a.teams])
    print(f"\nthe correct identity gives the lowest error for {wins} of {len(a.teams)} teams; "
          f"correct vs 'other': {gain:+.4f} m better on average; keep-velocity {r['keepvel_m'].mean():.3f} m")
    print(f"finished in {(time.time() - t0) / 60:.0f} min; wrote outputs/team_policy/{a.dataset}_model.pt")


if __name__ == "__main__":
    main()
