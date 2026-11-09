"""Team models against each other in GRF (Part B2/B3): a round robin with measurable scores.

Every pair of the chosen teams plays --games matches on each side (sides swapped, so neither team profits from being left or right).
Two baselines take part: "pooled" (the same model told 'other' instead of a team: what the team identity adds) and "GRF AI" (GRF's
own bot, which only plays on the right). Every match is GRF's full match (3000 steps, about 5 minutes of play). Matches run in
parallel processes; each finished match is appended to the results file at once, so a stop loses nothing.

Scores per team: points per game (3 win, 1 draw), goal difference per game with a 95% interval, shots, xG proxy, passes, pass
completion, possession, time with the ball in the opponent's half; and the same for every pairing.

Input : the team models (team_policy.py, team_decisions.py)
Output: outputs/team_league/<name>_games.csv (one row per match), _table.csv, _pairs.csv, _chart.png
Needs : the GRF environment with torch (.venv)

Usage :  .venv\\Scripts\\python.exe scripts\\team_league.py --teams Argentina France Croatia --games 6 --procs 4
         .venv\\Scripts\\python.exe scripts\\team_league.py --report                     (only the tables and the chart)
         .venv\\Scripts\\python.exe scripts\\team_league.py --video Argentina France     (record one match as GRF video)
"""
import argparse
import itertools
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

OUT = "outputs/team_league"
_M = {}


def _init(dataset):
    import torch
    torch.set_num_threads(1)
    import team_agent
    _M["models"] = team_agent.load_models(dataset)


def _play(job):
    import team_agent
    left, right, seed = job
    t0 = time.time()
    try:
        r = team_agent.play_game(left, right, *_M["models"], seed=seed)
    except Exception as e:                                       # one broken match must not stop the league
        return {"left": left, "right": right, "seed": seed, "error": repr(e)}
    r["minutes"] = round((time.time() - t0) / 60, 2)
    return r


def schedule(teams, games, ai_games):
    jobs, seed = [], 0
    for a, b in itertools.combinations(teams + ["pooled"], 2):
        for g in range(games):
            jobs += [(a, b, seed), (b, a, seed + 1)]
            seed += 2
    for t in teams + ["pooled"]:
        for g in range(ai_games):
            jobs.append((t, "GRF AI", seed))
            seed += 1
    return jobs


def team_rows(df):
    """One row per team per match, from that team's point of view."""
    rows = []
    for r in df.itertuples():
        for side, other in (("left", "right"), ("right", "left")):
            team, opp = getattr(r, side), getattr(r, other)
            gf, ga = getattr(r, f"goals_{side}"), getattr(r, f"goals_{other}")
            rows.append({"team": team, "opponent": opp, "side": side, "gf": gf, "ga": ga, "gd": gf - ga,
                         "points": 3 if gf > ga else 1 if gf == ga else 0,
                         "shots": getattr(r, f"{side}_shots", np.nan), "xg": getattr(r, f"{side}_xg", np.nan),
                         "passes": getattr(r, f"{side}_passes", np.nan), "pass_completion": getattr(r, f"{side}_pass_completion", np.nan),
                         "possession": getattr(r, f"{side}_possession", np.nan), "opp_half": getattr(r, f"{side}_opp_half", np.nan)})
    t = pd.DataFrame(rows)
    return t[t["team"] != "GRF AI"]


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def report(name):
    df = pd.read_csv(f"{OUT}/{name}_games.csv")
    df = df[df.get("error").isna()] if "error" in df else df
    t_all = team_rows(df)
    ai = t_all[t_all["opponent"] == "GRF AI"]
    t = t_all[t_all["opponent"] != "GRF AI"]                   # the table and chart: the models against each other only
    tab = t.groupby("team").agg(games=("gd", "size"), points_per_game=("points", "mean"), wins=("points", lambda p: int((p == 3).sum())),
                                draws=("points", lambda p: int((p == 1).sum())), losses=("points", lambda p: int((p == 0).sum())),
                                goals_for=("gf", "mean"), goals_against=("ga", "mean"), gd_per_game=("gd", "mean"), gd_ci95=("gd", ci),
                                shots=("shots", "mean"), xg=("xg", "mean"), passes=("passes", "mean"), pass_completion=("pass_completion", "mean"),
                                possession=("possession", "mean"), opp_half=("opp_half", "mean")).sort_values("points_per_game", ascending=False)
    tab.round(3).to_csv(f"{OUT}/{name}_table.csv")
    pairs = t.groupby(["team", "opponent"]).agg(games=("gd", "size"), points_per_game=("points", "mean"), gd_per_game=("gd", "mean"),
                                                 gd_ci95=("gd", ci), xg=("xg", "mean"), shots=("shots", "mean")).reset_index()
    pairs.round(3).to_csv(f"{OUT}/{name}_pairs.csv", index=False)
    print(f"\nLEAGUE TABLE, models against each other ({len(t) // 2} matches; per game; gd_ci95 = half-width of the 95% interval of the goal difference)\n")
    print(tab.round(2).to_string())
    if len(ai):
        a_tab = ai.groupby("team").agg(games=("gd", "size"), goals_for=("gf", "mean"), goals_against=("ga", "mean"), shots=("shots", "mean"))
        print("\nBASELINE: every model against GRF's own AI (per game)\n")
        print(a_tab.round(2).to_string())
        a_tab.round(3).to_csv(f"{OUT}/{name}_vs_grf_ai.csv")
    print("\nHEAD TO HEAD (row team against column team: goal difference per game)\n")
    print(pairs.pivot(index="team", columns="opponent", values="gd_per_game").round(2).to_string())
    print("\nPOINTS PER GAME HEAD TO HEAD\n")
    print(pairs.pivot(index="team", columns="opponent", values="points_per_game").round(2).to_string())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        order = list(tab.index)
        ax[0].bar(order, tab["points_per_game"], color="#3a7bd5")
        ax[0].set_title("Points per game")
        ax[1].bar(order, tab["gd_per_game"], yerr=tab["gd_ci95"], capsize=4, color="#e0473c")
        ax[1].axhline(0, color="0.4", lw=0.8)
        ax[1].set_title("Goal difference per game (95% interval)")
        ax[2].bar(np.arange(len(order)) - 0.2, tab["shots"], 0.4, label="shots", color="#888")
        ax[2].bar(np.arange(len(order)) + 0.2, tab["xg"] * 10, 0.4, label="xG proxy x10", color="#f2c14e")
        ax[2].set_xticks(range(len(order)), order)
        ax[2].legend()
        ax[2].set_title("Chances per game")
        for a in ax:
            a.tick_params(axis="x", rotation=20)
        fig.suptitle(f"Team models against each other in GRF: {len(t) // 2} matches (both sides swapped; games against GRF AI not included)")
        fig.tight_layout()
        fig.savefig(f"{OUT}/{name}_chart.png", dpi=120)
        print(f"\nwrote {OUT}/{name}_table.csv, _pairs.csv, _chart.png")
    except ImportError:
        print(f"\nwrote {OUT}/{name}_table.csv, _pairs.csv (no matplotlib here for the chart)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pff")
    ap.add_argument("--teams", nargs="+", default=["Argentina", "France", "Croatia"])
    ap.add_argument("--games", type=int, default=6, help="matches per pair per side")
    ap.add_argument("--ai-games", type=int, default=4, help="matches of each team against GRF's own bot")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--name", default="league")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--video", nargs=2, metavar=("LEFT", "RIGHT"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.report:
        report(a.name)
        return
    if a.video:
        import grf_names
        grf_names.setup_from_args()                     # the teams' real players' names in GRF's picture
        import cv2, team_agent
        path = f"{OUT}/{a.video[0]}_vs_{a.video[1]}.mp4".replace(" ", "_")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
        tr = []
        r = team_agent.play_game(a.video[0], a.video[1], *team_agent.load_models(a.dataset), seed=a.seed, video=vw, tracks=tr)
        vw.release()
        # the same match as positions, for the 3D viewer (export_viewer.py --grf-match)
        lineup = [grf_names.team_lineup(t) if t in grf_names.LATEST else ([f"{t} {k + 1}" for k in range(11)], [k + 1 for k in range(11)])
                  for t in a.video]
        np.savez_compressed(path.replace(".mp4", ".npz"), players=np.array([x[0] for x in tr]), ball=np.array([x[1] for x in tr]),
                            score=np.array([x[2] for x in tr]), game_mode=np.array([x[3] for x in tr]), teams=np.array(a.video),
                            names_left=np.array(lineup[0][0]), shirts_left=np.array([int(x) for x in lineup[0][1]]),
                            names_right=np.array(lineup[1][0]), shirts_right=np.array([int(x) for x in lineup[1][1]]), seed=a.seed)
        print(r, "\nwrote", path, "and", path.replace(".mp4", ".npz"))
        return
    path = f"{OUT}/{a.name}_games.csv"
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    have = set(zip(done["left"], done["right"], done["seed"])) if len(done) else set()
    jobs = [j for j in schedule(a.teams, a.games, a.ai_games) if j not in have]
    print(f"{len(jobs)} matches to play ({len(have)} already in {path}), {a.procs} at a time", flush=True)
    t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.dataset,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_play, jobs)):
            pd.DataFrame([r]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            el = (time.time() - t0) / 60
            print(f"  {i + 1}/{len(jobs)}  {r['left']} {r.get('goals_left', '?')}-{r.get('goals_right', '?')} {r['right']}"
                  f"  ({el:.0f} min, about {el / (i + 1) * (len(jobs) - i - 1):.0f} min to go)" + (f"  ERROR {r['error']}" if "error" in r else ""), flush=True)
    report(a.name)


if __name__ == "__main__":
    main()
