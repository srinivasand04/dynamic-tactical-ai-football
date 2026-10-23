"""Step 2 of the pass work: LEARN the numbers of the ball-owner weight from real passes, instead of guessing them.

What is learned (everything else in bo_graph.py stays the same)
    the five distances   free_m, reach_m, goal_m, lane_width_m, loft_m   (start: 8, 20, 30, 2.5, 15 m)
    the importance of each ingredient   free, reach, goal, lane          (start: equal, 0.25 each = geometric mean)

How
    For every completed open-play pass, each teammate gets a score  sum_k  a_k * log(ingredient_k)
    (the log of a weighted geometric mean). The scores become chances with a softmax, and the numbers are tuned so the
    REAL receiver gets as high a chance as possible (the same idea as a logistic regression, with one choice out of ~10).
    An offside teammate gets a learned penalty (our rule keeps flipping his sign in the graph itself).
    importance_k = a_k / sum(a); sum(a) only says how sure the model is, not which ingredient matters.

How it is judged (fair: never on the matches it learned from)
    leave-one-match-out: learn on 6 real matches, test on the 7th, repeat for each match.
    top1_% / top3_%  the real receiver is ranked first / in the first three
    log_loss         how surprised the model is by the real receiver (lower = better; chance = log(teammates))
    drop one         learn again without one ingredient; the loss of top-1 shows how much that ingredient matters

Output : the tables, outputs/bo_params_learned.json (learned on ALL real matches; bo_graph.learned() reads it),
         outputs/bo_fit.csv, outputs/bo_fit.png, a line in outputs/run_log.csv
Usage  :  python scripts\\fit_bo_weights.py
          python scripts\\fit_bo_weights.py --steps 600
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import bo_graph
import receiver_check
import runlog
import settings

PARTS = ["free", "reach", "goal", "lane"]
SCALES = ["free_m", "reach_m", "goal_m", "lane_width_m", "loft_m"]
T_MAX, O_MAX = 10, 11


def geometry(matches):
    """Per pass: the raw distances the ingredients are made of, padded to 10 teammates x 11 opponents."""
    rows = []
    for m in matches:
        z, idx = receiver_check.passes(m)
        for i in idx:
            xy, pres, team = z["xy"][i], z["present"][i], z["team"]
            bo, rec = int(z["passer"][i]), int(z["receiver"][i])
            g = bo_graph.build(xy, pres, team, None, bo=bo)
            if g is None:
                continue
            e = [x for x in g["edges"] if x["type"] == "teammate"]
            mates = [x["to"] for x in e]
            if rec not in mates:
                continue
            opps = g["opps"]
            goal = np.array([105.0 if g["side"] == 0 else 0.0, 34.0])
            r = {"match": m, "target": mates.index(rec), "n": len(mates),
                 "d_opp": np.zeros(T_MAX), "d_bo": np.zeros(T_MAX), "d_goal": np.zeros(T_MAX), "offside": np.zeros(T_MAX),
                 "along": np.zeros((T_MAX, O_MAX)), "across": np.zeros((T_MAX, O_MAX)), "between": np.zeros((T_MAX, O_MAX))}
            a = xy[bo]
            for t, x in enumerate(e):
                b = xy[x["to"]]
                r["d_opp"][t] = min(np.linalg.norm(b - xy[o]) for o in opps)
                r["d_bo"][t] = np.linalg.norm(b - a)
                r["d_goal"][t] = np.linalg.norm(b - goal)
                r["offside"][t] = x["offside"]
                v = b - a
                n2 = float(v @ v)
                for j, o in enumerate(opps):
                    u = float(((xy[o] - a) @ v) / n2) if n2 > 1e-6 else -1.0
                    if 0.0 < u < 1.0:
                        r["along"][t, j] = u * np.sqrt(n2)
                        r["across"][t, j] = np.linalg.norm(xy[o] - (a + u * v))
                        r["between"][t, j] = 1.0
            rows.append(r)
    return rows


def tensors(rows):
    t = {k: torch.tensor(np.stack([r[k] for r in rows]), dtype=torch.float32)
         for k in ["d_opp", "d_bo", "d_goal", "offside", "along", "across", "between"]}
    t["mask"] = torch.tensor(np.stack([np.arange(T_MAX) < r["n"] for r in rows]))
    t["target"] = torch.tensor([r["target"] for r in rows])
    t["n"] = torch.tensor([r["n"] for r in rows], dtype=torch.float32)
    return t


class Weight(torch.nn.Module):
    """The BO weight with learnable distances and importances. use = which ingredients are switched on."""

    def __init__(self, use=PARTS, learn=True):
        super().__init__()
        p = bo_graph.PARAMS
        self.log_scale = torch.nn.Parameter(torch.log(torch.tensor([p[s] for s in SCALES])), requires_grad=learn)
        self.raw_a = torch.nn.Parameter(torch.full((4,), 0.5413))             # softplus(0.5413) = 1
        self.raw_pen = torch.nn.Parameter(torch.tensor(2.0))
        self.on = torch.tensor([1.0 if k in use else 0.0 for k in PARTS])
        self.learn = learn

    def a(self):
        if self.learn:
            return torch.nn.functional.softplus(self.raw_a) * self.on
        return torch.nn.functional.softplus(self.raw_a).mean() * self.on      # hand-set: equal importance, only the sharpness learned

    def ingredients(self, t):
        free_m, reach_m, goal_m, width, loft = torch.exp(self.log_scale)
        free = torch.clamp(t["d_opp"] / free_m, max=1.0)
        reach = torch.exp(-t["d_bo"] / reach_m)
        goal = torch.exp(-t["d_goal"] / goal_m)
        block = torch.relu(1 - t["across"] / width) * torch.exp(-t["along"] / loft) * t["between"]
        lane = 1 - block.max(dim=2).values
        return torch.stack([free, reach, goal, lane], dim=-1)

    def forward(self, t):
        logp = torch.log(torch.clamp(self.ingredients(t), min=1e-3))
        pen = torch.nn.functional.softplus(self.raw_pen) if self.learn else torch.tensor(50.0)
        s = (logp * self.a()).sum(-1) - pen * t["offside"]
        return s.masked_fill(~t["mask"], -1e9)


def fit(t, use=PARTS, learn=True, steps=300):
    model = Weight(use, learn)
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(t), t["target"])
        loss.backward()
        opt.step()
    return model


def score(model, t):
    with torch.no_grad():
        s = model(t)
        loss = float(torch.nn.functional.cross_entropy(s, t["target"]))
        mine = s.gather(1, t["target"][:, None])
        rank = 1 + (s > mine).sum(1).float()
    return {"top1_%": 100 * float((rank <= 1).float().mean()), "top3_%": 100 * float((rank <= 3).float().mean()),
            "log_loss": loss}


def nearest(t):
    s = (-t["d_bo"]).masked_fill(~t["mask"], -1e9)
    rank = 1 + (s > s.gather(1, t["target"][:, None])).sum(1).float()
    return {"top1_%": 100 * float((rank <= 1).float().mean()), "top3_%": 100 * float((rank <= 3).float().mean()),
            "log_loss": float("nan")}


def chance(t):
    return {"top1_%": 100 * float((1 / t["n"]).mean()), "top3_%": 100 * float(torch.clamp(3 / t["n"], max=1).mean()),
            "log_loss": float(torch.log(t["n"]).mean())}


def describe(model):
    a = model.a().detach().numpy()
    return {"scales": {k: round(float(v), 2) for k, v in zip(SCALES, torch.exp(model.log_scale).detach().numpy())},
            "importance": {k: round(float(v / a.sum()), 3) for k, v in zip(PARTS, a)},
            "sharpness": round(float(a.sum()), 3),
            "offside_penalty": round(float(torch.nn.functional.softplus(model.raw_pen).detach()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    torch.manual_seed(0)
    matches = [m for m in settings.available(settings.REAL) if os.path.exists(settings.base(m) + "_passes.npz")]
    if len(matches) < 2:
        raise SystemExit("need _passes.npz for at least 2 real matches: run scripts/extract_passes.py first")
    rows = geometry(matches)
    print(f"{len(rows)} completed open-play passes with a known receiver, {len(matches)} real matches\n")

    folds = []
    for m in matches:
        tr = tensors([r for r in rows if r["match"] != m])
        te = tensors([r for r in rows if r["match"] == m])
        res = {"chance": chance(te), "nearest": nearest(te),
               "weight (hand)": score(fit(tr, learn=False, steps=a.steps), te),
               "weight (learned)": score(fit(tr, steps=a.steps), te)}
        for k in PARTS:
            res[f"learned without {k}"] = score(fit(tr, use=[x for x in PARTS if x != k], steps=a.steps), te)
        for name, r in res.items():
            folds.append({"held_out": m, "method": name, **r})
        print(f"  held out {m}: hand {res['weight (hand)']['top1_%']:.1f}%  learned {res['weight (learned)']['top1_%']:.1f}%", flush=True)

    df = pd.DataFrame(folds)
    order = list(dict.fromkeys(df["method"]))
    t = df.groupby("method")[["top1_%", "top3_%", "log_loss"]].mean().loc[order]
    print("\nLEAVE-ONE-MATCH-OUT (average over the held-out matches)\n")
    print(t.round(2).to_string())
    drop = {k: t.loc["weight (learned)", "top1_%"] - t.loc[f"learned without {k}", "top1_%"] for k in PARTS}
    print("\ntop-1 lost when an ingredient is removed (points):", {k: round(v, 1) for k, v in drop.items()})

    final = fit(tensors(rows), steps=a.steps)
    learned = describe(final)
    hand = {"scales": {k: bo_graph.PARAMS[k] for k in SCALES}, "importance": {k: 0.25 for k in PARTS}}
    print("\nNUMBERS LEARNED ON ALL REAL MATCHES (start value in brackets)\n")
    for k in SCALES:
        print(f"  {k:13s} {learned['scales'][k]:7.2f}   ({hand['scales'][k]})")
    for k in PARTS:
        print(f"  importance {k:6s} {learned['importance'][k]:5.2f}   (0.25)")
    print(f"  offside penalty  {learned['offside_penalty']}")

    learned.update({"passes": len(rows), "matches": matches,
                    "held_out_top1_%": {"hand": round(float(t.loc["weight (hand)", "top1_%"]), 1),
                                        "learned": round(float(t.loc["weight (learned)", "top1_%"]), 1),
                                        "nearest": round(float(t.loc["nearest", "top1_%"]), 1),
                                        "chance": round(float(t.loc["chance", "top1_%"]), 1)}})
    os.makedirs("outputs", exist_ok=True)
    with open(bo_graph.LEARNED, "w", encoding="utf-8") as f:
        json.dump(learned, f, indent=2)
    df.round(3).to_csv("outputs/bo_fit.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    names = ["chance", "nearest", "weight (hand)", "weight (learned)"] + [f"learned without {k}" for k in PARTS]
    y = np.arange(len(names))[::-1]
    v = t.loc[names, "top1_%"]
    ax[0].barh(y, v, height=0.6, color=["#2a78d6" if n.startswith("weight") else "#b4b2a9" for n in names])
    for yy, x in zip(y, v):
        ax[0].text(x + 0.8, yy, f"{x:.0f}%", va="center", fontsize=9, color="#3d3d3a")
    ax[0].set_yticks(y, names)
    ax[0].set_xlabel("real receiver ranked first, on matches not used for learning (%)")
    ax[0].set_title("Top-1 on held-out matches", loc="left", fontsize=10)
    x = np.arange(4)
    ax[1].bar(x, [learned["importance"][k] for k in PARTS], width=0.55, color="#2a78d6")
    ax[1].axhline(0.25, color="#73726c", lw=1, ls=(0, (4, 3)))
    for xx, k in zip(x, PARTS):
        ax[1].text(xx, learned["importance"][k] + 0.01, f"{learned['importance'][k]:.2f}", ha="center", fontsize=9, color="#3d3d3a")
    ax[1].set_xticks(x, PARTS)
    ax[1].set_title("Learned importance of each ingredient  (dashed = start, 0.25)", loc="left", fontsize=10)
    for axx in ax:
        for side in ("right", "top"):
            axx.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/bo_fit.png", dpi=150)
    runlog.log({"script": "fit_bo_weights", "passes": len(rows), "steps": a.steps,
                **{f"top1_{k}": v for k, v in learned["held_out_top1_%"].items()}})
    print(f"\nsaved {bo_graph.LEARNED}, outputs/bo_fit.csv, outputs/bo_fit.png")


if __name__ == "__main__":
    main()
