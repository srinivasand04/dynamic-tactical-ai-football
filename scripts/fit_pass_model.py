"""Steps 3 and 4 of the pass work: three small models, all learned from real passes and judged on matches they never saw.

    A  CHOICE      "who will the passer pick?"        softmax over the teammates (a conditional logit), scored from the numbers of
                   pass_features.py. Compared step by step: still picture only  ->  + arrival point  ->  + time margin  ->  + running.
    B  COMPLETION  "will the pass arrive?"            logistic regression on the pass the player really played (complete / failed).
    C  VALUE       "how dangerous is the place the ball arrives at?"   chance that the team shoots within 10 s of having the
                   ball there (an 'expected threat' map), learned from the real matches' tracking and shot events.

    expected gain of an option = chance it arrives (B) x danger of the arrival point (C)        <- used in decision_quality.py

How the tests are made: leave-one-match-out (learn on 6 matches, test on the 7th, repeat). The completion model is judged by AUC
(1 = perfect, 0.5 = guessing). The value map is smooth (8 numbers, ~30 000 samples), so it is fitted on all matches.

Output : the tables, outputs/pass_model.json (all models, read by decision_quality.py), outputs/pass_model.csv, outputs/pass_model.png
Usage  :  python scripts\\fit_pass_model.py
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import pass_features as P
import runlog
import settings

MODEL = "outputs/pass_model.json"
COL = {c: i for i, c in enumerate(P.COLUMNS)}
STILL, FULL = P.GROUPS["still picture"], P.COLUMNS
VALUE_HORIZON_S, VALUE_STEP = 10.0, 10                                    # a shot within 10 s; one sample per second


def prepare(z):
    """One match from its pass_features arrays: inputs (n, 10, 13), mask, target, complete, dest, offside, and when and who."""
    return {"I": P.to_inputs(z["X"]), "mask": z["mask"], "target": z["target"], "complete": z["complete"], "dest": z["dest"],
            "offside": z["X"][..., P.RAW.index("offside")] > 0.5, "frame": z["frame"], "time_s": z["time_s"], "passer": z["passer"]}


def load(matches):
    return {m: prepare(np.load(settings.base(m) + "_passfeat.npz")) for m in matches}


def cat(data, ms, key, only_complete=False):
    parts = [data[m][key][data[m]["complete"]] if only_complete else data[m][key] for m in ms]
    return np.concatenate(parts)


def idx(cols):
    return [COL[c] for c in cols]


# ---- A  choice ----------------------------------------------------------------------------------------------------------
def fit_choice(I, mask, target, cols, steps=300, l2=1e-3):
    X, M, T = torch.tensor(I[..., idx(cols)], dtype=torch.float32), torch.tensor(mask), torch.tensor(target)
    w = torch.zeros(len(cols), requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        s = (X @ w).masked_fill(~M, -1e9)
        loss = torch.nn.functional.cross_entropy(s, T) + l2 * (w ** 2).sum()
        loss.backward()
        opt.step()
    return w.detach().numpy()


def choice_scores(w, I, mask, cols):
    return np.where(mask, I[..., idx(cols)] @ w, -1e9)


def choice_metrics(s, target):
    mine = s[np.arange(len(s)), target]
    rank = 1 + (s > mine[:, None]).sum(1)
    lse = np.log(np.exp(s - s.max(1, keepdims=True)).sum(1)) + s.max(1)
    return {"top1_%": 100 * float((rank <= 1).mean()), "top3_%": 100 * float((rank <= 3).mean()), "log_loss": float((lse - mine).mean())}


# ---- B  completion ------------------------------------------------------------------------------------------------------
COMPLETION_COLS = [c for c in FULL if c != "offside"]


def chosen(I, target):
    return I[np.arange(len(I)), target]


def fit_logistic(X, y, steps=400, l2=1e-2):
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    w, b = torch.zeros(X.shape[1], requires_grad=True), torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + l2 * (w ** 2).sum()
        loss.backward()
        opt.step()
    return w.detach().numpy(), float(b.detach())


def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_completion(data, ms):
    X = chosen(cat(data, ms, "I"), cat(data, ms, "target"))[:, idx(COMPLETION_COLS)]
    return fit_logistic(X, cat(data, ms, "complete").astype(float))


def completion_prob(model, I):
    w, b = model
    return sigmoid(I[..., idx(COMPLETION_COLS)] @ w + b)


# ---- C  value -----------------------------------------------------------------------------------------------------------
def value_features(x_att, y):
    u, w = np.asarray(x_att) / P.L, np.abs(np.asarray(y) - P.W / 2) / (P.W / 2)
    return np.stack([u, u ** 2, u ** 3, w, w ** 2, u * w, u ** 2 * w], axis=-1)


def value_dataset(m):
    """One sample per second of live play: where the ball is (attacking direction of the team that has it), and whether
    that team shoots within the next 10 s."""
    f = pd.read_csv(settings.csv(m), usecols=["frame", "time_s", "ball_x_m", "ball_y_m", "ball_owner_team", "game_mode", "period"]).drop_duplicates("frame")
    period2 = float(f.loc[f["period"] == 2, "time_s"].min())
    pl = pd.read_csv(f"data/real/raw/{m}_players.csv")
    side = pl.drop_duplicates("team_id").set_index("team_id")["ground"]
    e = pd.read_csv(f"data/real/raw/{m}_events.csv.gz")
    e = e[e["event_type"] == "SHOT"].copy()
    e["time_s"] = pd.to_timedelta(e["timestamp"]).dt.total_seconds() + np.where(e["period_id"] == 2, period2, 0.0)
    shots = {s: np.sort(e.loc[e["team_id"].map(side) == s, "time_s"].to_numpy()) for s in ("home", "away")}
    g = f[(f["frame"] % VALUE_STEP == 0) & (f["game_mode"] == "normal") & f["ball_owner_team"].isin(["home", "away"]) & f["ball_x_m"].notna()]
    x_att = np.where(g["ball_owner_team"] == "home", g["ball_x_m"], P.L - g["ball_x_m"])
    y = np.zeros(len(g))
    t = g["time_s"].to_numpy()
    for s in ("home", "away"):
        sel = (g["ball_owner_team"] == s).to_numpy()
        j = np.searchsorted(shots[s], t[sel], side="right")
        nxt = np.where(j < len(shots[s]), shots[s][np.minimum(j, len(shots[s]) - 1)], np.inf)
        y[sel] = (nxt - t[sel]) <= VALUE_HORIZON_S
    return x_att, g["ball_y_m"].to_numpy(), y


def fit_value(matches):
    parts = [value_dataset(m) for m in matches]
    x, yy, lab = (np.concatenate(c) for c in zip(*parts))
    w, b = fit_logistic(value_features(x, yy), lab, steps=600, l2=1e-4)
    return (w, b), len(lab), float(lab.mean())


def value_at(model, x_att, y):
    w, b = model
    return sigmoid(value_features(x_att, y) @ w + b)


def to_json(choice_w, comp, val):
    return {"choice": {"cols": FULL, "w": [round(float(v), 4) for v in choice_w]},
            "completion": {"cols": COMPLETION_COLS, "w": [round(float(v), 4) for v in comp[0]], "b": round(comp[1], 4)},
            "value": {"w": [round(float(v), 4) for v in val[0]], "b": round(val[1], 4)}}


def from_json(path=MODEL):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run scripts/fit_pass_model.py first")
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    return (np.array(j["choice"]["w"]), (np.array(j["completion"]["w"]), j["completion"]["b"]), (np.array(j["value"]["w"]), j["value"]["b"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    torch.manual_seed(0)
    matches = [m for m in settings.available(settings.REAL) if os.path.exists(settings.base(m) + "_passfeat.npz")]
    if len(matches) < 2:
        raise SystemExit("run scripts/pass_features.py first")
    data = load(matches)
    n_all = sum(len(d["target"]) for d in data.values())
    n_ok = sum(int(d["complete"].sum()) for d in data.values())
    print(f"{n_all} real passes ({n_ok} completed) in {len(matches)} matches\n")

    variants = {"still picture": STILL,
                "+ arrival point": STILL + P.GROUPS["arrival point"],
                "+ time margin": STILL + P.GROUPS["arrival point"] + P.GROUPS["time margin"],
                "+ running (all)": FULL,
                "all, no arrival point": [c for c in FULL if c not in P.GROUPS["arrival point"]],
                "all, no time margin": [c for c in FULL if c != "margin"],
                "all, no running": [c for c in FULL if c not in P.GROUPS["running"]]}
    rows = []
    for m in matches:
        tr = [x for x in matches if x != m]
        I, M, T = cat(data, tr, "I", True), cat(data, tr, "mask", True), cat(data, tr, "target", True)
        Ite, Mte, Tte = (data[m][k][data[m]["complete"]] for k in ("I", "mask", "target"))
        n = Mte.sum(1)
        rows.append({"held_out": m, "model": "chance", "top1_%": 100 * float((1 / n).mean()), "top3_%": 100 * float(np.minimum(3 / n, 1).mean()),
                     "log_loss": float(np.log(n).mean())})
        rows.append({"held_out": m, "model": "nearest teammate", **choice_metrics(np.where(Mte, -Ite[..., COL["dist"]], -1e9), Tte)})
        for name, cols in variants.items():
            w = fit_choice(I, M, T, cols, a.steps)
            rows.append({"held_out": m, "model": name, **choice_metrics(choice_scores(w, Ite, Mte, cols), Tte)})
        print(f"  choice, held out {m}: still {rows[-len(variants)]['top1_%']:.1f}%  all {rows[-len(variants) + 3]['top1_%']:.1f}%", flush=True)
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df["model"]))
    t = df.groupby("model")[["top1_%", "top3_%", "log_loss"]].mean().loc[order]
    print("\nA  WHO WILL THE PASSER PICK?  (average over held-out matches; chance is about 10%)\n")
    print(t.round(2).to_string())
    print("\n  step-2 reference (static model with learned distances): top-1 41.5%, top-3 78.0%, log-loss 1.63")

    comp_rows = []
    for m in matches:
        tr = [x for x in matches if x != m]
        model = fit_completion(data, tr)
        te = chosen(data[m]["I"], data[m]["target"])
        y = data[m]["complete"].astype(int)
        p = sigmoid(te[:, idx(COMPLETION_COLS)] @ model[0] + model[1])
        base = fit_logistic(chosen(cat(data, tr, "I"), cat(data, tr, "target"))[:, [COL["dist"], COL["dist2"]]], cat(data, tr, "complete").astype(float))
        pb = sigmoid(te[:, [COL["dist"], COL["dist2"]]] @ base[0] + base[1])
        comp_rows.append({"held_out": m, "AUC all numbers": auc(y, p), "AUC pass length only": auc(y, pb), "completed_%": 100 * y.mean(),
                          "brier": float(((p - y) ** 2).mean()), "brier_always_mean": float(((y.mean() - y) ** 2).mean())})
    ct = pd.DataFrame(comp_rows).set_index("held_out")
    print("\nB  WILL THE PASS ARRIVE?  (held-out matches)\n")
    print(ct.round(3).to_string())
    print("\n  mean:", ct.mean().round(3).to_dict())
    os.makedirs("outputs", exist_ok=True)
    ct.round(4).to_csv("outputs/pass_model_completion.csv")

    val, n_val, rate = fit_value(matches)
    print(f"\nC  HOW DANGEROUS IS THE PLACE?  ({n_val} one-per-second samples, {100 * rate:.1f}% are followed by a shot within 10 s)\n")
    xs, ys = [10, 30, 50, 70, 85, 95], [34, 20, 8]
    print("  chance of a shot within 10 s, by where the ball is (attacking left to right)")
    print("            " + "".join(f"x={x:>3d}m  " for x in xs))
    for yy, nm in zip(ys, ["centre", "half-wide", "wide"]):
        print(f"  {nm:10s}" + "".join(f"{100 * float(value_at(val, np.array(x), np.array(yy))):6.1f}%  " for x in xs))

    w_all = fit_choice(cat(data, matches, "I", True), cat(data, matches, "mask", True), cat(data, matches, "target", True), FULL, a.steps)
    comp = fit_completion(data, matches)
    print("\nLEARNED CHOICE WEIGHTS (all real matches; positive = the passer likes it more)\n")
    for c, v in sorted(zip(FULL, w_all), key=lambda x: -abs(x[1])):
        print(f"  {c:14s} {v:+.2f}")
    os.makedirs("outputs", exist_ok=True)
    j = to_json(w_all, comp, val)
    j["cv"] = {"choice_top1_%": {k: round(float(v), 1) for k, v in t["top1_%"].items()}, "completion_auc": round(float(ct["AUC all numbers"].mean()), 3)}
    with open(MODEL, "w", encoding="utf-8") as f:
        json.dump(j, f, indent=2)
    df.round(3).to_csv("outputs/pass_model.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    names = ["chance", "nearest teammate"] + list(variants)[:4]
    y = np.arange(len(names))[::-1]
    ax[0].barh(y, t.loc[names, "top1_%"], height=0.6, color=["#b4b2a9"] * 2 + ["#2a78d6"] * 4)
    for yy, v in zip(y, t.loc[names, "top1_%"]):
        ax[0].text(v + 0.5, yy, f"{v:.0f}%", va="center", fontsize=9, color="#3d3d3a")
    ax[0].set_yticks(y, names)
    ax[0].set_xlabel("real receiver ranked first (%)")
    ax[0].set_title("A  who will the passer pick?", loc="left", fontsize=10)
    xc = np.arange(2)
    ax[1].bar(xc, [ct["AUC pass length only"].mean(), ct["AUC all numbers"].mean()], width=0.5, color=["#b4b2a9", "#2a78d6"])
    ax[1].axhline(0.5, color="#73726c", lw=1, ls=(0, (4, 3)))
    ax[1].set_ylim(0.4, 0.9)
    for xx, v in zip(xc, [ct["AUC pass length only"].mean(), ct["AUC all numbers"].mean()]):
        ax[1].text(xx, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, color="#3d3d3a")
    ax[1].set_xticks(xc, ["pass length only", "all numbers"])
    ax[1].set_title("B  will it arrive?  (AUC, dashed = guessing)", loc="left", fontsize=10)
    gx, gy = np.meshgrid(np.linspace(0, 105, 106), np.linspace(0, 68, 69))
    im = ax[2].imshow(100 * value_at(val, gx, gy), origin="lower", extent=[0, 105, 0, 68], cmap="YlOrRd", aspect="equal")
    ax[2].set_title("C  danger of the place (% shot within 10 s)", loc="left", fontsize=10)
    ax[2].set_xlabel("attacking direction  →")
    plt.colorbar(im, ax=ax[2], fraction=0.04)
    for axx in ax[:2]:
        for side in ("right", "top"):
            axx.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/pass_model.png", dpi=150)
    runlog.log({"script": "fit_pass_model", "passes": n_all, "top1_still": round(float(t.loc["still picture", "top1_%"]), 1),
                "top1_all": round(float(t.loc["+ running (all)", "top1_%"]), 1), "completion_auc": round(float(ct["AUC all numbers"].mean()), 3)})
    print(f"\nsaved {MODEL}, outputs/pass_model.csv, outputs/pass_model.png")


if __name__ == "__main__":
    main()
