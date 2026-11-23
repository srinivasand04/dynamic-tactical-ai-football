"""Trial of FootballBERT (achadj/footballbert-base, Apache-2.0) on our players.

The network is rebuilt here in plain PyTorch from the published modeling_footballbert.py and only the weights
(model.safetensors) are loaded: no downloaded code is run. The id->name map was read with a pickle loader that refuses
every Python class (plain dict of strings only) and saved as player_ids2names.json.

Tests
  0. coverage   how many of our World Cup 2022 (PFF) and Bundesliga (DFL) players the model knows (by name)
  A. roles      do the player embeddings know each player's real role? softmax regression, 5-fold cross-validation,
                vs the majority class and vs random embeddings (control)
  B. lineups    mask one starter of a real World Cup lineup; where does the model rank the real player among 99,943?
Usage : .venv-vision\\Scripts\\python.exe scripts\\footballbert_trial.py
Output: outputs/footballbert/
"""
import glob
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = os.path.join(ROOT, "data", "external", "hf", "footballbert")
WC = os.path.join(ROOT, "data", "external", "pff_worldcup", "raw", "FIFA World Cup 2022")
OUT = os.path.join(ROOT, "outputs", "footballbert")
torch.manual_seed(0); np.random.seed(0)

POS_MAP = {"GK": "GK", "LB": "LB", "RB": "RB", "LCB": "CB", "RCB": "CB", "MCB": "CB", "CB": "CB", "LWB": "WB", "RWB": "WB",
           "DM": "DM", "CM": "CM", "AM": "AM", "LM": "LM", "RM": "RM", "LW": "LW", "RW": "RW", "CF": "FW", "ST": "FW"}
LINE = {"GK": "GK", "LB": "DEF", "RB": "DEF", "CB": "DEF", "WB": "DEF", "DM": "MID", "CM": "MID", "AM": "MID", "LM": "MID",
        "RM": "MID", "LW": "FWD", "RW": "FWD", "FW": "FWD"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower().replace("-", " ")).strip()


# ------------------------------------------------------------------ the network (same as modeling_footballbert.py)
class Attn(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h, s.hd = h, d // h
        s.values, s.keys, s.queries = (nn.Linear(s.hd, s.hd, bias=False) for _ in range(3))
        s.fc_out = nn.Linear(d, d)

    def forward(s, x, mask):
        N, L, _ = x.shape
        v, k, q = (f(x.reshape(N, L, s.h, s.hd)) for f in (s.values, s.keys, s.queries))
        e = torch.einsum("nqhd,nkhd->nhqk", q, k).masked_fill(mask[:, None, None, :] == 0, -1e20)
        a = torch.softmax(e / s.hd ** 0.5, dim=3)
        return s.fc_out(torch.einsum("nhql,nlhd->nqhd", a, v).reshape(N, L, -1))


class Block(nn.Module):
    def __init__(s, d, h, ff):
        super().__init__(); s.attention = Attn(d, h); s.norm1, s.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.feed_forward = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(s, x, mask):
        x = s.norm1(s.attention(x, mask) + x)
        return s.norm2(s.feed_forward(x) + x)


class FootballBERT(nn.Module):
    def __init__(s, c):
        super().__init__()
        d = c["hidden_size"]
        s.players_embeddings = nn.Embedding(c["vocab_size"] + 1, d)
        s.positions_embeddings = nn.Embedding(c["num_positions"] + 1, d)
        s.team_affiliation_embeddings = nn.Embedding(3, d)
        s.seasons_embeddings = nn.Embedding(c["num_seasons"] + 1, d)
        s.layers = nn.ModuleList([Block(d, c["num_attention_heads"], c["intermediate_size"]) for _ in range(c["num_hidden_layers"])])
        s.decoder = nn.Linear(d, c["vocab_size"])

    def forward(s, pid, pos, team, season, mask):
        x = torch.relu(s.seasons_embeddings(season) + s.positions_embeddings(pos) + s.team_affiliation_embeddings(team)
                       + s.players_embeddings(pid))
        for b in s.layers:
            x = b(x, mask)
        return s.decoder(x)


def load_model():
    c = json.load(open(os.path.join(HF, "config.json")))
    m = FootballBERT(c)
    w = load_file(os.path.join(HF, "model.safetensors"))
    w = {k.replace("footballbert.player_encoder.", ""): v for k, v in w.items()}
    missing, unexpected = m.load_state_dict(w, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return m.eval(), c


# ------------------------------------------------------------------ our players
def wc_players():
    rows = []
    for f in sorted(glob.glob(os.path.join(WC, "Rosters", "*.json"))):
        for r in json.load(open(f, encoding="utf-8")):
            rows.append(dict(match=os.path.basename(f)[:-5], pff_id=r["player"]["id"], name=r["player"]["nickname"],
                             pos=r["positionGroupType"], started=r["started"], team=r["team"]["name"]))
    R = pd.DataFrame(rows)
    pl = pd.read_csv(os.path.join(WC, "players.csv"), dtype={"id": str})
    full = dict(zip(pl["id"], (pl["firstName"].fillna("") + " " + pl["lastName"].fillna("")).str.strip()))
    R["full"] = R["pff_id"].map(full)
    return R


def dfl_players():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "real", "raw", "*_players.csv"))):
        d = pd.read_csv(f)
        d["match"] = os.path.basename(f).split("_")[0]
        rows.append(d)
    return pd.concat(rows)


def softmax_cv(X, y, folds=5, l2=1e-3, epochs=300):
    """Multinomial logistic regression with k-fold CV, plain torch. Returns accuracy over held-out players."""
    classes = sorted(set(y)); yi = np.array([classes.index(v) for v in y])
    idx = np.random.permutation(len(y)); pred = np.zeros(len(y), int)
    for k in range(folds):
        te = idx[k::folds]; tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt, Xv = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32), torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
        lin = nn.Linear(X.shape[1], len(classes)); opt = torch.optim.LBFGS(lin.parameters(), max_iter=epochs)
        yt = torch.tensor(yi[tr])

        def closure():
            opt.zero_grad(); l = nn.functional.cross_entropy(lin(Xt), yt) + l2 * (lin.weight ** 2).sum(); l.backward(); return l
        opt.step(closure)
        pred[te] = lin(Xv).argmax(1).numpy()
    return (pred == yi).mean()


def main():
    os.makedirs(OUT, exist_ok=True)
    names = json.load(open(os.path.join(HF, "player_ids2names.json"), encoding="utf-8"))
    vocab = json.load(open(os.path.join(HF, "player_vocab.json"), encoding="utf-8"))
    by_name = defaultdict(list)
    for pid, n in names.items():
        if pid in vocab:
            by_name[norm(n)].append(vocab[pid])

    def find(*cands):
        for c in cands:
            ids = by_name.get(norm(c), []) if isinstance(c, str) and c else []
            if len(ids) == 1:
                return ids[0], "unique"
            if len(ids) > 1:
                return None, "ambiguous"
        return None, "missing"

    # 0. coverage
    R = wc_players()
    found = R.drop_duplicates("pff_id").apply(lambda r: find(r["name"], r["full"]), axis=1)
    U = R.drop_duplicates("pff_id").assign(tok=[f[0] for f in found], how=[f[1] for f in found])
    D = dfl_players().drop_duplicates("player_id")
    dfound = D["name"].apply(lambda n: find(n))
    print("coverage (unique name match):")
    print(f"  World Cup 2022: {(U['how'] == 'unique').sum()} of {len(U)} players "
          f"({(U['how'] == 'unique').mean() * 100:.0f}%), ambiguous {(U['how'] == 'ambiguous').sum()}")
    print(f"  Bundesliga (DFL names are 'F. Kainz' style): {(dfound.str[1] == 'unique').sum()} of {len(D)}")

    # A. roles from embeddings
    model, cfg = load_model()
    K = U[U["how"] == "unique"].copy()
    pos = R.groupby("pff_id")["pos"].agg(lambda s: Counter(s).most_common(1)[0][0])   # most common PFF role per player
    K["role"] = K["pff_id"].map(pos).map(POS_MAP)
    K = K.dropna(subset=["role"])
    K["line"] = K["role"].map(LINE)
    tok = torch.tensor(K["tok"].values.astype(int))
    with torch.no_grad():
        E_out = model.decoder.weight[tok].numpy()                 # 'fixed' embedding (the model's own definition)
        E_in = model.players_embeddings.weight[tok].numpy()       # input embedding
    rnd = np.random.randn(*E_out.shape)
    rows = []
    for target in ("line", "role"):
        y = K[target].values
        maj = Counter(y).most_common(1)[0][1] / len(y)
        rows.append(dict(task=f"{target} ({len(set(y))} classes)", players=len(y), majority=round(maj, 3),
                         random_embedding=round(softmax_cv(rnd, y), 3), footballbert_fixed=round(softmax_cv(E_out, y), 3),
                         footballbert_input=round(softmax_cv(E_in, y), 3)))
    A = pd.DataFrame(rows)
    print("\nA. Do the embeddings know the player's real role? (accuracy on held-out players, 5-fold)")
    print(A.to_string(index=False))
    A.to_csv(os.path.join(OUT, "roles.csv"), index=False)

    # B. masked starter in real World Cup lineups
    pvocab = json.load(open(os.path.join(HF, "position_vocab.json")))
    svocab = json.load(open(os.path.join(HF, "season_vocab.json")))
    tokmap = dict(zip(U["pff_id"], U["tok"]))
    res = []
    for season in ("2022", "2022-2023"):
        ranks = []
        for mid, g in R[R["started"]].groupby("match"):
            g = g.assign(tok=g["pff_id"].map(tokmap), p=g["pos"].map(POS_MAP))
            teams = list(g["team"].unique())
            if len(teams) != 2:
                continue
            g = pd.concat([g[g["team"] == teams[0]], g[g["team"] == teams[1]]])
            ok = g["tok"].notna().values
            pid = torch.tensor(np.where(ok, g["tok"].fillna(0).astype(int), cfg["vocab_size"]))[None]
            ppos = torch.tensor([pvocab.get(p, cfg["num_positions"]) for p in g["p"]])[None]
            team = torch.tensor([0 if t == teams[0] else 1 for t in g["team"]])[None]
            seas = torch.full_like(pid, svocab[season])
            mask = torch.tensor(ok.astype(int))[None]
            for i in np.where(ok)[0]:
                x = pid.clone(); x[0, i] = cfg["mask_token_id"]
                m = mask.clone(); m[0, i] = 1
                with torch.no_grad():
                    logit = model(x, ppos, team, seas, m)[0, i]
                ranks.append(int((logit > logit[int(pid[0, i])]).sum()) + 1)
        r = np.array(ranks)
        res.append(dict(season_token=season, masked_starters=len(r), top1=round((r <= 1).mean(), 3),
                        top10=round((r <= 10).mean(), 3), top100=round((r <= 100).mean(), 3), median_rank=int(np.median(r)),
                        chance_top10=round(10 / cfg["vocab_size"], 5)))
    B = pd.DataFrame(res)
    print("\nB. Hide one starter in a real World Cup 2022 lineup: does the model name the real player? (99,943 candidates)")
    print(B.to_string(index=False))
    B.to_csv(os.path.join(OUT, "masked_lineups.csv"), index=False)
    U.to_csv(os.path.join(OUT, "worldcup_coverage.csv"), index=False)
    print(f"\nsaved {os.path.relpath(OUT, ROOT)}/")


if __name__ == "__main__":
    main()
