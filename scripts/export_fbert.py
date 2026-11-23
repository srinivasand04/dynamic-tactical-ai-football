"""FootballBERT player vectors for our project (World Cup 2022 / PFF players).

1. graph-model input: for every PFF match with Delaunay graphs, a vector per Key State and node slot
   (PCA-16 of FootballBERT's fixed player embedding + a 'known' flag; unknown players and the ball are zeros)
   -> data/pff/processed/<match>_fbert.npz  (emb (K, 23, 17), aligned to the graph file's Key States)
   used by: train_gnn.py --player-emb
2. app: player cards -> viewer/data/player_cards.json (role, team, the 5 most similar World Cup players by cosine)

The model is rebuilt in plain torch (footballbert_trial.py); only safetensors weights are read.
Usage : .venv-vision\\Scripts\\python.exe scripts\\export_fbert.py
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import footballbert_trial as FB  # noqa: E402

ROOT = FB.ROOT
PFF = os.path.join(ROOT, "data", "pff", "processed")
DIM = 16


def main():
    names = json.load(open(os.path.join(FB.HF, "player_ids2names.json"), encoding="utf-8"))
    vocab = json.load(open(os.path.join(FB.HF, "player_vocab.json"), encoding="utf-8"))
    by_name = {}
    for pid, n in names.items():
        if pid in vocab:
            by_name.setdefault(FB.norm(n), []).append(vocab[pid])

    def find(*cands):
        for c in cands:
            ids = by_name.get(FB.norm(c), []) if isinstance(c, str) and c else []
            if len(ids) == 1:
                return ids[0]
            if len(ids) > 1:
                return None
        return None

    R = FB.wc_players()
    rost = []
    for f in sorted(glob.glob(os.path.join(FB.WC, "Rosters", "*.json"))):
        for r in json.load(open(f, encoding="utf-8")):
            rost.append(dict(match=os.path.basename(f)[:-5], pff_id=r["player"]["id"], shirt=int(r["shirtNumber"]), team=r["team"]["name"]))
    rost = pd.DataFrame(rost)
    U = R.drop_duplicates("pff_id").copy()
    U["tok"] = [find(a, b) for a, b in zip(U["name"], U["full"])]
    U = U.dropna(subset=["tok"]).astype({"tok": int})
    tok = dict(zip(U["pff_id"], U["tok"]))

    model, cfg = FB.load_model()
    E = model.decoder.weight.detach().numpy()
    X = E[U["tok"].values]
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:DIM]
    scale = (X - mu) @ P.T
    sd = scale.std(0) + 1e-6
    vec = {t: ((E[t] - mu) @ P.T) / sd for t in U["tok"].values}
    print(f"{len(U)} World Cup players known; PCA-{DIM} keeps {(S[:DIM] ** 2).sum() / (S ** 2).sum():.0%} of the variance")

    # 1. per-match slot vectors aligned to the graph files
    M = pd.read_csv(os.path.join(ROOT, "data", "pff", "matches.csv"), dtype={"match_id": str})
    done = 0
    for _, m in M.iterrows():
        gpath = os.path.join(PFF, f"{m.match_id}_graphs_delaunay.npz")
        if not os.path.exists(gpath):
            continue
        g = np.load(gpath)
        frames = g["frame"]
        d = pd.read_csv(os.path.join(PFF, f"{m.match_id}.csv.gz"), usecols=["frame", "team", "player_id", "jersey"])
        d = d[d["frame"].isin(set(frames.tolist()))]
        ro = rost[rost["match"] == m.match_id]
        side = {"home": m.home, "away": m.away}
        key = {(t, s): tok.get(p) for t, s, p in zip(ro["team"], ro["shirt"], ro["pff_id"])}
        fi = {f: i for i, f in enumerate(frames.tolist())}
        emb = np.zeros((len(frames), 23, DIM + 1), np.float32)
        hit = tot = 0
        for f, team, slot, jersey in d.itertuples(index=False):
            if pd.isna(jersey):
                continue
            s = int(slot) + (0 if team == "home" else 11)
            t = key.get((side[team], int(jersey)))
            tot += 1
            if t is not None:
                emb[fi[f], s, :DIM] = vec[t]; emb[fi[f], s, DIM] = 1.0; hit += 1
        np.savez_compressed(os.path.join(PFF, f"{m.match_id}_fbert.npz"), emb=emb, frame=frames)
        done += 1
        print(f"  {m.match_id} {m.home} v {m.away}: {hit / max(tot, 1):.0%} of player nodes have a vector", flush=True)
    print(f"wrote {done} files to data/pff/processed/*_fbert.npz")

    # 2. player cards for the app
    pos = R.groupby("pff_id")["pos"].agg(lambda s: Counter(s).most_common(1)[0][0])
    team = R.groupby("pff_id")["team"].first()
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -1)
    ids = U["pff_id"].tolist()
    cards = {}
    for i, pid in enumerate(ids):
        top = np.argsort(-sim[i])[:5]
        cards[pid] = dict(name=U.iloc[i]["name"], team=team[pid], role=FB.POS_MAP.get(pos.get(pid), pos.get(pid)),
                          line=FB.LINE.get(FB.POS_MAP.get(pos.get(pid), ""), ""),
                          similar=[dict(name=U.iloc[j]["name"], team=team[ids[j]], sim=round(float(sim[i, j]), 2)) for j in top])
    # slot lookup so the viewer can open a card from a PFF match: match -> side -> shirt -> pff_id
    lookup = {}
    for _, m in M.iterrows():
        ro = rost[rost["match"] == m.match_id]
        lookup[m.match_id] = {sd: {str(s): p for t, s, p in zip(ro["team"], ro["shirt"], ro["pff_id"]) if t == tn and p in cards}
                              for sd, tn in (("home", m.home), ("away", m.away))}
    out = os.path.join(ROOT, "viewer", "data", "player_cards.json")
    json.dump(dict(source="FootballBERT (achadj/footballbert-base @2f9107d), fixed player embeddings, cosine similarity among "
                          f"{len(cards)} World Cup 2022 players", cards=cards, matches=lookup), open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {os.path.relpath(out, ROOT)} ({len(cards)} cards)")
    ex = next(p for p in ids if cards[p]["name"] in ("Lionel Messi", "Messi")) if any(cards[p]["name"] in ("Lionel Messi", "Messi") for p in ids) else ids[0]
    print("example:", cards[ex]["name"], "->", [s["name"] for s in cards[ex]["similar"]])


if __name__ == "__main__":
    main()
