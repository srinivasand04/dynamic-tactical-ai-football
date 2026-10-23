"""Stage C: a spatio-temporal graph network over Key States.

Task (self-supervised, no tactical labels needed): given the last L Key-State graphs and the time to the next
Key State, predict where every player and the ball will have moved by the next Key State.

Model
  per Key State : a graph encoder, edge-type aware message passing over the 3 edge types (team / opponent / ball)
  over time     : a GRU that reads the L graph encodings of each node in order, with the time gaps as input
  output        : the movement (dx, dy) of every node, and a graph embedding (mean of the final node states)

Baselines to beat: "nobody moves" and "everybody keeps their velocity for the time gap".

Training uses real matches only. Evaluation:  the held-out real match (never seen in training) and the simulated
matches (a different source). The gap between those errors says how far simulated dynamics are from real ones.

Input : <name>_graphs.npz from build_graphs.py
Output: printed errors; with --save-embeddings, <name>_embeddings.npz (one vector per Key State) for the next stage

Usage :  python scripts\\train_gnn.py --holdout J03WOH
         python scripts\\train_gnn.py --holdout none --save-embeddings
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

REALS = ["J03WMX", "J03WN1", "J03WPY", "J03WOH"]
SIMS = ["match_001", "match_002", "match_003", "match_004"]
PATH = {**{m: f"data/real/processed/{m}" for m in REALS}, **{m: f"data/simulated/{m}" for m in SIMS}}
N, L = 23, 3
D_MOVE = 5.0          # metres: scale of the movement we predict


def load(match):
    g = np.load(PATH[match] + "_graphs.npz")
    xy, present = g["xy"], g["present"]
    K = len(xy)
    team, role = g["team"], g["role"]
    pos = np.nan_to_num((xy - np.array([52.5, 34.0])) / np.array([52.5, 34.0]))
    vel = np.clip(g["vel"], -8, 8) / 8.0
    db = np.nan_to_num(g["dist_ball"]) / 50.0
    team1 = np.eye(3)[team][None].repeat(K, 0)
    role1 = np.eye(11)[np.clip(role + 1, 0, 10)]
    feat = np.concatenate([pos, vel, db[..., None], team1, role1, present[..., None]], axis=2)
    feat = feat * present[..., None]                                       # empty slots are all zeros
    adj = np.stack([g["adj_team"], g["adj_opp"], g["adj_ball"]], axis=1).astype(np.float32)
    dist = np.nan_to_num(np.linalg.norm(xy[:, :, None] - xy[:, None], axis=3), nan=60.0)
    dist = np.clip(dist, 0, 30) / 30.0
    disp = np.zeros_like(xy)
    disp[:-1] = np.nan_to_num(xy[1:] - xy[:-1])                            # movement from this Key State to the next
    disp_ok = np.zeros((K, N), bool)
    disp_ok[:-1] = present[1:] & present[:-1]
    gap_next = np.zeros(K)
    gap_next[:-1] = g["time_s"][1:] - g["time_s"][:-1]
    gap_prev = g["gap_s"]
    vel_ms = np.clip(g["vel"], -12, 12)
    idx = np.arange(L - 1, K - 1)                                          # windows that end at k and have a next Key State
    return dict(feat=feat.astype(np.float32), adj=adj, dist=dist.astype(np.float32), disp=(disp / D_MOVE).astype(np.float32),
                disp_ok=disp_ok, gap_next=gap_next.astype(np.float32), gap_prev=gap_prev.astype(np.float32),
                vel_ms=vel_ms.astype(np.float32), idx=idx, present=present)


def batch(d, ks, device="cpu"):
    """Stack windows ending at each k in ks: tensors shaped (B, L, ...)."""
    win = ks[:, None] + np.arange(-L + 1, 1)[None]                         # (B, L)
    t = lambda a: torch.from_numpy(a).to(device)
    gaps = np.stack([d["gap_prev"][win], np.repeat(d["gap_next"][ks][:, None], L, 1)], axis=2)
    return dict(feat=t(d["feat"][win]), adj=t(d["adj"][win]), dist=t(d["dist"][win]), gaps=t(np.log1p(gaps).astype(np.float32)),
                target=t(d["disp"][ks]), ok=t(d["disp_ok"][ks]), gap_next=t(d["gap_next"][ks]), vel=t(d["vel_ms"][ks]))


class Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.Linear(d, d)) for _ in range(3)])
        self.upd = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, adj, dist):                                       # h (B,N,d), adj (B,3,N,N), dist (B,N,N)
        B, n, d = h.shape
        hi, hj = h[:, :, None].expand(B, n, n, d), h[:, None].expand(B, n, n, d)
        pair = torch.cat([hi, hj, dist[..., None]], dim=-1)
        outs = [h]
        for e in range(3):
            a = adj[:, e, :, :, None]
            outs.append((self.msg[e](pair) * a).sum(2) / (a.sum(2) + 1e-6))
        return self.norm(h + self.upd(torch.cat(outs, dim=-1)))


class Net(nn.Module):
    def __init__(self, f_in, d=48):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(f_in, d), nn.ReLU())
        self.layers = nn.ModuleList([Layer(d), Layer(d)])
        self.gru = nn.GRU(d + 2, d, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d + 2, d), nn.ReLU(), nn.Linear(d, 2))

    def encode(self, b):
        B, T, n, f = b["feat"].shape
        h = self.inp(b["feat"].reshape(B * T, n, f))
        adj, dist = b["adj"].reshape(B * T, 3, n, n), b["dist"].reshape(B * T, n, n)
        for layer in self.layers:
            h = layer(h, adj, dist)
        h = h.reshape(B, T, n, -1)
        g = b["gaps"][:, :, None, :].expand(B, T, n, 2)
        seq = torch.cat([h, g], dim=-1).permute(0, 2, 1, 3).reshape(B * n, T, -1)
        _, last = self.gru(seq)
        return last[0].reshape(B, n, -1)                                   # (B, N, d): state of every node after reading L Key States

    def forward(self, b):
        z = self.encode(b)
        g = b["gaps"][:, -1, None, :].expand(-1, z.shape[1], 2)
        return self.out(torch.cat([z, g], dim=-1)), z


def errors(pred, b):
    """Mean error in metres over nodes present at both Key States: players, ball."""
    err = torch.linalg.norm((pred - b["target"]) * D_MOVE, dim=-1)
    ok = b["ok"].float()
    ply, ball = ok.clone(), ok.clone()
    ply[:, 22], ball[:, :22] = 0, 0
    return (err * ply).sum().item(), ply.sum().item(), (err * ball).sum().item(), ball.sum().item()


def evaluate(model, d, bs=256):
    model.eval()
    tot = np.zeros(4)
    base0, basev = np.zeros(4), np.zeros(4)
    with torch.no_grad():
        for i in range(0, len(d["idx"]), bs):
            b = batch(d, d["idx"][i:i + bs])
            pred, _ = model(b)
            tot += errors(pred, b)
            zero = torch.zeros_like(pred)
            base0 += errors(zero, b)
            cv = b["vel"] * b["gap_next"][:, None, None] / D_MOVE
            basev += errors(cv, b)
    f = lambda a: (a[0] / a[1], a[2] / a[3])
    return f(tot), f(base0), f(basev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="J03WOH", help="a real match id to hold out, or 'none'")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--save-embeddings", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    data = {m: load(m) for m in REALS + SIMS}
    train = [m for m in REALS if m != a.holdout]
    print("train on:", train, "| held-out real:", a.holdout, "| windows:", {m: len(data[m]["idx"]) for m in train})

    model = Net(data[REALS[0]]["feat"].shape[2], a.dim)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    pool = [(m, k) for m in train for k in data[m]["idx"]]
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(len(pool))
        t0, tot, cnt = time.time(), 0.0, 0
        for i in range(0, len(perm), 128):
            chunk = [pool[j] for j in perm[i:i + 128]]
            parts = [batch(data[m], np.array([k for mm, k in chunk if mm == m])) for m in train if any(mm == m for mm, _ in chunk)]
            b = {key: torch.cat([p[key] for p in parts]) for key in parts[0]}
            pred, _ = model(b)
            ok = b["ok"].float()[..., None]
            loss = (nn.functional.smooth_l1_loss(pred, b["target"], reduction="none") * ok).sum() / (ok.sum() * 2 + 1e-6)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(chunk)
            cnt += len(chunk)
        msg = f"epoch {ep + 1:2d}  train loss {tot / cnt:.4f}  ({time.time() - t0:.0f}s)"
        if a.holdout != "none" and (ep + 1) % 4 == 0:
            (p, _), (p0, _), (pv, _) = evaluate(model, data[a.holdout])
            msg += f"  | held-out player error {p:.2f} m (nobody moves {p0:.2f}, keep velocity {pv:.2f})"
        print(msg, flush=True)

    print("\nMean error in metres between predicted and actual movement to the next Key State (lower is better)")
    rows = []
    for m in ([a.holdout] if a.holdout != "none" else []) + SIMS:
        (p, pb), (p0, pb0), (pv, pbv) = evaluate(model, data[m])
        rows.append((m, "held-out real" if m in REALS else "simulated", p, p0, pv, pb, pb0, pbv))
    print(f"{'match':10s} {'kind':14s} | players: model  still  keep-vel | ball: model  still  keep-vel")
    for r in rows:
        print(f"{r[0]:10s} {r[1]:14s} |          {r[2]:5.2f}  {r[3]:5.2f}  {r[4]:5.2f}   |       {r[5]:5.2f}  {r[6]:5.2f}  {r[7]:5.2f}")

    if a.save_embeddings:
        model.eval()
        for m in REALS + SIMS:
            d = data[m]
            K = len(d["feat"])
            emb = np.full((K, a.dim), np.nan, np.float32)
            ks = np.arange(L - 1, K)
            with torch.no_grad():
                for i in range(0, len(ks), 256):
                    kk = ks[i:i + 256]
                    b = batch(d, np.minimum(kk, K - 2))
                    b2 = dict(b)
                    _, z = model(b2)
                    pres = torch.from_numpy(d["present"][np.minimum(kk, K - 2)].astype(np.float32))[..., None]
                    emb[kk] = ((z * pres).sum(1) / pres.sum(1)).numpy()
            np.savez_compressed(PATH[m] + "_embeddings.npz", emb=emb)
        print("saved *_embeddings.npz for all matches")
        torch.save(model.state_dict(), "outputs/gnn_model.pt")


if __name__ == "__main__":
    main()
