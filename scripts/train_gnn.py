"""Stage C: a spatio-temporal graph network over Key States (version 2).

Task (self-supervised, no tactical labels needed): given the last L Key-State graphs and the time to the next
Key State, predict where every player and the ball will have moved by the next Key State.

Model
  per Key State : a graph encoder; edge-type aware message passing over the 3 edge types (team / opponent / ball)
  over time     : a GRU that reads the L graph encodings of each node in order, with the time gaps as input
  output        : the movement (dx, dy) of every node, and a graph embedding (mean of the final node states)

What is new in version 2
  faster   : messages are computed only along edges that exist (edge lists), early stopping on a validation slice
  better   : the ball has a velocity; training pictures are randomly mirrored (left-right, top-bottom) so the model
             learns football geometry, not which side a team happens to attack
  repeatable: all numbers and match lists come from settings.json, every run is appended to outputs/run_log.csv

Optional fourth edge type (--arch pass_score / pass_raw): the PASS edges of the ball-owner graph, from build_pass_edges.py. They
link the ball owner and each teammate in both directions and carry numbers (what the pass models say about that option), so the
network can use "who is the likely receiver, is the lane open" when it predicts where the ball and the players go.

Baselines to beat: "nobody moves" and "everybody keeps their velocity for the time gap" (the ball too).

Training uses real matches only. The last val_fraction of every training match is kept back for early stopping.
Evaluation: the held-out real match (never seen in training) and every simulated match.

Input : <name>_graphs.npz from build_graphs.py
Output: printed errors; with --save-embeddings, <name>_embeddings[_<tag>].npz (one vector per Key State)

Usage :  python scripts\\train_gnn.py --holdout J03WOH --tag test
         python scripts\\train_gnn.py --holdout none --save-embeddings
"""
import argparse
import copy
import os
import time
import zlib

import numpy as np
import torch
import torch.nn as nn

import runlog
import settings

REALS = [m for m in settings.available(settings.REAL) if os.path.exists(settings.graphs(m))]
SIMS = [m for m in settings.available(settings.SIM) if os.path.exists(settings.graphs(m))]
PATH = {m: settings.base(m) for m in REALS + SIMS}
N = 23
L = settings.get("gnn", "window", default=3)
D_MOVE = 5.0          # metres: scale of the movement we predict
SUFFIX = settings.graph_suffix()   # "" = radius graphs (_graphs.npz)   "_delaunay" = Delaunay graphs; set by graph.mode in settings.json
PASS_KEY = None       # None = no pass edges;  "pscore" / "praw" = which numbers ride on the pass edges (build_pass_edges.py)
PASS_ARCH = {"pass_score": "pscore", "pass_raw": "praw", "pass_shuffle": "praw"}
PASS_SHUFFLE = False  # pass_shuffle: the same pass edges, but the numbers on them are shuffled between teammates and Key States of a match


def load(match):
    g = np.load(PATH[match] + f"_graphs{SUFFIX}.npz")
    xy, present = g["xy"], g["present"]
    K = len(xy)
    team, role = g["team"], g["role"]
    pos = np.nan_to_num((xy - np.array([52.5, 34.0])) / np.array([52.5, 34.0]))
    vel = g["vel"].copy()
    vel[:, :22] = np.clip(vel[:, :22], -8, 8) / 8.0                        # players: up to 8 m/s
    vel[:, 22] = np.clip(vel[:, 22], -30, 30) / 30.0                       # the ball: up to 30 m/s
    db = np.nan_to_num(g["dist_ball"]) / 50.0
    team1 = np.eye(3)[team][None].repeat(K, 0)
    role1 = np.eye(11)[np.clip(role + 1, 0, 10)]
    feat = np.concatenate([pos, vel, db[..., None], team1, role1, present[..., None]], axis=2)
    feat = feat * present[..., None]                                       # empty slots are all zeros
    adj = np.stack([g["adj_team"], g["adj_opp"], g["adj_ball"]], axis=1)   # (K, 3, N, N) yes/no
    dist = np.nan_to_num(np.linalg.norm(xy[:, :, None] - xy[:, None], axis=3), nan=60.0)
    dist = np.clip(dist, 0, 30) / 30.0
    disp = np.zeros_like(xy)
    disp[:-1] = np.nan_to_num(xy[1:] - xy[:-1])                            # movement from this Key State to the next
    disp_ok = np.zeros((K, N), bool)
    disp_ok[:-1] = present[1:] & present[:-1]
    gap_next = np.zeros(K)
    gap_next[:-1] = g["time_s"][1:] - g["time_s"][:-1]
    vel_ms = g["vel"].copy()                                               # real units for the keep-velocity baseline
    idx = np.arange(L - 1, K - 1)                                          # windows that end at k and have a next Key State
    extra = {}
    if PASS_KEY:
        pe = np.load(PATH[match] + "_passedges.npz")
        pfeat = pe[PASS_KEY].astype(np.float32)
        if PASS_SHUFFLE:                                                   # control: same edges and same numbers, but the numbers no longer belong to the teammate
            valid = np.argwhere(pe["pmask"])
            perm = np.random.default_rng(zlib.crc32(match.encode())).permutation(len(valid))
            pfeat = pfeat.copy()
            pfeat[valid[:, 0], valid[:, 1]] = pfeat[valid[perm, 0], valid[perm, 1]]
        extra = dict(pbo=pe["pbo"], pmate=pe["pmate"], pmask=pe["pmask"], pfeat=pfeat)
    return dict(**extra, feat=feat.astype(np.float32), adj=adj, dist=dist.astype(np.float32), disp=(disp / D_MOVE).astype(np.float32),
                disp_ok=disp_ok, gap_next=gap_next.astype(np.float32), gap_prev=g["gap_s"].astype(np.float32),
                vel_ms=vel_ms.astype(np.float32), idx=idx, present=present)


def batch(d, ks):
    """Stack windows ending at each k in ks: tensors shaped (B, L, ...)."""
    win = ks[:, None] + np.arange(-L + 1, 1)[None]                         # (B, L)
    t = torch.from_numpy
    gaps = np.stack([d["gap_prev"][win], np.repeat(d["gap_next"][ks][:, None], L, 1)], axis=2)
    out = dict(feat=t(d["feat"][win]), adj=t(d["adj"][win]), dist=t(d["dist"][win]), gaps=t(np.log1p(gaps).astype(np.float32)),
               target=t(d["disp"][ks]), ok=t(d["disp_ok"][ks]), gap_next=t(d["gap_next"][ks]), vel=t(d["vel_ms"][ks]))
    if "pbo" in d:
        out.update(pbo=t(d["pbo"][win]), pmate=t(d["pmate"][win]), pmask=t(d["pmask"][win]), pfeat=t(d["pfeat"][win]))
    return out


def mirror(b):
    """Randomly mirror each picture left-right and top-bottom (the pitch is symmetric); targets move with it."""
    B = b["feat"].shape[0]
    sx = (torch.randint(0, 2, (B,)) * 2 - 1).float()
    sy = (torch.randint(0, 2, (B,)) * 2 - 1).float()
    b["feat"][..., 0] *= sx[:, None, None]
    b["feat"][..., 2] *= sx[:, None, None]
    b["feat"][..., 1] *= sy[:, None, None]
    b["feat"][..., 3] *= sy[:, None, None]
    b["target"][..., 0] *= sx[:, None]
    b["target"][..., 1] *= sy[:, None]
    b["vel"][..., 0] *= sx[:, None]
    b["vel"][..., 1] *= sy[:, None]
    return b


class Layer(nn.Module):
    """One round of message passing. Messages are computed only along the edges that exist."""

    def __init__(self, d):
        super().__init__()
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.Linear(d, d)) for _ in range(3)])
        self.upd = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, edges, dist):                                     # h (BT, N, d); edges: 3 x (graph, from, to) index lists
        outs = [h]
        for e in range(3):
            bi, i, j = edges[e]
            m = self.msg[e](torch.cat([h[bi, i], h[bi, j], dist[bi, i, j][:, None]], dim=-1))
            agg = h.new_zeros(h.shape).index_put((bi, i), m, accumulate=True)
            cnt = h.new_zeros(h.shape[0], h.shape[1], 1).index_put((bi, i), h.new_ones(len(bi), 1), accumulate=True)
            outs.append(agg / (cnt + 1e-6))                                # average over the neighbours of each node
        return self.norm(h + self.upd(torch.cat(outs, dim=-1)))


class PassLayer(Layer):
    """The same round of message passing plus a fourth edge type, the pass edges. Their messages also read the numbers on the edge."""

    def __init__(self, d, f_pass):
        super().__init__(d)
        self.pmsg = nn.Sequential(nn.Linear(2 * d + f_pass, d), nn.ReLU(), nn.Linear(d, d))
        self.upd = nn.Sequential(nn.Linear(5 * d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, h, edges, dist, pedges):
        outs = [h]
        for e in range(3):
            bi, i, j = edges[e]
            m = self.msg[e](torch.cat([h[bi, i], h[bi, j], dist[bi, i, j][:, None]], dim=-1))
            agg = h.new_zeros(h.shape).index_put((bi, i), m, accumulate=True)
            cnt = h.new_zeros(h.shape[0], h.shape[1], 1).index_put((bi, i), h.new_ones(len(bi), 1), accumulate=True)
            outs.append(agg / (cnt + 1e-6))
        bi, i, j, f = pedges
        m = self.pmsg(torch.cat([h[bi, i], h[bi, j], f], dim=-1))
        agg = h.new_zeros(h.shape).index_put((bi, i), m, accumulate=True)
        cnt = h.new_zeros(h.shape[0], h.shape[1], 1).index_put((bi, i), h.new_ones(len(bi), 1), accumulate=True)
        outs.append(agg / (cnt + 1e-6))
        return self.norm(h + self.upd(torch.cat(outs, dim=-1)))


class GCNLayer(nn.Module):
    """Plain graph convolution: every node averages its neighbours (and itself), then one shared linear map.
    No edge types and no distances: the standard baseline."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, edges, dist):
        bi, i, j = edges
        w = self.lin(h)
        agg = h.new_zeros(h.shape).index_put((bi, i), w[bi, j], accumulate=True)
        cnt = h.new_zeros(h.shape[0], h.shape[1], 1).index_put((bi, i), h.new_ones(len(bi), 1), accumulate=True)
        return self.norm(h + torch.relu(agg / (cnt + 1e-6)))


class NoGraphLayer(nn.Module):
    """The control experiment: the same size as the GCN layer but each node only looks at ITSELF. If this does as well as the
    graph layers, the links between players add nothing for this task."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, edges, dist):
        return self.norm(h + torch.relu(self.lin(h)))


class GATLayer(nn.Module):
    """Graph attention: every node weighs its neighbours by a learned importance (2 heads), then adds them up.
    No edge types and no distances: the second standard baseline."""

    def __init__(self, d, heads=2):
        super().__init__()
        self.heads, self.dk = heads, d // heads
        self.lin = nn.Linear(d, d, bias=False)
        self.att_i = nn.Parameter(torch.randn(heads, self.dk) * 0.1)
        self.att_j = nn.Parameter(torch.randn(heads, self.dk) * 0.1)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, edges, dist):
        bi, i, j = edges
        BT, n, d = h.shape
        w = self.lin(h).view(BT, n, self.heads, self.dk)
        e = nn.functional.leaky_relu((w[bi, i] * self.att_i).sum(-1) + (w[bi, j] * self.att_j).sum(-1), 0.2)    # (edges, heads)
        grp = bi * n + i                                                    # the node that is listening
        peak = e.new_full((BT * n, self.heads), -1e9).scatter_reduce(0, grp[:, None].expand(-1, self.heads), e.detach(), reduce="amax", include_self=True)
        ex = torch.exp(e - peak[grp])
        den = e.new_zeros(BT * n, self.heads).index_add(0, grp, ex)
        alpha = ex / (den[grp] + 1e-9)                                      # attention weights, sum to 1 over each node's neighbours
        out = e.new_zeros(BT * n, self.heads, self.dk).index_add(0, grp, alpha[..., None] * w[bi, j])
        return self.norm(h + torch.relu(out.reshape(BT, n, d)))


class Net(nn.Module):
    def __init__(self, f_in, d=48, arch="ours", f_pass=0):
        super().__init__()
        self.arch = arch
        self.with_pass = arch in PASS_ARCH
        self.inp = nn.Sequential(nn.Linear(f_in, d), nn.ReLU())
        if self.with_pass:
            self.layers = nn.ModuleList([PassLayer(d, f_pass), PassLayer(d, f_pass)])
        else:
            kind = {"ours": Layer, "gcn": GCNLayer, "gat": GATLayer, "mlp": NoGraphLayer}[arch]
            self.layers = nn.ModuleList([kind(d), kind(d)])
        self.gru = nn.GRU(d + 2, d, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d + 2, d), nn.ReLU(), nn.Linear(d, 2))

    def encode(self, b):
        B, T, n, f = b["feat"].shape
        h = self.inp(b["feat"].reshape(B * T, n, f))
        adj, dist = b["adj"].reshape(B * T, 3, n, n), b["dist"].reshape(B * T, n, n)
        pedges = None
        if self.with_pass:                                                          # the pass edges: ball owner <-> every teammate
            mask = b["pmask"].reshape(B * T, -1)
            gi = torch.arange(B * T)[:, None].expand_as(mask)[mask]
            bo = b["pbo"].reshape(B * T)[:, None].expand_as(mask)[mask]
            mt = b["pmate"].reshape(B * T, -1)[mask]
            ft = b["pfeat"].reshape(B * T, mask.shape[1], -1)[mask]
            pedges = (torch.cat([gi, gi]), torch.cat([bo, mt]), torch.cat([mt, bo]), torch.cat([ft, ft]))
        if self.arch == "ours" or self.with_pass:
            edges = [torch.nonzero(adj[:, e], as_tuple=True) for e in range(3)]    # three edge types, found once, used by both layers
        else:                                                                       # plain GCN / GAT: one edge type = any link, plus a link to itself
            edges = torch.nonzero(adj.any(dim=1) | torch.eye(n, dtype=torch.bool)[None], as_tuple=True)
        for layer in self.layers:
            h = layer(h, edges, dist, pedges) if self.with_pass else layer(h, edges, dist)
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
    """Summed error in metres over nodes present at both Key States: players, ball."""
    err = torch.linalg.norm((pred - b["target"]) * D_MOVE, dim=-1)
    ok = b["ok"].float()
    ply, ball = ok.clone(), ok.clone()
    ply[:, 22], ball[:, :22] = 0, 0
    return np.array([(err * ply).sum().item(), ply.sum().item(), (err * ball).sum().item(), ball.sum().item()])


def evaluate(model, d, ks=None, bs=256):
    """Returns (model, nobody-moves, keep-velocity), each as (player error, ball error) in metres."""
    model.eval()
    ks = d["idx"] if ks is None else ks
    tot, base0, basev = np.zeros(4), np.zeros(4), np.zeros(4)
    with torch.no_grad():
        for i in range(0, len(ks), bs):
            b = batch(d, ks[i:i + bs])
            pred, _ = model(b)
            tot += errors(pred, b)
            base0 += errors(torch.zeros_like(pred), b)
            basev += errors(b["vel"] * b["gap_next"][:, None, None] / D_MOVE, b)
    f = lambda a: (a[0] / max(a[1], 1), a[2] / max(a[3], 1))
    return f(tot), f(base0), f(basev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="J03WOH", help="a real match id to hold out, or 'none'")
    ap.add_argument("--epochs", type=int, default=settings.get("gnn", "max_epochs", default=30), help="maximum; early stopping usually ends sooner")
    ap.add_argument("--dim", type=int, default=settings.get("gnn", "dim", default=48))
    ap.add_argument("--patience", type=int, default=settings.get("gnn", "patience", default=3))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mirror", action="store_true", help="switch off the random mirroring of training pictures")
    ap.add_argument("--save-embeddings", action="store_true")
    ap.add_argument("--tag", default="", help="name of this run; with --save-embeddings it is added to the file names")
    ap.add_argument("--graphs", choices=["radius", "delaunay"], default=settings.get("graph", "mode", default="radius"), help="which edge definition to use (default: graph.mode in settings.json)")
    ap.add_argument("--arch", choices=["ours", "gcn", "gat", "mlp", "pass_score", "pass_raw", "pass_shuffle"], default="ours", help="ours = edge-type aware messages; gcn = plain graph convolution; gat = graph attention; mlp = no graph at all (control); pass_score / pass_raw = ours + the pass edges of the ball-owner graph (learned scores / plain numbers on the edge); pass_shuffle = the same with the numbers shuffled (control for extra capacity)")
    a = ap.parse_args()
    global SUFFIX, PASS_KEY, PASS_SHUFFLE
    SUFFIX = "" if a.graphs == "radius" else "_delaunay"
    PASS_KEY = PASS_ARCH.get(a.arch)
    PASS_SHUFFLE = a.arch == "pass_shuffle"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t_start = time.time()

    data = {m: load(m) for m in REALS + SIMS}
    train = [m for m in REALS if m != a.holdout]
    vf = settings.get("gnn", "val_fraction", default=0.1)
    tr_idx, va_idx = {}, {}
    for m in train:                                                        # the last part of each training match is for early stopping
        idx = data[m]["idx"]
        cut = int(len(idx) * (1 - vf))
        tr_idx[m], va_idx[m] = idx[:cut], idx[cut:]
    print("train on:", train, "| held-out real:", a.holdout, "| simulated matches to evaluate:", len(SIMS))
    print("windows for training:", sum(len(v) for v in tr_idx.values()), "| for early stopping:", sum(len(v) for v in va_idx.values()))

    model = Net(data[REALS[0]]["feat"].shape[2], a.dim, a.arch, data[REALS[0]]["pfeat"].shape[-1] if PASS_KEY else 0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"architecture: {a.arch}  ({n_params:,} parameters)")
    opt = torch.optim.Adam(model.parameters(), lr=settings.get("gnn", "lr", default=2e-3))
    bs = settings.get("gnn", "batch", default=128)
    pool = [(m, k) for m in train for k in tr_idx[m]]
    best, best_state, best_ep, stale = 1e9, None, 0, 0
    min_ep = settings.get("gnn", "min_epochs", default=4)
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(len(pool))
        t0, tot, cnt = time.time(), 0.0, 0
        for i in range(0, len(perm), bs):
            chunk = [pool[j] for j in perm[i:i + bs]]
            parts = [batch(data[m], np.array([k for mm, k in chunk if mm == m])) for m in train if any(mm == m for mm, _ in chunk)]
            b = {key: torch.cat([p[key] for p in parts]) for key in parts[0]}
            if not a.no_mirror and settings.get("gnn", "augment_flips", default=True):
                b = mirror(b)
            pred, _ = model(b)
            ok = b["ok"].float()[..., None]
            loss = (nn.functional.smooth_l1_loss(pred, b["target"], reduction="none") * ok).sum() / (ok.sum() * 2 + 1e-6)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(chunk)
            cnt += len(chunk)
        # early stopping on the kept-back slice of the training matches
        v = np.mean([evaluate(model, data[m], va_idx[m])[0][0] for m in train])
        improved = v < best - 0.005
        if v < best:
            best, best_state, best_ep = v, copy.deepcopy(model.state_dict()), ep + 1
        stale = 0 if improved else stale + 1
        print(f"epoch {ep + 1:2d}  train loss {tot / cnt:.4f}  validation error {v:.3f} m  ({time.time() - t0:.0f}s)", flush=True)
        if ep + 1 >= min_ep and stale >= a.patience:
            print(f"early stop: no improvement for {a.patience} epochs; best was epoch {best_ep}")
            break
    model.load_state_dict(best_state)
    epochs_run = ep + 1

    print("\nMean error in metres between predicted and actual movement to the next Key State (lower is better)")
    rows = []
    for m in ([a.holdout] if a.holdout != "none" else []) + SIMS:
        (p, pb), (p0, pb0), (pv, pbv) = evaluate(model, data[m])
        rows.append((m, "held-out real" if m in REALS else "simulated", p, p0, pv, pb, pb0, pbv))
    print(f"{'match':10s} {'kind':14s} | players: model  still  keep-vel | ball: model  still  keep-vel")
    for r in rows:
        print(f"{r[0]:10s} {r[1]:14s} |          {r[2]:5.2f}  {r[3]:5.2f}  {r[4]:5.2f}   |       {r[5]:5.2f}  {r[6]:5.2f}  {r[7]:5.2f}")

    sims = [r for r in rows if r[1] == "simulated"]
    real = [r for r in rows if r[1] == "held-out real"]
    entry = {"script": "train_gnn", "tag": a.tag, "holdout": a.holdout, "graphs": a.graphs, "mirror": not a.no_mirror,
             "arch": a.arch, "params": n_params,
             "team_k": settings.get("graph", "team_k"), "dim": a.dim, "seed": a.seed, "epochs_run": epochs_run, "best_epoch": best_ep,
             "best_validation_m": round(float(best), 3), "n_train_matches": len(train), "n_sim_matches": len(SIMS),
             "minutes": round((time.time() - t_start) / 60, 1),
             "sim_player_model_m": round(float(np.mean([r[2] for r in sims])), 3), "sim_player_keepvel_m": round(float(np.mean([r[4] for r in sims])), 3),
             "sim_ball_model_m": round(float(np.mean([r[5] for r in sims])), 3), "sim_ball_keepvel_m": round(float(np.mean([r[7] for r in sims])), 3)}
    if real:
        entry.update({"real_player_model_m": round(real[0][2], 3), "real_player_keepvel_m": round(real[0][4], 3), "real_player_still_m": round(real[0][3], 3),
                      "real_ball_model_m": round(real[0][5], 3), "real_ball_keepvel_m": round(real[0][7], 3)})
    runlog.log(entry)
    print(f"\nrun finished in {entry['minutes']} min after {epochs_run} epochs; logged to outputs/run_log.csv")

    if a.save_embeddings:
        model.eval()
        tag = f"_{a.tag}" if a.tag else ""
        for m in REALS + SIMS:
            d = data[m]
            K = len(d["feat"])
            emb = np.full((K, a.dim), np.nan, np.float32)
            ks = np.arange(L - 1, K)
            with torch.no_grad():
                for i in range(0, len(ks), 256):
                    kk = ks[i:i + 256]
                    b = batch(d, np.minimum(kk, K - 2))
                    _, z = model(b)
                    pres = torch.from_numpy(d["present"][np.minimum(kk, K - 2)].astype(np.float32))[..., None]
                    emb[kk] = ((z * pres).sum(1) / pres.sum(1)).numpy()
            np.savez_compressed(PATH[m] + f"_embeddings{tag}.npz", emb=emb)
        print(f"saved *_embeddings{tag}.npz for all matches")
        torch.save(model.state_dict(), f"outputs/gnn_model{tag}.pt")


if __name__ == "__main__":
    main()
