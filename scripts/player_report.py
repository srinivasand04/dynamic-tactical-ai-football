"""own_PlayerReport: one report for ANY player in our data, the backtest numbers behind it, and the target player
(Granit Xhaka) upgraded with the possession value model.

    reports    every player with enough minutes (Wyscout leagues 900+, tournaments / StatsBomb 270+): VAEP per 90
               (attack / defence), percentile in his role, where his value comes from (6 x 4 zones), his most valuable
               action types, his on-ball style with 90% bands (player_bayes), pressure response, most similar players,
               and whether a player model beats his role on matches it never saw
               -> viewer/data/reports.json
    backtest   all held-out accuracy numbers in one file (value model, Bayes player model, ball loss, movement, learning
               curves, rating reliability) -> viewer/data/backtest.json
    xhaka      M2 threat, M3 lanes and M7 what-if redone with VAEP instead of the static xT grid
               -> adds "vaep" to viewer/data/target_player.json, and viewer/data/showcase_xhaka.json (his actions with values)
Usage : python scripts/player_report.py reports | backtest | xhaka | all       (run from the project root)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import player_models as P  # noqa: E402
import player_models_big as BG  # noqa: E402
import target_player as TP  # noqa: E402

ROOT = BG.ROOT
VM = os.path.join(ROOT, "outputs", "value_model")
PB = os.path.join(ROOT, "outputs", "player_bayes")
VD = os.path.join(ROOT, "viewer", "data")
L, W = 105.0, 68.0
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TOURS = {"World Cup 2018", "Euro 2016", "Euro 2020", "World Cup 2022", "Euro 2024"}


def r3(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4)


# ------------------------------------------------------------------ reports for any player

def step_reports():
    styles = json.load(open(os.path.join(PB, "styles.json"), encoding="utf-8"))["players"]
    sim = pd.read_csv(os.path.join(BG.OUT, "wy_similar.csv"))
    sim_of = {(r.iloc[0], r.iloc[1]): [str(v) for v in r.iloc[4:9] if isinstance(v, str)] for _, r in sim.iterrows()}
    ev = {}
    for f, tag in (("wy_eval.csv", "wy"), ("sb_eval.csv", "sb")):
        e = pd.read_csv(os.path.join(BG.OUT, f))
        for r in e.itertuples():
            ev[f"{tag}:{r.player_id}"] = {"gain_pct": r3(r.gain_pct), "verdict": r.verdict, "matches": int(r.matches)}
    out, index = {}, []
    for tag in ("wy", "sb"):
        R = pd.read_csv(os.path.join(VM, f"{tag}_ratings.csv"))
        A = pd.read_csv(os.path.join(VM, f"{tag}_values.csv.gz"), usecols=["source", "player_id", "kind", "x", "y", "vaep"], low_memory=False)
        A["z"] = TP.zone(A["x"], A["y"])
        keep = R[((R["minutes"] >= 900) | (R["source"].isin(TOURS) & (R["minutes"] >= 270)) | ((tag == "sb") & (R["minutes"] >= 270)))]
        keep = keep[keep["role"].astype(str) != ""]
        ks = set(zip(keep["source"], keep["player_id"]))
        A = A[[k in ks for k in zip(A["source"], A["player_id"])]]
        zs = A.groupby(["source", "player_id", "z"])["vaep"].sum()
        kd = A.groupby(["source", "player_id", "kind"])["vaep"].sum()
        for r in keep.itertuples():
            k90 = 90 / r.minutes
            z = np.zeros(24)
            if (r.source, r.player_id) in zs.index:
                s = zs.loc[(r.source, r.player_id)]
                z[s.index.to_numpy(int)] = s.to_numpy() * k90
            kk = kd.loc[(r.source, r.player_id)].sort_values(ascending=False) * k90
            key = f"{tag}:{r.player_id}:{r.source}"
            st = styles.get(f"{tag}:{r.player_id}")
            out[key] = {"name": r.player, "team": r.team, "comp": r.source, "data": "Wyscout" if tag == "wy" else "StatsBomb 360",
                        "role": r.role, "minutes": int(r.minutes), "vaep90": r3(r.vaep90), "off90": r3(r.v_off90), "def90": r3(r.v_def90),
                        "loc90": r3(r.vaep_loc90), "pct": r3(r.pct_role), "pct_n": None if pd.isna(r.pct_n) else int(r.pct_n), "pct_group": ("all StatsBomb competitions" if tag == "sb" else r.source), "zones": [round(float(v), 4) for v in z],
                        "kinds": [[k, round(float(v), 4)] for k, v in kk.head(4).items()], "style": st,
                        "similar": sim_of.get((r.player, r.team), []), "model": ev.get(f"{tag}:{r.player_id}")}
            index.append([key, r.player, r.team, r.source, r.role, int(r.minutes)])
        print(f"{tag}: {len(keep)} player reports", flush=True)
    index.sort(key=lambda v: v[1])
    json.dump({"classes": P.CLASSES, "index": index, "players": out}, open(os.path.join(VD, "reports.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"viewer/data/reports.json: {len(index)} reports, {os.path.getsize(os.path.join(VD, 'reports.json')) // 1024} KB")


# ------------------------------------------------------------------ backtest numbers in one place

def step_backtest():
    V = json.load(open(os.path.join(VM, "eval.json"), encoding="utf-8"))
    B = json.load(open(os.path.join(PB, "eval.json"), encoding="utf-8"))
    T = json.load(open(os.path.join(TP.OUT, "target.json"), encoding="utf-8"))
    big = json.load(open(os.path.join(VD, "player_models_big.json"), encoding="utf-8"))
    out = {"value": {t: {k: {"scores": {kk: V[t][k]["scores"][kk] for kk in ("auc", "brier_skill", "logloss", "rate", "calibration")},
                             "concedes": {kk: V[t][k]["concedes"][kk] for kk in ("auc", "brier_skill", "rate")}} for k in ("location", "full")}
                     for t in ("sb", "wy") if t in V},
           "ratings": V.get("ratings"), "bayes": B, "ball_loss": {k: T["m5"][k] for k in ("gain_pct", "auc", "calib", "n")},
           "movement": big.get("gnn"), "decision_eval": big["eval"], "curve_wyscout": big["curve_wyscout"]}
    json.dump(out, open(os.path.join(VD, "backtest.json"), "w", encoding="utf-8"), default=float)
    print("viewer/data/backtest.json written")
    for t in ("sb", "wy"):
        if t in V:
            print(f"  {t}: scoring AUC location {V[t]['location']['scores']['auc']:.3f} -> full {V[t]['full']['scores']['auc']:.3f}")


# ------------------------------------------------------------------ Xhaka with VAEP

def step_xhaka():
    S = pd.read_csv(os.path.join(VM, "sb_values.csv.gz"), low_memory=False)
    Wv = pd.read_csv(os.path.join(VM, "wy_values.csv.gz"), low_memory=False)
    sbm, wym = TP.minutes()
    out = {"contexts": []}
    # VAEP per 90 per context and percentile among DM / CM
    for s in TP.CONTEXTS:
        if s in dict(TP.WY_KEYS).values():
            h, m = Wv[(Wv.player_id == TP.T_WY) & (Wv.source == s)], wym[(wym.player_id == TP.T_WY) & (wym.source == s)]["minutes"].sum()
        else:
            h, m = S[(S.player_id == TP.T_SB) & (S.source == s)], sbm[(sbm.player_id == TP.T_SB) & (sbm.source == s)]["minutes"].sum()
        if len(h) and m:
            out["contexts"].append({"context": TP.CTX_LABEL[s], "vaep90": 90 * h.vaep.sum() / m, "off90": 90 * h.v_off.sum() / m,
                                    "def90": 90 * h.v_def.sum() / m, "minutes": float(m)})
    Rs = pd.read_csv(os.path.join(VM, "sb_ratings.csv"))
    agg = Rs.groupby("player_id").agg(v=("vaep", "sum"), m=("minutes", "sum"), role=("role", lambda r: r.value_counts().index[0]), name=("player", "first"))
    pop = agg[agg.role.isin(TP.MIDS) & (agg.m >= 360)]
    v90 = 90 * pop.v / pop.m
    out["sb_vaep90"], out["sb_pct"], out["sb_n"] = float(v90[TP.T_SB]), float(100 * (v90 < v90[TP.T_SB]).mean()), int(len(pop))
    out["sb_top"] = [[pop.name[p], float(v)] for p, v in v90.sort_values(ascending=False).head(8).items()]
    Rw = pd.read_csv(os.path.join(VM, "wy_ratings.csv"))
    pl = Rw[(Rw.source == "Premier League") & Rw.role.isin(TP.MIDS) & (Rw.minutes >= 900)]
    out["pl_vaep90"] = float(pl.loc[pl.player_id == TP.T_WY, "vaep90"].iloc[0])
    out["pl_pct"], out["pl_n"] = float(100 * (pl.vaep90 < out["pl_vaep90"]).mean()), int(len(pl))
    # zones: VAEP per 90 by start zone vs DM / CM
    hm = sbm[sbm.player_id == TP.T_SB]["minutes"].sum()
    h = S[S.player_id == TP.T_SB]
    ids = pop.index
    pp = S[S.player_id.isin(ids)]
    hz = np.bincount(TP.zone(h.x, h.y), weights=h.vaep, minlength=24) * 90 / hm
    pz = np.bincount(TP.zone(pp.x, pp.y), weights=pp.vaep, minlength=24) * 90 / pop.m.sum()
    out["zones_his"], out["zones_pop"] = hz.tolist(), pz.tolist()
    out["top_zones"] = [{"zone": TP.zone_name(z), "his": float(hz[z]), "typical": float(pz[z])} for z in np.argsort(-hz)[:4]]
    # lanes with VAEP
    mv = h[h.kind.isin(["Pass", "Carry", "Cross"]) & (h.ok == 1)].copy()
    mv["z0"], mv["z1"] = TP.zone(mv.x, mv.y), TP.zone(mv.ex, mv.ey)
    G = mv.groupby(["z0", "z1"]).agg(n=("vaep", "size"), v=("vaep", "sum")).reset_index().sort_values("v", ascending=False)
    pos = mv.vaep.clip(lower=0).sum()
    out["lanes"] = [{"from": TP.zone_name(r.z0), "to": TP.zone_name(r.z1), "z0": int(r.z0), "z1": int(r.z1), "per90": r.n * 90 / hm,
                     "v90": r.v * 90 / hm, "share": r.v / pos} for r in G.head(8).itertuples()]
    # M7 with VAEP: value of each choice when it comes off / fails, per third (his, shrunk to DM / CM)
    A = TP.sb_actions()
    dec = TP.decisions_of(A).merge(S[["id", "vaep"]], on="id", how="left")
    dec["third"] = np.digitize(dec.x, [35, 70])
    role = TP.main_role(A[A.role != "GK"])
    hd = dec[dec.player_id == TP.T_SB].assign(role="DM").reset_index(drop=True)
    pdc = dec[dec.player_id.map(role).isin(TP.MIDS)]
    Vok, Vf = np.zeros((3, 6)), np.zeros((3, 6))
    for t in range(3):
        for c in range(6):
            for okv, M in ((1, Vok), (0, Vf)):
                a = hd[(hd.third == t) & (hd.choice == c) & (hd.ok == okv)]["vaep"].dropna()
                b = pdc[(pdc.third == t) & (pdc.choice == c) & (pdc.ok == okv)]["vaep"].dropna()
                M[t, c] = (a.sum() + 30 * (b.mean() if len(b) else 0)) / (len(a) + 30)
    D, X, Z, base = TP.sb_base()
    Bp = np.load(os.path.join(TP.OUT, "m1_player_offset.npy"))
    B0, Bl = np.load(os.path.join(TP.OUT, "m5_pop.npy")), np.load(os.path.join(TP.OUT, "m5_his.npy"))
    plans = []
    th = hd["third"].to_numpy()
    for name, sc in TP.SCEN:
        q = TP.apply(hd, sc)
        Xq, Zq = P.onball_X(q)
        pc = P.softmax(base.logits(q, Xq, Zq, np.arange(len(q))) + Zq @ Bp)
        Fq, Zl = TP.loss_X(q.assign(side=q["side"].fillna("none")))
        risk = P.softmax(Fq @ B0 + Zl @ Bl)[:, 1]
        val = (pc * ((1 - risk)[:, None] * Vok[th] + risk[:, None] * Vf[th])).sum(1)
        plans.append({"plan": name, "risk": float(risk.mean()), "vaep100": float(100 * val.mean())})
    free = next(p for p in plans if p["plan"].startswith("let him play"))["vaep100"]
    for p in plans:
        p["vs_free"] = 100 * (p["vaep100"] - free) / abs(free)
    best = min((p for p in plans if p["plan"] != "as played"), key=lambda p: p["vaep100"])
    out["plans"], out["best"], out["V_ok"], out["V_fail"] = plans, best["plan"], Vok.tolist(), Vf.tolist()
    print(f"Xhaka VAEP/90 {out['sb_vaep90']:.3f} (percentile {out['sb_pct']:.0f} of {out['sb_n']} DM/CM, StatsBomb); Arsenal "
          f"{out['pl_vaep90']:.3f} (percentile {out['pl_pct']:.0f} of {out['pl_n']})")
    for c in out["contexts"]:
        print(f"  {c['context']:18s} VAEP/90 {c['vaep90']:.3f} (attack {c['off90']:.3f}, defence {c['def90']:.3f})")
    print("  top zones: " + "; ".join(f"{z['zone']} {z['his']:.3f} (typ {z['typical']:.3f})" for z in out["top_zones"]))
    print("  top lanes: " + "; ".join(f"{l['from']} -> {l['to']} {l['v90']:.3f}/90" for l in out["lanes"][:4]))
    for p in plans:
        print(f"  plan {p['plan']:36s} his value per 100 actions {p['vaep100']:+.3f}  ({p['vs_free']:+.0f}% vs letting him play), risk {100 * p['risk']:.1f}%")
    print(f"  -> best plan with VAEP: {best['plan']}")
    T = json.load(open(TP.VIEW, encoding="utf-8"))
    T["vaep"] = out
    json.dump(T, open(TP.VIEW, "w", encoding="utf-8"), ensure_ascii=False, default=float)
    # showcase: his Leverkusen actions with values (largest |value| first, plus a random sample)
    hl = h[h.source == TP.LEV]
    big = hl.reindex(hl.vaep.abs().sort_values(ascending=False).index).head(500)
    rnd = hl.drop(big.index).sample(min(700, len(hl) - len(big)), random_state=1)
    sh = pd.concat([big, rnd])
    json.dump({"name": TP.NAME, "actions": [[round(a.x, 1), round(a.y, 1), round(a.ex, 1), round(a.ey, 1), round(a.vaep, 4), a.kind, int(a.ok)]
                                            for a in sh.itertuples()]},
              open(os.path.join(VD, "showcase_xhaka.json"), "w", encoding="utf-8"), separators=(",", ":"))
    print("viewer/data/showcase_xhaka.json written")


STEPS = {"reports": step_reports, "backtest": step_backtest, "xhaka": step_xhaka}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    a = ap.parse_args()
    for s in (list(STEPS) if a.step == "all" else [a.step]):
        STEPS[s]()
