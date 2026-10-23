"""Read settings.json: the one place for every number and every match list.

    import settings
    settings.REAL            -> ['J03WMX', ...]   all real matches, from every dataset with kind 'real'
    settings.SIM             -> ['match_001', ...]
    settings.base(m)         -> 'data/real/processed/J03WMX'        (add '_adaptive_ks.csv', '_graphs.npz', ... to it)
    settings.csv(m)          -> 'data/real/processed/J03WMX.csv.gz' (the common-format file of that match)
    settings.get('gnn', 'dim')
    settings.dataset(m)      -> 'dfl'

A new dataset (for example another tracking provider) is added to settings.json under "datasets" with
kind "real" or "simulated"; its matches then join every step that uses settings.REAL / settings.SIM.
"""
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, "settings.json"), encoding="utf-8") as f:
    S = json.load(f)

_DATASETS = S["datasets"]
REAL = [m for d in _DATASETS.values() if d["kind"] == "real" for m in d["matches"]]
SIM = [m for d in _DATASETS.values() if d["kind"] == "simulated" for m in d["matches"]]
_OWNER = {m: name for name, d in _DATASETS.items() for m in d["matches"]}


def dataset(m):
    return _OWNER[m]


def variant(m):
    """Name of the group a match belongs to: 'real' for real matches, else the simulator setting (stochastic, easy, hard, ...)."""
    d = _DATASETS[_OWNER[m]]
    return d.get("variant", "real" if d["kind"] == "real" else _OWNER[m])


def base(m):
    return f"{_DATASETS[_OWNER[m]]['dir']}/{m}"


def csv(m):
    d = _DATASETS[_OWNER[m]]
    return f"{d['dir']}/{m}{d['file_ext']}"


def get(*keys, default=None):
    cur = S
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def exists(m):
    """True when the common-format file of the match is on disk (so a not-yet-downloaded match is skipped)."""
    return os.path.exists(csv(m))


def available(matches):
    return [m for m in matches if exists(m)]


def palette(n_real, n_sim):
    """Colours for charts: greens for real matches, oranges for simulated ones."""
    import numpy as np
    import matplotlib.pyplot as plt
    g = plt.cm.Greens(np.linspace(0.95, 0.45, max(n_real, 1)))[:n_real]
    o = plt.cm.Oranges(np.linspace(0.95, 0.45, max(n_sim, 1)))[:n_sim]
    return [tuple(c) for c in g] + [tuple(c) for c in o]
