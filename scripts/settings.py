"""Read settings.json: the one place for every number and every match list.

    import settings
    settings.REAL            -> ['J03WMX', ...]   all real matches, from every dataset with kind 'real'
    settings.SIM             -> ['match_001', ...]
    settings.base(m)         -> 'data/real/processed/J03WMX'        (add '_adaptive_ks.csv', '_graphs.npz', ... to it)
    settings.graphs(m)       -> '.../J03WMX_graphs_delaunay.npz'   (graph file of the mode set in settings.json)
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


def _matches(d):
    """A dataset lists its matches in settings.json, or points to a registry csv (column match_id) written by its converter."""
    if "matches" in d:
        return d["matches"]
    reg = os.path.join(_ROOT, d["registry"])
    if not os.path.exists(reg):
        return []
    import csv as _csv
    with open(reg, encoding="utf-8") as f:
        return [r["match_id"] for r in _csv.DictReader(f)]


_MATCHES = {name: _matches(d) for name, d in _DATASETS.items()}
# 'pool': false keeps a dataset out of the pooled steps (the realism comparison, the DFL folds, the proof runs) so adding
# a dataset never changes the numbers already reported. Its matches are used by dataset-wise steps (--dataset NAME).
REAL = [m for n, d in _DATASETS.items() if d["kind"] == "real" and d.get("pool", True) for m in _MATCHES[n]]
SIM = [m for n, d in _DATASETS.items() if d["kind"] == "simulated" and d.get("pool", True) for m in _MATCHES[n]]
_OWNER = {m: name for name, ms in _MATCHES.items() for m in ms}


def dataset_matches(name):
    """Matches of one dataset (pooled or not) that have a common-format file on disk."""
    return [m for m in _MATCHES[name] if exists(m)]


def registry(name):
    """pandas table of one dataset's matches (home, away, score, date, ...) when it has a registry csv, else None."""
    import pandas as pd
    reg = _DATASETS[name].get("registry")
    return pd.read_csv(os.path.join(_ROOT, reg), dtype={"match_id": str}) if reg and os.path.exists(os.path.join(_ROOT, reg)) else None


def dataset(m):
    return _OWNER[m]


def variant(m):
    """Name of the group a match belongs to: 'real' for real matches, else the simulator setting (stochastic, easy, hard, ...)."""
    d = _DATASETS[_OWNER[m]]
    return d.get("variant", "real" if d["kind"] == "real" else _OWNER[m])


def base(m):
    return f"{_DATASETS[_OWNER[m]]['dir']}/{m}"


def graph_suffix():
    """'' for radius graphs, '_delaunay' for Delaunay graphs: set once in settings.json under graph.mode."""
    return "" if S.get("graph", {}).get("mode", "radius") == "radius" else "_delaunay"


def graphs(m):
    """The graph file of match m for the graph mode chosen in settings.json."""
    return f"{base(m)}_graphs{graph_suffix()}.npz"


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
