"""Download the open datasets for the bigger player-model run (player_models_big.py).

  statsbomb   StatsBomb open data, the three men's tournaments with 360 freeze frames: Euro 2020, World Cup 2022,
              Euro 2024 (events + three-sixty per match). Licence: StatsBomb public data user agreement (credit StatsBomb).
              -> data/external/statsbomb_360_tournaments/{matches_<comp>_<season>.json, events/, three-sixty/}
  wyscout     Wyscout public dataset (Pappalardo et al. 2019, Scientific Data; CC BY 4.0): events of the 5 big leagues
              2017/18, World Cup 2018 and Euro 2016, plus players, teams, matches.
              -> data/external/wyscout_public/{events/, matches/, players.json, teams.json, competitions.json}

Downloads are kept as plain data (JSON only, nothing is executed). Files already on disk are skipped, so a stopped run
resumes.
Usage: .venv-analysis\\Scripts\\python.exe scripts\\fetch_open_data.py statsbomb | wyscout
"""
import json
import os
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SB_RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
SB_OUT = os.path.join(ROOT, "data", "external", "statsbomb_360_tournaments")
SB_COMPS = [(55, 43, "Euro 2020"), (43, 106, "World Cup 2022"), (55, 282, "Euro 2024")]
WY_OUT = os.path.join(ROOT, "data", "external", "wyscout_public")
# figshare files of the Pappalardo et al. collection (events, matches, players, teams, competitions)
WY_FILES = {"events.zip": "https://ndownloader.figshare.com/files/14464685",
            "matches.zip": "https://ndownloader.figshare.com/files/14464622",
            "players.json": "https://ndownloader.figshare.com/files/15073721",
            "teams.json": "https://ndownloader.figshare.com/files/15073697",
            "competitions.json": "https://ndownloader.figshare.com/files/15073685"}
UA = {"User-Agent": "research-download/1.0"}


def get(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return os.path.getsize(path)
    tmp = path + ".part"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, path)
    return os.path.getsize(path)


def statsbomb():
    for d in ("events", "three-sixty"):
        os.makedirs(os.path.join(SB_OUT, d), exist_ok=True)
    jobs = []
    for c, s, name in SB_COMPS:
        mp = os.path.join(SB_OUT, f"matches_{c}_{s}.json")
        get(f"{SB_RAW}/matches/{c}/{s}.json", mp)
        ms = json.load(open(mp, encoding="utf-8"))
        print(f"{name}: {len(ms)} matches", flush=True)
        for m in ms:
            mid = m["match_id"]
            for d in ("events", "three-sixty"):
                jobs.append((f"{SB_RAW}/{d}/{mid}.json", os.path.join(SB_OUT, d, f"{mid}.json")))
    done = [0]

    def one(j):
        try:
            n = get(*j)
        except Exception as e:                                             # a match without 360 is reported, not fatal
            print("  failed", j[0], e, flush=True)
            return 0
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"  {done[0]}/{len(jobs)} files", flush=True)
        return n

    with ThreadPoolExecutor(8) as ex:
        tot = sum(ex.map(one, jobs))
    print(f"statsbomb: {len(jobs)} files, {tot / 1e9:.2f} GB in {SB_OUT}")


def wyscout():
    os.makedirs(WY_OUT, exist_ok=True)
    for name, url in WY_FILES.items():
        n = get(url, os.path.join(WY_OUT, name))
        print(f"  {name}: {n / 1e6:.0f} MB", flush=True)
    for z in ("events", "matches"):
        out = os.path.join(WY_OUT, z)
        if not os.path.isdir(out):
            with zipfile.ZipFile(os.path.join(WY_OUT, z + ".zip")) as f:
                for info in f.infolist():                                  # JSON files only, flat, no paths from the zip
                    if info.filename.endswith(".json"):
                        os.makedirs(out, exist_ok=True)
                        with f.open(info) as src, open(os.path.join(out, os.path.basename(info.filename)), "wb") as dst:
                            dst.write(src.read())
        print(f"  {z}: {sorted(os.listdir(out))}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    {"statsbomb": statsbomb, "wyscout": wyscout}[sys.argv[1]]()
