"""Real player names in GRF's picture, instead of GRF's invented ones ('Curie', 'Meitner', ...).

GRF's engine has 22 fixed surnames compiled into _gameplayfootball.pyd (one per player slot and side) and no setting for them.
This module writes the real names into those 22 places in a COPY of the engine (the installed GRF stays untouched) and makes
Python load that copy. It must run before anything imports gfootball, so the GRF scripts call setup_from_args() first.

Names: a real stretch (--segment): the players of the DFL match, in the segment's slot order (keeper, then by role), the same
order GRF's scenario adds them. A team model (Argentina, France, Croatia): the starting eleven of that team's latest World Cup
match (PFF rosters), placed on GRF's default line-up by position. 'pooled' and 'GRF AI' keep GRF's names.
Each name is the surname. GRF shows only as many letters as its own name in that slot had (4 to 12), so long surnames are cut.
Team names: GRF's 'Frequentists United' / 'Real Bayesians' (FRQ / RBA on the score board) become the real clubs (from
data/real/matches.csv) or the national teams; 19 and 14 letters at most ('SSV Jahn Regensburg' -> 'Regensburg').

Cache: .grf_engine/<names hash>/gfootball_engine (engine file + DLLs, ~10 MB; GRF's data folder is used from the install).
"""
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, ".grf_engine")
PFF_RAW = os.path.join(ROOT, "data", "external", "pff_worldcup", "raw", "FIFA World Cup 2022")
# GRF's names in engine slot order (src/data/teamdata.cpp builds the players in this order; left team, right team)
GRF_LEFT = ["Lovelace", "Turing", "Johnson", "da Vinci", "Newton", "Blackwell", "Borg", "Euler", "Pythagoras", "Curie", "Nixon Sutton"]
GRF_RIGHT = ["Meitner", "Einstein", "Vaughaun", "Archimedinho", "Banach", "Banneker", "Goodall", "Copernicus", "Feynman", "Franklin", "Roy Mouton"]
# GRF's default eleven (scenarios/11_vs_11_stochastic.py) and which PFF position groups fit each place, best first
GRF_LINEUP = ["GK", "RM", "CF", "LB", "CB", "CB", "RB", "CM", "CM", "CM", "LM"]
FITS = {"GK": ["GK"], "CB": ["RCB", "LCB", "MCB", "CB"], "LB": ["LB", "LWB"], "RB": ["RB", "RWB"], "CM": ["CM", "DM", "AM"],
        "LM": ["LW", "LM", "AM", "CF"], "RM": ["RW", "RM", "AM", "CF"], "CF": ["CF", "ST", "AM", "LW", "RW"]}
LATEST = {"Argentina": "10517", "France": "10515", "Croatia": "10516"}     # the team model's held-out match (team_policy.py)
# GRF's team names (long name, 3-letter code) for the left and the right team, compiled into the engine like the players
GRF_TEAMS = [("Frequentists United", "FRQ"), ("Real Bayesians", "RBA")]
MATCHES = os.path.join(ROOT, "data", "real", "matches.csv")


def ascii_name(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().strip()


def surname(full):
    """'F. Kastenmeier' -> 'Kastenmeier', 'Jordy de Wijs' -> 'de Wijs', 'Lionel Messi' -> 'Messi', 'Bono' -> 'Bono'."""
    s = ascii_name(full)
    if ". " in s:
        return s.split(". ", 1)[1]
    parts = s.split(" ", 1)
    return parts[1] if len(parts) > 1 else s


def label(name, cap):
    """The surname, cut to the `cap` letters GRF shows in that slot."""
    return surname(name)[:cap]


def engine_dir():
    spec = importlib.util.find_spec("gfootball_engine")
    return os.path.dirname(spec.origin)


def slot_rooms(blob):
    """For every GRF name: where it is in the engine file and how many letters fit there (up to the next text)."""
    out = {}
    for n in GRF_LEFT + GRF_RIGHT:
        key = n.encode() + b"\0"
        i = blob.find(key)
        assert i >= 0 and blob.count(key) == 1, f"GRF name {n} not found once in the engine"
        j = i + len(n)
        while j < len(blob) and blob[j] == 0:
            j += 1
        out[n] = (i, j - i - 1)                              # keep at least one closing zero
    return out


def team_label(name, cap):
    """A club name that fits `cap` letters: whole if it fits, else without its first words ('SSV Jahn Regensburg' ->
    'Regensburg'), else cut. Plus its 3-letter code from the last word ('DUS', 'REG', 'ARG')."""
    words = ascii_name(name).split()
    long = next((" ".join(words[k:]) for k in range(len(words)) if len(" ".join(words[k:])) <= cap), words[-1][:cap])
    return long, words[-1][:3].upper()


def segment_teams(segment):
    """(home, away) club names of a DFL segment, from the match registry (the file name starts with the match id)."""
    mid = os.path.basename(segment).split("_")[0]
    if not os.path.exists(MATCHES):
        return None
    import csv
    for r in csv.DictReader(open(MATCHES, encoding="utf-8")):
        if r["match_id"] == mid:
            return r["home"], r["away"]
    return None


def team_lineup(team):
    """(names, shirts) of a team model's eleven in GRF's default line-up order, from its latest World Cup match."""
    roster = json.load(open(os.path.join(PFF_RAW, "Rosters", f"{LATEST[team]}.json"), encoding="utf-8"))
    xi = [p for p in roster if p["started"] and p["team"]["name"] == team]
    names, shirts, used = [], [], set()
    for place in GRF_LINEUP:
        pick = None
        for group in FITS[place] + [None]:                  # best fitting player still free; else anybody left
            cand = [p for p in xi if id(p) not in used and (group is None or p["positionGroupType"] == group)]
            if cand:
                pick = cand[0]
                break
        used.add(id(pick))
        names.append(pick["player"]["nickname"])
        shirts.append(pick["shirtNumber"])
    return names, shirts


def segment_lineup(segment):
    import numpy as np
    z = np.load(segment, allow_pickle=True)
    if "name_home" not in z.files:
        return None, None
    return ([list(z["name_home"]), [int(x) for x in z["jersey_home"]]], [list(z["name_away"]), [int(x) for x in z["jersey_away"]]])


def use_names(left=None, right=None, teams=None):
    """left/right: (names, shirts) for GRF's left/right team in slot order, or None to keep GRF's names.
    teams: (left club, right club), a name or None each, for the score board and the goal banner.
    Call before gfootball is imported. Returns the labels written."""
    if "gfootball_engine" in sys.modules:
        print("grf_names: GRF is already loaded in this process; names unchanged")
        return None
    src = engine_dir()
    blob = bytearray(open(os.path.join(src, "_gameplayfootball.pyd"), "rb").read())
    rooms = slot_rooms(bytes(blob))
    labels = []
    for grf, team in ((GRF_LEFT, left), (GRF_RIGHT, right)):
        for k, n in enumerate(grf):
            i, room = rooms[n]
            new = n if team is None else label(team[0][k], len(n))     # the engine keeps the old name's length
            labels.append(new)
            raw = new.encode("ascii")[:room]
            blob[i:i + room + 1] = raw + b"\0" * (room + 1 - len(raw))
    for (grf_long, grf_code), club in zip(GRF_TEAMS, teams or (None, None)):
        new = (grf_long, grf_code) if club is None else team_label(club, len(grf_long))
        for old, txt in zip((grf_long, grf_code), new):
            i = blob.find(b"\0" + old.encode() + b"\0") + 1               # the engine keeps the old length here too
            assert i > 0, f"GRF team name {old} not found in the engine"
            raw = txt.encode("ascii")[:len(old)]
            blob[i:i + len(old)] = raw + b"\0" * (len(old) - len(raw))
            labels.append(txt)
    key = hashlib.sha1("|".join(labels).encode()).hexdigest()[:12]
    dst = os.path.join(CACHE, key, "gfootball_engine")
    if not os.path.exists(os.path.join(dst, "_gameplayfootball.pyd")):
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(src, "*.dll")) + [os.path.join(src, "__init__.py")]:
            shutil.copy2(f, dst)
        tmp = os.path.join(dst, "_gameplayfootball.pyd.part")
        open(tmp, "wb").write(bytes(blob))
        os.replace(tmp, os.path.join(dst, "_gameplayfootball.pyd"))
    os.environ.setdefault("GFOOTBALL_DATA_DIR", os.path.join(src, "data"))          # GRF's own data and font, from the install
    os.environ.setdefault("GFOOTBALL_FONT", os.path.join(src, "fonts", "AlegreyaSansSC-ExtraBold.ttf"))
    sys.path.insert(0, os.path.dirname(dst))
    return labels


def setup_from_args(argv=None, default_segment=None):
    """Pick the names from the script's own arguments (--segment ..., or --video LEFT RIGHT) and load them into GRF."""
    argv = sys.argv[1:] if argv is None else argv
    left = right = clubs = None
    if "--video" in argv:
        i = argv.index("--video")
        teams = argv[i + 1:i + 3]
        left = team_lineup(teams[0]) if len(teams) > 0 and teams[0] in LATEST else None
        right = team_lineup(teams[1]) if len(teams) > 1 and teams[1] in LATEST else None
        clubs = tuple(t if t in LATEST else None for t in (teams + [None, None])[:2])
    else:
        seg = argv[argv.index("--segment") + 1] if "--segment" in argv else default_segment
        if seg:
            seg = seg if os.path.exists(seg) else os.path.join(ROOT, seg)
            if os.path.exists(seg):
                left, right = segment_lineup(seg)
                clubs = segment_teams(seg)
    if left is None and right is None:
        return None
    return use_names(left, right, clubs)
