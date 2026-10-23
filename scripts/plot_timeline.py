"""E1: key-state timeline, real match vs simulated matches, on a 0-100% time axis."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ORDER = ["BUILD-UP", "PROGRESSION", "FINAL THIRD", "COUNTER-ATTACK",
         "TURNOVER", "LOOSE BALL", "STOPPAGE", "SET-PIECE", "GOAL"]
COLORS = ["#9CC5A1", "#4F9D69", "#14532D", "#E8833A",
          "#F2C14E", "#B0B8C0", "#6B7A89", "#2F6DB5", "#C0392B"]

ROWS = [
    ("Real: Köln 1–2 Bayern (97 min)", "data/real/processed/J03WMX_keystates.csv"),
    ("Simulated match 1 (5 min)", "data/simulated/match_001_keystates.csv"),
    ("Simulated match 2 (5 min)", "data/simulated/match_002_keystates.csv"),
]


def base_state(s):
    """Drop '+ pressure' and '(corner)' details so each second has one of the base states."""
    return s.str.replace(r" \+ pressure", "", regex=True).str.replace(r" \(.*\)", "", regex=True).str.strip()


fig, ax = plt.subplots(figsize=(11, 3.6))
cmap = ListedColormap(COLORS)
for i, (label, path) in enumerate(ROWS):
    df = pd.read_csv(path)
    y = len(ROWS) - 1 - i
    codes = [ORDER.index(s) for s in base_state(df["state"])]
    ax.imshow([codes], aspect="auto", cmap=cmap, vmin=0, vmax=len(ORDER) - 1,
              interpolation="nearest", extent=[0, 100, y, y + 0.8])
    goal_pos = df.index[df["goal"]] / len(df) * 100
    ax.scatter(goal_pos, [y + 0.93] * len(goal_pos), marker="v", color="black", s=40, zorder=3)
    ax.text(-1, y + 0.4, label, ha="right", va="center", fontsize=10)

ax.set_xlim(0, 100)
ax.set_ylim(0, len(ROWS) + 0.1)
ax.set_yticks([])
ax.set_xlabel("Share of the match (%)")
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.legend(handles=[Patch(color=c, label=n) for n, c in zip(ORDER, COLORS)],
          loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=9, frameon=False, fontsize=8)
ax.set_title("Key states over a match (black triangles = goals)", loc="left")
plt.tight_layout()
plt.savefig("outputs/key_state_timeline.png", dpi=150, bbox_inches="tight")
print("saved outputs/key_state_timeline.png")