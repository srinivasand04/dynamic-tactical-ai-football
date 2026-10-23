"""E2: share of time in each key state, real match vs simulated matches (grouped bars)."""
import pandas as pd
import matplotlib.pyplot as plt

ORDER = ["BUILD-UP", "PROGRESSION", "FINAL THIRD", "COUNTER-ATTACK",
         "TURNOVER", "LOOSE BALL", "STOPPAGE", "SET-PIECE", "GOAL"]

SERIES = [
    ("Real: Köln 1–2 Bayern", "data/real/processed/J03WMX_keystates.csv", "#14532D"),
    ("Simulated match 1", "data/simulated/match_001_keystates.csv", "#E8833A"),
    ("Simulated match 2", "data/simulated/match_002_keystates.csv", "#2F6DB5"),
]


def base_state(s):
    """Drop '+ pressure' and '(corner)' details so each second has one of the base states."""
    return s.str.replace(r" \+ pressure", "", regex=True).str.replace(r" \(.*\)", "", regex=True).str.strip()


shares = {}
for label, path, _ in SERIES:
    df = pd.read_csv(path)
    shares[label] = base_state(df["state"]).value_counts(normalize=True).mul(100).reindex(ORDER).fillna(0)
table = pd.DataFrame(shares).round(1)
print(table.to_string())

fig, ax = plt.subplots(figsize=(11, 4.6))
n = len(SERIES)
w = 0.8 / n
for k, (label, _, color) in enumerate(SERIES):
    xs = [i + (k - (n - 1) / 2) * w for i in range(len(ORDER))]
    bars = ax.bar(xs, table[label], width=w, label=label, color=color)
    ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, fontsize=9)
ax.set_ylabel("Share of time (%)")
ax.set_title("Where the time goes: key-state share per match", loc="left")
ax.legend(frameon=False)
for side in ("right", "top"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/key_state_share.png", dpi=150)
print("saved outputs/key_state_share.png")