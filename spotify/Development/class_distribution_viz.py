
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Palette / theme ────────────────────────────────────────────────────────
BG        = "#1D1D20"
FG        = "#fbfbff"
MUTED     = "#909094"
ACCENT    = "#A1C9F4"
WARN      = "#f04438"
OK        = "#17b26a"
HIGHLIGHT = "#ffd400"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "axes.titlecolor": FG, "grid.color": MUTED,
    "grid.alpha": 0.25, "font.family": "sans-serif",
})

# genre_value_counts comes from data_overview
genre_counts_clean = genre_value_counts.dropna()
n_genres = len(genre_counts_clean)
mean_val = genre_counts_clean.mean()
cv = genre_counts_clean.std() / mean_val * 100

# ── Insight commentary ─────────────────────────────────────────────────────
print("=" * 60)
print("CLASS DISTRIBUTION INSIGHTS")
print("=" * 60)
print(f"  Total genres       : {n_genres}")
print(f"  Min  class size    : {genre_counts_clean.min()}")
print(f"  Max  class size    : {genre_counts_clean.max()}")
print(f"  Mean class size    : {mean_val:.1f}")
print(f"  Std  class size    : {genre_counts_clean.std():.1f}")
print(f"  Coeff of variation : {cv:.1f}%  (< 15% = near-balanced)")
if cv < 15:
    print("  ✅ Dataset is NEAR-BALANCED across genres — mild imbalance only.")
else:
    print("  ⚠️  Significant class imbalance detected.")
print(f"\n  NOTE: genre_counts shows {n_genres} unique labels.")

# ── Plot 1: Full genre distribution (all genres sorted) ────────────────────
sorted_counts = genre_counts_clean.sort_values(ascending=False)
fig1, ax1 = plt.subplots(figsize=(16, 7), facecolor=BG)
ax1.set_facecolor(BG)

bar_colors = [ACCENT] * len(sorted_counts)
bar_colors[0] = HIGHLIGHT          # max
bar_colors[-1] = WARN              # min
ax1.bar(range(len(sorted_counts)), sorted_counts.values, color=bar_colors,
        edgecolor="none", width=0.85)
ax1.axhline(mean_val, color=OK, linestyle="--", linewidth=1.5, alpha=0.8,
            label=f"Mean = {mean_val:.0f}")

ax1.set_xticks(range(len(sorted_counts)))
ax1.set_xticklabels(sorted_counts.index, rotation=90, fontsize=6.5, ha="center")
ax1.set_ylabel("Track Count", fontsize=11)
ax1.set_title(f"Class Distribution Across {n_genres} Genres\n"
              "(Yellow = max class, Red = min class)",
              fontsize=13, pad=12)
ax1.legend(facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=10)
ax1.tick_params(axis="x", labelsize=6.5)
plt.tight_layout()
plt.show()
print("\n▶ Chart 1: All genres ranked by track count.")
print("  Yellow bar = most common genre, Red = least common.")

# ── Plot 2: Distribution of class sizes (histogram) ────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor=BG)
ax2.set_facecolor(BG)

ax2.hist(genre_counts_clean.values, bins=30, color=ACCENT, edgecolor=BG, linewidth=0.5)
ax2.axvline(mean_val, color=HIGHLIGHT, linestyle="--", linewidth=2,
            label=f"Mean {mean_val:.0f}")
ax2.axvline(genre_counts_clean.median(), color=OK, linestyle=":", linewidth=2,
            label=f"Median {genre_counts_clean.median():.0f}")
ax2.set_xlabel("Tracks per Genre", fontsize=11)
ax2.set_ylabel("Number of Genres", fontsize=11)
ax2.set_title("Histogram of Per-Genre Track Counts\n(How evenly is data distributed?)",
              fontsize=13, pad=12)
ax2.legend(facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=10)
plt.tight_layout()
plt.show()
print("\n▶ Chart 2: Histogram of class sizes — tightly clustered = balanced split.")
