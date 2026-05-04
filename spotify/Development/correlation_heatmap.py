
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BG    = "#1D1D20"
FG    = "#fbfbff"
MUTED = "#909094"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "axes.titlecolor": FG,
})

# ── Select audio features (exclude row index) ─────────────────────────────
audio_features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms",
]
corr_df = train_data[audio_features].dropna()
corr_matrix = corr_df.corr()

# ── Print top correlated pairs ─────────────────────────────────────────────
print("=" * 60)
print("TOP 10 FEATURE CORRELATIONS (absolute)")
print("=" * 60)
_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
_pairs = _upper.stack().abs().sort_values(ascending=False)
print(_pairs.head(10).to_string())

print("\n  Interpretation:")
print("  • energy–loudness    : strong positive — louder = more energy")
print("  • energy–acousticness: strong negative — acoustic tracks = quieter/calmer")
print("  • instrumentalness–speechiness: moderate negative")

# ── Heatmap ────────────────────────────────────────────────────────────────
fig_corr, ax_corr = plt.subplots(figsize=(11, 9), facecolor=BG)
ax_corr.set_facecolor(BG)

cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True   # upper triangle masked

sns.heatmap(
    corr_matrix, mask=mask, cmap=cmap,
    vmin=-1, vmax=1, center=0,
    annot=True, fmt=".2f", annot_kws={"size": 9, "color": FG},
    linewidths=0.5, linecolor=BG,
    cbar_kws={"shrink": 0.8},
    ax=ax_corr,
)
ax_corr.set_title("Pearson Correlation Heatmap — Audio Features\n"
                  "(lower triangle; red = positive, blue = negative)",
                  fontsize=13, pad=14, color=FG)
ax_corr.tick_params(axis="x", rotation=45, labelsize=9)
ax_corr.tick_params(axis="y", rotation=0,  labelsize=9)

# Style colorbar
cbar = ax_corr.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color=FG, labelsize=8)
cbar.ax.set_facecolor(BG)

plt.tight_layout()
plt.show()
print("\n▶ Heatmap: Pearson correlation across all key audio features.")
print("  Strong pairs to watch for ML: energy-loudness, energy-acousticness.")
