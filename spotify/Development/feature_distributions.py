
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BG        = "#1D1D20"
FG        = "#fbfbff"
MUTED     = "#909094"
ACCENT    = "#A1C9F4"
ORANGE    = "#FFB482"
OK        = "#17b26a"
HIGHLIGHT = "#ffd400"
PURPLE    = "#D0BBFF"
CORAL     = "#FF9F9B"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "axes.titlecolor": FG, "grid.color": MUTED,
    "grid.alpha": 0.25,
})

KEY_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms",
]
FEAT_COLORS = [ACCENT, ORANGE, OK, HIGHLIGHT, PURPLE, CORAL,
               "#8DE5A1", "#FFB482", "#A1C9F4", "#D0BBFF"]

# Print basic stats for context
print("=" * 60)
print("FEATURE DISTRIBUTION SUMMARY")
print("=" * 60)
for feat in KEY_FEATURES:
    col = train_data[feat].dropna()
    print(f"  {feat:<22}  mean={col.mean():.4f}  median={col.median():.4f}  "
          f"min={col.min():.3f}  max={col.max():.3f}")

# ── Plot 1-5: KDE + histogram for first 5 features ─────────────────────────
for i, feat in enumerate(KEY_FEATURES[:5]):
    col = train_data[feat].dropna()
    # Clip top 0.5% to avoid tail stretching
    hi_clip = col.quantile(0.995)
    lo_clip = col.quantile(0.005)
    col_c = col.clip(lo_clip, hi_clip)

    fig_d, ax_d = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax_d.set_facecolor(BG)
    ax_d.hist(col_c, bins=60, density=True, color=FEAT_COLORS[i],
              alpha=0.7, edgecolor="none")

    # Overlay KDE using numpy manual estimate
    from scipy.stats import gaussian_kde
    kde_fn = gaussian_kde(col_c.sample(min(5000, len(col_c)), random_state=42))
    x_kde = np.linspace(col_c.min(), col_c.max(), 300)
    ax_d.plot(x_kde, kde_fn(x_kde), color=FG, linewidth=2, label="KDE")

    ax_d.axvline(col.mean(),   color=HIGHLIGHT, linewidth=1.5, linestyle="--",
                 label=f"Mean {col.mean():.3f}")
    ax_d.axvline(col.median(), color=OK,        linewidth=1.5, linestyle=":",
                 label=f"Median {col.median():.3f}")
    ax_d.set_xlabel(feat.capitalize(), fontsize=11)
    ax_d.set_ylabel("Density", fontsize=11)
    ax_d.set_title(f"Distribution of '{feat}'\n"
                   f"skew={col.skew():.3f}  kurt={col.kurt():.3f}  "
                   f"n={len(col):,}",
                   fontsize=12, pad=10)
    ax_d.legend(facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=9)
    plt.tight_layout()
    plt.show()

print("\n▶ Charts 1-5: Distribution histograms + KDE for danceability, energy,")
print("  loudness, speechiness, acousticness.")

# ── Plot 6-10: Second group ─────────────────────────────────────────────────
for i, feat in enumerate(KEY_FEATURES[5:]):
    col = train_data[feat].dropna()
    hi_clip = col.quantile(0.995)
    lo_clip = col.quantile(0.005)
    col_c = col.clip(lo_clip, hi_clip)

    from scipy.stats import gaussian_kde
    fig_d2, ax_d2 = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax_d2.set_facecolor(BG)
    ax_d2.hist(col_c, bins=60, density=True, color=FEAT_COLORS[i + 5],
               alpha=0.7, edgecolor="none")

    kde_fn2 = gaussian_kde(col_c.sample(min(5000, len(col_c)), random_state=42))
    x_kde2  = np.linspace(col_c.min(), col_c.max(), 300)
    ax_d2.plot(x_kde2, kde_fn2(x_kde2), color=FG, linewidth=2, label="KDE")

    ax_d2.axvline(col.mean(),   color=HIGHLIGHT, linewidth=1.5, linestyle="--",
                  label=f"Mean {col.mean():.3f}")
    ax_d2.axvline(col.median(), color=OK,        linewidth=1.5, linestyle=":",
                  label=f"Median {col.median():.3f}")
    ax_d2.set_xlabel(feat.capitalize(), fontsize=11)
    ax_d2.set_ylabel("Density", fontsize=11)
    ax_d2.set_title(f"Distribution of '{feat}'\n"
                    f"skew={col.skew():.3f}  kurt={col.kurt():.3f}  "
                    f"n={len(col):,}",
                    fontsize=12, pad=10)
    ax_d2.legend(facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=9)
    plt.tight_layout()
    plt.show()

print("\n▶ Charts 6-10: Distribution histograms + KDE for instrumentalness, liveness,")
print("  valence, tempo, duration_ms.")

# ── Missing value heatmap ──────────────────────────────────────────────────
missing_cols = [c for c in train_data.columns if train_data[c].isnull().any()]
miss_pct_vals = [(c, train_data[c].isnull().mean() * 100) for c in missing_cols]
miss_pct_vals.sort(key=lambda x: x[1], reverse=True)
_miss_feats = [x[0] for x in miss_pct_vals]
_miss_pcts  = [x[1] for x in miss_pct_vals]

fig_mv, ax_mv = plt.subplots(figsize=(12, 5), facecolor=BG)
ax_mv.set_facecolor(BG)
bar_miss_colors = [CORAL if p > 10 else ORANGE if p > 5 else ACCENT for p in _miss_pcts]
ax_mv.barh(_miss_feats, _miss_pcts, color=bar_miss_colors, edgecolor="none")
ax_mv.axvline(5,  color=ORANGE, linewidth=1, linestyle="--", alpha=0.7, label="5% threshold")
ax_mv.axvline(10, color=CORAL,  linewidth=1, linestyle=":",  alpha=0.7, label="10% threshold")
ax_mv.set_xlabel("Missing (%)", fontsize=11)
ax_mv.set_title("Missing Value Profile — Train Set\n"
                "(red > 10%, orange 5–10%, blue < 5%)",
                fontsize=13, pad=12)
ax_mv.legend(facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=9)
plt.tight_layout()
plt.show()
print("\n▶ Missing value chart: artists (13.5%), album_name (9.5%) and explicit (8.8%)")
print("  are the most problematic — imputation or exclusion needed.")
