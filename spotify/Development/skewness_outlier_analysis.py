
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BG        = "#1D1D20"
FG        = "#fbfbff"
MUTED     = "#909094"
ACCENT    = "#A1C9F4"
ORANGE    = "#FFB482"
WARN      = "#f04438"
OK        = "#17b26a"
HIGHLIGHT = "#ffd400"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "axes.titlecolor": FG, "grid.color": MUTED,
    "grid.alpha": 0.25,
})

audio_feats = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms",
]

# ── Skewness analysis ──────────────────────────────────────────────────────
skew_vals = train_data[audio_feats].skew().sort_values(key=abs, ascending=False)
print("=" * 60)
print("SKEWNESS ANALYSIS")
print("=" * 60)
for feat, sk in skew_vals.items():
    flag = ""
    if abs(sk) > 2: flag = "  ⚠️  HIGHLY skewed"
    elif abs(sk) > 1: flag = "  ⚡ Moderately skewed"
    print(f"  {feat:<20} {sk:>8.4f}{flag}")
print("\n  Rule of thumb: |skew| > 1 = notable, > 2 = consider log transform")

# ── Outlier detection: IQR method ─────────────────────────────────────────
print("\n" + "=" * 60)
print("OUTLIER SUMMARY (IQR × 1.5)")
print("=" * 60)
for feat in audio_feats:
    col = train_data[feat].dropna()
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((col < lo) | (col > hi)).sum()
    pct   = n_out / len(col) * 100
    flag  = "  ⚠️" if pct > 5 else ""
    print(f"  {feat:<22} {n_out:>5,} outliers  ({pct:5.1f}%){flag}")

# ── Skewness bar chart ─────────────────────────────────────────────────────
fig_sk, ax_sk = plt.subplots(figsize=(11, 5), facecolor=BG)
ax_sk.set_facecolor(BG)
_colors_sk = [WARN if abs(v) > 2 else ORANGE if abs(v) > 1 else ACCENT
              for v in skew_vals.values]
bars_sk = ax_sk.barh(skew_vals.index, skew_vals.values, color=_colors_sk, edgecolor="none")
ax_sk.axvline(0, color=MUTED, linewidth=1)
ax_sk.axvline(1,  color=HIGHLIGHT, linewidth=1, linestyle="--", alpha=0.7, label="|1| threshold")
ax_sk.axvline(-1, color=HIGHLIGHT, linewidth=1, linestyle="--", alpha=0.7)
ax_sk.axvline(2,  color=WARN, linewidth=1, linestyle=":", alpha=0.7, label="|2| threshold")
ax_sk.axvline(-2, color=WARN, linewidth=1, linestyle=":", alpha=0.7)
ax_sk.set_xlabel("Skewness", fontsize=11)
ax_sk.set_title("Feature Skewness — Audio Features\n"
                "(blue = normal, orange = moderate, red = high skew)",
                fontsize=13, pad=12)
ax_sk.legend(facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=9)
plt.tight_layout()
plt.show()
print("\n▶ Skewness chart: features ordered by absolute skew magnitude.")
print("  duration_ms has extreme skew — log transform recommended.")

# ── Boxplot of audio features (normalized for comparison) ─────────────────
# Clip extreme outliers for better visual readability (quantile-based)
clip_df = train_data[audio_feats].copy()
for col in clip_df.columns:
    lo_c = clip_df[col].quantile(0.01)
    hi_c = clip_df[col].quantile(0.99)
    clip_df[col] = clip_df[col].clip(lo_c, hi_c)

# Standard-scale each feature for side-by-side comparison
from sklearn.preprocessing import StandardScaler
_scaler = StandardScaler()
scaled_arr = _scaler.fit_transform(clip_df.dropna())
scaled_for_box = pd.DataFrame(scaled_arr, columns=audio_feats)

fig_box, ax_box = plt.subplots(figsize=(14, 6), facecolor=BG)
ax_box.set_facecolor(BG)

bp = ax_box.boxplot(
    [scaled_for_box[f].dropna().values for f in audio_feats],
    labels=audio_feats,
    patch_artist=True,
    medianprops=dict(color=HIGHLIGHT, linewidth=2),
    whiskerprops=dict(color=MUTED),
    capprops=dict(color=MUTED),
    flierprops=dict(marker="o", color=WARN, markersize=2, alpha=0.3,
                    markeredgecolor="none"),
    boxprops=dict(facecolor=ACCENT, alpha=0.7, linewidth=0),
)
ax_box.axhline(0, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.5)
ax_box.set_xticklabels(audio_feats, rotation=40, ha="right", fontsize=9)
ax_box.set_ylabel("Standardised Value (z-score)", fontsize=11)
ax_box.set_title("Boxplots — Audio Features (z-scored, 1–99th pct clipped)\n"
                 "Red dots = outliers after clipping",
                 fontsize=13, pad=12)
plt.tight_layout()
plt.show()
print("\n▶ Boxplot: standardised distributions. Long whiskers & many red dots = outlier-rich.")
print("  speechiness, liveness, instrumentalness show heavy-tailed distributions.")
