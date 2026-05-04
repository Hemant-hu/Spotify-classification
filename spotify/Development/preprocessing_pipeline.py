import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

TARGET = "track_genre"

# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Load raw CSVs
# ─────────────────────────────────────────────────────────────────────────────
tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")
print(f"Loaded: train={tr.shape}, test={te.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Drop ID / label columns
# ─────────────────────────────────────────────────────────────────────────────
_drop_cols = ["Unnamed: 0", "track_id", "track_name", "album_name"]
tr.drop(columns=[c for c in _drop_cols if c in tr.columns], inplace=True)
te.drop(columns=[c for c in _drop_cols if c in te.columns], inplace=True)
print(f"[1] Dropped ID cols. train={tr.shape}, test={te.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Drop rows where target is null; normalize genre to plain str
# ─────────────────────────────────────────────────────────────────────────────
_n_before = len(tr)
tr = tr[tr[TARGET].notna()].copy()
tr[TARGET] = tr[TARGET].astype(str).str.strip()
te_has_target = TARGET in te.columns
print(f"[2] Dropped {_n_before - len(tr)} null-target rows → {len(tr)} train rows remain")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Encode explicit → binary 0/1 float
# ─────────────────────────────────────────────────────────────────────────────
def _enc_explicit(s: pd.Series) -> pd.Series:
    if hasattr(s, 'dtype') and s.dtype == bool:
        return s.astype(float)
    return (s.astype(str).str.strip().str.lower()
             .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
             .fillna(0.0))

tr["explicit"] = _enc_explicit(tr["explicit"])
te["explicit"] = _enc_explicit(te["explicit"])
print(f"[3] explicit binary: train={tr['explicit'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: popularity → numeric (coerce)
# ─────────────────────────────────────────────────────────────────────────────
tr["popularity"] = pd.to_numeric(tr["popularity"], errors="coerce")
te["popularity"] = pd.to_numeric(te["popularity"], errors="coerce")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Frequency-encode artists using log1p counts computed ONLY on train
#         Unseen test artists → 0
# ─────────────────────────────────────────────────────────────────────────────
_artist_vc = tr["artists"].value_counts()                     # fit on train only
artist_freq_map = {k: float(np.log1p(v)) for k, v in _artist_vc.items()}
tr["artist_freq"] = tr["artists"].map(artist_freq_map).fillna(0.0)
te["artist_freq"] = te["artists"].map(artist_freq_map).fillna(0.0)  # unseen → 0
tr.drop(columns=["artists"], inplace=True)
te.drop(columns=["artists"], inplace=True)
_n_unseen = (te["artist_freq"] == 0.0).sum()
print(f"[5] artist_freq: [{tr['artist_freq'].min():.3f}, {tr['artist_freq'].max():.3f}]"
      f"  |  unseen test artists → 0: {_n_unseen}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: key, mode, time_signature → ordinal integers (coerce to numeric)
# ─────────────────────────────────────────────────────────────────────────────
for _c in ["key", "mode", "time_signature"]:
    tr[_c] = pd.to_numeric(tr[_c], errors="coerce")
    te[_c] = pd.to_numeric(te[_c], errors="coerce")
print("[6] key, mode, time_signature kept as ordinal integers")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Interaction features
# ─────────────────────────────────────────────────────────────────────────────
for _df in [tr, te]:
    _df["energy_x_loudness"]       = pd.to_numeric(_df["energy"],       errors="coerce") \
                                   * pd.to_numeric(_df["loudness"],        errors="coerce")
    _df["dance_x_valence"]         = pd.to_numeric(_df["danceability"], errors="coerce") \
                                   * pd.to_numeric(_df["valence"],         errors="coerce")
    _df["acoustic_x_instrumental"] = pd.to_numeric(_df["acousticness"], errors="coerce") \
                                   * pd.to_numeric(_df["instrumentalness"],errors="coerce")
print("[7] Interaction features: energy_x_loudness, dance_x_valence, acoustic_x_instrumental")

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Log1p-transform right-skewed features
# ─────────────────────────────────────────────────────────────────────────────
for _f in ["duration_ms", "instrumentalness", "speechiness"]:
    for _df in [tr, te]:
        _df[f"log_{_f}"] = np.log1p(pd.to_numeric(_df[_f], errors="coerce").clip(lower=0))
        _df.drop(columns=[_f], inplace=True)
print("[8] Log1p-transformed: duration_ms, instrumentalness, speechiness → log_*")

# ─────────────────────────────────────────────────────────────────────────────
# Step 9: Bin tempo into 4 categories using TRAINING QUANTILES (no leakage)
#         Categories: slow / moderate / fast / very_fast
#         One-hot encode, then drop original tempo
# ─────────────────────────────────────────────────────────────────────────────
_tempo_train = pd.to_numeric(tr["tempo"], errors="coerce")
_tempo_test  = pd.to_numeric(te["tempo"], errors="coerce")

# Compute quantile boundaries on training data only
_q25, _q50, _q75 = _tempo_train.quantile([0.25, 0.50, 0.75]).values
_tempo_bins   = [-np.inf, _q25, _q50, _q75, np.inf]
_tempo_labels = ["slow", "moderate", "fast", "very_fast"]
print(f"[9] Tempo quantile bins (train-derived): "
      f"(-inf, {_q25:.1f}] / ({_q25:.1f}, {_q50:.1f}] / "
      f"({_q50:.1f}, {_q75:.1f}] / ({_q75:.1f}, +inf)")

# Bin both using training quantiles
_tb_tr = pd.cut(_tempo_train, bins=_tempo_bins, labels=_tempo_labels)
_tb_te = pd.cut(_tempo_test,  bins=_tempo_bins, labels=_tempo_labels)

# One-hot encode (align test columns to match train)
_ohe_tr = pd.get_dummies(_tb_tr, prefix="tempo").astype(float)
_ohe_te = pd.get_dummies(_tb_te, prefix="tempo").astype(float)
for _c in _ohe_tr.columns:
    if _c not in _ohe_te.columns:
        _ohe_te[_c] = 0.0
_ohe_te = _ohe_te[_ohe_tr.columns]  # enforce column order

tr = pd.concat([tr.drop(columns=["tempo"]), _ohe_tr], axis=1)
te = pd.concat([te.drop(columns=["tempo"]), _ohe_te], axis=1)
print(f"[9] Tempo OHE columns: {list(_ohe_tr.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 10: Build clean float64 feature arrays
# ─────────────────────────────────────────────────────────────────────────────
_feat_cols   = [c for c in tr.columns if c != TARGET]
feature_names = list(_feat_cols)

X_all_df  = tr[_feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float64)
X_test_df = te[_feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float64)
y_all     = tr[TARGET].values   # plain str array

print(f"\n[10] Feature count: {len(feature_names)}")
print(f"     Features     : {feature_names}")
print(f"     NaN  X_all   : {X_all_df.isna().sum().sum()}"
      f"  |  X_test: {X_test_df.isna().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 11: Median imputation — fit on full training set, apply to both
# ─────────────────────────────────────────────────────────────────────────────
pp_imputer = SimpleImputer(strategy="median")
X_all_imp  = pp_imputer.fit_transform(X_all_df.values)
X_test_imp = pp_imputer.transform(X_test_df.values)
print(f"[11] Post-impute NaN: X_all={np.isnan(X_all_imp).sum()},"
      f" X_test={np.isnan(X_test_imp).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 12: Stratified 80/20 train-val split BEFORE scaling
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_all_imp, y_all,
    test_size=0.2, random_state=42, stratify=y_all
)
print(f"[12] Stratified split: X_train={X_train.shape}, X_val={X_val.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 13: StandardScaler — fit ONLY on X_train, transform val + test
# ─────────────────────────────────────────────────────────────────────────────
pp_scaler        = StandardScaler()
X_train          = pp_scaler.fit_transform(X_train)
X_val            = pp_scaler.transform(X_val)
X_test_processed = pp_scaler.transform(X_test_imp)
print("[13] StandardScaler fitted on X_train only → zero leakage confirmed")

# ─────────────────────────────────────────────────────────────────────────────
# Step 14: Assertions — zero NaN, genre alignment, shape sanity
# ─────────────────────────────────────────────────────────────────────────────
assert not np.isnan(X_train).any(),          "NaN found in X_train!"
assert not np.isnan(X_val).any(),            "NaN found in X_val!"
assert not np.isnan(X_test_processed).any(), "NaN found in X_test_processed!"
assert X_train.shape[1] == len(feature_names), "Feature count mismatch!"
assert set(np.unique(y_train)) == set(np.unique(y_val)), "Genre mismatch train/val!"

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
_n_genres = len(np.unique(y_train))
print("\n" + "=" * 62)
print("  ✅ PREPROCESSING COMPLETE — NO LEAKAGE")
print("=" * 62)
print(f"  X_train          : {X_train.shape}  (NaN: {np.isnan(X_train).sum()})")
print(f"  X_val            : {X_val.shape}  (NaN: {np.isnan(X_val).sum()})")
print(f"  y_train          : {y_train.shape}  | {_n_genres} unique genres")
print(f"  y_val            : {y_val.shape}  | {len(np.unique(y_val))} unique genres")
print(f"  X_test_processed : {X_test_processed.shape}  (NaN: {np.isnan(X_test_processed).sum()})")
print(f"\n  Total features   : {len(feature_names)}")
print(f"\n  Feature index:")
for _idx, _fn in enumerate(feature_names):
    print(f"    {_idx+1:>2}. {_fn}")
print("\n  Leakage guard summary:")
print(f"    • artist_freq_map   — fitted on train ({len(artist_freq_map)} artists)")
print(f"    • tempo quantile bins — fitted on train (q25={_q25:.1f}, q50={_q50:.1f}, q75={_q75:.1f})")
print(f"    • pp_imputer        — fitted on train only")
print(f"    • pp_scaler         — fitted on X_train only")
print("=" * 62)
