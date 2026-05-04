
import pandas as pd
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

print("=" * 60)
print("DATASET SHAPES")
print("=" * 60)
print(f"  Train : {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
print(f"  Test  : {test_df.shape[0]:,}  rows × {test_df.shape[1]} columns")

print("\n" + "=" * 60)
print("COLUMN DTYPES (train)")
print("=" * 60)
print(train_df.dtypes.to_string())

print("\n" + "=" * 60)
print("MISSING VALUES (train)")
print("=" * 60)
missing = train_df.isnull().sum()
missing_pct = (missing / len(train_df) * 100).round(2)
missing_report = pd.DataFrame({"count": missing, "pct%": missing_pct})
missing_any = missing_report[missing_report["count"] > 0]
if missing_any.empty:
    print("  ✅ No missing values found in train set.")
else:
    print(missing_any.to_string())

print("\n" + "=" * 60)
print("MISSING VALUES (test)")
print("=" * 60)
missing_test = test_df.isnull().sum()
missing_any_test = missing_test[missing_test > 0]
if missing_any_test.empty:
    print("  ✅ No missing values found in test set.")
else:
    print(missing_any_test.to_string())

# ── Numerical features ─────────────────────────────────────────────────────
num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

print("\n" + "=" * 60)
print("NUMERICAL FEATURES")
print("=" * 60)
print(f"  {len(num_cols)} features: {num_cols}")

print("\n" + "=" * 60)
print("CATEGORICAL FEATURES")
print("=" * 60)
print(f"  {len(cat_cols)} features: {cat_cols}")

# ── Descriptive statistics ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS (train, numerical)")
print("=" * 60)
desc = train_df[num_cols].describe().T
desc["skewness"] = train_df[num_cols].skew()
desc["kurtosis"] = train_df[num_cols].kurt()
print(desc.round(4).to_string())

# ── Class ddistributionistribution ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLASS DISTRIBUTION  (track_genre — top 10 & summary)")
print("=" * 60)
genre_counts = train_df["track_genre"].value_counts()
print(f"  Total unique genres : {genre_counts.nunique()}")
print(f"  Min class count     : {genre_counts.min()}")
print(f"  Max class count     : {genre_counts.max()}")
print(f"  Mean class count    : {genre_counts.mean():.1f}")
print(f"  Std  class count    : {genre_counts.std():.1f}")
print("\n  Top 10 genres:")
print(genre_counts.head(10).to_string())
print("\n  Bottom 10 genres:")
print(genre_counts.tail(10).to_string())

# ── Duplicate check ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
n_dup_train = train_df.duplicated().sum()
n_dup_test  = test_df.duplicated().sum()
print(f"  Train duplicates : {n_dup_train:,}")
print(f"  Test  duplicates : {n_dup_test:,}")

# ── Export for downstream blocks ───────────────────────────────────────────
train_data = train_df
test_data  = test_df
audio_num_cols = [c for c in num_cols if c not in ("Unnamed: 0",)]
genre_value_counts = genre_counts
