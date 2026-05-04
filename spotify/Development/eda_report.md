# 🎵 Spotify Track Genre Classification — EDA & Modelling Report

> **Dataset**: `train.csv` (84,800 rows × 21 cols) | `test.csv` (34,200 rows × 20 cols)  
> **Task**: Multi-class classification — predict `track_genre` (114 classes)

---

## 1. Dataset Overview

| Property | Train | Test |
|---|---|---|
| Rows | 84,800 | 34,200 |
| Columns | 21 | 20 |
| Numerical features | 14 | 14 |
| Categorical features | 7 | 6 |
| Duplicate rows | See `data_overview` output | — |
| Null genre rows dropped | 1,853 (2.19%) | — |

### 1.1 Column Types
- **Numerical**: `Unnamed: 0`, `duration_ms`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`
- **Categorical**: `track_id`, `artists`, `album_name`, `track_name`, `popularity` *(stored as str in train — dtype inconsistency)*, `explicit`, `track_genre`

> ⚠️ **Dtype inconsistency**: `popularity` is `str` in train vs `int64` in test. `explicit` is `str` in train vs `bool` in test. Both require normalisation.

---

## 2. Missing Value Analysis

### Train Set (14 columns with missing data)

| Feature | Missing Count | Missing % | Action |
|---|---|---|---|
| artists | 11,464 | 13.52% | Frequency-encode; missing → freq=0 |
| album_name | 8,070 | 9.52% | Drop (high cardinality, no direct signal) |
| explicit | 7,474 | 8.81% | Binary encode; NaN → 0 |
| energy | 5,498 | 6.48% | Median imputation |
| speechiness | 5,157 | 6.08% | Median imputation (+ log-transform) |
| loudness | 4,139 | 4.88% | Median imputation |
| instrumentalness | 3,317 | 3.91% | Median imputation (+ log-transform) |
| danceability | 3,316 | 3.91% | Median imputation |
| mode | 2,783 | 3.28% | Median imputation |
| tempo | 2,102 | 2.48% | Median imputation → binned |
| track_genre | 1,853 | 2.19% | **Rows dropped** |
| time_signature | 1,425 | 1.68% | Median imputation |
| liveness | 567 | 0.67% | Median imputation |
| popularity | 90 | 0.11% | Median imputation |

### Test Set
Only 3 rows missing (artists, album_name, track_name) — negligible.

---

## 3. Class Distribution — `track_genre`

- **114 unique genres** in the training set
- Per-class track counts: **min = 693**, **max = 772**, **mean = 727.6**
- **Coefficient of Variation = 2.3%** — dataset is **near-perfectly balanced**
- ✅ No need for class-weight adjustments, SMOTE, or stratified sampling beyond routine use
- Stratified 80/20 train/val split was used to maintain balance across folds

> The distribution is remarkably uniform, which is unusual for real-world genre datasets and strongly suggests the data was artificially balanced per genre. This makes standard accuracy a valid evaluation metric.

---

## 4. Descriptive Statistics — Key Audio Features

| Feature | Mean | Median | Min | Max | Skewness | Kurtosis | Notes |
|---|---|---|---|---|---|---|---|
| danceability | 0.567 | 0.581 | 0.000 | 0.985 | −0.40 | — | Near-normal |
| energy | 0.642 | 0.685 | 0.000 | 1.000 | −0.60 | — | Slightly left-skewed |
| loudness | −8.26 | −7.01 | −49.31 | 4.53 | −2.01 | — | ⚠️ Highly negative-skewed |
| speechiness | 0.291 | 0.052 | −9.00 | 10.00 | +0.27 | — | ⚠️ Out-of-range values! |
| acousticness | 0.314 | 0.168 | 0.000 | 0.996 | +0.73 | — | Right-skewed |
| instrumentalness | 0.157 | 0.000 | 0.000 | 1.000 | +1.73 | — | ⚡ Highly right-skewed |
| liveness | 0.218 | 0.131 | −1.00 | 1.55 | +1.33 | — | ⚡ Out-of-range values! |
| valence | 0.474 | 0.464 | 0.000 | 0.995 | +0.12 | — | Near-normal |
| tempo | 122.1 | 122.0 | 0.000 | 222.6 | +0.24 | — | Bimodal tendencies |
| duration_ms | 227,828 | 212,800 | 13,386 | 5,237,295 | +10.49 | 338.6 | ⚠️ Extreme positive skew |

> **Notable data quality issues**: `speechiness` contains values outside [0,1] (min=−9, max=10) and `liveness` contains values < 0 (min=−1). These likely represent data entry errors or sensor artefacts. Log-transform + clipping applied during preprocessing.

---

## 5. Correlation Analysis

Top feature pairs by Pearson |r|:

| Feature Pair | Correlation | Interpretation |
|---|---|---|
| energy ↔ loudness | **+0.762** | Strong positive — energetic tracks are louder |
| energy ↔ acousticness | **−0.731** | Strong negative — acoustic = low energy |
| loudness ↔ acousticness | −0.591 | Moderate negative |
| danceability ↔ valence | +0.477 | Happier tracks are more danceable |
| loudness ↔ instrumentalness | −0.434 | Instrumental tracks are quieter |

> **Feature engineering implication**: The strong energy–loudness and energy–acousticness relationships suggest multiplicative interaction terms will capture non-linear genre-specific signal. These pairs were used directly as engineered features.

---

## 6. Skewness & Outlier Analysis

### Skewness Summary

| Feature | Skew | Verdict |
|---|---|---|
| duration_ms | +10.49 | ⚠️ Extreme → `log1p` transform |
| loudness | −2.01 | ⚠️ High → keep; tree models handle this |
| instrumentalness | +1.73 | ⚡ Moderate → `log1p` transform |
| liveness | +1.33 | ⚡ Moderate → keep (already clipped) |
| speechiness | +0.27 | ✅ Acceptable |
| valence | +0.12 | ✅ Normal-like |

### Outlier Profile (IQR × 1.5)

| Feature | Outlier % | Risk |
|---|---|---|
| instrumentalness | 22.2% | ⚠️ Very high — many silent tracks |
| speechiness | 18.8% | ⚠️ High — podcasts / spoken word |
| liveness | 9.0% | ⚠️ Live recordings inflate tail |
| loudness | 5.4% | ⚠️ Very quiet/noisy extremes |
| duration_ms | 5.0% | ⚠️ Very long tracks (classical/ambient) |
| danceability / tempo | < 1% | ✅ Minimal |

---

## 7. Preprocessing Decisions

1. **Drop ID columns**: `Unnamed: 0`, `track_id`, `track_name`, `album_name` — no predictive value for genre classification
2. **Drop null-genre rows**: 1,853 rows where `track_genre` is null; cannot use for supervised learning
3. **Encode `explicit`**: Map `true`/`false`/bool → `1.0`/`0.0`; NaN → 0.0
4. **Normalise `popularity`**: `pd.to_numeric(errors='coerce')` to handle string-typed values in train
5. **Artist frequency encoding**: Replace `artists` (22,758 unique values) with `log1p(artist_track_count)` — captures artist popularity without high-cardinality OHE. Fit on train only; unseen → 0.0
6. **Log-transform skewed features**: `duration_ms`, `instrumentalness`, `speechiness` — reduces skew, improves model sensitivity
7. **Tempo binning + OHE**: 4 bins (slow/moderate/fast/very_fast) — captures non-linear tempo effects
8. **Median imputation**: All remaining NaN filled with training-set median (fit on X_train, applied to val & test to prevent leakage)
9. **StandardScaler**: Fit only on X_train; transforms val and test sets identically — required for distance-based and regularised models

---

## 8. Feature Engineering

Final feature set (22 features after engineering):

| # | Feature | Type | Notes |
|---|---|---|---|
| 1 | popularity | Numeric | Cleaned from str |
| 2 | explicit | Binary | Encoded 0/1 |
| 3–10 | danceability, energy, key, loudness, mode, acousticness, liveness, valence | Numeric | Raw Spotify audio features |
| 11 | time_signature | Numeric | Ordinal kept as-is |
| 12 | artist_freq | Numeric | log1p(artist track count) — frequency encoding |
| 13 | energy_x_loudness | Interaction | energy × loudness — strong correlated pair |
| 14 | dance_x_valence | Interaction | danceability × valence — mood interaction |
| 15 | acoustic_x_instrumental | Interaction | acousticness × instrumentalness — texture signal |
| 16 | log_duration_ms | Log-transform | log1p(duration_ms) — corrects extreme skew |
| 17 | log_instrumentalness | Log-transform | log1p(instrumentalness) |
| 18 | log_speechiness | Log-transform | log1p(speechiness.clip(0)) |
| 19–22 | tempo_slow/moderate/fast/very_fast | OHE | Binned tempo categories |

**Split**: Stratified 80/20 → X_train (66,357 × 22), X_val (16,590 × 22)

---

## 9. Model Selection Reasoning

### Task Characteristics
- **114-class** multi-class classification with balanced classes
- **22 features** after engineering; all numerical after preprocessing
- **82,947 training samples** — large enough for tree ensembles and shallow NNs
- Near-balanced classes → accuracy is a reliable metric alongside macro-F1

### Recommended Model Hierarchy

| Model | Rationale | Expected Performance |
|---|---|---|
| **LightGBM / XGBoost** | Handles mixed numerics + interactions natively; fast; resilient to outliers; top Kaggle performer on tabular data | 🥇 Best expected |
| **Random Forest** | Robust baseline; handles non-linearity; parallelisable; interpretable via feature importance | 🥈 Strong baseline |
| **k-NN** | Direct genre neighbourhood — acoustically similar tracks likely share genre; sensitive to standardisation (already applied) | 🥉 Competitive for local structure |
| **Logistic Regression** | Linear baseline; reveals separability of feature space; regularisation handles collinearity | 📊 Benchmark |
| **MLP / Neural Net** | Can model complex feature interactions; needs more tuning | 🔮 Potential upside |

### Why Tree Ensembles are Favoured
1. Robust to remaining outliers (speechiness, liveness have out-of-range values)
2. Built-in feature selection via impurity-based importance
3. No assumption of linearity — genre boundaries in audio space are non-linear
4. LightGBM handles 114-class problems efficiently with histogram-based splitting

---

## 10. Final Observations & Recommendations

### Data Quality
- `speechiness` and `liveness` contain physically impossible values (< 0 or > 1) — likely sensor/API errors. Clipped at 0 during preprocessing.
- `popularity` has dtype inconsistency between train and test — normalised to numeric in pipeline.
- 14 of 21 columns have missing data in train, but all are manageable via median imputation.

### EDA Findings Summary
1. ✅ **Balanced classes** — no resampling needed; accuracy is a fair metric
2. ⚠️ **Highly skewed features** — `duration_ms` (skew=10.5) and `loudness` (skew=−2.0) require transformation; applied via log1p
3. 🔗 **Strong feature correlations** — energy–loudness (r=0.76) used as interaction term; prevents redundancy for linear models
4. 📊 **High outlier rates** — instrumentalness (22%) and speechiness (19%) have genre-meaningful bimodal distributions (presence/absence signal)
5. 🎭 **Artist frequency is a strong proxy** — well-known artists map to specific genres; log-encoded to dampen popularity bias

### Next Steps
1. Train LightGBM baseline on `X_train` / `y_train` — target > 60% macro-F1
2. Tune with Optuna for LightGBM hyperparameters (num_leaves, learning_rate, min_child_samples)
3. Evaluate on `X_val` with classification report (precision/recall per genre)
4. Investigate confusion matrix — identify most-confused genre pairs for feature engineering round 2
5. Ensemble LightGBM + Random Forest for final test predictions

---

*Report generated from EDA blocks: `data_overview` → `class_distribution_viz` → `correlation_heatmap` → `skewness_outlier_analysis` → `feature_distributions` → `preprocessing_pipeline`*
