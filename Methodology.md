# Methodology: Step-by-step pipeline for each Research Question

This document describes the complete computational pipeline that takes the raw TMDB CSV and produces the final figure (PDF) and table (CSV) for each of the 7 Research Questions.

All seven notebooks share **Steps 1–4** (loading, filtering, target definition, feature engineering). They diverge at **Step 5** based on what each RQ asks.

---

## Shared steps (all notebooks)

### Step 1 — Load raw dataset
```python
df = pd.read_csv("TMDB_all_movies.csv", low_memory=False)
# Shape: (1,188,481, 28)
```

### Step 2 — Filter to modeling subset
```python
m = df[(df["budget"] > 1000) & (df["revenue"] > 1000)].copy()
# Shape: (13,750, 28)
```

### Step 3 — Define binary target
```python
m["hit"] = (m["revenue"] >= 2 * m["budget"]).astype(int)
# Class balance: 56% flop, 44% hit
```

### Step 4 — Engineer 21 features
- `log_budget = log(1 + budget)`
- `log_popularity = log(1 + popularity)`
- `runtime` (raw)
- `release_year`, `release_month` (from `release_date`)
- `num_genres`, `num_production_companies`, `num_cast` (counted)
- `is_english` (binary)
- 12 binary genre indicators (`genre_drama`, `genre_comedy`, …)

---

## RQ1 — Baseline Model Comparison

### Step 5 — Train-test split
- Stratified 80/20 split, `random_state=42`.

### Step 6 — Train 5 classifiers
- Logistic Regression and SVM use `StandardScaler`-transformed features.
- Random Forest, Gradient Boosting, XGBoost use raw features.
- Each model is fit with default hyperparameters.

### Step 7 — Compute metrics on held-out test set
- For each model: Accuracy, Precision, Recall, F1, ROC-AUC.
- Save 5×6 results table → `table_rq1_model_comparison.csv`.

### Step 8 — Visualise
- Grouped bar chart with 5 model groups × 5 coloured metric bars.
- Save → `fig_rq1_model_comparison.pdf` and `.png`.

---

## RQ2 — Feature Importance

### Step 5 — Train the strongest model from RQ1 (XGBoost or GBM fallback)
- Same train/test split as RQ1.

### Step 6 — Extract feature importances
- Use `model.feature_importances_` (gain-based for tree ensembles).
- Sort descending, keep top 10.

### Step 7 — Categorise features
- Map each feature to a category: Financial / Engagement / Production / Content / Timing / Language / Genre.
- Save → `table_rq2_feature_importance.csv`.

### Step 8 — Visualise
- Horizontal bar chart, bars coloured by category.
- Save → `fig_rq2_feature_importance.pdf`.

---

## RQ3 — Performance by Budget Tier

### Step 5 — Assign tiers
- Low (< $1M), Mid ($1M–$20M), High ($20M–$100M), Blockbuster (> $100M).

### Step 6 — Train one global model on the full train set
- Same XGBoost/GBM as RQ2.

### Step 7 — Evaluate per tier on the test set
- Subset the test set by tier and compute per-tier metrics.
- Save → `table_rq3_budget_tiers.csv`.

### Step 8 — Visualise (two-panel figure)
- Panel (a): Bar chart of hit rate per tier + line plot of dataset size per tier (twin axis).
- Panel (b): Grouped bars of F1 and ROC-AUC per tier.
- Save → `fig_rq3_budget_tiers.pdf`.

---

## RQ4 — Genre Analysis

### Step 5 — Define primary genre
- `primary_genre = first comma-separated value in the genres field`.

### Step 6 — Train one global model
- Same setup as RQ2/RQ3.

### Step 7 — Evaluate per genre
- For each genre with ≥ 100 movies, compute hit rate, F1, accuracy on the test subset.
- Save → `table_rq4_genre_analysis.csv`.

### Step 8 — Visualise (bubble plot)
- X-axis: hit rate. Y-axis: F1. Bubble size: number of movies in that genre.
- Reference lines mark the overall hit rate and mean F1.
- Save → `fig_rq4_genre_analysis.pdf`.

---

## RQ5 — Temporal Stability

### Step 5 — Define forward-in-time splits
- Three (train, test) windows: (1980–2000, 2000–2010), (1990–2010, 2010–2020), (2000–2015, 2015–2025).

### Step 6 — Train and evaluate each split
- Train a fresh model on each train window, evaluate on the corresponding test window.
- Save → `table_rq5_temporal_stability.csv`.

### Step 7 — Compute year-by-year AUC
- Rolling 10-year training window for each evaluation year (1995 → 2024).
- Train on the prior 10 years, evaluate on the focal year.

### Step 8 — Visualise (two-panel figure)
- Panel (a): Grouped bar chart of accuracy/F1/AUC for the three forward-in-time splits.
- Panel (b): Year-by-year AUC scatter + smoothed trend line, with the streaming era highlighted.
- Save → `fig_rq5_temporal_stability.pdf`.

---

## RQ6 — Threshold Optimization

### Step 5 — Train the best model
- Same setup as RQ2.

### Step 6 — Sweep decision threshold
- For each `τ ∈ [0.05, 0.95]` step 0.01, compute precision, recall, F1.
- Identify the threshold that maximises F1.

### Step 7 — Operating-point table
- For τ ∈ {0.30, 0.40, 0.50, 0.60, 0.70}: report precision, recall, F1, predicted hits, true positives, false positives.
- Save → `table_rq6_threshold_optimization.csv`.

### Step 8 — Visualise (two-panel figure)
- Panel (a): Precision-Recall curve with the 5 operating points marked.
- Panel (b): Three lines (precision, recall, F1) vs threshold, with the optimal τ highlighted.
- Save → `fig_rq6_threshold_optimization.pdf`.

---

## RQ7 — Error Analysis

### Step 5 — Train the best model and predict on the test set

### Step 6 — Compute confusion matrix
- 2×2 matrix of TP, TN, FP, FN counts.

### Step 7 — Profile each error quadrant
- For each of TP / TN / FP / FN, compute the average value of: budget, runtime, popularity, vote_count, num_genres, num_production_companies.
- Save → `table_rq7_error_analysis.csv`.

### Step 8 — Visualise (two-panel figure)
- Panel (a): Confusion matrix heatmap with cell counts and percentages.
- Panel (b): Radar chart with 4 polygons (TP/TN/FP/FN) showing the relative average of 5 features.
- Save → `fig_rq7_error_analysis.pdf`.

---

## Verification

Each notebook also writes a PNG version of its figure for quick preview. The CSV tables are formatted for direct inclusion in a LaTeX or Word report.

All seven notebooks are independent — running RQ4 does not require running RQ1 first. Each notebook is self-contained and reproducible with `random_state=42`.
