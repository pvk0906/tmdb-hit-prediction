# Predicting Movie Profitability from Pre-Release Metadata

> A supervised binary classification project using the TMDB All Movies dataset.
> Submission 1 — Machine Learning assignment.

This repository contains the proposal, code, and outputs for predicting whether a movie will be a financial **hit** (revenue ≥ 2× budget) using only pre-release metadata such as budget, runtime, genre, language, production scale, and release timing.

---

## Project at a glance

| Field | Value |
|---|---|
| **Task** | Binary Classification (Supervised Learning) |
| **Target** | `hit = 1 if revenue ≥ 2 × budget, else 0` |
| **Dataset** | [TMDB All Movies on Kaggle](https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates) |
| **Raw size** | 1,188,481 rows × 28 columns |
| **Modeling subset** | 13,750 rows × 21 engineered features |
| **Class balance** | 56% flop / 44% hit |
| **Best F1-Score** | XGBoost (F1 = 0.641, Acc = 0.700) |
| **Best ROC-AUC** | Random Forest (AUC = 0.761) |
| **Research questions** | 7 |

---

## Repository structure

```
.
├── Proposal.docx            # Word version of the proposal
├── Proposal.pdf             # PDF version of the proposal
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── notebooks/
│   ├── RQ1.ipynb            # Baseline model comparison
│   ├── RQ2.ipynb            # Feature importance analysis
│   ├── RQ3.ipynb            # Performance by budget tier
│   ├── RQ4.ipynb            # Genre-specific success patterns
│   ├── RQ5.ipynb            # Temporal stability of predictions
│   ├── RQ6.ipynb            # Threshold optimization
│   └── RQ7.ipynb            # Error analysis on hard-to-classify movies
├── figures/
│   ├── fig_rq1_model_comparison.pdf      # + .png preview
│   ├── fig_rq2_feature_importance.pdf
│   ├── fig_rq3_budget_tiers.pdf
│   ├── fig_rq4_genre_analysis.pdf
│   ├── fig_rq5_temporal_stability.pdf
│   ├── fig_rq6_threshold_optimization.pdf
│   └── fig_rq7_error_analysis.pdf
└── tables/
    ├── table_rq1_model_comparison.csv
    ├── table_rq2_feature_importance.csv
    ├── table_rq3_budget_tiers.csv
    ├── table_rq4_genre_analysis.csv
    ├── table_rq5_temporal_stability.csv
    ├── table_rq6_threshold_optimization.csv
    └── table_rq7_error_analysis.csv
```

---

## Dataset

**Source:** https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates

The notebooks **auto-detect** whether they are running on Kaggle or locally:

- **On Kaggle:** the dataset is mounted at `/kaggle/input/tmdb-movies-daily-updates/TMDB_all_movies.csv` — no manual download needed. Just attach the dataset to your Kaggle notebook session.
- **Locally:** download `TMDB_all_movies.csv` from the Kaggle link above and place it in the same directory as the notebooks.

---

## How to run

### Option A: Kaggle (recommended)

1. Go to https://www.kaggle.com and create a new notebook.
2. Attach the dataset: `Add Data` → search for "TMDB Movies Dataset Daily Updates" → add to notebook.
3. Upload one of the `RQ*.ipynb` files.
4. Click **Run All**. Outputs (figures and tables) are saved to `/kaggle/working/`.

### Option B: Local

```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Place the TMDB CSV in the notebooks/ folder
# (download from https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates)
mv ~/Downloads/TMDB_all_movies.csv notebooks/

# Launch Jupyter
cd notebooks
jupyter lab
```

Then open and run any `RQ*.ipynb`. They are self-contained — each notebook loads the raw CSV, performs feature engineering, and produces its own figure and table.

---

## Research questions

| RQ | Question | Output |
|----|----------|--------|
| **RQ1** | How do classical supervised learning algorithms compare in predicting hits? | Figure 1.1 + table_rq1 |
| **RQ2** | Which pre-release features contribute most to predicting profitability? | Figure 2.1 + table_rq2 |
| **RQ3** | Does predictability vary across budget tiers? | Figure 3.1 + table_rq3 |
| **RQ4** | Which genres are most/least predictable? | Figure 4.1 + table_rq4 |
| **RQ5** | How stable is performance across time periods? | Figure 5.1 + table_rq5 |
| **RQ6** | What's the optimal decision threshold? | Figure 6.1 + table_rq6 |
| **RQ7** | What characterises misclassified movies? | Figure 7.1 + table_rq7 |

---

## Models compared

| Model | Library | Family |
|---|---|---|
| Logistic Regression | scikit-learn | Linear |
| SVM (RBF kernel) | scikit-learn | Kernel |
| Random Forest | scikit-learn | Bagging |
| Gradient Boosting | scikit-learn | Boosting |
| XGBoost | xgboost | Boosting |

## Evaluation metrics

Accuracy, Precision, Recall, F1-Score, and ROC-AUC are reported for every experiment.

---

## Headline findings

- **Tree-based ensembles dominate:** XGBoost wins on Accuracy (0.700), F1-Score (0.641), and Recall (0.617); Random Forest leads on ROC-AUC (0.761). All three tree ensembles substantially outperform Logistic Regression and SVM.
- **Engagement signals dominate feature importance** — `log_popularity` and `log_budget` are the top two predictors.
- **Blockbusters are the most predictable tier** (F1 = 0.859) — high-budget films exhibit clearer outcome patterns.
- **Horror has the highest hit rate** (53.4%) — confirms the low-budget high-ROI hypothesis.
- **Performance drops for recent test periods** — concept drift exists; the streaming era has changed what predicts theatrical profitability.
- **The default 0.50 threshold is suboptimal** — F1 peaks closer to 0.30–0.40, biasing toward recall.
- **False positives are big-budget flops** and **false negatives are sleeper hits** — the model fails on outliers in both directions.

---

## Reproducibility

- All notebooks use `random_state=42` for splits and model initialisation.
- The same feature engineering pipeline is applied identically in every notebook.
- No external API calls — only the static Kaggle CSV is required.

---

## License

Code released under MIT License. Dataset belongs to its original Kaggle uploader and TMDB.
