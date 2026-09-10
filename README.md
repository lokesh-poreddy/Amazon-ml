# Amazon ML Challenge 2026
## Competition-Ready Adaptive ML Engineering Framework

> **Team framework for the Amazon ML Challenge 2026.**
> Built to handle an *unknown* dataset — adaptable to any task type,
> metric, or modality the competition releases on Day 1.

---

## Architecture

```
                 UNKNOWN AMAZON DATASET
                         │
                         ▼
                 DATA DISCOVERY         src/data.py :: discover_files()
                         │
                         ▼
                  SCHEMA AUDIT          src/schema.py :: audit_dataset()
                         │
                         ▼
                  TASK DEFINITION       src/config.py :: TaskSpec
                         │
                         ▼
                  VALIDATION PLAN       src/split.py  :: make_splits()
                         │
                         ▼
                     BASELINE           src/train.py  :: cross_validate()
                         │
                         ▼
                FEATURE ENGINEERING     src/features.py
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
           TABULAR     TEXT        IMAGE
              └──────────┼───────────┘
                         │
                         ▼
                   MODEL COMPARISON     src/models.py
                         │
                         ▼
                    ERROR ANALYSIS      src/metrics.py :: error_analysis_regression()
                         │
                         ▼
                   OOF PREDICTIONS      src/train.py  :: cross_validate()
                         │
                         ▼
                      ENSEMBLE          src/ensemble.py
                         │
                         ▼
                  FINAL INFERENCE       src/predict.py
                         │
                         ▼
                 SUBMISSION VALIDATOR   src/submission.py :: write_submission()
                         │
                         ▼
                   FINAL CSV
```

---

## Project Structure

```
amazon-ml-challenge/
│
├── README.md                   ← You are here
├── COMPLIANCE.md               ← Fill from official rules on Day 1
├── requirements.txt
├── environment.yml
├── .gitignore
├── .env.example
│
├── notebooks/
│   ├── 00_schema_audit.ipynb   ← Day 1: Discover and inspect data
│   ├── 01_eda.ipynb            ← Day 1: Understand distributions
│   ├── 02_baseline.ipynb       ← Day 1: First valid model + submission
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_experiments.ipynb
│   ├── 05_validation.ipynb
│   ├── 06_ensemble.ipynb
│   └── 07_final_submission.ipynb
│
├── src/
│   ├── config.py               ← Config loader + TaskSpec
│   ├── data.py                 ← Multi-format loaders + file discovery
│   ├── schema.py               ← Automatic schema audit
│   ├── preprocessing.py        ← Fold-aware transformers + leakage guard
│   ├── features.py             ← Text/numeric/categorical features
│   ├── split.py                ← KFold/Group/Time splits
│   ├── metrics.py              ← SMAPE, MAE, F1, AUC, …
│   ├── models.py               ← LGBM/XGB/CatBoost/Linear/Dummy
│   ├── train.py                ← CV loop + OOF predictions
│   ├── predict.py              ← Test inference
│   ├── ensemble.py             ← OOF-based diversity + blending
│   ├── submission.py           ← Full validation pipeline
│   ├── experiment_tracker.py   ← CSV experiment log
│   ├── logging_utils.py        ← Structured logging
│   └── utils.py                ← seed_everything, device, timing, …
│
├── configs/config.yaml         ← All settings (fill after Day 1)
├── artifacts/                  ← Saved models, encoders, predictions
├── logs/experiments.csv        ← Experiment tracker
├── data/raw/                   ← Competition data (NOT committed)
├── submission/                 ← Validated CSV + metadata
└── tests/                      ← pytest suite
```

---

## Installation

### Option A — pip

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .            # install src as a package
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate amazon-ml-2026
pip install -e .
```

---

## Running the Test Suite

```bash
pytest tests/ -v
```

Run this before **every final submission**.

---

## Day 1 Competition Procedure

### Hour 0–1: Data and Schema

```bash
# 1. Place competition data
cp ~/Downloads/train.csv data/raw/
cp ~/Downloads/test.csv  data/raw/

# 2. Run schema audit (notebook 00)
jupyter lab
# Open 00_schema_audit.ipynb → Run All
```

The schema audit will automatically:
- Discover all files in `data/raw/`
- Classify column types (numeric / text / categorical / datetime / URL / image)
- Detect train/test mismatches
- Flag potential leakage columns
- Suggest probable target and ID columns

### Hour 1–2: Define Task and Update Config

Open `configs/config.yaml` and fill:

```yaml
data:
  train_path: "data/raw/train.csv"
  test_path:  "data/raw/test.csv"
  target_column: "entity_value"   # from problem statement
  id_column:     "index"          # from problem statement

task:
  type:      "regression"         # from problem statement
  metric:    "smape"              # from problem statement
  direction: "minimize"
```

### Hour 2–4: EDA + Baseline

```bash
# Run notebooks in order:
# 01_eda.ipynb     → understand distributions
# 02_baseline.ipynb → first LightGBM + first submission
```

Expected output from the baseline notebook:
- CV SMAPE score
- First submission file with sanity report
- Experiment E001 logged to `logs/experiments.csv`

### Hour 4–12: Feature Engineering Loop

```
Observation → Hypothesis → Experiment → Validation → Decision
```

Every experiment:
1. Adds features in `src/features.py` (or configures feature flags)
2. Runs `cross_validate()` with the new features
3. Compares OOF score to E001
4. Logs result to `logs/experiments.csv` with KEEP / REJECT

### Hour 12–24: Model Experiments

Progress:
```
Dummy → Ridge → LightGBM → XGBoost → CatBoost → Text embeddings → Ensemble
```

Never skip straight to transformers. Validate each step.

### Hour 24–48: Text / Image (if applicable)

Only if:
1. Schema audit identified text / image columns
2. Text structured features show diminishing returns on validation
3. Compute budget allows embedding computation

### Hour 48–66: Ensemble

1. Run `06_ensemble.ipynb`
2. Compare OOF correlation matrix
3. Reject correlated models (> 0.98 OOF correlation)
4. Optimise blend weights via `optimise_weights()`

### Hour 66–72: Final Submission

```bash
# Open 07_final_submission.ipynb
# Set FINAL_RUN = True in configs/config.yaml
# Run All cells
# Verify submission/sanity_report.txt shows PASS
# Submit submission/final_submission.csv
```

---

## Engineering Principles

### Notebooks vs. Modules

| Notebooks | `src/` modules |
|---|---|
| Research / experimentation | Reusable engineering system |
| Import from `src` | Never duplicate notebook logic |
| For human understanding | For correctness and speed |

### Leakage Prevention

All preprocessing transformers (imputers, scalers, encoders, frequency maps)
are fitted ONLY on training data within each CV fold.

```python
# WRONG — fits scaler on full data before splitting
scaler.fit(df)

# CORRECT — fits scaler only on training fold
for si in splits:
    train_fold = df.iloc[si.train_idx]
    val_fold   = df.iloc[si.val_idx]
    scaler.fit(train_fold[numeric_cols])           # ← only train!
    X_train = scaler.transform(train_fold[...])
    X_val   = scaler.transform(val_fold[...])      # ← apply to val
```

### Experiment Discipline

Every significant experiment must answer:
1. **What** did we change?
2. **Why** did we change it?
3. **What** happened?
4. **Keep** or **Reject**?

```python
tracker.log(ExperimentRecord(
    experiment_id="E003",
    hypothesis="TF-IDF on product_name reduces SMAPE",
    model="lgbm",
    feature_version="v2",
    cv_mean=0.391,
    cv_std=0.008,
    decision="KEEP",
))
```

---

## Team Workflow

| Member | Area |
|---|---|
| Member 1 | Data pipeline, schema audit, data quality |
| Member 2 | Feature engineering, classical ML baseline |
| Member 3 | Text embeddings, image pipeline (if needed) |
| Member 4 | Validation, ensemble, submission validation |

**Git branching:**
```
main                  ← stable, always runnable
feature/data-pipeline
feature/text-features
feature/image-pipeline
feature/ensemble
```

**Commit convention:**
```
feat: add automatic schema audit
feat: add fold-aware target encoding
fix: prevent leakage in frequency encoder
perf: cache text embeddings to disk
exp: E005 CatBoost + text features → SMAPE 0.391
```

---

## Final Package Check

```bash
python -c "from src.utils import check_final_package; check_final_package()"
```

Expected output:
```
FINAL PACKAGE STATUS: PASS ✅
```

If any check fails, resolve before submitting.

---

## Key Commands

```bash
# Schema audit
python -c "
from src.data import discover_files
from src.schema import audit_dataset, print_audit
from src.data import load_csv
files = discover_files('data/raw')
df = load_csv(files.probable_train[0])
report = audit_dataset(df, 'train')
print_audit(report)
"

# Run tests
pytest tests/ -v --tb=short

# Check final package
python -c "from src.utils import check_final_package; check_final_package()"

# View experiment log
python -c "
from src.experiment_tracker import get_tracker
t = get_tracker()
print(t.comparison_table().to_string())
"
```

---

## Reproducibility

Every experiment records:
- `seed` (set via `seed_everything(42)`)
- `feature_version`
- `split_version`
- `model` name + `params`
- `timestamp`

To reproduce an experiment from scratch:
1. Checkout the same git commit
2. Verify the same `feature_version` in `configs/config.yaml`
3. Run `python -m src.train` (or the corresponding notebook cell)
4. Compare `cv_mean` and `cv_std` from `logs/experiments.csv`

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `FileNotFoundError: data/raw/train.csv` | Place competition data in `data/raw/` |
| `Target column 'X' not found` | Set `data.target_column` in `config.yaml` |
| `LightGBM not installed` | `pip install lightgbm` |
| `TaskSpec has unresolved fields` | Fill `task.type`, `task.metric` in config after reading problem statement |
| Submission NaN check fails | Check `postprocess_predictions(clip_min=...)` |
| OOF correlation > 0.98 | Models are nearly identical — use only the better one |

---

*"The best model is not necessarily the biggest model."*
*Start simple. Validate rigorously. Improve incrementally.*
