# ML

This folder contains:

- `data/` for datasets
- `models/` for local trained model copies
- `reports/` for evaluation output
- `train_random_forest.py` for Random Forest training
- `train_xgboost.py` for XGBoost training
- `train_lightgbm.py` for LightGBM training

## Dataset and selected features

The prescribed dataset is the Kaggle *Insider Threat Dataset for Corporate Environments*: 118,614 employee-day records with an `is_malicious` label. The raw CSV has 21 behavioural and contextual source features. During preprocessing, `late_exit_flag` is removed as a zero-variance column, leaving 20 raw features and 110 one-hot encoded columns.

The retained features cover login timing, file-burning activity, privilege and role context, abnormal access frequency, data-transfer behaviour, travel/background context, and counter-intelligence risk context. See `../techdoc.md` for the behavioural-category table and XGBoost permutation-importance summary.

## Primary model

XGBoost is the primary classifier. At the tuned operating threshold it gives the best workload trade-off among the three trained models: 98.7% recall, 34.2% precision, 50.8% F1, 86.2% PR-AUC, and 2,426 false positives on the 23,723-row held-out test set. LightGBM remains the closest boosting comparator, while Random Forest remains the bagging-family comparator.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ml\requirements.txt
```

## Train

```powershell
python ml\train_random_forest.py
python ml\train_xgboost.py --target-recall 0.99
python ml\train_lightgbm.py --target-recall 0.99
python ml\feature_importance.py --model ml\models\xgboost_model.joblib --repeats 5
```

Training also writes backend-ready artifacts to `artifacts/models/`:

- `artifacts/models/random_forest_model.joblib`
- `artifacts/models/xgboost_model.joblib`
- `artifacts/models/lightgbm_model.joblib`

`--target-recall` tunes the decision threshold from out-of-fold training predictions before the final test evaluation. Lower thresholds usually catch more insiders but will raise false positives.

`feature_importance.py` permutes each raw feature on the held-out test set and measures how much recall, PR-AUC, F1, and false negatives worsen. The output JSON is saved under `ml/reports/`.
