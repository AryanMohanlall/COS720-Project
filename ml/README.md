# ML

This folder contains:

- `data/` for datasets
- `models/` for trained model artifacts
- `reports/` for evaluation output
- `train_random_forest.py` for Random Forest training

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ml\requirements.txt
```

## Train

```powershell
python ml\train_random_forest.py
python ml\train_boosting.py --target-recall 0.99
python ml\feature_importance.py --model ml\models\xgboost_model.joblib --repeats 5
```

`--target-recall` tunes the decision threshold from out-of-fold training predictions before the final test evaluation. Lower thresholds usually catch more insiders but will raise false positives.

`feature_importance.py` permutes each raw feature on the held-out test set and measures how much recall, PR-AUC, F1, and false negatives worsen. The output JSON is saved under `ml/reports/`.
