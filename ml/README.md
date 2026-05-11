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
```
