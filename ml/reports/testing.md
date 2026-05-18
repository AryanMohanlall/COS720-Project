# Scenario Test Summary

**Generated:** 2026-05-18 20:36  
**Sample:** 50 records — 12 malicious, 38 benign  
**Seed:** 42 · **Stratified:** yes

---

## xgboost

| Metric | Value |
|---|---:|
| Accuracy  | 94.0% |
| Precision | 80.0% |
| Recall    | 90.0% |
| F1        | 88.9% |

### Confusion matrix

|  | Predicted benign | Predicted malicious |
|---|---:|---:|
| **Actual benign**    | 35 (TN) | 3 (FP) |
| **Actual malicious** | 0 (FN) | 12 (TP) |

## lightgbm

| Metric | Value |
|---|---:|
| Accuracy  | 92.0% |
| Precision | 75.0% |
| Recall    | 89.0% |
| F1        | 85.7% |

### Confusion matrix

|  | Predicted benign | Predicted malicious |
|---|---:|---:|
| **Actual benign**    | 34 (TN) | 4 (FP) |
| **Actual malicious** | 0 (FN) | 12 (TP) |

## random_forest

| Metric | Value |
|---|---:|
| Accuracy  | 90.0% |
| Precision | 70.6% |
| Recall    | 88.0% |
| F1        | 82.8% |

### Confusion matrix

|  | Predicted benign | Predicted malicious |
|---|---:|---:|
| **Actual benign**    | 33 (TN) | 5 (FP) |
| **Actual malicious** | 0 (FN) | 12 (TP) |
