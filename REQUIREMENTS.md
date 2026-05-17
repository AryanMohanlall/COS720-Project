## 3. Functional Requirements

### FR1 — Data Ingestion and Preprocessing

**FR1.1** The system shall load training data exclusively from the prescribed Kaggle dataset.  
**FR1.2** The system shall handle missing values via a documented strategy (blanks → `None`; absent numeric keys → 0 via `DictVectorizer`).  
**FR1.3** The system shall apply one-hot encoding to all categorical features without train/test leakage.  
**FR1.4** The system shall remove zero-variance features and record them in the model artefact.  
**FR1.5** The system shall deduplicate rows before the train/test split.

### FR2 — Feature Selection and Behavioural Indicators

**FR2.1** The feature set shall cover login time patterns (e.g. `entry_during_weekend`, `num_entries`).  
**FR2.2** The feature set shall cover file access activity (`total_files_burned`, `burned_from_other`).  
**FR2.3** The feature set shall cover privilege usage (`employee_classification`).  
**FR2.4** The feature set shall cover abnormal access frequency (`num_entries`, `num_unique_campus`).  
**FR2.5** The feature set shall cover data transfer behaviour (`total_printed_pages`, `num_printed_pages_off_hours`).  
**FR2.6** Each feature's relevance shall be documented in `ml/FEATURES.md`.

### FR3 — Model Training

**FR3.1** The system shall train an XGBoost binary classifier with Optuna-tuned hyperparameters.  
**FR3.2** The system shall train a LightGBM binary classifier with Optuna-tuned hyperparameters.  
**FR3.3** The system shall train a Random Forest binary classifier with Optuna-tuned hyperparameters.  
**FR3.4** Each model shall save a self-contained `.joblib` artefact (model, vectorizer, threshold, constant columns, numeric columns).  
**FR3.5** The choice of tree-based ensembles shall be justified in `MODELS.md` with reference to tabular data, class imbalance, and SHAP compatibility.

### FR4 — Class Imbalance Handling

**FR4.1** XGBoost shall use `scale_pos_weight` set to the negative-to-positive ratio (≈17.78).  
**FR4.2** LightGBM shall use `is_unbalance=True`.  
**FR4.3** Random Forest shall use `class_weight="balanced_subsample"`.  
**FR4.4** Hyperparameter tuning shall optimise Average Precision (PR-AUC), not accuracy.

### FR5 — Decision Threshold Selection

**FR5.1** The threshold shall be selected using only out-of-fold predictions from the training set (no test-set leakage).  
**FR5.2** The selection rule shall target recall ≥ 0.99 and maximise precision among qualifying thresholds.  
**FR5.3** The selected threshold, selection rule, and OOF sweep shall be persisted in the metrics report.

### FR6 — Evaluation and Metrics Reporting

**FR6.1** The system shall report accuracy, precision, recall, F1, ROC-AUC, and PR-AUC on the held-out test set.  
**FR6.2** The system shall report the full confusion matrix (TP, TN, FP, FN counts).  
**FR6.3** The system shall report metrics at both the tuned threshold and at threshold = 0.5.  
**FR6.4** The system shall persist a threshold sweep (≥ 9 thresholds) for both OOF and test predictions.  
**FR6.5** The dashboard shall display all three models' metrics simultaneously for direct comparison.

### FR7 — Prototype Input

**FR7.1** The prototype shall accept CSV upload, parse the header row, and populate the form with the first data row's values.  
**FR7.2** The prototype shall provide at least three pre-populated sample employee profiles selectable from a dropdown (e.g. low-risk, high-risk, confirmed malicious).  
**FR7.3** Unrecognised CSV columns shall be ignored without failing the upload.

### FR8 — Prototype Output

**FR8.1** Each prediction shall display the binary classification label (Malicious / Not Malicious).  
**FR8.2** Each prediction shall display the model's probability as a percentage.  
**FR8.3** Each prediction shall display the decision threshold used.  
**FR8.4** Each prediction shall display a one-paragraph plain-English summary of the top risk drivers.  
**FR8.5** Each prediction shall display a ranked list of contributing features with direction (increases / decreases risk).

### FR9 — Explainability

**FR9.1** The system shall compute global SHAP explanations per model (bar chart, beeswarm, JSON summary).  
**FR9.2** Encoded-column SHAP values shall be aggregated back to raw features for interpretability.  
**FR9.3** Every API prediction shall include per-record SHAP contributions (top 10 raw features, all raw features, top 20 encoded features).  
**FR9.4** Each contribution shall include the feature name, input value, SHAP value, absolute SHAP value, and risk direction.  
**FR9.5** The non-technical explanation view shall hide raw SHAP numbers and use qualitative labels ("Low / Moderate / Strong impact").  
**FR9.6** Technical SHAP details shall be available in a collapsible "Technical details" section for analysts.

### FR10 — Testing and Failure Analysis

**FR10.1** The system shall evaluate each model on a 20% stratified held-out test set.  
**FR10.2** The system shall document TP, FP, and FN counts at the selected threshold.  
**FR10.3** The system shall isolate misclassified records (FP and FN) and produce a failure analysis report.  
**FR10.4** The failure analysis shall include per-feature descriptive statistics and SHAP contributions for misclassified subsets.  
**FR10.5** The failure analysis shall propose concrete improvements based on observed patterns.

---

## 4. Non-Functional Requirements

### NFR1 — Reproducibility

**NFR1.1** All random operations shall use fixed seeds (`random_state=42`, `seed=42`).  
**NFR1.2** All ML dependencies shall be pinned in `ml/requirements.txt`.  
**NFR1.3** Frontend dependencies shall be locked via `package-lock.json`.

### NFR2 — Performance

**NFR2.1** Single-record prediction (including SHAP) shall complete in under 2 seconds with the model loaded in memory.  
**NFR2.2** A latency benchmark script shall report mean, p50, and p95 over 100 sequential requests.

### NFR3 — Usability

**NFR3.1** The non-technical explanation panel shall be interpretable without ML background.  
**NFR3.2** The panel shall avoid statistical jargon in the primary summary.  
**NFR3.3** Risk direction shall be labelled "increases risk" / "decreases risk" rather than using sign conventions.

### NFR4 — Forensic Soundness

**NFR4.1** Every prediction response shall include a non-empty explanation list.  
**NFR4.2** Model artefacts shall record provenance: model name, vectorizer, threshold, constant columns, numeric columns.  
**NFR4.3** The system shall align with ISO/IEC 27043 evidential integrity principles.

### NFR5 — Modularity

**NFR5.1** Each model shall be independently trainable, replaceable, and servable.  
**NFR5.2** The backend shall select models by URL path parameter (`xgboost`, `lightgbm`, `random_forest`).

### NFR6 — Auditability

**NFR6.1** Each prediction request shall be logged with model name, timestamp, input hash, probability, threshold, and label.  
**NFR6.2** Logs shall be persisted to a file mountable as a Docker volume.

---

## 5. Status Summary

| Area | Implemented | Partial | Missing |
|---|---|---|---|
| Data ingestion (FR1) | FR1.3, FR1.4 | FR1.1, FR1.2 | FR1.5 |
| Features (FR2) | FR2.2, FR2.4, FR2.5 | FR2.1, FR2.3 | FR2.6 |
| Training (FR3) | FR3.1, FR3.2, FR3.3, FR3.4 | — | FR3.5 |
| Imbalance (FR4) | FR4.1, FR4.2, FR4.3, FR4.4 | — | — |
| Threshold (FR5) | FR5.1, FR5.2, FR5.3 | — | — |
| Evaluation (FR6) | FR6.1–FR6.5 | — | — |
| Prototype input (FR7) | FR7.1, FR7.3 | FR7.2 | — |
| Prototype output (FR8) | FR8.1–FR8.4 | FR8.5 | — |
| Explainability (FR9) | FR9.1–FR9.4 | — | FR9.5, FR9.6 |
| Testing (FR10) | FR10.1, FR10.2 | — | FR10.3, FR10.4, FR10.5 |
| Reproducibility (NFR1) | NFR1.1, NFR1.3 | NFR1.2 | — |
| Performance (NFR2) | — | — | NFR2.1, NFR2.2 |
| Usability (NFR3) | NFR3.3 | NFR3.1, NFR3.2 | — |
| Forensic (NFR4) | NFR4.1, NFR4.2, NFR4.3 | — | — |
| Modularity (NFR5) | NFR5.1, NFR5.2 | — | — |
| Auditability (NFR6) | — | — | NFR6.1, NFR6.2 |