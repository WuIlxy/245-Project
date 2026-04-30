# Complete or Collapse? Classifying Clinical Trial Outcomes

Predicting whether a clinical trial will be **Completed**, **Terminated**, or **Withdrawn** using trial design, sponsor characteristics, and FDA drug safety signals.

**Presentation:** [Video](https://drive.google.com/file/d/1HZONqkQukjSA9F9dGIcUlfuMU1ukdyFY/view?usp=sharing) | [Slides](CIS2450_FinalProject_Presentation.pdf)

## Group Members

**Zheng**

- Data collection and integration (ClinicalTrials.gov API pipeline)
- Data cleaning and preprocessing (ClinicalTrials.gov)
- Feature engineering
- Co-lead on model development
- Feature importance and explainability

**Andrew**

- Data collection and integration (OpenFDA API pipeline)
- Data cleaning and preprocessing (OpenFDA)
- Exploratory data analysis and visualization
- Co-lead on model development
- Dashboard development

## Data Sources

**ClinicalTrials.gov API (v2)** - U.S. government registry of 400,000+ clinical studies maintained by the National Library of Medicine. Provides structured metadata on trial phase, study type, intervention model, masking, primary purpose, enrollment count, sponsor class, number of sites, eligibility criteria, and trial timeline. The `overallStatus` field serves as the target label.

**OpenFDA API** - Maintained by the FDA, provides regulatory and safety information on approved drugs and biologics via the Drugs@FDA database. Fields include application type (NDA/ANDA/BLA), marketing status, therapeutic drug class (EPC), mechanism of action, approval year, and route of administration.

**Data Integration** - The two sources are joined on drug/intervention name using string normalization and fuzzy matching (token-sort ratio >= 85 via rapidfuzz). Trials with no FDA match are retained with a `has_fda_record = False` flag and null-filled FDA features. Final dataset: **53,628 trials x 150 features**.

## Objective

Predict the outcome of a clinical trial - Completed, Terminated, or Withdrawn - using design characteristics, sponsor profile, and drug-level regulatory signals available at or before trial start.

Clinical trials cost hundreds of millions of dollars per Phase 3 study. Identifying which design choices and drug profiles predict failure could benefit pharmaceutical companies, research institutions, regulatory agencies, and investors.

## Modeling Approach

- **Target:** 3-class outcome label (0 = Completed, 1 = Terminated, 2 = Withdrawn)
- **Split:** Temporal - train on trials starting before 2019, test on 2019 and later
- **Baseline:** Logistic Regression with `class_weight='balanced'` and StandardScaler - **Weighted F1: 0.520**
- **Random Forest:** Weighted F1: 0.612
- **XGBoost:** Weighted F1: 0.582
- **Models:** Logistic Regression -> Random Forest -> XGBoost (progressively more expressive)
- **XGBoost tuning:** Grid search, random search, and Bayesian tuning with Optuna on an inner temporal validation split
- **Evaluation metric:** Weighted F1 (accounts for class frequency; primary comparison metric)
- **Interpretability:** Feature importance + SHAP values
- **Leakage audit:** Tree models remove `trial_duration_days`, `enrollment_actual`, and `log_enrollment` - fields that can encode post-start information

## Why These Models?

> Each model was selected for a specific reason, not just tried at random. The progression from linear to ensemble to boosted reflects deliberate choices about the data's structure and the problem's constraints.

| Model | Why it was chosen |
|---|---|
| **Logistic Regression** | Fast, interpretable linear baseline. `class_weight='balanced'` handles the 53/37/11% class skew. StandardScaler required since LR is scale-sensitive. Sets the floor any more complex model must beat. |
| **Random Forest** | First non-linear step. Scale-invariant, handles mixed OHE + continuous features naturally, and bagging reduces variance on noisy real-world data. `class_weight='balanced_subsample'` rebalances per tree, which is more correct under bootstrap sampling than a global weight. |
| **XGBoost** | Primary model. Boosting sequentially corrects residual errors, making it especially effective when signal is spread across many weak features - a common property of clinical trial metadata. Handles sparse OHE efficiently, supports sample weighting for imbalance, and consistently outperforms RF on tabular benchmarks. Three complementary tuning strategies (grid, random, Bayesian) explore the hyperparameter space with increasing efficiency. |

## Notebooks

Each notebook contains detailed inline comments explaining every design decision, preprocessing step, and modeling choice - including why each transformation was applied, what alternatives were considered, and what the output means in context.

| Notebook                             | Description                                                          |
| ------------------------------------ | -------------------------------------------------------------------- |
| `CTAPI.ipynb`                        | ClinicalTrials.gov API data collection - pagination strategy, field selection rationale, and rate-limit handling |
| `CTDataEDA.ipynb`                    | CT data preprocessing and EDA - null audit, outlier analysis, distribution plots, and feature-level commentary |
| `OpenFDA_Data_Collection.ipynb`      | OpenFDA API data collection - endpoint selection, response parsing, and deduplication logic |
| `OpenFDA_Preprocess_EDA.ipynb`       | OpenFDA preprocessing and EDA - field normalization, EPC/mechanism parsing, and coverage analysis |
| `Join_and_Preprocess.ipynb`          | DuckDB join, fuzzy matching, feature engineering, and hypothesis testing - includes justification for the fuzzy threshold choice (85) and statistical test design |
| `Joined_EDA_and_Preprocessing.ipynb` | Joined dataset EDA, imputation strategy, one-hot encoding, and export to model-ready parquet - each imputation choice is explained per feature |
| `Baseline.ipynb`                     | Logistic regression baseline - scaling rationale, class weight motivation, and interpretation of coefficients |
| `Random_Forest_Model.ipynb`          | Random Forest training and evaluation - hyperparameter choices explained, feature importance discussion |
| `XGBoost_Model.ipynb`                | XGBoost training and evaluation - parameter rationale, comparison against baseline and RF, confusion matrix analysis |
| `xgboost_hyperparameter_tuning.py`   | XGBoost grid, random, and Bayesian (Optuna) hyperparameter tuning - search space design and strategy comparison |
| `Explainability_Analysis.ipynb`      | Feature importance, permutation importance, and SHAP summary plots - explains which features drive each outcome class |

## XGBoost Hyperparameter Tuning

Run all three tuning methods after preprocessing:

```powershell
python xgboost_hyperparameter_tuning.py
```

The script trains candidate models on older pre-2019 trials, validates on the later pre-2019 slice, and keeps the 2019+ temporal test set untouched until the best tuned model is refit. It writes `artifacts/xgboost_tuning_results.csv`, `artifacts/xgboost_tuning_summary.json`, and `artifacts/models/xgboost_tuned.joblib`.

## Dashboard

Install dependencies, run the model notebooks, then start the dashboard:

```powershell
pip install -r requirements.txt
python dashboard.py
```

Open `http://127.0.0.1:8050` in a browser.

## Key Challenges

- **Partial join coverage** - OpenFDA only covers drugs/biologics; device and behavioral trials have no FDA record. Retained with `has_fda_record` flag.
- **Class imbalance** - ~53% Completed / 37% Terminated / 11% Withdrawn. Addressed with `class_weight='balanced'` and weighted F1 evaluation across all three models.
- **Drug name matching** - Brand vs. generic names, combination therapies. Solved with fuzzy matching at threshold 85.
- **Label ambiguity** - Statuses like Suspended and Unknown excluded; only terminal statuses retained.
- **Leak features** - Presence of leaky features such as `log_enrollment` that dominated predictions and unfeasible in real employment. Selectively removed during model training.
