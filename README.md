# Complete or Collapse? Classifying Clinical Trial Outcomes

Predicting whether a clinical trial will be **Completed**, **Terminated**, or **Withdrawn** using trial design, sponsor characteristics, and FDA drug safety signals.

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
- **Baseline:** Logistic regression with `class_weight='balanced'` and StandardScaler on numeric features - **Macro F1: 0.805**
- **Models:** Logistic Regression baseline, Random Forest, XGBoost
- **Evaluation metric:** Macro F1 (treats all three classes equally regardless of frequency)
- **Interpretability:** Feature importance + SHAP values
- **Leakage audit:** Tree models remove `trial_duration_days`, `enrollment_actual`, and `log_enrollment` from the final saved feature set because these fields can encode information not available near trial start.

## Notebooks

| Notebook                             | Description                                                          |
| ------------------------------------ | -------------------------------------------------------------------- |
| `CTAPI.ipynb`                        | ClinicalTrials.gov API data collection                               |
| `CTDataEDA.ipynb`                    | CT data preprocessing and EDA                                        |
| `OpenFDA_Data_Collection.ipynb`      | OpenFDA API data collection                                          |
| `OpenFDA_Preprocess_EDA.ipynb`       | OpenFDA preprocessing and EDA                                        |
| `Join_and_Preprocess.ipynb`          | DuckDB join, fuzzy matching, feature engineering, hypothesis testing |
| `Joined_EDA_and_Preprocessing.ipynb` | Joined dataset EDA, imputation, OHE, model-ready parquet             |
| `Baseline.ipynb`                     | Logistic regression baseline                                         |
| `Random_Forest_Model.ipynb`          | Random Forest training and metrics                                   |
| `XGBoost_Model.ipynb`                | XGBoost training and metrics                                         |
| `Explainability_Analysis.ipynb`      | Feature importance, permutation importance, and SHAP summaries       |

## Dashboard

Install dependencies, run the model notebooks, then start the dashboard:

```powershell
pip install -r requirements.txt
python dashboard.py
```

Open `http://127.0.0.1:8050` in a browser.

## Key Challenges

- **Partial join coverage** - OpenFDA only covers drugs/biologics; device and behavioral trials have no FDA record. Retained with `has_fda_record` flag.
- **Class imbalance** - ~53% Completed / 37% Terminated / 11% Withdrawn. Addressed with `class_weight='balanced'` and macro F1 evaluation across all three models.
- **Drug name matching** - Brand vs. generic names, combination therapies. Solved with fuzzy matching at threshold 85.
- **Label ambiguity** - Statuses like Suspended and Unknown excluded; only terminal statuses retained.
- **Leak features** - Presence of leaky features such as `log_enrollment` that dominated predictions and unfeasible in real employment. Selectively removed during model training.
