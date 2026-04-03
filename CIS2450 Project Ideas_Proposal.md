**CIS 2450 Project Proposal**

**Title:** *Complete or Collapse? Classifying Clinical Trial Outcomes as Completed, Terminated, or Withdrawn Based on Trial Design, Sponsor Characteristics, and Drug Safety Signals*

**Section 1: Group Members and Initial Work Allocation**

Zheng

* Data collection and integration (ClinicalTrials.gov API pipeline)  
* Data cleaning and preprocessing (ClinicalTrials.gov)  
* Feature engineering  
* Co-lead on model development  
* Feature importance and explainability

Andrew

* Data collection and integration (OpenFDA API pipeline)  
* Data cleaning and preprocessing (OpenFDA)  
* Exploratory data analysis and visualization  
* Co-lead on model development  
* Dashboard development

**Section 2: Data Source**

We will use two distinct web APIs to construct our dataset:

**Source 1: ClinicalTrials.gov API (v2)** ClinicalTrials.gov is the U.S. government's official registry of clinical research studies, maintained by the National Library of Medicine. Its API is completely free, requires no key, and imposes no rate limit. It provides structured metadata on 400,000+ registered trials including trial phase, study type, intervention model, masking, primary purpose, enrollment count, number of arms, sponsor class, number of collaborators, number of study sites, eligibility criteria, conditions studied, intervention type, and trial timeline (start date, primary completion date). The trial's *overallStatus* field serves as our target label, providing a direct, objective classification of each trial's outcome.

**Source 2: OpenFDA API** The OpenFDA API is maintained by the U.S. Food and Drug Administration and is completely free with a generous rate limit of 240 requests per minute. It provides regulatory and safety information on drugs and biologics, including application type, marketing status, therapeutic drug class, regulatory action type, approval year, and adverse event counts (total, serious, and death-related) sourced from the FDA Adverse Event Reporting System (FAERS). These fields enrich each drug-intervention trial with regulatory context and real-world safety signals that are independent of the trial's own design.

**Data Integration** 

The two sources will be joined on the intervention/drug name field, which is present in both APIs. We will apply string normalization to handle minor naming inconsistencies. Since OpenFDA only covers drugs and biologics, trials testing devices or behavioral interventions will not have a matching FDA record. These rows will either be retained with OpenFDA features set to null and flagged with an is\_drug\_trial boolean, or filtered to drug trials only, a decision to be finalized during EDA. After joining and cleaning, we expect our final dataset to comfortably exceed the 50,000-row requirement given ClinicalTrials.gov's 400,000+ registered trials.

**Section 3: Objective and Value Proposition**

The objective of this project is to predict the outcome of a clinical trial, whether it will be completed, terminated, or withdrawn, using its design characteristics, sponsor profile, and drug-level safety signals available at or before the trial's start.

Clinical trials are among the most expensive and high-stakes endeavors in medicine, with Phase 3 trials alone costing hundreds of millions of dollars on average. Trial failures, like terminations and withdrawals, represent enormous losses in time, capital, and opportunity cost, and can delay or permanently prevent life-saving treatments from reaching patients. Despite this, a large fraction of registered trials never reach completion, and the factors driving failure are not always well understood.

By applying machine learning to large-scale, multi-source trial registry data, we aim to identify which design choices, sponsor characteristics, and drug safety profiles most strongly predict trial failure. These insights could benefit pharmaceutical companies optimizing trial design, research institutions allocating funding, regulatory agencies identifying high-risk trials early, and investors evaluating the viability of drug development pipelines.

**Section 4: Modeling Plan**

**Classification** 

Our primary target variable is a 3-class outcome label: **Completed**, **Terminated**, and **Withdrawn**, which are sourced directly from ClinicalTrials.gov's *overallStatus* field, which serves as a true gold standard label requiring no derivation. We will begin with a logistic regression baseline and then explore random forests and XGBoost. Model performance will be evaluated using macro F1-score to account for class imbalance across outcome tiers. We will split the dataset temporally, so training on trials that started before a cutoff year and testing on trials that started after, to prevent data leakage and more closely reflect real-world deployment.

**Feature Importance** 

To understand which trial characteristics most strongly drive outcome predictions, we will conduct feature importance analysis using tree-based model importance scores as well as SHAP (SHapley Additive exPlanations) for both global and local interpretability. This will allow us to identify, for example, whether sponsor class or adverse event count is the dominant predictor of trial termination, and to surface actionable design recommendations for trial planners.

**Section 5: Anticipated Obstacles and Challenges**

**Data Integration — Partial Join Coverage** 

* OpenFDA only covers drug and biologic interventions, meaning device and behavioral trials will lack FDA features. This results in a partial join with potentially significant null coverage.   
* Solution: retain non-matching rows with a boolean is\_drug\_trial flag and null-filled OpenFDA features, or restrict analysis to drug trials only; this is to be decided during EDA.

**Class Imbalance** 

* Completed trials are likely to outnumber terminated and withdrawn trials, creating imbalance across the three outcome classes.   
* Solution: stratified train/test split and class weighting during model training, with macro F1 as the primary evaluation metric.

**Intervention Name Matching** 

* Drug names across ClinicalTrials.gov and OpenFDA may differ in formatting, use of brand vs. generic names, or include combination therapies.   
* Solution: string normalization, and fuzzy matching with a conservative similarity threshold for ambiguous cases. 

**Label Ambiguity** 

* Some trials have statuses such as "Suspended" or "Unknown" that do not map cleanly to our three target classes.   
* Solution: exclude ambiguous status labels from the training set and document the filtering decision in EDA.

