# Pre-Registration: is oncology gene-editing trial completion predictable from registration metadata?

**Author:** Lakshay Mani (MS Analytics, Northeastern University)
**Date:** 2026-06-24
**Status:** Committed before any survival modeling or permutation testing is run. No results for the analyses below exist at commit time. The git commit timestamp fixes this plan.

## 1. Thesis

I claim a negative result: trial completion is close to unpredictable from information available at registration. I set out to prove it three independent ways:

1. A leakage audit (already done). Holding the model fixed, AUC falls 0.8963 (leaked) to 0.7185 (registration-only, random split) to 0.6047 (strict label, temporal split). These numbers live in `artifacts/results.json` and are the input to Phase 2, not a result of this pre-registration.
2. A permutation null for the honest classifier (Phase 2 here).
3. A survival model on registration-only features (Phase 1 here), which is the statistically correct frame because recent trials are right-censored rather than failed.

A negative result is only meaningful if the plan was fixed before the data spoke. That is the purpose of this document.

## 2. Data and cohort

- Source file: `data/crispr_trials_clean.csv` (interventional oncology gene-editing trials, ClinicalTrials.gov).
- Strict outcome label: completed = event of interest; terminated, withdrawn, or suspended = competing event; recruiting or active-not-recruiting = censored.
- Registration-only feature set (known at trial registration), identical to the classifier audit: `start_year`, `n_primary_outcomes`, `is_hematologic`, `is_recent`, `phase_clean`, `cancer_type`, `tumor_category`, `sponsor_class_clean`, `trial_era`.
- Random seed: 42 everywhere.
- Exact cohort sizes (after dropping trials with missing or implausible durations) will be reported in Phase 1. I am not stating them here because I have not run the survival construction yet.

## 3. Hypotheses (committed predictions)

**H1.** Under the strict label and temporal split, the registration-only classifier AUC (observed 0.6047) is statistically indistinguishable from chance (0.50) by a label-permutation test.

**H2.** In a Cox proportional-hazards model on registration-only features, most covariates have hazard ratios whose 95% confidence intervals cross 1 (no detectable effect on the completion hazard).

**H3 (the expected exceptions, with direction).** I predict one or two features carry real signal:
- **H3a, trial phase (high confidence).** Phase is a non-null predictor. Direction: later phases complete at a higher cumulative rate than Phase I, so the cause-specific completion hazard ratio is greater than 1 for higher phases. Rationale: later-phase trials have passed earlier gates, are better funded, and are more committed.
- **H3b, sponsor class (lower confidence).** Sponsor class is a non-null predictor. Direction: industry-sponsored trials complete at a higher rate than non-industry, hazard ratio greater than 1 for Industry versus the academic or other reference. I hold this with less confidence, since industry trials can also be terminated for commercial reasons.

## 4. Analysis plan

### 4.1 Survival construction (Phase 1)
- Time origin: `start_date`. Time scale: months.
- Duration `T`: months from start to the relevant end date. For completed trials, the completion date. For terminated, withdrawn, or suspended trials, their stop date. For active trials, the last-observed date.
- If a last-update date is not present in the cleaned data, I will censor active trials at the data snapshot date and report that choice explicitly. Trials with missing, negative, or implausible (greater than 480 months) durations are excluded, and the excluded count is reported.
- Primary analysis: cause-specific Cox for completion. The competing event (terminated, withdrawn, suspended) is censored at its stop time. This estimates the cause-specific hazard of completion.
- Secondary analysis: Aalen-Johansen cumulative incidence for completion versus the competing event, as the competing-risks view. This is reported so the cause-specific censoring choice is transparent.

### 4.2 Kaplan-Meier and log-rank
- KM completion curves overall, then stratified by `phase_clean` and by `sponsor_class_clean`.
- Log-rank (Mantel) tests across strata. Chi-square statistic and p-value reported for each stratification.

### 4.3 Cox proportional-hazards model
- Covariates: the registration-only set in section 2. Categorical features one-hot encoded with an explicit reference level; `phase_clean` modeled as ordered. Ties handled by the Efron method.
- Report for every covariate: hazard ratio, 95% CI, Wald p-value, and the same p-value after Benjamini-Hochberg correction. Report the overall likelihood-ratio test and Harrell's concordance index (C-index) as the survival analogue of AUC.

### 4.4 Proportional-hazards assumption
- Schoenfeld residual test (Grambsch and Therneau), global and per covariate. I will report whether the assumption holds. If it is violated for a covariate, I will note stratification or a time-varying term as the remedy and flag the affected estimates as approximate. I report the violation regardless.

### 4.5 Permutation null for the honest classifier (Phase 2)
- Config: registration-only features, strict label, temporal split, RandomForest seed 42 (the Phase-3 honest config of the leakage audit).
- Shuffle the training labels 1000 times, refit, and record the held-out AUC each time to build the null distribution.
- Report the observed AUC (0.6047), the null mean, the null 95% interval, and the empirical one-sided p-value for "observed greater than chance", computed as (count of null AUCs at least as large as observed, plus 1) divided by (1000 plus 1).

### 4.6 Multiple-comparison correction
- Benjamini-Hochberg false discovery rate at q = 0.05, applied across the full set of Cox covariate p-values jointly, and separately across the set of log-rank tests. Both raw and adjusted p-values are reported.

## 5. Falsification criteria

- **H1 is falsified** if the permutation empirical p-value is below 0.05 (observed AUC above the null 95th percentile). It is confirmed if the p-value is 0.05 or higher.
- **H2 is falsified** if, after Benjamini-Hochberg correction, more than half of the registration covariates have hazard-ratio CIs that exclude 1. It is confirmed if at most half do.
- **H3 is falsified** for a named feature if its hazard-ratio CI includes 1 (no detectable effect) or if the effect runs opposite to the predicted direction. It is confirmed if the CI excludes 1 in the predicted direction after correction.

## 6. Pre-specified interpretation rules (committed now to prevent post-hoc reframing)

- I separate statistical significance from practical significance in advance. Even if H1 is falsified statistically, an AUC below 0.65 and a Cox C-index below 0.65 are pre-declared here as "practically near-useless" discrimination. The thesis that registration metadata barely predicts completion is supported by effect size in that regime regardless of the permutation p-value.
- If H3 confirms phase or sponsor as a real but small effect, I will report it as the single positive edge, not as a contradiction of the thesis.

## 7. What I will report regardless of outcome

Every hazard ratio including null ones, the full null-distribution summary, any proportional-hazards violation, the exact censoring choice for active trials, the excluded-trial count, and any pre-registered prediction that turned out wrong. Nulls and failed predictions are part of the result.

## 8. Software and reproducibility

Python 3.11, scikit-learn 1.6.1, NumPy 2.2.3, pandas 2.3.3, lifelines 0.30.3, and matplotlib 3.10.0. Seed 42. Machine-readable outputs: `artifacts/survival.json` and `artifacts/permutation_null.json`. Plots to `outputs/`.

## 9. References

Aalen, O. O., & Johansen, S. (1978). An empirical transition matrix for non-homogeneous Markov chains based on censored observations. *Scandinavian Journal of Statistics, 5*(3), 141-150.

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289-300.

Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B, 34*(2), 187-220.

Davidson-Pilon, C. (2019). lifelines: Survival analysis in Python. *Journal of Open Source Software, 4*(40), 1317.

Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *Journal of the American Statistical Association, 94*(446), 496-509.

Grambsch, P. M., & Therneau, T. M. (1994). Proportional hazards tests and diagnostics based on weighted residuals. *Biometrika, 81*(3), 515-526.

Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association, 53*(282), 457-481.

Mantel, N. (1966). Evaluation of survival data and two new rank order statistics arising in its consideration. *Cancer Chemotherapy Reports, 50*(3), 163-170.

Ojala, M., & Garriga, G. C. (2010). Permutation tests for studying classifier performance. *Journal of Machine Learning Research, 11*, 1833-1863.
