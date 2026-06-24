# Findings: completion is near-unpredictable from registration metadata

**Author:** Lakshay Mani (MS Analytics, Northeastern University)
**Date:** 2026-06-24
**Pre-registration:** `docs/PREREGISTRATION.md`, committed and pushed (commit 70fb0da) before any survival or permutation result existed. I adjudicate every hypothesis below against the numbers, including the predictions I got wrong.

## Headline: three methods agree

Whether an oncology gene-editing trial will complete is close to unpredictable from the information available when it is registered. Three independent analyses converge on the same answer:

1. **Leakage audit.** Holding one random forest fixed and changing only what it is allowed to see, AUC falls from 0.8963 (with conduct-time leaks) to 0.7185 (registration-only, random split) to 0.6047 (registration-only, strict label, temporal split).
2. **Permutation null.** The honest 0.6047 sits about two standard deviations above a chance null (null mean 0.4951, 95% interval [0.3902, 0.5980]), with an empirical one-sided p of 0.0170. The signal is statistically real but small.
3. **Survival model.** A cause-specific Cox model gives a concordance index of 0.6188, next to the 0.6047 classifier AUC, and its proportional-hazards assumption fails, so even that modest number is an approximation.

Classifier AUC 0.60 and Cox C-index 0.62 agree, and both sit well below the 0.65 line I pre-declared as practically useless. The contribution is the negative result, established three ways, plus one stable positive association: industry sponsorship.

## A note on methodological honesty: the 2044 snapshot bug

My first survival run censored active trials at a snapshot date of 2044, because I derived the snapshot from the maximum completion date among resolved trials and the cleaned data contains erroneous future completion dates. That inflated the follow-up of the 1,935 active trials in the cohort by roughly eighteen years and pushed the Cox C-index to a contaminated 0.7433, with `start_year` showing an implausible hazard ratio of 0.54 at p near 1e-190. I caught this before drawing any conclusion, fixed the snapshot to the data collection date (2026-06-01), and excluded 50 resolved trials whose completion dates fell after that date as data errors. The corrected C-index is 0.6188. I report this because the bug, left unchecked, would have manufactured a positive result, and catching it is part of the analysis.

## Cohort

After fixing the snapshot, the survival cohort is 3,616 trials: 1,067 completions (the event of interest), 614 competing events (terminated, withdrawn, or suspended), and 1,935 active trials censored at 2026-06-01. I excluded 844 records: 715 of unknown status, 50 with future-dated completions, and 79 with missing or implausible durations.

## Hypothesis-by-hypothesis adjudication

### H1: statistically falsified, practically confirmed

I predicted that the honest AUC would be statistically indistinguishable from chance. That prediction is **falsified**. The permutation test (Ojala & Garriga, 2010) places the observed 0.6047 above the null 95th percentile, with p = 0.0170, so the model does extract a real signal from registration metadata. I do not bury that.

At the same time, the effect is **practically near-useless**. The 0.6047 AUC is below the 0.65 threshold I pre-declared in section 6 of the pre-registration, and it is corroborated by the independent Cox C-index of 0.6188. Section 6 governs the interpretation: a statistically detectable but practically worthless edge supports the thesis on effect size, and the permutation p does not overturn it. The honest one-line statement of H1 is that registration metadata carries a real but worthless signal for trial completion. This is a finding, not a failure, and it is exactly the outcome the pre-registration was written to handle.

### H2: falsified as written

I predicted that most Cox covariates would have hazard-ratio confidence intervals crossing 1. After Benjamini-Hochberg correction (Benjamini & Hochberg, 1995), 7 of 13 covariates have intervals that exclude 1: `start_year` (HR 1.086, adjusted p 0.036), Phase III (0.484, 0.0002), Phase IV (4.241, 0.005), Phase other (1.352, 0.005), Government-NIH sponsor (0.725, 0.005), Industry sponsor (1.572, 1.2e-08), and hematologic tumor type (0.810, 0.005). A majority show a detectable association, so H2 as I wrote it is **falsified**.

The nuance matters and does not rescue H2; it explains it. Individual associations are detectable largely because the cohort is large, yet they do not add up to useful prediction: the same model scores a C-index of only 0.62. The proportional-hazards assumption also fails for 7 of 13 covariates (Grambsch & Therneau, 1994), so these hazard ratios are time-averaged approximations, not stable effects. Detectable in aggregate, useless in aggregate.

The covariates that were genuinely null, reported so they are not hidden: number of primary outcomes (HR 0.980, adjusted p 0.215), Phase I/II (0.848, 0.114), Phase II (0.981, 0.872), Government-Federal sponsor (0.770, 0.872), Research Network sponsor (0.635, 0.104), and unknown sponsor (1.039, 0.909).

### H3a, trial phase: falsified, and my direction was wrong

I predicted, with high confidence, that later phases would complete at a higher rate, that is a cause-specific completion hazard ratio above 1. The data contradict me. Phase III has a hazard ratio of **0.484** (95% CI [0.346, 0.678]), the opposite of my prediction, and it is the well-powered later phase. Phase IV is above 1 (4.241) as I predicted, but it rests on roughly twenty trials with a wide interval, so I do not lean on it. I got the direction wrong for the phase that matters, and I am recording that.

The interpretation is consistent across the survival outputs. A hazard ratio describes the rate of completion within the observation window, not the eventual proportion. Larger, longer Phase III trials complete more slowly inside the window, which reads as a lower completion hazard. The Kaplan-Meier curves by phase (Kaplan & Meier, 1958) cross rather than staying ordered, the log-rank test still finds phases differ (chi-square 56.07, p = 2.8e-10; Mantel, 1966), and the proportional-hazards test flags phase as non-proportional. All three say the same thing: phase relates to completion timing, but not in the simple monotonic way I assumed.

### H3b, sponsor class: confirmed, and it is the one positive edge

I predicted, with lower confidence, that industry-sponsored trials would complete at a higher rate than non-industry trials. This is **confirmed**. Industry has a hazard ratio of **1.572** (95% CI [1.360, 1.817], adjusted p 1.2e-08) in the predicted direction. Industry is also one of the few covariates whose proportional-hazards assumption holds, so unlike the phase effects this one is stable rather than time-averaged. The log-rank test across sponsor classes is significant (chi-square 66.09, p = 1.5e-13). This is the single reliable, direction-correct signal in the study.

## Supporting survival detail

The Aalen-Johansen estimator (Aalen & Johansen, 1978) gives a cumulative completion incidence of 0.26 by 60 months against a competing-event incidence of 0.19, with completion continuing to accumulate past termination at longer follow-up. The Kaplan-Meier and cumulative-incidence plots are in `outputs/`, and the machine-readable numbers are in `artifacts/survival.json` and `artifacts/permutation_null.json`. All of this was produced with lifelines 0.30.3 (Davidson-Pilon, 2019) and a fixed seed of 42.

## What the negative result means

At registration, a planner cannot meaningfully predict whether an oncology gene-editing trial will complete. The metadata holds a whisper of signal that a permutation test can detect but that no decision-maker could act on, and two different modeling frames, a classifier and a survival model, agree on the magnitude. The value here is the rigor of the negative claim: a pre-registered plan, a leakage audit that explains why naive numbers look strong, a permutation null that quantifies how small the real signal is, and a survival model whose own diagnostics caution against over-reading its coefficients.

The one thing a planner can take away is sponsorship. Industry-sponsored trials complete at a meaningfully higher rate, the estimate is stable under the proportional-hazards check, and it survives multiple-comparison correction. Everything else I tested either carries no detectable signal or carries a detectable signal too small and too unstable to use.

## References

Aalen, O. O., & Johansen, S. (1978). An empirical transition matrix for non-homogeneous Markov chains based on censored observations. *Scandinavian Journal of Statistics, 5*(3), 141-150.

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289-300.

Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B, 34*(2), 187-220.

Davidson-Pilon, C. (2019). lifelines: Survival analysis in Python. *Journal of Open Source Software, 4*(40), 1317.

Grambsch, P. M., & Therneau, T. M. (1994). Proportional hazards tests and diagnostics based on weighted residuals. *Biometrika, 81*(3), 515-526.

Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association, 53*(282), 457-481.

Mantel, N. (1966). Evaluation of survival data and two new rank order statistics arising in its consideration. *Cancer Chemotherapy Reports, 50*(3), 163-170.

Ojala, M., & Garriga, G. C. (2010). Permutation tests for studying classifier performance. *Journal of Machine Learning Research, 11*, 1833-1863.
