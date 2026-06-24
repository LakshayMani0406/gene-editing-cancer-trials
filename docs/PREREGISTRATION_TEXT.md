# Pre-Registration (Study 2): do registration-time text fields predict trial completion?

**Author:** Lakshay Mani (MS Analytics, Northeastern University)
**Date:** 2026-06-25
**Status:** Committed before any text feature construction, embedding, or modeling is run. No results for the analyses below exist at commit time. The git commit timestamp fixes this plan.

## 1. Relationship to the prior study

This is a separate study with its own hypotheses. Study 1 (`docs/PREREGISTRATION.md`, `docs/FINDINGS.md`) showed that structured registration metadata barely predicts oncology gene-editing trial completion: under a strict label and a temporal split the honest classifier scored AUC 0.6047 and the cause-specific Cox a concordance of 0.6188, both below the 0.65 practical line, with a permutation test placing the classifier about two standard deviations above a chance null (p = 0.017). Study 2 asks whether the free-text fields a trial provides at registration carry predictive signal that the structured fields did not. Same honest frame: registration-only inputs, strict label, temporal split, permutation null, and a pre-declared practical line.

## 2. Question and thesis

Do registration-time text fields (eligibility criteria, brief summary, detailed description) add predictive signal for completion beyond the structured metadata? My thesis is that they add little: language is a richer re-encoding of the same registration-time information, and the prior study showed the registration-time ceiling is near chance. I expect text to be statistically detectable but not to move the practical needle.

## 3. Data and text fields

- Source: the three free-text fields as provided at registration: **eligibility criteria**, **brief_summary**, **detailed_description**, pulled from `raw/` (the ClinicalTrials.gov snapshot) keyed by NCT ID. The cleaned CSV does not retain these fields, so I will extract them from `raw/clinicaltrials_raw.json`.
- Registration-time fidelity. ClinicalTrials.gov serves the current text, which can differ from the first-posted version. Eligibility and brief_summary are typically set at registration and rarely revised; detailed_description is more often updated during conduct. To stay registration-time, the primary text features use **eligibility + brief_summary**, and detailed_description is treated as a secondary, sensitivity-only input. If `raw/` carries a first-posted vs last-updated date, I will report the gap and flag any field that looks conduct-edited. This is the text analogue of the conduct-time leakage concern in Study 1.
- Cohort: the same strict-label resolved set as Study 1 (completed = 1; terminated, withdrawn, or suspended = 0; in-progress dropped), restricted to trials that actually have the text fields. The exact N and the train/test sizes will be reported in T1, not invented here.
- Seed 42 everywhere.

## 4. Hypotheses (committed predictions)

**TH1.** Text-derived features, under the strict label and temporal split, achieve a test AUC statistically distinguishable from chance by a label-permutation test. **Directional prediction: yes, distinguishable.** Reason: text length and complexity proxy for phase, sponsor class, and disease area, each of which carried weak but detectable signal in Study 1, so with roughly 1,400 training trials a permutation test will likely detect a small effect. I expect the magnitude to remain near useless.

**TH2.** A model combining text features with the structured registration features does **not** beat the structured-only honest baseline by more than **0.03 AUC**. **Directional prediction: text adds little.** Reason: text mostly re-encodes the same registration-time information, and Study 1 showed that ceiling is near chance, so text cannot exceed what is knowable at registration about an outcome that is close to unpredictable.

**TH3.** Among the text signals, **eligibility-criteria specificity** (length, number of inclusion and exclusion constraints, and biomarker or genetic-eligibility terms) is the most predictive, more so than the readability or complexity of the brief summary. **Direction: negative.** More restrictive, longer, more biomarker-gated eligibility is associated with lower completion, through enrollment difficulty (narrow eligibility impedes accrual, a known driver of trial termination). I hold the direction with moderate confidence; specificity could instead proxy for better-resourced trials. If TH3 holds, eligibility specificity is the one place text may add marginal signal the structured set lacked, since the leaky enrollment feature was dropped in Study 1, but still within the TH2 margin.

## 5. Analysis plan

### 5.1 Feature families (built separately so cheap signals are not conflated with embeddings)
(a) **Interpretable text features**, computed deterministically: eligibility word count; number of inclusion and exclusion bullet points; Flesch-Kincaid grade level of the brief summary and of the eligibility text (Flesch, 1948; Kincaid et al., 1975); count of numeric constraints in eligibility (regex over numbers adjacent to comparison or unit tokens such as age, years, ECOG, percent, counts); presence and count of biomarker or genetic-eligibility terms from a fixed pre-declared list (mutation, mutant, biomarker, expression, positive, negative, HER2, EGFR, ALK, ROS1, BRAF, KRAS, PD-L1, PD-1, BRCA, MSI, TMB, CD19, BCMA, genotype, allele, amplification, fusion, wild-type, overexpression); brief_summary and detailed_description word counts.
(b) **Semantic embeddings**: embed brief_summary concatenated with eligibility using a sentence-transformer (Reimers & Gurevych, 2019). Primary model: all-MiniLM-L6-v2 (384-dim; Wang et al., 2020). Biomedical sensitivity model: a PubMedBERT sentence encoder (Gu et al., 2021). Mean-pooled sentence embeddings, reduced by PCA to 50 components (seed 42) before the classifier to keep dimensionality sane relative to the training size. This step needs sentence-transformers and torch, which I will not install without explicit approval, and a smoke test on 50 trials locally before any full run.

### 5.2 Models (all use the same classifier for a clean comparison)
RandomForest, identical hyperparameters and seed to Study 1 (n_estimators 400, max_depth 8, min_samples_leaf 5, max_features 0.4, random_state 42).
- **Structured-only**: the Study 1 registration-only feature set, re-fit and re-evaluated on the exact text cohort and split so all three models share identical trials. I will report this re-run baseline alongside the original 0.6047 reference.
- **Text-only**: interpretable text features plus the 50 PCA embedding components.
- **Combined**: structured features plus text-only features.

### 5.3 Split and permutation null
Temporal split, train on earlier start years and test on the most recent 20 percent (cut reported in T1). For the text-only and combined models: shuffle the training labels 1,000 times, refit, score the real held-out set, and report the observed AUC, the null mean, the null 95 percent interval, and the empirical one-sided p as (count of null AUCs at least as large as observed, plus 1) divided by 1,001 (Ojala & Garriga, 2010).

### 5.4 Combined-versus-structured comparison
Report delta AUC = combined minus structured, both fit on the identical cohort and split. Report text-only AUC against 0.50 and against the structured baseline.

### 5.5 Metrics, practical line, and multiple comparisons
Primary metric AUC-ROC, to match Study 1. The practical line stays at **0.65**: I will state whether text-only or combined crosses it. For TH3, univariate associations of each interpretable feature with the label are corrected with Benjamini-Hochberg at q = 0.05 (Benjamini & Hochberg, 1995), and feature importance is read from the text-only model.

## 6. Falsification criteria

- **TH1 falsified** if the text-only permutation p is at or above 0.05 (text not distinguishable from chance). Confirmed if p is below 0.05.
- **TH2 falsified** if combined AUC minus structured AUC, on the identical cohort and split, exceeds 0.03. Confirmed if it is 0.03 or less.
- **TH3 falsified** if eligibility specificity is not the top-ranked text feature, or if its association with completion runs positive rather than the predicted negative direction, after correction.

## 7. Pre-specified interpretation rules

- Statistical versus practical significance, as in Study 1. Even if TH1 confirms a statistically real text signal, an AUC below 0.65 is pre-declared practically useless, and the thesis that registration text barely predicts completion holds on effect size regardless of the permutation p.
- If TH2 is falsified (text adds more than 0.03 AUC), that is a positive result and I will report it as such, including which feature family (interpretable or embeddings) drove the gain.
- A confirmed TH3 with a small absolute contribution is reported as the one text edge, not as a contradiction of TH2.

## 8. What I will report regardless of outcome

Every model AUC including nulls, the full permutation null summary, the text-fidelity gap for any conduct-edited field, the cohort and exclusion counts, the interpretable-feature importances and corrected p-values, and any of TH1, TH2, or TH3 that turns out wrong.

## 9. Software and reproducibility

Python 3.11, scikit-learn 1.6.1, NumPy 2.2.3, pandas 2.3.3, seed 42. Flesch-Kincaid computed from a standard formula (no new dependency required) or via a small readability library if approved. The embedding step requires sentence-transformers and torch, and possibly a GPU; I will not install these or run any embedding without explicit approval, will smoke-test on 50 trials locally first, and if the full embedding run is heavy will prepare a single-GPU SLURM script for the cluster, write embeddings to `artifacts/`, and rsync them back. Machine-readable outputs: `artifacts/text_features.json` (or parquet for the matrices), `artifacts/text_eval.json`, and `artifacts/text_permutation_null.json`.

## 10. References

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289-300.

Flesch, R. (1948). A new readability yardstick. *Journal of Applied Psychology, 32*(3), 221-233.

Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., & Poon, H. (2021). Domain-specific language model pretraining for biomedical natural language processing. *ACM Transactions on Computing for Healthcare, 3*(1), Article 2.

Kincaid, J. P., Fishburne, R. P., Rogers, R. L., & Chissom, B. S. (1975). *Derivation of new readability formulas for Navy enlisted personnel* (Research Branch Report 8-75). Naval Air Station Memphis.

Ojala, M., & Garriga, G. C. (2010). Permutation tests for studying classifier performance. *Journal of Machine Learning Research, 11*, 1833-1863.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing* (pp. 3982-3992). Association for Computational Linguistics.

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. In *Advances in Neural Information Processing Systems 33* (pp. 5776-5788).
