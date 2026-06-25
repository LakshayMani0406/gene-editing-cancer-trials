# Findings (Study 2): does registration text predict completion? Barely, and only if you avoid a leakage trap

**Author:** Lakshay Mani (MS Analytics, Northeastern University)
**Date:** 2026-06-25
**Pre-registration:** `docs/PREREGISTRATION_TEXT.md`, committed (commit 99633ba) before any text feature was built. I adjudicate each pre-registered hypothesis against the numbers below, including the prediction that briefly looked falsified.

## Headline

Adding the language a trial provides at registration, eligibility criteria and brief summary, does not move completion prediction in any useful way. On genuine registration-time text the combined model reaches AUC 0.628, only +0.024 over the structured-only 0.605, below the 0.03 margin I pre-registered and well short of the 0.65 practical line. Sentence embeddings add nothing beyond simple surface features. Study 2 lands where Study 1 did: completion is near-unpredictable from registration data.

The reason this study is worth keeping is a trap I fell into and then caught: the **current** ClinicalTrials.gov text leaks. Used naively it inflated the text signal roughly twofold and manufactured a spurious crossing of the practical line. Only re-fetching the first-posted text exposed it.

## The arc (strict label, temporal split, RandomForest seed 42, n_train 1393, n_test 411)

| Model | AUC | vs structured |
|---|---|---|
| Structured-only (Study 1 baseline, re-run on this cohort) | 0.6047 | reference |
| Combined, **current** text (interpretable) | 0.6562 | +0.0515, crosses 0.65 |
| Combined, **first-posted** text (interpretable) | 0.6314 | +0.0267 |
| Combined, **first-posted** text (interpretable + embeddings) | 0.6283 | +0.0236 |
| Text-only, first-posted (interpretable + embeddings) | 0.5971 | below structured |
| Interpretable-only 0.5955, embeddings-only 0.5918 | near chance | |

The middle of that table is the whole story: the +0.0515 that crossed 0.65 was current text; the same features computed on the text as it was at registration give +0.0267, and adding embeddings does not help.

## The leakage I caught

I first evaluated on the current text and saw combined AUC 0.6562, a +0.05 gain over structured that crossed 0.65 and was statistically real (permutation p = 0.002, 1 of 1000 shuffles reached it). That would have falsified my own TH2. I did not trust it, because the gap between a trial's first-posted date and its last-updated date was large (median 1896 days).

A cheap proxy check was reassuring but wrong. The gain looked concentrated in low-update-gap trials and the top feature was uncorrelated with the gap, which suggested registration-time signal. The proxy understated the problem, because the update date measures when a record was touched, not how much the text changed.

The definitive test settled it. I re-fetched the first-posted version of every trial through the ClinicalTrials.gov history API and recomputed the identical features. The text had drifted enormously: **100% of eligibility texts had changed** since registration, 85.9% of summaries, with a median eligibility word-count change of 17% and 78.9% of trials changing by more than 10%. On the first-posted text the combined gain fell to +0.0267 and stopped crossing 0.65. About half the apparent text signal, and all of the practical-line crossing, was post-registration editing. This is the text analogue of the conduct-time leakage that collapsed Study 1, and it is the one reusable caution from this work: do not treat current ClinicalTrials.gov text as a registration-time feature.

## Hypothesis adjudication

**TH1 (text distinguishable from chance): confirmed.** On first-posted text, the text-only model scores AUC 0.5971 with a permutation p of 0.005 (4 of 1000 null shuffles reached it), outside the null 95% band. There is a real signal, as I predicted, but it is near useless in magnitude.

**TH2 (combined does not beat structured by more than 0.03): confirmed.** On first-posted text the combined gain is +0.0236 with embeddings and +0.0267 without, both under the 0.03 margin, and combined does not reach 0.65 (permutation p = 0.002, driven by the structured component, not the text increment). Text adds little, as I predicted. The current-text falsification was an artifact of the leakage above.

**TH3 (eligibility specificity is the top text feature, negative direction): partial.** Eligibility word count was the top text feature by importance, which supports the length half of my prediction, but readability ranked nearly as high and the specificity features, biomarker terms and numeric constraints, did not lead. I assessed this by model feature importance rather than the Benjamini-Hochberg univariate test I pre-registered, because the practical contribution is negligible and the ranking is the more honest lens. I did not get a clean directional confirmation, so I record TH3 as only partly supported.

## Comparison to Study 1

Study 1 found structured registration metadata barely beats chance (AUC 0.60, Cox C-index 0.62). Study 2 asked whether language closes the gap and the answer is no: registration-time text takes the combined model only to 0.628, still below the practical line, with the increment below my margin, and semantic embeddings add nothing that simple word counts and readability do not. The near-unpredictability of completion from registration data generalizes from structured fields to free text. Two studies, two honest negative results, each pre-registered and each with a leakage trap caught before it could become a false positive.

## Methods note

Interpretable features: eligibility and summary word counts, inclusion and exclusion counts, Flesch-Kincaid grade level (Flesch, 1948; Kincaid et al., 1975), numeric-constraint counts, and biomarker-term counts. Embeddings: eligibility concatenated with brief summary, encoded with all-MiniLM-L6-v2 (Reimers & Gurevych, 2019; Wang et al., 2020), reduced by PCA to 50 components fit on the training split (69% of variance retained). Significance by label-permutation test, 1000 shuffles, empirical one-sided p (Ojala & Garriga, 2010). All on the strict label and temporal split, seed 42, with sentence-transformers 5.6.0 and torch 2.12.1. Machine-readable numbers in `artifacts/text_eval_stage1.json`, `text_leakage_check.json`, `text_firstposted_eval.json`, `text_t2_eval.json`, and `text_t2_permutation.json`.

## References

Flesch, R. (1948). A new readability yardstick. *Journal of Applied Psychology, 32*(3), 221-233.

Kincaid, J. P., Fishburne, R. P., Rogers, R. L., & Chissom, B. S. (1975). *Derivation of new readability formulas for Navy enlisted personnel* (Research Branch Report 8-75). Naval Air Station Memphis.

Ojala, M., & Garriga, G. C. (2010). Permutation tests for studying classifier performance. *Journal of Machine Learning Research, 11*, 1833-1863.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing* (pp. 3982-3992). Association for Computational Linguistics.

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. In *Advances in Neural Information Processing Systems 33* (pp. 5776-5788).
