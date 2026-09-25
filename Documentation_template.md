# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

---

## 2. Methodology

### 2.1 Problem Analysis
*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** [Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]  
**Core Innovation:** [Brief description of your main technical contribution]

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** [e.g., PIN code, phonetic name encoding, TF-IDF, etc.]
- **Candidate pairs generated:** [total]
- **How you ensured true matches were not lost:**

---

## 4. Matching Model

**Features used:**
- Name features: Jaccard similarity, Levenshtein similarity, token-sort-ratio similarity
- Address features: Jaccard similarity, Levenshtein similarity, house number exact-match, city/region match, country match
- Other: Length differences (name and address)

**Most important features:** Address Jaccard and name Jaccard were by far the strongest signals.

**Model type:** LightGBM binary classifier

**Train/validation split:** Grouped by `source1_entity_id` (not pair-level) to prevent data leakage — 1,600 entities for training, 400 for validation.

**Threshold selection method:** Threshold of 0.91 chosen by maximising F_0.5 on the validation split.

---

## 5. Results & Error Analysis

Results below are on the 2,000-entity validation sample (grouped split, not full dataset). Full-dataset results are pending.

| Metric    | Value  |
|-----------|--------|
| F_0.5     | 0.9856 |
| Precision | 0.9897 |
| Recall    | 0.9697 |

**Note:** High precision relative to recall is consistent with the F_0.5 objective (precision is weighted more heavily). Full-scale results will be reported once the end-to-end pipeline has been run over the complete dataset.

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
