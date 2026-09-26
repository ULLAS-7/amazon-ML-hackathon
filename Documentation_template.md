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

**Strategy:** Multi-probe blocking with four complementary probe types:
1. **Name forward-prefix** — leading token prefix of the normalised entity name
2. **Name reversed-token** — leading token of the reversed token sequence (catches word-order variation)
3. **Address city segment** — city-level address token
4. **Address house-number digits** — numeric house/building identifier

**Full-scale tuning (2.2M Source-1 × 5M+ Source-2/3 records):**  
The sample-validated design required two adjustments at production scale:

- **Stopword list:** An explicit list of transliterated and generic legal-suffix tokens was built to remove degenerate high-frequency buckets (e.g. the Hindi abbreviation for "Limited", and English tokens such as "center", "partners", "group"). Without this, a small number of extremely common tokens produced unmanageably large buckets.
- **Per-probe safety caps:** Upper bounds on bucket size were applied per probe type to bound the total candidate-set size:
  - `city_cap = 80`
  - `name_cap = 8,000`
  - `nrev_cap = 3,000`
  - `housenum_cap = 200`

**Outcome:**  
- Recall on 2,000-entity ground-truth-aligned validation sample: **~68%**
- Average candidates per Source-1 entity: **~4,600**

This configuration reflects a deliberate trade-off: bounded output size (required by the challenge's candidate-set-size scoring criterion) was prioritised over maximum achievable recall.

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

This project demonstrates end-to-end entity resolution via multi-probe blocking and gradient-boosted matching. Iterative tuning was required to balance recall against candidate-set size at production scale — a core trade-off in real-world entity resolution systems — ultimately yielding a high-precision matcher (F_0.5 = 0.9856) within a bounded candidate footprint.

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
