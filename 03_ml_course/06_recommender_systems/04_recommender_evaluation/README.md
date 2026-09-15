# Recommender evaluation

**Method.** Chronological rating and ranking evaluation. `Workflow.hpp` in
this folder is the shared end-to-end project used by *every* recommender
module: it sorts MovieLens by timestamp, splits it into train (first ~60%),
validation (next ~20%) and test (last ~20%), selects the model's single
hyper-parameter on validation RMSE, refits on train + validation, and then
reports rating error (RMSE, MAE) *and* top-10 ranking quality (Recall@10,
NDCG@10) on the untouched test period. The end-to-end target of this module
runs that protocol on `MatrixFactorization` (lambda in {0.02, 0.1}).

**Core idea.** A recommender is judged by what it ranks, not only by how well
it guesses stars. Random splits leak the future into training (a user's later
ratings predict earlier ones), so every split here is by time and no timestamp
is cut in two. A test item counts as *relevant* when it was rated >= 4 and was
never seen in training; cold items stay in the denominator so catalog
cold-start honestly lowers recall. Every ranking metric is reported next to a
popularity baseline evaluated on the same users, so an improvement has a
reference point.

## Dataset

MovieLens 100K (`../data/u.data`, 100,000 ratings from 943 users on 1,682
items, with timestamps). Timestamps are what make a chronological protocol
possible; the 80/20 time split still leaves every user and most items in
training, so the numbers are stable enough to compare models.

## Prerequisites

The three previous modules (`01_popularity_baseline`,
`02_item_based_collaborative_filtering`, `03_matrix_factorization`), RMSE/MAE,
sets and sorting, logarithms (for NDCG), and the idea of a held-out period.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Offline evaluation, leakage, rating vs ranking metrics. | - |
| `math_intuition.md` | - | Precision@k, Recall@k and NDCG@k worked by hand. | - |
| `implementation.md` | - | Walkthrough of `Workflow.hpp`. | - |
| `exercises.md` | - | Practice questions. | - |
| `Workflow.hpp` | (header) | `recommend()`, `ranking()`, `chronological_boundary()` and `workflow<Model>()`; shared by all `rec_*` targets. | - |
| `01_theory.cpp` | `rec_evaluation_theory` | Lesson text; why random splits leak. | prints only |
| `02_math_intuition.cpp` | `rec_evaluation_math_intuition` | One ranking metric computed step by step on a toy list. | prints only |
| `03_implementation.cpp` | `rec_evaluation_implementation` | Precision@k / Recall@k curves for Popularity and ItemCF on a chronological 80/20 split. | `results/03_implementation_results/` |
| `04_end_to_end.cpp` | `rec_evaluation_end_to_end` | Full project on `MatrixFactorization`: chronological split, lambda selection, rating and ranking evaluation, archive reload check. | `results/04_end_to_end_results/{full,quick}/` |
| `predict.cpp` | `rec_evaluation_predict` | Reloads a saved model and lists ten unseen items for one user. | `results/predict_results/recommendations.csv` |
| `tests/model_test.cpp` | `rec_evaluation_tests` | Numerical fixture for the shared recommender stack. | prints only |

## Build and run

From the repository root (MinGW toolchain, `D:/msys64/ucrt64/bin` on PATH):

```powershell
cmake --preset course
cmake --build --preset course --target rec_evaluation_implementation rec_evaluation_end_to_end rec_evaluation_predict
build/03_ml_course/06_recommender_systems/04_recommender_evaluation/rec_evaluation_implementation.exe
build/03_ml_course/06_recommender_systems/04_recommender_evaluation/rec_evaluation_end_to_end.exe          # all ratings
build/03_ml_course/06_recommender_systems/04_recommender_evaluation/rec_evaluation_end_to_end.exe --quick  # earliest 12,000
```

`--quick` keeps the earliest 12,000 ratings (the file is time-sorted, so the
subset is deterministic) and writes to `quick/` instead of `full/`. It is a
smoke run for CTest, not a benchmark.

## Recommending for a user

```powershell
rec_evaluation_predict --model results/04_end_to_end_results/full/model/model.txt --user 1
```

Writes `results/predict_results/recommendations.csv` (`user,rank,item,score`):
the ten highest `rank_score` items the user did not rate in training. For an
unknown user id the factor and user-bias terms are skipped, so the list is
ordered by `mean + item_bias` (an item-popularity ordering).

## Results layout

- `results/03_implementation_results/` - `precision_recall_at_k.csv`,
  `precision_at_k.svg`, `recall_at_k.svg`.
- `results/04_end_to_end_results/full/` (or `quick/`) -
  `data/row_assignments.csv`, `data/quality_report.csv`, `eda/`,
  `validation/candidate_scores.csv`, `evaluation/metrics.json`,
  `evaluation/predictions.csv`, `evaluation/recommendations.csv`,
  `evaluation/user_ranking_metrics.csv`,
  `evaluation/popularity_user_metrics.csv`, `evaluation/figures/`,
  `model/model.txt`, `model/feature_schema.json`,
  `inference/reload_verification.json`, `run_manifest.json`, `report.md`,
  `execution.log`.
- `results/predict_results/recommendations.csv`.

Re-running a target replaces its folder. There is no `output.txt`; read
`report.md`, which explains every metric and its limitations.

## Tests

`ctest --test-dir build -R rec_evaluation` runs `rec_evaluation_numerical`
(finite predictions for unknown ids in all three models, `recommend()` never
returns a seen item, archive round trip reproduces predictions, corrupt
archive rejected, Recall@10 = NDCG@10 = 1 on a perfect toy ranking) and
`rec_evaluation_workflow` (the `--quick` project).

## Key takeaways

- Split by time, never at random, and never split a timestamp across the
  boundary.
- Select hyper-parameters on a validation *period*, then evaluate once on a
  later test period.
- Report ranking metrics next to a popularity baseline on the same users; a
  low absolute Recall@10 is normal on MovieLens.
- Keep cold items in the denominator so the metric reflects what a user would
  actually experience.

## Next module

This is the last module of the course. Return to `../README.md` to compare
the three models' `report.md` files, or revisit `05_text_classification` to
see the same train / select / evaluate / persist structure applied to text.
