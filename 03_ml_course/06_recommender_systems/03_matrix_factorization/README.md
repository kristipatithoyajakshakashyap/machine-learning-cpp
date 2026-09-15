# Biased matrix factorization

**Method.** Funk-SVD style explicit-feedback factorization. Every rating is
modelled as `mu + b_u + b_i + p_u . q_i` with 16-dimensional user and item
factors. `MatrixFactorization::fit` runs 30 epochs of stochastic gradient
descent over the observed ratings only (seeded shuffle, so runs are
reproducible), penalising biases and factors with an L2 strength `lambda`
(candidates 0.02 and 0.1). Ranking uses the unclamped score; rating
prediction clamps to [1, 5].

**Core idea.** Instead of comparing items through co-raters, learn a short
vector for every user and item so that their dot product explains the
observed ratings. Biases absorb "generous user" and "popular film" effects
before the factors have to; missing ratings contribute no error term. This
is the workhorse of explicit-rating recommenders and the model the evaluation
module analyses in depth.

## Dataset

MovieLens 100K (`../data/u.data`). 100,000 ratings is enough for SGD to learn
meaningful factors in seconds, and the per-epoch RMSE curve written by the
implementation lesson shows the fit/overfit trade-off clearly.

## Prerequisites

`01_popularity_baseline` and `02_item_based_collaborative_filtering`, dot
products, gradients and the idea of stochastic gradient descent
(`helper/math/optim.hpp`).

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Latent factors, biases, explicit vs implicit feedback. | - |
| `math_intuition.md` | - | The SGD update derived and worked by hand. | - |
| `implementation.md` | - | Walkthrough of `MatrixFactorization`. | - |
| `exercises.md` | - | Practice questions. | - |
| `Model.hpp` | (header) | `MatrixFactorization` (fit / predict / rank_score / save / load). | - |
| `01_theory.cpp` | `rec_mf_theory` | Lesson text and `recommend()` on a toy set. | prints only |
| `02_math_intuition.cpp` | `rec_mf_math_intuition` | One SGD step on a toy residual. | prints only |
| `03_implementation.cpp` | `rec_mf_implementation` | Training RMSE per epoch and the factors of the five most-rated items. | `results/03_implementation_results/` |
| `04_end_to_end.cpp` | `rec_mf_end_to_end` | Full project: chronological split, lambda in {0.02, 0.1}, rating error and ranking. | `results/04_end_to_end_results/{full,quick}/` |
| `predict.cpp` | `rec_mf_predict` | Reloads a saved model and lists ten unseen items for one user. | `results/predict_results/recommendations.csv` |
| `tests/model_test.cpp` | `rec_mf_tests` | Numerical fixture for the shared recommender stack. | prints only |

## Build and run

From the repository root (MinGW toolchain, `D:/msys64/ucrt64/bin` on PATH):

```powershell
cmake --preset course
cmake --build --preset course --target rec_mf_implementation rec_mf_end_to_end rec_mf_predict
build/03_ml_course/06_recommender_systems/03_matrix_factorization/rec_mf_implementation.exe
build/03_ml_course/06_recommender_systems/03_matrix_factorization/rec_mf_end_to_end.exe          # all ratings
build/03_ml_course/06_recommender_systems/03_matrix_factorization/rec_mf_end_to_end.exe --quick  # earliest 12,000
```

## Recommending for a user

```powershell
rec_mf_predict --model results/04_end_to_end_results/full/model/model.txt --user 1
```

Writes `results/predict_results/recommendations.csv` (`user,rank,item,score`).

## Results layout

- `results/03_implementation_results/` - `train_rmse_per_epoch.csv/.svg`,
  `item_factors_top5.csv`.
- `results/04_end_to_end_results/full/` (or `quick/`) - `data/`, `eda/`,
  `validation/candidate_scores.csv`, `evaluation/` (`metrics.json`,
  `predictions.csv`, `recommendations.csv`, per-user ranking metrics,
  `figures/`), `model/model.txt`, `model/feature_schema.json`,
  `inference/reload_verification.json`, `run_manifest.json`, `report.md`,
  `execution.log`.
- `results/predict_results/recommendations.csv`.

Re-running a target replaces its folder. There is no `output.txt`; read `report.md`.

## Tests

`ctest --test-dir build -R rec_mf` runs `rec_mf_numerical` (finite predictions
for unknown ids, archive round trip reproduces predictions, corrupt archive
rejection, ranking-metric fixture) and `rec_mf_workflow`.

## Key takeaways

- Read both factor vectors before updating either; the paired update must use
  the old values.
- Regularise per observed rating, and tune `lambda` on a chronological
  validation period, not a random one.
- Training RMSE keeps falling after validation RMSE stops; the epoch curve is
  the overfitting diagnostic.

## Next module

`04_recommender_evaluation` puts this model through the full chronological
rating and ranking protocol and explains every metric in `report.md`.
