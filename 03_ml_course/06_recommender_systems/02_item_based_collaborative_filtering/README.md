# Item-based collaborative filtering

**Method.** `ItemCF` centres every rating by its user's training mean, computes
adjusted-cosine similarity between item columns, shrinks it by
`overlap / (overlap + 10)` and keeps the `k` most similar positive neighbours
per item. A rating is predicted as
`user_mean + sum(sim * centred_rating) / sum(sim)` over the neighbours the
user has rated, falling back to the shrunken item mean when none apply.

**Core idea.** Users who rated the same items similarly provide indirect
evidence about an unrated item. Working with item-item rather than user-user
similarities keeps the table small (items are fewer and more stable than
users) and makes recommendations explainable: "because you liked X". The
neighbourhood size `k` is the single tuned hyper-parameter (candidates 15
and 40).

## Dataset

MovieLens 100K (`../data/u.data`, 100,000 ratings, 943 users, 1,682 films).
Dense enough that most popular films share many co-raters, sparse enough that
the overlap shrinkage and the item-mean fallback visibly matter.

## Prerequisites

`01_popularity_baseline` (model contract, loader, baseline statistics), dot
products and cosine similarity.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Neighbourhood methods, centring, why item-item. | - |
| `math_intuition.md` | - | Adjusted cosine and the weighted deviation by hand. | - |
| `implementation.md` | - | Walkthrough of `ItemCF`. | - |
| `exercises.md` | - | Practice questions. | - |
| `Model.hpp` | (header) | `ItemCF` (fit / predict / rank_score / save / load). | - |
| `01_theory.cpp` | `rec_itemcf_theory` | Lesson text and `recommend()` on a toy set. | prints only |
| `02_math_intuition.cpp` | `rec_itemcf_math_intuition` | Similarity arithmetic on a toy set. | prints only |
| `03_implementation.cpp` | `rec_itemcf_implementation` | Similarity block of the 12 most-rated items and sample top-10 lists. | `results/03_implementation_results/` |
| `04_end_to_end.cpp` | `rec_itemcf_end_to_end` | Full project: chronological split, k in {15, 40}, rating error and ranking. | `results/04_end_to_end_results/{full,quick}/` |
| `predict.cpp` | `rec_itemcf_predict` | Reloads a saved model and lists ten unseen items for one user. | `results/predict_results/recommendations.csv` |
| `tests/model_test.cpp` | `rec_itemcf_tests` | Numerical fixture for the shared recommender stack. | prints only |

## Build and run

From the repository root (MinGW toolchain, `D:/msys64/ucrt64/bin` on PATH):

```powershell
cmake --preset course
cmake --build --preset course --target rec_itemcf_implementation rec_itemcf_end_to_end rec_itemcf_predict
build/03_ml_course/06_recommender_systems/02_item_based_collaborative_filtering/rec_itemcf_implementation.exe
build/03_ml_course/06_recommender_systems/02_item_based_collaborative_filtering/rec_itemcf_end_to_end.exe          # all ratings
build/03_ml_course/06_recommender_systems/02_item_based_collaborative_filtering/rec_itemcf_end_to_end.exe --quick  # earliest 12,000
```

## Recommending for a user

```powershell
rec_itemcf_predict --model results/04_end_to_end_results/full/model/model.txt --user 1
```

Writes `results/predict_results/recommendations.csv` (`user,rank,item,score`)
with the ten best unseen catalog items for that user.

## Results layout

- `results/03_implementation_results/` - `item_similarity_top12.csv/.svg`,
  `sample_recommendations.csv`.
- `results/04_end_to_end_results/full/` (or `quick/`) - `data/`, `eda/`,
  `validation/candidate_scores.csv`, `evaluation/` (`metrics.json`,
  `predictions.csv`, `recommendations.csv`, `user_ranking_metrics.csv`,
  `popularity_user_metrics.csv`, `figures/`), `model/model.txt`,
  `model/feature_schema.json`, `inference/reload_verification.json`,
  `run_manifest.json`, `report.md`, `execution.log`.
- `results/predict_results/recommendations.csv`.

Re-running a target replaces its folder. There is no `output.txt`; read `report.md`.

## Tests

`ctest --test-dir build -R rec_itemcf` runs `rec_itemcf_numerical` (finite
predictions for unknown ids, archive round trip, corrupt archive rejection,
ranking-metric fixture) and `rec_itemcf_workflow` (the `--quick` project).

## Key takeaways

- Centre by user mean before measuring similarity or generous raters dominate.
- Shrink similarities computed from few co-raters; two shared ratings are not
  evidence.
- Keep only positive neighbours: negative similarity is noise at this
  overlap level.

## Next module

`03_matrix_factorization` replaces explicit neighbourhoods with learned
latent factors.
