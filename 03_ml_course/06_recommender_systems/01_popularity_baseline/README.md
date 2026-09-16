# Popularity recommendation

Method. The simplest recommender that must be beaten. `Popularity` learns
the global mean, per-item rating counts and shrunken item means
`(sum_i + 10 * global_mean) / (count_i + 10)`. Rating prediction returns the
shrunken item mean (the global mean for an item never seen in training).
Ranking scores an item by how many people rated it. The user is ignored.

Core idea. Rating accuracy and recommendation quality are different
objectives. The most-rated film is rarely the best-rated one. A
non-personalised baseline that ignores the user entirely is hard to beat on
Recall@10, so every later module reports its numbers next to this one.
`Model.hpp` also defines the `Rating` record, the MovieLens loader and the
model contract that ItemCF and matrix factorization inherit.

## Dataset

MovieLens 100K, checked in as `../data/u.data` (100,000 explicit 1-5 star
ratings by 943 users of 1,682 films, each with a timestamp, provenance and
usage conditions in `../data/PROVENANCE.md` and `../data/README`). Timestamps
allow honest chronological splits. The long-tailed item popularity makes the
baseline meaningful.

## Prerequisites

Vectors and sparse data, means and shrinkage, basic probability.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Why popularity is the reference baseline. | - |
| `math_intuition.md` | - | Shrunken mean worked by hand. | - |
| `implementation.md` | - | Walkthrough of `Popularity` and the loader. | - |
| `exercises.md` | - | Practice questions. | - |
| `Model.hpp` | (header) | `Rating`, `read_ratings`, `Popularity`, model contract. | - |
| `01_theory.cpp` | `rec_popularity_theory` | Lesson text and `recommend()` on a toy set. | prints only |
| `02_math_intuition.cpp` | `rec_popularity_math_intuition` | Shrinkage arithmetic on a toy set. | prints only |
| `03_implementation.cpp` | `rec_popularity_implementation` | Fits on all ratings. 20 most-rated items with counts and shrunken means. | `results/03_implementation_results/` |
| `04_end_to_end.cpp` | `rec_popularity_end_to_end` | Full project: chronological split, rating error and top-10 ranking. | `results/04_end_to_end_results/{full,quick}/` |
| `predict.cpp` | `rec_popularity_predict` | Reloads a saved model and lists ten unseen items for one user. | `results/predict_results/recommendations.csv` |
| `tests/model_test.cpp` | `rec_popularity_tests` | Numerical fixture for the shared recommender stack. | prints only |

## Build and run

From the repository root (MinGW toolchain, `D:/msys64/ucrt64/bin` on PATH):

```powershell
cmake --preset course
cmake --build --preset course --target rec_popularity_implementation rec_popularity_end_to_end rec_popularity_predict
build/03_ml_course/06_recommender_systems/01_popularity_baseline/rec_popularity_implementation.exe
build/03_ml_course/06_recommender_systems/01_popularity_baseline/rec_popularity_end_to_end.exe          # all 100,000 ratings
build/03_ml_course/06_recommender_systems/01_popularity_baseline/rec_popularity_end_to_end.exe --quick  # earliest 12,000 ratings
```

## Recommending for a user

```powershell
rec_popularity_predict --model results/04_end_to_end_results/full/model/model.txt --user 1
```

Writes `results/predict_results/recommendations.csv` with the ten highest-scored
catalog items the user has not rated in training (`user,rank,item,score`).

## Results layout

- `results/03_implementation_results/` - `top20_items.csv`,
  `top20_items_by_count.svg`, `top20_items_mean_rating.svg`.
- `results/04_end_to_end_results/full/` (or `quick/`) - `data/` (quality report,
  chronological row assignments), `eda/`, `validation/candidate_scores.csv`,
  `evaluation/` (`metrics.json`, `predictions.csv`, `recommendations.csv`,
  per-user ranking metrics for the model and the popularity reference,
  `figures/rating_errors.svg`, `figures/ranking.svg`), `model/model.txt`,
  `model/feature_schema.json`, `inference/reload_verification.json`,
  `run_manifest.json`, `report.md`, `execution.log`.
- `results/predict_results/recommendations.csv`.

Re-running a target replaces its folder. There is no `output.txt`. Read `report.md`.

## Tests

`ctest --test-dir build -R rec_popularity` runs `rec_popularity_numerical`
(finite predictions for unknown ids, rated items excluded from top-10, archive
round trip, corrupt archives rejected, perfect Recall@10 / NDCG@10 fixture) and
`rec_popularity_workflow` (the `--quick` project).

## Key takeaways

- Shrink sparse item means toward the global mean or one five-star rating
  tops the chart.
- Rank by count, predict by mean: two statistics, two objectives.
- Always report a personalised model next to this baseline.

## Next module

`02_item_based_collaborative_filtering` personalises the ranking with
item-item similarities computed from co-rating behaviour.