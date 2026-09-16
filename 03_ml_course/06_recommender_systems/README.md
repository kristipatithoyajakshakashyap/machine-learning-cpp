# Recommender systems in C++17

Four modules build explicit-feedback recommenders for MovieLens 100K from
scratch: a popularity baseline, adjusted-cosine item neighbours, biased matrix
factorization, and the chronological rating / ranking evaluation protocol that
judges all of them. Everything is plain C++17 on top of the course helpers.
There are no external ML libraries.

## Modules (read in order)

| Module | Target prefix | Adds | Final project |
|---|---|---|---|
| `01_popularity_baseline` | `rec_popularity` | `Model.hpp`: `Rating`, `read_ratings`, `Popularity` (global / user / item means, history) | Popularity baseline |
| `02_item_based_collaborative_filtering` | `rec_itemcf` | `ItemCF`: adjusted-cosine item similarity, k nearest neighbours | ItemCF (k tuned) |
| `03_matrix_factorization` | `rec_mf` | `MatrixFactorization`: Funk-SVD with biases, seeded SGD | MF (lambda tuned) |
| `04_recommender_evaluation` | `rec_evaluation` | `Workflow.hpp`: `recommend`, Recall@10 / NDCG@10, chronological split, `workflow<Model>()` | MF under the full protocol |

Each module has `theory.md`, `math_intuition.md`, `implementation.md`,
`exercises.md`, a `README.md`, the numbered lesson executables `01_theory`,
`02_math_intuition`, `03_implementation`, `04_end_to_end`, a `predict` tool
and `tests/model_test.cpp`. Target names are `<prefix>_theory`,
`<prefix>_math_intuition`, `<prefix>_implementation`, `<prefix>_end_to_end`,
`<prefix>_predict` and `<prefix>_tests`.

## Shared code

- `04_recommender_evaluation/Workflow.hpp` - `rec_course::workflow<Model>()`,
  the single end-to-end project every module's `04_end_to_end.cpp` calls, plus
  `recommend()` and `ranking()`. It also serves the inference CLI:
  `predict.cpp` simply includes `04_end_to_end.cpp`.
- `01_popularity_baseline/Model.hpp` - the `Rating` record, the `u.data`
  loader and the `Popularity` base class that `ItemCF` and
  `MatrixFactorization` derive from.
- Model contract used by the workflow (documented in
  `01_popularity_baseline/Model.hpp`): `explicit Model(double parameter)`,
  `fit(const std::vector<Rating>&)`, `predict(user, item)` (rating in
  [1, 5]), `rank_score(user, item)` (higher = better), `seen(user, item)`,
  `catalog()`, members `history` and `count`, `save(std::ostream&)`,
  `load(std::istream&)`.

## Dataset

`data/u.data` - MovieLens 100K: 100,000 explicit ratings (1-5 stars) from 943
users on 1,682 films, each with a timestamp. `data/README` and
`data/PROVENANCE.md` carry the upstream attribution. The upstream MovieLens
usage conditions remain applicable. It is small enough to train every model
in seconds yet has real sparsity (about 6% of the user x item grid is
observed), real popularity skew and timestamps that allow honest
chronological validation.

## Validation protocol

Ratings are sorted by timestamp and split into train (first ~60%), validation
(next ~20%) and test (last ~20%). No timestamp is split across a boundary. The
model's single hyper-parameter is selected on validation RMSE, the winner is
refitted on train + validation, and the test period is scored once for RMSE /
MAE and for Recall@10 / NDCG@10 against a popularity baseline on the same
users. The archive is reloaded and its predictions compared with the
in-memory model before the run is declared successful. `--quick` keeps the
earliest 12,000 ratings for smoke coverage only.

## Build, run, test

From the repository root:

```powershell
cmake --preset course
cmake --build --preset course
ctest --preset course -R "rec_"
```

Run a project: `rec_mf_end_to_end` (all ratings) or
`rec_mf_end_to_end --quick` (earliest 12,000).

Recommend ten unseen items for one user:

```powershell
rec_mf_predict --model 03_matrix_factorization/results/04_end_to_end_results/full/model/model.txt --user 1
```

## Results layout

Every executable owns `results/<cpp_filename_stem>_results/` inside its
module: `03_implementation.cpp` writes to `results/03_implementation_results/`,
the project to `results/04_end_to_end_results/full/` (or `quick/`), and the
predict tool to `results/predict_results/recommendations.csv`. Paths are
compiled in, so the working directory does not matter. A run contains
`data/`, `eda/`, `validation/`, `evaluation/` (metrics, predictions,
recommendations, per-user ranking metrics, figures), `model/model.txt`,
`model/feature_schema.json`, `inference/`, `run_manifest.json`, `report.md`
and `execution.log`. There is no `output.txt`.

## Tests

`<prefix>_numerical` compiles the shared fixture once per module.
`<prefix>_workflow` runs the `--quick` project. All of them are part of
`ctest --preset course`.