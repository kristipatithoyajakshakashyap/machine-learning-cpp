# K-means

## Method and core idea

K-means partitions rows into k groups around k centroids. Lloyd's algorithm alternates two steps:
assign every row to its nearest centroid, then move each centroid to the mean of its members. Each
step lowers the within-cluster sum of squares (inertia), so the loop converges to a local optimum.
Because the result depends on the starting centroids, this module seeds with k-means++ (later
centroids are sampled far from existing ones) and keeps the best of several seeded restarts. K-means
assumes roughly spherical, similar-size clusters in Euclidean space, which is why feature scaling is
part of the method rather than an afterthought.

## Dataset and why

Palmer Penguins (`helper/data/penguins.csv`, 344 rows, four numeric measurements) has three real
species that the algorithm never sees, missing values that must be imputed, and features in
different units (millimetres and grams). It shows why standardization matters and lets the report
compare the found clusters with species descriptively, after selection. The implementation lesson
uses Iris (150 rows) and a four-point fixture that can be checked by hand.

## Prerequisites

Euclidean distance, arithmetic means and variance, and the `helper/` types (`Mat`, `Vec`,
`Preprocessor`, `Artifacts`). Read `theory.md`, `math_intuition.md` and `implementation.md` first,
then `exercises.md` after running the lessons.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material, in that order | - |
| `KMeans.hpp` / `KMeans.cpp` | library `ml_kmeans` | `ml::KMeans`: k-means++ seeding, Lloyd iterations, `n_init` restarts, `predict` (nearest centroid), `save`/`load` | - |
| `01_theory.cpp` | `ukm_theory` | Prints the lesson text | prints only |
| `02_math_intuition.cpp` | `ukm_math_intuition` | Objective, centroid update and the four-point inertia example | prints only |
| `03_implementation.cpp` | `ukm_implementation` | k = 3 on Iris; clusters are best-matched to species and the agreement is printed | prints only |
| `04_end_to_end.cpp` | `ukm_end_to_end` | Full project: EDA, impute + standardize, choose k in 2..6 by silhouette, fit, evaluate, stability, snapshot, reload check | `results/04_end_to_end_results/` |
| `predict.cpp` | `ukm_predict` | Reloads the saved preprocessor and model and assigns new rows to clusters | `results/predict_results/new_predictions.csv` |

## Build and run

From the repository root with `D:\msys64\ucrt64\bin` on `PATH`:

```powershell
cmake --preset course
cmake --build --preset course --target ukm_end_to_end ukm_predict
build\03_ml_course\02_unsupervised\01_kmeans\ukm_end_to_end.exe
build\03_ml_course\02_unsupervised\01_kmeans\ukm_predict.exe new_rows.csv
```

The end-to-end program has no `--quick` flag (a full run takes seconds). `ukm_predict <numeric_csv>`
expects a header-less CSV with the four features in the order of
`results/04_end_to_end_results/model/feature_schema.csv`, in raw units; blank, `NA` or `nan` fields
are imputed with the saved training medians. Output column `output_0` is the cluster id.
`03_ml_course/02_unsupervised/tests/penguin_inference.csv` is a ready-made example.

## Results layout

`results/04_end_to_end_results/` contains `run_manifest.json`, `report.md`, `execution.log`,
`data/`, `eda/`, `validation/{candidate_scores.csv, selected_parameters.json, stability.csv}`,
`evaluation/{assignments.csv, metrics.json, cluster_profiles.csv, figures/*.svg}`,
`model/{model_state.txt, preprocessing_state.txt, feature_schema.csv}` and
`inference/reload_verification.json`. The `ukm_fresh_reload` test adds
`results/verify_reload_results/fresh_process_verification.json`.

## Tests

`ctest --preset course -R ukm` runs `ukm_workflow` (the project), `ukm_fresh_reload` (reload the
saved model in a new process and compare assignments) and `ukm_new_rows` (`ukm_predict` on the
inference CSV). `ctest --preset course -R unsupervised_numerical` runs the unit checks in
`../tests/test_unsupervised.cpp`: the four-point fixture gives centroids 1 and 9 with inertia 4,
duplicate points give zero inertia, and save/load reproduces labels exactly.

## Key takeaways

- Inertia always falls as k grows, so it cannot pick k by itself; the silhouette can.
- Restarts with k-means++ seeding remove most of the luck from the initialisation.
- Persist the preprocessor with the centroids: new rows must be imputed and scaled identically.

## Next

`../02_agglomerative/` builds a hierarchy instead of a flat partition and removes the need to fix k
before fitting.
