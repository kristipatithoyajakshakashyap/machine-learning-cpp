# DBSCAN

## Problem and core idea

DBSCAN finds clusters as dense regions separated by sparse ones. Two hyper-parameters: a radius `eps` and a minimum point count `min_pts`. A point with at least `min_pts` neighbours within `eps` is a core point. Points inside `eps` of a core point belong to its cluster. The rest are noise. You do not choose the number of clusters up front, and clusters have any shape. The implementation is header-only: it finds every `eps`-neighbourhood with an all-pairs Euclidean scan, then flood-fills from each unvisited core point to assemble the clusters. Border points that touch two components resolve by input order.

## Dataset and why

Palmer Penguins (`helper/data/penguins.csv`, 344 rows, 4 numeric columns, 3 species) in the end-to-end project. Two species overlap in feature space, so the selected radius tends to keep them in one cluster and label part of the third species as noise, which shows you exactly what DBSCAN does when the density assumption fails. The implementation lesson starts with a hand-checkable 1-D line, then a 2-D synthetic set of two blobs plus background noise. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

C++17 (sets, union-find or lambdas), Euclidean distance, and `01_kmeans` (you compare the two algorithms side by side). Read `theory.md`, `math_intuition.md` and `implementation.md`, then run the numbered lessons in order.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `DBSCAN.hpp` | interface library `ml_dbscan` | Header-only `ml::DBSCAN`: all-pairs Euclidean neighbourhood search, flood-fill clustering, noise label -1, `fit`, `labels`, `core_samples`, save/load | - |
| `01_theory.cpp` | `udb_theory` | Density, the `eps`-neighbourhood, core/border/noise | prints only |
| `02_math_intuition.cpp` | `udb_math_intuition` | Neighbourhood sizes by hand, core connectivity, border ties | prints only |
| `03_implementation.cpp` | `udb_implementation` | Four points on a line, then a 2-D synthetic set. An `eps` x `min_samples` sweep | `results/03_implementation_results/`: `labels.csv`, `sweep.csv`, `figures/dbscan_scatter.svg` |
| `04_end_to_end.cpp` | `udb_end_to_end` | Full project on penguins: impute, standardise, grid 4 `eps` x 3 `min_samples` = 12 candidates scored by the silhouette of their non-noise rows with a >= 80% coverage floor (there is no `--quick` flag) | `results/04_end_to_end_results/` (see below) |
| `CMakeLists.txt` | - | Registers the targets above. Interface library `ml_dbscan` headers only | - |

### Why there is no `udb_predict`

A new row would need a rule for reaching an existing cluster, and DBSCAN has none that is stable: labels depend on the whole neighbourhood structure. So this module has no `predict` target. Reassigning new rows is left to `01_kmeans`, `05_gaussian_mixture` and `06_pca`.

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target udb_end_to_end
build\03_ml_course\02_unsupervised\04_dbscan\udb_end_to_end            # full run, no --quick flag
```

The end-to-end program has no `--quick` flag, because a full run on 344 rows takes seconds. Every other lesson target runs without arguments.

## Results layout

Each executable owns `results/<source stem>_results/` inside this module. The project run writes:

```
results/04_end_to_end_results/
  data/          row sources and preprocessing notes
  eda/           per-feature summaries and figures
  model/         model_state.txt, preprocessing_state.txt, feature_schema.csv
  validation/    candidate_scores.csv, selected_parameters.json, stability.csv
  evaluation/    metrics.json, assignments.csv, cluster_profiles.csv, figures/
  inference/     reload_verification.json
  run_manifest.json, report.md, execution.log
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R udb
```

`udb_workflow` runs `udb_end_to_end` and fails if any stage or the reload verification fails. `udb_fresh_reload` starts a new process, loads the saved model and preprocessor, recomputes the labels and matches them against `evaluation/assignments.csv`. The unit checks in `../tests/test_unsupervised.cpp` run under `unsupervised_numerical`. There is no `udb_new_rows` CTest entry, because the module has no `predict` target.

## Key takeaways

- `eps` and `min_pts` encode a density. The same pair gives worse results on uneven densities.
- Noise is a first-class label. A model that refuses rows is a feature, not a failure.
- Compare the silhouette of DBSCAN with k-means on the same data to see what density buys you.

## Next module

`05_gaussian_mixture`: soft clusters with probabilities and a principled `k`.