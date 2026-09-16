# Agglomerative Clustering

## Problem and core idea

Agglomerative clustering builds a hierarchy from the bottom up: every row starts as its own cluster, and each step merges the two closest clusters until one cluster remains. The result is a dendrogram, and `k` is chosen after fitting by cutting it at the right height. Linkage defines "closest". Single linkage follows chains and elongated shapes. Ward behaves much like k-means on the average. Complete and average linkage sit between them. The implementation builds the full merge tree once, then `cut(n_clusters)` reads any `k` off it without refitting. Ties at the same merge distance are broken by the lowest cluster ids, so results are deterministic.

## Dataset and why

Iris (`helper/data/iris.csv`, 150 rows, 4 columns, 3 known species) for the implementation lesson and Palmer Penguins (`helper/data/penguins.csv`, 344 rows, 4 numeric columns, 3 species) for the end-to-end project. Iris appears in `01.knn` too, so you already know its geometry. Penguins mix units and have three nested groups, which makes the choice of linkage visibly different in the dendrogram. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

C++17 (priority queues, `std::function`), Euclidean distance, and `01_kmeans` (silhouette is reused to cut the dendrogram). Read `theory.md`, `math_intuition.md` and `implementation.md`, then run the numbered lessons in order.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `Agglomerative.hpp / Agglomerative.cpp` | library `ml_aggl` | `ml::AgglomerativeClustering`: single/complete/average/Ward linkage, `fit`, `cut(n_clusters)`, `labels`, `linkage_matrix`, `save`/`load` | - |
| `01_theory.cpp` | `uag_theory` | Bottom-up merging, linkage, the dendrogram | prints only |
| `02_math_intuition.cpp` | `uag_math_intuition` | Four points under single, complete, average and Ward linkage, by hand | prints only |
| `03_implementation.cpp` | `uag_implementation` | Iris, k=3, average linkage: cluster sizes, species best-match accuracy, merges, silhouette | `results/03_implementation_results/`: `iris_average.svg`, `iris_agglomerative_avg.labels` |
| `04_end_to_end.cpp` | `uag_end_to_end` | Full project on penguins: shared `workflow.hpp` helpers: impute, standardise, grid 4 linkages x `k` in {2, 3, 4, 5, 6} scored by silhouette, final fit, save and reload-check, linkage matrix and dendrogram | `results/04_end_to_end_results/` (see below) |
| `CMakeLists.txt` | - | Registers the targets above. Module library `ml_aggl (Agglomerative.cpp)` | - |

### Why there is no `uag_predict`

The dendrogram is built from the training rows as a whole. A new row would need a well-defined distance to a tree, and no single rule for that distance is standard or checkable. So this module has no `predict` target. `transform`/new-row assignment is left to `06_pca` and `07_tsne`. The saved snapshot stores the merge tree and the fitted labels. `uag_fresh_reload` recomputes them in a new process.

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target uag_end_to_end
build\03_ml_course\02_unsupervised\02_agglomerative\uag_end_to_end            # full run, no --quick flag
```

The end-to-end program takes no arguments, because a full run on 344 rows takes seconds. Every other lesson target runs without arguments.

## Results layout

Each executable owns `results/<source stem>_results/` inside this module. The project run writes:

```
results/04_end_to_end_results/
  data/          row sources and preprocessing notes
  eda/           per-feature summaries and figures
  model/         model_state.txt, preprocessing_state.txt, feature_schema.csv
  validation/    candidate_scores.csv, selected_parameters.json, stability.csv
  evaluation/    metrics.json, assignments.csv, cluster_profiles.csv
  evaluation/    linkage.csv, figures/ (clusters, silhouette, dendrogram)
  inference/     reload_verification.json
  run_manifest.json, report.md, execution.log
```

`evaluation/figures/` holds `clusters.svg`, `silhouette.svg`, `silhouette_distribution.svg` and `dendrogram.svg`, written by the shared report helper and the module's dendrogram stage.

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R uag
```

`uag_workflow` runs `uag_end_to_end` and fails if any stage or the reload verification fails. `uag_fresh_reload` starts a new process, loads the saved tree and preprocessor, recomputes the labels and matches them against `evaluation/assignments.csv`. The unit checks in `../tests/test_unsupervised.cpp` run under `unsupervised_numerical`. Because there is no `predict` target, there is no `uag_new_rows`.

## Key takeaways

- The dendrogram shows the whole merge order, not a single partition. Cut it at the height the silhouette likes.
- Single linkage finds chains and chains noise. Complete linkage makes compact balls but splits them.
- Building the tree once and cutting it at any `k` beats refitting for each `k`.

## Next module

`03_cluster_evaluation`: how to choose among partitions and algorithms with no labels.