# Hierarchical (agglomerative) clustering

## Method and core idea

Agglomerative clustering starts with every row as its own cluster and repeatedly merges the two
closest clusters until one remains. The sequence of merges is a tree (dendrogram); cutting the tree
at any height gives a flat partition, so k is chosen *after* fitting, not before. "Closest" depends
on the linkage rule: single (nearest pair), complete (farthest pair), average (mean pairwise
distance) or Ward (smallest increase in within-cluster variance). Single linkage follows chains and
elongated shapes; Ward behaves much like k-means and favours compact groups of similar size. The
whole tree costs O(n^2) memory and O(n^2 log n) or O(n^3) time, which limits it to modest n.

## Dataset and why

Palmer Penguins (344 rows, four measurements, missing values kept) for the project: the linkage
choice visibly changes the found groups, and the dendrogram of a few hundred rows is still readable.
The implementation lesson uses Iris with average linkage and compares the result with k-means.

## Prerequisites

`../01_kmeans/` (standardization, silhouette, the shared workflow) and basic distance metrics.
Read `theory.md`, `math_intuition.md` and `implementation.md`, then `exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `Agglomerative.hpp` / `Agglomerative.cpp` | library `ml_aggl` | `ml::Agglomerative`: builds the full merge tree for one linkage, `cut(k)` reads any partition, `linkage_matrix()`, `save`/`load` | - |
| `01_theory.cpp` | `uag_theory` | Prints the lesson text | prints only |
| `02_math_intuition.cpp` | `uag_math_intuition` | Linkage definitions on the four-point fixture | prints only |
| `03_implementation.cpp` | `uag_implementation` | Average linkage on Iris, 3 clusters, agreement with species printed | prints only |
| `04_end_to_end.cpp` | `uag_end_to_end` | Full project: EDA, impute + standardize, 4 linkages x k = 2..6 scored by silhouette, fit, evaluate, stability, snapshot, reload, dendrogram | `results/04_end_to_end_results/` |

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target uag_end_to_end
build\03_ml_course\02_unsupervised\02_agglomerative\uag_end_to_end.exe
```

No `--quick` flag is needed; the run takes seconds.

### Why there is no `uag_predict`

A merge tree is defined only over the rows it was built from. Adding a new row would change the
distance matrix and therefore the merges, so there is no principled "assign this row" operation
without refitting. The saved `model/model_state.txt` therefore stores the fitted tree and labels;
the `uag_fresh_reload` test reloads it in a separate process and checks that `cut(k)` reproduces
`evaluation/assignments.csv`. If you need to label new rows, use the cluster centroids with
`../01_kmeans/` or a mixture model (`../05_gaussian_mixture/`).

## Results layout

`results/04_end_to_end_results/` contains `run_manifest.json`, `report.md`, `execution.log`,
`data/`, `eda/`, `validation/{candidate_scores.csv, selected_parameters.json, stability.csv}`,
`evaluation/{assignments.csv, metrics.json, cluster_profiles.csv, linkage.csv, figures/*.svg,
figures/dendrogram.svg}`, `model/{model_state.txt, preprocessing_state.txt, feature_schema.csv}`
and `inference/reload_verification.json`. The reload test adds
`results/verify_reload_results/fresh_process_verification.json`.

## Tests

`ctest --preset course -R uag` runs `uag_workflow` and `uag_fresh_reload`.
`ctest --preset course -R unsupervised_numerical` covers the unit checks in
`../tests/test_unsupervised.cpp`: the four-point fixture yields a three-row linkage matrix, the
expected partition and the hand-computed final merge distance, `cut(1)` gives one cluster, and
save/load reproduces `cut(2)`.

## Key takeaways

- One fit gives every k; inspect the dendrogram before committing to a cut.
- Linkage is a modelling choice: single finds chains, Ward finds blobs.
- No native prediction for new rows; that is a property of the method, not a missing feature.

## Next

`../03_cluster_evaluation/` compares k-means and Ward on the same rows with internal metrics only.
