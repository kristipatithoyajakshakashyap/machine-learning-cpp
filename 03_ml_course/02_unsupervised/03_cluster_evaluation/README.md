# Cluster evaluation and silhouette

## Method and core idea

Without labels there is no accuracy, so clusterings are judged by *internal* criteria computed from
the data alone. The silhouette of a row compares its mean distance to its own cluster (a) with its
mean distance to the nearest other cluster (b): s = (b - a) / max(a, b) lies in [-1, 1], and the
mean over all rows rewards tight, well-separated groups. Inertia (within-cluster sum of squares)
only ever decreases with k, so it needs an "elbow" heuristic; the silhouette peaks at a sensible k.
Subsample stability (refit on 80% of the rows and compare with the adjusted Rand index) tells you
whether a partition is a property of the data or of one particular sample. This module applies these
tools to compare two algorithms on identical rows.

## Dataset and why

Palmer Penguins, standardized once and shared by both k-means and Ward agglomerative so that their
silhouettes live in the same feature space. The implementation lesson uses a synthetic three-blob
set where the correct k is known, so the silhouette-versus-k curve can be trusted.

## Prerequisites

`../01_kmeans/` and `../02_agglomerative/` (this module links both libraries).
Read `theory.md`, `math_intuition.md`, `implementation.md`, then `exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `01_theory.cpp` | `uce_theory` | Prints the lesson text | prints only |
| `02_math_intuition.cpp` | `uce_math_intuition` | Silhouette by hand on the four-point fixture (mean 47/63) | prints only |
| `03_implementation.cpp` | `uce_implementation` | Hand example, then k-means for k = 2..6 on three synthetic blobs with the silhouette curve | `results/03_implementation_results/{silhouette_per_k.csv, silhouette_samples.csv, figures/silhouette_per_k.svg, figures/clusters_best_k.svg}` |
| `04_end_to_end.cpp` | `uce_end_to_end` | Full project: shared preprocessing, k-means vs Ward for k = 2..6, comparison figure, k chosen by k-means silhouette, final fit, stability, snapshot, reload | `results/04_end_to_end_results/` |

The silhouette itself lives in `helper/math/metrics.hpp` and the ARI plus the `valid_silhouette`
guard in `../workflow.hpp`, because every clustering module uses them.

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target uce_end_to_end uce_implementation
build\03_ml_course\02_unsupervised\03_cluster_evaluation\uce_implementation.exe
build\03_ml_course\02_unsupervised\03_cluster_evaluation\uce_end_to_end.exe
```

No `--quick` flag; runs take seconds.

### Why there is no `uce_predict`

This module evaluates and compares fits; it does not introduce a new model. The final k-means fit it
saves is identical in kind to `../01_kmeans/`, whose `ukm_predict` shows how new rows are assigned.
Keeping a second copy of that tool here would add nothing.

## Results layout

`results/04_end_to_end_results/` contains `run_manifest.json`, `report.md`, `execution.log`,
`data/`, `eda/`, `validation/{candidate_scores.csv, stability.csv}`,
`evaluation/{assignments.csv, metrics.json, cluster_profiles.csv, figures/comparison.svg,
figures/*.svg}`, `model/{model_state.txt, preprocessing_state.txt, feature_schema.csv}` and
`inference/reload_verification.json`.

## Tests

This module registers no `uce_*` CTest entries of its own (`ctest --preset course -R uce` matches
nothing); its metrics are covered by the shared fixture.
`ctest --preset course -R unsupervised_numerical` checks the silhouette by hand (47/63 on the
fixture), a singleton cluster scoring 0, rejection of 1 or n clusters, the zero-denominator guard,
and that the adjusted Rand index is invariant to label permutation.

## Key takeaways

- Compare algorithms in one standardized feature space or the numbers are not comparable.
- Silhouette picks k; inertia alone cannot. Stability tells you whether to believe the pick.
- External agreement with species is descriptive context, reported only after selection.

## Next

`../04_dbscan/` clusters by density, finds arbitrary shapes and labels sparse rows as noise.
