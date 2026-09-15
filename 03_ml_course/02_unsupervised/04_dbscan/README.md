# DBSCAN

## Method and core idea

DBSCAN (Ester et al., 1996) clusters by density instead of by distance to a centre. A row is a
*core* point if at least `min_samples` rows (itself included) lie within radius `eps`; core points
whose neighbourhoods overlap are chained into one cluster, non-core rows inside a core
neighbourhood become *border* members, and everything else is *noise* (label -1). Consequently the
number of clusters is not chosen up front, clusters can have any shape, and outliers are excluded
rather than forced into a group. The price is two knobs that interact: a small `eps` fragments the
data into noise, a large one merges everything.

## Dataset and why

Palmer Penguins, median-imputed and standardized: `eps` is a single radius applied to every feature
at once, so unequal units would make it meaningless. The implementation lesson uses a 1-D line that
can be checked by hand and a 2-D set of two dense blobs on uniform background noise, where the
noise label is obviously correct.

## Prerequisites

`../01_kmeans/` and `../03_cluster_evaluation/` (silhouette, stability). Read `theory.md`,
`math_intuition.md` and `implementation.md`, then `exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `DBSCAN.hpp` | interface library `ml_dbscan` | Header-only `ml::DBSCAN`: neighbourhood search, core/border/noise labelling, `labels()`, `core_samples()`, `save`/`load` | - |
| `01_theory.cpp` | `udb_theory` | Prints the lesson text | prints only |
| `02_math_intuition.cpp` | `udb_math_intuition` | Core/border/noise on the points 0, 2, 8, 10 (+30) with eps 2.1 | prints only |
| `03_implementation.cpp` | `udb_implementation` | 1-D sanity check, 2-D blobs plus noise, and an eps x min_samples sweep | `results/03_implementation_results/{labels.csv, sweep.csv, figures/dbscan_scatter.svg}` |
| `04_end_to_end.cpp` | `udb_end_to_end` | Full project: EDA, impute + standardize, 4 eps x 3 min_samples scored by silhouette with an 80% coverage floor, fit, evaluate (noise excluded), stability, snapshot, reload | `results/04_end_to_end_results/` |

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target udb_end_to_end udb_implementation
build\03_ml_course\02_unsupervised\04_dbscan\udb_implementation.exe
build\03_ml_course\02_unsupervised\04_dbscan\udb_end_to_end.exe
```

No `--quick` flag; runs take seconds.

### Why there is no `udb_predict`

DBSCAN labels are a property of the whole neighbourhood graph. A new row could be called "noise" or
"member of cluster c" only by re-running the density reachability with that row inserted, which can
turn other rows from noise into border points and even merge clusters. The method therefore has no
stable single-row prediction; the project saves the fitted rows, labels and core flags, and the
`udb_fresh_reload` test checks in a separate process that the reloaded object returns the same
labels. For scoring new rows by "how far from dense regions", see `../../03_anomaly_detection/`.

## Results layout

`results/04_end_to_end_results/` contains `run_manifest.json`, `report.md`, `execution.log`,
`data/`, `eda/`, `validation/{candidate_scores.csv, selected_parameters.json, stability.csv}`,
`evaluation/{assignments.csv, metrics.json, cluster_profiles.csv, figures/*.svg}` (noise rows are
excluded from the silhouette and the noise share is reported in `metrics.json`),
`model/{model_state.txt, preprocessing_state.txt, feature_schema.csv}` and
`inference/reload_verification.json`.

## Tests

`ctest --preset course -R udb` runs `udb_workflow` and `udb_fresh_reload`.
`ctest --preset course -R unsupervised_numerical` includes the DBSCAN checks in
`../tests/test_unsupervised.cpp`: the fixture plus a far point gives the expected density labels
with the far point marked noise and not core.

## Key takeaways

- `eps` only makes sense after standardization; sweep it and watch the noise fraction.
- Noise is a first-class answer. Excluding it from the silhouette is required for a fair score.
- No native prediction for new rows, by construction of the method.

## Next

`../05_gaussian_mixture/` gives soft, probabilistic memberships and a principled way (BIC) to pick
the number of components.
