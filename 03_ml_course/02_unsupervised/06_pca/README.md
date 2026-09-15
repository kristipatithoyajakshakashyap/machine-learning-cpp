# Principal component analysis

## Method and core idea

PCA finds the orthogonal directions (principal components) along which standardized data varies
most, ordered by variance. Projecting rows onto the first k components gives the k-dimensional
representation with the smallest possible squared reconstruction error, and the explained-variance
ratios say how much of the total variance each component keeps. The components are the eigenvectors
of the covariance matrix; `PCA.hpp` computes them with cyclic Jacobi rotations, which are exact
for symmetric matrices of this size. Because PCA is a linear map fitted on training rows, it has a true out-of-sample
transform: new rows are centred, scaled and multiplied by the same loadings. PCA is sensitive to
scale (a feature in grams would own the first component), which is why standardization is part of
the saved preprocessing.

## Dataset and why

Palmer Penguins with a proper 80/20 split: PCA is often the first step of a supervised pipeline,
so the project demonstrates fitting imputation, scaling and the loadings on training rows only and
reporting the reconstruction error on unseen rows. The implementation lesson uses Iris to show a
scree plot and a 2-D score scatter, plus a four-point line that collapses to one component.

## Prerequisites

Variance and covariance, matrix-vector products, eigenvectors at an intuitive level.
Read `theory.md`, `math_intuition.md` and `implementation.md`, then `exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `PCA.hpp` | interface library `ml_pca` | Header-only `ml::PCA`: `fit`, `transform`, `inverse_transform`, `explained_variance_ratio`, `components`, `save`/`load` | - |
| `01_theory.cpp` | `upca_theory` | Prints the lesson text | prints only |
| `02_math_intuition.cpp` | `upca_math_intuition` | Covariance, eigen-decomposition and the projection worked on the fixture | prints only |
| `03_implementation.cpp` | `upca_implementation` | Four-point line to one component; Iris to two components with scree and score scatter | `results/03_implementation_results/{explained_variance.csv, scores.csv, components.csv, figures/scree.svg, figures/scores_scatter.svg}` |
| `04_end_to_end.cpp` | `upca_end_to_end` | Full project: seeded 80/20 split, training-only EDA and preprocessing, k chosen by a 95% explained-variance rule, hold-out reconstruction error, snapshot, reload | `results/04_end_to_end_results/` |
| `predict.cpp` | `upca_predict` | Reloads the saved preprocessor and loadings and projects new rows | `results/predict_results/new_predictions.csv` |

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target upca_end_to_end upca_predict
build\03_ml_course\02_unsupervised\06_pca\upca_end_to_end.exe
build\03_ml_course\02_unsupervised\06_pca\upca_predict.exe new_rows.csv
```

No `--quick` flag. `upca_predict <numeric_csv>` takes a header-less CSV in the order of
`model/feature_schema.csv`, raw units, blanks/`NA`/`nan` allowed; output columns
`output_0..output_{k-1}` are the component scores in standardized units.

## Results layout

`results/04_end_to_end_results/` contains `run_manifest.json`, `report.md`, `execution.log`,
`data/row_assignments.csv` (train/test membership), `eda/` (training rows only),
`validation/{candidate_scores.csv, selection_rule.md}`,
`evaluation/{projections.csv, metrics.json, figures/explained_variance.svg}`,
`model/{model_state.txt, preprocessing_state.txt, feature_schema.csv}` and
`inference/reload_verification.json`.

## Tests

`ctest --preset course -R upca` runs `upca_workflow`, `upca_fresh_reload` and `upca_new_rows`.
`ctest --preset course -R unsupervised_numerical` includes the PCA checks in
`../tests/test_unsupervised.cpp`: a collinear fixture gives an explained-variance ratio of 1, a
one-component reconstruction of it is exact, and the training mean projects to the origin.

## Key takeaways

- Fit preprocessing and loadings on training rows; evaluate reconstruction on held-out rows.
- Choose k by cumulative explained variance, then look at the scree plot to confirm.
- PCA is linear: it preserves global structure and large distances, not local neighbourhoods.

## Next

`../07_tsne/` embeds the same rows non-linearly to preserve local neighbourhoods instead.
