# Gaussian mixtures

## Method and core idea

A Gaussian mixture model (GMM) explains the data as a weighted sum of k Gaussian components,
p(x) = sum_c w_c N(x | mu_c, Sigma_c). Unlike k-means it gives every row a *responsibility*, the
posterior probability of belonging to each component, so memberships are soft and the model is a
density, not just a partition. Expectation-Maximisation fits it: the E-step computes
responsibilities from the current parameters, the M-step re-estimates weights, means and variances
from the responsibilities, and the log-likelihood never decreases. This module uses diagonal
covariances (axis-aligned ellipses), which keeps every step O(n k p) and avoids singular matrices;
a small variance floor guards against a component collapsing onto one point. Because a mixture is a
likelihood model, the number of components can be chosen with BIC, which penalises parameters.

## Dataset and why

Palmer Penguins: the species overlap on some measurements, so soft memberships are more honest
than hard labels, and the count of components is genuinely uncertain, which is what BIC is for. The
implementation lesson uses a four-point fixture and a synthetic 2-D mixture of three components
where the log-likelihood trajectory can be checked to be non-decreasing.

## Prerequisites

`../01_kmeans/` (EM's M-step generalises the centroid update), Gaussian densities, logarithms.
Read `theory.md`, `math_intuition.md` and `implementation.md`, then `exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `GaussianMixture.hpp` | interface library `ml_gmm` | Header-only `ml::GaussianMixture`: EM with k, iterations, seed and variance floor; `predict_proba`, `predict`, `score`, `bic`, `aic`, `save`/`load` | - |
| `01_theory.cpp` | `ugm_theory` | Prints the lesson text | prints only |
| `02_math_intuition.cpp` | `ugm_math_intuition` | E- and M-step formulas on the four-point fixture | prints only |
| `03_implementation.cpp` | `ugm_implementation` | Fixture, then a 3-component synthetic mixture; log-likelihood per iteration, responsibilities, BIC/AIC | `results/03_implementation_results/{log_likelihood.csv, responsibilities.csv, figures/log_likelihood.svg, figures/components_scatter.svg}` |
| `04_end_to_end.cpp` | `ugm_end_to_end` | Full project: EDA, impute + standardize, k = 2..6 x 3 seeds selected by BIC, fit, evaluate hard labels, stability, snapshot, reload | `results/04_end_to_end_results/` |
| `predict.cpp` | `ugm_predict` | Reloads the saved preprocessor and mixture and writes per-component probabilities for new rows | `results/predict_results/new_predictions.csv` |

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target ugm_end_to_end ugm_predict
build\03_ml_course\02_unsupervised\05_gaussian_mixture\ugm_end_to_end.exe
build\03_ml_course\02_unsupervised\05_gaussian_mixture\ugm_predict.exe new_rows.csv
```

No `--quick` flag. `ugm_predict <numeric_csv>` takes a header-less CSV in the order of
`model/feature_schema.csv`, raw units, blanks/`NA`/`nan` allowed; the output columns
`output_0..output_{k-1}` are responsibilities that sum to 1 per row. Use
`../tests/penguin_inference.csv` as an example input.

## Results layout

`results/04_end_to_end_results/` contains `run_manifest.json`, `report.md`, `execution.log`,
`data/`, `eda/`, `validation/{candidate_scores.csv, selected_parameters.json, stability.csv}`,
`evaluation/{assignments.csv, metrics.json, cluster_profiles.csv, figures/*.svg}`,
`model/{model_state.txt, preprocessing_state.txt, feature_schema.csv}` and
`inference/reload_verification.json`. `ugm_predict` writes `results/predict_results/`; the reload
test writes `results/verify_reload_results/fresh_process_verification.json`.

## Tests

`ctest --preset course -R ugm` runs `ugm_workflow`, `ugm_fresh_reload` and `ugm_new_rows`.
`ctest --preset course -R unsupervised_numerical` includes the mixture checks in
`../tests/test_unsupervised.cpp`: responsibilities sum to 1, BIC is finite, save/load reproduces
`predict`, and the variance floor keeps a degenerate (all-identical) fit finite.

## Key takeaways

- Soft memberships expose ambiguity that hard clustering hides.
- EM is a local optimiser: run several seeds and keep the best likelihood.
- BIC turns "how many clusters" into a model-comparison question with a defensible answer.

## Next

`../06_pca/` moves from grouping rows to compressing columns.
