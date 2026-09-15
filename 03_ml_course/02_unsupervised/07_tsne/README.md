# t-SNE

## Method and core idea

t-distributed Stochastic Neighbour Embedding maps high-dimensional rows to 2-D so that *local
neighbourhoods* are preserved. In the input space each row gets a Gaussian distribution over its
neighbours whose width is tuned per row until its entropy matches a chosen *perplexity* (roughly
"how many neighbours count"); these are symmetrised into joint affinities P. In the 2-D map the
same pairs get Student-t affinities Q, whose heavy tail lets dissimilar rows sit far apart. Gradient
descent on the Kullback-Leibler divergence KL(P || Q), with early exaggeration and momentum, moves
the points until neighbours in P are neighbours in Q. What t-SNE does *not* preserve: cluster
sizes, distances between clusters and the meaning of the axes, so an embedding is read
qualitatively. Trustworthiness (k = 10) measures how many map neighbours were genuine neighbours in
the input space and is used here to choose the perplexity.

## Dataset and why

Palmer Penguins: three species with partial overlap make a good test of whether local structure
survives the embedding, and 344 rows keep the O(n^2) affinity computation fast. The implementation
lesson embeds Iris and records the KL divergence per iteration so the effect of early exaggeration
is visible.

## Prerequisites

`../06_pca/` (the linear counterpart), softmax-style probabilities, gradient descent, KL divergence.
Read `theory.md`, `math_intuition.md` and `implementation.md`, then `exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `TSNE.hpp` | interface library `ml_tsne` | Header-only `ml::TSNE`: perplexity binary search, joint affinities, Student-t gradient with early exaggeration and momentum, `embedding()`, `save`/`load` | - |
| `01_theory.cpp` | `utsne_theory` | What t-SNE preserves (local neighbourhoods) and what it does not (sizes, distances, axes) | prints only |
| `02_math_intuition.cpp` | `utsne_math_intuition` | Prints P, Q, KL and gradient formulas and checks a three-point perplexity example numerically | prints only |
| `03_implementation.cpp` | `utsne_implementation` | Iris embedded to 2-D (perplexity 30) with the KL curve | `results/03_implementation_results/{embedding.csv, kl_per_iteration.csv, figures/embedding.svg, figures/kl_per_iteration.svg}` |
| `04_end_to_end.cpp` | `utsne_end_to_end` | Full project: EDA, impute + standardize all rows, perplexity grid {5, 15, 30} scored by trustworthiness, snapshot, reload-and-refit verification | `results/04_end_to_end_results/` |

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target utsne_end_to_end utsne_implementation
build\03_ml_course\02_unsupervised\07_tsne\utsne_implementation.exe
build\03_ml_course\02_unsupervised\07_tsne\utsne_end_to_end.exe
```

No `--quick` flag; the project runs three embeddings and finishes in seconds.

### Why there is no `utsne_predict`

t-SNE optimises the 2-D coordinates of the rows it was given; there is no function that maps a new
row into an existing map (a new row would change P for every other row and the whole embedding
would have to be re-optimised). The saved `model/model_state.txt` therefore holds the parameters,
the seed and the coordinates. Reproducibility is proven differently: the `utsne_fresh_reload` test
reloads that state in a separate process, refits on the same preprocessed rows and requires the
coordinates to match `evaluation/embedding.csv` bit for bit. To place new rows in a 2-D map use
`../06_pca/`, which has a true transform.

## Results layout

`results/04_end_to_end_results/` contains `run_manifest.json`, `report.md`, `execution.log`,
`data/row_assignments.csv`, `eda/`, `validation/{candidate_scores.csv, selection_rule.md}`
(perplexity, final KL and trustworthiness per candidate, selected row flagged),
`evaluation/{embedding.csv, metrics.json, figures/embedding.svg}` (species column is a reading
aid only), `model/{model_state.txt, preprocessing_state.txt, feature_schema.csv}` and
`inference/reload_verification.json`. The reload test writes
`results/verify_reload_results/fresh_process_verification.json`.

## Tests

`ctest --preset course -R utsne` runs `utsne_workflow` and `utsne_fresh_reload`.
`ctest --preset course -R unsupervised_numerical` includes the t-SNE checks in
`../tests/test_unsupervised.cpp`: the embedding has shape n x 2, three well-separated Gaussian
blobs stay separated, trustworthiness is high, save/load is exact and the same seed reproduces the
same coordinates.

## Key takeaways

- Read a t-SNE map for neighbourhoods, never for cluster sizes or inter-cluster distances.
- Perplexity is the main knob; compare several values and score them (trustworthiness).
- No out-of-sample transform; reproducibility comes from a fixed seed and saved state.

## Next

`../../03_anomaly_detection/` applies unsupervised scoring to find rows that do not belong.
