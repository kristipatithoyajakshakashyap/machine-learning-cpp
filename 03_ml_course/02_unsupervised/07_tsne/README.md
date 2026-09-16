# t-SNE

## Problem and core idea

t-SNE embeds the data in 2-D so that nearby rows stay nearby and distant rows stay distant. It turns pairwise distances into symmetric probabilities: a Student-t kernel measures similarity in the embedding, and gradient descent minimizes the Kullback-Leibler divergence between the two probability matrices. The perplexity parameter sets the number of neighbours each row tries to preserve. The result preserves local structure, and it does not preserve global distances, so the 2-D picture is for visualisation, not for inference.

## Dataset and why

Iris (`helper/data/iris.csv`, 150 rows, 4 columns) for the implementation lesson and Palmer Penguins (`helper/data/penguins.csv`, 344 rows, 4 numeric columns) in the end-to-end project. Both have three groups that PCA already separates, but t-SNE shows them pressed into tight, clearly separated islands, which is its signature behaviour and worth seeing on data you know. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

C++17 (gradient descent, `std::function`), the Kullback-Leibler divergence, and `06_pca` (the comparison embedding). Read `theory.md`, `math_intuition.md` and `implementation.md`, then run the numbered lessons in order.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | The KL cost and one gradient step, by hand | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `TSNE.hpp` | interface library `ml_tsne` | Header-only `ml::TSNE`: perplexity binary search, symmetrized probability matrix, gradient descent with momentum, save/load | - |
| `01_theory.cpp` | `utsne_theory` | Perplexity, the Student-t kernel, the KL cost | prints only |
| `02_math_intuition.cpp` | `utsne_math_intuition` | One gradient update on a 3-point fixture | prints only |
| `03_implementation.cpp` | `utsne_implementation` | Iris embedded at perplexity 30 for 500 iterations. KL divergence per iteration makes the early-exaggeration phase visible | `results/03_implementation_results/`: `embedding.csv`, `kl_per_iteration.csv`, `figures/embedding.svg`, `figures/kl_per_iteration.svg` |
| `04_end_to_end.cpp` | `utsne_end_to_end` | Full project on penguins: impute, standardise, embed 344 rows for perplexities {5, 15, 30}, select by trustworthiness (k = 10), save the state and refit-from-reload to prove bit-identical coordinates (there is no `--quick` flag) | `results/04_end_to_end_results/` (see below) |
| `CMakeLists.txt` | - | Registers the targets above. Interface library `ml_tsne` headers only | - |

### Why there is no `utsne_predict`

The embedding is a mapping of the training rows, not a function: each run optimizes a stochastic objective and re-relocates every point, so there is no well-defined `transform` for unseen rows. This module is for visualisation. For a reproducible out-of-sample projection, use `06_pca`.

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target utsne_end_to_end
build\03_ml_course\02_unsupervised\07_tsne\utsne_end_to_end            # full run, no --quick flag
```

The end-to-end program has no `--quick` flag, because a full run on 344 rows takes seconds. Every other lesson target runs without arguments.

## Results layout

Each executable owns `results/<source stem>_results/` inside this module. The project run writes:

```
results/04_end_to_end_results/
  data/          row_assignments.csv, preprocessing notes
  eda/           per-feature summaries and figures
  model/         model_state.txt, preprocessing_state.txt, feature_schema.csv
  validation/    candidate_scores.csv, selection_rule.md
  evaluation/    metrics.json, embedding.csv, figures/embedding.svg
  inference/     reload_verification.json
  run_manifest.json, report.md, execution.log
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R utsne
```

`utsne_workflow` runs `utsne_end_to_end` and fails if any stage or the reload verification fails. `utsne_fresh_reload` starts a new process, loads the saved seed and parameters, refits on the same rows and matches the coordinates against `evaluation/embedding.csv`. The unit checks in `../tests/test_unsupervised.cpp` run under `unsupervised_numerical`. There is no `utsne_new_rows` CTest entry, because the module has no `predict` target.

## Key takeaways

- t-SNE preserves local neighbourhoods. Distances between islands carry no meaning.
- Perplexity is a knob on how many neighbours count as local. 5 to 50 is the usual range.
- The same data re-run looks different. That is the cost history, not a bug. Embeddings are for eyes, not for numbers.

## Next module

This completes the track. `../03_anomaly_detection/` takes the same workflow classes to Isolation Forest, LOF and One-class SVM.