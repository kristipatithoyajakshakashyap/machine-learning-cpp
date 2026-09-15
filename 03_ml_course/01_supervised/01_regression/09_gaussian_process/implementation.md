# Implementation

`course::GaussianProcess` in Model.hpp stores the training inputs, the Cholesky factor L and alpha. `fit` builds the kernel matrix with the noise term on the diagonal, factorises it with a hand-written Cholesky (dense O(n^3)), solves for alpha with one forward and one backward substitution and caches the log marginal likelihood. `predict` returns the posterior mean; `predict_std` returns the latent posterior standard deviation (without the noise term) using one forward solve per query row. `save`/`load` use the shared archive with tag `GaussianProcess_V1`.

The pipeline parameter is the length scale; signal variance 1 and noise variance 0.1 match the standardised features and scaled target produced by the shared preprocessing. `04_end_to_end.cpp` runs `ml::run_supervised` on a 300-row diabetes subsample and then reloads model, preprocessing and target transform to write `evaluation/predictive_std.csv`. `03_implementation.cpp` fits 15 noisy sin(x) points and writes the posterior band and the marginal-likelihood curve as CSV + SVG.

The solver is educational: no hyperparameter gradients, no sparse approximations, and cost grows cubically with the number of rows.
