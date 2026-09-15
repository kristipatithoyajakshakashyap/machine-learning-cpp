# t-SNE: implementation walkthrough

`TSNE.hpp` is header-only in namespace `ml` and depends only on `02_unsupervised/validation.hpp` (`validate_dense`, `distance2`) and `helper/persistence/archive.hpp`.

## Constructor

`TSNE(size_t dims = 2, double perplexity = 30, size_t iterations = 500, double learning_rate = 200, unsigned seed = 42)`. The seed is a model parameter: it fixes the random initial layout and therefore the final picture.

## State

| member | meaning |
|---|---|
| `dims_`, `perplexity_`, `iterations_`, `learning_rate_`, `seed_` | hyper-parameters, all saved |
| `Y_` | embedding, n x dims, in input row order |
| `kl_` | KL(P || Q) of the final iteration (no exaggeration) |
| `kl_history_` | KL per iteration; filled by `fit`, not saved |

## joint_affinities(X)

1. `validate_dense(X)` rejects empty, ragged or non-finite input; a perplexity outside (0, n) throws `invalid_argument`.
2. For every row i compute squared distances to all rows and call `conditional_row`, which binary-searches the bandwidth beta_i (starting at 1, up to 50 halvings, tolerance 1e-5 in entropy) until the row's entropy equals log(perplexity). The upper bracket starts at infinity, so the first sharpening steps double beta.
3. Symmetrise: p_ij = (p_{j|i} + p_{i|j}) / (2n), clamped to at least 1e-12.

The matrix is O(n^2) in memory and time; it is built once per `fit`.

## fit(X)

1. Build P.
2. Initialise `Y_` from N(0, 1e-4) with `std::mt19937(seed_)`.
3. For each iteration t:
   - exaggeration = 12 while t < 100, else 1; momentum = 0.5 while t < 250, else 0.8;
   - compute the Student-t numerators W_ij = (1 + ||y_i - y_j||^2)^-1 and their sum;
   - accumulate KL (always with the *unexaggerated* P, so the history is comparable across the exaggeration switch) and the gradient 4 sum_j (exaggeration * p_ij - q_ij) W_ij (y_i - y_j);
   - update gains, velocity and position per coordinate, then subtract the mean so the embedding stays centred.
4. `kl_` is the last history entry.

Everything is deterministic: same rows, same parameters and same seed give bit-identical coordinates in the same binary, which is what the reload checks rely on.

## Accessors

`embedding()` returns a copy of `Y_`; `kl_divergence()` the final KL; `kl_history()` the per-iteration curve. Both embedding accessors throw `logic_error` before `fit`.

## Persistence

`save` writes the tag `"TSNE_V1"`, then `dims_`, `perplexity_`, `iterations_`, `learning_rate_`, `seed_`, `Y_` and `kl_` through `archive::write` (17 significant digits). `load` reads them back, rejects a wrong tag, invalid parameters or rows of the wrong width, and clears the history. A loaded model can return its stored embedding immediately or be refitted on the same preprocessed rows to reproduce it.

## trustworthiness(X, Y, k = 10)

A free function in the same header. For each row it sorts the other rows by original distance and by embedded distance (stable sort, so ties are deterministic), then adds (rank - k) for every embedded k-neighbour whose original rank exceeds k. It rejects k = 0 or 2n <= 3k + 1, where the normaliser would vanish. Cost is O(n^2 log n).

## End-to-end flow (`04_end_to_end.cpp`)

Load penguins (missing values preserved), write the feature schema, run EDA, fit the `Preprocessor` on all rows (there is no hold-out: t-SNE cannot embed unseen rows), fit one `TSNE` per perplexity in {5, 15, 30}, compute trustworthiness for each, record `validation/candidate_scores.csv` with KL and trustworthiness, select the highest trustworthiness, snapshot model and preprocessor, reload the model, refit it on the same rows and pass each row through `unsup::verify`, then write the embedding, row assignments, figure, metrics, manifest, report and log.

## Complexity and limits

- Affinities: O(n^2 d) plus 50 binary-search passes per row.
- Each iteration: O(n^2 dims); 500 iterations on 333 penguins take a few seconds per candidate.
- Memory: two n x n matrices (P and W).
- No out-of-sample transform, so no `predict.cpp`; the `unsupervised_reload` tool refits from the saved state instead.
