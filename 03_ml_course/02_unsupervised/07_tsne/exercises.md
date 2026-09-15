# t-SNE: exercises

1. **Perplexity by hand.** Redo the three-point example in `math_intuition.md` with beta_0 = 0.25 and beta_0 = 0.05. Compute p_{1|0}, p_{3|0}, the entropy and the perplexity each time, then explain how the binary search in `conditional_row` would move beta if the requested perplexity were 1.5.

2. **Perplexity sweep.** Run the Iris lesson with perplexities 2, 5, 30 and 100 (the last one must throw: explain why). For the others, compare `trustworthiness` and the final KL. Which comparison is legitimate and which is not, and why?

3. **Seeds and pictures.** Fit the penguin embedding with seeds 1, 2 and 3 at the selected perplexity. Overlay the three figures (or compare cluster memberships by running `KMeans(3)` on each embedding and computing `unsup::adjusted_rand` between them). What is stable and what changes?

4. **Early exaggeration.** Set `kEarlyExaggeration` to 1 in a scratch copy of the header and re-run the Iris lesson. Compare `kl_per_iteration.svg` and the final trustworthiness. Then try exaggeration 12 for all 500 iterations and describe the layout.

5. **Reload without refit.** Write a small driver that loads `model/model_state.txt`, calls `embedding()` directly and compares it with `evaluation/embedding.csv`. Then delete the seed from a copy of the state file and explain why the refit path of `unsupervised_reload` can no longer verify anything.

Acceptance: show equations, executable changes, saved results, and an interpretation of limitations.
