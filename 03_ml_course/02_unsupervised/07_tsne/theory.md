# t-SNE: theory

## What problem it solves

PCA finds the linear projection that keeps the most variance. When the interesting structure is a set of curved or well separated groups, the best linear projection can still overlap them. t-distributed Stochastic Neighbour Embedding (van der Maaten and Hinton, 2008) instead asks for a 2-D or 3-D layout in which every row keeps the same *neighbours* it had in the original space. It is a visualisation method: the output is a picture of local structure, not a set of features for downstream models.

## The idea in three steps

1. **Neighbourhood distributions in the original space.** For each row i, place a Gaussian on x_i and read off how much probability mass every other row receives. The width of the Gaussian is chosen per row so that the distribution has a fixed *perplexity*, which is roughly the effective number of neighbours. Dense regions get narrow Gaussians, sparse regions wide ones, so every row has the same number of "friends".
2. **Neighbourhood distributions in the embedding.** For the low-dimensional points y_i use a Student-t kernel with one degree of freedom (a Cauchy kernel). Its heavy tail lets moderately dissimilar rows be placed far apart without paying much cost, which is what opens gaps between clusters and cures the "crowding problem" of the earlier SNE method.
3. **Match the two.** Move the y_i by gradient descent so the Kullback-Leibler divergence KL(P || Q) between the two neighbourhood distributions is small. KL is asymmetric: a large p_ij with a small q_ij (true neighbours drawn apart) costs a lot, while a small p_ij with a large q_ij (strangers drawn together) costs little. t-SNE therefore protects local neighbourhoods and is relaxed about global arrangement.

## Perplexity

Perplexity is the single important hyper-parameter. Typical values lie between 5 and 50 and must be smaller than the number of rows.

- Small perplexity: tiny neighbourhoods, the embedding fragments into many small clumps and can invent structure.
- Large perplexity: the neighbourhoods span whole clusters and the embedding tends towards a global, PCA-like layout.

Because P depends on perplexity, the KL divergence reached at different perplexities measures different objectives and cannot be compared directly. The end-to-end project therefore compares candidates with *trustworthiness*, a rank-based measure of how many of each row's k nearest embedded neighbours were also close in the original space.

## Optimisation devices

- **Early exaggeration** multiplies P by 12 for the first 100 iterations so clusters form as tight, well separated groups before the fine layout is settled.
- **Momentum** (0.5 then 0.8 after iteration 250) and per-coordinate adaptive gains speed up the plain gradient step.
- **Random initialisation** from a tiny Gaussian: the seed is part of the model state because a different seed gives a different (equally valid) picture.

## What a t-SNE plot does not tell you

- Cluster **sizes** are meaningless: dense and sparse groups are expanded to similar areas.
- **Distances between clusters** are meaningless: two groups far apart in the plot may be close in the data.
- **Axes** have no interpretation; rotating or flipping the plot changes nothing.
- The picture is not a **test**: with a small perplexity even random noise forms clusters. Run several perplexities and seeds before claiming structure.
- There is **no out-of-sample map**; new rows cannot be placed without refitting everything. Parametric t-SNE and UMAP exist for that use case.

## Limitations of scale

The implementation forms the full n x n affinity matrix and runs O(n^2) gradient steps, which is fine for a few hundred to a few thousand rows. Barnes-Hut and FFT accelerated variants handle millions of rows but are far longer than a lesson.

Before modeling, inspect feature units, missingness, duplicates, outliers, distributions and correlations. The end-to-end lesson records this analysis and explains the metric limitations.
