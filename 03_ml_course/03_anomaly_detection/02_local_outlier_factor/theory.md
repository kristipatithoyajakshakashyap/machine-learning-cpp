# Local Outlier Factor: theory

## Density-based outliers

A global rule such as "flag rows far from the mean" or "flag rows whose nearest neighbour is far away" assumes one notion of "far" for the whole dataset. Real data rarely cooperates: one cluster may be tight, another diffuse, and a point that is perfectly ordinary inside the diffuse cluster would be a striking outlier inside the tight one.

Local Outlier Factor (Breunig, Kriegel, Ng and Sander, 2000) fixes the frame of reference. Instead of asking how far a point is from everything, it asks how dense the point's own neighbourhood is compared with the neighbourhoods of the points next to it. Each point gets a *local reachability density* (roughly the inverse of the average distance to its k nearest neighbours) and the factor is the ratio

    LOF(p) = mean density of p's neighbours / density of p.

A factor near 1 means p is as crowded as its neighbours: it belongs. A factor of 3 means the neighbours live in regions three times denser than p's own: p is a local outlier. Factors below 1 occur for points in the middle of a dense core.

## LOF versus Isolation Forest versus distance-based rules

- **Distance-based (k-NN distance, DBSCAN noise)** uses one absolute scale. It misses outliers that sit near a diffuse cluster and over-flags the fringe of that same cluster.
- **Isolation Forest** scores global isolation: how few random axis-aligned cuts separate a row from the rest. It is fast, scales to large data and handles many features, but a point lying between two clusters can be hard to isolate and therefore look ordinary.
- **LOF** is relative and local. It catches the between-cluster point and the point on the edge of a tight cluster, because their neighbours are all denser than they are. The price is O(n^2) neighbour search, storing the training set, and sensitivity to k.

The two anomaly modules in this course therefore complement each other: run both, and inspect rows that only one of them flags.

## Choosing k

k controls the size of the neighbourhood used to estimate density.

- Too small (k = 1 or 2): densities are estimated from one or two distances and fluctuate wildly; duplicated or near-duplicated points distort factors.
- Too large: the neighbourhood spans several clusters and the estimate becomes global, so local outliers disappear.
- The original paper suggests k between 10 and 50 and taking the maximum factor over a range of k when in doubt. This module uses k = 20 for the penguin project and k = 10 for the synthetic lesson.

The implementation clamps k to n - 1 when the training set is small, and refuses to fit fewer than two rows.

## Novelty versus outlier mode

- **Outlier mode** scores the training rows themselves, excluding each row from its own neighbourhood. `fitted_scores()` returns these values after `fit`.
- **Novelty mode** keeps the training matrix and scores *new* rows against it only. `score_samples` does this: a query's neighbours are always training rows, so a batch of new rows cannot shield one another. The penguin workflow calibrates a threshold on validation rows and scores held-out rows in novelty mode, which is also what `ulof_predict` reproduces in a fresh process.

## Limitations

- **High dimensions**: Euclidean distances concentrate as the number of features grows, the ratio between nearest and farthest neighbour shrinks, and density contrasts vanish. Standardise features and consider PCA first.
- **Cost**: fitting is O(n^2 d) and each query is O(n d); the whole training matrix is part of the model.
- **Scale**: without scaling, one feature with a large range dominates the distance. The workflow fits the preprocessor on the training partition only.
- **No labels**: a factor describes unusual measurement combinations, not fraud or error. The threshold is a review budget, and flagged rows must be inspected against their source records.
