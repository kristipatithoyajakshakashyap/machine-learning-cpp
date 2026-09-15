# Local Outlier Factor: math intuition

All distances below are Euclidean in the preprocessed feature space. N_k(p) is the set of the k nearest training points to p (p itself excluded when p is a training point).

## k-distance

    k-distance(p) = d(p, o_k)   where o_k is the k-th nearest neighbour of p

It is the radius of the smallest ball around p containing k other points; small radius means dense surroundings.

## Reachability distance

    reach_k(p, o) = max( k-distance(o), d(p, o) )

The distance from p to o, but never smaller than o's own k-distance. If p sits inside o's k-neighbourhood ball, the distance is replaced by the ball radius. This smoothing is what makes densities stable: points deep inside a cluster all "see" the same radius instead of their individual tiny distances.

## Local reachability density

    lrd_k(p) = k / sum_{o in N_k(p)} reach_k(p, o)

The inverse of the average reachability distance from p to its neighbours. Large lrd means the neighbours are close and dense.

## Local outlier factor

    LOF_k(p) = ( (1/k) sum_{o in N_k(p)} lrd_k(o) ) / lrd_k(p)

The mean density of the neighbours divided by the density of p. LOF is approximately 1 inside a homogeneous region, greater than 1 for points whose neighbours are denser than they are, and unbounded above.

## Worked example (k = 2)

One-dimensional points A = 0, B = 1, C = 2, D = 10.

**Neighbours and k-distances**

| point | nearest two | k-distance |
|---|---|---|
| A | B (1), C (2) | 2 |
| B | A (1), C (1) | 1 |
| C | B (1), A (2) | 2 |
| D | C (8), B (9) | 9 |

**Reachability distances and lrd**

- A: reach(A,B) = max(1, 1) = 1, reach(A,C) = max(2, 2) = 2, lrd(A) = 2 / 3
- B: reach(B,A) = max(2, 1) = 2, reach(B,C) = max(2, 1) = 2, lrd(B) = 2 / 4 = 1/2
- C: reach(C,B) = max(1, 1) = 1, reach(C,A) = max(2, 2) = 2, lrd(C) = 2 / 3
- D: reach(D,C) = max(2, 8) = 8, reach(D,B) = max(1, 9) = 9, lrd(D) = 2 / 17

Note how B, the middle point, has a *lower* density than A and C: its own k-distance is 1 but both neighbours have k-distance 2, and the reachability rule lifts each distance to 2.

**Factors**

- LOF(A) = ((1/2 + 2/3) / 2) / (2/3) = (7/12) / (2/3) = 7/8 = 0.875
- LOF(B) = ((2/3 + 2/3) / 2) / (1/2) = (2/3) / (1/2) = 4/3 = 1.333
- LOF(C) = 7/8 = 0.875
- LOF(D) = ((2/3 + 1/2) / 2) / (2/17) = (7/12) * (17/2) = 119/24 = 4.958

D is clearly the outlier. `02_math_intuition.cpp` fits `LocalOutlierFactor(2)` on these four points and asserts the computed factors match to 1e-12.

## Novelty scoring of a new point q

The same formulas apply with N_k(q) drawn from the stored training rows, using the stored k-distances and lrd values of those rows:

    reach_k(q, o) = max(k-distance(o), d(q, o))
    lrd_k(q)      = k / sum reach_k(q, o)
    LOF_k(q)      = mean lrd_k(o) / lrd_k(q)

Only quantities of the training rows appear on the right, so a saved model needs just the training matrix, the k-distance vector and the lrd vector to reproduce scores exactly.

## Degenerate distances

If every reachability distance is zero (a query identical to k or more duplicated training points), the sum in lrd is zero. The implementation divides by max(sum, 1e-12); identical points then share the same huge density and their factor is 1, which is the correct "nothing unusual here" answer.
