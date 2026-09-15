# t-SNE: math intuition

All distances are Euclidean in the preprocessed feature space (original) or in the embedding.

## Conditional affinities and perplexity

For row i with bandwidth beta_i = 1 / (2 sigma_i^2):

    p_{j|i} = exp(-beta_i ||x_i - x_j||^2) / sum_{k != i} exp(-beta_i ||x_i - x_k||^2),   p_{i|i} = 0

The Shannon entropy of the row is H(P_i) = -sum_j p_{j|i} log p_{j|i} and its perplexity is exp(H(P_i)). Perplexity behaves like an effective neighbour count: a distribution spread evenly over m rows has perplexity exactly m. beta_i is found by binary search so that H(P_i) = log(perplexity); larger beta sharpens the distribution and lowers the entropy.

## Symmetric joint affinities

    p_ij = (p_{j|i} + p_{i|j}) / (2n)

Symmetrising guarantees sum_j p_ij >= 1/(2n) for every i, so no row is ignored by the objective even if nobody else chose it as a neighbour. The implementation also clamps p_ij to at least 1e-12 so the logarithm is finite.

## Low-dimensional affinities

    q_ij = (1 + ||y_i - y_j||^2)^-1 / sum_{k != l} (1 + ||y_k - y_l||^2)^-1

The Student-t kernel decays like 1 / d^2 instead of exp(-d^2). Two rows that should be moderately far apart can be placed very far apart at almost no cost, which is what pulls clusters apart.

## Objective and gradient

    C = KL(P || Q) = sum_{i != j} p_ij log(p_ij / q_ij)

    dC / dy_i = 4 sum_j (p_ij - q_ij) (1 + ||y_i - y_j||^2)^-1 (y_i - y_j)

Read the gradient as a spring system: the term (p_ij - q_ij) is an attraction when the pair should be closer than it is and a repulsion otherwise, scaled by the same heavy-tailed kernel.

## Worked example: perplexity of one row

Three 1-D points x_0 = 0, x_1 = 1, x_3 = 3, and beta_0 = 1.

| j | d^2 | exp(-d^2) | p_{j|0} |
|---|---|---|---|
| 1 | 1 | 0.36788 | 0.99966 |
| 3 | 9 | 0.00012 | 0.00034 |

Entropy = -(0.99966 log 0.99966 + 0.00034 log 0.00034) = 0.0027 nats, perplexity = exp(0.0027) = 1.0027. Point 0 effectively has a single neighbour. If beta_0 -> 0 both weights tend to 1, the distribution becomes uniform, entropy = log 2 and perplexity = 2, which is the maximum for a row with two other points. Asking for perplexity 1.5 therefore lands beta_0 somewhere between 0 and 1; the binary search finds it in a few dozen halvings. `02_math_intuition.cpp` prints these numbers.

## Update rule

    v_i <- momentum * v_i - eta * gain_i * dC/dy_i,   y_i <- y_i + v_i

with momentum 0.5 for the first 250 iterations and 0.8 afterwards, learning rate eta = 200, and per-coordinate gains that grow by 0.2 when the gradient keeps its sign and shrink by 20% when it flips (never below 0.01). During the first 100 iterations P is multiplied by 12 (early exaggeration). The embedding is re-centred after every step, which removes the translation freedom of the objective.

## Trustworthiness

For row i let U_k(i) be the rows among its k nearest neighbours in the embedding that are *not* among its k nearest neighbours in the original space, and r(i, j) the rank of j in the original distances from i. Then

    T(k) = 1 - 2 / (n k (2n - 3k - 1)) * sum_i sum_{j in U_k(i)} (r(i, j) - k)

T = 1 when every embedded neighbourhood is an original neighbourhood; each intruder is penalised by how far down the original ranking it came from. The normalisation makes the worst case 0. Because T depends only on ranks it is comparable across perplexities, unlike KL.

Work through the three-point fixture by hand before running the numerical lesson. Change one point and explain how the entropy changes.
