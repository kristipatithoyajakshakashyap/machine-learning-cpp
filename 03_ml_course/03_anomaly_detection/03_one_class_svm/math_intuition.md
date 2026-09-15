# One-Class SVM: math intuition

## Primal problem

    min_{w, rho, xi}  1/2 ||w||^2 + 1/(nu n) sum_i xi_i - rho
    s.t.              w . phi(x_i) >= rho - xi_i,  xi_i >= 0

Maximising `rho` pushes the hyperplane away from the origin; the `||w||^2` term keeps the margin wide; each `xi_i` pays for a training row that lands inside the margin.

## Dual problem

Introducing multipliers `alpha_i` and eliminating `w = sum_i alpha_i phi(x_i)` gives

    min_alpha  1/2 sum_ij alpha_i alpha_j K(x_i, x_j)
    s.t.       0 <= alpha_i <= 1/(nu n),  sum_i alpha_i = 1

which involves the data only through the kernel. Let `C = 1/(nu n)`. Rows fall into three groups:

| alpha_i | position | name |
|---|---|---|
| 0 | strictly inside the region | non-support vector |
| (0, C) | exactly on the boundary | free support vector |
| C | outside the region (margin error) | bounded support vector |

Because `sum alpha = 1` and each `alpha <= C = 1/(nu n)`, at least `nu n` rows must have positive alpha, and because errors cost `C` each at most `nu n` rows can be bounded. That is the nu property.

## Recovering rho

For any free support vector the constraint is tight with zero slack:

    rho = sum_j alpha_j K(x_j, x_i)     for 0 < alpha_i < C

The implementation averages this over all free support vectors for numerical stability, and falls back to the mean over all support vectors when none is free (as in the two-point example below).

## Decision and score

    f(x)     = sum_i alpha_i K(x_i, x) - rho
    score(x) = rho - sum_i alpha_i K(x_i, x) = -f(x)

`f > 0` inside, `f = 0` on the boundary, `f < 0` outside. The score is reported so that larger means more anomalous.

## Solving the dual deterministically

The dual is a quadratic programme over a simplex intersected with a box. The header uses a pairwise (SMO-style) descent. With `g = K alpha` the gradient of the objective:

1. pick `up = argmax_i g_i` among rows with `alpha_i > 0` (can shrink) and `down = argmin_i g_i` among rows with `alpha_i < C` (can grow);
2. stop when `g_up - g_down < tolerance` (this is the KKT gap);
3. move mass `t` from `up` to `down` with `t = min((g_up - g_down) / (K_uu + K_dd - 2 K_ud), alpha_up, C - alpha_down)`, the exact line minimiser clipped to the box;
4. update `g_i += t (K_id - K_iu)` in O(n).

Every choice is deterministic (ties broken by index), so the same data always gives the same alphas, rho and scores; the fresh-process reload test relies on that.

## Hand example (two points)

`x_0 = 0`, `x_1 = 2`, `gamma = 1`, `nu = 1`, so `C = 1/2` and `sum alpha = 1` forces `alpha_0 = alpha_1 = 1/2` (both bounded).

    k = K(x_0, x_1) = exp(-4) = 0.018316
    g_0 = g_1 = (1 + k) / 2 = 0.509158
    rho = 0.509158          (no free vector, so mean over support vectors)
    f(x_0) = f(x_1) = 0     (training points sit on the boundary)
    f(1) = (2 exp(-1)) / 2 - rho = 0.367879 - 0.509158 = -0.141279
    score(1) = +0.141279    (the midpoint is outside: gamma = 1 makes each point its own blob)

`02_math_intuition.cpp` fits `OneClassSVM(1, 1)` on these two points and checks rho and the three scores against these values to 1e-12.

## Comparison with the other detectors

| | Isolation Forest | LOF | One-Class SVM |
|---|---|---|---|
| principle | random splits, short paths | density ratio to neighbours | margin from origin in kernel space |
| randomness | seeded ensemble | none | none |
| stored state | trees | all training rows | support vectors only |
| training cost | O(t n log n) | O(n^2 p) | O(n^2 p) kernel + solver sweeps |
| main knobs | trees, subsample | k | nu, gamma |
