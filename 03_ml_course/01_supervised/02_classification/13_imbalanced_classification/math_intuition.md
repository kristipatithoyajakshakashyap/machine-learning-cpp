# Math Intuition

## Weighted log-loss

Logistic regression models `p_i = sigma(theta . x_i + b)` with `sigma(z) = 1 / (1 + exp(-z))`. The ordinary loss is the mean binary cross-entropy. The cost-sensitive version multiplies each row's term by a class weight:

    L(theta, b) = -(1/W) * sum_i w_i * [ y_i * log p_i + (1 - y_i) * log(1 - p_i) ] + (lambda/2) * ||theta||^2

    w_i = w  if y_i = 1,   w_i = 1  if y_i = 0,   W = sum_i w_i.

Dividing by `W` instead of `n` keeps the learning rate meaningful whatever `w` is.

## Gradient

For the logistic link the derivative of the cross-entropy with respect to the logit is `p_i - y_i`. With the weights:

    dL/dtheta = (1/W) * sum_i w_i * (p_i - y_i) * x_i + lambda * theta
    dL/db     = (1/W) * sum_i w_i * (p_i - y_i)

A positive row that is currently predicted badly (`p_i` small) contributes `w * (p_i - 1) * x_i`, `w` times the unweighted pull. Negatives contribute `(p_i - 0) * x_i` as before. Full-batch gradient descent then updates `theta <- theta - lr * dL/dtheta`.

## Effect on the decision boundary

The boundary is `theta . x + b = 0`. At the optimum of the unweighted loss the boundary sits where the two class densities weighted by their priors cross: rare positives lose that contest and the boundary sits deep inside the positive region, producing few positive predictions. Multiplying the positive contribution by `w` is the same as multiplying the positive prior by `w`. The crossing point moves toward the negative cloud: more positives are recovered and more negatives are mislabelled. For well-separated data the boundary hardly moves because almost no rows are misclassified; for overlapping data the shift is large.

The same argument shows why reweighting damages calibration: the fitted probabilities now describe a population with `w` times as many positives. If calibrated probabilities matter, reweight and then recalibrate, or move the threshold instead.

## Tiny hand example

Two rows, one feature: a positive at `x = 1` with current `p = 0.2`, a negative at `x = -1` with current `p = 0.3`.

| weight `w` | positive term `w(p-y)x` | negative term `(p-y)x` | mean gradient |
|---|---|---|---|
| 1 | 1 * (0.2 - 1) * 1 = -0.80 | (0.3 - 0) * (-1) = -0.30 | -1.10 / 2 = -0.55 |
| 8 | 8 * (0.2 - 1) * 1 = -6.40 | -0.30 | -6.70 / 9 = -0.74 |

Both gradients are negative, so `theta` increases and the positive row's probability rises. With `w = 8` the step is larger relative to the negative's resistance; after convergence the negative at `x = -1` ends up with a higher probability than it would with `w = 1`. That is the recall-for-precision trade in one line.

## Numerical safety

The logit is clipped to `[-30, 30]` before the sigmoid. `sigma(30)` is already `1 - 9e-14`, so clipping changes nothing in normal operation but prevents `exp` overflow and `log(0)` when features are extreme. Inputs reach the model standardised by the pipeline's preprocessor, which keeps the logits small for typical data.
