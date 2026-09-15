# Theory

A Gaussian process (GP) places a prior directly on functions: any finite set of function values f(x_1..x_n) is jointly Gaussian with mean 0 and covariance K_ij = k(x_i, x_j). The RBF kernel k(x, z) = s^2 exp(-|x - z|^2 / (2 l^2)) encodes that nearby inputs have similar outputs; the length scale l sets how quickly correlation decays and the signal variance s^2 sets the amplitude.

Observations are y = f(x) + noise with noise variance sigma^2, so y ~ N(0, K + sigma^2 I). Conditioning this joint Gaussian on the training targets gives a closed-form posterior for any test input x*: mean k(x*, X)(K + sigma^2 I)^-1 y and variance k(x*, x*) - k(x*, X)(K + sigma^2 I)^-1 k(X, x*). Unlike ridge or kernel ridge regression, a GP returns calibrated uncertainty that grows away from the data and collapses to the noise level on top of it.

The log marginal likelihood log p(y | X) trades data fit against model complexity and is the standard objective for choosing kernel hyperparameters. In this module the length scale is chosen by cross-validation via the shared pipeline; the marginal likelihood is computed and plotted for comparison.
