# Math Intuition

Cholesky factorisation K + sigma^2 I = L L^T turns every posterior quantity into triangular solves. alpha = L^-T L^-1 y is computed once at fit time; the posterior mean at a new point is then the dot product k(x*, X) . alpha, a weighted sum of kernel bumps centred on the training inputs. The posterior variance subtracts |L^-1 k(X, x*)|^2 from the prior variance s^2: the more the test point resembles training points, the more variance is explained away.

The log marginal likelihood decomposes into -y^T alpha / 2 (data fit), -sum log L_ii (complexity, half the log-determinant of the covariance) and a constant. A very short length scale fits every wiggle but pays a large determinant penalty; a very long one is simple but fits poorly. The peak of the curve balances the two, which is what `03_implementation.cpp` plots.

Numerically the Cholesky requires K + sigma^2 I to be positive definite. Duplicate or near-duplicate inputs with zero noise make it singular, so a small jitter (the noise variance) is kept even when the data are noise free.
