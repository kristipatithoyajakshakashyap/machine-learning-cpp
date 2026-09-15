# One-Class SVM: theory

## The problem it solves

Anomaly detection without labels: learn where "ordinary" rows live and flag anything that falls outside. Isolation Forest answers this with random splits and LOF with local density ratios; the one-class SVM (Schoelkopf, Platt, Shawe-Taylor, Smola and Williamson, 2001) answers it with a single smooth boundary learned by a convex optimisation.

## Separating the data from the origin

An RBF kernel `K(x, z) = exp(-gamma ||x - z||^2)` implicitly maps every row to a unit vector `phi(x)` in a high-dimensional feature space (`K(x, x) = 1`). The one-class SVM finds the hyperplane `w . phi(x) = rho` that separates the mapped training rows from the origin with the largest margin `rho / ||w||`. Rows on the far side of the hyperplane are ordinary; rows on the origin side are anomalies. Because `phi` is nonlinear, the pre-image of the half-space is a union of smooth blobs around the training data.

## The role of nu

Slack variables let some training rows fall inside the margin at a cost of `1 / (nu n)` each. This produces the nu property:

- at most a fraction `nu` of training rows are outliers (strictly outside the boundary),
- at least a fraction `nu` of training rows are support vectors.

So nu is a budget: `nu = 0.05` says "assume roughly 5 percent of my training data is unusual and let the boundary ignore it". It is set by the analyst, not estimated from data.

## The role of gamma

gamma is the inverse squared width of the kernel. Small gamma: one smooth region enclosing all the data, close to an ellipsoid. Large gamma: every training row gets its own tight bump, and anything not almost identical to a training row is flagged. The course default is the "scale" rule `gamma = 1 / p` on standardised features, which is a sensible middle ground for a handful of features.

## Scores

The decision function `f(x) = sum_i alpha_i K(x_i, x) - rho` is positive inside the region and zero on the boundary. To match the other anomaly modules (higher = more anomalous), this course reports `score = rho - sum_i alpha_i K(x_i, x) = -f(x)`. Only the support vectors (rows with `alpha_i > 0`) contribute, so the saved model stores just those rows.

## Strengths and limits

- Smooth, deterministic boundary; no randomness at all.
- Convex objective, so the solver reaches a unique optimum regardless of starting point.
- Needs an `n x n` kernel matrix during training and a kernel evaluation per support vector at prediction time, so it does not scale like Isolation Forest.
- Sensitive to gamma and to feature scaling; always standardise first.
- Like every unsupervised detector, its flags are "unusual relative to training", never "wrong" - inspection is still required.
