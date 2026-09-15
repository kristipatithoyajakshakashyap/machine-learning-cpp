# Implementation

FISTA performs a gradient step on the quadratic dual, then soft-thresholds and clips coefficients. Bisection finds a multiplier enforcing the zero-sum constraint. Chosen step size uses a kernel row-sum upper bound. Interior support-vector KKT conditions recover b; the midpoint between the middle epsilon-loss knots minimizes intercept loss when none are interior. This educational solver uses dense O(n^2) memory and finite iterative tolerance; it is not LIBSVM.

