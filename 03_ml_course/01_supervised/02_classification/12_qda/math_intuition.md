# Mathematical intuition

log score_c = log(prior_c)-0.5*log|Sigma_c|-0.5*(x-mu_c)^T Sigma_c^-1(x-mu_c). The class-dependent quadratic terms do not cancel. In one dimension, compare N(0,1) and N(0,4): the narrow class wins near zero and the broad class can win in both tails.
