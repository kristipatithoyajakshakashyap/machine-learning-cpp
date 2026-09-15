# linear regression — theory

01_theory.cpp
Lesson: everything a linear model is in one page.
Theory summary:
  - A LINEAR MODEL predicts a weighted sum: yhat = w0 + w1 x1 + ... + wp xp.
    "Linear" means linear in the WEIGHTS, not in the features -- polynomial
    features still fit a linear model.
  - OLS finds the w that minimises   J(w) = (1/2n) sum_i (yhat_i - y_i)^2 .
    Setting the gradient to zero gives the NORMAL EQUATIONS
      (M'M) w = M'y    where M = [1, X] is the augmented design,
    solved via partial-pivot Gauss elimination -> w = (M'M)^{-1} M'y.
  - RIDGE adds lambda * ||w[1:]||^2 (the intercept is never penalised), so
    (M'M + lambda*I_slopes) w = M'y.  Bigger lambda shrinks slopes toward 0.
  - LASSO adds lambda * ||w[1:]||_1 instead; the L1 ball makes exact zeros,
    so lasso SELECTS features while ridge only shrinks them.
The tiny example below reproduces the normal equations by hand so you can
verify every number before trusting a real dataset.

Data: 3 toy points, x = 0,1,2  y = 1,3,5.  sklearn equivalents:
LinearRegression, Ridge(alpha), Lasso(alpha).



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
