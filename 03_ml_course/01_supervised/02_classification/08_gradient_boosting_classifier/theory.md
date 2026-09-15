# gradient boosting classifier — theory

01_theory.cpp
Lesson: gradient boosted classification in one page.
Theory summary:
  - GB regression fits trees to residuals; classification fits trees to
    PSEUDO-RESIDUALS of the logistic loss.  For binary:
      r_i = y_i - p_i,   p_i = sigmoid(F(x_i)).
  - Models are summed in LOG-ODDS space, then piped through the sigmoid
    (or softmax for K classes) to get probabilities:
      p = sigmoid( F0(x) + lr*h1(x) + lr*h2(x) + ... )
  - Leaf values are set so the round actually minimises the loss, then
    everything repeats.  This is why the same boosting loop covers both
    regression and classification.
sklearn equivalent: sklearn.ensemble.GradientBoostingClassifier.

Binary worksheet: y=[1,0], current p = [0.6224593, 0.3775407].



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
