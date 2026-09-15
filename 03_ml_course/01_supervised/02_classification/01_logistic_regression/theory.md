# logistic regression — theory

01_theory.cpp
Lesson: logistic regression in one page.
Theory summary:
  - Linear regression predicts on (-inf, +inf); classification needs a
    PROBABILITY in [0,1].  The logistic (sigmoid) link does that:
      p(class=1 | x) = 1 / (1 + exp(-(w.x + b)))
    A linear score w.x + b is therefore a LOG-ODDS: log(p/(1-p)).
  - The training loss is cross-entropy (the negative log-likelihood of the
    labels under the sigmoid).  Its gradient is beautiful:
      dL/dw = (1/n) sum_i (p_i - y_i) x_i
    i.e. we slide w uphill on the misclassification residuals, exactly like
    linear regression slides on raw residuals.
  - Here we fit a 1-D toy: features increase smoothly (0,0.5,1,1.5,2) with
    labels (0,0,1,1,1); a glance at the sigmoid curve shows the probability
    crossing 0.5 between x=1 and x=1.5.
sklearn equivalent: sklearn.linear_model.LogisticRegression.



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
