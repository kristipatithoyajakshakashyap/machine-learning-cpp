# naive bayes — theory

01_theory.cpp
Lesson: Gaussian Naive Bayes in one page.
Theory summary:
  - Bayes: P(y|x) = P(y) * f(x|y) / sum_y' P(y')*f(x|y'), where f is the
    Gaussian density with the class mean/variance.
  - "Naive": each feature is assumed independent, so the likelihood is the
    product over features - a strong assumption that still works well with
    few rows and little data.
  - Classification rule: pick the class with the largest posterior.  For
    numeric stability libraries compare LOG posteriors instead.
sklearn equivalent: sklearn.naive_bayes.GaussianNB.

Toy data: class 0 rows x={1,2,3}, class 1 rows x={7,8,9}, query x=4.5.
priors = 0.5 each, variances = 0.6666667 each.



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
