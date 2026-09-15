# random forest classifier — theory

01_theory.cpp
Lesson: random forest classification in one page.
Theory summary:
  - Same recipe as the regressing forest: T bootstrap trees, each split
    over a random subset of features, predictions combined.
  - For classification a tree emits CLASS PROBABILITIES (the leaf's class
    shares); the forest averages a probability vector:
        P(y=c|x) = (1/T) sum_t p_t(c|x).
  - Averaging probabilities is smoother than a hard vote and agrees with
    sklearn's predict_proba.  Hard votes = argmax of that average.
sklearn equivalent: sklearn.ensemble.RandomForestClassifier.

Three trees answer a two-class query with probability vectors.



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
