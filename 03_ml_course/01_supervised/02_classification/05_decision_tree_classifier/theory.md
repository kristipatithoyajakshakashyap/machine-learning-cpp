# decision tree classifier — theory

01_theory.cpp
Lesson: decision tree classification in one page.
Theory summary:
  - A tree classifies by a chain of "X[f] <= t" tests; leaves hold a class
    distribution and answer with the most common class.
  - Splits are chosen to reduce GINI IMPURITY
      G = 1 - sum_c p_c^2   (0 = pure leaf, max when classes mix).
  - max_depth / min_samples_leaf control how far the recursion goes - the
    classic bias-variance dial of every tree model.
sklearn equivalent: sklearn.tree.DecisionTreeClassifier(criterion='gini').

Toy leaf: 3 green, 1 red (classes {0,0,0,1}).



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
