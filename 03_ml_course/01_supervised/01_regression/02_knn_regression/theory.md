# knn regression — theory

01_theory.cpp
Lesson: k-nearest-neighbours regression in one page.
Theory summary:
  - kNN makes NO assumption about the shape of the function; it stores the
    training rows and predicts a new x by averaging the targets of its k
    closest training points (distance = Euclidean by default).
  - The ONLY hyper-parameter is k.  Small k -> high variance (wiggly fit,
    overfits noise); large k -> high bias (smooth fit, may miss structure).
  - This is why the data lesson sweeps k and watches train R2 collapse from
    1 (k=1 memorises) towards a stable test R2.
The toy below has two clusters; predicting between them shows the average
belonging to the nearest neighbours rather than to either cluster.
sklearn equivalent: sklearn.neighbors.KNeighborsRegressor(k).



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
