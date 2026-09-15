# knn classification — theory

01_theory.cpp
Lesson: k-nearest-neighbour classification in one page.
Theory summary:
  - Same distance engine as its regression sibling: store all training
    rows, answer a query by Euclidean distance, keep the k closest.
  - Predict = MAJORITY VOTE of the k neighbours' classes.  With k=1 a
    single neighbour decides (decision boundary is a Voronoi tessellation).
  - k controls smoothness: tiny k memorises the training set (near-perfect
    train accuracy, jagged test boundary); bigger k generalises but can blur
    rare classes.
  - Deterministic tie-break: ties resolved by class order (smallest class
    index wins), matching sklearn's argmax over bincount.
sklearn equivalent: sklearn.neighbors.KNeighborsClassifier.

Worked example: q=(0,0); (1,0) class 0, (1,1) class 0, (0,2) class 1.



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
