# Theory: grid search versus random search

A hyper-parameter is a knob the training algorithm cannot set for itself: tree depth, minimum leaf size, learning rate, regularisation strength. Fitting sees only the training loss, and the training loss almost always prefers the most flexible setting, so the choice must be made by an *outer* loop that scores each candidate on rows the fit never touched. Cross-validation is that loop.

## Grid search

Enumerate every combination of a small list of values per parameter and score each with k-fold CV. With d parameters and v values each the cost is v^d fits times k folds. It is exhaustive, reproducible and trivially parallel, but it wastes effort: if only one of the d parameters matters, the grid spends v^(d-1) evaluations re-measuring the same v useful points.

## Random search

Sample n combinations at random and score them. Bergstra and Bengio (2012) observed that most response surfaces have *low effective dimension*: a few parameters dominate. Random draws spread over every axis independently, so n draws explore n distinct values on the important axis, whereas the grid explores only v. For the same budget random search usually finds an equal or better point.

This module samples *without replacement* from the finite grid, so `random_search` with n at least the grid size is exactly the grid search, and no evaluation is repeated.

## What the score means

The mean CV score of the winner is not an unbiased estimate of the winner's generalisation performance. The selection step picked the maximum of several noisy numbers, and the maximum of noisy numbers is biased upward. `nested_cv` exists to report the honest number; see math_intuition.md.

## Budget

Every candidate costs k fits. A grid of 9 with k = 5 is 45 forest fits; a 5-draw random search is 25; nested CV with 5 outer folds, 3 inner folds and 5 draws is 5 x (5 x 3 + 1) = 80 fits plus the naive search. Choose the tool by budget: grid when the space is tiny, random when it is large, nested when you must report a number you trust.
