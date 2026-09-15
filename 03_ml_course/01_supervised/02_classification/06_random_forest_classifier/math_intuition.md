# random forest classifier — worked mathematics

02_math_intuition.cpp
Lesson: why feature subsampling decorrelates trees.
Classification forests draw max_features ~ sqrt(p) features per split.
The DELIBERATE randomness does two things: any single split is weaker, but
the trees become less correlated, and averaging many weakly-correlated
trees beats averaging near-twin trees.  The counts below show how many
different feature subsets each level can pick from.
sklearn equivalent: sklearn.ensemble.RandomForestClassifier
(max_features='sqrt').

EXPECTED OUTPUT:
  wine p=13 : sqrt(p) = 3.605551  -> max_features ~ 4
  feature subsets C(13,4) = 715      <- lots of freedom per split
  iris  p=4  : sqrt(p) = 2          -> C(4,2) = 6
  p=2        : C(2,1) = 2
  every tree is likely a genuinely different model



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
