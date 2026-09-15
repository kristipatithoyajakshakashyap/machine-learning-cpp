# model evaluation — worked mathematics

02_math_intuition.cpp
Lesson: every metric from one list of predictions, by hand.
Five binary rows, scores and a decision threshold.  All the standard
numbers (accuracy, precision, recall, F1, AUC) come from the SAME
confusion matrix - the threshold only moves the matrix.  This lesson
computes them twice (threshold 0.5 and 0.7) so the trade-off is visible:
precision stays perfect, recall pays for the stricter cut.
sklearn equivalent: sklearn.metrics.{accuracy_score, precision_score,
recall_score, f1_score, roc_auc_score}.

y = [1 0 1 1 0]   scores = [0.9 0.1 0.8 0.7 0.2]

EXPECTED OUTPUT:
  y = [1 0 1 1 0]   scores = [0.9 0.1 0.8 0.7 0.2]
    threshold 0.5 -> pred [1 0 1 1 0]
    accuracy 1  precision 1  recall 1  F1 1
    threshold 0.7 -> pred [1 0 1 0 0]
    accuracy 0.8  precision 1  recall 0.6666667  F1 0.8
    AUC = 1



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
