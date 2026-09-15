# Theory

## Two questions a fitted model should answer

1. Which inputs does the model actually use, and how much would its performance suffer without each one?
2. Is the model limited by too little data (variance) or by too weak a hypothesis class (bias), and is it worth collecting more rows?

Neither question is answered by a single test score. Permutation importance answers the first; learning curves answer the second.

## Permutation importance versus impurity importance

Tree ensembles offer a free importance score: the total impurity decrease attributed to each feature across all splits (sklearn's `feature_importances_`). It is cheap but biased. Features with many distinct values get more chances to produce a good-looking split, so continuous or high-cardinality columns are inflated; it is computed on training data, so it rewards features that help the trees memorise; and it cannot be computed for models without splits.

Permutation importance (Breiman 2001) is model-agnostic: take a fitted model, a scoring dataset and a metric; shuffle one column so its values are decoupled from the labels while the marginal distribution is unchanged; measure how much the metric falls. Repeat a few times to average out the shuffle. The score is measured wherever you choose to measure it: on held-out rows it reflects generalisation, on training rows it reflects what the model has fitted.

Its known weaknesses:

- Correlated features: if two columns carry the same information the model can lean on either, so shuffling one barely hurts and both look unimportant. Wine's phenol-related measurements show this.
- Extrapolation: shuffling creates feature combinations that never occur, and the drop then partly measures the model's behaviour off-distribution.
- Small scoring sets produce noisy, sometimes negative, drops; report the spread over repeats and do not over-interpret rank differences within it.

## Learning curves

A learning curve trains the same model on nested subsets of the training data (10%, 20%, ..., 100%) and records two numbers per size: the score on the rows it trained on and the cross-validated score on held-out folds. Plotting both against the number of training rows gives a picture of the bias-variance state of the model:

- High variance: the training score is high, the CV score is much lower, and the gap is still closing at the largest size. More data or more regularisation helps.
- High bias: both curves are low and have converged. More data does not help; the model needs more capacity or better features.
- Well matched: both curves are high and close; the CV curve has flattened.

Random forests usually show a training curve near the maximum score because each tree interpolates its bootstrap sample. That is not overfitting by itself; the CV curve carries the information.

## Using them together

Importance tells you where the signal is; the learning curve tells you whether the model has extracted it. A flat CV curve with a few dominant features suggests the problem is easy and the remaining errors are label noise. A rising CV curve with importance spread over many weak features suggests more rows would keep paying off.

## Related sklearn functions

`sklearn.inspection.permutation_importance` and `sklearn.model_selection.learning_curve` compute the same quantities. The helpers in `helper/eval/diagnostics.hpp` follow their conventions: importance is `baseline - permuted` (positive means important), and the learning curve reports means over folds for each training size.
