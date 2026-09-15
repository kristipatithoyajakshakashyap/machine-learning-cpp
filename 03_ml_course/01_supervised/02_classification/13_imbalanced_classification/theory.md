# Theory

## What imbalance is

A binary problem is imbalanced when one class (the positive class by convention) is much rarer than the other. Fraud, equipment failure, rare disease and defect detection are typical: prevalence of 1-10% is common, below 0.1% is not unusual. In this module prevalence is 30/387 = 7.8% by construction.

Imbalance is not a bug in the data. It is the true base rate, and the learner must respect it: a model that ignores prevalence produces badly calibrated probabilities, while a model that only respects prevalence never predicts the rare class.

## The accuracy paradox

Accuracy counts correct decisions over all rows. With 7.8% positives the constant "always negative" classifier scores 92.2% accuracy and is useless. Any metric that averages over rows is dominated by the majority class. The metrics that survive imbalance condition on the class:

- recall (true positive rate): of the real positives, how many were flagged;
- precision: of the flagged rows, how many were real positives;
- F1: harmonic mean of the two; macro F1 averages F1 over both classes so the rare class counts as much as the common one.

The pipeline selects parameters with macro F1 and reports accuracy only for context.

## Cost-sensitive learning

Missing a positive usually costs more than a false alarm. Cost-sensitive learning bakes that asymmetry into training by weighting the loss of each row by a class-dependent cost. With weight `w` on positives the optimiser minimises

    L = (1/W) * sum_i w_i * logloss(y_i, p_i),   w_i = w if y_i = 1 else 1,

which is equivalent to duplicating each positive row `w` times. The decision boundary moves toward the negative cloud, recall rises, precision falls. Random oversampling and undersampling are sampling-based relatives of the same idea; SMOTE synthesises interpolated positives.

## Threshold moving

A probabilistic classifier does not need to decide at 0.5. Lowering the threshold to 0.2 flags more rows as positive: recall goes up, precision goes down, and no retraining is required. Reweighting and threshold moving are two knobs on the same trade-off. Reweighting changes what the model learns (which rows shape the boundary); threshold moving changes only where the boundary is cut. Choose the threshold on training cross-validation folds, never on the hold-out.

## PR versus ROC

ROC plots recall against the false positive rate. When negatives are abundant, a large number of false alarms is still a small false positive rate, so ROC curves and AUC look flattering. Precision-recall plots recall against precision, whose denominator is the number of flagged rows, so every false alarm hurts. For rare positives inspect the PR curve; use ROC to compare rankers when the base rate may change between deployments.

## Reading the artifacts

`evaluation/metrics.json` reports accuracy, macro F1, ROC AUC and log loss on the hold-out. `evaluation/per_class.csv` gives precision and recall for class 1. `evaluation/pr_curve.csv` tabulates the threshold sweep. Compare `baseline_macro_f1` (majority class) to `macro_f1` to see how much the model adds beyond the base rate.
