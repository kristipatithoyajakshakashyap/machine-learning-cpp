# model evaluation — theory

12_binary_classification.cpp
Lesson: binary-classification deep dive on breast cancer.
One score (logistic probability) drives every binary metric, and the choice
of decision threshold is a business decision: lowering it catches more
positives but also produces false alarms.  This lesson sweeps the threshold
over p_pred, builds the full ROC curve, and prints the confusion
cells + F1 at a few interesting points, plus the AUC.  All metrics come
from ml/metrics.hpp and match sklearn digit-for-digit on the same split.
Equivalent: sklearn.metrics confusion_matrix / roc_curve / roc_auc_score.

EXPECTED OUTPUT (80/20 seed 7, softmax logistic, C=1):
th=0.3  tn 74 fp 0 fn 3 tp 37  f1 0.961039
th=0.5  tn 74 fp 0 fn 4 tp 36  f1 0.9473684
th=0.7  tn 74 fp 0 fn 4 tp 36  f1 0.9473684
auc 0.9858108



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
