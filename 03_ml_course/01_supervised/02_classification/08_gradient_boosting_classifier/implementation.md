# gradient boosting classifier — implementation walkthrough

Model interfaces: GradientBoostedTrees.hpp. Fit stores learned state; predict uses that state without updating it. The numbered implementation lesson is a small API example. The final lesson calls the shared C++ supervised runner.

1. Retain original row identities, reserve the holdout, and produce EDA on training rows.
2. Build validation folds inside the training partition.
3. Fit median imputation and population standardization on each fold's training rows, then transform its validation rows.
4. Fit each candidate from a fresh seeded state. Regressors also standardize training targets and invert predictions to original units.
5. Select negative RMSE for regression or macro F1 for classification, refit on the complete training partition, and evaluate the holdout once.
6. Save plots, fold scores, predictions, metrics, fitted model and preprocessing. Reload and compare predictions.

Serialization is versioned text with 17 significant digits. Inference reads named feature columns and applies saved preprocessing. See helper/pipeline/supervised.hpp for the complete control flow.
