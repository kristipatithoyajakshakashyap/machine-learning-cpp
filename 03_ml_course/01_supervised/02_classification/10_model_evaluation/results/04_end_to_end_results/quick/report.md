# Model comparison: 0 ridge, 1 kNN, 2 forest, 3 MLP

Quick verification uses a documented deterministic subset. Run without --quick for the full dataset.

Training rows: 320; untouched test rows: 80. 5 folds selected parameter 0 using negative RMSE.

Holdout RMSE: 81601. Bootstrap 95% percentile interval [63364.3, 100152]. These intervals describe sampling variation conditional on this fitted model, not repeated model-selection uncertainty.

See evaluation/metrics.json for baseline comparison, evaluation/predictions.csv for every test prediction and eda/findings.md for data quality. Scaling and median imputation were fitted separately inside every fold. Test outcomes did not select parameters. Model and preprocessing reload reproduced predictions.

Limitations: one holdout split; evaluate transfer to new populations before deployment. Correlations and feature relations are descriptive.
