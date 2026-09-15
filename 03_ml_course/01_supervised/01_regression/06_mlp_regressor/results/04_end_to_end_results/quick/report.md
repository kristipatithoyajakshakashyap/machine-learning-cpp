# mlp regressor

Quick verification uses a documented deterministic subset. Run without --quick for the full dataset.

Training rows: 320; untouched test rows: 80. 5 folds selected parameter 0.01 using negative RMSE.

Holdout RMSE: 75726.3. Bootstrap 95% percentile interval [59084.3, 91513.1]. These intervals describe sampling variation conditional on this fitted model, not repeated model-selection uncertainty.

See evaluation/metrics.json for baseline comparison, evaluation/predictions.csv for every test prediction and eda/findings.md for data quality. Scaling and median imputation were fitted separately inside every fold. Test outcomes did not select parameters. Model and preprocessing reload reproduced predictions.

Limitations: one holdout split; evaluate transfer to new populations before deployment. Correlations and feature relations are descriptive.
