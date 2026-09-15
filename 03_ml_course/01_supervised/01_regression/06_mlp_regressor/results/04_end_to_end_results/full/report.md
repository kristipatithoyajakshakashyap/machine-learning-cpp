# mlp regressor

Full supplied dataset run.

Training rows: 16512; untouched test rows: 4128. 5 folds selected parameter 0.01 using negative RMSE.

Holdout RMSE: 64974.4. Bootstrap 95% percentile interval [62830.9, 67190.2]. These intervals describe sampling variation conditional on this fitted model, not repeated model-selection uncertainty.

See evaluation/metrics.json for baseline comparison, evaluation/predictions.csv for every test prediction and eda/findings.md for data quality. Scaling and median imputation were fitted separately inside every fold. Test outcomes did not select parameters. Model and preprocessing reload reproduced predictions.

Limitations: one holdout split; evaluate transfer to new populations before deployment. Correlations and feature relations are descriptive.
