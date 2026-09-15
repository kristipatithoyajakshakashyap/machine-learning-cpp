# random forest regressor

Full supplied dataset run.

Training rows: 16512; untouched test rows: 4128. 5 folds selected parameter 12 using negative RMSE.

Holdout RMSE: 54849.3. Bootstrap 95% percentile interval [52641.1, 57065.4]. These intervals describe sampling variation conditional on this fitted model, not repeated model-selection uncertainty.

See evaluation/metrics.json for baseline comparison, evaluation/predictions.csv for every test prediction and eda/findings.md for data quality. Scaling and median imputation were fitted separately inside every fold. Test outcomes did not select parameters. Model and preprocessing reload reproduced predictions.

Limitations: one holdout split; evaluate transfer to new populations before deployment. Correlations and feature relations are descriptive.
