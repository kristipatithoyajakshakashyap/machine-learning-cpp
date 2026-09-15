# end_to_end

Full supplied dataset run.

Training rows: 353; untouched test rows: 89. 5 folds selected parameter 0.01 using negative RMSE.

Holdout RMSE: 57.4871. Bootstrap 95% percentile interval [50.3607, 64.2818]. These intervals describe sampling variation conditional on this fitted model, not repeated model-selection uncertainty.

See evaluation/metrics.json for baseline comparison, evaluation/predictions.csv for every test prediction and eda/findings.md for data quality. Scaling and median imputation were fitted separately inside every fold. Test outcomes did not select parameters. Model and preprocessing reload reproduced predictions.

Limitations: one holdout split; evaluate transfer to new populations before deployment. Correlations and feature relations are descriptive.
