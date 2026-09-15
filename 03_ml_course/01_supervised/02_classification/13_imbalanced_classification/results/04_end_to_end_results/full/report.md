# end_to_end

Full supplied dataset run.

Training rows: 309; untouched test rows: 78. 5 folds selected parameter 2 using macro F1.

Holdout macro F1: 0.951097. Bootstrap 95% percentile interval [0.775647, 1]. These intervals describe sampling variation conditional on this fitted model, not repeated model-selection uncertainty.

See evaluation/metrics.json for baseline comparison, evaluation/predictions.csv for every test prediction and eda/findings.md for data quality. Scaling and median imputation were fitted separately inside every fold. Test outcomes did not select parameters. Model and preprocessing reload reproduced predictions.

Limitations: one holdout split; evaluate transfer to new populations before deployment. Correlations and feature relations are descriptive.
