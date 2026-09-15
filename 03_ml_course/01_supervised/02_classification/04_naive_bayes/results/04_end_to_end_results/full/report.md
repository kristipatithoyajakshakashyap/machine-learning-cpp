# naive bayes

Full supplied dataset run.

Training rows: 454; untouched test rows: 115. 5 folds selected parameter 1e-09 using macro F1.

Holdout macro F1: 0.874675. Bootstrap 95% percentile interval [0.806397, 0.934686]. These intervals describe sampling variation conditional on this fitted model, not repeated model-selection uncertainty.

See evaluation/metrics.json for baseline comparison, evaluation/predictions.csv for every test prediction and eda/findings.md for data quality. Scaling and median imputation were fitted separately inside every fold. Test outcomes did not select parameters. Model and preprocessing reload reproduced predictions.

Limitations: one holdout split; evaluate transfer to new populations before deployment. Correlations and feature relations are descriptive.
