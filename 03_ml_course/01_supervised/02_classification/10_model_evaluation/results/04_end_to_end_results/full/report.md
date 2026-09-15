# Model comparison: 0 ridge, 1 kNN, 2 forest, 3 MLP

Full supplied dataset run.

Training rows: 16512; untouched test rows: 4128. 5 folds selected parameter 1 using negative RMSE.

Holdout RMSE: 60704.4. Bootstrap 95% percentile interval [58250.3, 63059]. These intervals describe sampling variation conditional on this fitted model, not repeated model-selection uncertainty.

See evaluation/metrics.json for baseline comparison, evaluation/predictions.csv for every test prediction and eda/findings.md for data quality. Scaling and median imputation were fitted separately inside every fold. Test outcomes did not select parameters. Model and preprocessing reload reproduced predictions.

Limitations: one holdout split; evaluate transfer to new populations before deployment. Correlations and feature relations are descriptive.
