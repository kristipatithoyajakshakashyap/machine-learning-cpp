# end_to_end

Quick verification uses a documented deterministic subset. Run without --quick for the full dataset.

Training rows: 319; untouched test rows: 81. 5 folds selected parameter 0.01 using macro F1.

Holdout macro F1: 0.959398. Bootstrap 95% percentile interval [0.907969, 1]. These intervals describe sampling variation conditional on this fitted model, not repeated model-selection uncertainty.

See evaluation/metrics.json for baseline comparison, evaluation/predictions.csv for every test prediction and eda/findings.md for data quality. Scaling and median imputation were fitted separately inside every fold. Test outcomes did not select parameters. Model and preprocessing reload reproduced predictions.

Limitations: one holdout split; evaluate transfer to new populations before deployment. Correlations and feature relations are descriptive.
