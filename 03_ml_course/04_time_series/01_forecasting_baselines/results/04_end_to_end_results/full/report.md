# Monthly passenger forecasting

Held-out RMSE: 46.0816 thousand passengers; seasonal-naive RMSE: 46.0816. Ridge alpha 0 was selected using expanding windows ending before the holdout.

This is one-step forecasting: each previous test observation becomes available before predicting the next month. Coefficients and scaling stay frozen across the final test. It is not a 29-month forecast made at a single origin. The increasing trend and seasonal amplitude violate simple stationarity assumptions; inspect training autocorrelations and residual figures. Lag features contain only already observed values. MASE scales errors by training seasonal-naive MAE. The saved archive is the pre-holdout model and history, so reload predicts the first held-out month.
