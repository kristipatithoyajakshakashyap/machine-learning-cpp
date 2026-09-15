# Theory

Random splits scramble the forecasting problem and can train on future regimes. Expanding-window validation repeatedly fits on a prefix and evaluates the next time block. Final chronological holdout remains untouched during parameter selection.
