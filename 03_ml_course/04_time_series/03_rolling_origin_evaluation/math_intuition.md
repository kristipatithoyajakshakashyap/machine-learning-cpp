# Math intuition

For origin o, train on [0,o) and validate [o,o+12). RMSE=sqrt(mean((y-yhat)^2)). Average the validation-window RMSE to choose regularization. One-step evaluation reveals y_t only after predicting it; multi-step recursive evaluation must instead feed previous forecasts.
