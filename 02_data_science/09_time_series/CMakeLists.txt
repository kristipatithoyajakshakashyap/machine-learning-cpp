# ------------------------------------------------------------
#  Module 09 - Time series
#  01_components       Raw series + 12-month moving average
#  02_seasonality      Month-of-year profile
#  03_forecast_bench   Naive / mean / seasonal-naive RMSE
#  04_trend_linreg     Linear trend via least squares
#  05_acf              Autocorrelation by lag
# ------------------------------------------------------------

add_lesson_executable(s09_01_components     01_components.cpp)
add_lesson_executable(s09_02_seasonality    02_seasonality.cpp)
add_lesson_executable(s09_03_bench          03_forecast_benchmark.cpp)
add_lesson_executable(s09_04_trend          04_trend_linreg.cpp)
add_lesson_executable(s09_05_acf            05_acf.cpp)