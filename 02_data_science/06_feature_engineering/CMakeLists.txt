# ------------------------------------------------------------
#  Module 06 - Feature engineering
#  01_derived_columns   Ratios and percentages via Series ops
#  02_binning           cut() with explicit edges, qcut edges
#  03_log_transform     log1p turns exponential growth linear
#  04_indicator_features Presence + child + family-size flags
#  05_feature_correlation  Which features track the target best
# ------------------------------------------------------------

add_lesson_executable(s06_01_derived   01_derived_columns.cpp)
add_lesson_executable(s06_02_binning   02_binning.cpp)
add_lesson_executable(s06_03_log       03_log_transform.cpp)
add_lesson_executable(s06_04_indicators 04_indicator_features.cpp)
add_lesson_executable(s06_05_corr      05_feature_correlation.cpp)