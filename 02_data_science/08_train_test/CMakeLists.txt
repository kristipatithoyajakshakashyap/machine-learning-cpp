# ------------------------------------------------------------
#  Module 08 - Train/test splits and sampling
#  01_random_split           Deterministic random split
#  02_stratified             Stratify on an outcome column
#  03_sampling_bias          First-N rows vs random rows
#  04_boot_sample_ci         Seeded indices + bootstrap CI
#  05_eval_split_size        Split-size / seed stability
# ------------------------------------------------------------

add_lesson_executable(s08_01_split   01_random_split.cpp)
add_lesson_executable(s08_02_strat   02_stratified.cpp)
add_lesson_executable(s08_03_bias    03_sampling_bias.cpp)
add_lesson_executable(s08_04_boot    04_boot_sample_ci.cpp)
add_lesson_executable(s08_05_size    05_eval_split_size.cpp)
add_lesson_executable(s08_06_cv      06_cross_validation.cpp)
