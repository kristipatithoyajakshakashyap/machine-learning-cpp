# ------------------------------------------------------------
#  Module 11 - Final pipeline capstone
#  01_data_clean            Titanic cleaning + features
#  02_feature_table         Numeric feature matrix
#  03_eda_summary           Descriptive stats + 2 plots
#  04_split_pca             Stratified split + PCA on train
#  05_eval_model            Threshold classifier + confusion matrix
# ------------------------------------------------------------

add_lesson_executable(s11_01_clean    01_data_clean.cpp)
add_lesson_executable(s11_02_feat     02_feature_table.cpp)
add_lesson_executable(s11_03_eda      03_eda_summary.cpp)
add_lesson_executable(s11_04_pca      04_split_pca.cpp)
add_lesson_executable(s11_05_eval     05_eval_model.cpp)
add_lesson_executable(s11_predict predict.cpp)
