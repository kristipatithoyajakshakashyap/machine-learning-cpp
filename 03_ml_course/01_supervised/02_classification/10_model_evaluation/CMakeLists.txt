# This module has no implementation of its own: it evaluates the algorithms
# implemented by other modules.  ml_eval compiles each algorithm's source
# exactly ONCE (from its home module) so the executables do not hit ODR
# duplicate-symbol errors.
set(ML_MODULE_LIB ml_eval)
add_library(${ML_MODULE_LIB} STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../01_regression/04_random_forest_regressor/RandomForest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../01_regression/01_linear_regression/LinearRegression.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/../01_logistic_regression/LogisticRegression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../04_naive_bayes/GaussianNB.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../03_svm_classification/SVM.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../09_mlp_classifier/MLP.cpp
)
target_include_directories(${ML_MODULE_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${ML_MODULE_LIB} PUBLIC ml_tree_models ml_rmlp ml_rknn)

set(ML_MODULE_RESULTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
add_lesson_executable(eval_theory 01_theory.cpp)
add_lesson_executable(eval_math_intuition 02_math_intuition.cpp)
add_lesson_executable(eval_implementation 03_implementation.cpp)
add_lesson_executable(eval_end_to_end 04_end_to_end.cpp)

add_lesson_executable(eval_predict predict.cpp)
