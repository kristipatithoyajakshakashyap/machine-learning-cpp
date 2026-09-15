# This module's own implementation, self-contained in this folder.
set(ML_MODULE_LIB ml_rlin)
add_library(${ML_MODULE_LIB} STATIC LinearRegression.cpp)
target_include_directories(${ML_MODULE_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${ML_MODULE_LIB} PUBLIC ml_core)

set(ML_MODULE_RESULTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
add_lesson_executable(lin_theory 01_theory.cpp)
add_lesson_executable(lin_math_intuition 02_math_intuition.cpp)
add_lesson_executable(lin_implementation 03_implementation.cpp)
add_lesson_executable(lin_04_ridge_regression 04_ridge_regression.cpp)
add_lesson_executable(lin_05_lasso_regression 05_lasso_regression.cpp)
add_lesson_executable(lin_end_to_end 08_end_to_end.cpp)
add_lesson_executable(lin_07_polynomial_regression 07_polynomial_regression.cpp)

add_lesson_executable(lin_predict predict.cpp)
