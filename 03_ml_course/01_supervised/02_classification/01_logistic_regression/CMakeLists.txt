set(ML_MODULE_LIB ml_clog)
add_library(${ML_MODULE_LIB} STATIC LogisticRegression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../01_regression/01_linear_regression/LinearRegression.cpp)
target_include_directories(${ML_MODULE_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${ML_MODULE_LIB} PUBLIC ml_core)

set(ML_MODULE_RESULTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
add_lesson_executable(clog_theory 01_theory.cpp)
add_lesson_executable(clog_math_intuition 02_math_intuition.cpp)
add_lesson_executable(clog_implementation 03_implementation.cpp)
add_lesson_executable(clog_04_multiclass_and_softmax 04_multiclass_and_softmax.cpp)
add_lesson_executable(clog_end_to_end 05_end_to_end.cpp)

add_lesson_executable(clog_predict predict.cpp)
