set(ML_MODULE_LIB ml_cmlp)
add_library(${ML_MODULE_LIB} STATIC MLP.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../01_regression/01_linear_regression/LinearRegression.cpp)
target_include_directories(${ML_MODULE_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${ML_MODULE_LIB} PUBLIC ml_rmlp)

set(ML_MODULE_RESULTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
add_lesson_executable(cmlp_theory 01_theory.cpp)
add_lesson_executable(cmlp_math_intuition 02_math_intuition.cpp)
add_lesson_executable(cmlp_implementation 03_implementation.cpp)
add_lesson_executable(cmlp_end_to_end 04_end_to_end.cpp)

add_lesson_executable(cmlp_predict predict.cpp)
