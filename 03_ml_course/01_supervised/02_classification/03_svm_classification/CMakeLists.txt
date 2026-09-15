set(ML_MODULE_LIB ml_csvm)
add_library(${ML_MODULE_LIB} STATIC SVM.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../01_regression/01_linear_regression/LinearRegression.cpp)
target_include_directories(${ML_MODULE_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${ML_MODULE_LIB} PUBLIC ml_core)

set(ML_MODULE_RESULTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
add_lesson_executable(csvm_theory 01_theory.cpp)
add_lesson_executable(csvm_math_intuition 02_math_intuition.cpp)
add_lesson_executable(csvm_implementation 03_implementation.cpp)
add_lesson_executable(csvm_04_rbf_kernel 04_rbf_kernel.cpp)
add_lesson_executable(csvm_05_c_curve 05_c_curve.cpp)
add_lesson_executable(csvm_end_to_end 06_end_to_end.cpp)

add_lesson_executable(csvm_predict predict.cpp)
