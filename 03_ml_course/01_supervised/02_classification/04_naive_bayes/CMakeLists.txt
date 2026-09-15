set(ML_MODULE_LIB ml_cnb)
add_library(${ML_MODULE_LIB} STATIC GaussianNB.cpp)
target_include_directories(${ML_MODULE_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${ML_MODULE_LIB} PUBLIC ml_core)

set(ML_MODULE_RESULTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
add_lesson_executable(cnb_theory 01_theory.cpp)
add_lesson_executable(cnb_math_intuition 02_math_intuition.cpp)
add_lesson_executable(cnb_implementation 03_implementation.cpp)
add_lesson_executable(cnb_end_to_end 04_end_to_end.cpp)

add_lesson_executable(cnb_predict predict.cpp)
