set(ML_MODULE_LIB ml_cada)
add_library(${ML_MODULE_LIB} STATIC AdaBoost.cpp)
target_include_directories(${ML_MODULE_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${ML_MODULE_LIB} PUBLIC ml_tree_models)

set(ML_MODULE_RESULTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
add_lesson_executable(cada_theory 01_theory.cpp)
add_lesson_executable(cada_math_intuition 02_math_intuition.cpp)
add_lesson_executable(cada_implementation 03_implementation.cpp)
add_lesson_executable(cada_end_to_end 04_end_to_end.cpp)

add_lesson_executable(cada_predict predict.cpp)
