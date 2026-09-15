set(ML_MODULE_LIB ml_cdtc)
add_library(${ML_MODULE_LIB} STATIC DecisionTree.cpp)
target_include_directories(${ML_MODULE_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${ML_MODULE_LIB} PUBLIC ml_tree_models)

set(ML_MODULE_RESULTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
add_lesson_executable(cdtc_theory 01_theory.cpp)
add_lesson_executable(cdtc_math_intuition 02_math_intuition.cpp)
add_lesson_executable(cdtc_implementation 03_implementation.cpp)
add_lesson_executable(cdtc_end_to_end 04_end_to_end.cpp)

add_lesson_executable(cdtc_predict predict.cpp)
