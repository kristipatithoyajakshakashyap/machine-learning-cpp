# Header-only module: the selection helpers live in helper/eval/
# feature_selection.hpp; the forest is ml_tree_models and the lasso used by
# l1_path is ml_rlin (01_regression/01_linear_regression).
set(ML_MODULE_LIB ml_fsel)
add_library(${ML_MODULE_LIB} INTERFACE)
target_include_directories(${ML_MODULE_LIB} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${ML_MODULE_LIB} INTERFACE ml_tree_models ml_rlin)
set(ML_MODULE_RESULTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
add_lesson_executable(fsel_theory 01_theory.cpp)
add_lesson_executable(fsel_math_intuition 02_math_intuition.cpp)
add_lesson_executable(fsel_implementation 03_implementation.cpp)
add_lesson_executable(fsel_end_to_end 04_end_to_end.cpp)
add_lesson_executable(fsel_predict predict.cpp)
add_lesson_executable(fsel_tests tests/model_test.cpp)
add_test(NAME fsel_numerical COMMAND fsel_tests)
add_test(NAME fsel_workflow COMMAND fsel_end_to_end --quick)
set_tests_properties(fsel_workflow PROPERTIES LABELS "integration")
