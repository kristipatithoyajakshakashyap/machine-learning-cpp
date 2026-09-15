set(ML_MODULE_LIB ml_rrf)
add_library(${ML_MODULE_LIB} STATIC RandomForest.cpp)
target_include_directories(${ML_MODULE_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${ML_MODULE_LIB} PUBLIC ml_tree_models)

set(ML_MODULE_RESULTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
add_lesson_executable(rrf_theory 01_theory.cpp)
add_lesson_executable(rrf_math_intuition 02_math_intuition.cpp)
add_lesson_executable(rrf_implementation 03_implementation.cpp)
add_lesson_executable(rrf_04_n_estimators 04_n_estimators.cpp)
add_lesson_executable(rrf_end_to_end 05_end_to_end.cpp)

add_lesson_executable(rrf_predict predict.cpp)
