# Agglomerative (bottom-up hierarchical) clustering.
set(ML_MODULE_LIB ml_aggl)
add_library(${ML_MODULE_LIB} STATIC Agglomerative.cpp)
target_include_directories(${ML_MODULE_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${ML_MODULE_LIB} PUBLIC ml_core)

set(ML_MODULE_RESULTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
add_lesson_executable(uag_theory 01_theory.cpp)
add_lesson_executable(uag_math_intuition 02_math_intuition.cpp)
add_lesson_executable(uag_implementation 03_implementation.cpp)
add_lesson_executable(uag_end_to_end 04_end_to_end.cpp)
