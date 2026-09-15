# k-means clustering: Lloyd's algorithm with deterministic k-means++ seeding.
set(ML_MODULE_LIB ml_kmeans)
add_library(${ML_MODULE_LIB} STATIC KMeans.cpp)
target_include_directories(${ML_MODULE_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${ML_MODULE_LIB} PUBLIC ml_core)

set(ML_MODULE_RESULTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
add_lesson_executable(ukm_theory 01_theory.cpp)
add_lesson_executable(ukm_math_intuition 02_math_intuition.cpp)
add_lesson_executable(ukm_implementation 03_implementation.cpp)
add_lesson_executable(ukm_end_to_end 04_end_to_end.cpp)
add_lesson_executable(ukm_predict predict.cpp)
