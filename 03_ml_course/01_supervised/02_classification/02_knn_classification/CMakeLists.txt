set(ML_MODULE_LIB ml_cknn)
add_library(${ML_MODULE_LIB} STATIC KNN.cpp)
target_include_directories(${ML_MODULE_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${ML_MODULE_LIB} PUBLIC ml_rknn)

set(ML_MODULE_RESULTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
add_lesson_executable(cknn_theory 01_theory.cpp)
add_lesson_executable(cknn_math_intuition 02_math_intuition.cpp)
add_lesson_executable(cknn_implementation 03_implementation.cpp)
add_lesson_executable(cknn_end_to_end 04_end_to_end.cpp)

add_lesson_executable(cknn_predict predict.cpp)
