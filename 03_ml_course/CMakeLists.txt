cmake_minimum_required(VERSION 3.16)

# ============================================================
#  MLCpp - Machine-learning algorithms in C++17.
#
#  The repository is a course.  Everything that is NOT a lesson
#  lives under helper/ (shared infrastructure + shared data), so the
#  only things at the repository root are 01_supervised/,
#  02_unsupervised/, helper/, this CMakeLists.txt and a README.
#
#  Every algorithm module is self-contained: it carries its own
#  implementation (header + .cpp) and its own numbered lessons, one
#  build target per lesson.
#
#  How a lesson builds:
#    1. cmake compiles every *.cpp under helper/ into one static
#       library `ml_core` (like sklean/numpy are pre-built libraries).
#    2. Each module compiles its own implementation into its own static
#       library `ml_<module>` which links `ml_core`.
#    3. add_lesson_executable() turns each lesson .cpp into its own
#       executable linked against the module library.
#    4. Every lesson gets DATA_DIR (absolute path to helper/data), and
#       every lesson is expected to write its artifacts into its own
#       module's results/ folder (reached via COURSE_ROOT).
# ============================================================

project(MLCpp LANGUAGES CXX)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Werror -pedantic)

# Absolute path to the repository root; lessons use it to reach helper/data
# and to write into their module's results/ folder.
get_filename_component(COURSE_ROOT ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
file(TO_CMAKE_PATH ${COURSE_ROOT} COURSE_ROOT)

# Absolute path to the shared data folder (helper/data holds the CSVs).
get_filename_component(ML_DATA_DIR ${COURSE_ROOT}/helper/data ABSOLUTE)
file(TO_CMAKE_PATH ${ML_DATA_DIR} ML_DATA_DIR)

# Central runtime folder for lesson artifacts (used during the transition;
# lessons later write into their own module's results/ folder).
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/runtime)
get_filename_component(RUNTIME_DIR ${CMAKE_BINARY_DIR}/runtime ABSOLUTE)
file(TO_CMAKE_PATH ${RUNTIME_DIR} RUNTIME_DIR)

# ------------------------------------------------------------
#  ml_core - the static library of every { supportive *.cpp under helper/ }.
#  This is the "framework" every algorithm module builds on.
# ------------------------------------------------------------
add_library(ml_core STATIC
  helper/dataframe/dataframe.cpp
  helper/math/matrix.cpp
  helper/math/optim.cpp
  helper/math/metrics.cpp
  helper/data/datasets.cpp
  helper/eval/cross_validation.cpp
  helper/reporting/artifacts.cpp
  helper/plot/plot_svg.cpp
)
# The repo root allows sources to `#include "helper/..."` directly.
target_include_directories(ml_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ------------------------------------------------------------
#  Helper: register a lesson executable.
#  Usage (inside a module CMakeLists):
#     set(ML_MODULE_LIB ml_<name>)           # module's impl library
#     add_lesson_executable(target lesson.cpp)
# ------------------------------------------------------------
function(add_lesson_executable target_name source_file)
  add_executable(${target_name} ${source_file})
  target_compile_features(${target_name} PRIVATE cxx_std_17)
  # Each C++ entry point owns a directory named after its source file.
  # For example: 04_end_to_end.cpp -> results/04_end_to_end_results/.
  get_filename_component(_lesson_stem "${source_file}" NAME_WE)
  set(_module_results "${CMAKE_CURRENT_SOURCE_DIR}/results")
  set(RUN_OUTPUT_DIR "${_module_results}/${_lesson_stem}_results")
  file(TO_CMAKE_PATH "${RUN_OUTPUT_DIR}" RUN_OUTPUT_DIR)
  set_property(TARGET ${target_name} PROPERTY ML_RESULTS_DIR "${RUN_OUTPUT_DIR}")
  set_property(GLOBAL APPEND PROPERTY LESSON_MANIFEST "${target_name}|$<TARGET_FILE:${target_name}>|${RUN_OUTPUT_DIR}")
  file(GLOB _end_to_end_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/[0-9]*_end_to_end.cpp")
  set(_end_to_end_results "")
  list(LENGTH _end_to_end_sources _end_to_end_count)
  if(_end_to_end_count EQUAL 1)
    list(GET _end_to_end_sources 0 _end_to_end_source)
    get_filename_component(_end_to_end_stem "${_end_to_end_source}" NAME_WE)
    set(_end_to_end_results "${_module_results}/${_end_to_end_stem}_results")
  endif()
  target_compile_definitions(${target_name} PRIVATE
    RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR}"
    ML_MODULE_RESULTS_DIR="${_module_results}"
    ML_END_TO_END_RESULTS_DIR="${_end_to_end_results}"
    DATA_DIR="${ML_DATA_DIR}"
    COURSE_ROOT="${COURSE_ROOT}")
  if(ML_MODULE_LIB)
    target_link_libraries(${target_name} PRIVATE ${ML_MODULE_LIB})
  else()
    target_link_libraries(${target_name} PRIVATE ml_core)
  endif()
endfunction()

add_library(ml_tree_models STATIC helper/models/tree_models.cpp)
target_link_libraries(ml_tree_models PUBLIC ml_core)
add_subdirectory(01_supervised)
add_subdirectory(02_unsupervised)
add_subdirectory(03_anomaly_detection)
add_subdirectory(04_time_series)
add_subdirectory(05_text_classification)
add_subdirectory(06_recommender_systems)
add_subdirectory(tests)
