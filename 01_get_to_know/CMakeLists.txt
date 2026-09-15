cmake_minimum_required(VERSION 3.16)

# ============================================================
#  Zero-to-Advanced C++ Course - Root Build File
#  Target standard: C++17
#  Every lesson .cpp compiles as its own executable.
# ============================================================

project(MachineLearningCppCourse LANGUAGES CXX)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Strict warnings everywhere: every warning becomes an error.
add_compile_options(-Wall -Wextra -Werror -pedantic)

# ------------------------------------------------------------
#  Helper: register a standalone lesson executable.
#  args:       target_name   source_file
# ------------------------------------------------------------
# Each lesson owns results/<source stem>_results/ next to its source, e.g.
# 04_loops.cpp -> results/04_loops_results/. The directory is created lazily
# by the lesson (or by scripts/run_all_lessons.ps1), never at configure time.
function(add_lesson_executable target_name source_file)
  get_filename_component(_lesson_stem "${source_file}" NAME_WE)
  set(RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/results/${_lesson_stem}_results")
  file(TO_CMAKE_PATH "${RUNTIME_DIR}" RUNTIME_DIR)
  add_executable(${target_name} ${source_file})
  target_compile_features(${target_name} PRIVATE cxx_std_17)
  target_compile_definitions(${target_name} PRIVATE RUN_OUTPUT_DIR="${RUNTIME_DIR}")
  set_property(GLOBAL APPEND PROPERTY LESSON_MANIFEST "${target_name}|$<TARGET_FILE:${target_name}>|${RUNTIME_DIR}")
endfunction()

add_subdirectory(00_setup)
add_subdirectory(01_basics)
add_subdirectory(02_memory_management)
add_subdirectory(03_oop)
add_subdirectory(04_error_handling)
add_subdirectory(05_file_handling)
add_subdirectory(06_logging)
add_subdirectory(07_stl_core)
add_subdirectory(08_templates_generics)
add_subdirectory(09_advanced_language)
add_subdirectory(10_project_structure)
add_subdirectory(11_capstone)
add_subdirectory(12_debugging_and_testing)
add_subdirectory(13_numerical_computing)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
