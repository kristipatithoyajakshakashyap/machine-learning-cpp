cmake_minimum_required(VERSION 3.16)
project(CppDataScienceCurriculum LANGUAGES CXX)
include(CTest)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_subdirectory(01_get_to_know)
add_subdirectory(02_data_science)
add_subdirectory(03_ml_course)

# One line per lesson executable: target|exe_path|results_dir.
# Consumed by scripts/run_all_lessons.ps1 to run every lesson and capture output.
get_property(_lesson_manifest GLOBAL PROPERTY LESSON_MANIFEST)
list(JOIN _lesson_manifest "\n" _lesson_manifest_text)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/lesson_manifest.txt" CONTENT "${_lesson_manifest_text}\n")
