cmake_minimum_required(VERSION 3.16)

# ============================================================
#  DataScienceCpp - Complete data-science toolkit in C++17
#  Every lesson .cpp compiles as its own executable that links
#  the dsts static library (our pandas/numpy/scipy equivalent).
# ============================================================

project(DataScienceCpp LANGUAGES CXX)
include(CTest)

option(DS_BUILD_LESSONS "Build the lesson executables" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Werror -pedantic)

# Absolute path to the checked-in data/ folder, exposed to every lesson.
get_filename_component(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data ABSOLUTE)
file(TO_CMAKE_PATH ${DATA_DIR} DATA_DIR)

# ------------------------------------------------------------
#  dsts - the Data Science toolkit static library.
# ------------------------------------------------------------
add_library(dsts STATIC
  src/dtypes.cpp
  src/series.cpp
  src/dataframe.cpp
  src/csv.cpp
  src/stats.cpp
  src/cleaning.cpp
  src/eda.cpp
  src/plots.cpp
src/join.cpp
  src/features.cpp
  src/sampling.cpp
  src/timeseries.cpp
  src/pca.cpp
  src/datetime.cpp
)
target_include_directories(dsts PUBLIC include)

# ------------------------------------------------------------
#  Helper: register a lesson executable.
#  args: target_name source_file
# ------------------------------------------------------------
# Each lesson owns results/<source stem>_results/ next to its source, e.g.
# 04_csv_io.cpp -> results/04_csv_io_results/. Created lazily at run time.
function(add_lesson_executable target_name source_file)
  get_filename_component(_lesson_stem "${source_file}" NAME_WE)
  set(RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/results/${_lesson_stem}_results")
  file(TO_CMAKE_PATH "${RUNTIME_DIR}" RUNTIME_DIR)
  set_property(GLOBAL APPEND PROPERTY LESSON_MANIFEST "${target_name}|$<TARGET_FILE:${target_name}>|${RUNTIME_DIR}")
  add_executable(${target_name} ${source_file})
  target_compile_features(${target_name} PRIVATE cxx_std_17)
  target_compile_definitions(${target_name} PRIVATE
    RUN_OUTPUT_DIR="${RUNTIME_DIR}"
    DATA_DIR="${DATA_DIR}")
  target_link_libraries(${target_name} PRIVATE dsts)
endfunction()

if(DS_BUILD_LESSONS)
add_subdirectory(01_series_dataframe)
add_subdirectory(02_cleaning)
add_subdirectory(03_eda)
add_subdirectory(04_plots)
add_subdirectory(05_joins_reshape)
add_subdirectory(06_feature_engineering)
add_subdirectory(07_statistics_tests)
add_subdirectory(08_train_test)
add_subdirectory(09_time_series)
add_subdirectory(10_dimensionality)
add_subdirectory(11_final_pipeline)
add_subdirectory(00_math_foundations)
add_subdirectory(12_advanced_eda)
add_subdirectory(13_datetime)
endif()
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
