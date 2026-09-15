# 10_project_structure - Real programs, real folders

# The static library: src/hello_utils.cpp is compiled ONCE...
add_library(gtk_utils STATIC src/hello_utils.cpp)
target_include_directories(gtk_utils PUBLIC include)

# ...and any lesson executable can link it and #include its headers.
add_lesson_executable(conc_program_layout 01_program_layout.cpp)
target_link_libraries(conc_program_layout PRIVATE gtk_utils)

add_lesson_executable(conc_include_guards 02_include_guards.cpp)
target_link_libraries(conc_include_guards PRIVATE gtk_utils)

add_lesson_executable(conc_matrix_library 03_matrix_library.cpp)
target_link_libraries(conc_matrix_library PRIVATE gtk_utils)