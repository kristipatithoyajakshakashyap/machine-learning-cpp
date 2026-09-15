# ------------------------------------------------------------
#  Module 10 - Dimensionality reduction (PCA)
#  01_explained_variance   Eigenvalues + scoop scrre bar
#  02_projection           Scores in PC1-PC2 space (SVG)
#  03_reconstruction       Loss from 1..4 kept axes
#  04_loadings             Variable weights per PC
#  05_standardize          Raw vs z-scored PCA
# ------------------------------------------------------------

add_lesson_executable(s10_01_eigval     01_explained_variance.cpp)
add_lesson_executable(s10_02_proj       02_projection.cpp)
add_lesson_executable(s10_03_reconstruct 03_reconstruction.cpp)
add_lesson_executable(s10_04_loadings   04_loadings.cpp)
add_lesson_executable(s10_05_standardize 05_standardize.cpp)