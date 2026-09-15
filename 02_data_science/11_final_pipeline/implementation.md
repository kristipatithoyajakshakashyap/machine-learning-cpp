# Implementation

Preprocessor::fit learns only from training. engineer is the shared deterministic feature definition; dense preserves one explicit feature order; transform reuses standardization and pca_transform. Model::fit learns its threshold from labeled training scores. Versioned persistence records medians, feature names, scaling, PCA means/loadings and class direction at 17-digit precision. Model::load checks version/schema. Each stage is independently runnable but delegates to the same pipeline, preventing drift between lessons.
