# Complete course structure

All implementations, examples and numerical tests are C++17. Markdown explains the theory; CMake builds the C++ programs and two PowerShell scripts run them and check their results. Study order: `learning_path.md`. Which files to write and which are generated: `file_guide.md`. CSV/text datasets and generated SVG/JSON/CSV reports support those programs. There are no Python scripts, notebooks or Python dependencies.

```text
MachineLearning/
├── CMakeLists.txt / CMakePresets.json
├── .vscode/
├── README.md
├── 01_get_to_know/
│   ├── 00_setup/
│   ├── 01_basics/
│   ├── 02_memory_management/
│   ├── 03_oop/
│   ├── 04_error_handling/
│   ├── 05_file_handling/
│   ├── 06_logging/
│   ├── 07_stl_core/
│   ├── 08_templates_generics/
│   ├── 09_advanced_language/
│   ├── 10_project_structure/
│   ├── 11_capstone/
│   ├── 12_debugging_and_testing/
│   ├── 13_numerical_computing/
│   └── tests/
├── 02_data_science/
│   ├── 00_math_foundations/
│   │   ├── 01_vectors_and_matrices/
│   │   ├── 02_linear_algebra/
│   │   ├── 03_probability/
│   │   ├── 04_statistics/
│   │   ├── 05_derivatives_and_gradients/
│   │   ├── 06_optimization/
│   │   └── 07_eigen_svd/
│   ├── 01_series_dataframe/
│   ├── 02_cleaning/
│   ├── 03_eda/
│   ├── 04_plots/
│   ├── 05_joins_reshape/
│   ├── 06_feature_engineering/
│   ├── 07_statistics_tests/
│   ├── 08_train_test/
│   ├── 09_time_series/
│   ├── 10_dimensionality/
│   ├── 11_final_pipeline/
│   ├── 12_advanced_eda/
│   │   ├── 01_data_quality/
│   │   ├── 02_missingness_analysis/
│   │   ├── 03_outliers_and_distributions/
│   │   ├── 04_categorical_relationships/
│   │   ├── 05_multivariate_analysis/
│   │   ├── 06_target_relationships/
│   │   ├── 07_temporal_and_geographic_analysis/
│   │   └── 08_eda_report/
│   ├── 13_datetime/
│   ├── include/ / src/                 C++ tabular/statistics/plotting toolkit
│   ├── data/                          Shared educational datasets
│   └── tests/
├── 03_ml_course/
│   ├── 01_supervised/
│   │   ├── 01_regression/
│   │   │   ├── 01_linear_regression/   OLS, ridge, lasso, polynomial lessons
│   │   │   ├── 02_knn_regression/
│   │   │   ├── 03_decision_tree_regressor/
│   │   │   ├── 04_random_forest_regressor/
│   │   │   ├── 05_gradient_boosting_regressor/
│   │   │   ├── 06_mlp_regressor/
│   │   │   ├── 07_elastic_net/
│   │   │   ├── 08_svr/
│   │   │   └── 09_gaussian_process/
│   │   └── 02_classification/
│   │       ├── 01_logistic_regression/
│   │       ├── 02_knn_classification/
│   │       ├── 03_svm_classification/
│   │       ├── 04_naive_bayes/
│   │       ├── 05_decision_tree_classifier/
│   │       ├── 06_random_forest_classifier/
│   │       ├── 07_adaboost_classifier/
│   │       ├── 08_gradient_boosting_classifier/
│   │       ├── 09_mlp_classifier/
│   │       ├── 10_model_evaluation/
│   │       ├── 11_lda/
│   │       ├── 12_qda/
│   │       └── 13_imbalanced_classification/
│   │   ├── 03_model_diagnostics/
│   │   │   ├── 01_feature_importance_and_learning_curves/
│   │   │   ├── 02_feature_selection/
│   │   │   └── 03_hyperparameter_search/
│   │   └── 04_ensembles/
│   │       └── 01_voting_and_stacking/
│   ├── 02_unsupervised/
│   │   ├── 01_kmeans/
│   │   ├── 02_agglomerative/           Hierarchical clustering and dendrograms
│   │   ├── 03_cluster_evaluation/      Silhouette and stability comparisons
│   │   ├── 04_dbscan/
│   │   ├── 05_gaussian_mixture/        Diagonal covariance EM
│   │   ├── 06_pca/
│   │   ├── 07_tsne/
│   │   └── tests/
│   ├── 03_anomaly_detection/
│   │   ├── 01_isolation_forest/
│   │   ├── 02_local_outlier_factor/
│   │   └── 03_one_class_svm/
│   ├── 04_time_series/
│   │   ├── 01_forecasting_baselines/
│   │   ├── 02_lag_feature_regression/
│   │   ├── 03_rolling_origin_evaluation/
│   │   ├── 04_exponential_smoothing/    Holt-Winters
│   │   └── 05_arima/
│   ├── 05_text_classification/
│   │   ├── 01_text_preprocessing/
│   │   ├── 02_count_and_tfidf/
│   │   ├── 03_multinomial_naive_bayes/
│   │   ├── 04_linear_text_classifier/
│   │   └── data/                      SMS corpus and provenance
│   ├── 06_recommender_systems/
│   │   ├── 01_popularity_baseline/
│   │   ├── 02_item_based_collaborative_filtering/
│   │   ├── 03_matrix_factorization/
│   │   ├── 04_recommender_evaluation/
│   │   └── data/                      MovieLens 100K and upstream conditions
│   ├── helper/                        Shared C++ data/math/models/pipelines
│   └── tests/                         C++ numerical and reload verification
├── docs/                              project_structure.md, learning_path.md, file_guide.md
├── scripts/                           run_all_lessons.ps1, verify_artifacts.ps1
└── build/                             Generated compiler output, not course source
```

Algorithm modules contain a README, theory, mathematical intuition, implementation notes, exercises, C++ lessons, an end-to-end program and CMake targets. Predict entry points are supplied where new-row prediction is defined. Hierarchical clustering and DBSCAN preserve fitted assignments instead of inventing a prediction rule.

Every C++ entry point owns a results folder named after its file (`<stem>_results`). It holds only files the lesson writes itself (data, plots, reports, model state); print-only lessons have no folder. For example:

```text
04_end_to_end.cpp
predict.cpp
results/
├── 03_implementation_results/   (only when the lesson writes files)
├── 04_end_to_end_results/
│   ├── full/       Supervised/applied full run: data, EDA, validation, model, evaluation
│   └── quick/      Separate smoke run
└── predict_results/
```

The filename numbering is preserved: `05_end_to_end.cpp` uses `05_end_to_end_results`, while `08_end_to_end.cpp` uses `08_end_to_end_results`. Unsupervised/anomaly runs write their data, EDA, validation, model, evaluation and report directly under their named end-to-end folder. Rerunning a workflow replaces its corresponding generated files. `scripts/run_all_lessons.ps1` runs every lesson from `build/lesson_manifest.txt` and regenerates all results.
