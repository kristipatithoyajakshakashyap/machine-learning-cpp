# Learning path

Three tracks, taken in order. Each module folder has a `README.md`; read it first, then the lesson `.cpp` files in numeric order, build, run, read the console output, and inspect any files the lesson wrote to `results/<lesson>_results/` (CSV, SVG, JSON, model state). Print-only lessons have no results folder.

```
cmake --preset course && cmake --build --preset course     # build everything once
./scripts/run_all_lessons.ps1 -Filter '<target regex>'     # run one lesson or a module
```

## Track 1: C++ basics (`01_get_to_know/`)

Do the modules top to bottom. Each lesson file is self-contained and prints what it teaches.

| Order | Module | Lessons (in order) | You can move on when |
|---|---|---|---|
| 1 | `00_setup` | hello | build + run works |
| 2 | `01_basics` | variables/types, operators, control flow, loops, functions, scope/lifetime, strings, user input | you can write a small console program |
| 3 | `02_memory_management` | arrays/vectors, pointers, references, dynamic memory, smart pointers | you know when to use `unique_ptr` vs raw pointer |
| 4 | `03_oop` | classes, constructors, encapsulation, inheritance, polymorphism, operator overloading, rule of five | you can design a class with correct copy/move |
| 5 | `04_error_handling` | exceptions, custom exceptions, noexcept, failures and status | you choose between exceptions and status returns |
| 6 | `05_file_handling` | text files, binary files, CSV files, filesystem | you can read/write CSV and walk directories |
| 7 | `06_logging` | logging basics, to file, multichannel, reusable logger | you have a logger you would reuse |
| 8 | `07_stl_core` | containers overview, vector/list, stack/queue, set/map, iterators, algorithms, array and string_view, random and chrono, regex | you reach for `<algorithm>` before writing loops |
| 9 | `08_templates_generics` | function templates, class templates, specialization, variadic templates, type traits and SFINAE | you can write a generic container/function |
| 10 | `09_advanced_language` | lambdas, move semantics, auto/decltype, const correctness, optional/variant, multithreading, namespaces, constexpr and enum class, mutex/atomic/async, std::function and callbacks | you read modern C++ fluently |
| 11 | `10_project_structure` | program layout, include guards, matrix library | you can split code into headers, sources, and a static library |
| 12 | `11_capstone` | stats calculator, CSV reader, normalizer, logger integration, tests | you built a small tested pipeline |
| 13 | `12_debugging_and_testing` | contracts, sanitizers and debugger, test harness | you assert preconditions, reproduce a bug under ASan/gdb, and write a test |
| 14 | `13_numerical_computing` | precision | you understand floating-point error and compensated sums |

## Track 2: Data science basics (`02_data_science/`)

Start with math foundations, then the tabular toolkit in order. Do `12_advanced_eda` before `11_final_pipeline`.

| Order | Module | What you learn |
|---|---|---|
| 1 | `00_math_foundations/01..07` | vectors/matrices, linear algebra, probability, statistics, derivatives/gradients, optimization, eigen decomposition and SVD (one worked example each) |
| 2 | `01_series_dataframe` | Series, DataFrame, CSV I/O, selection and masks |
| 3 | `02_cleaning` | missing values, duplicates, outliers, encoding, scaling, string columns |
| 4 | `03_eda` | summaries/counts, distributions, correlation, grouped analysis |
| 5 | `04_plots` | line, scatter, bar, histogram, box, heatmap, multi-panel (SVG output) |
| 6 | `05_joins_reshape` | merge/join, inner vs left, concat, pivot, melt |
| 7 | `06_feature_engineering` | derived columns, binning, log transform, indicators, feature correlation |
| 8 | `07_statistics_tests` | confidence intervals, t-tests, chi-square, normality, bootstrap, one-way ANOVA, Mann-Whitney and Kruskal-Wallis, effect sizes and multiple testing, Bayesian basics |
| 9 | `08_train_test` | random split, stratified split, sampling bias, bootstrap CI, split-size effects, k-fold cross-validation |
| 10 | `09_time_series` | components, seasonality, forecast benchmarks, trend regression, ACF |
| 11 | `10_dimensionality` | explained variance, projection, reconstruction, loadings, standardization (PCA) |
| 12 | `12_advanced_eda/01..08` | data quality, missingness, outliers/distributions, categorical relationships, multivariate, target relationships, temporal, full EDA report |
| 13 | `13_datetime` | date parsing and components, resampling monthly to quarterly/yearly, rolling windows and cumulative statistics |
| 14 | `11_final_pipeline` | Titanic end-to-end: clean, feature table, EDA summary, split + PCA, evaluate model, then `predict.cpp` reloads the saved model |

Rules that carry into ML: split before fitting anything; fit medians/scales/PCA on training rows only; evaluate the hold-out once.

## Track 3: Machine learning in C++ (`03_ml_course/`)

### How to learn every algorithm (same format in every module)

Each algorithm folder contains the same five steps. Do them in this order, never skip a step:

| Step | Read | Run | Goal |
|---|---|---|---|
| 1 | `README.md`, `theory.md` | `01_theory.cpp` | assumptions, when the algorithm applies, when it fails |
| 2 | `math_intuition.md` | `02_math_intuition.cpp` | objective function, derivation, update rule, hand-computed tiny example |
| 3 | `implementation.md`, `<Algorithm>.hpp/.cpp` | `03_implementation.cpp` | trace the C++ fit/predict code line by line |
| 4 | nothing new | `NN_end_to_end.cpp` (full, then `--quick`) | data prep, EDA, leakage-free model selection, evaluation, saved model; inspect `results/NN_end_to_end_results/full/report.md` and `evaluation/metrics.json` |
| 5 | `exercises.md` | `predict.cpp`, `tests/model_test.cpp` | fresh-process reload of the saved model; change data/parameters and explain what changed |

Some modules add numbered lessons between step 3 and the end-to-end run (for example ridge, lasso and polynomial variants in linear regression, or kernel and C-curve lessons in SVM). Do them in numeric order.

### Algorithm order

**Supervised: regression** (`01_supervised/01_regression/`)

1. `01_linear_regression` (OLS, ridge, lasso, polynomial)
2. `02_knn_regression`
3. `03_decision_tree_regressor`
4. `04_random_forest_regressor`
5. `05_gradient_boosting_regressor`
6. `06_mlp_regressor`
7. `07_elastic_net`
8. `08_svr`
9. `09_gaussian_process` (kernels, posterior uncertainty, marginal likelihood)

**Supervised: classification** (`01_supervised/02_classification/`)

10. `01_logistic_regression` (binary, multiclass/softmax)
11. `02_knn_classification`
12. `03_svm_classification` (linear, RBF, C-curve)
13. `04_naive_bayes`
14. `05_decision_tree_classifier`
15. `06_random_forest_classifier`
16. `07_adaboost_classifier`
17. `08_gradient_boosting_classifier`
18. `09_mlp_classifier`
19. `10_model_evaluation` (metrics, calibration, thresholds; revisit after every classifier)
20. `11_lda`
21. `12_qda`
22. `13_imbalanced_classification` (class weights, PR vs ROC, threshold moving)

**Supervised: model diagnostics** (`01_supervised/03_model_diagnostics/`)

23. `01_feature_importance_and_learning_curves` (permutation importance, learning curves, bias/variance diagnosis)
24. `02_feature_selection` (filter, wrapper, embedded)
25. `03_hyperparameter_search` (grid vs random search, nested cross-validation)

**Supervised: ensembles** (`01_supervised/04_ensembles/`)

26. `01_voting_and_stacking` (hard/soft voting, out-of-fold stacking)

**Unsupervised** (`02_unsupervised/`)

27. `01_kmeans`
28. `02_agglomerative` (hierarchical clustering)
29. `03_cluster_evaluation` (silhouette, stability)
30. `04_dbscan`
31. `05_gaussian_mixture`
32. `06_pca`
33. `07_tsne` (visualization embedding, perplexity, trustworthiness)

**Anomaly detection** (`03_anomaly_detection/`)

34. `01_isolation_forest`
35. `02_local_outlier_factor`
36. `03_one_class_svm`

**Time series** (`04_time_series/`)

37. `01_forecasting_baselines`
38. `02_lag_feature_regression`
39. `03_rolling_origin_evaluation`
40. `04_exponential_smoothing` (Holt-Winters)
41. `05_arima`

**Text classification** (`05_text_classification/`)

42. `01_text_preprocessing`
43. `02_count_and_tfidf`
44. `03_multinomial_naive_bayes`
45. `04_linear_text_classifier`

**Recommender systems** (`06_recommender_systems/`)

46. `01_popularity_baseline`
47. `02_item_based_collaborative_filtering`
48. `03_matrix_factorization`
49. `04_recommender_evaluation`

### Validation rules used everywhere

- Hold-out split is made before any preprocessing or model selection; classification splits are stratified.
- Medians, scales and PCA are fitted inside each cross-validation fold, never on the full data.
- Time series and interaction data use chronological splits; clustering scores are internal, not supervised.
- Every saved model is reloaded in a fresh process (`predict.cpp`) and must reproduce the evaluation predictions exactly.
- A successful build proves nothing about predictive quality; read `report.md` and `metrics.json` before drawing conclusions.

## Datasets

| File | Used by |
|---|---|
| `02_data_science/data/{iris,penguins,tips,titanic,flights,air_passengers}.csv` | data science track |
| `03_ml_course/helper/data/{iris,penguins,titanic,wine,breast_cancer,diabetes,california_housing,tips,air_passengers,flights}.csv` | supervised, unsupervised, anomaly, time series |
| `03_ml_course/05_text_classification/data/SMSSpamCollection` | text classification |
| `03_ml_course/06_recommender_systems/data/u.data` | recommender systems |

All datasets are checked in; nothing is downloaded at build or run time.
