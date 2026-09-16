# Regression

Predict a continuous target. Every module in this group follows the same shape: a `theory.md` / `math_intuition.md` / `implementation.md` / `exercises.md` set, numbered lesson executables, an end-to-end project built on the shared pipeline in `helper/pipeline/supervised.hpp`, and a `predict` target that serves a saved run. The datasets are California housing for the six classic modules, `diabetes` for the three later ones and for the small lessons, `tips` for one-feature plots.

| Module | Target prefix | What it teaches |
|---|---|---|
| [`01_linear_regression`](01_linear_regression/README.md) | `lin_*` | OLS, ridge, lasso and polynomial features (closed form and gradient descent) |
| [`02_knn_regression`](02_knn_regression/README.md) | `rknn_*` | Distance-weighted averaging of the k closest rows |
| [`03_decision_tree_regressor`](03_decision_tree_regressor/README.md) | `rdtr_*` | CART splits by variance reduction and the depth curve |
| [`04_random_forest_regressor`](04_random_forest_regressor/README.md) | `rrf_*` | Bagged trees with feature subsampling |
| [`05_gradient_boosting_regressor`](05_gradient_boosting_regressor/README.md) | `rgbr_*` | Sequential trees fitted to residuals with shrinkage |
| [`06_mlp_regressor`](06_mlp_regressor/README.md) | `rmlp_*` | Feed-forward network trained by backpropagation |
| [`07_elastic_net`](07_elastic_net/README.md) | `elastic_*` | Mixed L1/L2 penalty solved by coordinate descent |
| [`08_svr`](08_svr/README.md) | `svr_*` | Epsilon-insensitive loss, linear and RBF kernels |
| [`09_gaussian_process`](09_gaussian_process/README.md) | `gp_*` | Kernel posterior with predictive uncertainty |

## Build and run

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target <prefix>_end_to_end <prefix>_predict
build\03_ml_course\01_supervised\01_regression\<module>\<prefix>_end_to_end            # full dataset
build\03_ml_course\01_supervised\01_regression\<module>\<prefix>_end_to_end --quick    # seeded 400-row subset
build\03_ml_course\01_supervised\01_regression\<module>\<prefix>_predict --predict <holdout_features.csv> --model <run>/model
```

Every executable writes into `results/<source stem>_results/` inside its own module. The project run adds `full/` or `quick/` below that. Modules with a `tests/` folder register `<prefix>_numerical` and `<prefix>_workflow` in CTest (`ctest --preset course -R <prefix>`). Each module README lists its files, outputs, tests and takeaways.

## Next group

`../02_classification/01_logistic_regression`