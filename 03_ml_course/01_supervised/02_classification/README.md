# Classification

Predict a class label and, where the model allows, a probability per class. Every module in this group follows the same shape: a `theory.md` / `math_intuition.md` / `implementation.md` / `exercises.md` set, numbered lesson executables, an end-to-end project built on the shared pipeline in `helper/pipeline/supervised.hpp`, and a `predict` target that serves a saved run. The datasets are `breast_cancer` (569 x 30, binary) for every end-to-end project except `10_model_evaluation`, which compares four model families on California housing, so the modules stay comparable on the same holdout. `iris`, `wine`, `penguins`, `titanic` and the generated Gaussian fixtures serve in the lessons.

| Module | Target prefix | What it teaches |
|---|---|---|
| [`01_logistic_regression`](01_logistic_regression/README.md) | `clog_*` | Sigmoid/softmax on a linear score, trained by gradient descent |
| [`02_knn_classification`](02_knn_classification/README.md) | `cknn_*` | Majority vote among the k nearest rows |
| [`03_svm_classification`](03_svm_classification/README.md) | `csvm_*` | Maximum-margin boundary with hinge loss and RBF kernel |
| [`04_naive_bayes`](04_naive_bayes/README.md) | `cnb_*` | Generative Gaussian model with conditional independence |
| [`05_decision_tree_classifier`](05_decision_tree_classifier/README.md) | `cdtc_*` | CART splits by Gini impurity |
| [`06_random_forest_classifier`](06_random_forest_classifier/README.md) | `crfc_*` | Bagged trees, vote-fraction probabilities |
| [`07_adaboost_classifier`](07_adaboost_classifier/README.md) | `cada_*` | Row reweighting and weighted weak learners (SAMME) |
| [`08_gradient_boosting_classifier`](08_gradient_boosting_classifier/README.md) | `cgbc_*` | Per-class residual trees with a softmax head |
| [`09_mlp_classifier`](09_mlp_classifier/README.md) | `cmlp_*` | Neural network with a softmax output layer |
| [`10_model_evaluation`](10_model_evaluation/README.md) | `eval_*` | Metrics, curves, and the holdout/CV discipline, plus model-family selection |
| [`11_lda`](11_lda/README.md) | `lda_*` | Gaussian classes with a pooled covariance (linear boundaries) |
| [`12_qda`](12_qda/README.md) | `qda_*` | Gaussian classes with per-class covariance (quadratic boundaries) |
| [`13_imbalanced_classification`](13_imbalanced_classification/README.md) | `cimb_*` | Class-weighted log-loss and rare-class metrics |

## Build and run

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target <prefix>_end_to_end <prefix>_predict
build\03_ml_course\01_supervised\02_classification\<module>\<prefix>_end_to_end            # full dataset
build\03_ml_course\01_supervised\02_classification\<module>\<prefix>_end_to_end --quick    # seeded 400-row subset
build\03_ml_course\01_supervised\02_classification\<module>\<prefix>_predict --predict <holdout_features.csv> --model <run>/model
```

Every executable writes into `results/<source stem>_results/` inside its own module. The project run adds `full/` or `quick/` below that. Modules with a `tests/` folder register `<prefix>_numerical` and `<prefix>_workflow` in CTest (`ctest --preset course -R <prefix>`). Each module README lists its files, outputs, tests and takeaways.

## Next group

`../03_model_diagnostics/01_feature_importance_and_learning_curves`