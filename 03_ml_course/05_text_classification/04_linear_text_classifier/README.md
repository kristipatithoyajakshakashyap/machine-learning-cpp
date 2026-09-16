# Linear text classification

Method. L2-regularised binary logistic regression on sparse TF-IDF rows.
`p(spam|x) = sigmoid(w . x + b)`, trained by full-batch gradient descent on
mean cross-entropy plus `lambda * ||w||^2 / 2`. The step size is the reciprocal
of a Hessian upper bound, so no learning-rate tuning is needed and training is
deterministic.

Core idea. Unlike Naive Bayes, a discriminative model learns the label
boundary directly and gives a token a weight that contradicts its raw
frequency. Regularisation `lambda` controls how much rare but decisive spam
words pull the boundary. Too much shrinkage underfits them.

## Dataset

UCI SMS Spam Collection (`../data/SMSSpamCollection`). After TF-IDF the problem
is nearly linearly separable, which makes a linear model both accurate and
interpretable through its largest weights.

## Prerequisites

`02_count_and_tfidf` (sparse rows), logistic regression from
`01_supervised/02_classification/01_logistic_regression`, gradients.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Discriminative vs generative, regularisation. | - |
| `math_intuition.md` | - | Loss, gradient and the Lipschitz step bound. | - |
| `implementation.md` | - | Walkthrough of `LinearTextClassifier`. | - |
| `exercises.md` | - | Practice questions. | - |
| `Model.hpp` | (header) | `LinearTextClassifier` (fit / probabilities / save / load). | - |
| `01_theory.cpp` | `text_linear_theory` | Lesson text plus a toy encoding. | prints only |
| `02_math_intuition.cpp` | `text_linear_math_intuition` | Gradient and step-size derivation. | prints only |
| `03_implementation.cpp` | `text_linear_implementation` | Training-loss curve and the largest learned weights. | `results/03_implementation_results/` |
| `04_end_to_end.cpp` | `text_linear_end_to_end` | Full project: TF-IDF + logistic regression, lambda in {1e-4, 1e-3}. | `results/04_end_to_end_results/{full,quick}/` |
| `predict.cpp` | `text_linear_predict` | Reloads a saved run and scores new messages. | `results/predict_results/predictions.csv` |
| `tests/model_test.cpp` | `text_linear_tests` | Numerical fixture for the shared text stack. | prints only |

## Build and run

From the repository root (MinGW toolchain, `D:/msys64/ucrt64/bin` on PATH):

```powershell
cmake --preset course
cmake --build --preset course --target text_linear_implementation text_linear_end_to_end text_linear_predict
build/03_ml_course/05_text_classification/04_linear_text_classifier/text_linear_implementation.exe
build/03_ml_course/05_text_classification/04_linear_text_classifier/text_linear_end_to_end.exe          # full run
build/03_ml_course/05_text_classification/04_linear_text_classifier/text_linear_end_to_end.exe --quick  # smoke run
```

## Predicting new messages

```powershell
text_linear_predict --predict ../data/inference_messages.txt --model results/04_end_to_end_results/full/model
```

One message per line. Output `results/predict_results/predictions.csv`
(`row,p_spam,prediction`).

## Results layout

- `results/03_implementation_results/` - `training_loss_per_epoch.csv/.svg`, `largest_weights.csv/.svg`.
- `results/04_end_to_end_results/full/` (or `quick/`) - `data/`, `eda/`,
  `validation/`, `evaluation/`, `model/`, `inference/`, `run_manifest.json`,
  `report.md`, `execution.log` (same layout as the other text projects).
- `results/predict_results/predictions.csv`.

Re-running a target replaces its folder. There is no `output.txt`. Read `report.md`.

## Tests

`ctest --test-dir build -R text_linear` runs `text_linear_numerical` (the
logistic fixture must separate the two toy documents) and `text_linear_workflow`.

## Key takeaways

- With a Lipschitz-bounded step, batch gradient descent converges without a
  learning-rate search. The loss curve in `03_implementation` is monotone.
- The bias is not penalised. Penalising it would shift the decision threshold.
- Compare `evaluation/metrics.json` with the NB run. The linear model usually
  wins on log loss and calibration, NB on training speed.

## Next module

`06_recommender_systems/01_popularity_baseline` moves from labelled documents
to user-item ratings and chronological evaluation.