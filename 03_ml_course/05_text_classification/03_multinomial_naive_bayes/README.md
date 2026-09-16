# Multinomial Naive Bayes

Method. A generative text classifier. For each class `c` estimate a token
distribution `P(t|c) = (count(t,c) + alpha) / (tokens(c) + alpha * V)` and a
prior `P(c)`. Score a document by `log P(c) + sum_t count(t) * log P(t|c)` and
normalise the two log scores with a stable log-sum-exp to get `p(spam)`.

Core idea. Naive Bayes assumes tokens are conditionally independent given
the class. The assumption is false, yet the model is a strong, fast baseline
because spam and ham use visibly different vocabularies. Laplace smoothing
(`alpha`) keeps a single unseen word from zeroing out a whole class.

## Dataset

UCI SMS Spam Collection (`../data/SMSSpamCollection`). Short messages with a
few strongly class-specific tokens (`free`, `txt`, `claim`) are the textbook
case for a bag-of-words generative model.

## Prerequisites

`01_text_preprocessing`, `02_count_and_tfidf`, Bayes rule and logarithms.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Generative story, independence assumption, smoothing. | - |
| `math_intuition.md` | - | Smoothed token probabilities and log-score normalisation by hand. | - |
| `implementation.md` | - | Walkthrough of `MultinomialNB`. | - |
| `exercises.md` | - | Practice questions. | - |
| `Model.hpp` | (header) | `MultinomialNB` (fit / probabilities / save / load). | - |
| `01_theory.cpp` | `text_nb_theory` | Lesson text plus a toy encoding. | prints only |
| `02_math_intuition.cpp` | `text_nb_math_intuition` | Hand-computed smoothed probabilities. | prints only |
| `03_implementation.cpp` | `text_nb_implementation` | One 80/20 split: most discriminative words and the confusion matrix. | `results/03_implementation_results/` |
| `04_end_to_end.cpp` | `text_nb_end_to_end` | Full project: raw counts + NB, alpha in {0.5, 1.0}. | `results/04_end_to_end_results/{full,quick}/` |
| `predict.cpp` | `text_nb_predict` | Reloads a saved run and scores new messages. | `results/predict_results/predictions.csv` |
| `tests/model_test.cpp` | `text_nb_tests` | Numerical fixture for the shared text stack. | prints only |

## Build and run

From the repository root (MinGW toolchain, `D:/msys64/ucrt64/bin` on PATH):

```powershell
cmake --preset course
cmake --build --preset course --target text_nb_implementation text_nb_end_to_end text_nb_predict
build/03_ml_course/05_text_classification/03_multinomial_naive_bayes/text_nb_implementation.exe
build/03_ml_course/05_text_classification/03_multinomial_naive_bayes/text_nb_end_to_end.exe          # full run
build/03_ml_course/05_text_classification/03_multinomial_naive_bayes/text_nb_end_to_end.exe --quick  # smoke run
```

## Predicting new messages

```powershell
text_nb_predict --predict ../data/inference_messages.txt --model results/04_end_to_end_results/full/model
```

One message per line. Output `results/predict_results/predictions.csv`
(`row,p_spam,prediction`, threshold 0.5).

## Results layout

- `results/03_implementation_results/` - `discriminative_words.csv/.svg`, `confusion_matrix.csv`.
- `results/04_end_to_end_results/full/` (or `quick/`) - `data/`, `eda/`,
  `validation/candidate_scores.csv`, `evaluation/` (metrics, predictions, ROC,
  PR, calibration), `model/` (`model.txt`, `preprocessing.txt`, `vocabulary.csv`),
  `inference/reload_verification.json`, `run_manifest.json`, `report.md`, `execution.log`.
- `results/predict_results/predictions.csv`.

Re-running a target replaces its folder. There is no `output.txt`. Read `report.md`.

## Tests

`ctest --test-dir build -R text_nb` runs `text_nb_numerical` (the NB fixture
must give `p(spam) < 0.5` to the ham document and `> 0.5` to the spam one, plus
vectorizer and archive checks) and `text_nb_workflow`.

## Key takeaways

- Work in log space. Products of thousands of probabilities underflow.
- `alpha` trades bias for robustness to unseen words. Tune it by CV.
- NB probabilities are well ranked but poorly calibrated. Look at
  `evaluation/calibration.csv` before trusting them as risks.

## Next module

`04_linear_text_classifier` fits a discriminative logistic regression on the
same sparse features and compares the two approaches.