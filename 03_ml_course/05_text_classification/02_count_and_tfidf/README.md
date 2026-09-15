# Count and TF-IDF vectorization

**Method.** `Model.hpp` implements `Vectorizer`: fit a capped vocabulary on the
training documents (sorted by document frequency, ties broken alphabetically),
then encode each document as a sparse row of either raw counts or TF-IDF
weights `count * (log((1 + N) / (1 + df)) + 1)`, L2-normalised per document.

**Core idea.** Counts reward long messages and common words; TF-IDF discounts
words that appear in many training documents so rare, discriminative terms
carry more weight. Both the vocabulary and the IDF table are *learned*, so
they must be fitted on training rows only - inside every cross-validation fold
- or validation scores become optimistic. The project trains Multinomial NB on
TF-IDF features so the result can be compared with the raw-count run of
`01_text_preprocessing`.

## Dataset

UCI SMS Spam Collection (`../data/SMSSpamCollection`, 5,574 messages). Spam
messages reuse a small set of marketing words, which makes the difference
between count and TF-IDF weighting easy to see in the per-class term tables.

## Prerequisites

`01_text_preprocessing` (tokenizer, corpus loader), logarithms and vector norms.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Counts vs TF-IDF, vocabulary capping, unknown terms. | - |
| `math_intuition.md` | - | IDF formula and L2 normalisation worked by hand. | - |
| `implementation.md` | - | Walkthrough of `Vectorizer`. | - |
| `exercises.md` | - | Practice questions. | - |
| `Model.hpp` | (header) | `Sparse` row type and `Vectorizer` (fit / transform / save / load). | - |
| `01_theory.cpp` | `text_tfidf_theory` | Lesson text plus a toy TF-IDF encoding. | prints only |
| `02_math_intuition.cpp` | `text_tfidf_math_intuition` | IDF arithmetic on a three-document example. | prints only |
| `03_implementation.cpp` | `text_tfidf_implementation` | Top TF-IDF terms per class and vocabulary size vs `min_df`. | `results/03_implementation_results/` |
| `04_end_to_end.cpp` | `text_tfidf_end_to_end` | Full project: TF-IDF + Multinomial NB, alpha in {0.5, 1.0}. | `results/04_end_to_end_results/{full,quick}/` |
| `predict.cpp` | `text_tfidf_predict` | Reloads a saved run and scores new messages. | `results/predict_results/predictions.csv` |
| `tests/model_test.cpp` | `text_tfidf_tests` | Numerical fixture for the shared text stack. | prints only |

## Build and run

From the repository root (MinGW toolchain, `D:/msys64/ucrt64/bin` on PATH):

```powershell
cmake --preset course
cmake --build --preset course --target text_tfidf_implementation text_tfidf_end_to_end text_tfidf_predict
build/03_ml_course/05_text_classification/02_count_and_tfidf/text_tfidf_implementation.exe
build/03_ml_course/05_text_classification/02_count_and_tfidf/text_tfidf_end_to_end.exe          # full run
build/03_ml_course/05_text_classification/02_count_and_tfidf/text_tfidf_end_to_end.exe --quick  # smoke run
```

## Predicting new messages

```powershell
text_tfidf_predict --predict ../data/inference_messages.txt --model results/04_end_to_end_results/full/model
```

One message per line; output `results/predict_results/predictions.csv`
(`row,p_spam,prediction`). The saved `preprocessing.txt` carries the fitted
vocabulary and IDF, so unseen words are ignored exactly as during training.

## Results layout

- `results/03_implementation_results/` - `top_tfidf_terms_per_class.csv`,
  `top_tfidf_ham.svg`, `top_tfidf_spam.svg`, `vocabulary_size_vs_min_df.csv/.svg`.
- `results/04_end_to_end_results/full/` (or `quick/`) - same layout as every text
  project: `data/`, `eda/`, `validation/`, `evaluation/`, `model/`, `inference/`,
  `run_manifest.json`, `report.md`, `execution.log`.
- `results/predict_results/predictions.csv`.

Re-running a target replaces its folder. There is no `output.txt`; read `report.md`.

## Tests

`ctest --test-dir build -R text_tfidf` runs `text_tfidf_numerical` and the
`--quick` workflow `text_tfidf_workflow`.

## Key takeaways

- Vocabulary and IDF are fitted parameters; refit them in every fold.
- L2 normalisation removes the message-length effect that raw counts carry.
- A vocabulary cap (3000 terms here, 1000 in quick mode) bounds memory and
  removes hapax noise without hurting spam recall much.

## Next module

`03_multinomial_naive_bayes` studies the generative classifier that both
projects so far have used as a black box.
