# Text classification with C++17

Four modules build a spam filter for the UCI SMS Spam Collection from scratch:
tokenization, sparse count/TF-IDF features, Multinomial Naive Bayes and
L2-regularised logistic regression. Everything is plain C++17 on top of the
course helpers. There are no external ML libraries.

## Modules (read in order)

| Module | Target prefix | Adds | Final project |
|---|---|---|---|
| `01_text_preprocessing` | `text_preprocessing` | `Text.hpp`: tokenizer, corpus loader, deduplication | raw counts + Multinomial NB |
| `02_count_and_tfidf` | `text_tfidf` | `Vectorizer`: capped vocabulary, counts or TF-IDF, sparse rows | TF-IDF + Multinomial NB |
| `03_multinomial_naive_bayes` | `text_nb` | `MultinomialNB`: smoothed generative classifier | raw counts + Multinomial NB (alpha tuned) |
| `04_linear_text_classifier` | `text_linear` | `LinearTextClassifier`: logistic regression on sparse rows | TF-IDF + logistic regression (lambda tuned) |

Each module has `theory.md`, `math_intuition.md`, `implementation.md`,
`exercises.md`, a `README.md`, the numbered lesson executables
`01_theory`, `02_math_intuition`, `03_implementation`, `04_end_to_end`, a
`predict` tool and `tests/model_test.cpp`.

## Shared code

- `Workflow.hpp` (this folder) - `text_course::workflow<Model>()`, the single
  end-to-end project every module's `04_end_to_end.cpp` calls. It also serves
  the inference CLI: `predict.cpp` simply includes `04_end_to_end.cpp`.
- Model contract used by the workflow: `Model()`, `Model(double parameter)`,
  `fit(const Sparse&, const std::vector<double>&, size_t vocabulary_size)`,
  `probabilities(const Sparse&) -> std::vector<double>` (p(spam)),
  `save(std::ostream&)`, `load(std::istream&)`.

## Dataset

`data/SMSSpamCollection` - 5,574 messages, `ham`/`spam` label and text
separated by a tab. `data/PROVENANCE.md` and `data/readme` carry the upstream
attribution. `data/inference_messages.txt` is a five-line sample for the
predict tools. The corpus is tiny, imbalanced (about 13% spam) and full of the
punctuation and digits that make preprocessing choices matter.

## Validation protocol

Exact duplicates are removed first. A stratified 20% holdout is reserved.
The remaining rows go through 5-fold stratified cross-validation where the
vocabulary and IDF table are refitted inside every fold. The best candidate of
the model's single hyper-parameter is refitted on all training rows and
evaluated on the holdout once. The archive is reloaded and its probabilities
compared with the in-memory model before the run is declared successful.
`--quick` keeps a seeded ~1000-message subset and a 1000-term vocabulary.

## Build, run, test

From the repository root:

```powershell
cmake --preset course
cmake --build --preset course
ctest --preset course -R "text_"
```

Run a project: `text_nb_end_to_end` (full) or `text_nb_end_to_end --quick`.

Score new messages (one per line):

```powershell
text_nb_predict --predict messages.txt --model 03_multinomial_naive_bayes/results/04_end_to_end_results/full/model
```

## Results layout

Every executable owns `results/<cpp_filename_stem>_results/` inside its module:
`03_implementation.cpp` writes to `results/03_implementation_results/`, the
project to `results/04_end_to_end_results/full/` (or `quick/`), and the
predict tool to `results/predict_results/predictions.csv`. Paths are compiled
in, so the working directory does not matter. A run contains `data/`, `eda/`,
`validation/`, `evaluation/`, `model/` (`model.txt`, `preprocessing.txt`,
`vocabulary.csv`), `inference/`, `run_manifest.json`, `report.md` and
`execution.log`. There is no `output.txt`.

## Tests

`text_<prefix>_numerical` compiles the shared fixture once per module.
`text_<prefix>_workflow` runs the `--quick` project. All of them are part of
`ctest --preset course`.