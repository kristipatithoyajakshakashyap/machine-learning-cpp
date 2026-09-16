# SMS text preprocessing

Method. Turn raw messages into something a model counts. The tokenizer in
`Text.hpp` lowercases ASCII letters and digits and splits on every other byte,
so `"Win a FREE prize!"` becomes `[win, a, free, prize]`. `read_sms` loads the
corpus, drops exact-duplicate messages (keeping the first) and remembers the
source line number of every row so you trace any result back to the file.

Core idea. Every later text model sees only what the tokenizer produces.
Decisions made here shape the vocabulary, the class balance and the honesty of
the train/test split: lowercasing, no stemming, no stop-word list, duplicates
removed before splitting. The end-to-end project trains a Multinomial Naive
Bayes classifier on raw token counts to show the effect of those choices on a
real spam filter.

## Dataset

UCI SMS Spam Collection, checked in as `../data/SMSSpamCollection`
(5,574 tab-separated `label<TAB>message` rows, labels `ham`/`spam`, provenance
and licence in `../data/PROVENANCE.md`). It is small enough to process in
milliseconds, has a real class imbalance (about 13% spam) and contains the
punctuation, digits and shouting that make preprocessing decisions visible.

## Prerequisites

`std::string` and `std::vector`, basic probability, and the idea of a
stratified train/test split (see `01_supervised/02_classification`).

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Why tokenization and deduplication matter. | - |
| `math_intuition.md` | - | Bag-of-words counting worked by hand. | - |
| `implementation.md` | - | Walkthrough of `Text.hpp` and the lesson code. | - |
| `exercises.md` | - | Practice questions. | - |
| `Text.hpp` | (header) | `tokenize`, `Corpus`, `read_sms`, `select`. | - |
| `01_theory.cpp` | `text_preprocessing_theory` | Prints the lesson and encodes a toy corpus. | prints only |
| `02_math_intuition.cpp` | `text_preprocessing_math_intuition` | Bag-of-words counts on a toy example. | prints only |
| `03_implementation.cpp` | `text_preprocessing_implementation` | Tokenizes the whole corpus. Top-30 tokens and document-length histogram. | `results/03_implementation_results/` |
| `04_end_to_end.cpp` | `text_preprocessing_end_to_end` | Full project: raw-count NB spam filter with 5-fold CV over alpha in {0.5, 1.0}. | `results/04_end_to_end_results/{full,quick}/` |
| `predict.cpp` | `text_preprocessing_predict` | Reloads a saved run and scores new messages. | `results/predict_results/predictions.csv` |
| `tests/model_test.cpp` | `text_preprocessing_tests` | Numerical fixture for the shared text stack. | prints only |

## Build and run

From the repository root (MinGW toolchain, `D:/msys64/ucrt64/bin` on PATH):

```powershell
cmake --preset course
cmake --build --preset course --target text_preprocessing_implementation text_preprocessing_end_to_end text_preprocessing_predict
build/03_ml_course/05_text_classification/01_text_preprocessing/text_preprocessing_implementation.exe
build/03_ml_course/05_text_classification/01_text_preprocessing/text_preprocessing_end_to_end.exe          # full run
build/03_ml_course/05_text_classification/01_text_preprocessing/text_preprocessing_end_to_end.exe --quick  # ~1000-message smoke run
```

Output paths are compiled in (`RUN_OUTPUT_DIR`), so you start the executables
from any working directory.

## Predicting new messages

```powershell
text_preprocessing_predict --predict ../data/inference_messages.txt --model results/04_end_to_end_results/full/model
```

The text file holds one message per line. `results/predict_results/predictions.csv`
receives `row,p_spam,prediction` for every line.

## Results layout

- `results/03_implementation_results/` - `token_frequency_top30.csv/.svg`, `document_length_histogram.svg`.
- `results/04_end_to_end_results/full/` (or `quick/`):
  `data/` (quality report, row assignments), `eda/` (training-only tables and figures),
  `validation/candidate_scores.csv`, `evaluation/` (metrics.json, predictions.csv,
  ROC, precision-recall and calibration CSVs), `model/` (`model.txt`,
  `preprocessing.txt`, `vocabulary.csv`), `inference/reload_verification.json`,
  `run_manifest.json`, `report.md`, `execution.log`.
- `results/predict_results/predictions.csv`.

Re-running a target replaces its folder. There is no `output.txt`. Read `report.md`.

## Tests

`ctest --test-dir build -R text_preprocessing` runs `text_preprocessing_numerical`
(vectorizer counts, unknown-token handling, archive round trip, NB and logistic
fixtures) and `text_preprocessing_workflow` (the `--quick` project).

## Key takeaways

- Deduplicate before splitting or the same message leaks into train and test.
- The tokenizer is the first model: what it discards never gets recovered.
- Keep row ids so every prediction traces back to a source line.

## Next module

`02_count_and_tfidf` replaces raw counts with TF-IDF weights and shows why
vocabulary and IDF must be fitted inside each training fold.