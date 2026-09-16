# Module 11: Connected Titanic pipeline (capstone)

## Purpose
A five-stage, fully reproducible pipeline on the Titanic passenger table:
split, clean, engineer features, explore, reduce with PCA and evaluate a
transparent threshold classifier. Every stage executable calls
`capstone::run(stage)` from the shared `pipeline.hpp`, so all stages use the
same raw split (seed 42), the same training-fitted preprocessing and the same
feature schema. Higher stages write a superset of the lower stages' outputs.

## Prerequisites
Modules 01 through 10 and 12_advanced_eda. Also read `theory.md`,
`math_intuition.md`, `implementation.md` and `exercises.md` in this folder.

## Lessons

| File | Target | What it teaches | What it writes (under `results/<stem>_results/`) |
|------|--------|-----------------|--------------------------------------------------|
| `01_data_clean.cpp` | `s11_01_clean` | split raw passengers before learning medians, impute Age and Fare from training medians | `train_features.csv`, `test_features.csv`, `row_assignments.csv`, `report.md` |
| `02_feature_table.cpp` | `s11_02_feat` | family size, child indicator, cabin availability and sex encoding applied identically to both partitions | same files as stage 1 (stage 2 report) |
| `03_eda_summary.cpp` | `s11_03_eda` | summarise and plot training data before interpreting test performance | adds `age.svg`, `fare_by_sex.svg` |
| `04_split_pca.cpp` | `s11_04_pca` | fit scaling and PCA on training, transform hold-out with stored means and loadings | adds `pca_scores.csv` |
| `05_eval_model.cpp` | `s11_05_eval` | PC1 class-mean threshold classifier, majority baseline, confusion counts, save/reload check | adds `predictions.csv`, `metrics.json`, `model_state.txt`, `reload_verification.json` |
| `predict.cpp` | `s11_predict` | fresh-process inference: reload `model_state.txt`, predict a CSV and verify against expected predictions (argv: model, input CSV, expected CSV) | `predictions.csv`, `reload_verification.json` under `results/predict_results/` |

## Build and run

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s11_05_eval
.\build\02_data_science\11_final_pipeline\s11_05_eval.exe
```

Run the final target for the full workflow. Each target reads `titanic.csv`
from `02_data_science/data/` (`DATA_DIR`) and writes beneath this module's
`results/` directory (`RUN_OUTPUT_DIR`) regardless of working directory.
`s11_05_eval` and `s11_predict` are registered as the ctests `dsts_capstone`
and `dsts_capstone_reload`.

## dsts functions used
`read_csv`, `write_csv`, `stratified_split`, `pca`, `pca_transform`, `summary`,
`write_svg_histogram`, `write_svg_box`, `trim`, `Series::quantile`, plus
`DataFrame::numeric/strings/series/rows`.

## Key takeaways
- Split first. Learn medians, scaling and PCA axes on training rows only.
- Engineer features with one shared function so train and test never drift.
- Compare against the majority baseline before celebrating an accuracy number.
- Save the fitted state and prove that a reloaded model reproduces predictions.

This transparent teaching classifier is not logistic regression. The supervised
ML track supplies tuned classifiers and more extensive evaluation. No test-set
tuning occurs here. Family members are not kept together by the educational
stratified split. Grouped validation is appropriate before deployment claims.

## Next module
You have completed the data-science track. Continue with 03_ml_course.