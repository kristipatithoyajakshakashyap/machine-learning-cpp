# Exercises

Work in a copy of the lesson files. Each exercise names the artifact that shows whether your change worked.

## 1. Threshold moving without retraining

`evaluation/predictions.csv` contains `probability_1` for every hold-out row. Write a small program (or extend `03_implementation.cpp`) that sweeps thresholds 0.05, 0.10, ..., 0.95 on the `w = 1` model and prints recall and precision at each. Compare the row for threshold 0.2 with the `w = 4` row of `weight_sweep.csv`. Which knob gives better precision at equal recall on this split? Explain why they differ even though both move the boundary.

## 2. Sensitivity to the surviving positives

`make_imbalanced(..., 30, 42)` keeps one particular set of 30 malignant rows. Rerun `03_implementation.cpp` with seeds 1, 2, 3 and 4 and tabulate recall at `w = 1` and `w = 8`. How wide is the spread? Relate it to the fact that the hold-out contains only about 8 positives: one flipped decision moves recall by 12.5 points.

## 3. Class-weight equals oversampling

Build a training matrix in which every positive row is duplicated `w` times and fit `WeightedLogistic(1.0)` on it. Compare `weights()` and `bias()` with `WeightedLogistic(w)` fitted on the original rows (same `lr`, `epochs`, `l2`). They should agree closely but not exactly; identify the term in the update rule responsible for the difference.

## 4. Macro F1 versus accuracy for model selection

Change the selection metric in a local copy of `run_supervised` from `macro_f1` to `accuracy` and rerun `cimb_end_to_end --quick`. Read `validation/selected_parameters.json` and `evaluation/per_class.csv` under `quick/` for both runs. Which `w` is chosen in each case, and what happens to class-1 recall when accuracy drives selection?

## 5. Calibration after reweighting

`evaluation/calibration.csv` bins predicted probabilities and reports the observed positive fraction per bin. For `w = 8` the model is trained as if positives were eight times more common. Predict the direction of the miscalibration (over- or under-confident about positives) before looking, then verify with the file. Implement a correction: divide the odds `p / (1 - p)` by `w` and convert back to a probability. Recompute `binary_logloss` (helper/math/metrics.hpp) before and after the correction on the hold-out rows.
