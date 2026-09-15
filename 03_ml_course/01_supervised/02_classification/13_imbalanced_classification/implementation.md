# Implementation

`Model.hpp` is header-only; `Model.cpp` exists only so the module library `ml_cimb` has a translation unit, matching the other modules.

## Class layout

```cpp
class WeightedLogistic {
  explicit WeightedLogistic(double positive_weight = 1.0, double lr = 0.1,
                            int epochs = 2000, double l2 = 1e-3);
  void fit(const ml::Mat &X, const ml::Vec &y);
  ml::Mat predict_proba(const ml::Mat &X) const;   // n x 2
  ml::Vec predict(const ml::Mat &X) const;         // threshold 0.5
  void save(std::ostream &) const;  void load(std::istream &);
  // accessors: positive_weight(), learning_rate(), epochs(), l2(), bias(), weights()
};
```

The four constructor arguments are the only hyperparameters. `run_supervised` passes a single `double` to the factory, so the workflow varies `positive_weight` and leaves the others at their defaults.

## fit

1. `validate` rejects empty data, ragged rows, non-finite features, labels other than 0/1 and nonsensical hyperparameters, throwing `std::invalid_argument`.
2. `weights_` is zeroed to length `p` and `bias_` to 0.
3. For each epoch the code accumulates the weighted gradient in one pass over the rows: `err = w_i * (sigmoid(logit) - y_i)`, `grad[j] += err * x_ij`, `grad_b += err`, `total_weight += w_i`.
4. The update divides by `total_weight` (the `1/W` in the loss) and adds the L2 term `l2 * weights_[j]` to the feature gradients. The bias is not regularised.

Full-batch descent with 2000 epochs at learning rate 0.1 converges to a few decimals on standardised data of this size; it is deliberately simple rather than fast. The pipeline standardises inside every fold, so the class never sees raw feature scales.

## predict_proba and predict

`predict_proba` clips the logit to `[-30, 30]`, applies the sigmoid and returns `{1 - p, p}` per row so the pipeline's binary branch (two columns, second is the positive probability) turns on ROC, precision-recall and calibration outputs. `predict` thresholds column 1 at 0.5. Both throw if the model has not been fitted or the feature count differs from training.

## Persistence

`save` writes the tag `"WeightedLogistic_V1"` followed by the hyperparameters, bias and weight vector through `ml::archive::write`. `load` reads the tag first and throws `std::runtime_error` on mismatch, then reads the same fields in the same order and rejects a non-finite bias. `run_supervised` wraps this with `archive::save_file` / `load_file`, which add the `MLCPP_ARCHIVE_V1` header line.

## Lessons

- `01_theory.cpp`: prints the accuracy-paradox arithmetic for 30 positives in 387 rows.
- `02_math_intuition.cpp`: evaluates the weighted gradient on the two-row hand example for `w = 1` and `w = 8`, then fits a separable 1-D fixture.
- `03_implementation.cpp`: fixed stratified 75/25 split of the imbalanced set, standardisation fitted on the training part, weights 1, 2, 4, 8; prints accuracy/recall/precision/macro-F1 and writes `weight_sweep.csv` plus `recall_vs_weight.svg`.
- `04_end_to_end.cpp`: the full `run_supervised` workflow with grid `{1, 2, 4, 8}`; `predict.cpp` includes it to provide the `--predict` executable.

## Tests

`tests/model_test.cpp` (CTest name `cimb_numerical`) checks: perfect fit on a separable 2-D fixture; probabilities sum to one; recall with `w = 8` is at least recall with `w = 1` on an overlapping imbalanced fixture; save/load round trip reproduces probabilities to 1e-12; fitting empty data throws `std::invalid_argument`.
