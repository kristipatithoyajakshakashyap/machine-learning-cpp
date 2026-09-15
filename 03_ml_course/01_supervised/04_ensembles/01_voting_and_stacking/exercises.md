# Exercises

1. Run `ens_implementation` and compare `comparison.csv` with `validation/summary.csv` from `ens_end_to_end`. Explain why a single-split ranking of the three modes can differ from the five-fold ranking.
2. In `Model.hpp`, change `fit_meta` to score the members on the rows they were trained on (skip the folds). Rerun the tests and `ens_implementation`. Report how `meta_weights.csv` changes and why the holdout macro-F1 for stacking moves in the direction it does.
3. Replace the kNN member with a second random forest using a different seed. Measure the soft-voting score on the fixed split. Relate the result to the error-correlation formula in math_intuition.md.
4. Add member weights `{0.25, 0.25, 0.5}` to soft voting (weighted average instead of the mean). Is this still a special case of stacking? Where in the meta-weight vector does it sit?
5. Set `kMetaFolds` to 2 and to 10. Describe how the OOF matrix changes (how many training rows each member sees) and what that does to the meta-learner's estimate of member reliability.
6. Wine has 178 rows. Use `learning_curve` from `helper/eval/diagnostics.hpp` on modes 1 and 2 and decide, from the curves, whether stacking's extra parameters are justified at this data size.
