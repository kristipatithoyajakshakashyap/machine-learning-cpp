# adaboost classifier — theory

01_theory.cpp
Lesson: AdaBoost in one page.
Theory summary:
  - AdaBoost trains weak trees SEQUENTIALLY on the same rows, but each
    round re-weights the sample: rows the previous trees got wrong carry
    more weight, so the next tree must focus on them.
  - Round t with weighted error e:
      alpha_t = ln((1-e)/e) + ln(n_classes-1)
    down-weights correct rows by exp(-alpha) and up-weights errors by
    exp(+alpha), then all weights are normalised back.
  - Final answer = weighted vote, weights alpha_t.  The ensemble is
    adaptive: no two rounds have the same goals.
sklearn equivalent: sklearn.ensemble.AdaBoostClassifier (SAMME).

Weighted-error worksheet, 2-class round with e = 0.2.



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
