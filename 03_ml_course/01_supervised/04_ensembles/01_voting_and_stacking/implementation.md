# Implementation notes

All code is in `Model.hpp` (header-only, namespace `course`). The members are the course's own `ml::LogisticRegression`, `ml::KNNClassifier` and `ml::RandomForest`; the module library `ml_ens` is an INTERFACE target that links `ml_clog`, `ml_cknn` and `ml_tree_models`.

## Class layout

    class VotingStacking {
      int mode_;                       // 0 hard, 1 soft, 2 stacking
      size_t n_classes_;               // derived from y in fit()
      ml::LogisticRegression logistic_; // multinomial, C = 1, Adam 600 epochs
      ml::KNNClassifier knn_{5};
      ml::RandomForest forest_;        // 50 trees, depth 8, min leaf 2
      ml::LogisticRegression meta_;    // only fitted in stacking mode
    };

The constructor takes `double` so `run_supervised` can pass the parameter grid `{0, 1, 2}` directly; anything that is not 0, 1 or 2 throws `std::invalid_argument`.

## fit(X, y)

1. Validate: non-empty rectangular finite `X`, matching `y`, labels are integer codes 0..K-1 with K >= 2.
2. Stacking mode only: `fit_meta` builds the out-of-fold matrix. `ml::stratified_kfold(y, 5, 42)` gives the folds; for each fold a temporary `VotingStacking` fits its three members on the fold's training rows and `base_probabilities` scores the held-out rows into the OOF matrix (n x 3K). The meta-learner is `LogisticRegression(K)` fitted on that matrix.
3. `fit_bases` fits the three members on all rows.

Stochastic members (the forest) use the course RNG; callers that need bit-exact replays seed it with `ml::seed_rng` before `fit`, as `run_supervised` does.

## predict_proba(X)

- hard: `vote_fractions` collects each member's `predict` and accumulates 1/3 per vote.
- soft: block-average the 3K stacked probabilities.
- stacking: `meta_.predict_proba(base_probabilities(X))`.

`predict` is the argmax of `predict_proba`; `std::max_element` returns the first maximum, so hard-vote ties resolve to the smallest class index.

## Serialization

`save` writes the tag `VotingStacking_V1`, `mode_`, `n_classes_`, then calls each member's `save` in a fixed order (logistic, knn, forest, meta). `load` checks the tag, restores the scalars, validates them and calls each member's `load`. An unfitted meta-learner serializes as empty weight vectors, so modes 0 and 1 round-trip too. `run_supervised` verifies the round trip on every run (`inference/reload_verification.json`).

## Artifacts

- `03_implementation.cpp`: `comparison.csv` (members and modes on one stratified 80/20 split, features standardized by `ml::Preprocessor`), `macro_f1.svg`, `meta_weights.csv` with columns `meta_class,base,base_class,weight`.
- `04_end_to_end.cpp`: the standard pipeline set; `validation/candidate_scores.csv` and `validation/summary.csv` compare the three modes.

## Tests (`tests/model_test.cpp`)

Unanimous three-blob fixture: hard voting equals the members' shared prediction; soft probabilities sum to 1; stacking fits and predicts every row; `save`/`load` reproduce probabilities exactly for all three modes; invalid mode and empty data throw.
