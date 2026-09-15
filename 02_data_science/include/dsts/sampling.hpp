#ifndef DSTS_SAMPLING_HPP
#define DSTS_SAMPLING_HPP

// Sampling and train/test splitting helpers for module 08.
// sklearn equivalents: sklearn.model_selection.train_test_split and
// sklearn.model_selection.StratifiedShuffleSplit. Every routine is seeded,
// so the same seed always draws the same rows.

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "dsts/dataframe.hpp"

namespace dsts {

// k distinct row indices chosen uniformly from [0, n) without replacement.
std::vector<size_t> sample_indices(size_t n, size_t k, uint32_t seed);

// Randomly split rows into (train, test) carrying ~train_frac rows in
// train (at least 1 each). Column order is preserved; indices restart at 0.
std::pair<DataFrame, DataFrame> train_test_split(const DataFrame& df,
                                                 double train_frac,
                                                 uint32_t seed);

// train_test_split that balances on a string column: every group value is
// split at train_frac, so rare groups stay represented in both sets.
std::pair<DataFrame, DataFrame> stratified_split(const DataFrame& df,
                                                 const std::string& group_col,
                                                 double train_frac,
                                                 uint32_t seed);

// --- k-fold cross-validation -------------------------------------------
// Rows [0, n) are shuffled with `seed` and dealt into k folds; result[f] is
// the sorted list of row indices that form TEST fold f. Every row lands in
// exactly one fold and fold sizes differ by at most one.
// sklearn equivalent: sklearn.model_selection.KFold(shuffle=True).
std::vector<std::vector<size_t>> kfold_indices(size_t n, size_t k, uint32_t seed);

// Same contract, but each label value is dealt round-robin across folds so
// class proportions inside every fold match the full data (per-class fold
// counts differ by at most one).
// sklearn equivalent: sklearn.model_selection.StratifiedKFold(shuffle=True).
std::vector<std::vector<size_t>> stratified_kfold_indices(
    const std::vector<std::string>& labels, size_t k, uint32_t seed);

}  // namespace dsts

#endif  // DSTS_SAMPLING_HPP