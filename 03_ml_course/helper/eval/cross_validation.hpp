#ifndef HELPER_EVAL_CROSS_VALIDATION_HPP
#define HELPER_EVAL_CROSS_VALIDATION_HPP

// ===========================================================================
// helper/eval/cross_validation.hpp  (+ matching helper/eval/cross_validation.cpp)
// ---------------------------------------------------------------------------
// Train/test fold helpers used by every module's dataset + end-to-end
// lessons.  The sklearn equivalents are sklearn.model_selection.KFold and
// StratifiedKFold.  Every fold builder is deterministic: it seeds the shared
// RNG (helper/math/optim.hpp), shuffles the row indices [0, n), then cuts the
// shuffled order into contiguous blocks, so the same (dataset, seed) always
// reproduces the same folds and lesson output is stable across runs.
// ===========================================================================

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "helper/math/matrix.hpp"

namespace ml {

// One fold: the row indices (into the ORIGINAL dataset) used for training
// and testing.
struct Fold {
  std::vector<size_t> train;  // indices into the original rows
  std::vector<size_t> test;
};

// Plain k-fold: n rows split into k (as-equal-as-possible) blocks after a
// seeded shuffle; each block is one fold's test set, the rest its train set.
std::vector<Fold> kfold(size_t n, size_t k, std::uint32_t seed);

// Stratified k-fold: shuffle rows within each class first, then hand each
// fold a proportional slice of every class, so every fold keeps the global
// class balance.  y holds integer class codes in [0, n_classes).
std::vector<Fold> stratified_kfold(const Vec& y, size_t k,
                                   std::uint32_t seed);

// One deterministic split: after a seeded shuffle, the first frac*N rows are
// the train set and the remainder the test set.  frac is clamped to leave at
// least one row in each side.
Fold train_test_indices(size_t n, double frac, std::uint32_t seed);

// Serialise a fold to a side-car file ("train: i j k...\ntest: i j k...") so
// a Python verifier can feed the EXACT same row indices into sklearn and the
// two engines' outputs can be diffed digit-for-digit.
void write_split(const Fold& f, const std::string& path);

}  // namespace ml

#endif  // ML_CV_HPP