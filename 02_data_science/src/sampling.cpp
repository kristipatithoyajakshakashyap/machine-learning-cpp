// 02_data_science/src/sampling.cpp
// dsts random sampling and splitting: seeded index samples, plain and
// stratified train/test splits, and plain and stratified k-fold indices.
// All functions are deterministic for a given seed (std::mt19937).
//
// Reads/writes: nothing on disk. Build: part of the dsts static library
// (target dsts); used by 08_train_test and later modules. Exercised by
// ctest -R dsts_regression.

#include "dsts/sampling.hpp"

#include <algorithm>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

namespace dsts {

// k distinct indices from 0..n-1, returned sorted. k > n is clamped to n.
std::vector<size_t> sample_indices(size_t n, size_t k, uint32_t seed) {
  if (k > n) k = n;
  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; ++i) idx[i] = i;
  // Partial Fisher-Yates: shuffle only the first k positions.
  std::mt19937 rng(seed);
  for (size_t i = 0; i < k; ++i) {
    std::uniform_int_distribution<size_t> d(i, n - 1);
    std::swap(idx[i], idx[d(rng)]);
  }
  idx.resize(k);
  std::sort(idx.begin(), idx.end());
  return idx;
}

// Random split: round(train_frac * rows) training rows (at least 1, at most
// rows-1), the rest test. Both halves keep the original row order.
std::pair<DataFrame, DataFrame> train_test_split(const DataFrame& df,
                                                 double train_frac,
                                                 uint32_t seed) {
  if (train_frac <= 0.0 || train_frac >= 1.0) {
    throw std::invalid_argument("train_frac must be in (0, 1)");
  }
  if (df.rows() < 2) {
    throw std::invalid_argument("train_test_split needs at least 2 rows");
  }
  size_t k = static_cast<size_t>(train_frac * static_cast<double>(df.rows()));
  k = std::clamp(k, size_t{1}, df.rows() - 1);
  const std::vector<size_t> train_idx = sample_indices(df.rows(), k, seed);
  std::vector<bool> in_train(df.rows(), false);
  for (size_t i : train_idx) in_train[i] = true;
  std::vector<size_t> test_idx;
  for (size_t i = 0; i < df.rows(); ++i) {
    if (!in_train[i]) test_idx.push_back(i);
  }
  return {df.select_rows(train_idx), df.select_rows(test_idx)};
}

// Split each group_col value separately so class proportions are kept.
// Single-row groups go to train. Each group uses seed, seed+1, ... so the
// result is deterministic yet groups are not sampled identically.
std::pair<DataFrame, DataFrame> stratified_split(const DataFrame& df,
                                                 const std::string& group_col,
                                                 double train_frac,
                                                 uint32_t seed) {
  // Group row indices by the cell text (works for numeric and string data).
  std::map<std::string, std::vector<size_t>> groups;
  for (size_t r = 0; r < df.rows(); ++r) {
    groups[df.cell(r, group_col)].push_back(r);
  }
  std::vector<size_t> train_idx, test_idx;
  uint32_t s = seed;
  for (const auto& kv : groups) {
    const size_t m = kv.second.size();
    if (m < 2) {
      // A single-row group cannot be split: give it to train.
      train_idx.push_back(kv.second.front());
      continue;
    }
    const size_t k = std::clamp(
        static_cast<size_t>(train_frac * static_cast<double>(m)), size_t{1},
        m - 1);
    const std::vector<size_t> sel = sample_indices(m, k, s);
    std::vector<bool> in_train(m, false);
    for (size_t i : sel) in_train[i] = true;
    for (size_t i = 0; i < m; ++i) {
      (in_train[i] ? train_idx : test_idx).push_back(kv.second[i]);
    }
    ++s;  // different seed per group, deterministic cycle
  }
  std::sort(train_idx.begin(), train_idx.end());
  std::sort(test_idx.begin(), test_idx.end());
  return {df.select_rows(train_idx), df.select_rows(test_idx)};
}

namespace {

// Deal the (already shuffled) indices round-robin into k folds.
void deal_round_robin(const std::vector<size_t>& order,
                      std::vector<std::vector<size_t>>& folds) {
  for (size_t i = 0; i < order.size(); ++i) {
    folds[i % folds.size()].push_back(order[i]);
  }
}

// Validate 2 <= k <= n for k-fold helpers.
void require_k(size_t n, size_t k) {
  if (k < 2 || k > n) {
    throw std::invalid_argument("k-fold needs 2 <= k <= n");
  }
}

}  // namespace

// Shuffle 0..n-1 and deal into k folds of near-equal size; each fold sorted.
std::vector<std::vector<size_t>> kfold_indices(size_t n, size_t k, uint32_t seed) {
  require_k(n, k);
  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; ++i) order[i] = i;
  std::mt19937 rng(seed);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<std::vector<size_t>> folds(k);
  deal_round_robin(order, folds);
  for (auto& f : folds) std::sort(f.begin(), f.end());
  return folds;
}

// Like kfold_indices but each label group is shuffled and dealt separately,
// rotating the starting fold so every fold gets a fair share of small classes.
std::vector<std::vector<size_t>> stratified_kfold_indices(
    const std::vector<std::string>& labels, size_t k, uint32_t seed) {
  require_k(labels.size(), k);
  std::map<std::string, std::vector<size_t>> groups;
  for (size_t r = 0; r < labels.size(); ++r) groups[labels[r]].push_back(r);
  std::vector<std::vector<size_t>> folds(k);
  std::mt19937 rng(seed);
  size_t offset = 0;  // rotate the starting fold so small classes spread out
  for (auto& kv : groups) {
    std::vector<size_t> order = kv.second;
    std::shuffle(order.begin(), order.end(), rng);
    for (size_t i = 0; i < order.size(); ++i) {
      folds[(i + offset) % k].push_back(order[i]);
    }
    offset = (offset + order.size()) % k;
  }
  for (auto& f : folds) std::sort(f.begin(), f.end());
  return folds;
}

}  // namespace dsts