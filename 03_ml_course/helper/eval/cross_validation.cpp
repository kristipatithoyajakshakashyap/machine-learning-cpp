#include "helper/eval/cross_validation.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "helper/math/optim.hpp"

namespace ml {
// ---------------------------------------------------------------------------
// Cross-validation folds
//
// All fold builders share one idiom: seed the global RNG, materialise the
// row indices [0, n), shuffle them, then cut the shuffled order into blocks.
// This guarantees that a given (dataset, seed) always yields the same folds,
// so EXPECTED OUTPUT blocks stay stable and the python verifier can replay the
// exact same rows into sklearn.
// ---------------------------------------------------------------------------

// Plain k-fold: split the shuffled rows into k contiguous (as equal as
// possible) test blocks; every other row feeds that fold's training set.
std::vector<Fold> kfold(size_t n, size_t k, std::uint32_t seed) {
  if (k < 2 || k > n) throw std::runtime_error("kfold: invalid k");
  seed_rng(seed);
  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; ++i) idx[i] = i;
  shuffle_indices(idx);

  std::vector<Fold> folds(k);
  size_t base = n / k;      // blocks get `base` rows...
  size_t extra = n % k;     // ...and the first `extra` blocks get one extra
  size_t pos = 0;
  for (size_t f = 0; f < k; ++f) {
    size_t block = base + (f < extra ? 1 : 0);
    std::vector<size_t> test(idx.begin() + static_cast<long>(pos),
                             idx.begin() + static_cast<long>(pos + block));
    std::vector<size_t> train;
    for (size_t i = 0; i < n; ++i)
      if (std::find(test.begin(), test.end(), i) == test.end()) train.push_back(i);
    folds[f].train = train;
    folds[f].test = test;
    pos += block;
  }
  return folds;
}

// Stratified k-fold: shuffle rows *within each class* first, then hand each
// of the k folds its proportional slice of every class.  This keeps the class
// balance in each fold the same as in the full set (what StratifiedKFold does).
std::vector<Fold> stratified_kfold(const Vec& y, size_t k,
                                   std::uint32_t seed) {
  if(k < 2 || k > y.size()) throw std::invalid_argument("Invalid stratified fold count");
  // Group row indices by class label.
  std::map<double, std::vector<size_t>> by_class;
  for (size_t i = 0; i < y.size(); ++i) by_class[y[i]].push_back(i);

  seed_rng(seed);
  std::vector<Fold> folds(k);
  std::vector<std::vector<size_t>> test_sets(k);

  // Per class: cut the class's rows into k proportional blocks, appending each
  // block to the matching fold's test set.
  for (auto& kv : by_class) {
    auto& members = kv.second;
    if(members.size()<k) throw std::invalid_argument("Class has fewer rows than folds");
    shuffle_indices(members);
    size_t base = members.size() / k;
    size_t extra = members.size() % k;
    size_t pos = 0;
    for (size_t f = 0; f < k; ++f) {
      size_t block = base + (f < extra ? 1 : 0);
      for (size_t j = 0; j < block; ++j)
        test_sets[f].push_back(members[pos + j]);
      pos += block;
    }
  }

  // Complement of each fold's test set becomes its training set.
  for (size_t f = 0; f < k; ++f) {
    folds[f].test = test_sets[f];
    std::vector<bool> is_test(y.size(), false);
    for (size_t i : test_sets[f]) is_test[i] = true;
    for (size_t i = 0; i < y.size(); ++i)
      if (!is_test[i]) folds[f].train.push_back(i);
    std::sort(folds[f].train.begin(), folds[f].train.end());
  }
  return folds;
}

// Single deterministic split: shuffled rows, first frac*N go to train, the
// rest to test.  frac is clamped to keep at least one row in both sets.
Fold train_test_indices(size_t n, double frac, std::uint32_t seed) {
  if (n < 2 || !(frac > 0.0 && frac < 1.0))
    throw std::runtime_error("train_test_indices: frac out of range");
  seed_rng(seed);
  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; ++i) idx[i] = i;
  shuffle_indices(idx);
  size_t cut = static_cast<size_t>(frac * static_cast<double>(n));
  if (cut == 0) cut = 1;
  if (cut >= n) cut = n - 1;
  Fold f;
  f.train.assign(idx.begin(), idx.begin() + static_cast<long>(cut));
  f.test.assign(idx.begin() + static_cast<long>(cut), idx.end());
  return f;
}

// Serialise a fold as "train: i j k..." / "test: ..." so the side-car split
// file can be consumed verbatim by the Python sklearn verifier.
void write_split(const Fold& f, const std::string& path) {
  std::ofstream out(path);
  out << "train:";
  for (size_t i : f.train) out << " " << i;
  out << "\ntest:";
  for (size_t i : f.test) out << " " << i;
  out << "\n";
  out.flush(); if(!out) throw std::runtime_error("Could not save split: "+path);
}

}  // namespace ml