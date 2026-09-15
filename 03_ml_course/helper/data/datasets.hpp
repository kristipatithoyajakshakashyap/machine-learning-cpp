#ifndef HELPER_DATA_DATASETS_HPP
#define HELPER_DATA_DATASETS_HPP

// ===========================================================================
// helper/data/datasets.hpp  (+ the matching helper/data/datasets.cpp)
// ---------------------------------------------------------------------------
// The course's data hub.  Every lesson loads its real data through one of
// these loaders, which read the checked-in CSV files that live in this same
// helper/data/ folder via ml::read_csv (helper/dataframe/dataframe.hpp) and
// return a dense ml::Dataset (X = features, y = target) always in the same
// column order, so lesson EXPECTED OUTPUT blocks are stable.
//
// Files read: helper/data/{wine,breast_cancer,iris,penguins,titanic,
// diabetes,california_housing,tips}.csv
//
// Sources: UCI Wine, UCI Breast Cancer Wisconsin (WDBC), Bradley Efron / LARS
// diabetes data, StatLib California housing, Ronald Fisher iris, Palmer
// penguins, the real Titanic passenger list, and the seaborn tips study.
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "helper/math/matrix.hpp"

namespace ml {

// One loaded dataset.  X holds rows (observations); each inner vector is one
// observation's p features.  y[i] is the matching target: a class code for
// classification data (int-castable), a real number for regression data.
struct Dataset {
  std::vector<size_t> row_ids;        // original source row identity
  Mat X;                             // n x p rows = observations
  Vec y;                             // n (class codes int-castable, or target)
  std::vector<std::string> feature_names;
  std::string target_name;
  size_t n() const { return X.size(); }
  size_t p() const { return X.empty() ? 0 : X[0].size(); }
};

// --- classification data --------------------------------------------
Dataset load_wine(const std::string& data_dir);            // 178 x 13, 3 classes
Dataset load_breast_cancer(const std::string& data_dir);   // 569 x 30, 2 classes
Dataset load_iris(const std::string& data_dir);             // 150 x 4, 3 classes
Dataset load_penguins(const std::string& data_dir, bool preserve_missing = false); // raw 344 or legacy 342 complete cases
Dataset load_titanic(const std::string& data_dir, bool preserve_missing = false); // raw 7 features or legacy engineered features

// --- regression data ------------------------------------------------
Dataset load_diabetes(const std::string& data_dir);         // 442 x 10
Dataset load_california(const std::string& data_dir,
                        bool use_ocean = false, bool preserve_missing = false);            // 20640 x 8 (or +5 one-hots)
Dataset load_tips(const std::string& data_dir);             // 244 x 2 (total_bill, size) -> tip

// --- helpers ---------------------------------------------------------
// Keep only the rows listed in idx, preserving their order.  Drops no ids.
Dataset select_rows(const Dataset& d, const std::vector<size_t>& idx);
// Deterministic subsample of k rows (seeded shuffle of the row ids).
Dataset subsample(const Dataset& d, size_t k, std::uint32_t seed);
// Deliberate class imbalance: keep every row whose target differs from
// positive_label and a seeded random subset of keep_positives positive rows.
// Row identities (row_ids) are preserved so provenance survives subsampling.
Dataset make_imbalanced(const Dataset& d, double positive_label,
                        size_t keep_positives, unsigned seed);

}  // namespace ml

#endif  // ML_DATASETS_HPP
