#include "helper/data/datasets.hpp"

#include <algorithm>
#include <map>
#include <limits>
#include <set>
#include <stdexcept>

#include "helper/dataframe/dataframe.hpp"
#include "helper/math/optim.hpp"

namespace ml {
// ---------------------------------------------------------------------------
// Dataset loaders
//
// Every loader reads one checked-in CSV from helper/data, extracts the
// numeric features in a fixed column order, and returns a Dataset struct
// (dense X: n x p rows = observations, plus the target y).  The CSV reading
// layer is ml::read_csv from helper/dataframe.  Feature extraction helpers
// below keep each loader short and the column order identical across runs so
// lesson EXPECTED OUTPUT blocks are stable.
// ---------------------------------------------------------------------------

namespace {

// Extract a fully-numeric column; throws on any missing cell, because a
// silently dropped row would shift every downstream index.
Vec numeric_of(const DataFrame& df, const std::string& col) {
  const auto& raw = df.numeric(col);
  Vec out;
  out.reserve(raw.size());
  for (const auto& v : raw) {
    if (!v.has_value())
      throw std::runtime_error("numeric_of: missing value in " + col);
    out.push_back(v.value());
  }
  return out;
}

// Same as numeric_of but missing cells are replaced with `fill` (used for
// columns where missing values are meaningful, e.g. Age in titanic).
Vec numeric_or(const DataFrame& df, const std::string& col,
               double fill) {
  const auto& raw = df.numeric(col);
  Vec out;
  out.reserve(raw.size());
  for (const auto& v : raw) out.push_back(v.value_or(fill));
  return out;
}

// Map a string column to integer codes in first-seen order (0, 1, 2, ...).
// Used to turn "species", "diagnosis", etc. into numeric targets.
Vec codes_of(const DataFrame& df, const std::string& col) {
  const auto& raw = df.strings(col);
  std::map<std::string, double> code;
  Vec out;
  out.reserve(raw.size());
  for (const auto& s : raw) {
    auto it = code.find(s);
    if (it == code.end()) {
      code.emplace(s, static_cast<double>(code.size()));
      out.push_back(static_cast<double>(code.size() - 1));
    } else {
      out.push_back(it->second);
    }
  }
  return out;
}

// Shorthand: read a CSV under the data directory.
DataFrame read(const std::string& data_dir, const std::string& file) {
  return read_csv(data_dir + "/" + file);
}

}  // namespace

// UCI Wine: 178 samples, 13 chemical measurements, classes 1..3 (remapped to
// 0..2).  Source: UCI ML Repository (wine.data).
Dataset load_wine(const std::string& data_dir) {
  DataFrame df = read(data_dir, "wine.csv");
  Dataset d;
  d.feature_names = {"alcohol", "malic_acid", "ash", "alcalinity",
                     "magnesium", "total_phenols", "flavanoids",
                     "nonflavanoid_phenols", "proanthocyanins",
                     "color_intensity", "hue", "od280", "proline"};
  for (const auto& f : d.feature_names) d.X.push_back(numeric_of(df, f));
  d.y = numeric_of(df, "class");
  for (double& v : d.y) v -= 1.0;  // classes 1,2,3 -> 0,1,2
  d.target_name = "class";
  d.X = transpose(d.X);
  return d;
}

// Breast Cancer Wisconsin (WDBC): 569 samples, 30 features (mean/se/worst of
// 10 cell measurements), diagnosis "M" (malignant, label 1) vs "B" (benign, 0).
// Source: UCI ML Repository (wdbc.data).
Dataset load_breast_cancer(const std::string& data_dir) {
  DataFrame df = read(data_dir, "breast_cancer.csv");
  Dataset d;
  d.feature_names = {
      "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
      "smoothness_mean", "compactness_mean", "concavity_mean",
      "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
      "radius_se", "texture_se", "perimeter_se", "area_se",
      "smoothness_se", "compactness_se", "concavity_se",
      "concave_points_se", "symmetry_se", "fractal_dimension_se",
      "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
      "smoothness_worst", "compactness_worst", "concavity_worst",
      "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"};
  for (const auto& f : d.feature_names) d.X.push_back(numeric_of(df, f));
  d.y = numeric_of(df, "diagnosis");  // 1 = malignant, 0 = benign
  d.target_name = "diagnosis";
  d.X = transpose(d.X);
  return d;
}

// Ronald Fisher iris: 150 samples, 4 flower measurements, 3 species
// (setosa/versicolor/virginica -> 0/1/2 by first-seen order).
Dataset load_iris(const std::string& data_dir) {
  DataFrame df = read(data_dir, "iris.csv");
  Dataset d;
  d.feature_names = {"sepal_length", "sepal_width", "petal_length",
                     "petal_width"};
  for (const auto& f : d.feature_names) d.X.push_back(numeric_of(df, f));
  d.y = codes_of(df, "species");
  d.target_name = "species";
  d.X = transpose(d.X);
  return d;
}

// Palmer penguins: 344 observations, 4 measurements; the 2 rows with a
// missing measurement are dropped (keeps X dense and sklearn-comparable) so
// the loader returns 342 rows.
Dataset load_penguins(const std::string& data_dir, bool preserve_missing) {
  DataFrame df = read(data_dir, "penguins.csv");
  Dataset d;
  d.feature_names = {"bill_length_mm", "bill_depth_mm", "flipper_length_mm",
                     "body_mass_g"};
  const auto& raw0 = df.numeric("bill_length_mm");
  std::vector<size_t> keep;
  std::set<std::string> required = {"bill_length_mm", "bill_depth_mm",
                                    "flipper_length_mm", "body_mass_g"};
  // Row survives only if all four measurements are present.
  for (size_t i = 0; i < raw0.size(); ++i) {
    bool ok = true;
    for (const auto& f : d.feature_names) {
      const auto& col = df.numeric(f);
      if (!col[i].has_value()) {
        ok = false;
        break;
      }
    }
    if (ok || preserve_missing) keep.push_back(i);
  }
  for (const auto& f : d.feature_names) {
    const auto& col = df.numeric(f);
    Vec v;
    v.reserve(keep.size());
    for (size_t i : keep) v.push_back(col[i].value_or(std::numeric_limits<double>::quiet_NaN()));
    d.X.push_back(v);
  }
  d.row_ids = keep;
  Vec sp = codes_of(df, "species");
  d.y.reserve(keep.size());
  for (size_t i : keep) d.y.push_back(sp[i]);
  d.target_name = "species";
  d.X = transpose(d.X);
  return d;
}

// Titanic: 891 passengers predicting survival (Survived, 0/1).  Seven
// engineered features: sex (male=1), pclass, median-filled age, fare,
// family size (sibsp+parch+1), child flag (age<10), cabin present (1/0).
Dataset load_titanic(const std::string& data_dir, bool preserve_missing) {
  DataFrame df = read(data_dir, "titanic.csv");
  Dataset d;
  d.feature_names = {"sex_male", "pclass", "age", "fare", "family_size",
                     "is_child", "has_cabin"};
  std::vector<double> sex_male;
  const auto& sex = df.strings("Sex");
  const auto& pclass = df.numeric("Pclass");
  const auto& age = df.numeric("Age");
  const auto& fare = df.numeric("Fare");
  const auto& sibsp = df.numeric("SibSp");
  const auto& parch = df.numeric("Parch");
  const auto& cabin = df.strings("Cabin");

  if (preserve_missing) {
    // No learned quantities in this path: split first, then fit imputation
    // independently in every training fold. Keep the raw family counts;
    // an age-derived child indicator cannot be computed before age imputation.
    d.feature_names = {"sex_male", "pclass", "age", "fare", "sibsp", "parch", "has_cabin"};
    const double missing = std::numeric_limits<double>::quiet_NaN();
    for (size_t i=0;i<df.rows();++i) {
      if (sex[i]!="male" && sex[i]!="female") throw std::runtime_error("Unknown Titanic sex category");
      d.X.push_back({sex[i]=="male"?1.0:0.0, pclass[i].value_or(missing),
        age[i].value_or(missing),fare[i].value_or(missing),sibsp[i].value_or(missing),
        parch[i].value_or(missing),(cabin[i].empty() || cabin[i]=="NA")?0.0:1.0});
      d.row_ids.push_back(i);
    }
    d.y=numeric_of(df,"Survived");d.target_name="survived";
    return d;
  }

  double age_median = 28.0;  // dataset-wide median of the 177 known ages
  for (size_t i = 0; i < age.size(); ++i) {
    sex_male.push_back(sex[i] == "male" ? 1.0 : 0.0);
  }
  Vec age_fill = numeric_or(df, "Age", age_median);
  Vec pclass_v, fare_v, fam_v, is_child_v, has_cabin_v;
  pclass_v.reserve(pclass.size());
  fare_v.reserve(fare.size());
  fam_v.reserve(sibsp.size());
  is_child_v.reserve(age.size());
  has_cabin_v.reserve(cabin.size());
  for (size_t i = 0; i < sibsp.size(); ++i) {
    pclass_v.push_back(pclass[i].value_or(3.0));
    fare_v.push_back(fare[i].value_or(0.0));
    fam_v.push_back(sibsp[i].value_or(0.0) + parch[i].value_or(0.0) + 1.0);
    is_child_v.push_back(age_fill[i] < 10.0 ? 1.0 : 0.0);
    has_cabin_v.push_back(cabin[i].empty() ? 0.0 : 1.0);
  }
  d.X.push_back(sex_male);
  d.X.push_back(pclass_v);
  d.X.push_back(age_fill);
  d.X.push_back(fare_v);
  d.X.push_back(fam_v);
  d.X.push_back(is_child_v);
  d.X.push_back(has_cabin_v);
  d.y = numeric_of(df, "Survived");
  d.target_name = "survived";
  d.X = transpose(d.X);
  return d;
}

// Bradley Efron / Lars diabetes progression study (Stanford): 442 patients,
// 10 baseline variables, continuous one-year progression score.
Dataset load_diabetes(const std::string& data_dir) {
  DataFrame df = read(data_dir, "diabetes.csv");
  Dataset d;
  d.feature_names = {"age", "sex", "bmi", "bp", "s1", "s2",
                     "s3", "s4", "s5", "s6"};
  for (const auto& f : d.feature_names) d.X.push_back(numeric_of(df, f));
  d.y = numeric_of(df, "target");
  d.target_name = "target";
  d.X = transpose(d.X);
  return d;
}

// StatLib California housing (ageron copy): 20640 census-block groups
// predicting median house value in USD.  use_ocean adds one-hot columns for
// the ocean_proximity string column (included by default in csv).
Dataset load_california(const std::string& data_dir, bool use_ocean, bool preserve_missing) {
  DataFrame df = read(data_dir, "california_housing.csv");
  Dataset d;
  d.feature_names = {"longitude", "latitude", "housing_median_age",
                     "total_rooms", "total_bedrooms", "population",
                     "households", "median_income"};
  for (const auto& f : d.feature_names) d.X.push_back(numeric_or(df, f, preserve_missing ? std::numeric_limits<double>::quiet_NaN() : 0.0));
  if (use_ocean) {
    // One-hot encode each distinct ocean_proximity value as its own column.
    const auto& ocean = df.strings("ocean_proximity");
    std::set<std::string> cats;
    for (const auto& s : ocean) cats.insert(s);
    for (const auto& cat : cats) {
      d.feature_names.push_back("ocean_" + cat);
      Vec v;
      v.reserve(ocean.size());
      for (const auto& s : ocean) v.push_back(s == cat ? 1.0 : 0.0);
      d.X.push_back(v);
    }
  }
  d.y = numeric_of(df, "median_house_value");
  d.target_name = "median_house_value";
  d.X = transpose(d.X);
  return d;
}

// Seaborn tips study: 244 restaurant bills; predicts tip from total_bill and
// table size (the two numeric columns; size is included but many lessons use
// only column 0).
Dataset load_tips(const std::string& data_dir) {
  DataFrame df = read(data_dir, "tips.csv");
  Dataset d;
  d.feature_names = {"total_bill", "size"};
  d.X.push_back(numeric_of(df, "total_bill"));
  d.X.push_back(numeric_of(df, "size"));
  d.y = numeric_of(df, "tip");
  d.target_name = "tip";
  d.X = transpose(d.X);
  return d;
}

// ---------------------------------------------------------------------------
// Reworking / subsampling helpers
// ---------------------------------------------------------------------------

// Return a new Dataset holding exactly the rows in `idx` (in their order).
Dataset select_rows(const Dataset& d, const std::vector<size_t>& idx) {
  Dataset out;
  out.feature_names = d.feature_names;
  out.target_name = d.target_name;
  out.X.reserve(idx.size());
  out.y.reserve(idx.size());
  for (size_t i : idx) {
    if (i >= d.n()) throw std::out_of_range("Dataset row index");
    out.row_ids.push_back(d.row_ids.empty() ? i : d.row_ids.at(i));
    out.X.push_back(d.X[i]);
    out.y.push_back(d.y[i]);
  }
  return out;
}

// Deterministic subsample: seeded shuffle of [0, n), keep the first k rows.
// Lets a lesson zoom in on a cheap subset while remaining reproducible.
Dataset subsample(const Dataset& d, size_t k, std::uint32_t seed) {
  seed_rng(seed);
  std::vector<size_t> idx(d.n());
  for (size_t i = 0; i < d.n(); ++i) idx[i] = i;
  shuffle_indices(idx);
  idx.resize(std::min(k, d.n()));
  return select_rows(d, idx);
}

// Deliberate imbalance: all negatives plus a seeded subset of positives.
Dataset make_imbalanced(const Dataset& d, double positive_label,
                        size_t keep_positives, unsigned seed) {
  if (d.n() == 0 || d.y.size() != d.n())
    throw std::invalid_argument("make_imbalanced needs a labeled dataset");
  std::vector<size_t> negatives, positives;
  for (size_t i = 0; i < d.n(); ++i)
    (d.y[i] == positive_label ? positives : negatives).push_back(i);
  if (positives.empty() || keep_positives == 0)
    throw std::invalid_argument("make_imbalanced needs positive rows to keep");
  seed_rng(seed);
  shuffle_indices(positives);
  positives.resize(std::min(keep_positives, positives.size()));
  std::vector<size_t> keep = negatives;
  keep.insert(keep.end(), positives.begin(), positives.end());
  std::sort(keep.begin(), keep.end());
  return select_rows(d, keep);
}

}  // namespace ml
