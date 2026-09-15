// 05_feature_correlation.cpp
// Lesson: after engineering, check which features actually track the target.
// Pearson correlation of every candidate feature against "Survived" is
// printed sorted by absolute strength - a first look at what matters.
// Equivalent: df.corr()["Survived"].abs().sort_values().
//
// Data: titanic.csv (real data, 891 passengers).
//
// EXPECTED OUTPUT:
// ## correlation of each feature with Survived
//   Pclass: -0.338481
//   has_cabin: 0.3169115
//   Fare: 0.2573065
//   is_child: 0.122239
//   Parch: 0.08162941
//   Age: -0.07722109
//   SibSp: -0.0353225
//   family_size: 0.01663899

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/series.hpp"
#include "dsts/stats.hpp"

int main() {
  const dsts::DataFrame t = dsts::read_csv(DATA_DIR "/titanic.csv");
  const dsts::Series survived("Survived", t.numeric("Survived"));

  std::vector<dsts::OptD> family_size, has_cabin, is_child;
  family_size.reserve(t.rows());
  has_cabin.reserve(t.rows());
  is_child.reserve(t.rows());
  for (size_t r = 0; r < t.rows(); ++r) {
    family_size.push_back(*t.numeric("SibSp")[r] + *t.numeric("Parch")[r] + 1.0);
    has_cabin.push_back(t.strings("Cabin")[r].empty() ? 0.0 : 1.0);
    const dsts::OptD& age = t.numeric("Age")[r];
    is_child.push_back(age && *age < 18.0 ? 1.0 : 0.0);
  }

  struct RankedFeature {
    std::string name;
    double r;
  };
  std::vector<RankedFeature> table;
  auto add = [&](const std::string& name, const std::vector<dsts::OptD>& v) {
    table.push_back(
        RankedFeature{name, dsts::pearson(dsts::Series(name, v), survived)});
  };
  add("Pclass", t.numeric("Pclass"));
  add("Age", t.numeric("Age"));
  add("SibSp", t.numeric("SibSp"));
  add("Parch", t.numeric("Parch"));
  add("Fare", t.numeric("Fare"));
  add("family_size", family_size);
  add("has_cabin", has_cabin);
  add("is_child", is_child);

  std::sort(table.begin(), table.end(),
            [](const RankedFeature& a, const RankedFeature& b) {
              return std::fabs(a.r) > std::fabs(b.r);
            });

  std::cout << "## correlation of each feature with Survived\n";
  for (const auto& f : table) {
    std::cout << "  " << f.name << ": " << dsts::fmt(f.r) << "\n";
  }
  return 0;
}