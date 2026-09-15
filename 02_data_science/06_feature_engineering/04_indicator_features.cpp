// 04_indicator_features.cpp
// Lesson: presence and group indicators turn text or counts into model-ready
// numbers: has_cabin, is_child and family_size are built from the real
// titanic file, then compared against actual survival rates.
// Equivalent: df["has_cabin"] = df.Cabin.notna(); df.family_size = SibSp+Parch+1.
//
// Data: titanic.csv (real data, 891 passengers).
//
// EXPECTED OUTPUT:
// ## engineered columns
//    has_cabin  is_child  family_size  Survived
// 0         no         0            2         0
// 1        yes         0            2         1
// 2         no         0            1         1
// 3        yes         0            2         1
// 4         no         0            1         0
// 5         no         0            1         0
//
// ## survival rate by cabin presence
//   cabin=no  survival 0.2998544  (n 687)
//   cabin=yes  survival 0.6666667  (n 204)
// ## survival rate by family size
//   family_size=1  survival 0.3035382  (n 537)
//   family_size=11  survival 0  (n 7)
//   family_size=2  survival 0.552795  (n 161)
//   family_size=3  survival 0.5784314  (n 102)
//   family_size=4  survival 0.7241379  (n 29)
//   family_size=5  survival 0.2  (n 15)
//   family_size=6  survival 0.1363636  (n 22)
//   family_size=7  survival 0.3333333  (n 12)
//   family_size=8  survival 0  (n 6)

#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/eda.hpp"
#include "dsts/series.hpp"

int main() {
  const dsts::DataFrame t = dsts::read_csv(DATA_DIR "/titanic.csv");

  std::vector<std::string> has_cabin_txt, family_size_txt;
  std::vector<dsts::OptD> is_child;
  has_cabin_txt.reserve(t.rows());
  family_size_txt.reserve(t.rows());
  is_child.reserve(t.rows());
  for (size_t r = 0; r < t.rows(); ++r) {
    has_cabin_txt.push_back(t.strings("Cabin")[r].empty() ? "no" : "yes");
    const dsts::OptD& age = t.numeric("Age")[r];
    is_child.push_back(age && *age < 18.0 ? 1.0 : 0.0);
    family_size_txt.push_back(
        dsts::fmt(*t.numeric("SibSp")[r] + *t.numeric("Parch")[r] + 1.0));
  }

  dsts::DataFrame eng;
  eng.add_string("has_cabin", has_cabin_txt);
  eng.add_numeric("is_child", is_child);
  eng.add_string("family_size", family_size_txt);
  eng.add_numeric("Survived", t.numeric("Survived"));

  std::cout << "## engineered columns\n" << eng.head(6) << "\n";

  std::cout << "## survival rate by cabin presence\n";
  for (const auto& g : dsts::group_mean(eng, "has_cabin", "Survived")) {
    std::cout << "  cabin=" << g.group << "  survival " << dsts::fmt(g.mean)
              << "  (n " << g.count << ")\n";
  }

  std::cout << "## survival rate by family size\n";
  for (const auto& g : dsts::group_mean(eng, "family_size", "Survived")) {
    std::cout << "  family_size=" << g.group << "  survival "
              << dsts::fmt(g.mean) << "  (n " << g.count << ")\n";
  }
  return 0;
}