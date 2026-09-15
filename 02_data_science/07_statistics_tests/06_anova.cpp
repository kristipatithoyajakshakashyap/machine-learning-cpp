// 06_anova.cpp
// Lesson: one-way ANOVA asks whether k group means differ more than the
// noise inside the groups would explain. Kruskal-Wallis asks the same
// question on ranks, so it does not care about normality or outliers.
// Equivalent: scipy.stats.f_oneway / scipy.stats.kruskal.
//
// Data: tips.csv (real data, 244 restaurant tips), tip grouped by day.
//
// EXPECTED OUTPUT:
// ## tip by day
//   Thur: n 62   mean 2.771452  sd 1.240223
//   Fri : n 19   mean 2.734737  sd 1.019577
//   Sat : n 87   mean 2.993103  sd 1.631014
//   Sun : n 76   mean 3.255132  sd 1.23488
//
// one-way ANOVA:   F=1.672355  df=(3, 240)  p=0.1735886
// Kruskal-Wallis:  H=8.565588  df=3  p=0.03566056
// interpretation lines

#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/series.hpp"
#include "dsts/stats.hpp"

// Raw (non-missing) values of `col` for rows whose `by` column equals `level`.
std::vector<double> group_values(const dsts::DataFrame& df, const std::string& by,
                                 const std::string& level, const std::string& col) {
  std::vector<double> out;
  const auto& keys = df.strings(by);
  const auto& vals = df.numeric(col);
  for (size_t r = 0; r < df.rows(); ++r) {
    if (keys[r] == level && vals[r].has_value()) out.push_back(*vals[r]);
  }
  return out;
}

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const std::vector<std::string> days{"Thur", "Fri", "Sat", "Sun"};

  std::cout << "## tip by day\n";
  std::vector<std::vector<double>> groups;
  for (const std::string& d : days) {
    groups.push_back(group_values(tips, "day", d, "tip"));
    std::vector<dsts::OptD> opt(groups.back().begin(), groups.back().end());
    const dsts::Series s(d, opt);
    std::cout << "  " << d << (d.size() == 3 ? " " : "") << ": n " << s.count()
              << "   mean " << dsts::fmt(s.mean()) << "  sd " << dsts::fmt(s.stddev()) << "\n";
  }

  const dsts::AnovaResult a = dsts::anova_oneway(groups);
  const dsts::KruskalResult k = dsts::kruskal_wallis(groups);
  std::cout << "\none-way ANOVA:   F=" << dsts::fmt(a.f) << "  df=(" << dsts::fmt(a.df_between)
            << ", " << dsts::fmt(a.df_within) << ")  p=" << dsts::fmt(a.p_value) << "\n";
  std::cout << "Kruskal-Wallis:  H=" << dsts::fmt(k.h) << "  df=" << k.df
            << "  p=" << dsts::fmt(k.p_value) << "\n\n";

  std::cout << "interpretation:\n"
            << "  H0: all four day means are equal. ANOVA compares between-day\n"
            << "  spread of means to within-day spread of tips (F ~ 1 under H0).\n"
            << "  " << (a.p_value < 0.05 ? "p < 0.05: at least one day differs."
                                          : "p >= 0.05: no evidence that tips differ by day.")
            << "\n"
            << "  Kruskal-Wallis works on ranks, so the skewed right tail of tips\n"
            << "  (a few very large tips) cannot dominate the answer. "
            << (k.p_value < 0.05 ? "It rejects H0." : "It also does not reject H0.") << "\n"
            << "  Agreement between both tests is the reassuring case; when they\n"
            << "  disagree (as here), suspect outliers or heavy skew: the rank test\n"
            << "  sees a shift in the TYPICAL tip that a few huge tips hide from the mean.\n";
  return 0;
}
