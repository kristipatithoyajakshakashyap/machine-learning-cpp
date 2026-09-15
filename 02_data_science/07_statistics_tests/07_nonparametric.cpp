// 07_nonparametric.cpp
// Lesson: when a variable is heavily skewed (Titanic fares), the t-test's
// mean-based statistic is fragile. The Mann-Whitney U test compares RANKS,
// so it asks "does one group tend to have larger values?" rather than
// "do the means differ?".
// Equivalent: scipy.stats.mannwhitneyu(method="asymptotic") vs
// scipy.stats.ttest_ind(equal_var=False).
//
// Data: titanic.csv (891 passengers), Fare split by Survived.
//
// EXPECTED OUTPUT:
// ## Fare by Survived
//   died    : n 549  mean 22.11789  median 10.5  sd 31.38821
//   survived: n 342  mean 48.39541  median 26   sd 66.59700
//
// Welch t-test:    t=... df=... p=...
// Mann-Whitney U:  U=... z=... p=...
// when to prefer which (text)

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/series.hpp"
#include "dsts/stats.hpp"

std::vector<double> fares_where(const dsts::DataFrame& df, double survived) {
  std::vector<double> out;
  const auto& s = df.numeric("Survived");
  const auto& f = df.numeric("Fare");
  for (size_t r = 0; r < df.rows(); ++r) {
    if (s[r].has_value() && f[r].has_value() && *s[r] == survived) out.push_back(*f[r]);
  }
  return out;
}

dsts::Series to_series(const std::string& name, const std::vector<double>& v) {
  return dsts::Series(name, std::vector<dsts::OptD>(v.begin(), v.end()));
}

void describe(const std::string& label, const dsts::Series& s) {
  std::cout << "  " << label << ": n " << s.count() << "  mean " << dsts::fmt(s.mean())
            << "  median " << dsts::fmt(s.median()) << "  sd " << dsts::fmt(s.stddev()) << "\n";
}

int main() {
  const dsts::DataFrame titanic = dsts::read_csv(DATA_DIR "/titanic.csv");
  const std::vector<double> died = fares_where(titanic, 0.0);
  const std::vector<double> survived = fares_where(titanic, 1.0);
  const dsts::Series s_died = to_series("died", died);
  const dsts::Series s_surv = to_series("survived", survived);

  std::cout << "## Fare by Survived\n";
  describe("died    ", s_died);
  describe("survived", s_surv);

  const dsts::TTestResult t = dsts::ttest_ind(s_died, s_surv);
  const dsts::MannWhitneyResult mw = dsts::mann_whitney_u(died, survived);
  std::cout << "\nWelch t-test:    t=" << dsts::fmt(t.t) << "  df=" << dsts::fmt(t.df)
            << "  p=" << dsts::fmt(t.p_value) << "\n";
  std::cout << "Mann-Whitney U:  U=" << dsts::fmt(mw.u) << "  z=" << dsts::fmt(mw.z)
            << "  p=" << dsts::fmt(mw.p_value) << "\n\n";

  // How skewed is Fare? mean >> median and max far beyond the 75% quantile.
  std::cout << "skew check (all fares): mean " << dsts::fmt(titanic.series("Fare").mean())
            << "  median " << dsts::fmt(titanic.series("Fare").median()) << "  max "
            << dsts::fmt(titanic.series("Fare").max()) << "\n\n";

  std::cout << "when to prefer which:\n"
            << "  - t-test: roughly symmetric data, or large n where the CLT rescues the\n"
            << "    mean; it answers 'are the MEANS different?'.\n"
            << "  - Mann-Whitney: skewed data, outliers, ordinal scales, small n; it\n"
            << "    answers 'is one group STOCHASTICALLY larger?' and is invariant to\n"
            << "    monotone transforms (log(fare) gives the same U).\n"
            << "  Here both reject H0 decisively, but the t statistic leans on a handful\n"
            << "  of 500-pound fares; the rank test would give the same verdict if\n"
            << "  those were capped at 100.\n";
  return 0;
}
