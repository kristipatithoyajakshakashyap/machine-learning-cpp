#ifndef DSTS_STATS_HPP
#define DSTS_STATS_HPP

// Statistical helpers used across the course.
// The numpy/scipy equivalents live in numpy + scipy.stats.
// Every routine ignores missing cells (std::nullopt) as pandas does;
// functions with too few valid observations return NaN.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "dsts/dtypes.hpp"
#include "dsts/series.hpp"

namespace dsts {

// --- descriptive + correlation ---------------------------------------
// Pearson correlation over complete pairs; NaN if < 2 pairs.
double pearson(const Series& x, const Series& y);
// Spearman rank correlation (average ranks for ties).
double spearman(const Series& x, const Series& y);
// Interquartile range of the present values.
double iqr(const Series& x);
// Z-scores of each present value against the sample mean/stddev.
std::vector<double> zscore(const Series& x);

// --- hypothesis tests -------------------------------------------------
// Welch's two-sample t-test; returns {t, degrees_of_freedom, p}.
struct TTestResult {
  double t;
  double df;
  double p_value;
};
TTestResult ttest_ind(const Series& a, const Series& b);

// Chi-square independence test on an observed R x C contingency table.
struct ChiSquareResult {
  double chi2;
  int dof;
  double p_value;
};
ChiSquareResult chi2_independence(const std::vector<std::vector<double>>& observed);

// One-way ANOVA over k groups of raw values (no missing cells).
// F = (SS_between/(k-1)) / (SS_within/(N-k)); p from the F distribution.
// scipy equivalent: scipy.stats.f_oneway.
struct AnovaResult {
  double f;
  double df_between;
  double df_within;
  double p_value;
};
AnovaResult anova_oneway(const std::vector<std::vector<double>>& groups);

// Mann-Whitney U rank-sum test, two-sided, normal approximation with tie and
// continuity corrections. u = min(U1, U2); z is computed on max(U1, U2).
// scipy equivalent: scipy.stats.mannwhitneyu(a, b, method="asymptotic").
struct MannWhitneyResult {
  double u;
  double z;
  double p_value;
};
MannWhitneyResult mann_whitney_u(const std::vector<double>& a, const std::vector<double>& b);

// Kruskal-Wallis H test over k groups (rank ANOVA), tie-corrected; p from
// chi-square with k-1 dof. scipy equivalent: scipy.stats.kruskal.
struct KruskalResult {
  double h;
  int df;
  double p_value;
};
KruskalResult kruskal_wallis(const std::vector<std::vector<double>>& groups);

// One-sample Kolmogorov-Smirnov test against a standard normal.
struct KsResult {
  double statistic;  // D
  double p_value;    // asymptotic p (one sample), two-sided
};
KsResult ks_normal(const Series& x);

// --- confidence intervals ----------------------------------------------
// 95% z-interval for the mean of the present values.
struct ConfidenceInterval {
  double lower;
  double upper;
};
ConfidenceInterval mean_ci95(const Series& x);

// --- resampling ---------------------------------------------------------
// Bootstrap the mean: B resamples (with replacement) of size SEED_FIXED,
// returning the B sample means in a fixed (seeded) order.
std::vector<double> bootstrap_mean(const Series& x, size_t samples, uint32_t seed);

}  // namespace dsts

#endif  // DSTS_STATS_HPP