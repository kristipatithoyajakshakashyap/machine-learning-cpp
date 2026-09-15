// 02_data_science/src/stats.cpp
// dsts statistics: correlations, z-scores, and classical hypothesis tests
// (Welch t, chi-square independence, one-way ANOVA, Mann-Whitney U,
// Kruskal-Wallis, Kolmogorov-Smirnov normality), a normal-approximation
// confidence interval and a bootstrap of the mean. p-values are computed
// from hand-rolled incomplete gamma/beta functions (Numerical Recipes
// algorithms) so no external library is needed.
//
// Reads/writes: nothing on disk. Build: part of the dsts static library
// (target dsts); used by 07_statistics_tests and 12_advanced_eda. Exercised
// by ctest -R dsts_regression.

#include "dsts/stats.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace dsts {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Present (non-missing) values of s, in original order.
std::vector<double> present(const Series& s) {
  std::vector<double> out;
  for (const OptD& v : s.values()) {
    if (v.has_value()) out.push_back(*v);
  }
  return out;
}

// Regularized lower incomplete gamma P(a, x) via series (Numerical Recipes gamser).
double gamser(double a, double x) {
  static const double kEps = 3e-12;
  double ap = a, sum = 1.0 / a, del = sum;
  for (int i = 0; i < 500; ++i) {
    ++ap;
    del *= x / ap;
    sum += del;
    if (std::fabs(del) < std::fabs(sum) * kEps) {
      return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
    }
  }
  return kNaN;
}

// Upper regularized incomplete gamma Q(a, x) via continued fraction (gamcf).
double gamcf(double a, double x) {
  static const double kEps = 3e-12;
  double b = x + 1.0 - a, c = 1.0 / kEps, d = 1.0 / b, h = d;
  for (int i = 1; i <= 500; ++i) {
    const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
    b += 2.0;
    d = an * d + b;
    if (std::fabs(d) < kEps) d = kEps;
    c = b + an / c;
    if (std::fabs(c) < kEps) c = kEps;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < kEps) break;
  }
  return h * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

// Q(a, x): 1 - P from the series for small x, the CF directly for large x.
double gammq(double a, double x) {
  return x < a + 1.0 ? 1.0 - gamser(a, x) : gamcf(a, x);
}

// Regularized incomplete beta I_x(a, b) via continued fraction (Numerical
// Recipes betacf), with the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) for fast
// convergence. Shared by the t, F and (through F) ANOVA p-values.
double betai(double a, double b, double x) {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  const double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                             a * std::log(x) + b * std::log(1.0 - x));
  if (x >= (a + 1.0) / (a + b + 2.0)) return 1.0 - betai(b, a, 1.0 - x);
  const double kEps = 3e-12;
  const double kFpmin = 1e-300;
  const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::fabs(d) < kFpmin) d = kFpmin;
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m <= 500; ++m) {
    const double m2 = 2.0 * static_cast<double>(m);
    double aa = static_cast<double>(m) * (b - static_cast<double>(m)) * x /
                ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kFpmin) d = kFpmin;
    c = 1.0 + aa / c;
    if (std::fabs(c) < kFpmin) c = kFpmin;
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + static_cast<double>(m)) * (qab + static_cast<double>(m)) * x /
         ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kFpmin) d = kFpmin;
    c = 1.0 + aa / c;
    if (std::fabs(c) < kFpmin) c = kFpmin;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < kEps) break;
  }
  return bt * h / a;
}

// Student-t survival value P(T > |t|) via the incomplete beta function.
double t_survival(double t_value, double df) {
  // CDF_t(t) = 1 - 0.5 * I_z(a=df/2, b=1/2) with z = df/(df+t^2).
  const double z = df / (df + t_value * t_value);
  return 0.5 * betai(df / 2.0, 0.5, z);
}

// F-distribution survival P(F > f) = I_{d2/(d2+d1 f)}(d2/2, d1/2).
double f_survival(double f, double d1, double d2) {
  if (!(f > 0.0)) return 1.0;
  return betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f));
}

// Standard normal survival P(Z > z).
double normal_survival(double z) { return 0.5 * std::erfc(z / std::sqrt(2.0)); }

// Asymptotic two-sided p-value for a one-sample KS statistic D (Stephens).
double ks_p_value(double d, size_t n) {
  const double sn = std::sqrt(static_cast<double>(n)) + 0.12 +
                    0.11 / std::sqrt(static_cast<double>(n));
  const double lam = sn * d;
  double sum = 0.0;
  for (int k = 1; k <= 200; ++k) {
    const double term = std::exp(-2.0 * static_cast<double>(k) * k * lam * lam);
    sum += (k % 2 == 1) ? term : -term;
    if (term < 1e-300) break;
  }
  return std::clamp(2.0 * sum, 0.0, 1.0);
}

// Ranks 1..n with ties given the average of their positions (needed by
// Spearman, Mann-Whitney and Kruskal-Wallis).
std::vector<double> ranks(const std::vector<double>& data) {
  // Average ranks of each element when sorted ascending.
  std::vector<double> out(data.size());
  std::vector<size_t> order(data.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return data[a] < data[b]; });
  size_t i = 0;
  while (i < order.size()) {
    size_t j = i;
    while (j + 1 < order.size() && data[order[j + 1]] == data[order[i]]) ++j;
    const double r = 0.5 * (static_cast<double>(i) + static_cast<double>(j)) + 1.0;
    for (size_t k = i; k <= j; ++k) out[order[k]] = r;
    i = j + 1;
  }
  return out;
}

// Sum over tie groups of (t^3 - t), used by the rank-test tie corrections.
double tie_term(std::vector<double> sorted_values) {
  std::sort(sorted_values.begin(), sorted_values.end());
  double sum = 0.0;
  size_t i = 0;
  while (i < sorted_values.size()) {
    size_t j = i;
    while (j + 1 < sorted_values.size() && sorted_values[j + 1] == sorted_values[i]) ++j;
    const double t = static_cast<double>(j - i + 1);
    sum += t * t * t - t;
    i = j + 1;
  }
  return sum;
}

// Pearson correlation of two equal-length vectors; NaN when n < 2 or either
// side is constant.
double corr(const std::vector<double>& x, const std::vector<double>& y) {
  if (x.size() != y.size() || x.size() < 2) return kNaN;
  double mx = 0, my = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    mx += x[i];
    my += y[i];
  }
  mx /= static_cast<double>(x.size());
  my /= static_cast<double>(y.size());
  double num = 0, sx = 0, sy = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double dx = x[i] - mx, dy = y[i] - my;
    num += dx * dy;
    sx += dx * dx;
    sy += dy * dy;
  }
  if (sx == 0.0 || sy == 0.0) return kNaN;
  return num / std::sqrt(sx * sy);
}

}  // namespace

// Pearson correlation over positions where both series are present.
double pearson(const Series& x, const Series& y) {
  const auto vx = x.values(), vy = y.values();
  if (vx.size() != vy.size()) return kNaN;
  std::vector<double> px, py;
  for (size_t i = 0; i < vx.size(); ++i) {
    if (vx[i].has_value() && vy[i].has_value()) {
      px.push_back(*vx[i]);
      py.push_back(*vy[i]);
    }
  }
  return corr(px, py);
}

// Spearman rank correlation: Pearson on average ranks of the complete pairs.
double spearman(const Series& x, const Series& y) {
  const auto vx = x.values(), vy = y.values();
  if (vx.size() != vy.size()) return kNaN;
  std::vector<double> px, py;
  for (size_t i = 0; i < vx.size(); ++i) {
    if (vx[i].has_value() && vy[i].has_value()) {
      px.push_back(*vx[i]);
      py.push_back(*vy[i]);
    }
  }
  if (px.size() < 2) return kNaN;
  const auto rx = ranks(px), ry = ranks(py);
  return corr(rx, ry);
}

// Interquartile range Q3 - Q1.
double iqr(const Series& x) { return x.quantile(0.75) - x.quantile(0.25); }

// (x - mean) / sample sd for every present value (missing cells are dropped).
std::vector<double> zscore(const Series& x) {
  const auto p = present(x);
  std::vector<double> out;
  out.reserve(p.size());
  const double m = x.mean();
  const double sd = x.stddev();
  for (double v : p) out.push_back((v - m) / sd);
  return out;
}

// Welch's two-sample t-test (unequal variances) with the Welch-Satterthwaite
// degrees of freedom. Two-sided p-value. NaN fields if a group has < 2 values
// or the standard error is zero.
TTestResult ttest_ind(const Series& a, const Series& b) {
  const auto pa = present(a), pb = present(b);
  TTestResult r{kNaN, kNaN, kNaN};
  if (pa.size() < 2 || pb.size() < 2) return r;
  const double na = static_cast<double>(pa.size()),
               nb = static_cast<double>(pb.size());
  double ma = 0, mb = 0;
  for (double v : pa) ma += v;
  for (double v : pb) mb += v;
  ma /= na;
  mb /= nb;
  double va = 0, vb = 0;
  for (double v : pa) va += (v - ma) * (v - ma);
  for (double v : pb) vb += (v - mb) * (v - mb);
  va /= (na - 1.0);
  vb /= (nb - 1.0);
  const double se = std::sqrt(va / na + vb / nb);
  if (se == 0.0) return r;
  r.t = (ma - mb) / se;
  r.df = (va / na + vb / nb) * (va / na + vb / nb);
  r.df /= (va / na) * (va / na) / (na - 1.0) +
          (vb / nb) * (vb / nb) / (nb - 1.0);
  // t_survival returns the one-sided tail P(T > |t|); the two-sided p-value
  // doubles it (matches scipy.stats.ttest_ind).
  r.p_value = 2.0 * t_survival(std::fabs(r.t), r.df);
  return r;
}

// Pearson chi-square test on a contingency table of counts. Expected count
// for cell (i,j) = row_i * col_j / total; dof = (rows-1)(cols-1);
// p = Q(dof/2, chi2/2).
ChiSquareResult chi2_independence(const std::vector<std::vector<double>>& observed) {
  ChiSquareResult r{0.0, 0, kNaN};
  if (observed.empty()) return r;
  const int rows = static_cast<int>(observed.size());
  const int cols = static_cast<int>(observed[0].size());
  std::vector<double> row_sum(rows, 0.0), col_sum(cols, 0.0);
  double total = 0.0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      row_sum[i] += observed[i][j];
      col_sum[j] += observed[i][j];
      total += observed[i][j];
    }
  }
  double chi2 = 0.0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const double expected = row_sum[i] * col_sum[j] / total;
      chi2 += (observed[i][j] - expected) * (observed[i][j] - expected) / expected;
    }
  }
  r.chi2 = chi2;
  r.dof = (rows - 1) * (cols - 1);
  r.p_value = r.dof > 0 ? gammq(static_cast<double>(r.dof) / 2.0, chi2 / 2.0) : kNaN;
  return r;
}

// One-way ANOVA F = (SS_between/df_between) / (SS_within/df_within) with an
// F-distribution p-value. NaN fields for < 2 groups, an empty group or zero
// within-group variance.
AnovaResult anova_oneway(const std::vector<std::vector<double>>& groups) {
  AnovaResult r{kNaN, kNaN, kNaN, kNaN};
  const size_t k = groups.size();
  if (k < 2) return r;
  double grand = 0.0, n_total = 0.0;
  for (const auto& g : groups) {
    if (g.empty()) return r;
    for (double v : g) grand += v;
    n_total += static_cast<double>(g.size());
  }
  grand /= n_total;
  double ss_between = 0.0, ss_within = 0.0;
  for (const auto& g : groups) {
    double m = 0.0;
    for (double v : g) m += v;
    m /= static_cast<double>(g.size());
    ss_between += static_cast<double>(g.size()) * (m - grand) * (m - grand);
    for (double v : g) ss_within += (v - m) * (v - m);
  }
  r.df_between = static_cast<double>(k - 1);
  r.df_within = n_total - static_cast<double>(k);
  if (r.df_within <= 0.0 || ss_within == 0.0) return r;
  r.f = (ss_between / r.df_between) / (ss_within / r.df_within);
  r.p_value = f_survival(r.f, r.df_between, r.df_within);
  return r;
}

// Mann-Whitney U with tie-corrected normal approximation and continuity
// correction (scipy's asymptotic method). Reports min(U1, U2).
MannWhitneyResult mann_whitney_u(const std::vector<double>& a, const std::vector<double>& b) {
  MannWhitneyResult r{kNaN, kNaN, kNaN};
  if (a.empty() || b.empty()) return r;
  std::vector<double> pooled(a);
  pooled.insert(pooled.end(), b.begin(), b.end());
  const auto rk = ranks(pooled);
  const double n1 = static_cast<double>(a.size()), n2 = static_cast<double>(b.size());
  double r1 = 0.0;
  for (size_t i = 0; i < a.size(); ++i) r1 += rk[i];
  const double u1 = r1 - n1 * (n1 + 1.0) / 2.0;
  const double u2 = n1 * n2 - u1;
  r.u = std::min(u1, u2);  // scipy reports U1; we report min(U1, U2)
  const double n = n1 + n2;
  const double mean_u = n1 * n2 / 2.0;
  const double var_u = n1 * n2 / 12.0 * ((n + 1.0) - tie_term(pooled) / (n * (n - 1.0)));
  if (!(var_u > 0.0)) return r;
  // scipy two-sided asymptotic: z from max(U1,U2) with continuity correction.
  const double z = (std::max(u1, u2) - mean_u - 0.5) / std::sqrt(var_u);
  r.z = z;
  r.p_value = std::min(1.0, 2.0 * normal_survival(z));
  return r;
}

// Kruskal-Wallis H test (rank-based one-way ANOVA) with tie correction;
// p-value from the chi-square distribution with k-1 dof.
KruskalResult kruskal_wallis(const std::vector<std::vector<double>>& groups) {
  KruskalResult r{kNaN, 0, kNaN};
  if (groups.size() < 2) return r;
  std::vector<double> pooled;
  for (const auto& g : groups) {
    if (g.empty()) return r;
    pooled.insert(pooled.end(), g.begin(), g.end());
  }
  const auto rk = ranks(pooled);
  const double n = static_cast<double>(pooled.size());
  double h = 0.0;
  size_t offset = 0;
  for (const auto& g : groups) {
    double sum = 0.0;
    for (size_t i = 0; i < g.size(); ++i) sum += rk[offset + i];
    offset += g.size();
    h += sum * sum / static_cast<double>(g.size());
  }
  h = 12.0 / (n * (n + 1.0)) * h - 3.0 * (n + 1.0);
  const double correction = 1.0 - tie_term(pooled) / (n * n * n - n);
  if (!(correction > 0.0)) return r;
  r.h = h / correction;
  r.df = static_cast<int>(groups.size()) - 1;
  r.p_value = gammq(static_cast<double>(r.df) / 2.0, r.h / 2.0);
  return r;
}

// One-sample KS test of the z-scored data against N(0,1). D is the largest
// gap between the empirical CDF (checked just before and just after each
// point) and the normal CDF.
KsResult ks_normal(const Series& x) {
  KsResult r{kNaN, kNaN};
  const auto p = present(x);
  if (p.empty()) return r;
  const double m = x.mean();
  const double sd = x.stddev();
  if (!(sd > 0.0)) return r;
  // Empirical CDF vs N(0,1) on z-scores.
  std::vector<double> z;
  z.reserve(p.size());
  for (double v : p) z.push_back((v - m) / sd);
  std::sort(z.begin(), z.end());
  double d = 0.0;
  for (size_t i = 0; i < z.size(); ++i) {
    const double cdf = 0.5 * (1.0 + std::erf(z[i] / std::sqrt(2.0)));
    const double fn = static_cast<double>(i + 1) / static_cast<double>(z.size());
    const double fpm = static_cast<double>(i) / static_cast<double>(z.size());
    d = std::max(d, std::max(cdf - fpm, fn - cdf));
  }
  r.statistic = d;
  r.p_value = ks_p_value(d, z.size());
  return r;
}

// Normal-approximation 95% CI for the mean: mean +/- 1.96 * sd/sqrt(n).
ConfidenceInterval mean_ci95(const Series& x) {
  const auto p = present(x);
  ConfidenceInterval ci{kNaN, kNaN};
  if (p.size() < 2) return ci;
  const double m = x.mean();
  const double se = x.stddev() / std::sqrt(static_cast<double>(p.size()));
  const double z = 1.959963984540054;  // Phi^-1(0.975)
  ci.lower = m - z * se;
  ci.upper = m + z * se;
  return ci;
}

// `samples` bootstrap replicates of the mean: resample the present values
// with replacement (seeded mt19937) and average each resample.
std::vector<double> bootstrap_mean(const Series& x, size_t samples, uint32_t seed) {
  const auto p = present(x);
  std::vector<double> out;
  if (p.empty()) return out;
  out.reserve(samples);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> pick(0, p.size() - 1);
  for (size_t b = 0; b < samples; ++b) {
    double s = 0.0;
    for (size_t i = 0; i < p.size(); ++i) s += p[pick(rng)];
    out.push_back(s / static_cast<double>(p.size()));
  }
  return out;
}

}  // namespace dsts