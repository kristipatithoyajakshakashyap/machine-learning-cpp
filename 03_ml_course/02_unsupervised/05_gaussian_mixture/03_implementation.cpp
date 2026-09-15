// 03_ml_course/02_unsupervised/05_gaussian_mixture/03_implementation.cpp
// Purpose: lesson showing a Gaussian mixture fitted by EM. A tiny four-point
//          example is followed by a 2-D synthetic mixture of three components.
//          Because EM is deterministic for a fixed seed, refitting with
//          iterations = 1, 2, ... N reproduces the log-likelihood trajectory,
//          which must be non-decreasing.
// Inputs:  none from disk; data is generated in code (seed 3), EM seed 42.
// Outputs: results/03_implementation_results/{log_likelihood.csv,
//          responsibilities.csv,figures/log_likelihood.svg,
//          figures/components_scatter.svg}; also prints labels, LL, BIC, AIC.
// Run target: ugm_implementation.
// sklearn equivalent: sklearn.mixture.GaussianMixture(covariance_type='diag').

#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "GaussianMixture.hpp"
#include "helper/reporting/artifacts.hpp"

int main() {
  using namespace ml;
  // Part 1 - four points on a line, two components: the pairs {0,2} and {8,10}
  // should end up in different components, so the printed labels are two of
  // one value followed by two of the other.
  Mat X{{0, 0}, {2, 0}, {8, 0}, {10, 0}};
  GaussianMixture model(2);
  model.fit(X);
  for (double y : model.predict(X)) std::cout << y << ' ';
  std::cout << '\n';

  // Part 2 - three 2-D blobs of 60 points each with different spreads
  // (sd 0.5, 1.0, 0.7) centred at (0,0), (5,1), (2,5). Seed 3 fixes the data.
  Artifacts a(RUN_OUTPUT_DIR, ".");
  std::mt19937 gen(3);
  std::normal_distribution<double> n1(0, .5), n2(0, 1.0), n3(0, .7);
  Mat S;
  for (int i = 0; i < 60; ++i) S.push_back({n1(gen), n1(gen)});
  for (int i = 0; i < 60; ++i) S.push_back({5 + n2(gen), 1 + n2(gen)});
  for (int i = 0; i < 60; ++i) S.push_back({2 + n3(gen), 5 + n3(gen)});

  // Log-likelihood trajectory: fit t = 1..40 iterations from the same seed.
  // Each fit repeats the earlier steps exactly, so the t-th value is the LL
  // after t EM steps. (Refitting is O(t) per point; fine for a lesson.)
  const size_t k = 3;
  const size_t max_iter = 40;
  std::ostringstream ll;
  ll << std::setprecision(17) << "iteration,log_likelihood\n";
  Vec it_axis, ll_axis;
  for (size_t t = 1; t <= max_iter; ++t) {
    GaussianMixture g(k, t, 42);
    g.fit(S);
    const double v = g.score(S);
    ll << t << ',' << v << '\n';
    it_axis.push_back(double(t));
    ll_axis.push_back(v);
  }
  a.write("log_likelihood.csv", ll.str());
  std::cout << std::setprecision(6) << "log-likelihood after 1 EM step "
            << ll_axis.front() << ", after " << max_iter << " steps "
            << ll_axis.back() << '\n';
  Plot curve;
  curve.title("EM log-likelihood per iteration (k = 3)");
  curve.xlabel("iteration");
  curve.ylabel("log-likelihood");
  curve.line(it_axis, ll_axis, "log-likelihood");
  a.figure("figures/log_likelihood.svg", curve);

  // Final fit: soft responsibilities per row plus the hard arg-max label.
  GaussianMixture g(k, max_iter, 42);
  g.fit(S);
  const Mat R = g.predict_proba(S);
  const Vec hard = g.predict(S);
  std::ostringstream resp;
  resp << std::setprecision(17) << "row,x,y";
  for (size_t c = 0; c < k; ++c) resp << ",resp_" << c;
  resp << ",component\n";
  for (size_t i = 0; i < S.size(); ++i) {
    resp << i << ',' << S[i][0] << ',' << S[i][1];
    for (size_t c = 0; c < k; ++c) resp << ',' << R[i][c];
    resp << ',' << hard[i] << '\n';
  }
  a.write("responsibilities.csv", resp.str());
  // Information criteria used for choosing k in the end-to-end lesson.
  std::cout << "BIC " << g.bic(S) << "  AIC " << g.aic(S) << '\n';

  Plot scatter;
  scatter.title("Points coloured by most responsible component");
  scatter.xlabel("x");
  scatter.ylabel("y");
  scatter.scatter(S, hard, "component");
  a.figure("figures/components_scatter.svg", scatter);
  std::cout << "Saved responsibilities.csv, log_likelihood.csv and figures "
               "under "
            << RUN_OUTPUT_DIR << '\n';
}
