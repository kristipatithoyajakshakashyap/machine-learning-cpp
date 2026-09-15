// 10_mlp_regression.cpp
// Lesson: multi-layer perceptron (feedforward neural net) for regression on a
// California housing subsample.
// The net is input -> [ReLU hidden 24] -> 1 linear output unit, trained by
// backpropagation over the mean squared error using Adam.  Unlike the linear
// models, the hidden layer lets the net capture non-linear interactions, but
// with only one hidden layer the expressiveness is still limited.
// Comparison vs sklearn is tolerance-based: weight initialisation order and
// Adam schedules differ (seeded next_normal init here).
// Equivalent: sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(24,),
//             activation='relu', solver='adam', max_iter=150).
//
// California rows are subsampled to 2048 (seed 7) for speed.
//
// EXPECTED OUTPUT (2048 rows, 80/20 seed 7, hidden 24, epoch 200, lr 0.01):
// mlp  train R2 0.4110672  test R2 0.3214215

#include <iomanip>
#include <iostream>
#include <cmath>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "../01_linear_regression/LinearRegression.hpp"
#include "helper/math/metrics.hpp"
#include "MLP.hpp"
#include "helper/math/optim.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto full = ml::load_california(DATA_DIR);
  auto data = ml::subsample(full, 2048, 7);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/s10_mlp.split");

  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  // Neural nets are scale-sensitive: standardise the features with the
  // training fold's statistics, and standardise the target too so the MSE
  // gradient is well conditioned.  Predictions are mapped back afterwards.
  ml::Vec xm, xs;
  ml::Mat Xtr = ml::scale_fit(tr.X, &xm, &xs);
  ml::Mat Xte = ml::scale_apply(te.X, xm, xs);

  double ym = ml::mean(tr.y);
  double ys = std::sqrt(ml::variance(tr.y, true));
  if (ys <= 0.0) ys = 1.0;
  ml::Vec ytr(tr.y.size());
  for (size_t i = 0; i < tr.y.size(); ++i) ytr[i] = (tr.y[i] - ym) / ys;

  ml::seed_rng(7);
  ml::MLP model({24}, 1, 0.01, 200);
  model.fit(Xtr, ytr);
  auto tr_pred = model.predict(Xtr);
  auto te_pred = model.predict(Xte);
  for (double& v : tr_pred) v = v * ys + ym;
  for (double& v : te_pred) v = v * ys + ym;
  std::cout << "mlp  train R2 " << ml::r2(tr.y, tr_pred) << "  test R2 "
            << ml::r2(te.y, te_pred) << "\n";
  return 0;
}
