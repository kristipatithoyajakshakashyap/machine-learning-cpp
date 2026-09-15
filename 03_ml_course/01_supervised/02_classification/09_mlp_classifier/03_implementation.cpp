// 10_mlp_classifier.cpp
// Lesson: multi-layer perceptron (neural net) for classification on wine.
// The net is input -> [ReLU hidden 24] -> softmax(3 classes), trained by
// backpropagation on cross-entropy with Adam.  Weight initialisation is
// Xavier-normal (seeded) so the net is reproducible but the exact values
// differ from sklearn's init, making the comparison tolerance-based.
// Equivalent: sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(24,),
//             activation='relu', solver='adam', max_iter=200).
//
// Wine: 178 x 13, 3 classes.
// EXPECTED OUTPUT (80/20 seed 7, hidden 24, 200 epochs, lr 0.01):
// mlp  train accuracy 1  test accuracy 0.9444444

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "../../01_regression/01_linear_regression/LinearRegression.hpp"
#include "helper/math/metrics.hpp"
#include "MLP.hpp"
#include "helper/math/optim.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_wine(DATA_DIR);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/c10_mlp.split");
  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  // Standardise so hidden pre-activations stay well-scaled.
  ml::Vec mu, sd;
  ml::Mat Xtr = ml::scale_fit(tr.X, &mu, &sd);
  ml::Mat Xte = ml::scale_apply(te.X, mu, sd);

  ml::seed_rng(5);
  ml::MLP model({24}, 3, 0.01, 200);
  model.fit(Xtr, tr.y);
  std::cout << "mlp  train accuracy "
            << ml::accuracy(tr.y, model.predict(Xtr)) << "  test accuracy "
            << ml::accuracy(te.y, model.predict(Xte)) << "\n";
  return 0;
}
