#ifndef MLC_MODULE_MLP_HPP
#define MLC_MODULE_MLP_HPP

// Multi-layer perceptron (MLP) — a fully-connected feedforward neural network
// trained by backpropagation with Adam optimiser.
//
// sklearn equivalent: sklearn.neural_network.MLPRegressor /
//                     sklearn.neural_network.MLPClassifier
//
// Architecture: input → hidden layers (ReLU) → output layer.
//   Regression:  linear output, MSE loss.
//   Classification: softmax output (n_classes > 1) or sigmoid (binary),
//                   cross-entropy loss.
//
// Weights are initialised with Xavier-normal (Glorot) initialisation
// using ml::next_normal for determinism.  Training uses ml::adam from
// optim.hpp.
//
// Approximate tolerance-based comparison with sklearn (~1e-4 R² / accuracy)
// because weight initialisation order differs.

#include <cstddef>
#include <vector>

#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"

namespace ml {

class MLP {
 public:
  // hidden_layers: sizes of each hidden layer, e.g. {64, 32}.
  // n_classes: 1 for regression, 2 for binary, >2 for multiclass.
  // lr / epochs: Adam learning rate and passes.
  MLP(const std::vector<size_t>& hidden_layers, size_t n_classes = 1,
      double lr = 0.001, size_t epochs = 500);

  void fit(const Mat& X, const Vec& y);
  Vec predict(const Mat& X) const;
  Mat predict_proba(const Mat& X) const;

  void save(std::ostream& out) const { archive::write(out,std::string("MLP"),hidden_layers_,n_classes_,lr_,epochs_,weights_,biases_); }
  void load(std::istream& in) { std::string tag; archive::read(in,tag); if(tag!="MLP") throw std::runtime_error("Wrong model type"); archive::read(in,hidden_layers_,n_classes_,lr_,epochs_,weights_,biases_); }
 private:
  void forward(const Vec& x, std::vector<Vec>& activations) const;
  void backward(const std::vector<Vec>& activations, const Vec& y_onehot,
                std::vector<Vec>& grads_w, std::vector<Vec>& grads_b);

  std::vector<size_t> hidden_layers_;
  size_t n_classes_;
  double lr_;
  size_t epochs_;

  // Network weights: weights_[l][j][i] connects unit i in layer l to unit j
  // in layer l+1; biases_[l][j] is the bias for unit j in layer l+1.
  std::vector<Mat> weights_;
  std::vector<Vec> biases_;
};

}  // namespace ml

#endif  // ML_NEURAL_HPP
