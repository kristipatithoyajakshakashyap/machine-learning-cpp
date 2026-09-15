// 03_ml_course/01_supervised/01_regression/06_mlp_regressor/MLP.cpp
// ml::MLP implementation: a fully connected feed-forward network with ReLU hidden
// layers trained by Adam on a flattened parameter vector. n_classes == 1 gives a
// regressor (linear output, MSE loss); n_classes > 1 gives a softmax classifier
// (cross-entropy loss). Weights use Glorot-style random initialisation, so results
// depend on the global RNG seed set by the calling lesson.
//
// Reads:   nothing on disk (data arrives as ml::Mat / ml::Vec).
// Writes:  nothing; save()/load() in MLP.hpp serialise the tensors to a stream.
// Build:   compiled into the static library ml_rmlp (with LinearRegression.cpp) and
//          linked by targets rmlp_theory, rmlp_math_intuition, rmlp_implementation,
//          rmlp_end_to_end and rmlp_predict. Tests: ctest -R rmlp.
// Layout:  weights_[l][j][i] is the weight from unit i of layer l to unit j of
//          layer l+1; biases_[l][j] is the bias of unit j in layer l+1.
#include "MLP.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "helper/math/optim.hpp"

namespace ml {

namespace {
// ReLU activation and its derivative.
double relu(double z) { return z > 0.0 ? z : 0.0; }
// d relu / dz: 1 for positive inputs, 0 otherwise (the value at 0 is a convention).
double relu_deriv(double z) { return z > 0.0 ? 1.0 : 0.0; }

// Stable softmax (row vector normalised to probabilities).
// Subtracting the row maximum before exp() prevents overflow; the result is the
// same because the constant cancels in the normalisation.
void softmax(Vec& v) {
  double mx = *std::max_element(v.begin(), v.end());
  double s = 0.0;
  for (double& x : v) {
    x = std::exp(x - mx);
    s += x;
  }
  for (double& x : v) x /= s;
}

// Xavier/Glorot-init variance: sqrt(6/(fan_in+fan_out)) keeps pre-activations
// on a comparable scale across layer sizes (sklearn's MLP default var is
// 2/(fan_in+fan_out); the sqrt(6) factor matches the Glorot uniform bound).
double init_std(size_t fan_in, size_t fan_out) {
  return std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
}
}  // namespace

// Construct an untrained network. hidden_layers lists the width of each hidden
// layer (may be empty for a linear model); lr and epochs are Adam settings.
// Throws std::invalid_argument for n_classes == 0, non-positive/non-finite lr,
// zero epochs, or a zero-width hidden layer.
MLP::MLP(const std::vector<size_t>& hidden_layers, size_t n_classes, double lr,
         size_t epochs)
    : hidden_layers_(hidden_layers), n_classes_(n_classes), lr_(lr),
      epochs_(epochs) {
  if(n_classes_==0||!std::isfinite(lr_)||lr_<=0||epochs_==0)throw std::invalid_argument("Invalid MLP parameters");
  for(size_t width:hidden_layers_)if(width==0)throw std::invalid_argument("MLP hidden layers cannot have zero units");
}

// Allocate and Xavier-initialise the weight/bias tensors from the layer sizes
// [p, h1, h2, ..., out].
// Train on X (n x p) and targets y (length n). For a classifier every y must be an
// integer label in [0, n_classes). Validates shapes and finiteness, then
// initialises weights, flattens them into theta and runs Adam for `epochs_`
// full-batch steps. Throws std::runtime_error / std::invalid_argument on bad input.
void MLP::fit(const Mat& X, const Vec& y) {
  if (X.empty() || X[0].empty() || X.size() != y.size())
    throw std::runtime_error("mlp: dimension mismatch");
  for(const auto& row:X){
    if(row.size()!=X[0].size())throw std::invalid_argument("Ragged MLP feature matrix");
    for(double value:row)if(!std::isfinite(value))throw std::invalid_argument("MLP features must be finite");
  }
  for(double value:y){
    if(!std::isfinite(value))throw std::invalid_argument("MLP targets must be finite");
    if(n_classes_>1&&(value<0||value!=std::floor(value)||value>=static_cast<double>(n_classes_)))
      throw std::invalid_argument("MLP label outside configured class range");
  }
  size_t p = X[0].size();
  size_t out = n_classes_ == 1 ? 1 : n_classes_;

  // Build the layer-size ladder: input + hidden + output.
  std::vector<size_t> sizes;
  sizes.push_back(p);
  for (size_t h : hidden_layers_) sizes.push_back(h);
  sizes.push_back(out);

  weights_.clear();
  biases_.clear();
  for (size_t l = 0; l + 1 < sizes.size(); ++l) {
    size_t fan_in = sizes[l], fan_out = sizes[l + 1];
    double sd = init_std(fan_in, fan_out);
    Mat W(fan_out, Vec(fan_in));
    Vec b(fan_out, 0.0);
    for (auto& row : W)
      for (double& v : row) v = next_normal(0.0, sd);
    weights_.push_back(std::move(W));
    biases_.push_back(std::move(b));
  }

  size_t n = X.size();

  // Flattened view of every parameter for Adam.
  size_t nparams = 0;
  for (size_t l = 0; l < weights_.size(); ++l)
    nparams += weights_[l].size() * weights_[l][0].size() + biases_[l].size();

  // Loss: MSE for regression, softmax cross-entropy for classification.
  auto loss = [&](const Vec& theta) -> double {
    // Re-materialise the weights from the flat vector.
    std::vector<Mat> W = weights_;
    std::vector<Vec> b = biases_;
    size_t c = 0;
    for (size_t l = 0; l < W.size(); ++l) {
      for (auto& row : W[l])
        for (double& v : row) v = theta[c++];
      for (double& v : b[l]) v = theta[c++];
    }
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
      // Forward pass, storing activations per layer.
      std::vector<Vec> act = {X[i]};
      for (size_t l = 0; l + 1 < weights_.size(); ++l) {
        Vec z(b[l].size(), b[l][0]);
        for (size_t j = 0; j < z.size(); ++j) {
          z[j] = b[l][j];
          for (size_t i2 = 0; i2 < act.back().size(); ++i2)
            z[j] += W[l][j][i2] * act.back()[i2];
        }
        for (double& v : z) v = relu(v);
        act.push_back(std::move(z));
      }
      // Final linear layer -> output.
      Vec z = b.back();
      for (size_t j = 0; j < z.size(); ++j)
        for (size_t i2 = 0; i2 < act.back().size(); ++i2)
          z[j] += W.back()[j][i2] * act.back()[i2];
      if (n_classes_ == 1) {
        // Regression: squared error. Classification: -log p(true class), clamped at
        // 1e-15 so a confident wrong prediction cannot produce log(0).
        double diff = z[0] - y[i];
        total += diff * diff;
      } else {
        softmax(z);
        total -= std::log(std::max(z[static_cast<size_t>(y[i])], 1e-15));
      }
    }
    return total / static_cast<double>(n);
  };

  // Backpropagation gradient.  Implemented as a direct function of theta so
  // Adam sees a uniform 1-D vector, exactly like the linear/logistic models.
  auto grad = [&](const Vec& theta, Vec& g) -> void {
    std::vector<Mat> W = weights_;
    std::vector<Vec> b = biases_;
    size_t c = 0;
    for (size_t l = 0; l < W.size(); ++l) {
      for (auto& row : W[l])
        for (double& v : row) v = theta[c++];
      for (double& v : b[l]) v = theta[c++];
    }
    g.assign(nparams, 0.0);

    std::vector<Vec> grad_W(W.size(), Vec());
    std::vector<Vec> grad_b(b.size(), Vec());
    for (size_t l = 0; l < W.size(); ++l) {
      grad_W[l].assign(W[l].size() * W[l][0].size(), 0.0);
      grad_b[l].assign(b[l].size(), 0.0);
    }

    for (size_t i = 0; i < n; ++i) {
      // Forward pass storing both pre-activations (z) and activations (a).
      std::vector<Vec> a = {X[i]};
      std::vector<Vec> z(weights_.size());
      for (size_t l = 0; l + 1 < weights_.size(); ++l) {
        z[l] = b[l];
        for (size_t j = 0; j < z[l].size(); ++j)
          for (size_t i2 = 0; i2 < a.back().size(); ++i2)
            z[l][j] += W[l][j][i2] * a.back()[i2];
        Vec act = z[l];
        for (double& v : act) v = relu(v);
        a.push_back(std::move(act));
      }
      z.back() = b.back();
      for (size_t j = 0; j < z.back().size(); ++j)
        for (size_t i2 = 0; i2 < a.back().size(); ++i2)
          z.back()[j] += W.back()[j][i2] * a.back()[i2];

      // Output-layer error.
      size_t L = weights_.size();  // number of weight matrices
      Vec delta(b.back().size(), 0.0);
      if (n_classes_ == 1) {
        double out = z.back()[0];
        // dMSE/dz for one sample, already divided by n so the sum over samples is the
        // mean-loss gradient (matches what loss() returns).
        delta[0] = 2.0 * (out - y[i]) / static_cast<double>(n);
      } else {
        Vec p = z.back();
        softmax(p);
        for (size_t k = 0; k < p.size(); ++k)
          delta[k] = (p[k] - (k == static_cast<size_t>(y[i]) ? 1.0 : 0.0)) /
                     static_cast<double>(n);
      }

      // Backpropagate.
      for (long l = static_cast<long>(L) - 1; l >= 0; --l) {
        size_t ll = static_cast<size_t>(l);
        const Vec& a_prev = a[ll];
        for (size_t j = 0; j < delta.size(); ++j) {
          for (size_t i2 = 0; i2 < a_prev.size(); ++i2)
            grad_W[ll][j * a_prev.size() + i2] += delta[j] * a_prev[i2];
          grad_b[ll][j] += delta[j];
        }
        // Next (lower) layer's delta = W^T delta . relu'(z_prev).
        if (ll > 0) {
          Vec d_next(a[ll].size(), 0.0);
          for (size_t i2 = 0; i2 < d_next.size(); ++i2) {
            double acc = 0.0;
            for (size_t j = 0; j < delta.size(); ++j)
              acc += W[ll][j][i2] * delta[j];
            d_next[i2] = acc * relu_deriv(z[ll - 1][i2]);
          }
          delta = std::move(d_next);
        } else {
          // For the input projection, the hidden delta shape follows the
          // number of hidden units of layer 0 (already handled above).
          break;
        }
      }
    }

    // Scatter back into the flat buffer.
    c = 0;
    for (size_t l = 0; l < W.size(); ++l) {
      for (size_t k = 0; k < grad_W[l].size(); ++k) g[c++] = grad_W[l][k];
      for (double v : grad_b[l]) g[c++] = v;
    }
  };

  Vec theta(nparams, 0.0);
  // Initialise theta from the (already random) weights.
  {
    size_t c = 0;
    for (size_t l = 0; l < weights_.size(); ++l) {
      for (const auto& row : weights_[l])
        for (double v : row) theta[c++] = v;
      for (double v : biases_[l]) theta[c++] = v;
    }
  }

  // adam() lives in helper/math/optim.hpp and returns the best parameter vector.
  Vec best = adam(theta, epochs_, lr_, loss, grad).w;

  // Write the optimised parameters back into the tensors.
  {
    size_t c = 0;
    for (size_t l = 0; l < weights_.size(); ++l) {
      for (auto& row : weights_[l])
        for (double& v : row) v = best[c++];
      for (double& v : biases_[l]) v = best[c++];
    }
  }
}

// One forward pass for prediction (no dropout, deterministic).
// Returns one value per row: the linear output for regression, or the argmax class
// index for classification. Throws if the model is unfitted, if a row has the wrong
// feature count, or if a feature is not finite.
Vec MLP::predict(const Mat& X) const {
  if (weights_.empty()) throw std::runtime_error("mlp: model not fitted");
  for(const auto& row:X){
    if(row.size()!=weights_[0][0].size())throw std::invalid_argument("MLP prediction feature count mismatch");
    for(double value:row)if(!std::isfinite(value))throw std::invalid_argument("MLP features must be finite");
  }
  Vec out(X.size(), 0.0);
  for (size_t i = 0; i < X.size(); ++i) {
    std::vector<Vec> act = {X[i]};
    for (size_t l = 0; l + 1 < weights_.size(); ++l) {
      Vec z(biases_[l].size(), 0.0);
      for (size_t j = 0; j < z.size(); ++j) {
        z[j] = biases_[l][j];
        for (size_t i2 = 0; i2 < act.back().size(); ++i2)
          z[j] += weights_[l][j][i2] * act.back()[i2];
      }
      for (double& v : z) v = relu(v);
      act.push_back(std::move(z));
    }
    Vec z(biases_.back().size(), 0.0);
    for (size_t j = 0; j < z.size(); ++j) {
      z[j] = biases_.back()[j];
      for (size_t i2 = 0; i2 < act.back().size(); ++i2)
        z[j] += weights_.back()[j][i2] * act.back()[i2];
    }
    if (n_classes_ == 1)
      out[i] = z[0];
    else {
      softmax(z);
      double best = z[0];
      size_t best_k = 0;
      for (size_t k = 1; k < z.size(); ++k)
        if (z[k] > best) {
          best = z[k];
          best_k = k;
        }
      out[i] = static_cast<double>(best_k);
    }
  }
  return out;
}

// Class probabilities (n x n_classes) from the softmax output. Only valid for a
// classifier (n_classes > 1); regression models throw std::runtime_error.
// The forward pass duplicates predict() so that both stay allocation-light.
Mat MLP::predict_proba(const Mat& X) const {
  if (weights_.empty()) throw std::runtime_error("mlp: model not fitted");
  if (n_classes_ == 1)
    throw std::runtime_error("mlp: predict_proba on a regression MLP");
  for(const auto& row:X){
    if(row.size()!=weights_[0][0].size())throw std::invalid_argument("MLP prediction feature count mismatch");
    for(double value:row)if(!std::isfinite(value))throw std::invalid_argument("MLP features must be finite");
  }
  Mat proba(X.size(), Vec(n_classes_, 0.0));
  for (size_t i = 0; i < X.size(); ++i) {
    std::vector<Vec> act = {X[i]};
    for (size_t l = 0; l + 1 < weights_.size(); ++l) {
      Vec z(biases_[l].size(), 0.0);
      for (size_t j = 0; j < z.size(); ++j) {
        z[j] = biases_[l][j];
        for (size_t i2 = 0; i2 < act.back().size(); ++i2)
          z[j] += weights_[l][j][i2] * act.back()[i2];
      }
      for (double& v : z) v = relu(v);
      act.push_back(std::move(z));
    }
    Vec z(biases_.back().size(), 0.0);
    for (size_t j = 0; j < z.size(); ++j) {
      z[j] = biases_.back()[j];
      for (size_t i2 = 0; i2 < act.back().size(); ++i2)
        z[j] += weights_.back()[j][i2] * act.back()[i2];
    }
    softmax(z);
    proba[i] = z;
  }
  return proba;
}

}  // namespace ml
