#ifndef MLC_MODULE_LOGISTIC_HPP
#define MLC_MODULE_LOGISTIC_HPP

// Logistic regression: binary and multinomial (softmax) logistic regression
// trained by gradient descent on the cross-entropy loss.
//
// sklearn equivalents:
//   sklearn.linear_model.LogisticRegression(C=1.0, solver='lbfgs',
//                                           multi_class='multinomial')
//
// Loss for binary:
//   L = -(1/n) sum [ y_i log p_i + (1-y_i) log(1-p_i) ]  +  (l2/2) ||w||^2
//   where p_i = sigmoid(x_i . w + b)
//
// Loss for multiclass (softmax):
//   L = -(1/n) sum_k sum_i [ y_ik log p_ik ]  +  (l2/2) ||w||^2
//   where p_ik = exp(x_i . w_k + b_k) / sum_j exp(x_i . w_j + b_j)
//
// Trained via ml::adam from optim.hpp with a gradient-check call at the
// start of training for correctness assurance.

#include <cstddef>
#include <vector>

#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"

namespace ml {

class LogisticRegression {
 public:
  // n_classes: 2 for binary, >2 for multinomial softmax.
  // C: inverse regularisation strength (like sklearn); l2_lambda = 1/C.
  // lr / epochs: Adam parameters.
  LogisticRegression(size_t n_classes = 2, double C = 1.0,
                     double lr = 0.1, size_t epochs = 1000);

  void fit(const Mat& X, const Vec& y);
  Vec predict(const Mat& X) const;
  Mat predict_proba(const Mat& X) const;

  // Model accessors.
  size_t n_classes() const { return n_classes_; }
  size_t n_features() const { return weights_.empty() ? 0 : weights_[0].size(); }
  const Vec& intercept() const { return intercept_; }
  const Vec& feature_weights(size_t k) const { return weights_[k]; }

  void save(std::ostream& out) const { archive::write(out,std::string("LogisticRegression"),n_classes_,C_,lr_,epochs_,weights_,intercept_); }
  void load(std::istream& in) { std::string tag; archive::read(in,tag); if(tag!="LogisticRegression") throw std::runtime_error("Wrong model type"); archive::read(in,n_classes_,C_,lr_,epochs_,weights_,intercept_); }
 private:
  void fit_binary(const Mat& X, const Vec& y);
  void fit_multiclass(const Mat& X, const Vec& y);

  size_t n_classes_;
  double C_;
  double lr_;
  size_t epochs_;
  std::vector<Vec> weights_;  // [n_classes][n_features]
  Vec intercept_;             // [n_classes]
};

}  // namespace ml

#endif  // ML_LOGISTIC_HPP
