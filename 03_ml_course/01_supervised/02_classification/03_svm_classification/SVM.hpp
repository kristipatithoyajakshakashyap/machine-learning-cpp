#ifndef MLC_MODULE_SVM_HPP
#define MLC_MODULE_SVM_HPP

// Support Vector Machine: linear SVM (hinge loss) and kernel SVM (Platt SMO).
//
// sklearn equivalents:
//   Linear kernel:  sklearn.svm.SVC(kernel='linear')  or
//                   sklearn.linear_model.SGDClassifier(loss='hinge')
//   RBF kernel:     sklearn.svm.SVC(kernel='rbf')
//
// Linear SVM is trained by hinge-loss SGD using ml::adam from optim.hpp
// and matches sklearn SGDClassifier(hinge) or SVC(linear) closely.
//
// RBF-kernel SVM uses Platt's SMO algorithm (Sequential Minimal
// Optimisation) as described in "Working Set Selection Using Second Order
// Information" (Fan, Chen, Lin 2005).  The implementation maintains dual
// coefficients alpha_i and a bias b, with the decision function:
//
//   f(x) = sum_i  alpha_i * y_i * K(x_i, x)  +  b
//
// where K is the RBF kernel exp(-gamma * ||x - x'||^2).
// Predicted label = sign(f(x)).
//
// The RBF solver is approximate (sklearn's libsvm differs in working-set
// selection) so comparisons are tolerance-based (~1e-4 accuracy).

#include <cstddef>
#include <vector>

#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"

namespace ml {

// ---------------------------------------------------------------------------
// LinearSVM: trained by Adam-optimised hinge loss with L2 regularisation.
// ---------------------------------------------------------------------------
class LinearSVM {
 public:
  // C: regularisation parameter (like sklearn SVC(C=...)); lr, epochs for Adam.
  LinearSVM(double C = 1.0, double lr = 0.01, size_t epochs = 2000);

  // Requires finite rectangular features and both classes encoded as 0 and 1.
  // A fitted model accepts an empty prediction batch and returns an empty Vec.
  void fit(const Mat& X, const Vec& y);
  Vec predict(const Mat& X) const;
  Vec decision_function(const Mat& X) const;

  const Vec& coef() const { return w_; }
  double intercept() const { return b_; }

  void save(std::ostream& out) const {
    archive::write(out, std::string("LinearSVM"), C_, lr_, epochs_, w_, b_);
  }
  void load(std::istream& in);

 private:
  double C_;
  double lr_;
  size_t epochs_;
  Vec w_;
  double b_ = 0.0;
};

// ---------------------------------------------------------------------------
// KernelSVM: Platt SMO for binary classification with RBF kernel.
// ---------------------------------------------------------------------------
class KernelSVM {
 public:
  // Positive finite C, gamma and tol; gamma=-1 selects 1/n_features at fit.
  KernelSVM(double C = 1.0, double gamma = -1.0, double tol = 1e-3,
            size_t max_passes = 100);

  // Requires finite rectangular features and both classes encoded as 0 and 1.
  // A fitted model accepts an empty prediction batch and returns an empty Vec.
  void fit(const Mat& X, const Vec& y);
  Vec predict(const Mat& X) const;
  Vec decision_function(const Mat& X) const;

  void save(std::ostream& out) const {
    archive::write(out, std::string("KernelSVM"), C_, gamma_, tol_, max_passes_,
                   alpha_, b_, X_train_, y_train_, n_features_);
  }
  void load(std::istream& in);

 private:
  double rbf_kernel(const Vec& a, const Vec& b, size_t n_features) const;

  double C_;
  double gamma_;
  double tol_;
  size_t max_passes_;
  Vec alpha_;  // dual coefficients
  double b_ = 0.0;
  Mat X_train_;
  Vec y_train_;
  size_t n_features_ = 0;
};

}  // namespace ml

#endif  // ML_SVM_HPP
