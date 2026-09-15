#ifndef MLCPP_CANONICAL_TREES_HPP
#define MLCPP_CANONICAL_TREES_HPP

// ===========================================================================
// 01_supervised\01_regression\04_random_forest_regressor/RandomForest.hpp  (+ the matching 01_supervised\01_regression\04_random_forest_regressor/RandomForest.cpp)
// ---------------------------------------------------------------------------
// SELF-CONTAINED module implementation: everything this module's lessons need
// lives in this one file pair, so the module folder stands alone (no shared
// treebase dependency).  It embeds its own copy of the CART decision tree
// (DecisionTree), which is also the base learner of every ensemble below.
// ===========================================================================

#include <cstddef>
#include <vector>

#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"


// RandomForest (bag of CART trees) -- regression focus.  sklearn: sklearn.ensemble.RandomForestRegressor.

namespace ml {
enum class TreeTask { Regression, Classification };
enum class SplitCriterion { Gini, Entropy };

struct TreeNode {
  bool is_leaf = true;
  int feature = -1;       // split feature index; -1 for leaves
  double threshold = 0.0; // X[feature] <= threshold goes left
  size_t left = 0;        // child index in nodes vector
  size_t right = 0;
  double value = 0.0;     // leaf value: mean (regression) or majority (classification)
  Vec proba;              // classification: [p_class0, p_class1, ...] at leaf
  size_t n_samples = 0;   // training rows reaching this node
  void save(std::ostream& out) const { archive::write(out,is_leaf,feature,threshold,left,right,value,proba,n_samples); }
  void load(std::istream& in) { archive::read(in,is_leaf,feature,threshold,left,right,value,proba,n_samples); }
};

class DecisionTree {
 public:
  // max_depth:     deepest allowed tree (0 = unlimited).
  // min_samples_leaf: each leaf must contain at least this many training rows.
  // min_impurity_decrease: a split is accepted only if it decreases weighted
  //   impurity by at least this amount (matches sklearn's min_impurity_decrease).
  // max_features:  number of features to consider per split; 0 = all (default).
  //                RF sets this to int(sqrt(p)) or int(log2(p)).
  DecisionTree(TreeTask task = TreeTask::Regression,
               SplitCriterion criterion = SplitCriterion::Gini,
               size_t max_depth = 10, size_t min_samples_leaf = 1,
               double min_impurity_decrease = 0.0, size_t max_features = 0);

  void fit(const Mat& X, const Vec& y);
  Vec predict(const Mat& X) const;
  Mat predict_proba(const Mat& X) const;  // classification only

  const std::vector<TreeNode>& nodes() const { return nodes_; }
  size_t node_count() const { return nodes_.size(); }

  // For ensembles: the index of the leaf each row lands in (useful to
  // recompute leaf values, e.g. GBDT's terminal-region estimates).
  std::vector<size_t> apply(const Mat& X) const;
  // Overwrite a leaf's stored value (used after fitting a tree on residuals).
  void set_leaf_value(size_t node_id, double value);
  std::vector<TreeNode>& mutable_nodes() { return nodes_; }

  // Accessors for ensemble use.
  TreeTask task() const { return task_; }
  SplitCriterion criterion() const { return criterion_; }
  size_t max_depth() const { return max_depth_; }
  size_t min_samples_leaf() const { return min_samples_leaf_; }
  size_t max_features() const { return max_features_; }
  double min_impurity_decrease() const { return min_impurity_decrease_; }
  size_t n_classes() const { return n_classes_; }

  void save(std::ostream& out) const { archive::write(out,std::string("DecisionTree"),task_,criterion_,max_depth_,min_samples_leaf_,min_impurity_decrease_,max_features_,n_classes_,nodes_); }
  void load(std::istream& in) { std::string tag; archive::read(in,tag); if(tag!="DecisionTree") throw std::runtime_error("Wrong model type"); archive::read(in,task_,criterion_,max_depth_,min_samples_leaf_,min_impurity_decrease_,max_features_,n_classes_,nodes_); }
 private:
  size_t build(const Mat& X, const Vec& y,
               const std::vector<size_t>& idx, size_t depth);
  size_t traverse(const Vec& x) const;

  TreeTask task_;
  SplitCriterion criterion_;
  size_t max_depth_;
  size_t min_samples_leaf_;
  double min_impurity_decrease_;
  size_t max_features_;
  size_t n_classes_ = 0;
  std::vector<TreeNode> nodes_;
};
#include <cstddef>
#include <vector>

#include "helper/math/matrix.hpp"


// ---------------------------------------------------------------------------
// RandomForest: bag of decision trees.  Each tree is trained on a bootstrap
// sample with max_features features randomly chosen per split.
//   Regression:  yhat = average of tree predictions.
//   Classification: yhat = class with highest averaged probability.
// ---------------------------------------------------------------------------
class RandomForest {
 public:
  // n_trees:          number of trees in the forest.
  // max_features:     features per split; 0 = sqrt(p) for classification,
  //                   p/3 for regression (sklearn defaults).
  // max_depth:        per-tree depth limit.
  // min_samples_leaf: per-tree leaf constraint.
  RandomForest(size_t n_trees = 100, size_t max_features = 0,
               size_t max_depth = 10, size_t min_samples_leaf = 1);

  void fit(const Mat& X, const Vec& y);
  Vec predict(const Mat& X) const;
  Mat predict_proba(const Mat& X) const;

  size_t n_trees() const { return n_trees_; }

  void set_task(TreeTask task) { task_ = task; }
  void save(std::ostream& out) const { archive::write(out,std::string("RandomForest"),n_trees_,max_features_,max_depth_,min_samples_leaf_,trees_,classification_,n_classes_,task_); }
  void load(std::istream& in) { std::string tag; archive::read(in,tag); if(tag!="RandomForest") throw std::runtime_error("Wrong model type"); archive::read(in,n_trees_,max_features_,max_depth_,min_samples_leaf_,trees_,classification_,n_classes_,task_); }
 private:
  TreeTask task_ = TreeTask::Regression;
  size_t n_trees_;
  size_t max_features_;
  size_t max_depth_;
  size_t min_samples_leaf_;
  std::vector<DecisionTree> trees_;
  bool classification_ = false;
  size_t n_classes_ = 0;
};

// ---------------------------------------------------------------------------
// GBDT: additive gradient boosted decision trees.
//
// Regression (squared error):  F_0 = mean(y); r_i = y_i - F; tree fits r.
// Classification (log-loss / multinomial):
//   K classes, K trees per iteration.  Pseudo-residual for class k:
//     r_ik = y_ik - p_ik    (y one-hot, p softmax probabilities).
//   Leaf value = K * sum(r) / sum(|r|(1-|r|)).
//   F_k += lr * tree_k(x).
// ---------------------------------------------------------------------------
class GradientBoostedTrees {
 public:
  // n_estimators: number of boosting rounds.
  // lr:           learning rate (shrinkage).
  // max_depth:    depth of each regression tree.
  // min_samples_leaf: minimum leaf size.
  // subsample:    fraction of rows used per round (1.0 = no subsampling).
  GradientBoostedTrees(size_t n_estimators = 100, double lr = 0.1,
                       size_t max_depth = 3, size_t min_samples_leaf = 1,
                       double subsample = 1.0);

  void fit(const Mat& X, const Vec& y);
  Vec predict(const Mat& X) const;
  Mat predict_proba(const Mat& X) const;

  void set_task(TreeTask task) { task_ = task; }
  void save(std::ostream& out) const { archive::write(out,std::string("GradientBoostedTrees"),n_estimators_,lr_,max_depth_,min_samples_leaf_,subsample_,classification_,n_classes_,init_value_,trees_,init_,task_); }
  void load(std::istream& in) { std::string tag; archive::read(in,tag); if(tag!="GradientBoostedTrees") throw std::runtime_error("Wrong model type"); archive::read(in,n_estimators_,lr_,max_depth_,min_samples_leaf_,subsample_,classification_,n_classes_,init_value_,trees_,init_,task_); }
 private:
  TreeTask task_ = TreeTask::Regression;
  size_t n_estimators_;
  double lr_;
  size_t max_depth_;
  size_t min_samples_leaf_;
  double subsample_;
  bool classification_ = false;
  size_t n_classes_ = 0;
  double init_value_ = 0.0;       // regression: F_0 = mean(y)
  std::vector<DecisionTree> trees_; // each "round" holds K trees (K=1 reg, K=classes)
  std::vector<double> init_;        // classification: init_[k] = log(pi_k)
};

// ---------------------------------------------------------------------------
// AdaBoost: SAMME (Stagewise Additive Modeling using a Multi-class
// Exponential loss function).
//
// Each iteration:
//   1. Fit a stump (max_depth=1) weighted by sample weights.
//   2. Weighted error = sum w_i * (pred_i != y_i) / sum w_i.
//   3. alpha = log((1-err)/err) + log(K-1).
//   4. Update weights: w_i *= exp(alpha * (pred_i != y_i)).
//   5. Renormalise weights.
// ---------------------------------------------------------------------------
class AdaBoost {
 public:
  // n_estimators: number of stumps.
  // learning_rate: weight shrinkage (applied to alpha).
  AdaBoost(size_t n_estimators = 100, double learning_rate = 1.0);

  void fit(const Mat& X, const Vec& y);
  Vec predict(const Mat& X) const;
  Mat predict_proba(const Mat& X) const;

  void save(std::ostream& out) const { archive::write(out,std::string("AdaBoost"),n_estimators_,learning_rate_,n_classes_,stumps_,alphas_); }
  void load(std::istream& in) { std::string tag; archive::read(in,tag); if(tag!="AdaBoost") throw std::runtime_error("Wrong model type"); archive::read(in,n_estimators_,learning_rate_,n_classes_,stumps_,alphas_); }
 private:
  size_t n_estimators_;
  double learning_rate_;
  size_t n_classes_ = 0;
  std::vector<DecisionTree> stumps_;
  std::vector<double> alphas_;
};
}  // namespace ml
#endif  // MLCPP_CANONICAL_TREES_HPP
