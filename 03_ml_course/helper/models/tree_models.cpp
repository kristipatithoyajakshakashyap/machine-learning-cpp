#include "helper/models/tree_models.hpp"


#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "helper/math/optim.hpp"

namespace ml {

namespace {
void validate_training(const Mat& X,const Vec& y,bool classification) {
  if(X.empty()||X[0].empty()||X.size()!=y.size())throw std::invalid_argument("Tree training requires nonempty matching data");
  for(const auto& row:X){
    if(row.size()!=X[0].size())throw std::invalid_argument("Ragged tree feature matrix");
    for(double value:row)if(!std::isfinite(value))throw std::invalid_argument("Tree features must be finite");
  }
  for(double value:y){
    if(!std::isfinite(value))throw std::invalid_argument("Tree targets must be finite");
    if(classification&&(value<0||value!=std::floor(value)||value>=static_cast<double>(std::numeric_limits<size_t>::max())))
      throw std::invalid_argument("Classification labels must be nonnegative integer codes");
  }
}
// ---------------------------------------------------------------------------
// Internal helpers for the CART split-finder.
// ---------------------------------------------------------------------------

// Node impurity:
//   regression      : MSE  = (1/n) sum (y - mean)^2
//   gini            : 1 - sum_k p_k^2
//   entropy         : - sum_k p_k log2(p_k)
// `count` is a per-class counter for classification nodes (size = n_classes)
// or {n} for regression nodes.
double impurity(TreeTask task, SplitCriterion criterion, double n,
                const std::vector<double>& count) {
  if (n <= 0.0) return 0.0;
  if (task == TreeTask::Regression) {
    // count[0] = sum(y), count[1] = sum(y^2) for regression nodes.
    double sum = count[0], sum2 = count[1];
    return sum2 / n - (sum / n) * (sum / n);
  }
  // Classification: count[k] = number of rows of class k.
  double gini = 1.0;
  double entropy = 0.0;
  for (double c : count) {
    if (c <= 0.0) continue;
    double p = c / n;
    gini -= p * p;
    entropy -= p * std::log2(p);
  }
  return criterion == SplitCriterion::Gini ? gini : entropy;
}

// Best split for a node described by `rows` (indices into the full data).
// Returns true if a split with positive improvement was found.
// Sklearn semantics:
//   - features are tried in order; candidate thresholds are midpoints of
//     consecutive DISTINCT feature values,
//   - the weighted impurity of children must be strictly better (first-max
//     wins ties, matching sklearn's strictly-improving scan),
//   - both children must hold at least min_samples_leaf rows,
//   - improvement must exceed min_impurity_decrease.
bool find_best_split(const Mat& X, const Vec& y, TreeTask task,
                     SplitCriterion criterion,
                     const std::vector<size_t>& rows,
                     size_t p, size_t n_classes, size_t min_samples_leaf,
                     double min_impurity_decrease,
                     size_t max_features, int* best_feature,
                     double* best_threshold, double* best_improvement) {
  double n = static_cast<double>(rows.size());

  // Per-node aggregators.
  std::vector<double> node_count(task == TreeTask::Regression
                                     ? 2
                                     : n_classes,
                                 0.0);
  if (task == TreeTask::Regression) {
    for (size_t i : rows) {
      node_count[0] += y[i];
      node_count[1] += y[i] * y[i];
    }
  } else {
    for (size_t i : rows) ++node_count[static_cast<size_t>(y[i])];
  }
  double parent_imp = impurity(task, criterion, n, node_count);

  // Feature candidates: either a random subset (ensemble use) or all.
  std::vector<size_t> feats(p);
  for (size_t j = 0; j < p; ++j) feats[j] = j;
  size_t feature_count = p;
  if (max_features > 0 && max_features < p) {
    // Deterministic shuffle then keep the first max_features (seeded by the
    // caller; this mirrors the randomness in sklearn's RF feature subsets).
    shuffle_indices(feats);
    feature_count = max_features;
  }

  *best_feature = -1;
  *best_threshold = 0.0;
  *best_improvement = 0.0;

  for (size_t feature_index = 0; feature_index < feature_count; ++feature_index) {
    size_t fj = feats[feature_index];
    int f = static_cast<int>(fj);
    // Sort rows by feature value.  The sort key ties row index so the scan
    // is deterministic even when values repeat.
    std::vector<size_t> order = rows;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      double va = X[a][f], vb = X[b][f];
      return va < vb || (va == vb && a < b);
    });

    // Left-side running counters as we sweep i from 0 upward:
    // left = order[0..i], right = order[i+1..end).
    std::vector<double> left_count(task == TreeTask::Regression ? 2
                                                                : n_classes,
                                   0.0);
    for (size_t i = 0; i + 1 < order.size(); ++i) {
      // Add order[i] to the left side.
      size_t row_i = order[i];
      if (task == TreeTask::Regression) {
        left_count[0] += y[row_i];
        left_count[1] += y[row_i] * y[row_i];
      } else {
        ++left_count[static_cast<size_t>(y[row_i])];
      }

      // A split is only a candidate between DISTINCT feature values, and
      // both children must respect min_samples_leaf.
      if (X[order[i]][f] == X[order[i + 1]][f]) continue;
      size_t left_size = i + 1;
      size_t right_size = order.size() - left_size;
      if (left_size < min_samples_leaf || right_size < min_samples_leaf)
        continue;

      double nl = static_cast<double>(left_size);
      double nr = static_cast<double>(right_size);
      double left_imp = impurity(task, criterion, nl, left_count);

      // Right-side aggregators = node minus left.
      std::vector<double> right_count;
      if (task == TreeTask::Regression) {
        right_count = {node_count[0] - left_count[0],
                       node_count[1] - left_count[1]};
      } else {
        right_count.resize(n_classes);
        for (size_t k = 0; k < n_classes; ++k)
          right_count[k] = node_count[k] - left_count[k];
      }
      double right_imp = impurity(task, criterion, nr, right_count);
      double improvement =
          parent_imp - (nl / n) * left_imp - (nr / n) * right_imp;

      // Strictly-better improvement keeps the FIRST split (feature order,
      // then threshold order) that matches — same as sklearn's scan.
      if (improvement > *best_improvement + 1e-15 &&
          improvement > min_impurity_decrease) {
        *best_feature = f;
        *best_threshold = (X[order[i]][f] + X[order[i + 1]][f]) / 2.0;
        *best_improvement = improvement;
      }
    }
  }
  return *best_feature >= 0;
}

// Partition `rows` into children by the chosen threshold.
void partition(const Mat& X, const std::vector<size_t>& rows, int feature,
               double threshold, std::vector<size_t>* left,
               std::vector<size_t>* right) {
  for (size_t i : rows) {
    if (X[i][feature] <= threshold)
      left->push_back(i);
    else
      right->push_back(i);
  }
}
}  // namespace

// ---------------------------------------------------------------------------
// DecisionTree
// ---------------------------------------------------------------------------

DecisionTree::DecisionTree(TreeTask task, SplitCriterion criterion,
                           size_t max_depth, size_t min_samples_leaf,
                           double min_impurity_decrease, size_t max_features)
    : task_(task), criterion_(criterion), max_depth_(max_depth),
      min_samples_leaf_(min_samples_leaf),
      min_impurity_decrease_(min_impurity_decrease),
      max_features_(max_features) {
  if(min_samples_leaf_==0||!std::isfinite(min_impurity_decrease_)||min_impurity_decrease_<0)
    throw std::invalid_argument("Invalid tree parameters");
}

// Recursively build the tree.  Returns the index of the built node.
// NOTE: recursive calls append to nodes_ and can reallocate it, so we never
// hold a TreeNode& across a recursive call — every field write re-looks-up
// the node by its stable index.
size_t DecisionTree::build(const Mat& X, const Vec& y,
                           const std::vector<size_t>& rows, size_t depth) {
  size_t p = X.empty() ? 0 : X[0].size();
  size_t node_id = nodes_.size();
  nodes_.emplace_back();
  TreeNode& node = nodes_[node_id];  // valid until the next recursive call
  node.n_samples = rows.size();

  // Leaf value / class distribution for this node (used directly if we
  // decide to stop, or as fallbacks).
  if (task_ == TreeTask::Regression) {
    double sum = 0.0;
    for (size_t i : rows) sum += y[i];
    node.value = rows.empty() ? 0.0 : sum / static_cast<double>(rows.size());
    node.proba.clear();
  } else {
    Vec counts(n_classes_, 0.0);
    for (size_t i : rows) ++counts[static_cast<size_t>(y[i])];
    node.proba = counts;
    for (double& v : node.proba)
      v /= static_cast<double>(std::max<size_t>(1, rows.size()));
    double best = node.proba[0];
    size_t best_k = 0;
    for (size_t k = 1; k < n_classes_; ++k) {
      if (node.proba[k] > best) {
        best = node.proba[k];
        best_k = k;
      }
    }
    node.value = static_cast<double>(best_k);
  }

  // Stopping rules: depth limit (root = depth 0), too few samples, or a
  // single remaining sample.
  if (max_depth_ > 0 && depth >= max_depth_) return node_id;
  if (rows.size() < 2 || rows.size() < 2 * min_samples_leaf_) return node_id;

  // Find the best split.
  int feature = -1;
  double threshold = 0.0, improvement = 0.0;
  bool found = find_best_split(X, y, task_, criterion_, rows, p, n_classes_,
                               min_samples_leaf_, min_impurity_decrease_,
                               max_features_, &feature, &threshold,
                               &improvement);
  if (!found) return node_id;

  // Split the rows and recurse.  The recursion reallocates nodes_, so the
  // `node` reference above is no longer valid from here on.
  std::vector<size_t> left, right;
  partition(X, rows, feature, threshold, &left, &right);
  size_t left_id = build(X, y, left, depth + 1);
  size_t right_id = build(X, y, right, depth + 1);

  // Re-fetch by index and wire the children in.
  TreeNode& self = nodes_[node_id];
  self.is_leaf = false;
  self.feature = feature;
  self.threshold = threshold;
  self.left = left_id;
  self.right = right_id;
  return node_id;
}

void DecisionTree::fit(const Mat& X, const Vec& y) {
  validate_training(X,y,task_==TreeTask::Classification);
  n_classes_ = 1;
  if(task_==TreeTask::Classification)
    for (double v : y) n_classes_ = std::max(n_classes_, static_cast<size_t>(v) + 1);
  nodes_.clear();
  std::vector<size_t> rows(X.size());
  for (size_t i = 0; i < X.size(); ++i) rows[i] = i;
  build(X, y, rows, 0);
}

// Follow the decision path for one test row down to a leaf.
size_t DecisionTree::traverse(const Vec& x) const {
  if(nodes_.empty())throw std::runtime_error("tree: model not fitted");
  for(double value:x)if(!std::isfinite(value))throw std::invalid_argument("Nonfinite prediction feature");
  size_t id = 0;
  while (!nodes_[id].is_leaf) {
    if(nodes_[id].feature<0||static_cast<size_t>(nodes_[id].feature)>=x.size())throw std::invalid_argument("Tree prediction feature count mismatch");
    if (x[nodes_[id].feature] <= nodes_[id].threshold)
      id = nodes_[id].left;
    else
      id = nodes_[id].right;
  }
  return id;
}

Vec DecisionTree::predict(const Mat& X) const {
  if (nodes_.empty()) throw std::runtime_error("tree: model not fitted");
  Vec out(X.size(), 0.0);
  for (size_t i = 0; i < X.size(); ++i) out[i] = nodes_[traverse(X[i])].value;
  return out;
}

Mat DecisionTree::predict_proba(const Mat& X) const {
  if (nodes_.empty()) throw std::runtime_error("tree: model not fitted");
  if (task_ != TreeTask::Classification)
    throw std::runtime_error("tree: predict_proba on a regressor");
  Mat out(X.size(), Vec(n_classes_, 0.0));
  for (size_t i = 0; i < X.size(); ++i)
    out[i] = nodes_[traverse(X[i])].proba;
  return out;
}

std::vector<size_t> DecisionTree::apply(const Mat& X) const {
  std::vector<size_t> ids(X.size());
  for (size_t i = 0; i < X.size(); ++i) ids[i] = traverse(X[i]);
  return ids;
}

void DecisionTree::set_leaf_value(size_t node_id, double value) {
  nodes_[node_id].value = value;
}

}  // namespace ml

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "helper/math/optim.hpp"

namespace ml {

namespace {
// Draw `n` bootstrap indices with replacement from [0, N).
std::vector<size_t> bootstrap(size_t n, size_t N) {
  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; ++i) idx[i] = next_rng(0, static_cast<std::uint32_t>(N - 1));
  return idx;
}

// Deterministic weighted resample: draw indices proportional to `weights`.
// New weights are re-normalised internally; used by AdaBoost to give each
// stump a sample-with-weight view of the data.
std::vector<size_t> weighted_resample(size_t n, const Vec& weights) {
  Vec cdf = weights;
  double total = 0.0;
  for (double w : cdf) total += w;
  if (total <= 0.0) return bootstrap(n, cdf.size());
  for (double& w : cdf) w /= total;
  for (size_t i = 1; i < cdf.size(); ++i) cdf[i] += cdf[i - 1];
  cdf.back() = 1.0;  // edge: round-off may leave last < 1

  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; ++i) {
    double u = next_unit();
    size_t lo = 0, hi = cdf.size() - 1;
    while (lo < hi) {  // upper_bound on the CDF
      size_t mid = (lo + hi) / 2;
      if (u < cdf[mid])
        hi = mid;
      else
        lo = mid + 1;
    }
    idx[i] = lo;
  }
  return idx;
}

// Subset the rows of a dataset into another Mat/Vec.
void gather_rows(const Mat& X, const Vec& y, const std::vector<size_t>& idx,
                 Mat* X_out, Vec* y_out) {
  X_out->clear();
  y_out->clear();
  X_out->reserve(idx.size());
  y_out->reserve(idx.size());
  for (size_t i : idx) {
    X_out->push_back(X[i]);
    y_out->push_back(y[i]);
  }
}

// Number of classes in a label vector (max+1).
size_t class_count(const Vec& y) {
  size_t c = 1;
  for (double v : y) c = std::max(c, static_cast<size_t>(v) + 1);
  if(c>y.size())throw std::invalid_argument("Class labels must be contiguous from zero");
  std::vector<bool> present(c,false);for(double v:y)present[static_cast<size_t>(v)]=true;
  for(bool found:present)if(!found)throw std::invalid_argument("Class labels must be contiguous from zero");
  return c;
}

// Heuristic used by the ensembles to tell "integer class codes" from
// "continuous target": regression values are typically non-integer or outside
// the small integer range, labels are codes in [0, 10).
[[maybe_unused]] bool classify_task(const Vec& y) {
  bool looks_class = true;
  size_t c = 0;
  for (double v : y) {
    if (v < 0.0 || v != std::floor(v) || c > 200) {
      looks_class = false;
      break;
    }
    c = std::max(c, static_cast<size_t>(v) + 1);
  }
  return looks_class && c > 1 && c <= 10;
}

// One-hot row: y_onehot[i][k] = (y[i] == k).
Mat one_hot(const Vec& y, size_t n_classes) {
  Mat out(y.size(), Vec(n_classes, 0.0));
  for (size_t i = 0; i < y.size(); ++i)
    out[i][static_cast<size_t>(y[i])] = 1.0;
  return out;
}

// Softmax over the rows of a score matrix (class probabilities).
Mat softmax_rows(const Mat& scores) {
  Mat p = scores;
  for (auto& row : p) {
    double mx = *std::max_element(row.begin(), row.end());
    double s = 0.0;
    for (double& v : row) {
      v = std::exp(v - mx);
      s += v;
    }
    for (double& v : row) v /= s;
  }
  return p;
}

// Argmax per row.
Vec argmax_rows(const Mat& scores) {
  Vec out(scores.size(), 0.0);
  for (size_t i = 0; i < scores.size(); ++i) {
    size_t best = 0;
    for (size_t k = 1; k < scores[i].size(); ++k)
      if (scores[i][k] > scores[i][best]) best = k;
    out[i] = static_cast<double>(best);
  }
  return out;
}

// Size of the random feature subset for a node: sklearn defaults are
//   classification: sqrt(p)   regression: p/3
size_t default_max_features(bool classification, size_t p) {
  if (classification)
    return std::max<size_t>(1, static_cast<size_t>(std::sqrt(p)));
  return std::max<size_t>(1, p / 3);
}
}  // namespace

// ---------------------------------------------------------------------------
// RandomForest
// ---------------------------------------------------------------------------

RandomForest::RandomForest(size_t n_trees, size_t max_features,
                           size_t max_depth, size_t min_samples_leaf)
    : n_trees_(n_trees), max_features_(max_features), max_depth_(max_depth),
      min_samples_leaf_(min_samples_leaf) {
  if(n_trees_==0||min_samples_leaf_==0)throw std::invalid_argument("Invalid forest parameters");
}

void RandomForest::fit(const Mat& X, const Vec& y) {
  validate_training(X,y,task_==TreeTask::Classification);
  size_t n = X.size();
  size_t p = X[0].size();

  classification_ = (task_ == TreeTask::Classification);
  n_classes_ = classification_ ? class_count(y) : 1;

  size_t mf = max_features_ > 0 ? max_features_
                                : default_max_features(classification_, p);

  trees_.clear();
  trees_.reserve(n_trees_);
  for (size_t t = 0; t < n_trees_; ++t) {
    // Bootstrap sample (with replacement) is this tree's training view.
    std::vector<size_t> idx = bootstrap(n, n);
    Mat Xt;
    Vec yt;
    gather_rows(X, y, idx, &Xt, &yt);
    DecisionTree tree(classification_ ? TreeTask::Classification
                                      : TreeTask::Regression,
                      SplitCriterion::Gini, max_depth_, min_samples_leaf_,
                      0.0, mf);
    tree.fit(Xt, yt);
    trees_.push_back(std::move(tree));
  }
}

Vec RandomForest::predict(const Mat& X) const {
  if (trees_.empty()) throw std::runtime_error("forest: model not fitted");
  if (classification_) return argmax_rows(predict_proba(X));
  // Regression: average the trees' predictions.
  Vec out = trees_[0].predict(X);
  for (size_t t = 1; t < trees_.size(); ++t) {
    Vec v = trees_[t].predict(X);
    for (size_t i = 0; i < X.size(); ++i) out[i] += v[i];
  }
  for (double& v : out) v /= static_cast<double>(trees_.size());
  return out;
}

Mat RandomForest::predict_proba(const Mat& X) const {
  if (trees_.empty()) throw std::runtime_error("forest: model not fitted");
  Mat proba(X.size(), Vec(n_classes_, 0.0));
  for (const auto& tree : trees_) {
    Mat p = tree.predict_proba(X);
    for (size_t i = 0; i < X.size(); ++i)
      for (size_t k = 0; k < std::min(n_classes_,p[i].size()); ++k) proba[i][k] += p[i][k];
  }
  for (auto& row : proba)
    for (double& v : row) v /= static_cast<double>(trees_.size());
  return proba;
}

// ---------------------------------------------------------------------------
// GradientBoostedTrees
// ---------------------------------------------------------------------------

GradientBoostedTrees::GradientBoostedTrees(size_t n_estimators, double lr,
                                           size_t max_depth,
                                           size_t min_samples_leaf,
                                           double subsample)
    : n_estimators_(n_estimators), lr_(lr), max_depth_(max_depth),
      min_samples_leaf_(min_samples_leaf), subsample_(subsample) {
  if(n_estimators_==0||min_samples_leaf_==0||!std::isfinite(lr_)||lr_<0||!std::isfinite(subsample_)||subsample_<=0||subsample_>1)
    throw std::invalid_argument("Invalid gradient boosting parameters");
}

void GradientBoostedTrees::fit(const Mat& X, const Vec& y) {
  validate_training(X,y,task_==TreeTask::Classification);
  size_t n = X.size();
  trees_.clear();

  // The task is configured explicitly; integer-valued regression stays regression.
  classification_ = (task_ == TreeTask::Classification);
  n_classes_ = classification_ ? class_count(y) : 1;

  // ---- Regression: squared-error boosting -----------------------------
  // F_0 = mean(y).  Each round fits a tree on the negative gradient, which
  // for squared error is simply the current residual r = y - F(x).
  // The tree's leaf value IS the mean residual in each leaf, which is the
  // optimal least-squares update.
  if (!classification_) {
    n_classes_ = 1;
    double mean_y = 0.0;
    for (double v : y) mean_y += v;
    mean_y /= static_cast<double>(n);
    init_value_ = mean_y;
    Vec F(n, mean_y);

    for (size_t t = 0; t < n_estimators_; ++t) {
      // Residuals drive the next tree.
      Vec r(n);
      for (size_t i = 0; i < n; ++i) r[i] = y[i] - F[i];

      // Optional row subsampling per round.
      Mat Xt = X;
      Vec rt = r;
      if (subsample_ < 1.0) {
        size_t k = std::max<size_t>(1, static_cast<size_t>(n * subsample_));
        std::vector<size_t> idx(n);
        for (size_t i = 0; i < n; ++i) idx[i] = i;
        shuffle_indices(idx);
        idx.resize(k);
        Xt.clear();
        rt.clear();
        for (size_t i : idx) {
          Xt.push_back(X[i]);
          rt.push_back(r[i]);
        }
      }

      DecisionTree tree(TreeTask::Regression, SplitCriterion::Gini,
                        max_depth_, min_samples_leaf_);
      tree.fit(Xt, rt);
      // Update the full-sample prediction: new F = F + lr * tree(x).
      Vec pred = tree.predict(X);
      for (size_t i = 0; i < n; ++i) F[i] += lr_ * pred[i];
      trees_.push_back(std::move(tree));
    }
    return;
  }

  // ---- Classification: multinomial log-loss boosting --------------------
  // K trees are added per round, one per class.  Scores F_k(x) start from
  // the class log-priors and are updated by lr * tree_k(x); probabilities
  // are the softmax of the scores.  Each tree is fit on the pseudo-residual
  // r_ik = y_ik - p_ik, and its leaf values are re-estimated as
  //   K * sum(r) / sum(|r| (1 - |r|))   per leaf (Friedman 2001).
  n_classes_ = class_count(y);
  classification_ = true;

  Mat Y = one_hot(y, n_classes_);
  std::vector<Vec> F(n_classes_, Vec(n, 0.0));
  init_.assign(n_classes_, 0.0);
  for (size_t k = 0; k < n_classes_; ++k) {
    size_t cnt = 0;
    for (double v : y)
      if (static_cast<size_t>(v) == k) ++cnt;
    init_[k] = std::log(static_cast<double>(cnt) / static_cast<double>(n));
    for (size_t i = 0; i < n; ++i) F[k][i] = init_[k];
  }

  for (size_t t = 0; t < n_estimators_; ++t) {
    Mat scores(n, Vec(n_classes_));
    for (size_t i = 0; i < n; ++i)
      for (size_t k = 0; k < n_classes_; ++k) scores[i][k] = F[k][i];
    Mat P = softmax_rows(scores);

    // Row subsampling round.
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    size_t ksub = n;
    if (subsample_ < 1.0) {
      ksub = std::max<size_t>(1, static_cast<size_t>(n * subsample_));
      shuffle_indices(idx);
      idx.resize(ksub);
    }

    for (size_t k = 0; k < n_classes_; ++k) {
      // Pseudo-residual: true minus predicted probability for class k.
      Vec r(n);
      for (size_t i = 0; i < n; ++i)
        r[i] = Y[i][k] - P[i][k];

      // Fit tree k on the residual over the subsample.
      Mat Xt;
      Vec rt;
      for (size_t ii : idx) {
        Xt.push_back(X[ii]);
        rt.push_back(r[ii]);
      }
      DecisionTree tree(TreeTask::Regression, SplitCriterion::Gini,
                        max_depth_, min_samples_leaf_);
      tree.fit(Xt, rt);

      // Re-estimate leaf values for classification.  sklearn (Multinomial
      // Deviance, _gb.py update_terminal_region) sets each leaf to
      //   sum(residual) / sum(p(1-p))
      // where the denominator here equals sum(|r|(1-|r|)) because the
      // residual r = y_true - p takes values in [-1, 1].
      std::vector<size_t> leaves = tree.apply(Xt);
      std::vector<double> denom(tree.node_count(), 0.0);
      std::vector<double> num(tree.node_count(), 0.0);
      for (size_t a = 0; a < idx.size(); ++a) {
        size_t leaf = leaves[a];
        size_t i = idx[a];
        num[leaf] += r[i];
        denom[leaf] += std::fabs(r[i]) * (1.0 - std::fabs(r[i]));
      }
      for (size_t leaf = 0; leaf < denom.size(); ++leaf) {
        if (denom[leaf] > 1e-12)
          tree.set_leaf_value(leaf, num[leaf] / denom[leaf]);
      }

      // Update raw scores for the full sample.
      Vec pred = tree.predict(X);
      for (size_t i = 0; i < n; ++i) F[k][i] += lr_ * pred[i];
      trees_.push_back(std::move(tree));
    }
  }
}

Vec GradientBoostedTrees::predict(const Mat& X) const {
  if (trees_.empty()) throw std::runtime_error("gbt: model not fitted");
  if (!classification_) {
    // F(x) = F0 + sum_t lr * tree_t(x).
    Vec out = trees_[0].predict(X);
    for (double& v : out) v *= lr_;
    for (size_t t = 1; t < trees_.size(); ++t) {
      Vec v = trees_[t].predict(X);
      for (size_t i = 0; i < X.size(); ++i) out[i] += lr_ * v[i];
    }
    for (double& v : out) v += init_value_;
    return out;
  }

  // Classification: scores per class = init_k + lr * sum tree_k(x).
  Mat scores(X.size(), init_);
  size_t t = 0;
  for (size_t round = 0; round < n_estimators_; ++round) {
    for (size_t k = 0; k < n_classes_; ++k, ++t) {
      Vec v = trees_[t].predict(X);
      for (size_t i = 0; i < X.size(); ++i) scores[i][k] += lr_ * v[i];
    }
  }
  return argmax_rows(scores);
}

Mat GradientBoostedTrees::predict_proba(const Mat& X) const {
  if (trees_.empty()) throw std::runtime_error("gbt: model not fitted");
  if (!classification_) return {X.size(), Vec(1, init_value_)};
  Mat scores(X.size(), init_);
  size_t t = 0;
  for (size_t round = 0; round < n_estimators_; ++round) {
    for (size_t k = 0; k < n_classes_; ++k, ++t) {
      Vec v = trees_[t].predict(X);
      for (size_t i = 0; i < X.size(); ++i) scores[i][k] += lr_ * v[i];
    }
  }
  return softmax_rows(scores);
}

// ---------------------------------------------------------------------------
// AdaBoost (SAMME)
// ---------------------------------------------------------------------------

AdaBoost::AdaBoost(size_t n_estimators, double learning_rate)
    : n_estimators_(n_estimators), learning_rate_(learning_rate) {
  if(n_estimators_==0||!std::isfinite(learning_rate_)||learning_rate_<=0)throw std::invalid_argument("Invalid AdaBoost parameters");
}

void AdaBoost::fit(const Mat& X, const Vec& y) {
  validate_training(X,y,true);
  size_t n = X.size();
  n_classes_ = class_count(y);
  if(n_classes_<2)throw std::invalid_argument("AdaBoost requires at least two classes");

  // Start with uniform sample weights.
  Vec w(n, 1.0 / static_cast<double>(n));
  stumps_.clear();
  alphas_.clear();

  for (size_t t = 0; t < n_estimators_; ++t) {
    // Weighted resample -> each stump effectively weights the data.
    std::vector<size_t> idx = weighted_resample(n, w);
    Mat Xt;
    Vec yt;
    gather_rows(X, y, idx, &Xt, &yt);

    DecisionTree stump(TreeTask::Classification, SplitCriterion::Gini, 1,
                       1, 0.0);
    stump.fit(Xt, yt);

    // Weighted error on the ORIGINAL data.
    Vec pred = stump.predict(X);
    double err = 0.0;
    for (size_t i = 0; i < n; ++i)
      if (pred[i] != y[i]) err += w[i];
    // A chance-or-worse learner is not a valid SAMME boosting step. Keep
    // earlier accepted learners, or fail clearly if no learner was useful.
    if(err >= 1.0-1.0/static_cast<double>(n_classes_)-1e-15){
      if(stumps_.empty())throw std::runtime_error("AdaBoost weak learner is no better than chance");
      break;
    }
    stumps_.push_back(std::move(stump));
    if(err<=1e-15){
      // A perfect weak learner terminates training before log(0). Its finite
      // vote dominates previous imperfect learners and preserves exact labels.
      double dominant=1;for(double a:alphas_)dominant+=std::fabs(a);
      alphas_.push_back(dominant);
      break;
    }

    // SAMME's classifier weight: ln((1-err)/err) + ln(K-1).
    double alpha = learning_rate_ *
                   (std::log((1.0 - err) / err) + std::log(static_cast<double>(n_classes_) - 1));
    alphas_.push_back(alpha);

    // Update and renormalise sample weights: misclassified weight grows.
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      if (pred[i] != y[i]) w[i] *= std::exp(alpha);
      sum += w[i];
    }
    if (sum > 0.0)
      for (double& wi : w) wi /= sum;
  }
}

Vec AdaBoost::predict(const Mat& X) const {
  if (stumps_.empty()) throw std::runtime_error("adaboost: model not fitted");
  // Class score = sum_t alpha_t * [pred_t(x) == k]; argmax the score.
  Mat scores(X.size(), Vec(n_classes_, 0.0));
  for (size_t t = 0; t < stumps_.size(); ++t) {
    Vec pred = stumps_[t].predict(X);
    for (size_t i = 0; i < X.size(); ++i)
      scores[i][static_cast<size_t>(pred[i])] += alphas_[t];
  }
  return argmax_rows(scores);
}

Mat AdaBoost::predict_proba(const Mat& X) const {
  if (stumps_.empty()) throw std::runtime_error("adaboost: model not fitted");
  // Weighted class scores normalised into probabilities.
  Mat scores(X.size(), Vec(n_classes_, 0.0));
  for (size_t t = 0; t < stumps_.size(); ++t) {
    Vec pred = stumps_[t].predict(X);
    for (size_t i = 0; i < X.size(); ++i)
      scores[i][static_cast<size_t>(pred[i])] += alphas_[t];
  }
  return softmax_rows(scores);
}

}  // namespace ml
