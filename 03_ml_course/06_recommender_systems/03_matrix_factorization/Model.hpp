// 03_ml_course/06_recommender_systems/03_matrix_factorization/Model.hpp
//
// Purpose : biased matrix factorization (Funk-SVD style) trained with
//           stochastic gradient descent. A rating is modelled as
//           mean + user_bias + item_bias + <user_factors, item_factors>.
// Inputs  : training ratings via fit(); no files are read here.
// Outputs : none. This header writes no files; workflows write results.
// Used by : rec_mf_* and rec_evaluation_* targets and every rec_* test
//           through 04_recommender_evaluation/Workflow.hpp (no target).
// Contract: satisfies the recommender model contract documented in
//           01_popularity_baseline/Model.hpp; the tunable parameter is the
//           L2 regularization strength.
#pragma once
#include "06_recommender_systems/01_popularity_baseline/Model.hpp"
#include <numeric>
#include <random>
namespace rec_course {
// Latent-factor recommender. Inherits the Popularity statistics (used for
// cold-start detection through history/count) and adds biases and factors.
// Invariants after fit(): users/user_bias are indexed by user id and
// items/item_bias by item id; every factor row has `dimensions` entries.
class MatrixFactorization : public Popularity {
public:
  double regularization;
  size_t dimensions = 16, epochs = 30;
  std::vector<double> user_bias, item_bias;
  std::vector<std::vector<double>> users, items;
  // Parameters: reg - L2 penalty on biases and factors (>= 0).
  // Throws std::invalid_argument on a negative value.
  explicit MatrixFactorization(double reg = 0.05) : regularization(reg) {
    if (reg < 0)
      throw std::invalid_argument("Negative regularization");
  }
  // Unclamped score. Bias and factor terms are only added for users/items
  // that appeared in training, so a cold user gets mean + item_bias and a
  // cold pair degrades gracefully to the global mean. O(dimensions).
  double raw(size_t user, size_t item) const {
    double score = mean;
    bool known_user = user < users.size() && !history[user].empty(),
         known_item = item < items.size() && count[item] > 0;
    if (known_user)
      score += user_bias[user];
    if (known_item)
      score += item_bias[item];
    if (known_user && known_item)
      for (size_t d = 0; d < dimensions; ++d)
        score += users[user][d] * items[item][d];
    return score;
  }
  // Fit the baseline statistics, then run `epochs` passes of SGD.
  // Seed 42 fixes both the N(0, 0.1) factor initialisation and the per-epoch
  // shuffle, so repeated runs give identical models. The learning rate
  // decays as 0.01 / (1 + 0.03 * epoch). Each update is the gradient step
  // of the squared error plus the L2 penalty for the touched parameters.
  // Complexity: O(epochs * rows * dimensions).
  void fit(const std::vector<Rating> &rows) {
    Popularity::fit(rows);
    std::mt19937 rng(42);
    std::normal_distribution<double> init(0, 0.1);
    users.assign(history.size(), std::vector<double>(dimensions));
    items.assign(count.size(), std::vector<double>(dimensions));
    user_bias.assign(users.size(), 0);
    item_bias.assign(items.size(), 0);
    for (auto &r : users)
      for (double &v : r)
        v = init(rng);
    for (auto &r : items)
      for (double &v : r)
        v = init(rng);
    std::vector<size_t> order(rows.size());
    std::iota(order.begin(), order.end(), 0);
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
      std::shuffle(order.begin(), order.end(), rng);
      double lr = 0.01 / (1 + 0.03 * epoch);
      for (size_t idx : order) {
        const auto &r = rows[idx];
        double err = r.value - raw(r.user, r.item);
        user_bias[r.user] += lr * (err - regularization * user_bias[r.user]);
        item_bias[r.item] += lr * (err - regularization * item_bias[r.item]);
        // Read both factors before updating either so the item step uses the
        // pre-update user factor (simultaneous update).
        for (size_t d = 0; d < dimensions; ++d) {
          double u = users[r.user][d], v = items[r.item][d];
          users[r.user][d] += lr * (err * v - regularization * u);
          items[r.item][d] += lr * (err * u - regularization * v);
        }
      }
    }
  }
  // Rating estimate clamped to the valid [1, 5] range.
  double predict(size_t user, size_t item) const {
    return std::clamp(raw(user, item), 1.0, 5.0);
  }
  // Ranking uses the unclamped score so items that saturate at 5 can still
  // be ordered among themselves.
  double rank_score(size_t user, size_t item) const { return raw(user, item); }
  // Archive: own tag, the Popularity block, then hyper-parameters, biases
  // and factor matrices.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("MatrixFactorization_V1"));
    Popularity::save(out);
    ml::archive::write(out, regularization, dimensions, epochs, user_bias,
                       item_bias, users, items);
  }
  // Inverse of save(): tag check, base block, then the same member order.
  // Throws std::runtime_error if the tag belongs to another model type.
  void load(std::istream &in) {
    std::string model_type;
    ml::archive::read(in, model_type);
    if (model_type != std::string("MatrixFactorization_V1"))
      throw std::runtime_error("Wrong model type/version");
    Popularity::load(in);
    ml::archive::read(in, regularization, dimensions, epochs, user_bias,
                      item_bias, users, items);
  }
};
} // namespace rec_course
