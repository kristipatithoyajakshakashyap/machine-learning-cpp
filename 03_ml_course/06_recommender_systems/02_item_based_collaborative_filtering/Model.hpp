// 03_ml_course/06_recommender_systems/02_item_based_collaborative_filtering/Model.hpp
//
// Purpose : item-based collaborative filtering (ItemCF). Extends the
//           Popularity baseline with an item-item similarity table built
//           from mean-centred user ratings and predicts a rating as the
//           similarity-weighted deviation over the user's own ratings.
// Inputs  : training ratings via fit(); no files are read here.
// Outputs : none. This header writes no files; workflows write results.
// Used by : rec_itemcf_* targets and every rec_* test through
//           04_recommender_evaluation/Workflow.hpp (no dedicated target).
// Contract: satisfies the recommender model contract documented in
//           01_popularity_baseline/Model.hpp; the tunable parameter is the
//           neighbourhood size k.
#pragma once
#include "06_recommender_systems/01_popularity_baseline/Model.hpp"
#include <unordered_map>
namespace rec_course {
// Item-item k-nearest-neighbour recommender.
// State after fit(): nearest[i] holds up to `neighbors` (item, similarity)
// pairs with similarity > 0, sorted by similarity descending then item id;
// lookup[u] maps item -> rating for user u (rebuilt from history on load so
// it is never archived).
class ItemCF : public Popularity {
public:
  size_t neighbors;
  std::vector<std::vector<std::pair<size_t, double>>> nearest;
  std::vector<std::unordered_map<size_t, double>> lookup;
  // Parameters: k - neighbourhood size (truncated to an integer, >= 1).
  // Throws std::invalid_argument when k < 1.
  explicit ItemCF(double k = 30) : neighbors(static_cast<size_t>(k)) {
    if (k < 1)
      throw std::invalid_argument("Neighbors must be positive");
  }
  // Rebuild the per-user hash lookup from history. Called after fit() and
  // load() so predict() can find a user's rating for an item in O(1).
  void rebuild() {
    lookup.clear();
    lookup.resize(history.size());
    for (size_t u = 0; u < history.size(); ++u)
      for (const auto &r : history[u])
        lookup[u][r.first] = r.second;
  }
  // Fit the baseline statistics, then the item-item similarity table.
  // Similarity is an adjusted cosine on user-mean-centred ratings, damped by
  // common / (common + 10) so pairs co-rated by few users are shrunk toward
  // zero. Only positive similarities are kept, then each list is truncated
  // to `neighbors`.
  // Complexity: O(sum over users of history^2) for the products plus
  // O(items^2) memory for the dense accumulators.
  void fit(const std::vector<Rating> &rows) {
    Popularity::fit(rows);
    const size_t n = count.size();
    std::vector<std::vector<double>> products(n, std::vector<double>(n)),
        common(n, std::vector<double>(n));
    std::vector<double> norm(n, 0);
    // Accumulate, for every pair of items rated by the same user, the dot
    // product of their centred ratings and the number of co-raters. Only
    // the upper triangle (a < b) is filled.
    for (size_t u = 0; u < history.size(); ++u)
      for (const auto &a : history[u]) {
        double ca = a.second - user_mean[u];
        norm[a.first] += ca * ca;
        for (const auto &b : history[u])
          if (a.first < b.first) {
            products[a.first][b.first] += ca * (b.second - user_mean[u]);
            ++common[a.first][b.first];
          }
      }
    nearest.assign(n, {});
    for (size_t i = 1; i < n; ++i)
      for (size_t j = i + 1; j < n; ++j)
        if (norm[i] > 0 && norm[j] > 0) {
          double sim = products[i][j] / std::sqrt(norm[i] * norm[j]) *
                       common[i][j] / (common[i][j] + 10);
          if (sim > 0) {
            nearest[i].push_back({j, sim});
            nearest[j].push_back({i, sim});
          }
        }
    // Deterministic order: similarity descending, item id ascending on ties.
    for (auto &row : nearest) {
      std::sort(row.begin(), row.end(), [](const auto &a, const auto &b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
      });
      if (row.size() > neighbors)
        row.resize(neighbors);
    }
    rebuild();
  }
  // Rating estimate with explicit cold-start fallbacks:
  //   unknown user            -> Popularity::predict (shrunken item mean)
  //   unknown item            -> the user's mean rating
  //   no rated neighbours     -> Popularity::predict
  //   otherwise user_mean + weighted mean of the user's centred neighbour
  //   ratings, clamped to the [1, 5] rating range.
  // Complexity: O(neighbors) hash lookups.
  double predict(size_t user, size_t item) const {
    if (user >= lookup.size() || history[user].empty())
      return Popularity::predict(user, item);
    if (item >= nearest.size() || !count[item])
      return user_mean[user];
    double num = 0, denom = 0;
    for (const auto &n : nearest[item]) {
      auto it = lookup[user].find(n.first);
      if (it != lookup[user].end()) {
        num += n.second * (it->second - user_mean[user]);
        denom += n.second;
      }
    }
    return std::clamp(denom > 0 ? user_mean[user] + num / denom
                                : Popularity::predict(user, item),
                      1.0, 5.0);
  }
  // Ranking uses the predicted rating directly.
  double rank_score(size_t user, size_t item) const {
    return predict(user, item);
  }
  // Archive: own tag, then the Popularity block, then k and the neighbour
  // lists. lookup is derived state and is not stored.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("ItemCF_V1"));
    Popularity::save(out);
    ml::archive::write(out, neighbors, nearest);
  }
  // Inverse of save(): checks the tag, restores base + neighbour state and
  // rebuilds the derived lookup table. Throws on a foreign archive.
  void load(std::istream &in) {
    std::string model_type;
    ml::archive::read(in, model_type);
    if (model_type != std::string("ItemCF_V1"))
      throw std::runtime_error("Wrong model type/version");
    Popularity::load(in);
    ml::archive::read(in, neighbors, nearest);
    rebuild();
  }
};
} // namespace rec_course
