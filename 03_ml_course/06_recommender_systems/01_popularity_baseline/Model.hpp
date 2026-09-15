// 03_ml_course/06_recommender_systems/01_popularity_baseline/Model.hpp
//
// Purpose : the MovieLens rating record, its loader, and the Popularity
//           baseline recommender. Popularity is also the base class of
//           ItemCF and MatrixFactorization, so the members declared here
//           (mean, count, item_mean, user_mean, history) form the shared
//           state every recommender exposes to Workflow.hpp.
// Inputs  : read_ratings(path) reads MovieLens "u.data" (whitespace-separated
//           "user item rating timestamp" rows). Lessons build the path from
//           COURSE_ROOT: <COURSE_ROOT>/06_recommender_systems/data/u.data.
// Outputs : none. This header writes no files; workflows write results.
// Used by : every rec_* lesson, predict and test target through
//           04_recommender_evaluation/Workflow.hpp (no dedicated target).
//
// Model contract required by rec_course::workflow / recommend / ranking:
//   explicit Model(double parameter);        // single tunable hyper-parameter
//   void fit(const std::vector<Rating> &);   // learn from training rows only
//   double predict(size_t user, size_t item) const;    // rating in [1, 5]
//   double rank_score(size_t user, size_t item) const; // higher = better
//   bool seen(size_t user, size_t item) const;
//   std::vector<size_t> catalog() const;     // items with >= 1 training rating
//   members history and count (inherited from Popularity)
//   void save(std::ostream &) const;  void load(std::istream &);
#pragma once
#include "helper/persistence/archive.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>
namespace rec_course {
// One explicit rating. user/item are the original 1-based MovieLens ids
// (0 is never a valid id and is used as "unknown"); id is the 1-based row
// number in the source file so results can be traced back; value is the
// star rating in [1, 5]; timestamp is seconds since the epoch and drives the
// chronological splits.
struct Rating {
  size_t user = 0, item = 0, id = 0;
  double value = 0;
  long long timestamp = 0;
};
// Load MovieLens ratings and sort them chronologically.
// Parameters: path - u.data style file, four whitespace-separated columns.
// Returns rows sorted by timestamp (stable, so file order breaks ties).
// Throws std::runtime_error on an unreadable file, a zero user/item id, a
// rating outside [1, 5], trailing garbage or an empty file.
// Complexity: O(n log n) for the sort.
inline std::vector<Rating> read_ratings(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot read MovieLens ratings");
  std::vector<Rating> rows;
  Rating r;
  while (in >> r.user >> r.item >> r.value >> r.timestamp) {
    r.id = rows.size() + 1;
    if (!r.user || !r.item || r.value < 1 || r.value > 5)
      throw std::runtime_error("Invalid rating");
    rows.push_back(r);
  }
  // Reaching EOF is the only acceptable way for the loop to stop.
  if (!in.eof() || rows.empty())
    throw std::runtime_error("Malformed rating file");
  // Stable sort keeps the original order among equal timestamps, which makes
  // the chronological boundaries in Workflow.hpp deterministic.
  std::stable_sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
    return a.timestamp < b.timestamp;
  });
  return rows;
}
// Non-personalised baseline: predicts a shrunken per-item mean rating and
// ranks items by how many training ratings they received.
// Invariants after fit(): count/item_mean are indexed by item id (size
// max_item + 1, index 0 unused); user_mean/history are indexed by user id.
// The constructor parameter is ignored; it exists only so that
// workflow<Popularity>() can pass a candidate value like for other models.
class Popularity {
public:
  double mean = 0;
  std::vector<double> count, item_mean, user_mean;
  std::vector<std::vector<std::pair<size_t, double>>> history;
  explicit Popularity(double unused = 0) { (void)unused; }
  // Learn global mean, per-item counts and means, per-user means and the
  // per-user rating history from training rows.
  // Parameters: rows - training ratings (must be non-empty).
  // Throws std::invalid_argument on empty input. Complexity: O(rows).
  void fit(const std::vector<Rating> &rows) {
    if (rows.empty())
      throw std::invalid_argument("Empty recommender data");
    size_t users = 0, items = 0;
    mean = 0;
    for (const auto &r : rows) {
      users = std::max(users, r.user);
      items = std::max(items, r.item);
      mean += r.value / rows.size();
    }
    count.assign(items + 1, 0);
    item_mean.assign(items + 1, 0);
    user_mean.assign(users + 1, 0);
    history.assign(users + 1, {});
    for (const auto &r : rows) {
      ++count[r.item];
      item_mean[r.item] += r.value;
      user_mean[r.user] += r.value;
      history[r.user].push_back({r.item, r.value});
    }
    // Bayesian shrinkage: an item's mean is pulled towards the global mean
    // with the weight of 10 pseudo-ratings, so rarely rated items do not get
    // extreme averages.
    for (size_t i = 1; i < count.size(); ++i)
      item_mean[i] = (item_mean[i] + 10 * mean) / (count[i] + 10);
    for (size_t u = 0; u < history.size(); ++u)
      user_mean[u] =
          history[u].empty() ? mean : user_mean[u] / history[u].size();
  }
  // Rating estimate: the item's shrunken mean, or the global mean for an
  // item never seen in training (cold item). The user is ignored.
  double predict(size_t user, size_t item) const {
    (void)user;
    return item < item_mean.size() && count[item] > 0 ? item_mean[item] : mean;
  }
  // Ranking score: number of training ratings (popularity). Unknown items
  // score 0. The user is ignored, so every user gets the same ranking
  // apart from the items they have already seen.
  double rank_score(size_t user, size_t item) const {
    (void)user;
    return item < count.size() ? count[item] : 0;
  }
  // True if the user rated the item in the training data. Linear scan of the
  // user's history, which is short for MovieLens 100K.
  bool seen(size_t user, size_t item) const {
    if (user >= history.size())
      return false;
    for (const auto &r : history[user])
      if (r.first == item)
        return true;
    return false;
  }
  // All item ids with at least one training rating, ascending. This is the
  // candidate set that recommend() ranks.
  std::vector<size_t> catalog() const {
    std::vector<size_t> ids;
    for (size_t i = 1; i < count.size(); ++i)
      if (count[i])
        ids.push_back(i);
    return ids;
  }
  // Archive format: a type/version tag followed by every fitted member.
  // Derived classes write their own tag first and then call this.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("Popularity_V1"));
    ml::archive::write(out, mean, count, item_mean, user_mean, history);
  }
  // Reject archives of other model types before reading any state, so a
  // MatrixFactorization file cannot be loaded as a Popularity model.
  void load(std::istream &in) {
    std::string model_type;
    ml::archive::read(in, model_type);
    if (model_type != std::string("Popularity_V1"))
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, mean, count, item_mean, user_mean, history);
  }
};
} // namespace rec_course
