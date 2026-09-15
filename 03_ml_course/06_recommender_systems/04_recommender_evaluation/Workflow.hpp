// 03_ml_course/06_recommender_systems/04_recommender_evaluation/Workflow.hpp
//
// Purpose : the shared end-to-end project for every recommender module.
//           Provides top-k recommendation, Recall@10 / NDCG@10 ranking
//           evaluation, chronological splitting and the workflow<Model>()
//           template that trains, selects, evaluates and persists a model
//           (or, with --model, serves recommendations for one user).
// Inputs  : path - MovieLens u.data (built from COURSE_ROOT by each
//           04_end_to_end.cpp); root - RUN_OUTPUT_DIR of the calling target.
//           CLI: [--quick] | --model MODEL_FILE --user USER_ID.
// Outputs : training run: <root>/full/ (or quick/) with data/, eda/,
//           validation/, evaluation/, model/model.txt, inference/,
//           run_manifest.json, report.md, execution.log.
//           Inference run: <root>/recommendations.csv.
// Used by : rec_*_end_to_end and rec_*_predict (predict.cpp includes
//           04_end_to_end.cpp so both share this code), 03_implementation
//           lessons and every rec_*_tests fixture.
// Contract: Model must satisfy the interface documented in
//           01_popularity_baseline/Model.hpp (Popularity, ItemCF and
//           MatrixFactorization all do).
#pragma once
#include "06_recommender_systems/02_item_based_collaborative_filtering/Model.hpp"
#include "06_recommender_systems/03_matrix_factorization/Model.hpp"
#include "helper/math/metrics.hpp"
#include "helper/reporting/artifacts.hpp"
#include <iostream>
#include <map>
#include <set>
#include <sstream>
namespace rec_course {
// Top-k recommendations for one user.
// Parameters: model - fitted recommender; user - original user id; k - list
// length (default 10). Candidates are every catalog item the user has not
// rated in training. Returns (item, rank_score) pairs sorted by score
// descending, item id ascending on ties, truncated to k (fewer if the
// catalog is small). Complexity: O(catalog * cost of rank_score + n log k).
template <class Model>
std::vector<std::pair<size_t, double>> recommend(const Model &model,
                                                 size_t user, size_t k = 10) {
  std::set<size_t> seen;
  if (user < model.history.size())
    for (const auto &r : model.history[user])
      seen.insert(r.first);
  std::vector<std::pair<size_t, double>> scores;
  for (size_t item : model.catalog())
    if (!seen.count(item))
      scores.push_back({item, model.rank_score(user, item)});
  k = std::min(k, scores.size());
  std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                    [](const auto &a, const auto &b) {
                      return a.second != b.second ? a.second > b.second
                                                  : a.first < b.first;
                    });
  scores.resize(k);
  return scores;
}
// Aggregate ranking metrics over evaluated users. recall/ndcg are means of
// the per-user Recall@10 / NDCG@10; standard_error is the standard error of
// the mean recall across users; rows and recommendations hold ready-to-write
// CSV text (per-user metrics and the full top-10 lists).
struct Ranking {
  double recall = 0, ndcg = 0, standard_error = 0;
  size_t users = 0;
  std::string rows, recommendations;
};
// Evaluate top-10 ranking quality on held-out ratings.
// Parameters: model - fitted on earlier data only; test - later ratings.
// A test item is "relevant" for a user when rated >= 4 and not seen in
// training. Users with no relevant item are skipped. Unseen (cold) items
// stay in the denominator, so catalog cold-start lowers recall honestly.
// Returns the Ranking summary. Complexity: O(users * recommend()).
template <class Model>
Ranking ranking(const Model &model, const std::vector<Rating> &test) {
  std::map<size_t, std::set<size_t>> relevant;
  for (const auto &r : test)
    if (r.value >= 4 && !model.seen(r.user, r.item))
      relevant[r.user].insert(r.item);
  Ranking result;
  std::vector<double> recalls;
  std::ostringstream rows, recs;
  rows << "user,relevant_count,hits_at_10,recall_at_10,ndcg_at_10\n";
  recs << "user,rank,item,score\n";
  for (const auto &u : relevant) {
    auto selected = recommend(model, u.first);
    double hits = 0, dcg = 0, idcg = 0;
    // DCG uses binary gains with the standard 1 / log2(rank + 1) discount
    // (rank is 1-based, hence rank + 2 for the 0-based loop index).
    for (size_t rank = 0; rank < selected.size(); ++rank) {
      recs << u.first << ',' << rank + 1 << ',' << selected[rank].first << ','
           << selected[rank].second << '\n';
      if (u.second.count(selected[rank].first)) {
        ++hits;
        dcg += 1 / std::log2(rank + 2.0);
      }
    }
    // Ideal DCG: all relevant items (at most 10) at the top of the list.
    for (size_t rank = 0; rank < std::min(size_t(10), u.second.size()); ++rank)
      idcg += 1 / std::log2(rank + 2.0);
    double recall = hits / u.second.size(), ndcg = idcg > 0 ? dcg / idcg : 0;
    recalls.push_back(recall);
    result.recall += recall;
    result.ndcg += ndcg;
    rows << u.first << ',' << u.second.size() << ',' << hits << ',' << recall
         << ',' << ndcg << '\n';
  }
  result.users = recalls.size();
  if (result.users) {
    result.recall /= result.users;
    result.ndcg /= result.users;
    // Sample standard error of the mean recall (needs >= 2 users).
    double ss = 0;
    for (double v : recalls)
      ss += (v - result.recall) * (v - result.recall);
    if (result.users > 1)
      result.standard_error = std::sqrt(ss / (result.users - 1) / result.users);
  }
  result.rows = rows.str();
  result.recommendations = recs.str();
  return result;
}
// Index at which to cut chronologically sorted rows so that roughly
// `fraction` of them fall before the cut and no timestamp is split across
// the boundary (the cut moves forward past equal timestamps).
// Throws std::runtime_error if that leaves either side empty.
inline size_t chronological_boundary(const std::vector<Rating> &rows,
                                     double fraction) {
  size_t cut = static_cast<size_t>(rows.size() * fraction);
  while (cut < rows.size() && cut > 0 &&
         rows[cut].timestamp == rows[cut - 1].timestamp)
    ++cut;
  if (!cut || cut >= rows.size())
    throw std::runtime_error("Cannot create timestamp-disjoint split");
  return cut;
}
// Run the complete recommender project (or serve one user's list).
// Parameters:
//   root       - results directory for this executable (RUN_OUTPUT_DIR).
//   path       - MovieLens u.data file.
//   candidates - values of the model's single hyper-parameter to compare.
//   argc/argv  - command line; see file header for the accepted flags.
// Returns 0 on success, 1 on any exception (message printed to stderr).
// Protocol: sort by time -> optional quick truncation to the first 12000
// rows -> 60/20/20 chronological train/validation/test split -> pick the
// candidate with the lowest validation RMSE -> refit on train+validation
// -> evaluate rating error and top-10 ranking on the test period against
// a Popularity baseline -> archive the model and verify a reload.
template <class Model>
int workflow(const std::string &root, const std::string &path,
             const ml::Vec &candidates, int argc, char **argv) {
  try {
    // ---- command line -----------------------------------------------------
    bool quick = false;
    std::string model_path;
    size_t user = 1;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--quick")
        quick = true;
      else if (arg == "--model" && i + 1 < argc)
        model_path = argv[++i];
      else if (arg == "--user" && i + 1 < argc)
        user = std::stoul(argv[++i]);
      else
        throw std::invalid_argument(
            "Use --quick or --model MODEL_FILE --user USER_ID");
    }
    // ---- inference mode ---------------------------------------------------
    // Restores the archive and writes <root>/recommendations.csv holding the
    // top-10 unseen items for --user (default user 1).
    if (!model_path.empty()) {
      Model model;
      ml::archive::load_file(model_path, model);
      std::ostringstream csv;
      csv << "user,rank,item,score\n";
      size_t rank = 0;
      for (const auto &r : recommend(model, user))
        csv << user << ',' << ++rank << ',' << r.first << ',' << r.second
            << '\n';
      ml::Artifacts(root, ".").write("recommendations.csv", csv.str());
      return 0;
    }
    // ---- data and chronological split -------------------------------------
    // --quick keeps only the earliest 12000 ratings (data is time-sorted), a
    // deterministic subset that is still large enough to exercise every step.
    auto data = read_ratings(path);
    if (quick && data.size() > 12000)
      data.resize(12000);
    // train = first ~60%, validation = next ~20%, test = last ~20%.
    // "development" (train + validation) is what the final model sees.
    const size_t train_end = chronological_boundary(data, 0.6),
                 test_start = chronological_boundary(data, 0.8);
    if (train_end >= test_start)
      throw std::runtime_error("Validation partition empty");
    std::vector<Rating> train(data.begin(), data.begin() + train_end),
        valid(data.begin() + train_end, data.begin() + test_start),
        development(data.begin(), data.begin() + test_start),
        test(data.begin() + test_start, data.end());
    ml::Artifacts a(root, quick ? "quick" : "full");
    // ---- model selection on validation RMSE -------------------------------
    // Strict '<' keeps the first candidate on ties (deterministic choice).
    std::ostringstream scores;
    scores << "parameter,validation_rmse\n";
    double chosen = candidates.front(), best = 1e300;
    for (double candidate : candidates) {
      Model model(candidate);
      model.fit(train);
      ml::Vec actual, pred;
      for (const auto &r : valid) {
        actual.push_back(r.value);
        pred.push_back(model.predict(r.user, r.item));
      }
      double error = ml::rmse(actual, pred);
      scores << candidate << ',' << error << '\n';
      if (error < best) {
        best = error;
        chosen = candidate;
      }
    }
    a.write("validation/candidate_scores.csv", scores.str());
    // ---- final fit on train + validation, single test evaluation ----------
    // The Popularity baseline is fitted on the same rows so both rating error
    // and ranking metrics have a like-for-like reference.
    Model model(chosen);
    model.fit(development);
    Popularity baseline;
    baseline.fit(development);
    ml::Vec actual, pred, base;
    // Cold rows: test user/item absent from the final training data.
    size_t cold_users = 0, cold_items = 0;
    std::ostringstream predictions;
    predictions << "source_row,user,item,timestamp,actual,prediction,"
                   "popularity_prediction\n";
    for (const auto &r : test) {
      double estimate = model.predict(r.user, r.item),
             reference = baseline.predict(r.user, r.item);
      actual.push_back(r.value);
      pred.push_back(estimate);
      base.push_back(reference);
      cold_users +=
          r.user >= model.history.size() || model.history[r.user].empty();
      cold_items += r.item >= model.count.size() || model.count[r.item] == 0;
      predictions << r.id << ',' << r.user << ',' << r.item << ','
                  << r.timestamp << ',' << r.value << ',' << estimate << ','
                  << reference << '\n';
    }
    a.write("evaluation/predictions.csv", predictions.str());
    auto ranked = ranking(model, test), ranked_base = ranking(baseline, test);
    a.write("evaluation/user_ranking_metrics.csv", ranked.rows);
    a.write("evaluation/recommendations.csv", ranked.recommendations);
    a.write("evaluation/popularity_user_metrics.csv", ranked_base.rows);
    std::ostringstream metrics;
    metrics << "{\"rmse\":" << ml::rmse(actual, pred)
            << ",\"mae\":" << ml::mae(actual, pred)
            << ",\"popularity_rmse\":" << ml::rmse(actual, base)
            << ",\"recall_at_10\":" << ranked.recall
            << ",\"ndcg_at_10\":" << ranked.ndcg
            << ",\"recall_standard_error\":" << ranked.standard_error
            << ",\"popularity_recall_at_10\":" << ranked_base.recall
            << ",\"popularity_ndcg_at_10\":" << ranked_base.ndcg
            << ",\"ranking_users\":" << ranked.users
            << ",\"cold_user_rows\":" << cold_users
            << ",\"cold_item_rows\":" << cold_items << "}\n";
    a.write("evaluation/metrics.json", metrics.str());
    // ---- EDA, split log and quality report --------------------------------
    // The EDA dataset describes final training interactions with three dense
    // descriptive columns; it is not what the recommenders are trained on.
    ml::Dataset description;
    description.feature_names = {"timestamp", "user_training_ratings",
                                 "item_training_ratings"};
    description.target_name = "rating";
    for (const auto &r : development) {
      description.X.push_back({double(r.timestamp),
                               double(model.history[r.user].size()),
                               model.count[r.item]});
      description.y.push_back(r.value);
    }
    ml::eda(description, a);
    std::ostringstream split;
    split << "source_row,timestamp,partition\n";
    for (size_t i = 0; i < data.size(); ++i)
      split << data[i].id << ',' << data[i].timestamp << ','
            << (i < train_end    ? "train"
                : i < test_start ? "validation"
                                 : "test")
            << '\n';
    a.write("data/row_assignments.csv", split.str());
    std::set<size_t> users, items;
    for (const auto &r : development) {
      users.insert(r.user);
      items.insert(r.item);
    }
    std::ostringstream quality;
    quality << "training_ratings,users,items,density\n"
            << development.size() << ',' << users.size() << ',' << items.size()
            << ',' << double(development.size()) / (users.size() * items.size())
            << '\n';
    a.write("data/quality_report.csv", quality.str());
    ml::Plot compare;
    compare.bar({ml::rmse(actual, pred), ml::rmse(actual, base)},
                {"selected", "popularity"});
    compare.title("Held-out rating RMSE");
    a.figure("evaluation/figures/rating_errors.svg", compare);
    ml::Plot rank_plot;
    rank_plot.bar({ranked.ndcg, ranked_base.ndcg}, {"selected", "popularity"});
    rank_plot.title("Held-out NDCG@10");
    a.figure("evaluation/figures/ranking.svg", rank_plot);
    // ---- persistence and reload verification ------------------------------
    // Reload into a fresh object and require identical predictions on the
    // first 100 test rows (tolerance 1e-12) plus an identical top-10 list.
    ml::archive::save_file(a.path("model/model.txt"), model);
    Model restored;
    ml::archive::load_file(a.path("model/model.txt"), restored);
    for (size_t i = 0; i < std::min(size_t(100), test.size()); ++i)
      if (std::abs(restored.predict(test[i].user, test[i].item) - pred[i]) >
          1e-12)
        throw std::runtime_error("Recommender reload mismatch");
    if (recommend(restored, test.front().user) !=
        recommend(model, test.front().user))
      throw std::runtime_error("Reload ranking mismatch");
    a.write("inference/reload_verification.json", "{\"passed\":true}\n");
    // ---- manifest, schema, report, log ------------------------------------
    std::ostringstream manifest;
    manifest
        << "{\"dataset\":\"MovieLens 100K\",\"seed\":42,\"selected_parameter\":"
        << chosen << ",\"train_end_timestamp\":" << train.back().timestamp
        << ",\"validation_end_timestamp\":" << valid.back().timestamp
        << ",\"test_start_timestamp\":" << test.front().timestamp
        << ",\"training_fingerprint\":\"" << ml::fingerprint(description)
        << "\"}\n";
    a.write("run_manifest.json", manifest.str());
    a.write("model/feature_schema.json",
            "{\"input\":[\"original_user_id\",\"original_item_id\"],\"rating_"
            "range\":[1,5],\"relevant_rating_minimum\":4,\"ranking_k\":10}\n");
    std::ostringstream report;
    report << "# MovieLens recommendation\n\nHeld-out RMSE: "
           << ml::rmse(actual, pred) << "; Recall@10: " << ranked.recall
           << "; NDCG@10: " << ranked.ndcg
           << ". Popularity baseline NDCG@10: " << ranked_base.ndcg
           << ".\n\nGlobal chronological splits use approximately 60/20/20 "
              "percent with equal timestamps kept together. Validation RMSE "
              "chooses the parameter; final fit uses the first 80 percent. No "
              "future interactions enter similarities, biases, factors or item "
              "popularity. Ranking evaluates all users with at least one "
              "unseen test rating >=4; candidates are every training-catalog "
              "item not previously seen. Unseen test items remain in relevance "
              "denominators, so catalog cold-start costs are visible. Missing "
              "ratings are unknown, not verified dislikes; offline ranking is "
              "subject to exposure bias. Models fall back to item/global or "
              "known-user bias estimates for cold starts. The reported recall "
              "standard error summarizes variation across users and does not "
              "model temporal dependence. EDA describes final training "
              "interactions; user/item history counts are descriptive "
              "aggregates, not chronological prediction features.\n";
    a.write("report.md", report.str());
    a.write("execution.log",
            "Chronological selection, full-catalog ranking, rating evaluation "
            "and archive reload completed.\n");
    std::cout << metrics.str();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
} // namespace rec_course
