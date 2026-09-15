// 03_ml_course/06_recommender_systems/04_recommender_evaluation/03_implementation.cpp
//
// Purpose : implementation lesson for ranking evaluation. Trains Popularity
//           and ItemCF on the chronologically earliest 80% of MovieLens and
//           measures mean precision@k and recall@k on the last 20% for a
//           grid of k values.
// Inputs  : <COURSE_ROOT>/06_recommender_systems/data/u.data
//           (COURSE_ROOT and RUN_OUTPUT_DIR are compile definitions from
//           add_lesson_executable in the root CMakeLists.txt).
// Outputs : results/03_implementation_results/
//             precision_recall_at_k.csv  - k, both models' precision/recall
//             precision_at_k.svg         - precision@k curves
//             recall_at_k.svg            - recall@k curves
// Target  : rec_evaluation_implementation
#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
#include <iostream>

// Lesson: ranking evaluation with precision@k and recall@k.
// A chronological 80/20 split trains a popularity model and an item-CF model;
// for each k the mean precision@k / recall@k over users with at least one
// relevant (rating >= 4, unseen) test item is recorded.
// Artifacts: precision_recall_at_k.csv + line SVG.
namespace {
// Mean precision and recall at one cutoff k.
struct AtK {
  double precision = 0, recall = 0;
};
// Compute mean precision@k and recall@k for every k in `ks`.
// Parameters: model - fitted recommender; test - later ratings; ks - cutoff
// values in ascending order (the last one bounds the list length).
// Relevance follows Workflow.hpp: rating >= 4 and unseen in training. Only
// users with at least one relevant item count. precision@k divides hits by
// k, recall@k by the user's number of relevant items.
// Complexity: O(users * recommend(kmax) + users * sum(ks)).
template <class Model>
std::vector<AtK> precision_recall_at_k(
    const Model &model, const std::vector<rec_course::Rating> &test,
    const std::vector<size_t> &ks) {
  std::map<size_t, std::set<size_t>> relevant;
  for (const auto &r : test)
    if (r.value >= 4 && !model.seen(r.user, r.item))
      relevant[r.user].insert(r.item);
  std::vector<AtK> out(ks.size());
  const size_t kmax = ks.back();
  for (const auto &u : relevant) {
    // One ranked list of length kmax serves every smaller k as a prefix.
    auto ranked = rec_course::recommend(model, u.first, kmax);
    for (size_t i = 0; i < ks.size(); ++i) {
      double hits = 0;
      for (size_t r = 0; r < std::min(ks[i], ranked.size()); ++r)
        hits += u.second.count(ranked[r].first);
      out[i].precision += hits / ks[i];
      out[i].recall += hits / u.second.size();
    }
  }
  for (auto &v : out) {
    v.precision /= relevant.size();
    v.recall /= relevant.size();
  }
  return out;
}
} // namespace

int main() {
  std::cout
      << R"LESSON(Train only on earlier interactions, then rank unseen items from the training catalog.)LESSON"
      << "\n";
  // Toy demonstration of the recommend() API on the Popularity baseline.
  rec_course::Popularity m;
  m.fit({{1, 1, 1, 5, 1}, {2, 2, 2, 4, 2}});
  for (const auto &r : rec_course::recommend(m, 1))
    std::cout << "item " << r.first << " score " << r.second << "\n";

  try {
    auto data = rec_course::read_ratings(
        std::string(COURSE_ROOT) + "/06_recommender_systems/data/u.data");
    // Time-ordered 80/20 split: no test interaction predates training data.
    const size_t cut = rec_course::chronological_boundary(data, 0.8);
    std::vector<rec_course::Rating> train(data.begin(), data.begin() + cut),
        test(data.begin() + cut, data.end());
    rec_course::Popularity pop;
    pop.fit(train);
    rec_course::ItemCF cf(30);
    cf.fit(train);
    const std::vector<size_t> ks = {1, 2, 3, 5, 10, 20, 50};
    auto pop_k = precision_recall_at_k(pop, test, ks);
    auto cf_k = precision_recall_at_k(cf, test, ks);
    std::ostringstream csv;
    csv << "k,popularity_precision,popularity_recall,itemcf_precision,"
           "itemcf_recall\n";
    ml::Vec xs, pp, pr, cp, cr;
    for (size_t i = 0; i < ks.size(); ++i) {
      csv << ks[i] << ',' << pop_k[i].precision << ',' << pop_k[i].recall
          << ',' << cf_k[i].precision << ',' << cf_k[i].recall << '\n';
      xs.push_back(double(ks[i]));
      pp.push_back(pop_k[i].precision);
      pr.push_back(pop_k[i].recall);
      cp.push_back(cf_k[i].precision);
      cr.push_back(cf_k[i].recall);
    }
    // Artifacts(root, ".") writes directly into RUN_OUTPUT_DIR (no sub-run).
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("precision_recall_at_k.csv", csv.str());
    ml::Plot precision;
    precision.line(xs, pp, "popularity");
    precision.line(xs, cp, "item-CF");
    precision.title("Precision@k (chronological holdout)");
    precision.xlabel("k");
    precision.ylabel("precision@k");
    a.figure("precision_at_k.svg", precision);
    ml::Plot recall;
    recall.line(xs, pr, "popularity");
    recall.line(xs, cr, "item-CF");
    recall.title("Recall@k (chronological holdout)");
    recall.xlabel("k");
    recall.ylabel("recall@k");
    a.figure("recall_at_k.svg", recall);
    std::cout << "popularity recall@10 " << pop_k[4].recall
              << "  item-CF recall@10 " << cf_k[4].recall << "\n";
    for (const auto &f : {"precision_recall_at_k.csv", "precision_at_k.svg",
                          "recall_at_k.svg"})
      std::cout << "wrote " << a.path(f) << "\n";
  } catch (const std::exception &e) {
    std::cerr << "artifact error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
