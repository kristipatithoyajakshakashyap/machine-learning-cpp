// 03_ml_course/06_recommender_systems/02_item_based_collaborative_filtering/03_implementation.cpp
//
// Purpose : implementation lesson for item-based CF. Fits ItemCF on the
//           whole MovieLens file with an uncapped neighbourhood, extracts
//           the similarity block of the 12 most-rated items and prints
//           sample top-10 lists for four users.
// Inputs  : <COURSE_ROOT>/06_recommender_systems/data/u.data
//           (COURSE_ROOT and RUN_OUTPUT_DIR are compile definitions from
//           add_lesson_executable in the root CMakeLists.txt).
// Outputs : results/03_implementation_results/
//             item_similarity_top12.csv   - 12x12 adjusted-cosine matrix
//             item_similarity_top12.svg   - heatmap of the same matrix
//             sample_recommendations.csv  - top-10 for users 1, 13, 100, 405
// Target  : rec_itemcf_implementation
#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
#include <iostream>

// Lesson: item-based collaborative filtering on MovieLens 100K.
// Artifacts: item-item similarity among the 12 most-rated items (CSV +
// heatmap SVG) and sample top-10 recommendations for a few users (CSV).
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
    // Keep the full neighbour list so similarities between popular items are
    // available even when they are not in each other's top-30.
    rec_course::ItemCF cf(2000);
    cf.fit(data);
    std::vector<size_t> items = cf.catalog();
    const size_t top = std::min<size_t>(12, items.size());
    std::partial_sort(items.begin(), items.begin() + top, items.end(),
                      [&](size_t x, size_t y) {
                        return cf.count[x] != cf.count[y]
                                   ? cf.count[x] > cf.count[y]
                                   : x < y;
                      });
    items.resize(top);
    // Dense 12x12 block: diagonal 1, off-diagonal from the neighbour lists.
    // Pairs with non-positive similarity are absent from nearest, so they
    // stay 0.
    ml::Mat sim(top, ml::Vec(top, 0.0));
    for (size_t r = 0; r < top; ++r) {
      sim[r][r] = 1.0;
      for (const auto &n : cf.nearest[items[r]])
        for (size_t c = 0; c < top; ++c)
          if (items[c] == n.first)
            sim[r][c] = n.second;
    }
    std::ostringstream csv;
    std::vector<std::string> labels;
    csv << "item";
    for (size_t i : items) {
      csv << ',' << i;
      labels.push_back(std::to_string(i));
    }
    csv << '\n';
    for (size_t r = 0; r < top; ++r) {
      csv << items[r];
      for (size_t c = 0; c < top; ++c)
        csv << ',' << sim[r][c];
      csv << '\n';
    }
    // Artifacts(root, ".") writes directly into RUN_OUTPUT_DIR (no sub-run).
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("item_similarity_top12.csv", csv.str());
    ml::Plot heat(640, 560);
    heat.heatmap(sim, labels, labels, -1.0, 1.0);
    heat.title("Item-item similarity (12 most-rated items)");
    heat.xlabel("item id");
    heat.ylabel("item id");
    a.figure("item_similarity_top12.svg", heat);
    std::ostringstream recs;
    recs << "user,rank,item,predicted_rating\n";
    for (size_t user : {1, 13, 100, 405}) {
      size_t rank = 0;
      for (const auto &r : rec_course::recommend(cf, user))
        recs << user << ',' << ++rank << ',' << r.first << ',' << r.second
             << '\n';
    }
    a.write("sample_recommendations.csv", recs.str());
    std::cout << "items with at least one positive neighbour: ";
    size_t connected = 0;
    for (const auto &row : cf.nearest)
      connected += !row.empty();
    std::cout << connected << "\n";
    for (const auto &f : {"item_similarity_top12.csv",
                          "item_similarity_top12.svg",
                          "sample_recommendations.csv"})
      std::cout << "wrote " << a.path(f) << "\n";
  } catch (const std::exception &e) {
    std::cerr << "artifact error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
