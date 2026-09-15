// 03_ml_course/06_recommender_systems/01_popularity_baseline/03_implementation.cpp
//
// Purpose : implementation lesson for the popularity baseline. Shows the
//           recommend() API on a two-rating toy set, then fits Popularity on
//           the whole MovieLens file and reports the 20 most-rated items.
// Inputs  : <COURSE_ROOT>/06_recommender_systems/data/u.data
//           (COURSE_ROOT and RUN_OUTPUT_DIR are compile definitions from
//           add_lesson_executable in the root CMakeLists.txt).
// Outputs : results/03_implementation_results/
//             top20_items.csv              - rank, item, count, shrunken mean
//             top20_items_by_count.svg     - bar chart of rating counts
//             top20_items_mean_rating.svg  - bar chart of shrunken means
// Target  : rec_popularity_implementation
#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
#include <iostream>
#include <numeric>

// Lesson: popularity baseline on MovieLens 100K.
// Artifacts: top-20 items by rating count with their shrunken mean rating
// (CSV + bar SVG).
int main() {
  std::cout
      << R"LESSON(Train only on earlier interactions, then rank unseen items from the training catalog.)LESSON"
      << "\n";
  // Toy demonstration: user 1 rated item 1 only, so item 2 is the single
  // unseen candidate and is recommended with its rating count as the score.
  rec_course::Popularity m;
  m.fit({{1, 1, 1, 5, 1}, {2, 2, 2, 4, 2}});
  for (const auto &r : rec_course::recommend(m, 1))
    std::cout << "item " << r.first << " score " << r.second << "\n";

  try {
    auto data = rec_course::read_ratings(
        std::string(COURSE_ROOT) + "/06_recommender_systems/data/u.data");
    rec_course::Popularity pop;
    pop.fit(data);
    // Order the catalog by rating count descending; ties broken by item id
    // so the output is deterministic across platforms.
    std::vector<size_t> items = pop.catalog();
    const size_t top = std::min<size_t>(20, items.size());
    std::partial_sort(items.begin(), items.begin() + top, items.end(),
                      [&](size_t x, size_t y) {
                        return pop.count[x] != pop.count[y]
                                   ? pop.count[x] > pop.count[y]
                                   : x < y;
                      });
    std::ostringstream csv;
    csv << "rank,item,rating_count,shrunken_mean_rating\n";
    ml::Vec counts, means;
    std::vector<std::string> labels;
    for (size_t r = 0; r < top; ++r) {
      size_t i = items[r];
      csv << r + 1 << ',' << i << ',' << pop.count[i] << ',' << pop.item_mean[i]
          << '\n';
      counts.push_back(pop.count[i]);
      means.push_back(pop.item_mean[i]);
      labels.push_back(std::to_string(i));
    }
    // Artifacts(root, ".") writes directly into RUN_OUTPUT_DIR (no sub-run).
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("top20_items.csv", csv.str());
    ml::Plot by_count(820, 400);
    by_count.bar(counts, labels);
    by_count.title("Top 20 items by rating count");
    by_count.xlabel("item id");
    by_count.ylabel("ratings");
    a.figure("top20_items_by_count.svg", by_count);
    ml::Plot by_mean(820, 400);
    by_mean.bar(means, labels);
    by_mean.title("Shrunken mean rating of the top 20 items");
    by_mean.xlabel("item id");
    by_mean.ylabel("mean rating");
    a.figure("top20_items_mean_rating.svg", by_mean);
    std::cout << "global mean " << pop.mean << "  catalog "
              << pop.catalog().size() << " items\n";
    for (const auto &f : {"top20_items.csv", "top20_items_by_count.svg",
                          "top20_items_mean_rating.svg"})
      std::cout << "wrote " << a.path(f) << "\n";
  } catch (const std::exception &e) {
    std::cerr << "artifact error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
