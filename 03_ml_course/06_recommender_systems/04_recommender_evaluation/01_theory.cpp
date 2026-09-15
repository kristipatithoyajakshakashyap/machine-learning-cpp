#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Rating accuracy and recommendation quality answer different questions. RMSE measures numerical rating error; Recall@10 measures the fraction of relevant items retrieved; NDCG@10 rewards high placement. Global temporal splits expose cold-start and catalog changes.)LESSON"
      << "\n";
  rec_course::Popularity m;
  m.fit({{1, 1, 1, 5, 1}, {2, 2, 2, 4, 2}});
  for (const auto &r : rec_course::recommend(m, 1))
    std::cout << "item " << r.first << " score " << r.second << "\n";
}
