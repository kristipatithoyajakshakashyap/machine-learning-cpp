#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Users who rated similar items provide indirect evidence about an unrated item. Center each user rating by that user's training mean; compute adjusted-cosine similarities between item columns. Positive neighbors can raise or lower predictions depending on how the user rated them.)LESSON"
      << "\n";
  rec_course::Popularity m;
  m.fit({{1, 1, 1, 5, 1}, {2, 2, 2, 4, 2}});
  for (const auto &r : rec_course::recommend(m, 1))
    std::cout << "item " << r.first << " score " << r.second << "\n";
}
