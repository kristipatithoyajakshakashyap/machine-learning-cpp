#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Popularity recommends broadly consumed items and supplies a necessary ranking baseline. Rating prediction uses an item mean shrunk toward the global mean. These are different objectives: the most frequently rated item need not have the highest mean score.)LESSON"
      << "\n";
  rec_course::Popularity m;
  m.fit({{1, 1, 1, 5, 1}, {2, 2, 2, 4, 2}});
  for (const auto &r : rec_course::recommend(m, 1))
    std::cout << "item " << r.first << " score " << r.second << "\n";
}
