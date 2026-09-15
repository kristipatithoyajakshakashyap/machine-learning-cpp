#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
int main(int argc, char **argv) {
  return rec_course::workflow<rec_course::MatrixFactorization>(
      RUN_OUTPUT_DIR,
      std::string(COURSE_ROOT) + "/06_recommender_systems/data/u.data",
      {0.02, 0.1}, argc, argv);
}
