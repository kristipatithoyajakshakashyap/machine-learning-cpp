#include "05_text_classification/Workflow.hpp"
int main(int argc, char **argv) {
  return text_course::workflow<text_course::MultinomialNB>(
      RUN_OUTPUT_DIR,
      std::string(COURSE_ROOT) +
          "/05_text_classification/data/SMSSpamCollection",
      false, {0.5, 1.0}, argc, argv);
}
