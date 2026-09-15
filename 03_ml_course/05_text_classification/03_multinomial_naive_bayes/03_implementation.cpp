// 03_ml_course/05_text_classification/03_multinomial_naive_bayes/03_implementation.cpp
//
// Purpose : implementation lesson for Multinomial NB. Trains on one
//           stratified 80/20 split of the SMS corpus, lists the words with
//           the most extreme spam/ham log-odds and reports the holdout
//           confusion matrix.
// Inputs  : <COURSE_ROOT>/05_text_classification/data/SMSSpamCollection
//           (COURSE_ROOT, RUN_OUTPUT_DIR from add_lesson_executable).
// Outputs : results/03_implementation_results/
//             discriminative_words.csv / .svg, confusion_matrix.csv
// Target  : text_nb_implementation
#include "05_text_classification/Workflow.hpp"
#include <iostream>
#include <numeric>

// Lesson: multinomial naive Bayes on SMS spam.
// Artifacts: per-class log-probabilities of the most discriminative words
// (CSV + bar SVG of the log-odds) and a confusion matrix on a holdout split.
int main() {
  std::cout
      << R"LESSON(Fit on training text, then transform a new message. Unknown tokens contribute no feature.)LESSON"
      << "\n";
  // Toy demonstration of the count vectorizer (see 01_text_preprocessing).
  text_course::Vectorizer v(false);
  v.fit({"win win now", "hello friend"});
  auto encoded = v.transform({"win unknown"});
  for (const auto &term : encoded[0])
    std::cout << v.vocabulary[term.first] << "=" << term.second << "\n";

  try {
    const std::string data = std::string(COURSE_ROOT) +
                             "/05_text_classification/data/SMSSpamCollection";
    auto corpus = text_course::read_sms(data);
    // First fold of a seeded (42) stratified 5-fold split = ~80/20 holdout
    // that keeps the spam proportion in both parts.
    auto split = ml::stratified_kfold(corpus.y, 5, 42).front();
    auto train = text_course::select(corpus, split.train),
         test = text_course::select(corpus, split.test);
    // Vocabulary is learned on training text only (no test leakage).
    text_course::Vectorizer counts(false, 3000);
    counts.fit(train.text);
    text_course::MultinomialNB nb(1.0);
    nb.fit(counts.transform(train.text), train.y, counts.vocabulary.size());
    // Discriminative words: largest |log P(w|spam) - log P(w|ham)|.
    const size_t p = counts.vocabulary.size();
    std::vector<size_t> idx(p);
    std::iota(idx.begin(), idx.end(), 0);
    auto odds = [&](size_t j) { return nb.logprob[1][j] - nb.logprob[0][j]; };
    const size_t top = std::min<size_t>(20, p);
    std::partial_sort(idx.begin(), idx.begin() + top, idx.end(),
                      [&](size_t x, size_t y) {
                        return std::abs(odds(x)) > std::abs(odds(y));
                      });
    std::ostringstream csv;
    csv << "rank,term,logprob_ham,logprob_spam,log_odds_spam\n";
    ml::Vec h;
    std::vector<std::string> lab;
    for (size_t r = 0; r < top; ++r) {
      size_t j = idx[r];
      csv << r + 1 << ',' << ml::csv_quote(counts.vocabulary[j]) << ','
          << nb.logprob[0][j] << ',' << nb.logprob[1][j] << ',' << odds(j)
          << '\n';
      h.push_back(odds(j));
      lab.push_back(counts.vocabulary[j]);
    }
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("discriminative_words.csv", csv.str());
    ml::Plot bars(820, 400);
    bars.bar(h, lab);
    bars.title("Naive Bayes log-odds spam vs ham (top |log-odds| words)");
    bars.xlabel("term");
    bars.ylabel("log P(w|spam) - log P(w|ham)");
    a.figure("discriminative_words.svg", bars);
    // Holdout evaluation at the default 0.5 threshold (hard() in Workflow.hpp).
    auto prob = nb.probabilities(counts.transform(test.text));
    auto score = ml::binary_scores(test.y, text_course::hard(prob), &prob);
    std::ostringstream cm;
    cm << "actual,predicted_ham,predicted_spam\n"
       << "ham," << score.tn << ',' << score.fp << '\n'
       << "spam," << score.fn << ',' << score.tp << '\n';
    a.write("confusion_matrix.csv", cm.str());
    std::cout << "holdout accuracy " << score.accuracy << "  f1 " << score.f1
              << "\n";
    for (const auto &f : {"discriminative_words.csv",
                          "discriminative_words.svg", "confusion_matrix.csv"})
      std::cout << "wrote " << a.path(f) << "\n";
  } catch (const std::exception &e) {
    std::cerr << "artifact error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
