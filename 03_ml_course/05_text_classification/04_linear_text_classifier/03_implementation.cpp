// 03_ml_course/05_text_classification/04_linear_text_classifier/03_implementation.cpp
//
// Purpose : implementation lesson for the linear text classifier. Traces the
//           training log-loss of full-batch gradient descent on TF-IDF
//           features and lists the terms with the largest learned weights.
// Inputs  : <COURSE_ROOT>/05_text_classification/data/SMSSpamCollection
//           (COURSE_ROOT, RUN_OUTPUT_DIR from add_lesson_executable).
// Outputs : results/03_implementation_results/
//             training_loss_per_epoch.csv / .svg, largest_weights.csv / .svg
// Target  : text_linear_implementation
#include "05_text_classification/Workflow.hpp"
#include <iostream>
#include <numeric>

// Lesson: L2-regularised logistic regression on TF-IDF features.
// Artifacts: terms with the largest |weight| (CSV + bar SVG) and the
// training log-loss per epoch (CSV + line SVG).
int main() {
  std::cout
      << R"LESSON(Fit on training text, then transform a new message. Unknown tokens contribute no feature.)LESSON"
      << "\n";
  // Toy demonstration of the TF-IDF vectorizer (see 02_count_and_tfidf).
  text_course::Vectorizer v(true);
  v.fit({"win win now", "hello friend"});
  auto encoded = v.transform({"win unknown"});
  for (const auto &term : encoded[0])
    std::cout << v.vocabulary[term.first] << "=" << term.second << "\n";

  try {
    const std::string data = std::string(COURSE_ROOT) +
                             "/05_text_classification/data/SMSSpamCollection";
    auto corpus = text_course::read_sms(data);
    // Exploratory lesson: fit on the whole corpus to study optimisation
    // behaviour; the reported loss is a training loss, not a generalisation
    // estimate. The end-to-end workflow uses proper held-out evaluation.
    text_course::Vectorizer tfidf(true, 3000);
    tfidf.fit(corpus.text);
    auto X = tfidf.transform(corpus.text);
    // Loss curve: refit with a growing epoch budget. The model is
    // deterministic full-batch gradient descent, so every prefix is identical
    // to the corresponding prefix of the longest run.
    const size_t total_epochs = 200, step = 10;
    std::ostringstream loss;
    loss << "epoch,train_log_loss\n";
    ml::Vec xs, ys;
    text_course::LinearTextClassifier model;
    for (size_t e = step; e <= total_epochs; e += step) {
      model = text_course::LinearTextClassifier(0.001, e);
      model.fit(X, corpus.y, tfidf.vocabulary.size());
      double l = ml::binary_logloss(corpus.y, model.probabilities(X));
      loss << e << ',' << l << '\n';
      xs.push_back(double(e));
      ys.push_back(l);
    }
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("training_loss_per_epoch.csv", loss.str());
    ml::Plot curve;
    curve.line(xs, ys, "train log-loss");
    curve.title("Training log-loss vs epoch");
    curve.xlabel("epoch");
    curve.ylabel("log-loss");
    a.figure("training_loss_per_epoch.svg", curve);
    // `model` now holds the 200-epoch fit; rank features by |weight|.
    // Positive weights push towards spam, negative towards ham.
    const size_t p = model.weights.size();
    std::vector<size_t> idx(p);
    std::iota(idx.begin(), idx.end(), 0);
    const size_t top = std::min<size_t>(20, p);
    std::partial_sort(idx.begin(), idx.begin() + top, idx.end(),
                      [&](size_t x, size_t y) {
                        return std::abs(model.weights[x]) >
                               std::abs(model.weights[y]);
                      });
    std::ostringstream csv;
    csv << "rank,term,weight\n";
    ml::Vec h;
    std::vector<std::string> lab;
    for (size_t r = 0; r < top; ++r) {
      csv << r + 1 << ',' << ml::csv_quote(tfidf.vocabulary[idx[r]]) << ','
          << model.weights[idx[r]] << '\n';
      h.push_back(model.weights[idx[r]]);
      lab.push_back(tfidf.vocabulary[idx[r]]);
    }
    a.write("largest_weights.csv", csv.str());
    ml::Plot bars(820, 400);
    bars.bar(h, lab);
    bars.title("Largest |weight| terms (positive = spam)");
    bars.xlabel("term");
    bars.ylabel("weight");
    a.figure("largest_weights.svg", bars);
    std::cout << "final train log-loss " << ys.back() << "  bias "
              << model.bias << "\n";
    for (const auto &f :
         {"training_loss_per_epoch.csv", "training_loss_per_epoch.svg",
          "largest_weights.csv", "largest_weights.svg"})
      std::cout << "wrote " << a.path(f) << "\n";
  } catch (const std::exception &e) {
    std::cerr << "artifact error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
