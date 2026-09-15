// 03_ml_course/05_text_classification/Workflow.hpp
//
// Purpose : the shared end-to-end project for every text module. One
//           function template, workflow<Model>(), performs the complete
//           training / model-selection / evaluation / persistence run, and
//           doubles as the inference CLI when --predict is given.
// Inputs  : data - path to SMSSpamCollection (built from COURSE_ROOT by each
//           04_end_to_end.cpp); root - RUN_OUTPUT_DIR of the calling target.
//           CLI: [--quick] | --predict TEXT_FILE --model MODEL_DIRECTORY.
// Outputs : training run: <root>/full/ (or quick/) with data/, eda/,
//           validation/, evaluation/, model/, inference/, run_manifest.json,
//           report.md, execution.log. Inference run: <root>/predictions.csv.
// Used by : text_*_end_to_end and text_*_predict (predict.cpp includes
//           04_end_to_end.cpp so both share this code), 03_implementation
//           lessons (for hard()) and the module tests.
//
// Model contract (satisfied by MultinomialNB and LinearTextClassifier):
//   Model();  Model(double parameter);
//   void fit(const Sparse &X, const std::vector<double> &y, size_t p);
//   std::vector<double> probabilities(const Sparse &X) const;  // p(spam)
//   void save(std::ostream &) const;  void load(std::istream &);
#pragma once
#include "05_text_classification/03_multinomial_naive_bayes/Model.hpp"
#include "05_text_classification/04_linear_text_classifier/Model.hpp"
#include "helper/eval/cross_validation.hpp"
#include "helper/math/metrics.hpp"
#include "helper/reporting/artifacts.hpp"
#include <iostream>
#include <sstream>
namespace text_course {
// Threshold probabilities at 0.5 into hard labels (1 = spam).
// Parameters: p - p(spam) per row. Returns a label vector of equal length.
inline ml::Vec hard(const ml::Vec &p) {
  ml::Vec y;
  for (double v : p)
    y.push_back(v >= 0.5 ? 1 : 0);
  return y;
}
// Summarise a corpus with four dense descriptive features per message
// (characters, tokens, digits, exclamation marks). Used only for EDA tables
// and the dataset fingerprint; the classifiers never see these columns.
// Complexity: O(total characters).
inline ml::Dataset describe(const Corpus &c) {
  ml::Dataset d;
  d.feature_names = {"characters", "tokens", "digits", "exclamations"};
  d.target_name = "spam";
  d.y = c.y;
  for (const auto &s : c.text) {
    double digits = 0, bangs = 0;
    for (unsigned char ch : s) {
      digits += std::isdigit(ch) ? 1 : 0;
      bangs += ch == '!' ? 1 : 0;
    }
    d.X.push_back(
        {double(s.size()), double(tokenize(s).size()), digits, bangs});
  }
  return d;
}
// Run the complete text-classification project (or inference).
// Parameters:
//   root       - results directory for this executable (RUN_OUTPUT_DIR).
//   data       - path to the SMS corpus.
//   tfidf      - true: TF-IDF features, false: raw counts.
//   parameters - candidate values of the model's single hyper-parameter
//                (alpha for NB, lambda for logistic regression).
//   argc/argv  - command line; see file header for the accepted flags.
// Returns 0 on success, 1 on any exception (message printed to stderr).
// Protocol: dedupe -> optional quick subsample (seed 42) -> stratified
// holdout (fold 0 of a seeded 5-fold split) -> 5-fold CV on the training
// part (seed 43) with vocabulary/IDF refitted inside every fold -> refit
// the winner on all training rows -> evaluate the holdout once -> save the
// archive and verify a reload reproduces the probabilities.
template <class Model>
int workflow(const std::string &root, const std::string &data, bool tfidf,
             const ml::Vec &parameters, int argc, char **argv) {
  try {
    // ---- command line -----------------------------------------------------
    bool quick = false;
    std::string predict_file, model_dir;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--quick")
        quick = true;
      else if (arg == "--predict" && i + 1 < argc)
        predict_file = argv[++i];
      else if (arg == "--model" && i + 1 < argc)
        model_dir = argv[++i];
      else
        throw std::invalid_argument(
            "Use --quick or --predict TEXT_FILE --model MODEL_DIRECTORY");
    }
    // ---- inference mode ---------------------------------------------------
    // Reads one message per line, restores the saved vectorizer + model from
    // <model_dir> and writes <root>/predictions.csv (row, p_spam, prediction).
    if (!predict_file.empty()) {
      if (model_dir.empty())
        throw std::invalid_argument("--model required for inference");
      Vectorizer vectorizer;
      Model model;
      ml::archive::load_file(model_dir + "/preprocessing.txt", vectorizer);
      ml::archive::load_file(model_dir + "/model.txt", model);
      std::ifstream input(predict_file);
      if (!input)
        throw std::runtime_error("Cannot read inference text");
      std::vector<std::string> docs;
      std::string line;
      while (std::getline(input, line))
        docs.push_back(line);
      auto p = model.probabilities(vectorizer.transform(docs));
      std::ostringstream csv;
      csv << "row,p_spam,prediction\n";
      for (size_t i = 0; i < p.size(); ++i)
        csv << i << ',' << p[i] << ',' << (p[i] >= 0.5) << '\n';
      ml::Artifacts(root, ".").write("predictions.csv", csv.str());
      return 0;
    }
    // ---- data -------------------------------------------------------------
    Corpus corpus = read_sms(data);
    // --quick keeps a seeded random subset of ~1000 messages for smoke tests.
    if (quick && corpus.y.size() > 1000) {
      auto ids =
          ml::train_test_indices(corpus.y.size(), 1000.0 / corpus.y.size(), 42)
              .train;
      corpus = select(corpus, ids);
    }
    // Taking one stratified fold as the final holdout gives an approximately
    // 80/20 split while retaining minority class representation.
    auto split = ml::stratified_kfold(corpus.y, 5, 42).front();
    auto train = select(corpus, split.train), test = select(corpus, split.test);
    ml::Artifacts artifacts(root, quick ? "quick" : "full");
    // EDA is computed on training rows only so the holdout stays untouched.
    ml::eda(describe(train), artifacts);
    std::ostringstream quality;
    quality << "rows,duplicates_removed,train_rows,test_rows\n"
            << corpus.y.size() << ',' << corpus.duplicates << ','
            << train.y.size() << ',' << test.y.size() << '\n';
    artifacts.write("data/quality_report.csv", quality.str());
    std::ostringstream assignments;
    assignments << "source_row,split\n";
    for (size_t id : train.ids)
      assignments << id << ",train\n";
    for (size_t id : test.ids)
      assignments << id << ",test\n";
    artifacts.write("data/row_assignments.csv", assignments.str());
    // ---- model selection: 5-fold CV inside the training part -------------
    // A different seed (43) keeps the inner folds independent of the outer
    // split. The vectorizer is refitted per fold so IDF never sees fold
    // validation text; the vocabulary cap is smaller in quick mode.
    auto folds = ml::stratified_kfold(train.y, 5, 43);
    double best = -1, chosen = parameters.front();
    std::ostringstream scores;
    scores << "parameter,fold,f1\n";
    for (double value : parameters) {
      double total = 0;
      size_t f = 0;
      for (const auto &fold : folds) {
        auto a = select(train, fold.train), b = select(train, fold.test);
        Vectorizer v(tfidf, quick ? 1000 : 3000);
        v.fit(a.text);
        Model m(value);
        m.fit(v.transform(a.text), a.y, v.vocabulary.size());
        auto p = m.probabilities(v.transform(b.text));
        double s = ml::binary_scores(b.y, hard(p), &p).f1;
        scores << value << ',' << f++ << ',' << s << '\n';
        total += s / folds.size();
      }
      // Strict '>' keeps the first candidate on ties (deterministic choice).
      if (total > best) {
        best = total;
        chosen = value;
      }
    }
    artifacts.write("validation/candidate_scores.csv", scores.str());
    // ---- final fit on all training rows and single holdout evaluation ----
    Vectorizer vectorizer(tfidf, quick ? 1000 : 3000);
    vectorizer.fit(train.text);
    Model model(chosen);
    model.fit(vectorizer.transform(train.text), train.y,
              vectorizer.vocabulary.size());
    auto p = model.probabilities(vectorizer.transform(test.text));
    auto prediction = hard(p);
    auto score = ml::binary_scores(test.y, prediction, &p);
    std::ostringstream metrics;
    metrics << "{\"accuracy\":" << score.accuracy
            << ",\"precision\":" << score.precision
            << ",\"recall\":" << score.recall << ",\"f1\":" << score.f1
            << ",\"roc_auc\":" << score.auc
            << ",\"log_loss\":" << ml::binary_logloss(test.y, p)
            << ",\"ham_baseline_accuracy\":"
            << ml::accuracy(test.y, ml::Vec(test.y.size(), 0))
            << ",\"tp\":" << score.tp << ",\"fp\":" << score.fp
            << ",\"tn\":" << score.tn << ",\"fn\":" << score.fn << "}\n";
    artifacts.write("evaluation/metrics.json", metrics.str());
    std::ostringstream predictions;
    predictions << "source_row,actual,prediction,p_spam\n";
    for (size_t i = 0; i < p.size(); ++i)
      predictions << test.ids[i] << ',' << test.y[i] << ',' << prediction[i]
                  << ',' << p[i] << '\n';
    artifacts.write("evaluation/predictions.csv", predictions.str());
    // ROC curve points (FPR, TPR) over every distinct score threshold.
    std::ostringstream roc;
    roc << "false_positive_rate,true_positive_rate\n";
    for (const auto &v : ml::roc_curve(test.y, p))
      roc << v.first << ',' << v.second << '\n';
    artifacts.write("evaluation/roc_curve.csv", roc.str());
    // Precision/recall on a fixed grid of 101 thresholds (0.00 .. 1.00).
    std::ostringstream pr;
    pr << "threshold,precision,recall\n";
    for (int k = 0; k <= 100; ++k) {
      double threshold = k / 100.0;
      ml::Vec labels;
      for (double v : p)
        labels.push_back(v >= threshold ? 1 : 0);
      auto s = ml::binary_scores(test.y, labels);
      pr << threshold << ',' << s.precision << ',' << s.recall << '\n';
    }
    artifacts.write("evaluation/precision_recall_curve.csv", pr.str());
    // Reliability table: ten equal-width probability bins; p == 1 goes to
    // bin 9 via the min(9, ...) clamp. Empty bins are omitted.
    std::ostringstream calibration;
    calibration << "bin,count,mean_probability,observed_fraction\n";
    for (int b = 0; b < 10; ++b) {
      double count = 0, sum = 0, positive = 0;
      for (size_t i = 0; i < p.size(); ++i)
        if (std::min(9, int(p[i] * 10)) == b) {
          ++count;
          sum += p[i];
          positive += test.y[i];
        }
      if (count)
        calibration << b << ',' << count << ',' << sum / count << ','
                    << positive / count << '\n';
    }
    artifacts.write("evaluation/calibration.csv", calibration.str());
    // ---- persistence and reload verification ------------------------------
    std::ostringstream terms;
    terms << "index,term,idf\n";
    for (size_t i = 0; i < vectorizer.vocabulary.size(); ++i)
      terms << i << ',' << ml::csv_quote(vectorizer.vocabulary[i]) << ','
            << vectorizer.idf[i] << '\n';
    artifacts.write("model/vocabulary.csv", terms.str());
    ml::archive::save_file(artifacts.path("model/model.txt"), model);
    ml::archive::save_file(artifacts.path("model/preprocessing.txt"),
                           vectorizer);
    // Reload both archives into fresh objects and demand bit-for-bit equal
    // probabilities (tolerance 1e-12) before declaring the run reproducible.
    Vectorizer restored;
    Model copy;
    ml::archive::load_file(artifacts.path("model/model.txt"), copy);
    ml::archive::load_file(artifacts.path("model/preprocessing.txt"), restored);
    auto again = copy.probabilities(restored.transform(test.text));
    double error = 0;
    for (size_t i = 0; i < p.size(); ++i)
      error = std::max(error, std::abs(p[i] - again[i]));
    if (error > 1e-12)
      throw std::runtime_error("Reload prediction mismatch");
    artifacts.write("inference/reload_verification.json",
                    "{\"passed\":true}\n");
    // ---- manifest, report, log --------------------------------------------
    std::ostringstream manifest;
    manifest << "{\"seed\":42,\"fold_seed\":43,\"parameter\":" << chosen
             << ",\"tfidf\":" << (tfidf ? "true" : "false")
             << ",\"vocabulary\":" << vectorizer.vocabulary.size()
             << ",\"dataset\":\"UCI SMS Spam "
                "Collection\",\"description_fingerprint\":\""
             << ml::fingerprint(describe(corpus)) << "\"}\n";
    artifacts.write("run_manifest.json", manifest.str());
    std::ostringstream report;
    report
        << "# SMS spam classification\n\nHeld-out F1: " << score.f1
        << "; precision: " << score.precision << "; recall: " << score.recall
        << ". Selected parameter " << chosen << " with mean training CV F1 "
        << best
        << ".\n\nExact duplicate messages were removed before splitting. "
           "Vocabulary and IDF were learned separately within each fold. The "
           "cap retains frequent training terms; unknown words are ignored and "
           "an all-unknown message uses the model intercept/prior. The ASCII "
           "tokenizer does not perform Unicode normalization. Source "
           "collection dates and sender identities are unavailable; random "
           "evaluation does not establish future or sender-independent "
           "performance. Inspect false positives before deployment; flagging "
           "legitimate SMS has a real user cost. EDA summarizes training rows "
           "only. Calibration and threshold tables describe the held-out set; "
           "do not tune a deployment threshold on it.\n";
    artifacts.write("report.md", report.str());
    artifacts.write("execution.log",
                    "Completed training-only CV, final evaluation, model "
                    "persistence and reload verification.\n");
    std::cout << metrics.str();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
} // namespace text_course
