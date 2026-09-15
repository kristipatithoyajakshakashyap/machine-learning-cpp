// 03_ml_course/05_text_classification/02_count_and_tfidf/03_implementation.cpp
//
// Purpose : implementation lesson for TF-IDF. Fits a TF-IDF Vectorizer on the
//           whole SMS corpus, reports the terms with the highest mean weight
//           in each class and shows how a min_df cut-off shrinks the
//           vocabulary.
// Inputs  : <COURSE_ROOT>/05_text_classification/data/SMSSpamCollection
//           (COURSE_ROOT, RUN_OUTPUT_DIR from add_lesson_executable).
// Outputs : results/03_implementation_results/
//             top_tfidf_terms_per_class.csv, top_tfidf_ham.svg,
//             top_tfidf_spam.svg, vocabulary_size_vs_min_df.csv / .svg
// Target  : text_tfidf_implementation
#include "05_text_classification/Workflow.hpp"
#include <iostream>
#include <map>
#include <numeric>

// Lesson: TF-IDF weighting on the SMS corpus.
// Artifacts: top TF-IDF terms per class (CSV + bar SVG) and vocabulary size
// as a function of the min_df cut-off (CSV + line SVG).
int main() {
  std::cout
      << R"LESSON(Fit on training text, then transform a new message. Unknown tokens contribute no feature.)LESSON"
      << "\n";
  // Toy demonstration with TF-IDF weighting: the single known token "win"
  // gets weight 1 after L2 normalisation; "unknown" is dropped.
  text_course::Vectorizer v(true);
  v.fit({"win win now", "hello friend"});
  auto encoded = v.transform({"win unknown"});
  for (const auto &term : encoded[0])
    std::cout << v.vocabulary[term.first] << "=" << term.second << "\n";

  try {
    const std::string data = std::string(COURSE_ROOT) +
                             "/05_text_classification/data/SMSSpamCollection";
    auto corpus = text_course::read_sms(data);
    // Exploratory lesson: the vectorizer is fitted on all rows here. The
    // end-to-end workflow fits it on training folds only.
    text_course::Vectorizer tfidf(true, 3000);
    tfidf.fit(corpus.text);
    auto X = tfidf.transform(corpus.text);
    // Mean TF-IDF weight of each term within each class.
    // sum[c][j] accumulates the weight of feature j over class-c documents;
    // n[c] counts documents per class (0 = ham, 1 = spam).
    std::vector<ml::Vec> sum(2, ml::Vec(tfidf.vocabulary.size(), 0.0));
    std::vector<double> n(2, 0.0);
    for (size_t i = 0; i < X.size(); ++i) {
      size_t c = size_t(corpus.y[i]);
      ++n[c];
      for (const auto &t : X[i])
        sum[c][t.first] += t.second;
    }
    const size_t top = std::min<size_t>(15, tfidf.vocabulary.size());
    std::ostringstream csv;
    csv << "class,rank,term,mean_tfidf\n";
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    const char *names[2] = {"ham", "spam"};
    for (size_t c = 0; c < 2; ++c) {
      // partial_sort only orders the first `top` indices: O(V log top).
      std::vector<size_t> idx(tfidf.vocabulary.size());
      std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin() + top, idx.end(),
                        [&](size_t p, size_t q) {
                          return sum[c][p] > sum[c][q];
                        });
      ml::Vec h;
      std::vector<std::string> lab;
      for (size_t r = 0; r < top; ++r) {
        double m = sum[c][idx[r]] / n[c];
        csv << names[c] << ',' << r + 1 << ','
            << ml::csv_quote(tfidf.vocabulary[idx[r]]) << ',' << m << '\n';
        h.push_back(m);
        lab.push_back(tfidf.vocabulary[idx[r]]);
      }
      ml::Plot p(760, 400);
      p.bar(h, lab);
      p.title(std::string("Top mean TF-IDF terms: ") + names[c]);
      p.xlabel("term");
      p.ylabel("mean TF-IDF");
      a.figure(std::string("top_tfidf_") + names[c] + ".svg", p);
    }
    a.write("top_tfidf_terms_per_class.csv", csv.str());
    // Vocabulary size vs min_df: terms whose document frequency >= min_df.
    // Document frequency counts a term once per document, like Vectorizer.
    std::map<std::string, size_t> df;
    for (const auto &doc : corpus.text) {
      auto words = text_course::tokenize(doc);
      std::set<std::string> uniq(words.begin(), words.end());
      for (const auto &w : uniq)
        ++df[w];
    }
    std::ostringstream vocab;
    vocab << "min_df,vocabulary_size\n";
    ml::Vec xs, ys;
    for (size_t min_df : {1, 2, 3, 5, 10, 20, 50, 100}) {
      size_t kept = 0;
      for (const auto &e : df)
        kept += e.second >= min_df;
      vocab << min_df << ',' << kept << '\n';
      xs.push_back(double(min_df));
      ys.push_back(double(kept));
    }
    a.write("vocabulary_size_vs_min_df.csv", vocab.str());
    ml::Plot line;
    line.line(xs, ys, "vocabulary");
    line.title("Vocabulary size vs min_df");
    line.xlabel("min_df");
    line.ylabel("terms kept");
    a.figure("vocabulary_size_vs_min_df.svg", line);
    for (const auto &f :
         {"top_tfidf_terms_per_class.csv", "top_tfidf_ham.svg",
          "top_tfidf_spam.svg", "vocabulary_size_vs_min_df.csv",
          "vocabulary_size_vs_min_df.svg"})
      std::cout << "wrote " << a.path(f) << "\n";
  } catch (const std::exception &e) {
    std::cerr << "artifact error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
