// 03_ml_course/05_text_classification/01_text_preprocessing/03_implementation.cpp
//
// Purpose : implementation lesson for text preprocessing. Shows how the
//           Vectorizer behaves on a toy corpus, then tokenizes the whole SMS
//           corpus to inspect token frequencies and document lengths.
// Inputs  : <COURSE_ROOT>/05_text_classification/data/SMSSpamCollection
//           (COURSE_ROOT and RUN_OUTPUT_DIR are compile definitions from
//           add_lesson_executable in the root CMakeLists.txt).
// Outputs : results/03_implementation_results/
//             token_frequency_top30.csv / .svg  - 30 most frequent tokens
//             document_length_histogram.svg     - tokens per message
// Target  : text_preprocessing_implementation
#include "05_text_classification/Workflow.hpp"
#include <iostream>
#include <map>

// Lesson: tokenize the SMS corpus and inspect what the vectorizer sees.
// Artifacts: token_frequency_top30.csv (+ bar SVG) and a histogram of
// document lengths in tokens.
int main() {
  std::cout
      << R"LESSON(Fit on training text, then transform a new message. Unknown tokens contribute no feature.)LESSON"
      << "\n";
  // Toy demonstration: raw counts (tfidf=false). "unknown" was never seen
  // during fit, so the encoded row only contains the "win" feature.
  text_course::Vectorizer v(false);
  v.fit({"win win now", "hello friend"});
  auto encoded = v.transform({"win unknown"});
  for (const auto &term : encoded[0])
    std::cout << v.vocabulary[term.first] << "=" << term.second << "\n";

  try {
    const std::string data = std::string(COURSE_ROOT) +
                             "/05_text_classification/data/SMSSpamCollection";
    auto corpus = text_course::read_sms(data);
    // Corpus-wide token counts and per-document lengths (no labels needed).
    std::map<std::string, size_t> freq;
    ml::Vec lengths;
    for (const auto &doc : corpus.text) {
      auto words = text_course::tokenize(doc);
      lengths.push_back(double(words.size()));
      for (const auto &w : words)
        ++freq[w];
    }
    // Sort by count descending; ties broken alphabetically so output is
    // deterministic across runs and platforms.
    std::vector<std::pair<std::string, size_t>> sorted(freq.begin(),
                                                       freq.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
      return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    const size_t top = std::min<size_t>(30, sorted.size());
    std::ostringstream csv;
    csv << "rank,token,count\n";
    ml::Vec heights;
    std::vector<std::string> labels;
    for (size_t i = 0; i < top; ++i) {
      csv << i + 1 << ',' << ml::csv_quote(sorted[i].first) << ','
          << sorted[i].second << '\n';
      heights.push_back(double(sorted[i].second));
      labels.push_back(sorted[i].first);
    }
    // Artifacts(root, ".") writes directly into RUN_OUTPUT_DIR (no sub-run).
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("token_frequency_top30.csv", csv.str());
    ml::Plot bars(900, 420);
    bars.bar(heights, labels);
    bars.title("Top 30 tokens in SMS corpus");
    bars.xlabel("token");
    bars.ylabel("count");
    a.figure("token_frequency_top30.svg", bars);
    ml::Plot hist;
    hist.histogram(lengths, 30);
    hist.title("Document length (tokens)");
    hist.xlabel("tokens per message");
    hist.ylabel("messages");
    a.figure("document_length_histogram.svg", hist);
    std::cout << "documents " << corpus.text.size() << "  distinct tokens "
              << freq.size() << "\n";
    for (const auto &f :
         {"token_frequency_top30.csv", "token_frequency_top30.svg",
          "document_length_histogram.svg"})
      std::cout << "wrote " << a.path(f) << "\n";
  } catch (const std::exception &e) {
    std::cerr << "artifact error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
