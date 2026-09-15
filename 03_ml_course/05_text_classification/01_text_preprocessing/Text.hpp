// 03_ml_course/05_text_classification/01_text_preprocessing/Text.hpp
//
// Purpose : text primitives shared by every text-classification module:
//           an ASCII tokenizer, the in-memory SMS corpus and its loader.
// Inputs  : read_sms(path) reads the UCI SMS Spam Collection (tab-separated
//           "label<TAB>message" rows). Lessons build the path from COURSE_ROOT:
//           <COURSE_ROOT>/05_text_classification/data/SMSSpamCollection.
// Outputs : none. This header writes no files; workflows write results.
// Used by : Vectorizer (02_count_and_tfidf/Model.hpp), Workflow.hpp and every
//           text lesson/test through those headers (no dedicated target).
#pragma once
#include <cctype>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace text_course {
// Split text into lowercase ASCII alphanumeric tokens.
// Rules: a token is a maximal run of bytes < 128 that pass std::isalnum;
// every other byte (punctuation, whitespace, any non-ASCII byte) is a
// separator. No Unicode normalization, stemming or stop-word removal.
// Parameters: text - raw message. Returns tokens in order of appearance.
// Complexity: O(length of text).
inline std::vector<std::string> tokenize(const std::string &text) {
  std::vector<std::string> words;
  std::string word;
  for (unsigned char c : text) {
    if (c < 128 && std::isalnum(c))
      word += static_cast<char>(std::tolower(c));
    else if (!word.empty()) {
      words.push_back(word);
      word.clear();
    }
  }
  if (!word.empty())
    words.push_back(word);
  return words;
}
// A labelled SMS corpus. All three vectors are parallel (same length):
// text[i] is the message, y[i] its label (1 = spam, 0 = ham) and ids[i] the
// 1-based source line number so results can be traced back to the raw file.
// duplicates counts exact-duplicate messages dropped by read_sms.
struct Corpus {
  std::vector<std::string> text;
  std::vector<double> y;
  std::vector<size_t> ids;
  size_t duplicates = 0;
};
// Load the SMS Spam Collection from disk.
// Parameters: path - file with one "ham|spam<TAB>message" row per line.
// Returns a Corpus with exact duplicate messages removed (first occurrence
// kept) so that a split can never place the same text in train and test.
// Throws std::runtime_error on missing file, missing tab or unknown label.
// Complexity: O(total characters * log rows) due to the duplicate set.
inline Corpus read_sms(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot open SMS corpus: " + path);
  Corpus c;
  std::set<std::string> seen;
  std::string line;
  size_t id = 0;
  while (std::getline(in, line)) {
    ++id;
    // Tolerate Windows line endings without treating '\r' as message text.
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    auto tab = line.find('\t');
    if (tab == std::string::npos)
      throw std::runtime_error("Malformed SMS row");
    const auto label = line.substr(0, tab), message = line.substr(tab + 1);
    if (label != "ham" && label != "spam")
      throw std::runtime_error("Unknown SMS label");
    // Duplicate messages are counted and skipped; ids keep the raw line number.
    if (!seen.insert(message).second) {
      ++c.duplicates;
      continue;
    }
    c.ids.push_back(id);
    c.text.push_back(message);
    c.y.push_back(label == "spam" ? 1 : 0);
  }
  return c;
}
// Build a sub-corpus from row positions (used for train/test and CV folds).
// Parameters: c - source corpus; rows - positions into c (0-based).
// Returns a new Corpus containing those rows in the given order; the
// duplicates counter is not carried over. Throws std::out_of_range on a bad
// position. Complexity: O(rows.size()).
inline Corpus select(const Corpus &c, const std::vector<size_t> &rows) {
  Corpus out;
  for (size_t i : rows) {
    out.text.push_back(c.text.at(i));
    out.y.push_back(c.y.at(i));
    out.ids.push_back(c.ids.at(i));
  }
  return out;
}
} // namespace text_course
