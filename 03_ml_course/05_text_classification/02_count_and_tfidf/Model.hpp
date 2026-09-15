// 03_ml_course/05_text_classification/02_count_and_tfidf/Model.hpp
//
// Purpose : sparse bag-of-words feature extraction. Vectorizer learns a capped
//           vocabulary plus IDF weights on training text and turns documents
//           into sparse rows of either raw counts or L2-normalised TF-IDF.
// Inputs  : documents as std::vector<std::string>; tokenization comes from
//           01_text_preprocessing/Text.hpp. No files or defines are read here.
// Outputs : none directly. Workflow.hpp archives a fitted Vectorizer as
//           <run>/model/preprocessing.txt via save()/load().
// Used by : MultinomialNB, LinearTextClassifier, Workflow.hpp, all text
//           lessons and tests (header-only, no dedicated target).
#pragma once
#include "05_text_classification/01_text_preprocessing/Text.hpp"
#include "helper/persistence/archive.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>
namespace text_course {
// One document as (feature index, weight) pairs sorted by index with no
// duplicate indices. Absent indices are zero. This is the input format every
// text model in this course consumes.
using SparseRow = std::vector<std::pair<size_t, double>>;
// A batch of documents, one SparseRow per document.
using Sparse = std::vector<SparseRow>;
// Bag-of-words vectorizer with optional TF-IDF weighting.
// Invariants after fit(): vocabulary, idf and index have the same size;
// index[vocabulary[i]] == i; vocabulary.size() <= max_features.
// Model contract used by Workflow.hpp: fit(docs), transform(docs) -> Sparse,
// vocabulary.size() as the feature count p, save(ostream)/load(istream).
class Vectorizer {
public:
  size_t max_features;                          // vocabulary cap (> 0)
  bool tfidf;                                   // false = raw term counts
  std::vector<std::string> vocabulary;          // feature index -> term
  std::vector<double> idf;                      // smoothed IDF per feature
  std::unordered_map<std::string, size_t> index; // term -> feature index
  // tf  : true for TF-IDF weighting, false for raw counts.
  // cap : maximum number of retained terms (most frequent by document count).
  explicit Vectorizer(bool tf = true, size_t cap = 3000)
      : max_features(cap), tfidf(tf) {}
  // Learn the vocabulary and IDF from training documents only.
  // Terms are ranked by document frequency (ties alphabetical) and cut at
  // max_features. idf = log((1 + N) / (1 + df)) + 1, the smoothed form used
  // by scikit-learn, so unseen-in-any-document terms cannot divide by zero.
  // Throws std::invalid_argument on an empty corpus, zero cap or no tokens.
  // Complexity: O(total tokens * log V + V log V) with V distinct terms.
  void fit(const std::vector<std::string> &documents) {
    if (documents.empty() || !max_features)
      throw std::invalid_argument("Empty corpus or vocabulary cap");
    // Document frequency: count each term once per document.
    std::map<std::string, size_t> df;
    for (const auto &doc : documents) {
      auto words = tokenize(doc);
      std::set<std::string> unique(words.begin(), words.end());
      for (const auto &w : unique)
        ++df[w];
    }
    std::vector<std::pair<std::string, size_t>> sorted(df.begin(), df.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
      return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    if (sorted.size() > max_features)
      sorted.resize(max_features);
    vocabulary.clear();
    idf.clear();
    index.clear();
    for (const auto &entry : sorted) {
      index[entry.first] = vocabulary.size();
      vocabulary.push_back(entry.first);
      idf.push_back(std::log((1.0 + documents.size()) / (1.0 + entry.second)) +
                    1);
    }
    if (vocabulary.empty())
      throw std::invalid_argument("Corpus contains no tokens");
  }
  // Encode documents with the fitted vocabulary. Tokens outside the
  // vocabulary are silently dropped, so an all-unknown document yields an
  // empty row. With tfidf each count is multiplied by its IDF and the row is
  // L2-normalised (rows with norm 0 stay empty). Rows are sorted by index
  // because they are built from a std::map.
  // Throws std::runtime_error if fit() has not run.
  // Complexity: O(total tokens * log(terms per document)).
  Sparse transform(const std::vector<std::string> &documents) const {
    if (vocabulary.empty())
      throw std::runtime_error("Vectorizer not fitted");
    Sparse result;
    for (const auto &doc : documents) {
      std::map<size_t, double> count;
      for (const auto &word : tokenize(doc)) {
        auto it = index.find(word);
        if (it != index.end())
          ++count[it->second];
      }
      SparseRow row(count.begin(), count.end());
      double norm = 0;
      for (auto &v : row) {
        if (tfidf)
          v.second *= idf[v.first];
        norm += v.second * v.second;
      }
      if (tfidf && norm > 0)
        for (auto &v : row)
          v.second /= std::sqrt(norm);
      result.push_back(row);
    }
    return result;
  }
  // Serialise as a versioned text archive: type tag, cap, mode, vocabulary,
  // idf. The index is rebuilt on load rather than stored.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("TextVectorizer_V1"));
    ml::archive::write(out, max_features, tfidf, vocabulary, idf);
  }
  // Restore from save(). Rejects a foreign type tag or a vocabulary/idf size
  // mismatch, then rebuilds the term -> index map.
  void load(std::istream &in) {
    std::string model_type;
    ml::archive::read(in, model_type);
    if (model_type != std::string("TextVectorizer_V1"))
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, max_features, tfidf, vocabulary, idf);
    if (vocabulary.size() != idf.size())
      throw std::runtime_error("Invalid vocabulary archive");
    index.clear();
    for (size_t i = 0; i < vocabulary.size(); ++i)
      index[vocabulary[i]] = i;
  }
};
} // namespace text_course
