// 03_ml_course/helper/persistence/archive.hpp
//
// Purpose : the course's single model persistence format. A tiny text
//           serializer that every model, preprocessor and vectorizer uses
//           for save()/load(), plus file helpers that add a magic header.
// Inputs  : any std::ostream / std::istream; save_file/load_file take a path.
// Outputs : text archives (one value per line) written by save_file; the
//           workflows place them under results/<stem>_results/<run>/model/.
// Used by : every Model.hpp, Preprocessor, TargetTransform, the workflows
//           in helper/pipeline and the module Workflow.hpp files, and the
//           tests that hand-craft corrupt archives (part of ml_core, no
//           dedicated target).
//
// Contract other modules rely on:
//   * put/get handle enums, arithmetic types, std::string, std::vector<T>
//     and std::pair<A, B> recursively; any other T must provide
//     `void save(std::ostream&) const` and `void load(std::istream&)`.
//   * Doubles are written with 17 significant digits, so a save/load round
//     trip reproduces predictions bit-for-bit (the workflows assert this).
//   * Strings are std::quoted, so they may contain spaces.
//   * Vectors are written as "size" followed by the elements; get() rejects
//     sizes above 10,000,000 to stop corrupt archives from allocating GBs.
//   * Files begin with the line "MLCPP_ARCHIVE_V1"; models usually add their
//     own type tag as the first value so mismatched archives are rejected.
//   * Every stream failure throws std::runtime_error; nothing fails silently.
#pragma once
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ml {
namespace archive {
template <class T> void put(std::ostream &out, const T &value);
template <class T> void get(std::istream &in, T &value);
// Type traits used by put/get to dispatch on containers at compile time.
template <class T> struct is_vector : std::false_type {};
template <class T, class A>
struct is_vector<std::vector<T, A>> : std::true_type {};
template <class T> struct is_pair : std::false_type {};
template <class A, class B> struct is_pair<std::pair<A, B>> : std::true_type {};
// Write one value (recursively for vectors/pairs, via save() otherwise).
// Throws std::runtime_error if the stream is in a failed state afterwards.
template <class T> void put(std::ostream &out, const T &value) {
  if constexpr (std::is_enum_v<T>)
    out << static_cast<int>(value) << '\n';
  else if constexpr (std::is_arithmetic_v<T>)
    out << std::setprecision(17) << value << '\n';
  else if constexpr (std::is_same_v<T, std::string>)
    out << std::quoted(value) << '\n';
  else if constexpr (is_vector<T>::value) {
    out << value.size() << '\n';
    for (const auto &v : value)
      put(out, v);
  } else if constexpr (is_pair<T>::value) {
    put(out, value.first);
    put(out, value.second);
  } else
    value.save(out);
  if (!out)
    throw std::runtime_error("Failed to write model archive");
}
// Read one value written by put(). Mirrors the dispatch above.
// Throws std::runtime_error on a truncated stream or an absurd vector size.
template <class T> void get(std::istream &in, T &value) {
  if constexpr (std::is_enum_v<T>) {
    int v = 0;
    in >> v;
    value = static_cast<T>(v);
  } else if constexpr (std::is_arithmetic_v<T>)
    in >> value;
  else if constexpr (std::is_same_v<T, std::string>)
    in >> std::quoted(value);
  else if constexpr (is_vector<T>::value) {
    size_t n = 0;
    in >> n;
    if (!in || n > 10000000)
      throw std::runtime_error("Invalid archive vector size");
    value.resize(n);
    for (auto &v : value)
      get(in, v);
  } else if constexpr (is_pair<T>::value) {
    get(in, value.first);
    get(in, value.second);
  } else
    value.load(in);
  if (!in)
    throw std::runtime_error("Truncated or invalid model archive");
}
// Variadic conveniences: write(out, a, b, c) / read(in, a, b, c) serialize
// the arguments in order. Models list their members once in each.
template <class... T> void write(std::ostream &out, const T &...value) {
  (put(out, value), ...);
}
template <class... T> void read(std::istream &in, T &...value) {
  (get(in, value), ...);
}
// Save a model to `path`: magic line, then model.save(). Throws
// std::runtime_error if the file cannot be opened or fully flushed.
template <class T> void save_file(const std::string &path, const T &model) {
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("Cannot save: " + path);
  out << "MLCPP_ARCHIVE_V1\n";
  model.save(out);
  out.flush();
  if (!out)
    throw std::runtime_error("Incomplete save: " + path);
}
// Load a model from `path`: verifies the magic line, then model.load().
// A missing file yields an empty magic line and is reported as unsupported.
template <class T> void load_file(const std::string &path, T &model) {
  std::ifstream in(path);
  std::string magic;
  std::getline(in, magic);
  if (magic != "MLCPP_ARCHIVE_V1")
    throw std::runtime_error("Unsupported archive: " + path);
  model.load(in);
  if (!in)
    throw std::runtime_error("Incomplete archive: " + path);
}
} // namespace archive
} // namespace ml
