// 03_ml_course/helper/reporting/artifacts.cpp
//
// Purpose : implementation of the Artifacts sandbox, CSV/JSON quoting,
//           the dataset fingerprint and the standard EDA report. See
//           artifacts.hpp for the path rules every lesson relies on.
// Inputs  : see artifacts.hpp; eda() takes an in-memory ml::Dataset.
// Outputs : files below the Artifacts root; eda() writes data/schema.json,
//           <prefix>/findings.md, <prefix>/tables/{descriptive_statistics,
//           correlations,grouped_analysis}.csv and <prefix>/figures/*.svg.
// Target  : compiled into the ml_core static library (no lesson target).
#include "helper/reporting/artifacts.hpp"
#include "helper/math/metrics.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ml {
// CSV field quoting: always quote, double any embedded quote. O(length).
std::string csv_quote(const std::string &s) {
  std::string r = "\"";
  for (char c : s) {
    if (c == '\"')
      r += '\"';
    r += c;
  }
  return r + '\"';
}
// JSON string quoting: escape quote/backslash/newline/CR/tab and drop other
// control bytes (< 32) so the output is always valid JSON. O(length).
std::string json_quote(const std::string &s) {
  std::string r = "\"";
  for (unsigned char c : s) {
    if (c == '\"' || c == '\\') {
      r += '\\';
      r += static_cast<char>(c);
    } else if (c == '\n')
      r += "\\n";
    else if (c == '\r')
      r += "\\r";
    else if (c == '\t')
      r += "\\t";
    else if (c >= 32)
      r += static_cast<char>(c);
  }
  return r + '\"';
}
// Resolve <root>/<name>. The run name must be relative and must not contain
// ".." so a run can never be placed outside the results root. "." is a
// valid name meaning "write into root itself" (used by lessons).
Artifacts::Artifacts(const std::string &root, const std::string &name) {
  auto base = std::filesystem::absolute(root).lexically_normal();
  const std::filesystem::path run(name);
  if (name.empty() || run.has_root_name() || run.has_root_directory())
    throw std::invalid_argument("Invalid run name");
  for (const auto &p : std::filesystem::path(name))
    if (p == "..")
      throw std::invalid_argument("Run escapes results folder");
  root_ = base / name;
  std::filesystem::create_directories(root_);
}
// Same guards for every file path, then create the parent directory so the
// caller can open the file immediately. Returned with forward slashes.
std::string Artifacts::path(const std::string &rel) const {
  std::filesystem::path p(rel);
  if (p.empty() || p.has_root_name() || p.has_root_directory())
    throw std::invalid_argument("Artifact requires relative path");
  for (const auto &part : p)
    if (part == "..")
      throw std::invalid_argument("Artifact escapes run directory");
  auto result = root_ / p;
  std::filesystem::create_directories(result.parent_path());
  return result.generic_string();
}
// Binary mode keeps '\n' as-is on Windows so CSV/JSON bytes are identical
// across platforms; flush before checking so a disk-full error is caught.
void Artifacts::write(const std::string &rel, const std::string &text) const {
  std::ofstream out(path(rel), std::ios::binary);
  out << text;
  out.flush();
  if (!out)
    throw std::runtime_error("Cannot write artifact " + rel);
}
// Render a Plot to SVG text and store it under `rel` via write() (same
// directory-creation and overwrite rules as any other artifact).
void Artifacts::figure(const std::string &rel, const Plot &p) const {
  write(rel, p.render());
}
// Serialise names, every X cell (17 digits) and y into one byte string and
// hash it with 64-bit FNV-1a. Any change in the data changes the hash.
// Complexity: O(n * p).
std::string fingerprint(const Dataset &d) {
  std::ostringstream data;
  data << std::setprecision(17);
  for (const auto &n : d.feature_names)
    data << n << '\0';
  for (size_t i = 0; i < d.n(); ++i) {
    for (double v : d.X[i])
      data << v << ',';
    if (i < d.y.size())
      data << d.y[i];
    data << '\n';
  }
  uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : data.str()) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  std::ostringstream out;
  out << std::hex << h;
  return "fnv1a64:" + out.str();
}
// Standard EDA report (see artifacts.hpp for the list of outputs).
// Missing (NaN) cells are excluded from every statistic; correlations use
// pairwise-complete rows. Complexity: O(n * p^2) for the correlation matrix.
void eda(const Dataset &d, const Artifacts &a, const std::string &prefix) {
  if (d.n() == 0 || d.p() == 0)
    throw std::invalid_argument("EDA requires nonempty data");
  std::ostringstream table, notes, schema;
  table
      << std::setprecision(17)
      << "feature,count,missing,min,q25,median,q75,max,mean,std,iqr_outliers\n";
  notes << "# Data understanding\n\nRows: " << d.n() << ". Features: " << d.p()
        << ". Target: " << d.target_name << ".\n\n";
  schema << "{\"features\":[";
  Mat columns(d.p());
  Vec mus(d.p()), sds(d.p());
  for (size_t j = 0; j < d.p(); ++j) {
    std::string name = j < d.feature_names.size() ? d.feature_names[j]
                                                  : "x" + std::to_string(j);
    if (j)
      schema << ',';
    schema << json_quote(name);
    Vec v;
    for (const auto &r : d.X) {
      if (r.size() != d.p())
        throw std::invalid_argument("Ragged EDA matrix");
      if (std::isfinite(r[j]))
        v.push_back(r[j]);
    }
    columns[j] = v;
    if (v.empty()) {
      notes << "- " << name
            << ": entirely missing; training must reject or explicitly remove "
               "this feature.\n";
      continue;
    }
    std::sort(v.begin(), v.end());
    // Linear-interpolated quantile on the sorted finite values.
    auto q = [&](double f) {
      double pos = f * static_cast<double>(v.size() - 1);
      size_t lo = static_cast<size_t>(pos);
      return v[lo] + (v[std::min(lo + 1, v.size() - 1)] - v[lo]) *
                         (pos - static_cast<double>(lo));
    };
    // Tukey rule: outside [q1 - 1.5 IQR, q3 + 1.5 IQR] counts as an outlier.
    double q1 = q(.25), med = q(.5), q3 = q(.75), iqr = q3 - q1;
    size_t outliers = 0;
    for (double x : v)
      if (x < q1 - 1.5 * iqr || x > q3 + 1.5 * iqr)
        ++outliers;
    mus[j] = ml::mean(v);
    sds[j] = std::sqrt(variance(v, true));
    table << csv_quote(name) << ',' << v.size() << ',' << d.n() - v.size()
          << ',' << v.front() << ',' << q1 << ',' << med << ',' << q3 << ','
          << v.back() << ',' << mus[j] << ',' << sds[j] << ',' << outliers
          << '\n';
    notes << "- " << name << ": " << d.n() - v.size() << " missing, "
          << outliers << " IQR outliers. Mean " << mus[j] << ", median " << med
          << ". "
          << (sds[j] == 0
                  ? "Constant feature."
                  : "Investigate domain validity before removing outliers.")
          << "\n";
    Plot hist;
    hist.title(name + " distribution");
    hist.histogram(v, 24);
    a.figure(prefix + "/figures/feature_" + std::to_string(j) + ".svg", hist);
  }
  schema << "],\"target\":" << json_quote(d.target_name) << "}\n";
  a.write("data/schema.json", schema.str());
  a.write(prefix + "/tables/descriptive_statistics.csv", table.str());
  // Pearson correlation on rows where both features are finite; a constant
  // column yields 0 rather than NaN.
  Mat corr(d.p(), Vec(d.p()));
  for (size_t i = 0; i < d.p(); ++i)
    for (size_t j = 0; j < d.p(); ++j) {
      Vec x, y;
      for (const auto &row : d.X)
        if (std::isfinite(row[i]) && std::isfinite(row[j])) {
          x.push_back(row[i]);
          y.push_back(row[j]);
        }
      double mx = ml::mean(x), my = ml::mean(y), num = 0, sx = 0, sy = 0;
      for (size_t r = 0; r < x.size(); ++r) {
        double dx = x[r] - mx, dy = y[r] - my;
        num += dx * dy;
        sx += dx * dx;
        sy += dy * dy;
      }
      corr[i][j] = (sx > 0 && sy > 0) ? num / std::sqrt(sx * sy) : 0;
    }
  Plot heat(1000, 800);
  heat.title("Pairwise complete feature correlations");
  heat.heatmap(corr, d.feature_names, d.feature_names, -1, 1);
  a.figure(prefix + "/figures/correlation.svg", heat);
  std::ostringstream ct;
  ct << "feature_i,feature_j,pearson\n";
  for (size_t i = 0; i < d.p(); ++i)
    for (size_t j = 0; j < d.p(); ++j)
      ct << i << ',' << j << ',' << corr[i][j] << '\n';
  a.write(prefix + "/tables/correlations.csv", ct.str());
  // Exact duplicate feature rows, detected on the 17-digit text form.
  std::set<std::string> seen;
  size_t dup = 0;
  for (const auto &row : d.X) {
    std::ostringstream s;
    s << std::setprecision(17);
    for (double v : row)
      s << v << ',';
    if (!seen.insert(s.str()).second)
      ++dup;
  }
  notes << "\nDuplicate feature rows: " << dup
        << ". Duplicate features do not necessarily mean duplicate "
           "observations.\n";
  if (d.y.size() == d.n()) {
    Plot target;
    target.title("Target distribution");
    target.histogram(d.y, 24);
    a.figure(prefix + "/figures/target.svg", target);
    // Treat targets with <= 20 distinct values as classes and report
    // per-class feature means; continuous targets skip this table.
    std::map<double, size_t> counts;
    for (double v : d.y)
      ++counts[v];
    if (counts.size() <= 20) {
      std::ostringstream grouped;
      grouped << "class,count,feature,mean\n";
      for (const auto &kv : counts)
        for (size_t j = 0; j < d.p(); ++j) {
          Vec v;
          for (size_t i = 0; i < d.n(); ++i)
            if (d.y[i] == kv.first && std::isfinite(d.X[i][j]))
              v.push_back(d.X[i][j]);
          grouped << kv.first << ',' << kv.second << ',' << j << ','
                  << ml::mean(v) << '\n';
        }
      a.write(prefix + "/tables/grouped_analysis.csv", grouped.str());
    }
    // Feature-versus-target scatter plots, capped at the first 8 features.
    for (size_t j = 0; j < std::min<size_t>(d.p(), 8); ++j) {
      Vec x, y;
      for (size_t i = 0; i < d.n(); ++i)
        if (std::isfinite(d.X[i][j])) {
          x.push_back(d.X[i][j]);
          y.push_back(d.y[i]);
        }
      Plot p;
      p.title("Feature " + std::to_string(j) + " versus target");
      p.scatter(x, y);
      a.figure(
          prefix + "/figures/target_relation_" + std::to_string(j) + ".svg", p);
    }
  }
  notes << "\nCorrelation is descriptive, not causal. Supervised workflows "
           "produce this EDA on training rows only; preprocessing is "
           "independently fitted inside each CV fold.\n";
  a.write(prefix + "/findings.md", notes.str());
}
} // namespace ml
