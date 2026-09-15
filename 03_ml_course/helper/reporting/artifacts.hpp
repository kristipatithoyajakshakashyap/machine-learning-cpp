// 03_ml_course/helper/reporting/artifacts.hpp
//
// Purpose : the only way lessons write files. Artifacts turns a results
//           root plus a run name into a sandboxed directory and offers
//           write()/figure() for text and SVG; eda() and fingerprint()
//           produce the standard data-understanding outputs.
// Inputs  : root - normally the RUN_OUTPUT_DIR compile definition
//           (results/<stem>_results of the calling lesson); name - the run
//           sub-folder ("full", "quick", or "." for the root itself).
// Outputs : files under <root>/<name>/... created on demand; eda() writes
//           data/schema.json, <prefix>/findings.md, <prefix>/tables/*.csv
//           and <prefix>/figures/*.svg.
// Used by : every lesson, workflow and module Workflow.hpp (implemented in
//           artifacts.cpp, part of ml_core; no dedicated target).
//
// Path rules other modules rely on (enforced in artifacts.cpp):
//   * The run name and every relative path must be relative, non-empty and
//     free of ".." components; anything else throws std::invalid_argument,
//     so a lesson can never write outside its own results folder.
//   * path() creates parent directories, so callers may use nested names
//     such as "evaluation/figures/roc.svg" without preparing folders.
//   * write() is binary and flushes; a failed write throws, never silently
//     produces a partial file.
//   * The convention results/<stem>_results/{full,quick}/ comes from the
//     callers: Artifacts(root, quick ? "quick" : "full").
#pragma once
#include "helper/data/datasets.hpp"
#include "helper/plot/plot_svg.hpp"
#include <filesystem>
#include <string>

namespace ml {
// Wrap text in double quotes, doubling embedded quotes (RFC 4180 CSV).
std::string csv_quote(const std::string &text);
// Wrap text in double quotes with JSON escapes for quote, backslash, newline,
// carriage return and tab; other control bytes are dropped.
std::string json_quote(const std::string &text);
// Sandboxed output directory for one run of one lesson.
class Artifacts {
public:
  // Create <root>/<name> (and parents). Throws on an invalid run name.
  Artifacts(const std::string &root, const std::string &name);
  // Absolute, forward-slash path for a relative file; creates its parent
  // directories. Throws on an absolute or escaping path.
  std::string path(const std::string &relative) const;
  // Write text to path(relative). Throws std::runtime_error on failure.
  void write(const std::string &relative, const std::string &text) const;
  // Render a Plot to SVG and write it to path(relative).
  void figure(const std::string &relative, const Plot &plot) const;

private:
  std::filesystem::path root_;
};
// Exploratory data analysis of a dense dataset: per-feature descriptive
// statistics, IQR outlier counts, histograms, a correlation heatmap and
// table, duplicate-row count, target distribution and (for <= 20 classes)
// per-class feature means. Supervised workflows call this on training rows
// only. Throws std::invalid_argument on empty or ragged data.
void eda(const Dataset &data, const Artifacts &artifacts,
         const std::string &prefix = "eda");
// Deterministic 64-bit FNV-1a hash of feature names, X and y, formatted as
// "fnv1a64:<hex>". Written into run_manifest.json so a run can be tied to
// the exact data it saw.
std::string fingerprint(const Dataset &data);
} // namespace ml
