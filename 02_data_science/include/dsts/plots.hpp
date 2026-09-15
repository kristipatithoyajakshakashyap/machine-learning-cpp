#ifndef DSTS_PLOTS_HPP
#define DSTS_PLOTS_HPP

// Minimal dependency-free SVG chart writer (the matplotlib/seaborn
// equivalent of this file). Every chart is a standalone .svg file with a
// deterministic layout, so the exact same bytes come out on every run.

#include <string>
#include <vector>

#include "dsts/dataframe.hpp"
#include "dsts/series.hpp"

namespace dsts {

struct Panel {
  std::string kind;                 // "line" | "scatter" | "bar" | "hist"
  std::string title, x_label, y_label;
  std::vector<std::string> x_labels;  // categorical x (line / bar)
  Series x;                           // scatter x (or histogram source)
  Series y;                           // line/scatter/hist values
  std::vector<std::string> groups;    // scatter group per point
};

// All functions write a file and return its path for convenience.
std::string write_svg_line(const std::string& path, const std::string& title,
                           const std::string& x_label,
                           const std::string& y_label,
                           const std::vector<std::string>& x_labels,
                           const Series& y);

std::string write_svg_scatter(const std::string& path,
                              const std::string& title,
                              const std::string& x_label,
                              const std::string& y_label, const Series& x,
                              const Series& y,
                              const std::vector<std::string>& groups);

std::string write_svg_bar(const std::string& path, const std::string& title,
                          const std::string& x_label,
                          const std::string& y_label,
                          const std::vector<std::string>& categories,
                          const std::vector<double>& values);

std::string write_svg_histogram(const std::string& path,
                                const std::string& title,
                                const std::string& x_label,
                                const std::string& y_label, const Series& x,
                                size_t bins);

std::string write_svg_box(const std::string& path, const std::string& title,
                          const std::string& y_label,
                          const std::vector<std::string>& groups,
                          const std::vector<OptD>& values);

std::string write_svg_heatmap(const std::string& path,
                              const std::string& title,
                              const std::vector<std::string>& labels,
                              const std::vector<std::vector<double>>& matrix);

std::string write_svg_multi(const std::string& path, const std::string& title,
                            const std::vector<Panel>& panels);

}  // namespace dsts

#endif  // DSTS_PLOTS_HPP