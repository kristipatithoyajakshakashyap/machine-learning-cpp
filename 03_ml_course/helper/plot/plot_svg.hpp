#ifndef HELPER_PLOT_PLOT_SVG_HPP
#define HELPER_PLOT_PLOT_SVG_HPP

// ===========================================================================
// helper/plot/plot_svg.hpp  (+ the matching helper/plot/plot_svg.cpp)
// ---------------------------------------------------------------------------
// A tiny, dependency-free chart library that renders plain-SVG figures, used
// by lessons to draw results into their module's results/ folder.
//
// The API is deliberately matplotlib-ish: you build one Plot, add series
// (line / scatter / bar / histogram), set title and axis labels, then call
// render() and write the returned string to a .svg file.  Nothing is shown
// on screen - the output is a self-contained SVG document any browser (or
// image viewer) can open.
//
// Supported:
//   line(x, y)                       - polyline through the points
//   scatter(x, y)                    - one-colour point cloud
//   scatter(xy, cls)                 - points coloured by class (cluster plots)
//   bar(heights, labels)             - category bars (x axis shows the labels)
//   histogram(values, bins)          - frequency bars over a numeric range
//   heatmap(values, xl, yl, vmin, vmax)
//                                    - diverging red-white-blue heatmap cells
//
// Every series is stored as vector geometry; render() computes nice axis
// limits, ticks and a legend, exactly the way a plotting library should.
// ===========================================================================

#include <cstddef>
#include <string>
#include <vector>

#include "helper/math/matrix.hpp"

namespace ml {

class Plot {
 public:
  // Canvas size, in SVG user units.  Aspect ratio is preserved on open.
  explicit Plot(double width = 640.0, double height = 400.0);

  // -- series ------------------------------------------------------------
  // Scatter of the points (x[i], y[i]), all in one colour.
  void scatter(const Vec& x, const Vec& y, const std::string& label = "");

  // Scatter of the 2-D points xy (n x 2), coloured by class id cls[i].
  void scatter(const Mat& xy, const Vec& cls, const std::string& label = "");

  // Scatter of the 2-D points xy (n x 2), one colour (a scatter of the
  // first two columns of a dataset drops naturally into this overload).
  void scatter(const Mat& xy, const std::string& label = "");

  // Line plot through (x[i], y[i]) in sorted x order.
  void line(const Vec& x, const Vec& y, const std::string& label = "");

  // Vertical bars; labels[i] is drawn under the i-th bar.
  void bar(const Vec& heights, const std::vector<std::string>& labels,
           const std::string& label = "");

  // Histogram (frequency counts) of `values` split into `bins` equal bins.
  void histogram(const Vec& values, size_t bins = 10,
                 const std::string& label = "");

  // Heatmap of the matrix `values` (n_rows x n_cols), one coloured cell per
  // entry on a diverging red-white-blue scale.  Cells below vmin are clamped
  // to vmin and cells above vmax to vmax; when vmax <= vmin the range is
  // derived from the data instead.  Optional per-row / per-column labels are
  // drawn when the matrix is small enough to stay legible.
  void heatmap(const Mat& values,
               const std::vector<std::string>& xlabels = {},
               const std::vector<std::string>& ylabels = {},
               double vmin = 0.0, double vmax = 0.0);

  // -- figure furniture --------------------------------------------------
  void title(const std::string& t) { title_ = t; }
  void xlabel(const std::string& t) { xlabel_ = t; }
  void ylabel(const std::string& t) { ylabel_ = t; }

  // Class names for the legend of class-coloured scatters: the legend then
  // shows one colour-keyed entry per class ("benign", "malignant", species
  // names, ...).  class_labels[i] names class code i; codes without a name
  // fall back to "class <code>".
  void class_labels(const std::vector<std::string>& names) { class_labels_ = names; }

  // Render the current figure as a complete SVG document (a <svg> root with
  // everything inside).  Write the result to a .svg file to view it.
  std::string render() const;

 private:
  struct ScatterItem {
    Mat xy;            // n x 2 points
    Vec cls;           // class id per point (empty => single colour)
    std::string label;
  };
  struct LineItem {
    Mat xy;            // n x 2 polyline vertices
    std::string label;
  };
  struct BarItem {
    Vec heights;
    std::vector<std::string> labels;
    std::string label;
  };
  struct HistItem {
    Vec values;
    size_t bins;
    std::string label;
  };
  struct HeatItem {
    Mat values;
    std::vector<std::string> xlabels;
    std::vector<std::string> ylabels;
    double vmin = 0.0, vmax = 0.0;
  };

  double width_;
  double height_;
  std::string title_;
  std::string xlabel_;
  std::string ylabel_;
  std::vector<std::string> class_labels_;
  std::vector<ScatterItem> scatters_;
  std::vector<LineItem> lines_;
  std::vector<BarItem> bars_;
  std::vector<HistItem> hists_;
  std::vector<HeatItem> heatmaps_;
};

}  // namespace ml

#endif  // HELPER_PLOT_PLOT_SVG_HPP