// 02_data_science/src/plots.cpp
// dsts SVG chart writer: line, scatter, bar, histogram, box and heatmap charts
// plus a 2x2 multi-panel page. No external plotting library is used; every
// chart is assembled as SVG text by a small Picture/Canvas pair.
//
// Reads:   nothing on disk (data arrives as Series / vectors).
// Writes:  one .svg file per write_svg_* call at the caller-supplied path
//          (lessons pass paths under results/<lesson>_results/).
// Build:   part of the dsts static library (target dsts); linked into every
//          02_data_science lesson. Exercised by ctest -R dsts_regression.
// Layout:  standalone pages are 680x420 with a 610x316 plotting rectangle at
//          (62,44); multi-panel pages are 640x500 with four 300x215 panels.
//          All coordinates are deterministic, so the output bytes are stable.
#include "dsts/plots.hpp"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "dsts/eda.hpp"

namespace dsts {

namespace {

// ---------------------------------------------------------------
// small color type + fixed palette of 10 categorical colors
// ---------------------------------------------------------------
// RGB triple (0..255 each). Values outside the range are clamped by hexc().
struct Color {
  int r, g, b;
};

// Categorical colour i, cycling through the 10-entry palette (i % 10).
const Color& palette(size_t i) {
  static const Color kPalette[] = {
      {76, 120, 168},   // blue
      {245, 133, 24},   // orange
      {228, 87, 86},    // red
      {114, 183, 178},  // teal
      {84, 162, 75},    // green
      {238, 202, 59},   // yellow
      {178, 121, 162},  // purple
      {255, 157, 166},  // pink
      {150, 150, 150},  // gray
      {96, 99, 112},    // slate
  };
  return kPalette[i % 10];
}

// Linear interpolation between two colours: t = 0 gives a, t = 1 gives b.
Color mix(const Color& a, const Color& b, double t) {
  return {static_cast<int>(a.r + (b.r - a.r) * t),
          static_cast<int>(a.g + (b.g - a.g) * t),
          static_cast<int>(a.b + (b.b - a.b) * t)};
}

// "#rrggbb" text for an SVG fill/stroke attribute; each channel clamped to 0..255.
std::string hexc(const Color& c) {
  std::ostringstream os;
  os << '#' << std::hex << std::setfill('0')
     << std::setw(2) << std::max(0, std::min(255, c.r))
     << std::setw(2) << std::max(0, std::min(255, c.g))
     << std::setw(2) << std::max(0, std::min(255, c.b)) << std::dec;
  return os.str();
}

// ---------------------------------------------------------------
// Picture: raw SVG builder
// ---------------------------------------------------------------
// Accumulates SVG elements for one page of size w x h (white background).
// Coordinates are absolute page pixels; y grows downwards as in SVG.
class Picture {
 public:
  // Open the <svg> root and paint the white background rectangle.
  Picture(int w, int h) : w_(w), h_(h) {
    out_ << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w
         << "\" height=\"" << h << "\" viewBox=\"0 0 " << w << " " << h
         << "\">\n";
    out_ << "<rect width=\"" << w << "\" height=\"" << h
         << "\" fill=\"white\"/>\n";
  }
  // Filled axis-aligned rectangle with top-left corner (x, y).
  void rect(double x, double y, double w, double h, const Color& fill) {
    out_ << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << w
         << "\" height=\"" << h << "\" fill=\"" << hexc(fill) << "\"/>\n";
  }
  // Straight line segment with the given stroke colour and width.
  void line(double x1, double y1, double x2, double y2, const Color& stroke,
            double stroke_w = 1.0) {
    out_ << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2
         << "\" y2=\"" << y2 << "\" stroke=\"" << hexc(stroke)
         << "\" stroke-width=\"" << stroke_w << "\"/>\n";
  }
  // Open path through pts ("M x y L x y ..."), no fill; used for line charts.
  void polyline(const std::vector<std::pair<double, double>>& pts,
                const Color& stroke, double stroke_w = 2.0) {
    std::ostringstream d;
    for (size_t i = 0; i < pts.size(); ++i) {
      d << (i ? " L" : " M") << pts[i].first << " " << pts[i].second;
    }
    out_ << "<path d=\"" << d.str() << "\" fill=\"none\" stroke=\""
         << hexc(stroke) << "\" stroke-width=\"" << stroke_w
         << "\" stroke-linejoin=\"round\"/>\n";
  }
  // Filled circle centred at (cx, cy); used for markers and legend swatches.
  void circle(double cx, double cy, double r, const Color& fill) {
    out_ << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << r
         << "\" fill=\"" << hexc(fill) << "\"/>\n";
  }
  // Text label; `anchor` is the SVG text-anchor ("start" | "middle" | "end").
  // The string is written unescaped, so callers pass plain labels only.
  void text(double x, double y, const std::string& s, int size,
            const std::string& anchor, const Color& fill) {
    out_ << "<text x=\"" << x << "\" y=\"" << y << "\" font-size=\"" << size
         << "\" font-family=\"sans-serif\" text-anchor=\"" << anchor
         << "\" fill=\"" << hexc(fill) << "\">" << s << "</text>\n";
  }
  // Close the root element and return the complete SVG document.
  std::string save() {
    out_ << "</svg>\n";
    return out_.str();
  }

 private:
  std::ostringstream out_;
  int w_, h_;
};

// Write the SVG text to `path` in binary mode (no newline translation).
void write_file(const std::string& path, const std::string& svg) {
  std::ofstream out(path, std::ios::binary);
  out << svg;
}

// ---------------------------------------------------------------
// nice numbers for axis ticks
// ---------------------------------------------------------------
// Round x up to a "nice" tick step: 1, 2, 5 or 10 times a power of ten.
double nice_num(double x) {
  if (x == 0.0) return 0.0;
  const double exponent = std::floor(std::log10(std::fabs(x)));
  const double frac = x / std::pow(10.0, exponent);
  double nice;
  if (frac < 1.0) nice = 1.0;
  else if (frac < 2.0) nice = 2.0;
  else if (frac < 5.0) nice = 5.0;
  else nice = 10.0;
  return nice * std::pow(10.0, exponent);
}

// Tick positions covering [lo, hi] with at most about max_ticks steps.
// Degenerate ranges (hi <= lo) fall back to a single tick at the midpoint.
// The 1e-9 slack keeps a tick that lands exactly on hi despite rounding.
std::vector<double> ticks(double lo, double hi, int max_ticks) {
  std::vector<double> out;
  double range = hi - lo;
  if (range <= 0.0) range = 1.0;
  const double step =
      nice_num(range / static_cast<double>(std::max(1, max_ticks)));
  if (step == 0.0) {
    out.push_back((lo + hi) / 2.0);
    return out;
  }
  const double first = std::ceil(lo / step) * step;
  for (double v = first; v <= hi + 1e-9; v += step) out.push_back(v);
  if (out.empty()) out.push_back((lo + hi) / 2.0);
  return out;
}

// Tick label using the shared dsts number formatter (same text as printing).
std::string fmt_num(double v) { return fmt(v); }
// Compact number for tick/cell labels: at most 2 decimals, trailing zeros removed.
std::string fmt_short(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f", v);
  std::string t(buf);
  if (t.find('.') != std::string::npos) {
    while (!t.empty() && t.back() == '0') t.pop_back();
    if (!t.empty() && t.back() == '.') t.pop_back();
  }
  if (t == "-0") t = "0";
  return t;
}

// ---------------------------------------------------------------
// Canvas: plotting area with axes, in absolute picture coordinates
// ---------------------------------------------------------------
// One plotting rectangle inside a Picture: origin (ox_, oy_) and size w_ x h_.
// It maps data values and category indices into page coordinates and draws
// titles, axis labels, gridlines and tick labels around the rectangle.
class Canvas {
 public:
  Canvas(Picture& pic, double ox, double oy, double w, double h)
      : pic_(pic), ox_(ox), oy_(oy), w_(w), h_(h) {}

  // Numeric x -> page x; a zero span is treated as 1 to avoid division by zero.
  double map_x_value(double v, double lo, double hi) const {
    const double span = (hi - lo) != 0.0 ? (hi - lo) : 1.0;
    return ox_ + w_ * (v - lo) / span;
  }
  // Category i of n -> page x at the centre of its slot.
  double map_x_cat(size_t i, size_t n) const {
    return ox_ + w_ * (static_cast<double>(i) + 0.5) / static_cast<double>(n);
  }
  // Numeric y -> page y; flipped because SVG y grows downwards.
  double map_y(double v, double lo, double hi) const {
    const double span = (hi - lo) != 0.0 ? (hi - lo) : 1.0;
    return oy_ + h_ * (1.0 - (v - lo) / span);
  }
  // Page y of the x axis line.
  double plot_bottom() const { return oy_ + h_; }

  // Centred title just above the plotting rectangle.
  void title(const std::string& t, int size) const {
    pic_.text(ox_ + w_ / 2.0, oy_ - 12, t, size, "middle", {30, 30, 30});
  }
  // x label below the axis, y label to the left of the rectangle's midpoint.
  void axis_labels(const std::string& xl, const std::string& yl, int size) const {
    pic_.text(ox_ + w_ / 2.0, oy_ + h_ + 24, xl, size, "middle", {60, 60, 60});
    pic_.text(ox_ - 10, oy_ + h_ / 2.0, yl, size, "middle", {60, 60, 60});
  }
  // y axis: light horizontal gridlines with tick labels, plus the two axis lines.
  void frame(double lo, double hi, int max_ticks) const {
    for (double t : ticks(lo, hi, max_ticks)) {
      const double y = map_y(t, lo, hi);
      pic_.line(ox_, y, ox_ + w_, y, {230, 230, 230}, 1.0);
      pic_.text(ox_ - 6, y + 4, fmt_num(t), 11, "end", {60, 60, 60});
    }
    pic_.line(ox_, oy_, ox_, oy_ + h_, {90, 90, 90});
    pic_.line(ox_, oy_ + h_, ox_ + w_, oy_ + h_, {90, 90, 90}, 1.5);
  }
  // x axis with numeric ticks drawn along the bottom line
  void xframe_value(double lo, double hi, int max_ticks) const {
    for (double t : ticks(lo, hi, max_ticks)) {
      const double x = map_x_value(t, lo, hi);
      pic_.text(x, oy_ + h_ + 16, fmt_num(t), 10, "middle", {60, 60, 60});
    }
    pic_.line(ox_, oy_ + h_, ox_ + w_, oy_ + h_, {90, 90, 90}, 1.5);
  }
  // Largest label width (approx. 0.6 * font size per character) decides how
  // many category labels fit without overlapping; the rest are skipped.
  static size_t label_step(const std::vector<std::string>& labels, double w,
                           int font_size) {
    size_t longest = 1;
    for (const auto& l : labels) longest = std::max(longest, l.size());
    const double label_px = static_cast<double>(longest) * font_size * 0.6 + 10.0;
    const size_t fit = std::max<size_t>(1, static_cast<size_t>(w / label_px));
    if (labels.size() <= fit) return 1;
    return (labels.size() + fit - 1) / fit;
  }
  // Category labels along the bottom edge. Every category gets a small tick;
  // labels are drawn only every `step` categories so they stay readable.
  void xlabels(const std::vector<std::string>& labels, int step) const {
    const size_t n = labels.size();
    const size_t auto_step = label_step(labels, w_, 10);
    const size_t use = std::max<size_t>(static_cast<size_t>(std::max(step, 1)), auto_step);
    for (size_t i = 0; i < n; ++i) {
      const double x = map_x_cat(i, n);
      const bool labelled = i % use == 0;
      pic_.line(x, oy_ + h_, x, oy_ + h_ + (labelled ? 5.0 : 2.5), {120, 120, 120}, 1.0);
      if (!labelled) continue;
      pic_.text(x, oy_ + h_ + 17, labels[i], 10, "middle", {60, 60, 60});
    }
  }
  // Category labels with automatic thinning only (step = 1 requested).
  void xlabels_spaced(const std::vector<std::string>& labels) const {
    xlabels(labels, 1);
  }

 private:
  Picture& pic_;
  double ox_, oy_, w_, h_;
};

// Min/max of the present values; [0, 1] when every value is missing.
void data_scale(const std::vector<OptD>& values, double& lo, double& hi) {
  lo = 0.0;
  hi = 1.0;
  bool any = false;
  for (const OptD& v : values) {
    if (!v.has_value()) continue;
    if (!any) {
      lo = hi = *v;
      any = true;
    } else {
      lo = std::min(lo, *v);
      hi = std::max(hi, *v);
    }
  }
}

// Plain doubles from a Series, substituting `missing_as` for empty cells.
std::vector<double> as_doubles(const Series& s, double missing_as = 0.0) {
  std::vector<double> out;
  out.reserve(s.size());
  for (const OptD& v : s.values()) out.push_back(v.value_or(missing_as));
  return out;
}

// ---------------------------------------------------------------
// chart drawing functions (ox, oy, w, h = canvas rectangle)
// ---------------------------------------------------------------
// Line chart of y over categorical x labels. Missing points are skipped
// (the line simply connects the neighbours). 8% vertical padding.
void chart_line(const std::string& title, const std::string& x_label,
                const std::string& y_label,
                const std::vector<std::string>& labels, const Series& y,
                Picture& pic, double ox, double oy, double w, double h,
                bool as_panel) {
  Canvas c(pic, ox, oy, w, h);
  c.title(title, as_panel ? 12 : 15);
  c.axis_labels(x_label, y_label, as_panel ? 10 : 12);
  double lo, hi;
  data_scale(y.values(), lo, hi);
  const double pad = (hi - lo) * 0.08 + 1e-9;
  lo -= pad;
  hi += pad;
  c.frame(lo, hi, 5);
  c.xlabels_spaced(labels);
  const auto values = y.values();
  std::vector<std::pair<double, double>> pts;
  for (size_t i = 0; i < values.size(); ++i) {
    if (!values[i].has_value()) continue;
    pts.push_back({c.map_x_cat(i, values.size()), c.map_y(*values[i], lo, hi)});
  }
  pic.polyline(pts, palette(0), 2.0);
  for (const auto& p : pts) pic.circle(p.first, p.second, 2.6, palette(0));
}

// Scatter of (x, y) pairs; when `groups` has one entry per point, colours follow
// first-appearance order of the group labels and a legend is drawn top-left.
void chart_scatter(const std::string& title, const std::string& x_label,
                   const std::string& y_label, const Series& x, const Series& y,
                   const std::vector<std::string>& groups, Picture& pic,
                   double ox, double oy, double w, double h, bool as_panel) {
  Canvas c(pic, ox, oy, w, h);
  c.title(title, as_panel ? 12 : 15);
  c.axis_labels(x_label, y_label, as_panel ? 10 : 12);
  double xlo, xhi, ylo, yhi;
  data_scale(x.values(), xlo, xhi);
  data_scale(y.values(), ylo, yhi);
  const double xpad = (xhi - xlo) * 0.04 + 1e-9;
  const double ypad = (yhi - ylo) * 0.06 + 1e-9;
  xlo -= xpad;
  xhi += xpad;
  ylo -= ypad;
  yhi += ypad;
  c.frame(ylo, yhi, 5);
  c.xframe_value(xlo, xhi, 5);
  for (size_t i = 0; i < x.size(); ++i) {
    if (!x.at(i).has_value() || !y.at(i).has_value()) continue;
    const double px = c.map_x_value(*x.at(i), xlo, xhi);
    const double py = c.map_y(*y.at(i), ylo, yhi);
    Color col = palette(0);
    if (groups.size() == x.size()) {
      int g = 0;
      for (const auto& seen : groups) {
        if (seen == groups[i]) break;
        ++g;
      }
      col = palette(static_cast<size_t>(g));
    }
    pic.circle(px, py, 4.0, col);
  }
  // legend
  std::vector<std::string> legend;
  for (const auto& g : groups) {
    if (std::find(legend.begin(), legend.end(), g) == legend.end())
      legend.push_back(g);
  }
  for (size_t i = 0; i < legend.size(); ++i) {
    pic.circle(ox + 14, oy + 18 + 18.0 * i, 5.0, palette(i));
    pic.text(ox + 26, oy + 22 + 18.0 * i, legend[i], 11, "start", {60, 60, 60});
  }
}

// Vertical bars from a y baseline of 0; each bar is labelled with its value.
void chart_bar(const std::string& title, const std::string& x_label,
               const std::string& y_label,
               const std::vector<std::string>& cats,
               const std::vector<double>& vals, Picture& pic, double ox,
               double oy, double w, double h, bool as_panel) {
  Canvas c(pic, ox, oy, w, h);
  c.title(title, as_panel ? 12 : 15);
  c.axis_labels(x_label, y_label, as_panel ? 10 : 12);
  double hi = 0.0;
  for (double v : vals) hi = std::max(hi, v);
  hi += hi * 0.08 + 1e-9;
  c.frame(0.0, hi, 5);
  c.xlabels(cats, 1);
  const double slot = w / static_cast<double>(cats.size());
  const double bw = slot * 0.62;
  for (size_t i = 0; i < cats.size(); ++i) {
    const double x = c.map_x_cat(i, cats.size()) - bw / 2.0;
    const double y = c.map_y(vals[i], 0.0, hi);
    pic.rect(x, y, bw, c.plot_bottom() - y, palette(i));
    pic.text(x + bw / 2.0, y - 4, fmt_num(vals[i]), 10, "middle", {60, 60, 60});
  }
}

// Histogram from dsts::histogram(x, bins); bars sit at their left bin edge and
// bin-edge labels are thinned so that they never overlap.
void chart_histogram(const std::string& title, const std::string& x_label,
                     const std::string& y_label, const Series& x,
                     size_t bins, Picture& pic, double ox, double oy, double w,
                     double h, bool as_panel) {
  Canvas c(pic, ox, oy, w, h);
  c.title(title, as_panel ? 12 : 15);
  c.axis_labels(x_label, y_label, as_panel ? 10 : 12);
  const Histogram hist = histogram(x, bins);
  double hi = 0.0;
  for (size_t cnt : hist.counts) hi = std::max(hi, static_cast<double>(cnt));
  hi += 1.0;
  c.frame(0.0, hi, 5);
  const double slot = w / static_cast<double>(hist.counts.size());
  const double bw = slot * 0.92;
  std::vector<std::string> edge_labels;
  for (double e : hist.edges) edge_labels.push_back(fmt_short(e));
  // Label bin edges (left edge of each bar) only as densely as they fit.
  const size_t step = Canvas::label_step(edge_labels, w, 9);
  for (size_t i = 0; i < hist.counts.size(); ++i) {
    const double bx = ox + slot * static_cast<double>(i);
    const double by = c.map_y(static_cast<double>(hist.counts[i]), 0.0, hi);
    pic.rect(bx, by, bw, c.plot_bottom() - by,
             mix(palette(0), {255, 255, 255}, 0.25));
    pic.line(bx, c.plot_bottom(), bx, c.plot_bottom() + (i % step == 0 ? 5.0 : 2.5),
             {120, 120, 120}, 1.0);
    if (i % step != 0) continue;
    pic.text(bx, c.plot_bottom() + 15, edge_labels[i], 9, "middle", {90, 90, 90});
  }
  // Right edge of the last bar: label it only when it will not collide with
  // the last labelled left edge.
  const size_t last_labelled = ((hist.counts.size() - 1) / step) * step;
  const double gap = slot * static_cast<double>(hist.counts.size() - last_labelled);
  if (gap >= static_cast<double>(edge_labels.back().size()) * 9 * 0.6 + 10.0) {
    pic.line(ox + w, c.plot_bottom(), ox + w, c.plot_bottom() + 5.0, {120, 120, 120}, 1.0);
    pic.text(ox + w, c.plot_bottom() + 15, edge_labels.back(), 9, "middle", {90, 90, 90});
  }
}

// One Tukey box per distinct group (first-appearance order): box = Q1..Q3,
// thick line = median, whiskers to the last value inside 1.5*IQR, points beyond.
void chart_box(const std::string& title, const std::string& y_label,
               const std::vector<std::string>& groups,
               const std::vector<OptD>& values, Picture& pic, double ox,
               double oy, double w, double h, bool as_panel) {
  Canvas c(pic, ox, oy, w, h);
  c.title(title, as_panel ? 12 : 15);
  c.axis_labels("", y_label, as_panel ? 10 : 12);
  // distinct groups in first-appearance order
  std::vector<std::string> g_order;
  for (const auto& g : groups) {
    if (std::find(g_order.begin(), g_order.end(), g) == g_order.end())
      g_order.push_back(g);
  }
  std::vector<Series> box_series;
  box_series.reserve(g_order.size());
  for (const auto& g : g_order) {
    std::vector<OptD> vals;
    for (size_t i = 0; i < groups.size(); ++i) {
      if (groups[i] == g) vals.push_back(values[i]);
    }
    box_series.push_back(Series(g, vals));
  }
  double lo = 0.0, hi = 1.0;
  bool any = false;
  for (auto& s : box_series) {
    if (s.count() == 0) continue;
    const double a = s.min(), b = s.max();
    if (!any) {
      lo = a;
      hi = b;
      any = true;
    } else {
      lo = std::min(lo, a);
      hi = std::max(hi, b);
    }
  }
  const double pad = (hi - lo) * 0.06 + 1e-9;
  lo -= pad;
  hi += pad;
  c.frame(lo, hi, 5);
  const double slot = w / static_cast<double>(g_order.size());
  const double bw = slot * 0.5;
  for (size_t gi = 0; gi < g_order.size(); ++gi) {
    const double cx = c.map_x_cat(gi, g_order.size());
    const auto& s = box_series[gi];
    if (s.count() == 0) continue;
    const double q1 = s.quantile(0.25), med = s.quantile(0.50),
                 q3 = s.quantile(0.75);
    const double iq = q3 - q1;
    const double wlo = std::max(s.min(), q1 - 1.5 * iq);
    const double whi = std::min(s.max(), q3 + 1.5 * iq);
    const double x = cx - bw / 2.0;
    const double y1 = c.map_y(q1, lo, hi), y3 = c.map_y(q3, lo, hi);
    pic.rect(x, y3, bw, y1 - y3, mix(palette(gi), {255, 255, 255}, 0.25));
    pic.line(x, y3, x + bw, y3, palette(gi), 1.4);
    pic.line(x, y1, x + bw, y1, palette(gi), 1.4);
    const double ym = c.map_y(med, lo, hi);
    pic.line(x - 2, ym, x + bw + 2, ym, palette(gi), 2.2);
    const double ywlo = c.map_y(wlo, lo, hi), ywhi = c.map_y(whi, lo, hi);
    pic.line(cx, ywhi, cx, y1, palette(gi), 1.4);
    pic.line(cx - 4, ywhi, cx + 4, ywhi, palette(gi), 1.4);
    pic.line(cx, y3, cx, ywlo, palette(gi), 1.4);
    pic.line(cx - 4, ywlo, cx + 4, ywlo, palette(gi), 1.4);
    const auto valsvec = s.values();
    for (const OptD& v : valsvec) {
      if (!v.has_value()) continue;
      if (*v < wlo || *v > whi) pic.circle(cx, c.map_y(*v, lo, hi), 2.8, palette(gi));
    }
    pic.text(cx, oy + h + 16, g_order[gi], 10, "middle", {60, 60, 60});
  }
}

// Square cell grid for a symmetric matrix in [-1, 1]: negative values shade
// towards red, positive towards blue, |v| = 1 is fully saturated.
void chart_heatmap(const std::string& title,
                   const std::vector<std::string>& labels,
                   const std::vector<std::vector<double>>& matrix,
                   Picture& pic, double ox, double oy, double w, double h,
                   bool as_panel) {
  Canvas c(pic, ox, oy, w, h);
  c.title(title, as_panel ? 12 : 15);
  const size_t n = labels.size();
  const double cell =
      std::min(w / static_cast<double>(n), h / static_cast<double>(n));
  const double base_x = ox + 8;
  const double base_y = oy + 8;
  const Color lo = {228, 87, 86};
  const Color hi = {76, 120, 168};
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const double v = matrix[i][j];
      const double t = v < 0.0 ? std::min(1.0, -v) : std::min(1.0, v);
      const Color cell_col =
          v < 0.0 ? mix(lo, {255, 255, 255}, 1.0 - t)
                  : mix(hi, {255, 255, 255}, 1.0 - t);
      const double x = base_x + static_cast<double>(j) * cell;
      const double y = base_y + static_cast<double>(i) * cell;
      pic.rect(x, y, cell, cell, cell_col);
      pic.text(x + cell / 2.0, y + cell / 2.0 + 4, fmt_short(v), 10, "middle",
               {40, 40, 40});
    }
  }
  for (size_t i = 0; i < n; ++i) {
    pic.text(base_x - 6, base_y + cell * (static_cast<double>(i) + 0.5) + 4,
             labels[i], 10, "end", {60, 60, 60});
    pic.text(base_x + cell * (static_cast<double>(i) + 0.5),
             base_y + cell * n + 16, labels[i], 10, "middle", {60, 60, 60});
  }
  (void)as_panel;
}

}  // namespace

// ---------------------------------------------------------------
// public entry points - standalone pages (680x420)
// ---------------------------------------------------------------
// Standalone line chart page; returns `path` for chaining into log messages.
std::string write_svg_line(const std::string& path, const std::string& title,
                           const std::string& x_label,
                           const std::string& y_label,
                           const std::vector<std::string>& x_labels,
                           const Series& y) {
  Picture pic(680, 420);
  chart_line(title, x_label, y_label, x_labels, y, pic, 62, 44, 610, 316,
             false);
  write_file(path, pic.save());
  return path;
}

// Standalone scatter page (optional per-point group colouring).
std::string write_svg_scatter(const std::string& path, const std::string& title,
                              const std::string& x_label,
                              const std::string& y_label, const Series& x,
                              const Series& y,
                              const std::vector<std::string>& groups) {
  Picture pic(680, 420);
  chart_scatter(title, x_label, y_label, x, y, groups, pic, 62, 44, 610, 316,
                false);
  write_file(path, pic.save());
  return path;
}

// Standalone bar chart page.
std::string write_svg_bar(const std::string& path, const std::string& title,
                          const std::string& x_label,
                          const std::string& y_label,
                          const std::vector<std::string>& categories,
                          const std::vector<double>& values) {
  Picture pic(680, 420);
  chart_bar(title, x_label, y_label, categories, values, pic, 62, 44, 610, 316,
            false);
  write_file(path, pic.save());
  return path;
}

// Standalone histogram page with `bins` equal-width bins.
std::string write_svg_histogram(const std::string& path,
                                const std::string& title,
                                const std::string& x_label,
                                const std::string& y_label, const Series& x,
                                size_t bins) {
  Picture pic(680, 420);
  chart_histogram(title, x_label, y_label, x, bins, pic, 62, 44, 610, 316,
                  false);
  write_file(path, pic.save());
  return path;
}

// Standalone box plot page; `groups[i]` labels `values[i]`.
std::string write_svg_box(const std::string& path, const std::string& title,
                          const std::string& y_label,
                          const std::vector<std::string>& groups,
                          const std::vector<OptD>& values) {
  Picture pic(680, 420);
  chart_box(title, y_label, groups, values, pic, 62, 44, 610, 316, false);
  write_file(path, pic.save());
  return path;
}

// Standalone heatmap page (e.g. a correlation matrix).
std::string write_svg_heatmap(const std::string& path, const std::string& title,
                              const std::vector<std::string>& labels,
                              const std::vector<std::vector<double>>& matrix) {
  Picture pic(680, 420);
  chart_heatmap(title, labels, matrix, pic, 62, 44, 610, 316, false);
  write_file(path, pic.save());
  return path;
}

// Up to four panels on one 640x500 page laid out in a 2x2 grid (row-major).
// Panel kinds: "line", "scatter", "bar", "hist" (10 bins); others are skipped.
std::string write_svg_multi(const std::string& path, const std::string& title,
                            const std::vector<Panel>& panels) {
  Picture pic(640, 500);
  pic.text(320, 22, title, 15, "middle", {30, 30, 30});
  const double pw = 300, ph = 215;
  const double ox[4] = {10, 10 + pw + 12, 10, 10 + pw + 12};
  const double oy[4] = {38, 38, 38 + ph + 16, 38 + ph + 16};
  for (size_t p = 0; p < panels.size() && p < 4; ++p) {
    const Panel& panel = panels[p];
    if (panel.kind == "line") {
      chart_line(panel.title, panel.x_label, panel.y_label, panel.x_labels,
                 panel.y, pic, ox[p], oy[p], pw, ph, true);
    } else if (panel.kind == "scatter") {
      chart_scatter(panel.title, panel.x_label, panel.y_label, panel.x,
                    panel.y, panel.groups, pic, ox[p], oy[p], pw, ph, true);
    } else if (panel.kind == "bar") {
      chart_bar(panel.title, panel.x_label, panel.y_label, panel.x_labels,
                as_doubles(panel.y), pic, ox[p], oy[p], pw, ph, true);
    } else if (panel.kind == "hist") {
      chart_histogram(panel.title, panel.x_label, panel.y_label, panel.x, 10,
                      pic, ox[p], oy[p], pw, ph, true);
    }
  }
  write_file(path, pic.save());
  return path;
}

}  // namespace dsts