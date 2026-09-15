#include "helper/plot/plot_svg.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ml {

namespace {

// Ordered colour palette (cycles when more series are added than colours).
const char* PALETTE[] = {"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                         "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                         "#bcbd22", "#17becf"};
constexpr size_t NPAL = 10;

// XML-escape text that lands inside SVG markup.
std::string esc(const std::string& s) {
  std::string o;
  o.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '&': o += "&amp;"; break;
      case '<': o += "&lt;"; break;
      case '>': o += "&gt;"; break;
      case '"': o += "&quot;"; break;
      default: o += c;
    }
  }
  return o;
}

// A "nice" tick step so that ~6 ticks span the range (1/2/5 x 10^k).
double nice_step(double range) {
  if (!(range > 0.0)) return 1.0;
  double rough = range / 6.0;
  double pow10 = std::pow(10.0, std::floor(std::log10(rough)));
  double m = rough / pow10;
  double step = (m < 1.5) ? 1.0 : (m < 3.0) ? 2.0 : (m < 7.0) ? 5.0 : 10.0;
  return step * pow10;
}

struct Ticks {
  std::vector<double> values;
  int precision = 0;
};

Ticks make_ticks(double lo, double hi) {
  Ticks t;
  double step = nice_step(hi - lo);
  double start = std::ceil(lo / step) * step;
  for (double v = start; v <= hi + 1e-9 * std::max(1.0, std::abs(hi));
       v += step)
    t.values.push_back(v);
  if (t.values.size() < 2) {
    t.values = {lo, hi};
    t.precision = 4;
    return t;
  }
  if (step < 1.0) {
    t.precision = static_cast<int>(std::ceil(-std::log10(step)));
    if (t.precision > 6) t.precision = 6;
  } else {
    t.precision = 0;
  }
  return t;
}

std::string fmt_double(double v, int prec) {
  if (v == 0.0) v = 0.0;  // normalise -0.0 so labels show "0", not "-0"
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*f", prec, v);
  std::string s = buf;
  if (prec > 0) {
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
  }
  return s;
}

// Diverging red-white-blue colormap for heatmaps: t in [0,1] -> a CSS colour.
// Negative values get reds, zero gets white, positive values get blues, so a
// correlation matrix is readable at a glance.
std::string diverging(double t) {
  struct Stop {
    double t, r, g, b;
  };
  static const Stop STOPS[] = {
      {0.00, 0.40, 0.00, 0.10},  // deep red
      {0.25, 0.85, 0.20, 0.20},
      {0.50, 1.00, 1.00, 1.00},  // white at the midpoint
      {0.75, 0.30, 0.55, 0.90},
      {1.00, 0.05, 0.20, 0.60},  // deep blue
  };
  constexpr size_t N = sizeof(STOPS) / sizeof(STOPS[0]);
  if (t <= STOPS[0].t) t = 0.0;
  if (t >= STOPS[N - 1].t) t = STOPS[N - 1].t;
  size_t i = 0;
  while (i + 1 < N && STOPS[i + 1].t < t) ++i;
  double span = STOPS[i + 1].t - STOPS[i].t;
  double w = span == 0.0 ? 0.0 : (t - STOPS[i].t) / span;
  auto mix = [&](double a, double b) { return a + (b - a) * w; };
  char buf[64];
  std::snprintf(buf, sizeof buf, "#%02x%02x%02x",
                static_cast<int>(mix(STOPS[i].r, STOPS[i + 1].r) * 255 + 0.5),
                static_cast<int>(mix(STOPS[i].g, STOPS[i + 1].g) * 255 + 0.5),
                static_cast<int>(mix(STOPS[i].b, STOPS[i + 1].b) * 255 + 0.5));
  return buf;
}

// Truncate a long label so heatmap cell annotations stay on screen.
std::string clip_label(const std::string& s, size_t max_len) {
  if (s.size() <= max_len) return s;
  return s.substr(0, max_len - 1) + "...";
}

}  // namespace

// ===========================================================================
//  Plot  (builder + renderer)
// ===========================================================================

Plot::Plot(double width, double height)
    : width_(width), height_(height) {
  if (width_ <= 0.0 || height_ <= 0.0)
    throw std::runtime_error("plot: canvas must have positive size");
}

void Plot::scatter(const Vec& x, const Vec& y, const std::string& label) {
  if (x.size() != y.size())
    throw std::runtime_error("plot::scatter: x and y differ in length");
  ScatterItem it;
  it.xy.resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    it.xy[i] = {x[i], y[i]};
  }
  it.label = label;
  scatters_.push_back(std::move(it));
}

void Plot::scatter(const Mat& xy, const Vec& cls, const std::string& label) {
  if (xy.size() && xy[0].size() != 2)
    throw std::runtime_error("plot::scatter: expected n x 2 point matrix");
  if (!cls.empty() && cls.size() != xy.size())
    throw std::runtime_error("plot::scatter: class vector length mismatch");
  ScatterItem it;
  it.xy = xy;
  it.cls = cls;
  it.label = label;
  scatters_.push_back(std::move(it));
}

void Plot::scatter(const Mat& xy, const std::string& label) {
  if (xy.size() && xy[0].size() != 2)
    throw std::runtime_error("plot::scatter: expected n x 2 point matrix");
  ScatterItem it;
  it.xy = xy;
  it.label = label;
  scatters_.push_back(std::move(it));
}

void Plot::line(const Vec& x, const Vec& y, const std::string& label) {
  if (x.size() != y.size())
    throw std::runtime_error("plot::line: x and y differ in length");
  LineItem it;
  std::vector<size_t> order(x.size());
  for (size_t i = 0; i < x.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return x[a] < x[b]; });
  it.xy.resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    size_t k = order[i];
    it.xy[i] = {x[k], y[k]};
  }
  it.label = label;
  lines_.push_back(std::move(it));
}

void Plot::bar(const Vec& heights, const std::vector<std::string>& labels,
               const std::string& label) {
  if (!labels.empty() && labels.size() != heights.size())
    throw std::runtime_error("plot::bar: labels and heights differ in length");
  BarItem it;
  it.heights = heights;
  it.labels = labels;
  it.label = label;
  bars_.push_back(std::move(it));
}

void Plot::histogram(const Vec& values, size_t bins, const std::string& label) {
  if (bins == 0) throw std::runtime_error("plot::histogram: bins must be > 0");
  HistItem it;
  it.values = values;
  it.bins = bins;
  it.label = label;
  hists_.push_back(std::move(it));
}

void Plot::heatmap(const Mat& values,
                   const std::vector<std::string>& xlabels,
                   const std::vector<std::string>& ylabels, double vmin,
                   double vmax) {
  if (values.empty() || values[0].empty())
    throw std::runtime_error("plot::heatmap: empty value matrix");
  size_t c0 = values[0].size();
  for (const auto& row : values)
    if (row.size() != c0)
      throw std::runtime_error("plot::heatmap: ragged row found");
  if (!xlabels.empty() && xlabels.size() != c0)
    throw std::runtime_error("plot::heatmap: x labels length mismatch");
  if (!ylabels.empty() && ylabels.size() != values.size())
    throw std::runtime_error("plot::heatmap: y labels length mismatch");
  HeatItem it;
  it.values = values;
  it.xlabels = xlabels;
  it.ylabels = ylabels;
  it.vmin = vmin;
  it.vmax = vmax;
  heatmaps_.push_back(std::move(it));
}

std::string Plot::render() const {
  // ------------------------------------------------------------------ layout
  const double L = 64.0, R = 24.0, T = 36.0, B = 52.0;
  const double plotW = width_ - L - R;
  const double plotH = height_ - T - B;
  if (plotW <= 0.0 || plotH <= 0.0)
    throw std::runtime_error("plot: canvas too small for its axes");

  // ------------------------------------------------------------- data ranges
  auto minmax = [](const std::vector<double>& v, bool* ok) {
    double lo = 0, hi = 0;
    bool first = true;
    for (double x : v) {
      if (first) { lo = hi = x; first = false; }
      lo = std::min(lo, x);
      hi = std::max(hi, x);
    }
    *ok = !first;
    return std::make_pair(lo, hi);
  };

  bool anyX = false, anyY = false;
  double xlo = 0, xhi = 0, ylo = 0, yhi = 0;
  bool categorical = false;
  double catN = 0;

  auto extend_x = [&](double a, double b) {
    if (!anyX) { xlo = a; xhi = b; anyX = true; }
    else { xlo = std::min(xlo, a); xhi = std::max(xhi, b); }
  };
  auto extend_y = [&](double a, double b) {
    if (!anyY) { ylo = a; yhi = b; anyY = true; }
    else { ylo = std::min(ylo, a); yhi = std::max(yhi, b); }
  };

  for (const auto& s : scatters_) {
    for (const auto& p : s.xy) {
      extend_x(p[0], p[0]);
      extend_y(p[1], p[1]);
    }
  }
  for (const auto& l : lines_) {
    for (const auto& p : l.xy) {
      extend_x(p[0], p[0]);
      extend_y(p[1], p[1]);
    }
  }

  // Bars must appear above the 0 line on the y axis.
  for (const auto& b : bars_) {
    for (size_t i = 0; i < b.heights.size(); ++i) {
      extend_y(0.0, b.heights[i]);
      if (b.heights[i] < 0.0) extend_y(b.heights[i], 0.0);
    }
    if (b.heights.empty()) extend_y(0.0, 1.0);
  }

  // Histogram counts; x domain is the value range itself.
  for (const auto& h : hists_) {
    if (h.values.empty()) continue;
    bool ok = false;
    auto mm = minmax(h.values, &ok);
    extend_x(mm.first, mm.second);
    std::vector<double> counts(h.bins, 0.0);
    for (double v : h.values) {
      if (h.bins == 1) { counts[0] += 1.0; continue; }
      size_t bin = static_cast<size_t>((v - mm.first) / (mm.second - mm.first) *
                                       static_cast<double>(h.bins));
      if (bin == h.bins) bin = h.bins - 1;
      counts[bin] += 1.0;
    }
    for (double c : counts) extend_y(0.0, c);
  }

  // Categorical mode: bars get the x axis; number placement at integer slots.
  size_t nCat = 0;
  for (const auto& b : bars_) nCat = std::max(nCat, b.heights.size());
  if (nCat > 0 && !anyX) categorical = true;
  if (categorical) {
    catN = static_cast<double>(nCat);
    xlo = -0.5;
    xhi = catN - 0.5;
    anyX = true;
    // Bars must appear above the 0 line on the y axis.
    extend_y(0.0, yhi);
  }

  if (!anyX) { xlo = 0.0; xhi = 1.0; }
  if (!anyY) { ylo = 0.0; yhi = 1.0; }

  // 5% padding (categorical x keeps default padding so bars fill the range).
  double padX = (xhi - xlo) * 0.05;
  double padY = (yhi - ylo) * 0.05;
  if (padX == 0.0) padX = 1.0;
  if (padY == 0.0) padY = 1.0;
  if (!categorical) { xlo -= padX; xhi += padX; }
  ylo -= padY;
  yhi += padY;

  // -------------------------------------------------------------- transforms
  auto sx = [&](double px) { return L + (px - xlo) / (xhi - xlo) * plotW; };
  auto sy = [&](double py) { return T + plotH - (py - ylo) / (yhi - ylo) * plotH; };

  Ticks tx = categorical ? Ticks{} : make_ticks(xlo, xhi);
  Ticks ty = make_ticks(ylo, yhi);

  // Each series type gets its own colour base (scatters first, lines next,
  // then bars, then histograms), so a scatter and a line on the same plot
  // always use different colours.  The renderer and the legend both index
  // with these offsets, keeping swatches in perfect agreement with the drawn
  // geometry.
  const size_t nScat = scatters_.size();
  const size_t nLine = lines_.size();
  const size_t nBar = bars_.size();

  // ------------------------------------------------------------- legend data
  // Each entry remembers what it depicts (a dot for scattered points, a stroke
  // for a line, a filled bar for bars / histograms) so the legend swatch looks
  // like the geometry it names.  Class-coloured scatters expand into one
  // colour-keyed entry per distinct class, so the legend actually keys the
  // colours on screen.
  enum class EntryKind { Dot, Line, Rect, Faded };
  struct LegendEntry { std::string label; std::string colour; EntryKind kind; };
  std::vector<LegendEntry> legend;
  // Colours must match the renderer, which cycles the palette per series type
  // (PALETTE[k % NPAL] for the k-th line / scatter / bar / histogram) and not
  // across types - a running counter here would offset every line, bar and
  // histogram colour by the number of scatters drawn before it.
  for (size_t k = 0; k < scatters_.size(); ++k) {
    const auto& s = scatters_[k];
    if (!s.cls.empty()) {
      std::vector<size_t> codes;
      for (double c : s.cls) {
        size_t ci = static_cast<size_t>(std::max(0.0, c));
        if (std::find(codes.begin(), codes.end(), ci) == codes.end()) codes.push_back(ci);
      }
      std::sort(codes.begin(), codes.end());
      if (codes.size() >= 2 && codes.size() <= 8) {
        for (size_t ci : codes) {
          std::string lab = ci < class_labels_.size() ? class_labels_[ci] : "";
          if (lab.empty()) lab = "class " + std::to_string(ci);
          legend.push_back({lab, PALETTE[ci % NPAL], EntryKind::Dot});
        }
      } else if (!s.label.empty()) {
        legend.push_back({s.label, PALETTE[k % NPAL], EntryKind::Dot});
      }
    } else {
      if (!s.label.empty()) legend.push_back({s.label, PALETTE[k % NPAL], EntryKind::Dot});
    }
  }
  for (size_t k = 0; k < lines_.size(); ++k)
    if (!lines_[k].label.empty())
      legend.push_back({lines_[k].label, PALETTE[(k + nScat) % NPAL], EntryKind::Line});
  for (size_t k = 0; k < bars_.size(); ++k)
    if (!bars_[k].label.empty())
      legend.push_back({bars_[k].label, PALETTE[(k + nScat + nLine) % NPAL], EntryKind::Rect});
  for (size_t k = 0; k < hists_.size(); ++k)
    if (!hists_[k].label.empty())
      legend.push_back({hists_[k].label, PALETTE[(k + nScat + nLine + nBar) % NPAL], EntryKind::Faded});

  // --------------------------------------------------------------------- svg
  std::ostringstream o;
  o << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  o << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
    << "width=\"" << width_ << "\" height=\"" << height_ << "\" "
    << "viewBox=\"0 0 " << width_ << " " << height_ << "\">\n";
  o << "  <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n";

  // Title.
  if (!title_.empty())
    o << "  <text x=\"" << (L + plotW / 2.0) << "\" y=\"" << (T / 2.0 + 2)
      << "\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\" "
      << "fill=\"#111111\">" << esc(title_) << "</text>\n";

  // ---------------------------------------------- heatmap-only rendering
  // A heatmap is its own figure: when present it is the only content.
  if (!heatmaps_.empty()) {
    const auto& hm = heatmaps_[0];
    size_t nR = hm.values.size();
    size_t nC = hm.values[0].size();
    double xlo = hm.vmin;
    double xhi = hm.vmax;
    if (xhi <= xlo) {  // auto range from the data
      xlo = hm.values[0][0];
      xhi = hm.values[0][0];
      for (const auto& row : hm.values)
        for (double v : row) {
          xlo = std::min(xlo, v);
          xhi = std::max(xhi, v);
        }
      if (xhi == xlo) xhi = xlo + 1.0;
    }
    const double hL = 70.0, hR = 76.0, hT = 42.0, hB = 50.0;
    const double hpW = width_ - hL - hR;
    const double hpH = height_ - hT - hB;
    const double cw = hpW / static_cast<double>(nC);
    const double chh = hpH / static_cast<double>(nR);
    bool labels = nR <= 14 && nC <= 14 && hpW / nC >= 26.0 && hpH / nR >= 16.0;

    o << "  <line x1=\"" << hL << "\" y1=\"" << (hT + hpH) << "\" x2=\""
      << (hL + hpW) << "\" y2=\"" << (hT + hpH)
      << "\" stroke=\"#111111\" stroke-width=\"1\"/>\n";
    o << "  <line x1=\"" << hL << "\" y1=\"" << hT << "\" x2=\"" << hL
      << "\" y2=\"" << (hT + hpH) << "\" stroke=\"#111111\" stroke-width=\"1\"/>\n";

    for (size_t r = 0; r < nR; ++r) {
      for (size_t c = 0; c < nC; ++c) {
        double v = hm.values[r][c];
        double t = (v - xlo) / (xhi - xlo);
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        o << "  <rect x=\"" << (hL + c * cw) << "\" y=\"" << (hT + r * chh)
          << "\" width=\"" << (cw - 0.5) << "\" height=\"" << (chh - 0.5)
          << "\" fill=\"" << diverging(t) << "\"/>\n";
      }
    }

    // Print the numeric value in every cell that is big enough to hold text,
    // so a correlation matrix reads colour-wise AND numerically.  Small
    // matrices (correlation tables, confusion matrices) always show numbers;
    // dense grids (many features) keep colours only.
    const double minCellW = 28.0, minCellH = 16.0;
    if (cw >= minCellW && chh >= minCellH && nR * nC <= 144) {
      int prec = (xhi - xlo) < 10.0 ? 2 : 0;
      double fs = std::max(7.0, std::min(10.0, std::min(cw, chh) * 0.5));
      for (size_t rr = 0; rr < nR; ++rr) {
        for (size_t cc = 0; cc < nC; ++cc) {
          o << "  <text x=\"" << (hL + cc * cw + cw / 2.0) << "\" y=\""
            << (hT + rr * chh + chh / 2.0 + fs * 0.36) << "\" text-anchor=\"middle\" "
            << "font-size=\"" << fs << "\" fill=\"#111111\">"
            << fmt_double(hm.values[rr][cc], prec) << "</text>\n";
        }
      }
    }
    if (labels) {
      for (size_t c = 0; c < nC; ++c) {
        std::string lab = c < hm.xlabels.size()
                              ? clip_label(hm.xlabels[c], 12)
                              : std::to_string(c + 1);
        o << "  <text x=\"" << (hL + c * cw + cw / 2.0) << "\" y=\""
          << (hT + hpH + 18) << "\" text-anchor=\"middle\" font-size=\"10\" "
          << "fill=\"#333333\">" << esc(lab) << "</text>\n";
      }
      for (size_t r = 0; r < nR; ++r) {
        std::string lab = r < hm.ylabels.size()
                              ? clip_label(hm.ylabels[r], 12)
                              : std::to_string(r + 1);
        o << "  <text x=\"" << (hL - 6) << "\" y=\"" << (hT + r * chh + chh / 2.0 + 3.5)
          << "\" text-anchor=\"end\" font-size=\"10\" fill=\"#333333\">"
          << esc(lab) << "</text>\n";
      }
    }

    // Vertical colour bar with min / mid / max annotations, right of the grid.
    const double cbarX = hL + hpW + 14.0;
    const double cbarW = 14.0;
    const size_t NSTEPS = 40;
    for (size_t s = 0; s < NSTEPS; ++s) {
      double t0 = static_cast<double>(s) / static_cast<double>(NSTEPS);
      double t1 = static_cast<double>(s + 1) / static_cast<double>(NSTEPS);
      o << "  <rect x=\"" << cbarX << "\" y=\"" << (hT + hpH - t1 * hpH)
        << "\" width=\"" << cbarW << "\" height=\"" << ((t1 - t0) * hpH)
        << "\" fill=\"" << diverging(t0) << "\"/>\n";
    }
    o << "  <text x=\"" << (cbarX + cbarW / 2.0) << "\" y=\"" << (hT + hpH + 18)
      << "\" text-anchor=\"middle\" font-size=\"10\" fill=\"#333333\">"
      << fmt_double(xlo, 3) << "</text>\n";
    o << "  <text x=\"" << (cbarX + cbarW / 2.0) << "\" y=\"" << (hT + 3)
      << "\" text-anchor=\"middle\" font-size=\"10\" fill=\"#333333\">"
      << fmt_double(xhi, 3) << "</text>\n";

    o << "</svg>\n";
    return o.str();
  }

  // Plot frame + grid.
  o << "  <rect x=\"" << L << "\" y=\"" << T << "\" width=\"" << plotW
    << "\" height=\"" << plotH
    << "\" fill=\"none\" stroke=\"#999999\" stroke-width=\"1\"/>\n";

  for (double v : ty.values) {
    o << "  <line x1=\"" << L << "\" y1=\"" << sy(v) << "\" x2=\""
      << (L + plotW) << "\" y2=\"" << sy(v)
      << "\" stroke=\"#e0e0e0\" stroke-width=\"1\"/>\n";
  }
  if (!categorical) {
    for (double v : tx.values) {
      o << "  <line x1=\"" << sx(v) << "\" y1=\"" << T << "\" x2=\""
        << sx(v) << "\" y2=\"" << (T + plotH)
        << "\" stroke=\"#e0e0e0\" stroke-width=\"1\"/>\n";
    }
  }

  // Axes.
  o << "  <line x1=\"" << L << "\" y1=\"" << (T + plotH) << "\" x2=\""
    << (L + plotW) << "\" y2=\"" << (T + plotH)
    << "\" stroke=\"#111111\" stroke-width=\"1\"/>\n";
  o << "  <line x1=\"" << L << "\" y1=\"" << T << "\" x2=\"" << L
    << "\" y2=\"" << (T + plotH) << "\" stroke=\"#111111\" stroke-width=\"1\"/>\n";

  // Y tick labels.
  for (double v : ty.values) {
    o << "  <text x=\"" << (L - 6) << "\" y=\"" << (sy(v) + 3.5)
      << "\" text-anchor=\"end\" font-size=\"10\" fill=\"#333333\">"
      << fmt_double(v, ty.precision) << "</text>\n";
  }

  // X tick labels.
  if (categorical) {
    for (size_t i = 0; i < nCat; ++i) {
      if (i >= bars_.size()) break;
      const auto& labels = bars_[i].labels;
      if (labels.size() != nCat) continue;
      o << "  <text x=\"" << sx(static_cast<double>(i)) << "\" y=\""
        << (T + plotH + 16) << "\" text-anchor=\"middle\" font-size=\"10\" "
        << "fill=\"#333333\">" << esc(labels[i]) << "</text>\n";
    }
  } else {
    for (double v : tx.values) {
      o << "  <text x=\"" << sx(v) << "\" y=\"" << (T + plotH + 16)
        << "\" text-anchor=\"middle\" font-size=\"10\" fill=\"#333333\">"
        << fmt_double(v, tx.precision) << "</text>\n";
    }
  }

  // X / Y axis labels.
  if (!xlabel_.empty())
    o << "  <text x=\"" << (L + plotW / 2.0) << "\" y=\"" << (height_ - 10)
      << "\" text-anchor=\"middle\" font-size=\"12\" fill=\"#111111\">"
      << esc(xlabel_) << "</text>\n";
  if (!ylabel_.empty())
    o << "  <text x=\"16\" y=\"" << (T + plotH / 2.0)
      << "\" text-anchor=\"middle\" font-size=\"12\" fill=\"#111111\" "
      << "transform=\"rotate(-90 16 " << (T + plotH / 2.0) << ")\">"
      << esc(ylabel_) << "</text>\n";

  // ------------------------------------------------------------------ series
  // Histograms (numeric bins) first, so lines overlay them.
  for (size_t k = 0; k < hists_.size(); ++k) {
    const auto& h = hists_[k];
    if (h.values.empty()) continue;
    bool ok = false;
    auto mm = minmax(h.values, &ok);
    double span = mm.second - mm.first;
    if (span == 0.0) span = 1.0;
    std::vector<double> counts(h.bins, 0.0);
    for (double v : h.values) {
      if (h.bins == 1) { counts[0] += 1.0; continue; }
      size_t bin = static_cast<size_t>((v - mm.first) / span *
                                       static_cast<double>(h.bins));
      if (bin == h.bins) bin = h.bins - 1;
      counts[bin] += 1.0;
    }
    double bw = plotW / static_cast<double>(h.bins);
    for (size_t i = 0; i < counts.size(); ++i) {
      double x0 = mm.first + span * (static_cast<double>(i) / static_cast<double>(h.bins));
      double wpx = std::max(0.5, bw - 1.0);
      o << "  <rect x=\"" << (sx(x0) - wpx / 2.0) << "\" y=\"" << sy(counts[i])
        << "\" width=\"" << wpx << "\" height=\"" << (sy(0.0) - sy(counts[i]))
        << "\" fill=\"" << PALETTE[(k + nScat + nLine + nBar) % NPAL]
        << "\" opacity=\"0.65\"/>\n";
    }
  }

  // Category bars.
  for (size_t k = 0; k < bars_.size(); ++k) {
    const auto& b = bars_[k];
    for (size_t i = 0; i < b.heights.size(); ++i) {
      double wpx = std::max(2.0, plotW / static_cast<double>(nCat) * 0.55);
      double x0 = sx(static_cast<double>(i));
      double yBase = sy(0.0);
      double yTop = sy(b.heights[i]);
      if (yTop > yBase) std::swap(yTop, yBase);
      o << "  <rect x=\"" << (x0 - wpx / 2.0) << "\" y=\"" << yTop
        << "\" width=\"" << wpx << "\" height=\"" << (yBase - yTop)
        << "\" fill=\"" << PALETTE[(k + nScat + nLine) % NPAL] << "\" opacity=\"0.8\"/>\n";
    }
  }

  // Lines.  Offset by the number of scatters so lines never share a colour
  // with a scatter on the same figure.
  for (size_t k = 0; k < lines_.size(); ++k) {
    const auto& ln = lines_[k];
    std::ostringstream pts;
    pts << "  <polyline points=\"";
    for (size_t i = 0; i < ln.xy.size(); ++i) {
      if (i > 0) pts << " ";
      pts << sx(ln.xy[i][0]) << "," << sy(ln.xy[i][1]);
    }
    pts << "\" fill=\"none\" stroke=\"" << PALETTE[(k + nScat) % NPAL]
        << "\" stroke-width=\"2\" stroke-linejoin=\"round\"/>";
    o << pts.str() << "\n";
  }

  // Scatter points.
  for (size_t k = 0; k < scatters_.size(); ++k) {
    const auto& s = scatters_[k];
    bool coloured = !s.cls.empty();
    for (size_t i = 0; i < s.xy.size(); ++i) {
      std::string col = PALETTE[k % NPAL];
      if (coloured)
        col = PALETTE[static_cast<size_t>(s.cls[i]) % NPAL];
      o << "  <circle cx=\"" << sx(s.xy[i][0]) << "\" cy=\"" << sy(s.xy[i][1])
        << "\" r=\"4\" fill=\"" << col
        << "\" fill-opacity=\"0.85\""
        << " stroke=\"#ffffff\" stroke-width=\"0.5\"/>\n";
    }
  }

  // ------------------------------------------------------------- legend box
  // The box is placed in the corner that covers the fewest drawn points so it
  // never hides the data it explains (ties prefer top-right).
  if (!legend.empty()) {
    const double padL = 12.0, padV = 8.0, rowH = 16.0, swW = 20.0, gap = 6.0;
    double maxW = 0.0;
    for (const auto& e : legend)
      maxW = std::max(maxW, static_cast<double>(e.label.size()) * 6.5);
    const double boxW = padL + swW + gap + maxW + padL;
    const double boxH = padV * 2.0 + static_cast<double>(legend.size()) * rowH;

    double cx[4], cy[4];
    cx[0] = L + plotW - boxW - padL;  cy[0] = T + padV;                  // top-right
    cx[1] = L + padL;                 cy[1] = T + padV;                  // top-left
    cx[2] = L + plotW - boxW - padL;  cy[2] = T + plotH - boxH - padV;   // bottom-right
    cx[3] = L + padL;                 cy[3] = T + plotH - boxH - padV;   // bottom-left

    size_t best = 0;
    size_t bestHit = std::numeric_limits<size_t>::max();
    for (size_t ci = 0; ci < 4; ++ci) {
      size_t hit = 0;
      const double bx1 = cx[ci], by1 = cy[ci], bx2 = cx[ci] + boxW, by2 = cy[ci] + boxH;
      auto in_box = [&](double px, double py) {
        return px >= bx1 && px <= bx2 && py >= by1 && py <= by2;
      };
      for (const auto& s : scatters_)
        for (const auto& pt : s.xy)
          if (in_box(sx(pt[0]), sy(pt[1]))) ++hit;
      for (const auto& ln : lines_)
        for (const auto& pt : ln.xy)
          if (in_box(sx(pt[0]), sy(pt[1]))) ++hit;
      if (hit < bestHit) { bestHit = hit; best = ci; }
    }
    const double lx = cx[best], ly = cy[best];

    o << "  <rect x=\"" << lx << "\" y=\"" << ly << "\" width=\"" << boxW
      << "\" height=\"" << boxH << "\" fill=\"#ffffff\" fill-opacity=\"0.88\" "
      << "stroke=\"#cccccc\" stroke-width=\"1\"/>\n";
    double iy = ly + padV + 11.0;
    for (const auto& e : legend) {
      const double sw = lx + padL;
      if (e.kind == EntryKind::Line) {
        o << "  <line x1=\"" << sw << "\" y1=\"" << (iy - 4.0) << "\" x2=\""
          << (sw + swW) << "\" y2=\"" << (iy - 4.0) << "\" stroke=\""
          << e.colour << "\" stroke-width=\"2\"/>\n";
      } else if (e.kind == EntryKind::Rect || e.kind == EntryKind::Faded) {
        o << "  <rect x=\"" << sw << "\" y=\"" << (iy - 9.0) << "\" width=\"14\" "
          << "height=\"10\" fill=\"" << e.colour << "\" opacity=\""
          << (e.kind == EntryKind::Faded ? 0.65 : 0.8) << "\"/>\n";
      } else {
        o << "  <circle cx=\"" << (sw + 7.0) << "\" cy=\"" << (iy - 4.0)
          << "\" r=\"4\" fill=\"" << e.colour << "\" fill-opacity=\"0.85\"/>\n";
      }
      o << "  <text x=\"" << (sw + swW + gap) << "\" y=\"" << iy
        << "\" font-size=\"11\" fill=\"#111111\">" << esc(e.label) << "</text>\n";
      iy += rowH;
    }
  }

  o << "</svg>\n";
  return o.str();
}

}  // namespace ml