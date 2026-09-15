// 07_multi_panel.cpp
// Lesson: a 2x2 grid of panels puts several chart types into one figure:
// line (passengers), scatter (iris), bar (average bill by day) and a
// histogram (tip size). The Panel struct describes each sub-plot.
// Equivalent: matplotlib.pyplot.subplots, seaborn.FacetGrid.
//
// Data: flights.csv, iris.csv, tips.csv (real data).
//
// EXPECTED OUTPUT:
// wrote 07_multi_grid.svg (29555 bytes)

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/eda.hpp"
#include "dsts/plots.hpp"
#include <filesystem>

size_t file_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  return f ? static_cast<size_t>(f.tellg()) : 0;
}

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
  const dsts::DataFrame flights = dsts::read_csv(DATA_DIR "/flights.csv");
  const dsts::Series passengers("passengers",
                                flights.numeric("passengers"));
  std::vector<std::string> months;
  for (size_t i = 0; i < flights.rows(); i += 24) {
    months.push_back(flights.strings("month")[i] + " " +
                     dsts::fmt(*flights.numeric("year")[i]));
  }

  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const dsts::Series pl("petal_length", iris.numeric("petal_length"));
  const dsts::Series pw("petal_width", iris.numeric("petal_width"));
  const auto& species = iris.strings("species");

  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const dsts::Series tip("tip", tips.numeric("tip"));
  const auto days = dsts::group_mean(tips, "day", "total_bill");
  std::vector<std::string> day_names;
  std::vector<double> day_means;
  for (const auto& d : days) {
    day_names.push_back(d.group);
    day_means.push_back(d.mean);
  }

  dsts::Panel line;
  line.kind = "line";
  line.title = "Air passengers";
  line.x_label = "Sample";
  line.y_label = "Passengers";
  line.x_labels = months;
  line.y = passengers;

  dsts::Panel scatter;
  scatter.kind = "scatter";
  scatter.title = "Iris petals";
  scatter.x_label = "Petal length";
  scatter.y_label = "Petal width";
  scatter.x = pl;
  scatter.y = pw;
  for (size_t i = 0; i < iris.rows(); ++i) scatter.groups.push_back(species[i]);

  dsts::Panel bar;
  bar.kind = "bar";
  bar.title = "Average bill by day";
  bar.x_label = "Day";
  bar.y_label = "Total bill ($)";
  bar.x_labels = day_names;
  std::vector<dsts::OptD> day_mean_opts;
  for (double m : day_means) day_mean_opts.push_back(m);
  bar.y = dsts::Series("mean", day_mean_opts);

  dsts::Panel hist;
  hist.kind = "hist";
  hist.title = "Tip sizes";
  hist.x_label = "Tip ($)";
  hist.y_label = "Count";
  hist.x = tip;

  const std::string name = "07_multi_grid.svg";
  const std::string out = std::string(RUN_OUTPUT_DIR) + "/" + name;
  dsts::write_svg_multi(out.c_str(), "Four views of the data",
                        {line, scatter, bar, hist});

  std::cout << "wrote " << name << " (" << file_bytes(out) << " bytes)\n";
  return 0;
}