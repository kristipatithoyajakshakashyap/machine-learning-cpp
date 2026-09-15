// 01_components.cpp
// Lesson: log-and-plot the raw series, then overlay a 12-month moving
// average to separate the rising trend from the seasonal wobble.
// Equivalent: matplotlib line plot + pandas.rolling(12).mean().
//
// Data: air_passengers.csv (real data, 144 monthly totals, 1949-1960).
//
// EXPECTED OUTPUT:
// rows 144  mean 280.2986  recent (1960) 417 vs first (1949) 112
//
// 12-month moving average (last year)  passengers  ma12
//   1960-02   391   437.1667
//   1960-03   419   438.25
//   1960-04   461   443.6667
//   1960-05   472   448
//   1960-06   535   453.25
//   1960-07   622   459.4167
//   1960-08   606   463.3333
//   1960-09   508   467.0833
//   1960-10   461   471.5833
//   1960-11   390   473.9167
//   1960-12   432   476.1667
//
// wrote <results/01_components_results>/m09_01_passengers.svg

#include <iostream>

#include "dsts/csv.hpp"
#include "dsts/plots.hpp"
#include "dsts/series.hpp"
#include "dsts/timeseries.hpp"
#include <filesystem>

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
  const dsts::DataFrame ap = dsts::read_csv(DATA_DIR "/air_passengers.csv");
  const dsts::Series p("passengers", ap.numeric("Passengers"));

  std::cout << "rows " << p.values().size() << "  mean "
            << dsts::fmt(p.mean()) << "  recent (1960) "
            << dsts::fmt(*p.values()[132]) << " vs first (1949) "
            << dsts::fmt(*p.values()[0]) << "\n\n";

  const dsts::Series ma = dsts::rolling_mean(p, 12);
  std::cout << "12-month moving average (last year)  passengers  ma12\n";
  for (size_t r = 133; r < 144; ++r) {
    std::cout << "  " << ap.strings("Month")[r] << "   " << dsts::fmt(*p.values()[r])
              << "   " << dsts::fmt(*ma.values()[r]) << "\n";
  }

  dsts::write_svg_line(RUN_OUTPUT_DIR "/m09_01_passengers.svg",
                       "Air passengers 1949-1960", "month", "passengers",
                       ap.strings("Month"), p);
  std::cout << "\nwrote " RUN_OUTPUT_DIR "/m09_01_passengers.svg\n";
  return 0;
}