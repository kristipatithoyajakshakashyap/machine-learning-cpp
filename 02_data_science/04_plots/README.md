# Module 04 — Data visualisation (SVG charts)

## Purpose
Every lesson writes a standalone `.svg` file into the lesson's `results/<source stem>_results/`
folder. The layout is deterministic, so the same bytes are produced on every
run. The chart engine lives in `include/dsts/plots.hpp` / `src/plots.cpp`.

## Prerequisites

Module 03_eda (the statistics each chart visualises).

## Lessons

| File | Target | What it teaches | Python equivalent | What it writes |
|------|--------|-----------------|-------------------|----------------|
| `01_line_chart.cpp` | `s04_01_line` | Line plot of a time series (flights); data: flights.csv (real data, 1949-1960) | `matplotlib.pyplot.plot` | `01_line_flights.svg` under `results/01_line_chart_results/` |
| `02_scatter_plot.cpp` | `s04_02_scatter` | XY scatter coloured by group (iris); data: iris.csv (real data, 150 flowers of three species) | `plt.scatter`, `seaborn.scatterplot` | `02_scatter_iris.svg` under `results/02_scatter_plot_results/` |
| `03_bar_chart.cpp` | `s04_03_bar` | One value per category as bars (day means); data: tips.csv (real data, 244 restaurant tips) | `plt.bar`, `seaborn.barplot` | `03_bar_tips.svg` under `results/03_bar_chart_results/` |
| `04_histogram.cpp` | `s04_04_histogram` | Distribution of a numeric column; data: tips.csv and iris.csv (real data) | `plt.hist` | `04_hist_iris.svg`, `04_hist_tips.svg` under `results/04_histogram_results/` |
| `05_box_plot.cpp` | `s04_05_box` | Quartiles, median and 1.5×IQR outliers per group; data: iris.csv (real data, 150 flowers of three species) | `plt.boxplot` | `05_box_iris.svg` under `results/05_box_plot_results/` |
| `06_heatmap.cpp` | `s04_06_heatmap` | Pearson correlation grid coloured red/blue; data: iris.csv and tips.csv (real data) | `seaborn.heatmap` | `06_heatmap_iris.svg`, `06_heatmap_tips.svg` under `results/06_heatmap_results/` |
| `07_multi_panel.cpp` | `s04_07_multi` | 2×2 figure mixing several chart types; data: flights.csv, iris.csv, tips.csv (real data) | `plt.subplots` | `07_multi_grid.svg` under `results/07_multi_panel_results/` |

## Build and run

Configure once from the repository root, build a lesson target, then run the
executable from `build/02_data_science/04_plots/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s04_01_line
.\build\02_data_science\04_plots\s04_01_line.exe
```

Every lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes any output files
to `results/<source stem>_results/` next to its source (injected as
`RUN_OUTPUT_DIR`, created on first run).

## dsts functions used

Types: `dsts::DataFrame`, `dsts::OptD`, `dsts::Panel`, `dsts::Series`.

- `01_line_chart.cpp`: `read_csv`, `write_svg_line`
- `02_scatter_plot.cpp`: `pearson`, `read_csv`, `write_svg_scatter`
- `03_bar_chart.cpp`: `fmt`, `group_mean`, `read_csv`, `write_svg_bar`
- `04_histogram.cpp`: `read_csv`, `write_svg_histogram`
- `05_box_plot.cpp`: `read_csv`, `write_svg_box`
- `06_heatmap.cpp`: `correlation_matrix`, `read_csv`, `write_svg_heatmap`
- `07_multi_panel.cpp`: `fmt`, `group_mean`, `read_csv`, `write_svg_multi`

## Key ideas
- **Encoding matters** — position (line/scatter) is the most readable channel;
  colour separates groups; area/count (bar/histogram) emphasises magnitude.
- **Box plots** compress an entire distribution into five numbers plus
  outliers, making cross-group comparison cheap.
- **Correlation heatmaps** highlight strong relations at a glance; remember
  the caveat from module 03: correlation is not causation.

## Key takeaways
- Every chart is a deterministic SVG written by `dsts` with no external library.
- Position encodes numbers best; colour separates groups; area emphasises magnitude.
- A box plot is five numbers plus outliers; a heatmap is a correlation matrix you can read at a glance.

## Next module

**05_joins_reshape** - merge, inner vs left joins, concat, pivot and melt.
