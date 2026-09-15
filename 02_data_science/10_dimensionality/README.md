# Module 10 — Dimensionality reduction (PCA)

## Purpose
The iris dataset has 4 axes. PCA finds the 4 axes that maximally spread
the points (eigenvectors of the covariance matrix, via Jacobi rotations in
`include/dsts/pca.hpp`) and ranks them by how much of the original
variation they capture.

## Prerequisites

Module 02_cleaning (scaling) and 00_math_foundations/07_eigen_svd.

## Lessons

| File | Target | What it teaches | Python equivalent | What it writes |
|------|--------|-----------------|-------------------|----------------|
| `01_explained_variance.cpp` | `s10_01_eigval` | Scree: explained ratios + cumulative %; data: iris.csv (real measurements, 150 flowers, 4 traits) | `PCA().explained_variance_ratio_` | `m10_01_scree.svg` under `results/01_explained_variance_results/` |
| `02_projection.cpp` | `s10_02_proj` | First 6 rows in PC1-PC2; SVG scatter by species; data: iris.csv (real measurements, 150 flowers) | `fit_transform` | `m10_02_pca_scatter.svg` under `results/02_projection_results/` |
| `03_reconstruction.cpp` | `s10_03_reconstruct` | Loss from 1 to 4 kept PCs (RMSE on original scale); data: iris.csv (real measurements, 150 flowers, 4 traits) | `inverse_transform` | prints only |
| `04_loadings.cpp` | `s10_04_loadings` | Variable weights per PC; data: iris.csv (real measurements, 150 flowers) | `PCA().components_` | prints only |
| `05_standardize.cpp` | `s10_05_standardize` | Raw PCA vs z-scored PCA; data: iris.csv (real measurements, 150 flowers; all four traits are in mm) | `StandardScaler().fit_transform` then PCA | prints only |

## Build and run

Configure once from the repository root, build a lesson target, then run the
executable from `build/02_data_science/10_dimensionality/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s10_01_eigval
.\build\02_data_science\10_dimensionality\s10_01_eigval.exe
```

Every lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes any output files
to `results/<source stem>_results/` next to its source (injected as
`RUN_OUTPUT_DIR`, created on first run).

## dsts functions used

Types: `dsts::DataFrame`, `dsts::OptD`, `dsts::PcaResult`, `dsts::Series`.

- `01_explained_variance.cpp`: `fmt`, `pca`, `read_csv`, `write_svg_bar`
- `02_projection.cpp`: `fmt`, `pca`, `read_csv`, `write_svg_scatter`
- `03_reconstruction.cpp`: `fmt`, `pca`, `read_csv`
- `04_loadings.cpp`: `fmt`, `pca`, `read_csv`
- `05_standardize.cpp`: `fmt`, `pca`, `read_csv`

## Key ideas
- **First two PCs keep 97.8% of the variance** — iris really lives on a 2D
  plane dominated by petal traits.
- **Loadings say what matters** — PC1 is petal-heavy (0.857 on petal_length);
  PC2 adds sepal-width in the opposite direction (-0.730 vs +0.657).
- **The reconstruction shows what was lost** — k=1 recovers 0.29 RMSE;
  k=3 drops to 0.08, almost exact. k=4 is machine-zero.
- **Raw vs standardized** — scaling removes the petal-length dominance; PC1
  drops from 92.5% to 73.0%, giving all traits an equal voice.

## Key takeaways
- PCA rotates the data onto orthogonal axes ordered by explained variance.
- Loadings say which original variables drive each component.
- Standardise first unless one variable's scale should dominate.

## Next module

**12_advanced_eda** (recommended before the capstone), then **11_final_pipeline**.
