// 03_ml_course/02_unsupervised/06_pca/03_implementation.cpp
// Purpose: lesson showing PCA. The four-point line collapses to one component;
//          then iris (4 features) is projected to two components so we can
//          write the explained variance ratios (scree) and a 2-D score scatter
//          coloured by species.
// Inputs:  DATA_DIR/iris.csv (150 rows, 4 features, 3 species in ds.y).
// Outputs: results/03_implementation_results/{explained_variance.csv,
//          scores.csv,components.csv,figures/scree.svg,
//          figures/scores_scatter.svg}; also prints the ratios.
// Run target: upca_implementation.
// sklearn equivalent: sklearn.decomposition.PCA(n_components=2).

#include <iomanip>
#include <iostream>
#include <sstream>

#include "PCA.hpp"
#include "helper/data/datasets.hpp"
#include "helper/reporting/artifacts.hpp"

int main() {
  using namespace ml;
  // Part 1 - four collinear points: all variance lies along x, so a single
  // component reproduces the centred x coordinates (-5, -3, 3, 5 up to sign).
  Mat X{{0, 0}, {2, 0}, {8, 0}, {10, 0}};
  PCA model(1);
  model.fit(X);
  for (const auto& row : model.transform(X)) std::cout << row[0] << ' ';
  std::cout << '\n';

  // Part 2 - iris in raw units (no standardisation here on purpose, so the
  // ratios match the textbook ~0.92 / ~0.05 split for PC1 / PC2).
  Artifacts a(RUN_OUTPUT_DIR, ".");
  auto ds = load_iris(std::string(DATA_DIR));
  PCA pca(2);
  pca.fit(ds.X);
  // Explained variance covers all 4 axes even though only 2 are kept.
  const Vec ratio = pca.explained_variance_ratio();
  std::ostringstream ev;
  ev << std::setprecision(17) << "component,explained_variance_ratio,cumulative\n";
  std::vector<std::string> names;
  double cum = 0;
  std::cout << std::setprecision(4) << "iris explained variance ratio:";
  for (size_t c = 0; c < ratio.size(); ++c) {
    cum += ratio[c];
    ev << c + 1 << ',' << ratio[c] << ',' << cum << '\n';
    names.push_back("PC" + std::to_string(c + 1));
    std::cout << ' ' << ratio[c];
  }
  std::cout << '\n';
  a.write("explained_variance.csv", ev.str());

  Plot scree;
  scree.title("Scree plot: iris explained variance ratio");
  scree.xlabel("component");
  scree.ylabel("ratio");
  scree.bar(ratio, names, "explained variance");
  a.figure("figures/scree.svg", scree);

  // Scores: each row projected onto PC1/PC2; species is only used for colour.
  const Mat scores = pca.transform(ds.X);
  std::ostringstream sc;
  sc << std::setprecision(17) << "row,pc1,pc2,species\n";
  for (size_t i = 0; i < scores.size(); ++i)
    sc << i << ',' << scores[i][0] << ',' << scores[i][1] << ',' << ds.y[i]
       << '\n';
  a.write("scores.csv", sc.str());

  Plot scatter;
  scatter.title("iris PCA scores (PC1 vs PC2)");
  scatter.xlabel("PC1");
  scatter.ylabel("PC2");
  scatter.class_labels({"setosa", "versicolor", "virginica"});
  scatter.scatter(scores, ds.y, "species");
  a.figure("figures/scores_scatter.svg", scatter);

  // Loadings: how much each original feature contributes to each axis.
  std::ostringstream comp;
  comp << std::setprecision(17) << "component";
  for (const auto& f : ds.feature_names) comp << ',' << f;
  comp << '\n';
  for (size_t c = 0; c < pca.components().size(); ++c) {
    comp << names[c];
    for (double v : pca.components()[c]) comp << ',' << v;
    comp << '\n';
  }
  a.write("components.csv", comp.str());
  std::cout << "Saved explained_variance.csv, scores.csv, components.csv and "
               "figures under "
            << RUN_OUTPUT_DIR << '\n';
}
