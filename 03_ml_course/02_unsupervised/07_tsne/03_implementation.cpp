// 03_implementation.cpp
// Lesson: t-SNE.  Iris (4 features, 150 rows) is embedded into two
// coordinates.  We write the embedding as CSV and SVG (coloured by species,
// which is used only for colouring) and the KL divergence per iteration as
// CSV and SVG so the effect of early exaggeration is visible.
// sklearn equivalent: sklearn.manifold.TSNE(n_components=2, perplexity=30).

#include <iomanip>
#include <iostream>
#include <sstream>

#include "TSNE.hpp"
#include "helper/data/datasets.hpp"
#include "helper/preprocessing/preprocessor.hpp"
#include "helper/reporting/artifacts.hpp"

int main() {
  using namespace ml;
  constexpr double kPerplexity = 30;
  constexpr size_t kIterations = 500;
  Artifacts a(RUN_OUTPUT_DIR, ".");
  auto ds = load_iris(std::string(DATA_DIR));
  Preprocessor prep;
  prep.fit(ds.X);
  const Mat X = prep.transform(ds.X);
  TSNE model(2, kPerplexity, kIterations);
  model.fit(X);
  const Mat Y = model.embedding();
  std::cout << std::setprecision(4) << "iris t-SNE: final KL "
            << model.kl_divergence() << ", trustworthiness(k=10) "
            << trustworthiness(X, Y, 10) << '\n';

  std::ostringstream emb;
  emb << std::setprecision(17) << "row,dim1,dim2,species\n";
  for (size_t i = 0; i < Y.size(); ++i)
    emb << i << ',' << Y[i][0] << ',' << Y[i][1] << ',' << ds.y[i] << '\n';
  a.write("embedding.csv", emb.str());

  Plot scatter;
  scatter.title("iris t-SNE embedding (perplexity 30)");
  scatter.xlabel("dim1");
  scatter.ylabel("dim2");
  scatter.class_labels({"setosa", "versicolor", "virginica"});
  scatter.scatter(Y, ds.y, "species");
  a.figure("figures/embedding.svg", scatter);

  const Vec& history = model.kl_history();
  std::ostringstream kl;
  kl << std::setprecision(17) << "iteration,kl_divergence\n";
  Vec iteration(history.size());
  for (size_t t = 0; t < history.size(); ++t) {
    iteration[t] = double(t + 1);
    kl << t + 1 << ',' << history[t] << '\n';
  }
  a.write("kl_per_iteration.csv", kl.str());
  Plot curve;
  curve.title("KL(P||Q) per iteration (exaggeration ends at 100)");
  curve.xlabel("iteration");
  curve.ylabel("KL divergence");
  curve.line(iteration, history, "kl");
  a.figure("figures/kl_per_iteration.svg", curve);
  std::cout << "Saved embedding.csv, kl_per_iteration.csv and figures under "
            << RUN_OUTPUT_DIR << '\n';
}
