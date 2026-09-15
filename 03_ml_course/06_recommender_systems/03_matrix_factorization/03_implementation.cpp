// 03_ml_course/06_recommender_systems/03_matrix_factorization/03_implementation.cpp
//
// Purpose : implementation lesson for biased matrix factorization. Runs the
//           SGD update from Model.hpp one epoch at a time on the whole
//           MovieLens file so the training RMSE curve can be recorded, then
//           dumps the learned bias and factors of the 5 most-rated items.
// Inputs  : <COURSE_ROOT>/06_recommender_systems/data/u.data
//           (COURSE_ROOT and RUN_OUTPUT_DIR are compile definitions from
//           add_lesson_executable in the root CMakeLists.txt).
// Outputs : results/03_implementation_results/
//             train_rmse_per_epoch.csv / .svg  - RMSE after each of 20 epochs
//             item_factors_top5.csv            - item, count, bias, f0..f15
// Target  : rec_mf_implementation
#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
#include <iostream>
#include <numeric>
#include <random>

// Lesson: biased matrix factorization trained by SGD on MovieLens 100K.
// The lesson runs the same SGD update as Model.hpp but one epoch at a time so
// the training RMSE can be logged after every pass.
// Artifacts: train RMSE per epoch (CSV + line SVG) and the learned bias and
// latent factors of the 5 most-rated items (CSV).
int main() {
  std::cout
      << R"LESSON(Train only on earlier interactions, then rank unseen items from the training catalog.)LESSON"
      << "\n";
  // Toy demonstration of the recommend() API on the Popularity baseline.
  rec_course::Popularity m;
  m.fit({{1, 1, 1, 5, 1}, {2, 2, 2, 4, 2}});
  for (const auto &r : rec_course::recommend(m, 1))
    std::cout << "item " << r.first << " score " << r.second << "\n";

  try {
    auto data = rec_course::read_ratings(
        std::string(COURSE_ROOT) + "/06_recommender_systems/data/u.data");
    // epochs = 0 makes fit() only compute the baseline statistics and draw
    // the seeded N(0, 0.1) initial factors; the SGD passes run below with
    // their own shuffle seed (7) so the curve is reproducible.
    rec_course::MatrixFactorization mf(0.05);
    mf.epochs = 0; // initialise factors only; epochs run below
    mf.fit(data);
    const size_t epochs = 20;
    std::mt19937 rng(7);
    std::vector<size_t> order(data.size());
    std::iota(order.begin(), order.end(), 0);
    std::ostringstream csv;
    csv << "epoch,train_rmse\n";
    ml::Vec xs, ys;
    // One SGD epoch = one shuffled pass over all ratings with the same
    // decaying learning rate and L2-regularised updates as Model.hpp.
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
      std::shuffle(order.begin(), order.end(), rng);
      const double lr = 0.01 / (1 + 0.03 * epoch), reg = mf.regularization;
      for (size_t idx : order) {
        const auto &r = data[idx];
        double err = r.value - mf.raw(r.user, r.item);
        mf.user_bias[r.user] += lr * (err - reg * mf.user_bias[r.user]);
        mf.item_bias[r.item] += lr * (err - reg * mf.item_bias[r.item]);
        for (size_t d = 0; d < mf.dimensions; ++d) {
          double u = mf.users[r.user][d], v = mf.items[r.item][d];
          mf.users[r.user][d] += lr * (err * v - reg * u);
          mf.items[r.item][d] += lr * (err * u - reg * v);
        }
      }
      // Training RMSE after this epoch (clamped predictions, all rows).
      double sse = 0;
      for (const auto &r : data) {
        double e = r.value - mf.predict(r.user, r.item);
        sse += e * e;
      }
      double rmse = std::sqrt(sse / data.size());
      csv << epoch + 1 << ',' << rmse << '\n';
      xs.push_back(double(epoch + 1));
      ys.push_back(rmse);
    }
    // Artifacts(root, ".") writes directly into RUN_OUTPUT_DIR (no sub-run).
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("train_rmse_per_epoch.csv", csv.str());
    ml::Plot curve;
    curve.line(xs, ys, "train RMSE");
    curve.title("Matrix factorization: training RMSE per epoch");
    curve.xlabel("epoch");
    curve.ylabel("RMSE");
    a.figure("train_rmse_per_epoch.svg", curve);
    // Five most-rated items (ties by id) with their bias and 16 factors.
    std::vector<size_t> items = mf.catalog();
    const size_t top = std::min<size_t>(5, items.size());
    std::partial_sort(items.begin(), items.begin() + top, items.end(),
                      [&](size_t x, size_t y) {
                        return mf.count[x] != mf.count[y]
                                   ? mf.count[x] > mf.count[y]
                                   : x < y;
                      });
    std::ostringstream factors;
    factors << "item,rating_count,item_bias";
    for (size_t d = 0; d < mf.dimensions; ++d)
      factors << ",f" << d;
    factors << '\n';
    for (size_t r = 0; r < top; ++r) {
      size_t i = items[r];
      factors << i << ',' << mf.count[i] << ',' << mf.item_bias[i];
      for (double f : mf.items[i])
        factors << ',' << f;
      factors << '\n';
    }
    a.write("item_factors_top5.csv", factors.str());
    std::cout << "final train RMSE " << ys.back() << "\n";
    for (const auto &f : {"train_rmse_per_epoch.csv",
                          "train_rmse_per_epoch.svg", "item_factors_top5.csv"})
      std::cout << "wrote " << a.path(f) << "\n";
  } catch (const std::exception &e) {
    std::cerr << "artifact error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
