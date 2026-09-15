// 03_ml_course/tests/numerical/core.cpp
//
// Purpose : the course-wide correctness suite. One process walks through the
//           shared helpers (splits, preprocessing, metrics, archive) and the
//           dense supervised models with hand-computable fixtures, checking
//           numerical results, input guards and leakage-safety invariants.
// Inputs  : none (small in-memory matrices; no data files, no defines used).
// Outputs : prints only ("<n> numerical and leakage checks passed").
//           Exit 0 on success, 1 after the first failed check.
// Target  : ml_correctness_tests (CTest: ml_correctness)
#include "01_supervised/01_regression/01_linear_regression/LinearRegression.hpp"
#include "01_supervised/01_regression/02_knn_regression/KNN.hpp"
#include "01_supervised/01_regression/04_random_forest_regressor/RandomForest.hpp"
#include "01_supervised/02_classification/01_logistic_regression/LogisticRegression.hpp"
#include "01_supervised/02_classification/09_mlp_classifier/MLP.hpp"
#include "helper/pipeline/supervised.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
int main() {
  size_t n = 0;
  // check: count the assertion and throw on failure (non-zero exit for CTest).
  auto check = [&](bool b, const char *what) {
    ++n;
    if (!b)
      throw std::runtime_error(what);
  };
  // fails: true when the callable throws any std::exception.
  auto fails = [&](const std::function<void()> &f) {
    try {
      f();
    } catch (const std::exception &) {
      return true;
    }
    return false;
  };
  try {
    // --- Logistic regression: regularisation strength and persistence ---
    // Model a uses C = 0.01 (strong L2 penalty), b uses C = 100 (weak). The
    // strongly penalised weight must be far smaller in magnitude.
    ml::Mat X{{-2}, {-1}, {.2}, {.6}, {1.4}, {2}};
    ml::Vec y{0, 0, 1, 0, 1, 1};
    ml::LogisticRegression a(2, .01, .01, 400), b(2, 100, .01, 400);
    a.fit(X, y);
    b.fit(X, y);
    check(std::fabs(a.feature_weights(0)[0]) <
              std::fabs(b.feature_weights(0)[0]) * .2,
          "regularization must shrink coefficients");
    // C = 0 would mean infinite regularisation; the constructor rejects it.
    check(fails([] { ml::LogisticRegression bad(2, 0); }), "C zero must fail");
    // save -> load into a fresh object must reproduce probabilities exactly;
    // a truncated archive must be rejected.
    std::stringstream ar;
    b.save(ar);
    ml::LogisticRegression reload;
    reload.load(ar);
    check(b.predict_proba(X) == reload.predict_proba(X),
          "model reload probabilities");
    check(fails([] {
            std::stringstream broken("LogisticRegression 2");
            ml::LogisticRegression m;
            m.load(broken);
          }),
          "truncated archive must fail");
    // --- Silhouette: hand fixture on three 1-D points ---
    // Point 10 is alone in its cluster -> silhouette defined as 0. Point 0:
    // a = 1 (to point 1), b = 10 (to point 10) -> (10 - 1) / 10 = 0.9.
    auto silhouettes = ml::silhouette_samples({{0}, {1}, {10}}, {0, 0, 1});
    check(silhouettes[2] == 0, "singleton silhouette zero");
    check(std::fabs(silhouettes[0] - .9) < 1e-12, "hand silhouette");
    check(fails([] { ml::silhouette({{0}, {1}, {10}}, {0, 0, 0}); }),
          "one cluster silhouette invalid");
    check(fails([] { ml::silhouette({{0}, {1}, {10}}, {0, 1, 2}); }),
          "all singleton clusters invalid");
    // --- Split helpers reject degenerate requests ---
    check(fails([] { ml::kfold(3, 1, 42); }), "kfold needs two folds");
    check(fails([] { ml::train_test_indices(0, .8, 42); }),
          "empty split invalid");
    check(fails([] { ml::stratified_kfold({0, 0, 1}, 2, 42); }),
          "minority support check");
    // --- Preprocessor: statistics come from training rows only ---
    // Column 0 has values {1, 3, NaN}: median 2 (imputation value) and mean 2.
    // Column 1 is constant (7): its scaled value must be 0, never NaN.
    ml::Preprocessor prep;
    prep.fit({{1, 7}, {3, 7}, {std::numeric_limits<double>::quiet_NaN(), 7}});
    auto held = prep.transform({{999, 7}});
    check(prep.median[0] == 2 && prep.mean[0] == 2 && held[0][1] == 0,
          "train-only median and constant handling");
    check(fails([] {
            ml::Preprocessor p;
            p.fit({{std::numeric_limits<double>::quiet_NaN()}});
          }),
          "all missing rejected");
    // --- Leakage guard: CV folds are drawn from the outer train split only ---
    // Every index inside any inner fold must map to an outer-train row, never
    // to a row that belongs to the held-out test set.
    auto outer =
        ml::stratified_holdout({0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}, .8, 42);
    std::set<size_t> test(outer.test.begin(), outer.test.end());
    auto folds = ml::kfold(outer.train.size(), 4, 123);
    for (const auto &f : folds) {
      for (auto i : f.train)
        check(!test.count(outer.train.at(i)), "test row in CV train");
      for (auto i : f.test)
        check(!test.count(outer.train.at(i)), "test row in CV validation");
    }
    // --- KNN: tie-break and k bounds ---
    // Query 0.1 with k = 2 sees one vote per class; the tie goes to class 0.
    ml::KNNClassifier knn(2);
    knn.fit({{0}, {2}}, {1, 0});
    check(knn.predict({{.1}})[0] == 0, "knn class tie chooses lowest class");
    check(fails([] {
            ml::KNNRegressor k(0);
            k.fit({{0}}, {1});
          }),
          "zero k invalid");
    check(fails([] {
            ml::KNNRegressor k(3);
            k.fit({{0}, {1}}, {1, 2});
          }),
          "oversized k invalid");
    // --- Gradient boosting: zero learning rate leaves only the class priors ---
    // Labels {0,1,1,2,2,2}: priors 1/6, 2/6, 3/6. With learning rate 0 the
    // single tree adds nothing, so predict_proba must return those priors.
    ml::GradientBoostedTrees gb(1, 0, 1, 1);
    gb.set_task(ml::TreeTask::Classification);
    gb.fit({{0}, {1}, {2}, {3}, {4}, {5}}, {0, 1, 1, 2, 2, 2});
    auto pr = gb.predict_proba({{1}});
    check(std::fabs(pr[0][0] - 1.0 / 6) < 1e-10 &&
              std::fabs(pr[0][2] - .5) < 1e-10,
          "class-specific GBT priors");
    // --- Random forest: integer-valued targets stay a regression task ---
    ml::RandomForest rf(10, 1, 4, 1);
    rf.set_task(ml::TreeTask::Regression);
    ml::seed_rng(42);
    rf.fit({{0}, {1}, {2}, {3}}, {1, 2, 3, 4});
    auto pred = rf.predict({{1.5}});
    check(std::isfinite(pred[0]), "integer-valued regression explicit task");
    // --- OLS on an exact line y = 2x + 1: prediction at x = 3 is 7 ---
    ml::LinearRegression lr;
    lr.fit_ols({{0}, {1}, {2}}, {1, 3, 5});
    check(std::fabs(lr.predict({{3}})[0] - 7) < 1e-12, "OLS hand fixture");
    // Metrics reject length mismatches and impossible degrees of freedom.
    check(fails([] { ml::mse({1, 2}, {1}); }), "metric mismatch rejected");
    check(fails([] { ml::r2_adjusted({1, 2}, {1, 2}, 2); }),
          "adjusted R2 invalid df");
    // --- MLP: fits an affine target (y = 2x + 1) and survives a round trip ---
    ml::MLP neural({3}, 1, .03, 300);
    ml::seed_rng(42);
    neural.fit({{-2}, {-1}, {0}, {1}, {2}}, {-3, -1, 1, 3, 5});
    auto np = neural.predict({{-2}, {-1}, {0}, {1}, {2}});
    check(ml::mse({-3, -1, 1, 3, 5}, np) < .2, "MLP can fit offset regression");
    std::stringstream nar;
    neural.save(nar);
    ml::MLP nr({1});
    nr.load(nar);
    check(nr.predict({{.5}}) == neural.predict({{.5}}), "MLP reload");
    // --- Decision tree: negative continuous targets mean regression ---
    // The task must be inferred as regression (n_classes == 1), and a leaf
    // must return the raw target -3.5 rather than a class code.
    ml::DecisionTree negative_tree;
    negative_tree.fit({{0}, {1}, {2}}, {-3.5, -1.25, 2});
    check(negative_tree.n_classes() == 1,
          "regression tree must not cast negative targets into class codes");
    check(negative_tree.predict({{0}})[0] == -3.5,
          "negative continuous tree target retained");
    // --- Input guards across tree, forest, boosting, logistic and MLP ---
    // Ragged rows, non-finite values and out-of-range class codes must all
    // be rejected at fit time; wrong query width at predict time.
    check(fails([] {
            ml::DecisionTree t;
            t.fit({{0}, {1, 2}}, {0, 1});
          }),
          "ragged tree training rejected");
    check(fails([] {
            ml::DecisionTree t;
            t.fit({{0}, {1}}, {0, std::numeric_limits<double>::infinity()});
          }),
          "nonfinite tree target rejected");
    check(fails([] {
            ml::DecisionTree t(ml::TreeTask::Classification);
            t.fit({{0}, {1}}, {-1, 1});
          }),
          "negative class code rejected before conversion");
    check(fails([] {
            ml::RandomForest t;
            t.fit({{0}, {std::numeric_limits<double>::quiet_NaN()}}, {0, 1});
          }),
          "forest missing feature rejected");
    check(fails([] {
            ml::GradientBoostedTrees t;
            t.fit({{0}, {1, 2}}, {0, 1});
          }),
          "boosted tree ragged input rejected");
    check(fails([] {
            ml::LogisticRegression m(3);
            m.fit({{0}, {1}}, {0, 3});
          }),
          "multiclass logistic out-of-range label rejected");
    check(fails([] {
            ml::LogisticRegression m;
            m.fit({{0}, {1}}, {0, .5});
          }),
          "fractional logistic class rejected");
    check(fails([] {
            ml::LogisticRegression m;
            m.fit({{0}, {1, 2}}, {0, 1});
          }),
          "ragged logistic input rejected");
    check(fails([] {
            ml::LogisticRegression m;
            m.fit({{0}, {std::numeric_limits<double>::infinity()}}, {0, 1});
          }),
          "nonfinite logistic input rejected");
    check(b.predict({}).empty() && b.predict_proba({}).empty(),
          "fitted logistic accepts empty prediction batch safely");
    check(fails([&] { b.predict({{1, 2}}); }),
          "logistic prediction shape rejected");
    check(fails([] { ml::MLP m({0}); }), "zero MLP hidden width rejected");
    check(fails([] {
            ml::MLP m({2}, 2);
            m.fit({{0}, {1}}, {0, 2});
          }),
          "MLP class label bounds");
    check(fails([] {
            ml::MLP m({2});
            m.fit({{0}, {1, 2}}, {0, 1});
          }),
          "ragged MLP input rejected");
    check(fails([] {
            ml::MLP m({2});
            m.fit({{0}, {1}}, {0, std::numeric_limits<double>::quiet_NaN()});
          }),
          "nonfinite MLP target rejected");
    check(fails([&] { neural.predict({{1, 2}}); }),
          "MLP prediction shape rejected");
    // --- AdaBoost: degenerate stumps are rejected, perfect stumps converge ---
    // Identical features give a chance-level stump (error 0.5) -> reject.
    check(fails([] {
            ml::AdaBoost m;
            ml::seed_rng(42);
            m.fit({{1}, {1}, {1}, {1}}, {0, 1, 0, 1});
          }),
          "chance AdaBoost stump rejected");
    check(fails([] {
            ml::AdaBoost m;
            m.fit({{0}, {1}}, {0, 0});
          }),
          "one-class AdaBoost rejected");
    // A perfectly separable problem yields a zero-error stump; boosting must
    // stop cleanly (no division by zero) and keep probabilities finite.
    ml::Mat separable;
    ml::Vec labels;
    for (size_t i = 0; i < 40; ++i) {
      separable.push_back({i < 20 ? 0.0 : 1.0});
      labels.push_back(i < 20 ? 0.0 : 1.0);
    }
    ml::AdaBoost perfect(100);
    ml::seed_rng(42);
    perfect.fit(separable, labels);
    check(perfect.predict(separable) == labels,
          "perfect AdaBoost stump terminates with exact classification");
    for (const auto &row : perfect.predict_proba(separable))
      for (double probability : row)
        check(std::isfinite(probability),
              "perfect AdaBoost finite probabilities");
    // --- Linear regression guards: shapes, penalties and iteration counts ---
    check(fails([] {
            ml::LinearRegression m;
            m.fit_ols({{0}, {1, 2}}, {1, 3});
          }),
          "ragged OLS rejected");
    check(fails([] {
            ml::LinearRegression m;
            m.fit_ridge({{0}, {1}}, {1, 3}, -1);
          }),
          "negative ridge rejected");
    check(fails([] {
            ml::LinearRegression m;
            m.fit_lasso({{0}, {1}}, {1, 3}, 1, 0);
          }),
          "zero lasso iterations rejected");
    check(fails([] {
            ml::LinearRegression m;
            m.set_lambda(std::numeric_limits<double>::quiet_NaN());
          }),
          "nonfinite ridge rejected");
    check(fails([&] { lr.predict({{1, 2}}); }),
          "linear prediction shape rejected");
    check(
        fails([&] { lr.predict({{std::numeric_limits<double>::infinity()}}); }),
        "nonfinite linear prediction rejected");
    // --- Preprocessor archive: a non-positive saved scale is rejected on load ---
    check(fails([] {
            std::stringstream s;
            ml::archive::write(s, ml::Vec{0}, ml::Vec{0}, ml::Vec{0});
            ml::Preprocessor p;
            p.load(s);
          }),
          "invalid saved scale rejected");
    // --- scale_fit / scale_apply: population std, constant columns -> 0 ---
    // Column 0 {1,3}: mean 2, std 1 -> {-1, 1}. Column 1 is constant -> {0, 0}.
    ml::Vec sm, ss;
    auto scaled = ml::scale_fit({{1, 7}, {3, 7}}, &sm, &ss);
    check(scaled == ml::Mat{{-1, 0}, {1, 0}},
          "population scaling centers constant features");
    check(fails([] { ml::scale_fit({}, nullptr, nullptr); }),
          "empty scaling training rejected");
    check(ml::scale_apply(ml::Mat{}, sm, ss).empty(),
          "empty scaling query batch safe");
    std::cout << n << " numerical and leakage checks passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Failure after " << n << " checks: " << e.what() << '\n';
    return 1;
  }
}
