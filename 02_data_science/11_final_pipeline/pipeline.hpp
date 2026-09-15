// 11_final_pipeline/pipeline.hpp
// Shared Titanic capstone pipeline: preprocessing, a PC1 threshold classifier,
// model save/load, and the staged driver used by lessons 01..05.
//
// Every stage executable (s11_01_clean .. s11_05_eval) calls capstone::run(stage)
// with its own stage number; higher stages produce a superset of the outputs
// of lower stages, always from the same seed-42 stratified split.
//
// Reads:   DATA_DIR "/titanic.csv" (DATA_DIR is injected by CMake).
// Writes:  under RUN_OUTPUT_DIR (results/<stage stem>_results/):
//            train_features.csv, test_features.csv, row_assignments.csv, report.md
//            stage>=3: age.svg, fare_by_sex.svg
//            stage>=4: pca_scores.csv
//            stage>=5: predictions.csv, metrics.json, model_state.txt,
//                      reload_verification.json
// Run:     cmake --build --preset course --target s11_05_eval (or s11_01_clean ...)
//          then build/02_data_science/11_final_pipeline/s11_05_eval
//
// dsts functions used: read_csv, write_csv, stratified_split, pca, pca_transform,
// summary, write_svg_histogram, write_svg_box, trim, Series::quantile.

#pragma once
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include "dsts/csv.hpp"
#include "dsts/eda.hpp"
#include "dsts/pca.hpp"
#include "dsts/plots.hpp"
#include "dsts/sampling.hpp"

namespace capstone {
// Row-major dense matrix: x[row][feature].
using Matrix = std::vector<std::vector<double>>;
// Fixed feature schema. Order matters: it is the column order of every Matrix
// and is written into model_state.txt so load() can verify it.
inline const std::vector<std::string> features = {"Pclass", "Age", "Fare", "family_size", "is_child", "has_cabin", "Sex_male"};

// Open `path` for writing, creating parent directories. Stream exceptions are
// enabled so a failed write throws instead of silently producing a short file.
// Precision 17 makes doubles round-trip exactly through text.
inline std::ofstream output(const std::filesystem::path& path) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out;
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out.open(path);
  out << std::setprecision(17);
  return out;
}

// Fitted preprocessing state: imputation medians, per-feature standardisation
// and PCA. All statistics are learned from the training frame only (fit) and
// then applied unchanged to any frame (transform), which prevents leakage.
struct Preprocessor {
  double age_median = 0, fare_median = 0;   // training medians used to fill missing Age/Fare
  std::vector<double> mean, scale;          // per-feature mean and std-dev (std-dev 0 -> 1)
  dsts::PcaResult pc;                       // PCA fitted on standardised training features

  // Impute Age/Fare and add the four engineered columns. Pure function of the
  // input table plus the stored medians; it does not touch mean/scale/pc.
  //   table: raw Titanic frame (by value so the caller's copy is untouched).
  //   returns: the same frame with Age/Fare filled and family_size, is_child,
  //            has_cabin, Sex_male appended.
  //   throws: invalid_argument if Sex is neither "male" nor "female".
  dsts::DataFrame engineer(dsts::DataFrame table) const {
    // Replace missing (nullopt) Age/Fare cells with the stored medians.
    for (const auto& item : std::vector<std::pair<std::string,double>>{{"Age",age_median},{"Fare",fare_median}}) {
      auto values = table.numeric(item.first);
      for (auto& value : values) if (!value) value = item.second;
      table.add_numeric(item.first, std::move(values));
    }
    std::vector<dsts::OptD> family, child, cabin, male;
    for (size_t r = 0; r < table.rows(); ++r) {
      // family_size = self + siblings/spouses + parents/children; missing counts as 0.
      family.push_back(1 + table.numeric("SibSp")[r].value_or(0) + table.numeric("Parch")[r].value_or(0));
      child.push_back(*table.numeric("Age")[r] < 10 ? 1 : 0);   // Age is non-missing after imputation
      // Cabin absence is an empty string or the literal "NA".
      const auto c = dsts::trim(table.strings("Cabin")[r]);
      cabin.push_back(c.empty() || c == "NA" ? 0 : 1);
      const auto& sex = table.strings("Sex")[r];
      if (sex != "male" && sex != "female") throw std::invalid_argument("unknown sex category");
      male.push_back(sex == "male" ? 1 : 0);
    }
    table.add_numeric("family_size", family); table.add_numeric("is_child", child);
    table.add_numeric("has_cabin", cabin); table.add_numeric("Sex_male", male);
    return table;
  }

  // engineer() then pack the `features` columns into a dense Matrix.
  Matrix dense(const dsts::DataFrame& raw) const {
    const auto table = engineer(raw);
    Matrix x(table.rows(), std::vector<double>(features.size()));
    for (size_t r=0; r<x.size(); ++r) for (size_t j=0; j<features.size(); ++j)
      x[r][j] = *table.numeric(features[j])[r];
    return x;
  }

  // (x - mean) / scale per column. Requires fit() to have run.
  Matrix standardize(Matrix x) const {
    if (mean.size()!=features.size() || scale.size()!=features.size()) throw std::logic_error("preprocessor not fitted");
    for (auto& row:x) for (size_t j=0;j<row.size();++j) row[j]=(row[j]-mean[j])/scale[j];
    return x;
  }

  // Learn medians, mean/scale and PCA from the training frame.
  //   throws: invalid_argument if Age or Fare is entirely missing (median NaN).
  void fit(const dsts::DataFrame& train) {
    age_median=train.series("Age").quantile(0.5); fare_median=train.series("Fare").quantile(0.5);
    if (!std::isfinite(age_median)||!std::isfinite(fare_median)) throw std::invalid_argument("all-missing imputation feature");
    const auto x=dense(train); const size_t p=features.size();
    mean.assign(p,0);scale.assign(p,0);
    // Population mean and variance (divide by n, not n-1), accumulated per column.
    for (const auto& row:x) for (size_t j=0;j<p;++j) mean[j]+=row[j]/static_cast<double>(x.size());
    for (const auto& row:x) for (size_t j=0;j<p;++j) scale[j]+=(row[j]-mean[j])*(row[j]-mean[j])/static_cast<double>(x.size());
    // Constant columns get scale 1 so standardisation never divides by zero.
    for (double& s:scale) {s=std::sqrt(s);if(s==0)s=1;}
    pc=dsts::pca(standardize(x));
  }

  // Full pipeline for any frame: engineer -> standardise -> PCA scores.
  Matrix transform(const dsts::DataFrame& raw) const {return dsts::pca_transform(pc,standardize(dense(raw)));}
};

// Deliberately simple classifier: threshold on the first principal component
// halfway between the two class means of PC1. Not logistic regression.
struct Model {
  Preprocessor prep;
  double threshold=0;         // midpoint of the two class means on PC1
  bool positive_above=true;   // true if survivors have the larger PC1 mean
  int majority=0;             // majority class in training; used for baseline accuracy

  // Fit preprocessing, then compute PC1 class means and the midpoint threshold.
  //   throws: invalid_argument unless both classes appear in `train`.
  void fit(const dsts::DataFrame& train) {
    prep.fit(train);const auto x=prep.transform(train);
    double means[2]={0,0};size_t counts[2]={0,0};
    for(size_t r=0;r<train.rows();++r){const int y=static_cast<int>(*train.numeric("Survived")[r]);means[y]+=x[r][0];++counts[y];}
    if(!counts[0]||!counts[1])throw std::invalid_argument("both target classes required");
    for(int c=0;c<2;++c)means[c]/=static_cast<double>(counts[c]);
    threshold=(means[0]+means[1])/2;positive_above=means[1]>means[0];majority=counts[1]>counts[0]?1:0;
  }

  // 0/1 prediction per row: which side of the threshold PC1 falls on.
  std::vector<int> predict(const dsts::DataFrame& table) const {
    std::vector<int> out;
    for(const auto& row:prep.transform(table))out.push_back(positive_above?row[0]>threshold:row[0]<threshold);
    return out;
  }

  // Text serialisation (format version 1):
  //   line 1: DSTS_CAPSTONE 1 <p>
  //   line 2: quoted feature names (schema check on load)
  //   line 3: age_median fare_median threshold positive_above majority
  //   3 lines: prep.mean, prep.scale, pc.mean (p values each)
  //   p lines: PCA loadings rows
  void save(const std::filesystem::path& path) const {
    auto out=output(path);out<<"DSTS_CAPSTONE 1 "<<features.size()<<'\n';
    for(const auto& name:features){out<<std::quoted(name)<<' ';}out<<'\n';
    out<<prep.age_median<<' '<<prep.fare_median<<' '<<threshold<<' '<<positive_above<<' '<<majority<<'\n';
    for(const auto* values:{&prep.mean,&prep.scale,&prep.pc.mean}){for(double v:*values)out<<v<<' ';out<<'\n';}
    for(const auto& row:prep.pc.loadings){for(double v:row)out<<v<<' ';out<<'\n';}
  }

  // Inverse of save(). Validates tag, version, feature count, feature names
  // and that every saved scale is strictly positive.
  static Model load(const std::filesystem::path& path) {
    std::ifstream in(path);in.exceptions(std::ios::failbit|std::ios::badbit);
    std::string tag;int version;size_t p;in>>tag>>version>>p;
    if(tag!="DSTS_CAPSTONE"||version!=1||p!=features.size())throw std::invalid_argument("unsupported pipeline state");
    for(const auto& expected:features){std::string name;in>>std::quoted(name);if(name!=expected)throw std::invalid_argument("feature schema mismatch");}
    Model m;in>>m.prep.age_median>>m.prep.fare_median>>m.threshold>>m.positive_above>>m.majority;
    for(auto* values:{&m.prep.mean,&m.prep.scale,&m.prep.pc.mean}){values->resize(p);for(double& v:*values)in>>v;}
    for(double s:m.prep.scale)if(!(s>0))throw std::invalid_argument("invalid saved scale");
    m.prep.pc.loadings.assign(p,std::vector<double>(p));for(auto& row:m.prep.pc.loadings)for(double& v:row)in>>v;
    return m;
  }
};

// Staged driver shared by lessons 01..05. `stage` selects how far to go:
//   1-2: split, fit preprocessing, write feature tables, assignments, report
//   3:   + training histograms/box plots
//   4:   + PCA scores for train and test
//   5:   + predictions, metrics, model save + reload verification
// Returns 0 on success; any failure throws.
inline int run(int stage) {
  const auto raw=dsts::read_csv(DATA_DIR "/titanic.csv");
  // Stratified 80/20 split on Survived with a fixed seed so every stage agrees.
  auto [train,test]=dsts::stratified_split(raw,"Survived",0.8,42);
  const std::filesystem::path dir=RUN_OUTPUT_DIR;std::filesystem::create_directories(dir);
  Model model;model.fit(train);
  // Engineered (not standardised) frames are what the CSV outputs contain.
  const auto train_clean=model.prep.engineer(train),test_clean=model.prep.engineer(test);
  dsts::write_csv((dir/"train_features.csv").string(),train_clean);
  dsts::write_csv((dir/"test_features.csv").string(),test_clean);
  auto split=output(dir/"row_assignments.csv");split<<"PassengerId,split\n";
  for(const auto* table:{&train,&test})for(const auto& id:table->numeric("PassengerId"))split<<*id<<','<<(table==&train?"train":"test")<<'\n';
  auto report=output(dir/"report.md");
  report<<"# Titanic pipeline: stage "<<stage<<"\n\nPredict recorded survival using passenger attributes. PassengerId is an identifier, not a feature.\n\n"
        <<"Rows: "<<raw.rows()<<"; training: "<<train.rows()<<"; hold-out: "<<test.rows()<<". Split seed 42.\n\n"
        <<"Age median "<<model.prep.age_median<<" and Fare median "<<model.prep.fare_median<<" are learned from training only. Cabin absence is an empty string or NA.\n\n"
        <<"The model is a nearest-class-mean threshold on PC1, not logistic regression. PCA optimizes feature variance, not predictive accuracy. No hyperparameters are selected on test.\n\n"
        <<"Training numeric summary:\n```\n"<<dsts::summary(train_clean)<<"```\n\n"
        <<"Historical observational associations do not establish causality; family groups can cross this educational stratified split. A deployment study should evaluate family-group hold-outs.\n";
  if(stage>=3){
    dsts::write_svg_histogram((dir/"age.svg").string(),"Training age distribution","years","passengers",train_clean.series("Age"),16);
    dsts::write_svg_box((dir/"fare_by_sex.svg").string(),"Training fare by sex","fare",train_clean.strings("Sex"),train_clean.numeric("Fare"));
  }
  if(stage>=4){
    auto out=output(dir/"pca_scores.csv");out<<"PassengerId,split,PC1,PC2\n";
    for(const auto* table:{&train,&test}){const auto scores=model.prep.transform(*table);for(size_t r=0;r<table->rows();++r)
      out<<*table->numeric("PassengerId")[r]<<','<<(table==&train?"train":"test")<<','<<scores[r][0]<<','<<scores[r][1]<<'\n';}
  }
  if(stage>=5){
    // Confusion counts on the hold-out set; `base` counts rows the majority
    // classifier would get right (baseline accuracy).
    const auto pred=model.predict(test);size_t tp=0,tn=0,fp=0,fn=0,base=0;
    auto out=output(dir/"predictions.csv");out<<"PassengerId,actual,prediction\n";
    for(size_t r=0;r<test.rows();++r){int y=static_cast<int>(*test.numeric("Survived")[r]);
      if(y==1){if(pred[r])++tp;else ++fn;}else{if(pred[r])++fp;else ++tn;}if(y==model.majority)++base;
      out<<*test.numeric("PassengerId")[r]<<','<<y<<','<<pred[r]<<'\n';}
    const double n=static_cast<double>(test.rows()),acc=(tp+tn)/n;
    // Precision/recall guard against division by zero when a class is never predicted.
    auto metrics=output(dir/"metrics.json");metrics<<"{\"accuracy\":"<<acc<<",\"baseline_accuracy\":"<<base/n<<",\"precision\":"<<(tp+fp?static_cast<double>(tp)/(tp+fp):0)
      <<",\"recall\":"<<(tp+fn?static_cast<double>(tp)/(tp+fn):0)<<",\"tn\":"<<tn<<",\"fp\":"<<fp<<",\"fn\":"<<fn<<",\"tp\":"<<tp<<"}\n";
    // Round-trip check: a reloaded model must reproduce the same predictions.
    model.save(dir/"model_state.txt");const auto loaded=Model::load(dir/"model_state.txt");
    if(loaded.predict(test)!=pred)throw std::runtime_error("reload predictions differ");
    auto reload=output(dir/"reload_verification.json");reload<<"{\"identical_predictions\":true}\n";
    report<<"\nHold-out accuracy: "<<acc<<"; majority baseline: "<<base/n<<". Inspect the confusion counts in metrics.json before drawing conclusions.\n";
  }
  std::cout<<"Stage "<<stage<<" saved to "<<dir<<'\n';return 0;
}
} // namespace capstone
