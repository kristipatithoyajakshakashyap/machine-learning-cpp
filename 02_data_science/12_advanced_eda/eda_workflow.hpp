// 12_advanced_eda/eda_workflow.hpp
// Shared advanced-EDA workflow: one function per analysis stage plus a driver.
// Each sub-lesson (01_data_quality .. 08_eda_report) is a one-line main() that
// calls advanced_eda::run(stage); stage 8 runs every analysis and adds report.md.
//
// Reads:   DATA_DIR "/titanic.csv" (all Titanic stages use only the seeded
//          80% training partition) and DATA_DIR "/air_passengers.csv" (stage 7).
// Writes:  under RUN_OUTPUT_DIR (results/<lesson stem>_results/), per stage:
//            1 quality.csv, quality.md
//            2 age_missing_by_sex.csv, missingness.md
//            3 quantiles_outliers.csv, Age.svg, Fare.svg, outliers.md
//            4 sex_by_class.csv, categorical.md
//            5 correlation.svg, pairwise_association.csv
//            6 survival_by_sex.csv, target.md
//            7 temporal_diagnostics.csv, passengers.svg, temporal.md
//            8 all of the above plus report.md
// Run:     cmake --build --preset course --target s12_eda_01 (.. s12_eda_08)
//          then build/02_data_science/12_advanced_eda/<sub-module>/s12_eda_0N
//
// dsts functions used: read_csv, stratified_split, duplicated, outlier_mask,
// chi2_independence, correlation_matrix, pearson, spearman, group_mean,
// select_columns, write_svg_histogram, write_svg_heatmap, write_svg_line,
// Series::min/max/median/quantile/missing.
#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include "dsts/cleaning.hpp"
#include "dsts/csv.hpp"
#include "dsts/eda.hpp"
#include "dsts/plots.hpp"
#include "dsts/sampling.hpp"
#include "dsts/stats.hpp"

namespace advanced_eda {
// Open RUN_OUTPUT_DIR/<name> for writing with exceptions enabled and 17-digit
// precision; creates the results directory on first use.
inline std::ofstream file(const std::string& name){
  std::filesystem::create_directories(RUN_OUTPUT_DIR);std::ofstream out;
  out.exceptions(std::ios::failbit|std::ios::badbit);out.open(std::string(RUN_OUTPUT_DIR)+"/"+name);
  out<<std::setprecision(17);return out;
}
// Stage 1: per-column type and missing counts (quality.csv), duplicate count and
// sanity notes (quality.md). Throws if Age/Fare contain negative, non-finite or
// implausible (Age > 120) values.
inline void quality(const dsts::DataFrame& t){
  auto out=file("quality.csv");out<<"feature,type,missing,missing_fraction\n";
  for(const auto& col:t.columns()){
    size_t missing=0;
    // Numeric columns track missing cells; string columns treat "" as missing.
    if(t.is_numeric(col))missing=t.series(col).missing();else for(const auto& value:t.strings(col))if(value.empty())++missing;
    out<<col<<','<<(t.is_numeric(col)?"numeric":"categorical")<<','<<missing<<','<<static_cast<double>(missing)/t.rows()<<'\n';
  }
  auto report=file("quality.md");const auto dup=dsts::duplicated(t);
  report<<"# Data quality\n\nTraining rows: "<<t.rows()<<"; full-row duplicates: "<<std::count(dup.begin(),dup.end(),true)
    <<". PassengerId is an identifier and is excluded from correlation/model features.\n\nCheck nonnegative fares, ages in [0,120], target in {0,1}, and Pclass in {1,2,3}. Missingness may be informative; do not silently remove all incomplete passengers.\n";
  for(const auto& c:std::vector<std::string>{"Age","Fare"})for(const auto& v:t.numeric(c))
    if(v&&(!std::isfinite(*v)||*v<0||(c=="Age"&&*v>120)))throw std::runtime_error("invalid "+c);
}
// Stage 2: Age missing-rate split by Sex (age_missing_by_sex.csv) to probe
// whether missingness depends on an observed group.
inline void missingness(const dsts::DataFrame& t){
  std::map<std::string,std::pair<size_t,size_t>> groups;
  for(size_t r=0;r<t.rows();++r){auto& g=groups[t.strings("Sex")[r]];++g.first;if(!t.numeric("Age")[r])++g.second;}
  auto out=file("age_missing_by_sex.csv");out<<"sex,total,missing,rate\n";
  for(const auto& [sex,g]:groups)out<<sex<<','<<g.first<<','<<g.second<<','<<static_cast<double>(g.second)/g.first<<'\n';
  auto text=file("missingness.md");text<<"# Missingness\n\nCompare rates across observed groups. Unequal rates provide evidence against a simple missing-completely-at-random story; equal rates do not prove it. Missing-not-at-random cannot generally be identified from observed values alone. Fit imputers after splitting.\n";
}
// Stage 3: five-number summary and IQR outlier count for Age and Fare
// (quantiles_outliers.csv) plus one histogram SVG per feature.
inline void distributions(const dsts::DataFrame& t){
  auto out=file("quantiles_outliers.csv");out<<"feature,min,q1,median,q3,max,iqr_outliers\n";
  for(const auto& name:std::vector<std::string>{"Age","Fare"}){
    const auto s=t.series(name);const auto mask=dsts::outlier_mask(s);
    out<<name<<','<<s.min()<<','<<s.quantile(.25)<<','<<s.median()<<','<<s.quantile(.75)<<','<<s.max()<<','<<std::count(mask.begin(),mask.end(),true)<<'\n';
    dsts::write_svg_histogram(std::string(RUN_OUTPUT_DIR)+"/"+name+".svg",name+" training distribution",name,"count",s,20);
  }
  auto text=file("outliers.md");text<<"# Outliers\n\nIQR fences Q1-1.5*IQR and Q3+1.5*IQR flag unusual values, not errors. Fare is right-skewed: compare median to mean before selecting transformations. Preserve legitimate expensive tickets; investigate provenance before deletion.\n";
}
// Stage 4: Sex x Pclass contingency table (sex_by_class.csv), chi-square test of
// independence and Cramer's V (categorical.md).
inline void categorical(const dsts::DataFrame& t){
  std::map<std::string,std::vector<double>> table;
  // Pclass is 1..3, so Pclass-1 indexes the three class columns.
  for(size_t r=0;r<t.rows();++r){auto& row=table[t.strings("Sex")[r]];row.resize(3);++row[static_cast<size_t>(*t.numeric("Pclass")[r])-1];}
  std::vector<std::vector<double>> observed;auto out=file("sex_by_class.csv");out<<"sex,class1,class2,class3\n";
  for(const auto& [group,row]:table){out<<group;for(double x:row)out<<','<<x;out<<'\n';observed.push_back(row);}
  const auto test=dsts::chi2_independence(observed);const double n=static_cast<double>(t.rows());
  auto report=file("categorical.md");report<<"# Categorical association\n\nChi-square="<<test.chi2<<", df="<<test.dof<<", p="<<test.p_value
    <<", Cramer's V="<<std::sqrt(test.chi2/n)<<" (two sex rows). Review expected counts before trusting asymptotic p-values. Association is not causation, and exploratory multiple comparisons inflate false positives.\n";
}
// Stage 5: Pearson correlation heat map over the numeric predictors and a
// pairwise table with complete-pair counts, Pearson and Spearman coefficients.
inline void multivariate(const dsts::DataFrame& t){
  const auto numeric=t.select_columns({"Pclass","Age","SibSp","Parch","Fare"});const auto corr=dsts::correlation_matrix(numeric);
  dsts::write_svg_heatmap(std::string(RUN_OUTPUT_DIR)+"/correlation.svg","Training Pearson correlation",numeric.columns(),corr);
  auto out=file("pairwise_association.csv");out<<"left,right,complete_pairs,pearson,spearman\n";
  for(const auto& a:numeric.columns())for(const auto& b:numeric.columns()){
    size_t complete=0;for(size_t r=0;r<t.rows();++r)if(t.numeric(a)[r]&&t.numeric(b)[r])++complete;
    out<<a<<','<<b<<','<<complete<<','<<dsts::pearson(t.series(a),t.series(b))<<','<<dsts::spearman(t.series(a),t.series(b))<<'\n';
  }
}
// Stage 6: survival rate by Sex with Wilson 95% score intervals
// (survival_by_sex.csv).
inline void target(const dsts::DataFrame& t){
  const auto groups=dsts::group_mean(t,"Sex","Survived");auto out=file("survival_by_sex.csv");out<<"sex,n,survival_rate,wilson_low,wilson_high\n";
  for(const auto& g:groups){const double n=static_cast<double>(g.count),z=1.96,den=1+z*z/n;
    // Wilson interval: centre and half-width for a binomial proportion g.mean of n.
    const double mid=(g.mean+z*z/(2*n))/den,half=z*std::sqrt(g.mean*(1-g.mean)/n+z*z/(4*n*n))/den;
    out<<g.group<<','<<g.count<<','<<g.mean<<','<<mid-half<<','<<mid+half<<'\n';}
  auto report=file("target.md");report<<"# Target relationships\n\nRates include Wilson 95% intervals to show sampling uncertainty. They assume independent passengers; family dependence may narrow these intervals too much. These are training associations, not fairness certification or causal effects.\n";
}
// Stage 7: AirPassengers lag-12 differences and ratios (temporal_diagnostics.csv)
// and a line plot. The first 12 rows have no lag-12 partner and get empty cells.
inline void temporal(){
  const auto t=dsts::read_csv(DATA_DIR "/air_passengers.csv");const auto y=t.numeric("Passengers");auto out=file("temporal_diagnostics.csv");
  out<<"month,passengers,lag12_difference,lag12_ratio\n";
  for(size_t r=0;r<t.rows();++r){out<<t.strings("Month")[r]<<','<<*y[r]<<',';if(r>=12)out<<*y[r]-*y[r-12]<<','<<*y[r]/ *y[r-12];else out<<',';out<<'\n';}
  dsts::write_svg_line(std::string(RUN_OUTPUT_DIR)+"/passengers.svg","Air passengers: trend and seasonality","month","passengers",t.strings("Month"),t.series("Passengers"));
  auto text=file("temporal.md");text<<"# Temporal structure\n\nThe series trends upward and its seasonal amplitude grows: inspect ratios/log differences rather than assuming stationary additive noise. Lag-12 comparisons preserve chronology. Do not randomly split forecasting data. No coordinates exist here: geographic analysis is not applicable; add coordinates and spatial block validation before interpreting maps.\n";
}
// Driver. Loads Titanic, takes the seed-42 stratified 80% training split, and
// runs the stage(s) selected by `stage` (1..7 single stage, 8 = everything).
// Returns 0; failures throw.
inline int run(int stage){
  std::filesystem::create_directories(RUN_OUTPUT_DIR);
  const auto raw=dsts::read_csv(DATA_DIR "/titanic.csv");const auto split=dsts::stratified_split(raw,"Survived",.8,42);const auto& train=split.first;
  if(stage==1||stage==8)quality(train);
  if(stage==2||stage==8)missingness(train);
  if(stage==3||stage==8)distributions(train);
  if(stage==4||stage==8)categorical(train);
  if(stage==5||stage==8)multivariate(train);
  if(stage==6||stage==8)target(train);
  if(stage==7||stage==8)temporal();
  if(stage==8){auto report=file("report.md");report<<"# Advanced EDA report\n\nTitanic goal: understand survival predictors. Data dictionary: Pclass=ticket class; Age=years; Fare=ticket fare; SibSp/Parch=family counts; Sex=recorded category; Survived=binary target. PassengerId is excluded.\n\nAll Titanic analysis uses the seeded 80% training partition. Read quality.csv, age_missing_by_sex.csv, quantiles_outliers.csv, pairwise_association.csv and survival_by_sex.csv with the companion interpretations.\n\nRecommended modeling actions: training-fitted age/fare imputation, standardized numeric predictors, explicit categorical encoding and majority baseline. Compare family-group validation before deployment. No test-set-derived preprocessing decisions. AirPassengers is a separate temporal example, not a Titanic feature.\n";}
  std::cout<<"Saved advanced EDA stage "<<stage<<" to "<<RUN_OUTPUT_DIR<<'\n';return 0;
}
}
