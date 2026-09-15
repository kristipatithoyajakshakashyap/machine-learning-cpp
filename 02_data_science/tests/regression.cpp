// 02_data_science/tests/regression.cpp
// Regression test suite for the dsts library and the Titanic capstone pipeline.
// Each check() guards a behaviour that was once wrong or is easy to break:
// CSV escaping, scaler edge cases, PCA identities, hold-out isolation, model
// persistence, eigen/SVD, rank tests, calendar maths and k-fold splitting.
//
// Reads:   DATA_DIR/titanic.csv and DATA_DIR/iris.csv (DATA_DIR from CMake).
// Writes:  RUN_OUTPUT_DIR/roundtrip.csv and RUN_OUTPUT_DIR/pipeline.txt
//          (02_data_science/tests/results/regression_results/); scratch files
//          only, re-created on every run.
// Run:     cmake --build --preset course --target dsts_regression_tests
//          ctest --preset course -R dsts_regression
//          Prints "DS regression failures: N" and exits 1 when N > 0.
//
// dsts functions used: read_csv, write_csv, read_csv_string, minmax_scale, pca,
//   pca_transform, stratified_split, symmetric_eigen, svd_thin, anova_oneway,
//   mann_whitney_u, kruskal_wallis, parse_date, is_leap_year, weekday,
//   days_from_civil, civil_from_days, add_months, month_key, quarter_key,
//   to_string, kfold_indices, stratified_kfold_indices, Series::median/sum.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include "dsts/cleaning.hpp"
#include "dsts/datetime.hpp"
#include "dsts/pca.hpp"
#include "dsts/sampling.hpp"
#include "dsts/stats.hpp"
#include "../11_final_pipeline/pipeline.hpp"
// Runs every fixture in sequence; a failed check prints its name and is counted,
// but never aborts, so one run reports all failures at once.
int main(){
  int failed=0;
  // check(ok, name): record a failure with a readable label; silent on success.
  auto check=[&](bool ok,const char* name){if(!ok){std::cerr<<"FAIL: "<<name<<'\n';++failed;}};
  const std::filesystem::path dir=RUN_OUTPUT_DIR;std::filesystem::create_directories(dir);
  // --- CSV round trip: quotes, commas, embedded newlines and CR/LF inside fields must
  // survive write -> read unchanged; numeric cells keep 17-digit precision and missing.
  dsts::DataFrame csv;
  csv.add_string("quoted\"name",{" leading ","comma,quote\"", "two\nlines", "carriage\r\nreturn"});
  csv.add_numeric("value",{1.2345678901234567,std::nullopt,3,4});
  dsts::write_csv((dir/"roundtrip.csv").string(),csv);
  const auto read=dsts::read_csv((dir/"roundtrip.csv").string());
  check(read.strings("quoted\"name")==csv.strings("quoted\"name"),"CSV escaped fields preserve contents");
  check(read.numeric("value")==csv.numeric("value"),"CSV numeric precision and missing roundtrip");
  // A quoted field that never closes is a parse error, not silently truncated data.
  bool threw=false;try{dsts::read_csv_string("a\n\"unclosed");}catch(const std::invalid_argument&){threw=true;}
  check(threw,"unclosed CSV rejected");
  // Constant series: min == max would divide by zero; expect 0 and missing preserved.
  const auto scaled=dsts::minmax_scale(dsts::Series("c",{5,std::nullopt,5}));
  check(scaled.at(0)==0&& !scaled.at(1)&&scaled.at(2)==0,"constant scaling preserves missing");
  // --- PCA on collinear points (rank 1): PC1 must explain 100% of the variance,
  // transform() must reproduce the fitted scores, and mean + scores * loadings
  // must reconstruct the original rows.
  const std::vector<std::vector<double>> x{{1,2},{2,4},{3,6}};const auto pc=dsts::pca(x);const auto scores=dsts::pca_transform(pc,x);
  check(std::abs(pc.explained_ratio[0]-1)<1e-9,"rank-one PCA explains all variance");
  for(size_t r=0;r<x.size();++r)for(size_t j=0;j<2;++j){
    check(std::abs(scores[r][j]-pc.scores[r][j])<1e-10,"PCA transform matches fitted scores");
    double reconstructed=pc.mean[j];for(size_t k=0;k<2;++k)reconstructed+=scores[r][k]*pc.loadings[k][j];
    check(std::abs(reconstructed-x[r][j])<1e-9,"PCA roundtrip reconstruction");
  }
  // --- Capstone leakage guards on the real Titanic data (seed 42):
  // no PassengerId may appear in both train and test.
  const auto raw=dsts::read_csv(DATA_DIR "/titanic.csv");auto [train,test]=dsts::stratified_split(raw,"Survived",.8,42);
  std::set<double> ids;for(const auto& id:train.numeric("PassengerId"))ids.insert(*id);
  for(const auto& id:test.numeric("PassengerId"))check(!ids.count(*id),"hold-out row isolation");
  // Imputation statistics must come from the training split only.
  capstone::Model model;model.fit(train);
  check(model.prep.age_median==train.series("Age").median(),"age median uses training only");
  // Shifting hold-out ages by +100 must not change the fitted PCA axes: transform()
  // applies the learned parameters and never refits on new data.
  auto altered=test;auto age=test.numeric("Age");for(auto& v:age)if(v)v=*v+100;altered.add_numeric("Age",age);
  const auto before=model.prep.pc.loadings;model.prep.transform(altered);
  check(before==model.prep.pc.loadings,"transform never learns axes from hold-out");
  // Feature engineering is deterministic: 204 passengers have a cabin recorded.
  const auto clean=model.prep.engineer(raw);
  check(clean.series("has_cabin").sum()==204,"consistent cabin absence marker");
  // Save -> load -> predict must give exactly the same labels as the in-memory model.
  model.save(dir/"pipeline.txt");const auto loaded=capstone::Model::load(dir/"pipeline.txt");
  check(loaded.predict(test)==model.predict(test),"saved full pipeline reproduces predictions");
  // --- eigen / SVD fixtures ---
  // Diagonal matrix: eigenvalues are the diagonal entries (order not guaranteed).
  // Thin SVD of diag(3,4) stacked with a zero row: singular values 4, 3 sorted descending.
  {std::vector<double> ev;std::vector<std::vector<double>> V;dsts::symmetric_eigen({{2,0},{0,3}},ev,V);
   check(std::abs(std::min(ev[0],ev[1])-2)<1e-12&&std::abs(std::max(ev[0],ev[1])-3)<1e-12,"diagonal eigenvalues");
   std::vector<std::vector<double>> U,Vs;std::vector<double> S;dsts::svd_thin({{3,0},{0,4},{0,0}},U,S,Vs);
   check(std::abs(S[0]-4)<1e-12&&std::abs(S[1]-3)<1e-12,"thin SVD singular values sorted desc");}
  // --- ANOVA / rank tests (hand-computed) ---
  // Hand-computed expectations so the tests do not depend on another library.
  {// groups {1,2,3},{2,3,4},{5,6,7}: grand mean 11/3; means 2,3,6; SSB=3*((5/3)^2+(2/3)^2+(7/3)^2)=26; SSW=6; F=(26/2)/(6/6)=13.
   const auto a=dsts::anova_oneway({{1,2,3},{2,3,4},{5,6,7}});
   check(std::abs(a.f-13)<1e-12&&a.df_between==2&&a.df_within==6,"ANOVA F=13 on hand-computed groups");
   check(a.p_value>0.006&&a.p_value<0.007,"ANOVA p for F(2,6)=13 is ~0.0065");
   // Identical groups: no between-group variance, so F = 0 and p = 1 exactly.
   const auto e=dsts::anova_oneway({{1,2,3},{1,2,3}});check(std::abs(e.f)<1e-12&&std::abs(e.p_value-1)<1e-12,"ANOVA identical groups F=0 p=1");
   // Fully separated samples give U = 0 (smallest possible); identical samples give
   // U = n1*n2/2 = 8 and p = 1.
   const auto mw=dsts::mann_whitney_u({1,2,3},{4,5,6});check(mw.u==0,"Mann-Whitney U=0 for separated samples");
   check(mw.p_value>0&&mw.p_value<0.1,"Mann-Whitney p small for separated samples");
   const auto mw2=dsts::mann_whitney_u({1,2,3,4},{1,2,3,4});check(mw2.u==8&&std::abs(mw2.p_value-1)<1e-12,"Mann-Whitney identical samples U=n1n2/2 p=1");
   // Kruskal on {1,2,3},{4,5,6}: ranks 1..6, R1=6, R2=15,
   // H = 12/(6*7)*(36/3+225/3) - 3*7 = 12/42*87 - 21 = 24.857 - 21 = 3.857142857 (= 27/7).
   const auto kw=dsts::kruskal_wallis({{1,2,3},{4,5,6}});check(std::abs(kw.h-27.0/7.0)<1e-9&&kw.df==1,"Kruskal-Wallis H=27/7");
   check(std::abs(kw.p_value-0.04953)<5e-4,"Kruskal-Wallis p ~ 0.0495");}
  // --- datetime fixtures ---
  // Calendar rules: Feb 29 only in leap years; 1900 is not leap (century rule),
  // 2000 is (400-year rule). Known weekdays and day numbers anchor the algorithms.
  // add_months clamps the day to the target month's length and rolls across years.
  {const auto leap=dsts::parse_date("2024-02-29");check(leap&&leap->day==29,"2024-02-29 parses (leap year)");
   check(!dsts::parse_date("2023-02-29"),"2023-02-29 rejected");
   check(dsts::is_leap_year(2000)&&!dsts::is_leap_year(1900)&&dsts::is_leap_year(2024),"leap year rules");
   check(dsts::weekday({2000,1,1})==6,"2000-01-01 is Saturday");
   check(dsts::weekday({1970,1,1})==4,"1970-01-01 is Thursday");
   check(dsts::days_from_civil({1970,1,1})==0&&dsts::days_from_civil({2000,3,1})==11017,"days_from_civil epoch and 2000-03-01");
   check(dsts::civil_from_days(11017)==dsts::Date{2000,3,1},"civil_from_days roundtrip");
   check(dsts::add_months({2024,1,31},1)==dsts::Date{2024,2,29},"add_months clamps to Feb 29");
   check(dsts::add_months({2023,11,15},3)==dsts::Date{2024,2,15},"add_months rolls into next year");
   check(dsts::add_months({2024,1,15},-2)==dsts::Date{2023,11,15},"add_months negative rollover");
   const auto ym=dsts::parse_date("1949-01");check(ym&&ym->year==1949&&ym->month==1&&ym->day==1,"YYYY-MM parses with day=1");
   check(dsts::month_key({2024,7,4})=="2024-07"&&dsts::quarter_key({2024,7,4})=="2024-Q3"&&dsts::to_string({2024,7,4})=="2024-07-04","date keys");}
  // --- k-fold fixtures ---
  // k-fold on 23 rows: every index in range, each row in exactly one test fold,
  // fold sizes differ by at most 1, and the same seed reproduces the same folds.
  {const auto folds=dsts::kfold_indices(23,5,3);std::vector<int> seen(23,0);size_t mn=99,mx=0;
   for(const auto& f:folds){mn=std::min(mn,f.size());mx=std::max(mx,f.size());for(size_t i:f){check(i<23,"kfold index in range");++seen[i];}}
   bool once=true;for(int c:seen)once=once&&c==1;check(once,"kfold: each row in exactly one test fold");
   check(folds.size()==5&&mx-mn<=1,"kfold: 5 folds, sizes differ by at most 1");
   check(dsts::kfold_indices(23,5,3)==folds,"kfold: same seed same folds");
   // Stratified k-fold on iris species: each row appears once and every class is
   // spread across the 4 folds as evenly as possible (per-class counts within 1).
   const auto iris=dsts::read_csv(DATA_DIR "/iris.csv");const auto& sp=iris.strings("species");
   const auto sf=dsts::stratified_kfold_indices(sp,4,11);std::vector<int> seen2(sp.size(),0);
   std::map<std::string,std::vector<size_t>> per_class;
   for(size_t f=0;f<sf.size();++f)for(size_t i:sf[f]){++seen2[i];auto& v=per_class[sp[i]];if(v.empty())v.assign(4,0);++v[f];}
   bool once2=true;for(int c:seen2)once2=once2&&c==1;check(once2,"stratified kfold: each row in exactly one fold");
   bool balanced=true;for(const auto& kv:per_class){const auto [lo,hi]=std::minmax_element(kv.second.begin(),kv.second.end());balanced=balanced&&(*hi-*lo<=1);}
   check(balanced,"stratified kfold: per-class fold counts differ by at most 1");
   // k = 1 is not a split; the function must reject it instead of returning one fold.
   bool threw2=false;try{dsts::kfold_indices(3,1,0);}catch(const std::invalid_argument&){threw2=true;}check(threw2,"kfold rejects k<2");}
  std::cout<<"DS regression failures: "<<failed<<'\n';return failed?1:0;
}
