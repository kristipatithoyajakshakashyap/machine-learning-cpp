#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <vector>
#include "dsts/pca.hpp"
#include "dsts/stats.hpp"
int main(){
std::filesystem::create_directories(RUN_OUTPUT_DIR);
std::ofstream out;out.exceptions(std::ios::failbit|std::ios::badbit);out.open(RUN_OUTPUT_DIR "/worked_example.csv");out<<std::setprecision(17);
std::vector<std::vector<double>> x{{1,2},{2,4},{3,6}}; auto pc=dsts::pca(x); auto z=dsts::pca_transform(pc,x); out<<"explained_PC1,"<<pc.explained_ratio[0]<<"\n"; for(size_t i=0;i<x.size();++i)for(size_t j=0;j<2;++j){double r=pc.mean[j];for(size_t k=0;k<2;++k)r+=z[i][k]*pc.loadings[k][j];if(std::abs(r-x[i][j])>1e-9)return 1;}
return 0;
}
