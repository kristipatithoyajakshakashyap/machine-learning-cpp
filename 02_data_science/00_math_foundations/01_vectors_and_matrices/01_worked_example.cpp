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
std::vector<double> x{1,2,3},w{2,-1,4}; double dot=std::inner_product(x.begin(),x.end(),w.begin(),0.0); out<<"dot,"<<dot<<"\n"; if(std::abs(dot-12)>1e-12) return 1;
return 0;
}
