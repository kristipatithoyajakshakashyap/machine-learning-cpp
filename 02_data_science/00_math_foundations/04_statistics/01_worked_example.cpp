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
dsts::Series s("sample",{2,4,6});out<<"mean,"<<s.mean()<<"\nvariance,"<<s.variance()<<"\n";auto boot=dsts::bootstrap_mean(s,1000,42);out<<"bootstrap_draws,"<<boot.size()<<"\n";if(std::abs(s.variance()-4)>1e-12)return 1;
return 0;
}
