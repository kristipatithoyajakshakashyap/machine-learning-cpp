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
double prior=.01,sensitivity=.9,fpr=.05; double positive=prior*sensitivity+(1-prior)*fpr; double posterior=prior*sensitivity/positive; out<<"posterior,"<<posterior<<"\n";if(std::abs(posterior-2.0/13)>1e-12)return 1;
return 0;
}
