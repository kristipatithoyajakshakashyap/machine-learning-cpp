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
auto f=[](double w){return (w-3)*(w-3);};double w=1,h=1e-5;double numeric=(f(w+h)-f(w-h))/(2*h),analytic=2*(w-3);out<<"analytic,"<<analytic<<"\nnumerical,"<<numeric<<"\n";if(std::abs(numeric-analytic)>1e-8)return 1;
return 0;
}
