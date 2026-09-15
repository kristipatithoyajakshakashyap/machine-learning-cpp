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
double w=0,lambda=1,rate=.1;out<<"iteration,weight,loss\n";for(int i=0;i<100;++i){double grad=2*(w-3)+2*lambda*w;w-=rate*grad;out<<i<<","<<w<<","<<(w-3)*(w-3)+lambda*w*w<<"\n";}if(std::abs(w-1.5)>1e-9)return 1;
return 0;
}
