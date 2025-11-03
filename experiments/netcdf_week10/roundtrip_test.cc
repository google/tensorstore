#include "ts_netcdf_io.h"
#include <vector>
#include <cmath>
#include <iostream>
using namespace ncutil;
int main(){try{std::string path="week10_out.nc";{auto f=File::Create(path,true);auto v=define_2d(f,"var2d",DType::FLOAT32,Dim{"rows",4},Dim{"cols",5});std::vector<size_t>start={1,1},count={2,3};std::vector<float>d={10,11,12,20,21,22};v.write<float>(start,count,d.data(),d.size());f.Sync();f.Close();}{auto f=File::Open(path,false);int vid=-1;NC_CHECK(nc_inq_varid(f.id(),"var2d",&vid));Var v{f.id(),vid,DType::FLOAT32,{}};std::vector<size_t>start={1,1},count={2,3};std::vector<float>b(count[0]*count[1]);v.read<float>(start,count,b.data(),b.size());std::vector<float>e={10,11,12,20,21,22};for(size_t i=0;i<e.size();++i)if(std::fabs(e[i]-b[i])>1e-6)return 1;}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
