#include "ts_netcdf_io.h"
#include <vector>
#include <iostream>
using namespace ncutil;
int main(int argc,char**argv){try{std::string path=argc>1?argv[1]:"week10_out.nc";auto f=File::Create(path,true);Dim r{"rows",4};Dim c{"cols",5};auto v=define_2d(f,"var2d",DType::FLOAT32,r,c);std::vector<size_t> start={1,1},count={2,3};std::vector<float>d={10,11,12,20,21,22};v.write<float>(start,count,d.data(),d.size());f.Sync();f.Close();return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
