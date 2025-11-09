#include <iostream>
#include "tensorstore/driver/netcdf/reader.h"
int main(int argc, char** argv){
  if(argc<2){ std::cerr<<"usage: "<<argv[0]<<" path\n"; return 2; }
  ts_netcdf::Slice s{{0,0,0},{1,3,2},{1,1,2}};
  std::vector<double> out; std::string err;
  int rc = ts_netcdf::ReadDoubles(argv[1],"v",s,&out,&err);
  if(rc){ std::cerr<<"error: "<<err<<"\n"; return 1; }
  for(size_t i=0;i<out.size();++i) std::cout<<out[i]<<(i+1<out.size()?", ":"\n");
  return 0;
}
