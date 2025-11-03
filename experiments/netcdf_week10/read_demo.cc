#include "ts_netcdf_io.h"
#include <vector>
#include <iostream>
#include <iomanip>
using namespace ncutil;
int main(int argc,char**argv){try{std::string path=argc>1?argv[1]:"week10_out.nc";auto f=File::Open(path,false);int varid=-1;NC_CHECK(nc_inq_varid(f.id(),"var2d",&varid));nc_type nct;int ndims=0;NC_CHECK(nc_inq_var(f.id(),varid,nullptr,&nct,&ndims,nullptr,nullptr));std::vector<int>dimids(ndims);NC_CHECK(nc_inq_var(f.id(),varid,nullptr,&nct,&ndims,dimids.data(),nullptr));std::vector<size_t>ds(ndims);for(int i=0;i<ndims;++i){char n[NC_MAX_NAME+1];size_t l=0;NC_CHECK(nc_inq_dim(f.id(),dimids[i],n,&l));ds[i]=l;}Var v{f.id(),varid,DType::FLOAT32,dimids};std::vector<size_t>start(ndims,0),count=ds;std::vector<float>b(product(count));v.read<float>(start,count,b.data(),b.size());size_t R=count[0],C=count[1];for(size_t r=0;r<R;++r){for(size_t c=0;c<C;++c)std::cout<<std::setw(6)<<b[r*C+c]<<" ";std::cout<<"\n";}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
