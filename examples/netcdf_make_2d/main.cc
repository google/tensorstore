#include <netcdf.h>
#include <iostream>
#include <vector>
#include <cstring>
static void fail(int rc,const char* where){ std::cerr<<where<<": "<<nc_strerror(rc)<<"\n"; std::exit(1); }
int main(){
  const char* path="/tmp/demo2d.nc";
  int ncid; int rc=nc_create(path,NC_NETCDF4|NC_CLOBBER,&ncid); if(rc) fail(rc,"nc_create");
  int dimY,dimX; rc=nc_def_dim(ncid,"y",3,&dimY); if(rc) fail(rc,"nc_def_dim(y)");
  rc=nc_def_dim(ncid,"x",4,&dimX); if(rc) fail(rc,"nc_def_dim(x)");
  int dims[2]={dimY,dimX}, varid;
  rc=nc_def_var(ncid,"z",NC_DOUBLE,2,dims,&varid); if(rc) fail(rc,"nc_def_var(z)");
  rc=nc_enddef(ncid); if(rc) fail(rc,"nc_enddef");
  double vals[3*4]={ 1,2,3,4,  5,6,7,8,  9,10,11,12 };
  rc=nc_put_var_double(ncid,varid,vals); if(rc) fail(rc,"nc_put_var_double");
  rc=nc_close(ncid); if(rc) fail(rc,"nc_close");
  std::cout<<"Created /tmp/demo2d.nc (z: 3x4)\n";
  return 0;
}
