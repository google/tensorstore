#include <netcdf.h>
#include <iostream>
#include <vector>
static void fail(int rc,const char*w){std::cerr<<w<<": "<<nc_strerror(rc)<<"\n";std::exit(1);}
int main(){
  const char* p="/tmp/demo3d.nc"; int ncid; int rc=nc_create(p,NC_NETCDF4|NC_CLOBBER,&ncid); if(rc) fail(rc,"nc_create");
  int d0,d1,d2; rc=nc_def_dim(ncid,"t",2,&d0); if(rc) fail(rc,"nc_def_dim");
  rc=nc_def_dim(ncid,"y",3,&d1); if(rc) fail(rc,"nc_def_dim");
  rc=nc_def_dim(ncid,"x",4,&d2); if(rc) fail(rc,"nc_def_dim");
  int dims[3]={d0,d1,d2}, varid; rc=nc_def_var(ncid,"v",NC_INT,3,dims,&varid); if(rc) fail(rc,"nc_def_var");
  rc=nc_enddef(ncid); if(rc) fail(rc,"nc_enddef");
  int vals[2*3*4]; for(int t=0,idx=0;t<2;t++)for(int y=0;y<3;y++)for(int x=0;x<4;x++,idx++) vals[idx]=100*t+10*y+x+1;
  rc=nc_put_var_int(ncid,varid,vals); if(rc) fail(rc,"nc_put_var_int");
  rc=nc_close(ncid); if(rc) fail(rc,"nc_close");
  std::cout<<"Created /tmp/demo3d.nc (v: 2x3x4 INT)\n"; return 0;
}
