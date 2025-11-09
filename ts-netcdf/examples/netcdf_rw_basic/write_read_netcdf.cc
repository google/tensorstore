#include <netcdf.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void chk(int rc,const char*where){ if(rc!=NC_NOERR){ std::fprintf(stderr,"ERR %s: %s\n",where,nc_strerror(rc)); std::exit(2);} }
int main(){
  const char* path="/tmp/ts_week4.nc";
  int ncid, xdim, ydim, varid;
  int rc = nc_create(path, NC_CLOBBER, &ncid); chk(rc,"nc_create");
  rc = nc_def_dim(ncid,"y",2,&ydim); chk(rc,"nc_def_dim y");
  rc = nc_def_dim(ncid,"x",3,&xdim); chk(rc,"nc_def_dim x");
  int dims[2] = {ydim, xdim};
  rc = nc_def_var(ncid,"var",NC_FLOAT,2,dims,&varid); chk(rc,"nc_def_var");
  rc = nc_enddef(ncid); chk(rc,"nc_enddef");
  float w[6] = {1,2,3,4,5,6};
  rc = nc_put_var_float(ncid,varid,w); chk(rc,"nc_put_var_float");
  rc = nc_close(ncid); chk(rc,"nc_close write");

  rc = nc_open(path, NC_NOWRITE, &ncid); chk(rc,"nc_open");
  float r[6] = {0};
  rc = nc_inq_varid(ncid,"var",&varid); chk(rc,"nc_inq_varid");
  rc = nc_get_var_float(ncid,varid,r); chk(rc,"nc_get_var_float");
  rc = nc_close(ncid); chk(rc,"nc_close read");

  double sum=0; for(float v: r) sum+=v;
  std::printf("OK %s sum=%.0f vals=", path, sum);
  for(int i=0;i<6;++i) std::printf("%s%.0f", i?",":"", r[i]);
  std::printf("\n");
  return 0;
}
