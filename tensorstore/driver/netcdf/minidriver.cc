#include "tensorstore/driver/netcdf/minidriver.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>

extern "C" {
#include <netcdf.h>
}

namespace ts_netcdf {

static std::string nerr(int rc){ return rc==NC_NOERR?"" : std::string(nc_strerror(rc)); }

int Inspect(const std::string& path, const std::string& var, Info* out, std::string* err){
  int ncid=-1, varid=-1, rc = nc_open(path.c_str(), NC_NOWRITE, &ncid);
  if(rc){ if(err) *err=nerr(rc); return rc; }
  rc = nc_inq_varid(ncid, var.c_str(), &varid);
  if(rc){ if(err) *err=nerr(rc); nc_close(ncid); return rc; }
  char name[NC_MAX_NAME+1]={0};
  nc_type vtype=NC_NAT; int vndims=0, vnatts=0;
  int dimids[NC_MAX_VAR_DIMS]={0};
  rc = nc_inq_var(ncid, varid, name, &vtype, &vndims, dimids, &vnatts);
  if(rc){ if(err) *err=nerr(rc); nc_close(ncid); return rc; }
  out->dtype = (vtype==NC_DOUBLE)?DType::kDouble : (vtype==NC_FLOAT)?DType::kFloat :
               (vtype==NC_INT)?DType::kInt32 : DType::kUnknown;
  out->shape.assign(vndims, 0);
  for(int i=0;i<vndims;i++){
    char dname[NC_MAX_NAME+1]={0}; size_t dlen=0;
    rc = nc_inq_dim(ncid, dimids[i], dname, &dlen);
    if(rc){ if(err) *err=nerr(rc); nc_close(ncid); return rc; }
    out->shape[i] = dlen;
  }
  nc_close(ncid);
  return 0;
}

template <class T, class NCGetter>
static int ReadTyped(const std::string& path, const std::string& var,
                     const Slice& s, std::vector<T>* out, std::string* err,
                     NCGetter getter)
{
  int ncid=-1, varid=-1, rc = nc_open(path.c_str(), NC_NOWRITE, &ncid);
  if(rc){ if(err) *err=nerr(rc); return rc; }
  rc = nc_inq_varid(ncid, var.c_str(), &varid);
  if(rc){ if(err) *err=nerr(rc); nc_close(ncid); return rc; }

  size_t n = 1; for(size_t c: s.count) n *= c;
  out->assign(n, T{});

  if(!s.stride.empty()){
    rc = getter(true, ncid, varid, s.start.data(), s.count.data(), s.stride.data(), out->data());
  }else{
    rc = getter(false, ncid, varid, s.start.data(), s.count.data(), nullptr, out->data());
  }
  if(rc && err) *err=nerr(rc);
  nc_close(ncid);
  return rc;
}

int ReadDoubles(const std::string& path, const std::string& var,
                const Slice& s, std::vector<double>* out, std::string* err){
  auto g = [](bool str, int ncid, int varid, const size_t* st, const size_t* ct,
              const ptrdiff_t* sd, double* buf){
    return str ? nc_get_vars_double(ncid,varid,st,ct,sd,buf)
               : nc_get_vara_double(ncid,varid,st,ct,buf);
  };
  return ReadTyped<double>(path,var,s,out,err,g);
}

int ReadFloats(const std::string& path, const std::string& var,
               const Slice& s, std::vector<float>* out, std::string* err){
  auto g = [](bool str, int ncid, int varid, const size_t* st, const size_t* ct,
              const ptrdiff_t* sd, float* buf){
    return str ? nc_get_vars_float(ncid,varid,st,ct,sd,buf)
               : nc_get_vara_float(ncid,varid,st,ct,buf);
  };
  return ReadTyped<float>(path,var,s,out,err,g);
}

int ReadInts(const std::string& path, const std::string& var,
             const Slice& s, std::vector<int>* out, std::string* err){
  auto g = [](bool str, int ncid, int varid, const size_t* st, const size_t* ct,
              const ptrdiff_t* sd, int* buf){
    return str ? nc_get_vars_int(ncid,varid,st,ct,sd,buf)
               : nc_get_vara_int(ncid,varid,st,ct,buf);
  };
  return ReadTyped<int>(path,var,s,out,err,g);
}

} // namespace ts_netcdf
