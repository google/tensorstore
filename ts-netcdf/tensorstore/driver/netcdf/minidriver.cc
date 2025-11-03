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

// Write operations implementation
template <class T, class NCPutter>
static int WriteTyped(const std::string& path, const std::string& var,
                      const Slice& s, const T* data, std::string* err,
                      NCPutter putter)
{
  int ncid=-1, varid=-1, rc = nc_open(path.c_str(), NC_WRITE, &ncid);
  if(rc){ if(err) *err=nerr(rc); return rc; }
  rc = nc_inq_varid(ncid, var.c_str(), &varid);
  if(rc){ if(err) *err=nerr(rc); nc_close(ncid); return rc; }

  if(!s.stride.empty()){
    rc = putter(true, ncid, varid, s.start.data(), s.count.data(), s.stride.data(), data);
  }else{
    rc = putter(false, ncid, varid, s.start.data(), s.count.data(), nullptr, data);
  }
  if(rc && err) *err=nerr(rc);

  // Sync to ensure data is written
  int sync_rc = nc_sync(ncid);
  if(sync_rc && !rc) {
    rc = sync_rc;
    if(err) *err=nerr(rc);
  }

  nc_close(ncid);
  return rc;
}

int WriteDoubles(const std::string& path, const std::string& var,
                 const Slice& s, const double* data, std::string* err){
  auto p = [](bool str, int ncid, int varid, const size_t* st, const size_t* ct,
              const ptrdiff_t* sd, const double* buf){
    return str ? nc_put_vars_double(ncid,varid,st,ct,sd,buf)
               : nc_put_vara_double(ncid,varid,st,ct,buf);
  };
  return WriteTyped<double>(path,var,s,data,err,p);
}

int WriteFloats(const std::string& path, const std::string& var,
                const Slice& s, const float* data, std::string* err){
  auto p = [](bool str, int ncid, int varid, const size_t* st, const size_t* ct,
              const ptrdiff_t* sd, const float* buf){
    return str ? nc_put_vars_float(ncid,varid,st,ct,sd,buf)
               : nc_put_vara_float(ncid,varid,st,ct,buf);
  };
  return WriteTyped<float>(path,var,s,data,err,p);
}

int WriteInts(const std::string& path, const std::string& var,
              const Slice& s, const int* data, std::string* err){
  auto p = [](bool str, int ncid, int varid, const size_t* st, const size_t* ct,
              const ptrdiff_t* sd, const int* buf){
    return str ? nc_put_vars_int(ncid,varid,st,ct,sd,buf)
               : nc_put_vara_int(ncid,varid,st,ct,buf);
  };
  return WriteTyped<int>(path,var,s,data,err,p);
}

// File and variable creation operations
int CreateFile(const std::string& path, bool clobber, std::string* err){
  int ncid=-1;
  int mode = clobber ? NC_CLOBBER : NC_NOCLOBBER;
  int rc = nc_create(path.c_str(), mode, &ncid);
  if(rc){ if(err) *err=nerr(rc); return rc; }
  rc = nc_close(ncid);
  if(rc && err) *err=nerr(rc);
  return rc;
}

int CreateDimension(const std::string& path, const std::string& dim_name, size_t size, std::string* err){
  int ncid=-1, dimid=-1;
  int rc = nc_open(path.c_str(), NC_WRITE, &ncid);
  if(rc){ if(err) *err=nerr(rc); return rc; }

  // First check if dimension already exists
  int check_rc = nc_inq_dimid(ncid, dim_name.c_str(), &dimid);
  if(check_rc == NC_NOERR){
    // Dimension already exists, verify size matches
    size_t existing_size = 0;
    rc = nc_inq_dimlen(ncid, dimid, &existing_size);
    if(rc == NC_NOERR && existing_size != size){
      if(err) *err = "Dimension '" + dim_name + "' exists with different size";
      nc_close(ncid);
      return NC_EBADDIM;
    }
    nc_close(ncid);
    return NC_NOERR;
  }

  // Enter define mode
  rc = nc_redef(ncid);
  if(rc != NC_NOERR && rc != NC_EINDEFINE){
    if(err) *err=nerr(rc);
    nc_close(ncid);
    return rc;
  }

  // Create dimension
  rc = nc_def_dim(ncid, dim_name.c_str(), size, &dimid);
  if(rc){ if(err) *err=nerr(rc); nc_close(ncid); return rc; }

  // Exit define mode
  rc = nc_enddef(ncid);
  if(rc){ if(err) *err=nerr(rc); nc_close(ncid); return rc; }

  nc_close(ncid);
  return NC_NOERR;
}

int CreateVariable(const std::string& path, const std::string& var_name,
                   DType dtype, const std::vector<std::string>& dim_names, std::string* err){
  int ncid=-1, varid=-1;
  int rc = nc_open(path.c_str(), NC_WRITE, &ncid);
  if(rc){ if(err) *err=nerr(rc); return rc; }

  // First check if variable already exists
  int check_rc = nc_inq_varid(ncid, var_name.c_str(), &varid);
  if(check_rc == NC_NOERR){
    // Variable already exists
    nc_close(ncid);
    return NC_NOERR;
  }

  // Get dimension IDs
  std::vector<int> dimids;
  for(const auto& dname : dim_names){
    int dimid=-1;
    rc = nc_inq_dimid(ncid, dname.c_str(), &dimid);
    if(rc){
      if(err) *err = "Dimension '" + dname + "' not found: " + nerr(rc);
      nc_close(ncid);
      return rc;
    }
    dimids.push_back(dimid);
  }

  // Convert DType to nc_type
  nc_type nctype;
  switch(dtype){
    case DType::kDouble: nctype = NC_DOUBLE; break;
    case DType::kFloat:  nctype = NC_FLOAT;  break;
    case DType::kInt32:  nctype = NC_INT;    break;
    default:
      if(err) *err = "Unknown data type";
      nc_close(ncid);
      return NC_EBADTYPE;
  }

  // Enter define mode
  rc = nc_redef(ncid);
  if(rc != NC_NOERR && rc != NC_EINDEFINE){
    if(err) *err=nerr(rc);
    nc_close(ncid);
    return rc;
  }

  // Create variable
  rc = nc_def_var(ncid, var_name.c_str(), nctype, (int)dimids.size(),
                  dimids.empty() ? nullptr : dimids.data(), &varid);
  if(rc){ if(err) *err=nerr(rc); nc_close(ncid); return rc; }

  // Exit define mode
  rc = nc_enddef(ncid);
  if(rc){ if(err) *err=nerr(rc); nc_close(ncid); return rc; }

  nc_close(ncid);
  return NC_NOERR;
}

} // namespace ts_netcdf
