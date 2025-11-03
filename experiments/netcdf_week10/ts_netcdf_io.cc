#include "ts_netcdf_io.h"
#include <cstring>
#include <cmath>

namespace ncutil {

size_t product(const std::vector<size_t>& v){size_t p=1;for(auto x:v)p*=x;return p;}

int ensure_dim(File& f,const std::string& name,size_t size){int ncid=f.id();int dimid=-1;int s=nc_inq_dimid(ncid,name.c_str(),&dimid);if(s==NC_NOERR)return dimid;int redef_rc=nc_redef(ncid);if(redef_rc!=NC_NOERR&&redef_rc!=NC_EINDEFINE)throw NcError(redef_rc);NC_CHECK(nc_def_dim(ncid,name.c_str(),size,&dimid));int enddef_rc=nc_enddef(ncid);if(enddef_rc!=NC_NOERR&&enddef_rc!=NC_ENOTINDEFINE)throw NcError(enddef_rc);return dimid;}

static int to_nc_type(DType t){switch(t){case DType::FLOAT32:return NC_FLOAT;case DType::FLOAT64:return NC_DOUBLE;case DType::INT32:return NC_INT;case DType::INT16:return NC_SHORT;case DType::UINT8:return NC_UBYTE;case DType::CHAR8:return NC_CHAR;}return NC_NAT;}

Var ensure_var(File& f,const std::string& name,DType dtype,const std::vector<int>& dimids){int ncid=f.id();int varid=-1;int s=nc_inq_varid(ncid,name.c_str(),&varid);if(s==NC_NOERR)return Var(ncid,varid,dtype,dimids);int redef_rc=nc_redef(ncid);if(redef_rc!=NC_NOERR&&redef_rc!=NC_EINDEFINE)throw NcError(redef_rc);NC_CHECK(nc_def_var(ncid,name.c_str(),to_nc_type(dtype),(int)dimids.size(),dimids.data(),&varid));int enddef_rc=nc_enddef(ncid);if(enddef_rc!=NC_NOERR&&enddef_rc!=NC_ENOTINDEFINE)throw NcError(enddef_rc);return Var(ncid,varid,dtype,dimids);}

// Template specializations for write operations
template<> void Var::write<float>(const std::vector<size_t>& start,const std::vector<size_t>& count,const float* d,size_t n) const{
  if(product(count)!=n)throw std::invalid_argument("size mismatch");
  NC_CHECK(nc_put_vara_float(ncid_,varid_,start.data(),count.data(),d));
}

template<> void Var::write<double>(const std::vector<size_t>& start,const std::vector<size_t>& count,const double* d,size_t n) const{
  if(product(count)!=n)throw std::invalid_argument("size mismatch");
  NC_CHECK(nc_put_vara_double(ncid_,varid_,start.data(),count.data(),d));
}

template<> void Var::write<int32_t>(const std::vector<size_t>& start,const std::vector<size_t>& count,const int32_t* d,size_t n) const{
  if(product(count)!=n)throw std::invalid_argument("size mismatch");
  NC_CHECK(nc_put_vara_int(ncid_,varid_,start.data(),count.data(),d));
}

template<> void Var::write<int16_t>(const std::vector<size_t>& start,const std::vector<size_t>& count,const int16_t* d,size_t n) const{
  if(product(count)!=n)throw std::invalid_argument("size mismatch");
  NC_CHECK(nc_put_vara_short(ncid_,varid_,start.data(),count.data(),d));
}

template<> void Var::write<uint8_t>(const std::vector<size_t>& start,const std::vector<size_t>& count,const uint8_t* d,size_t n) const{
  if(product(count)!=n)throw std::invalid_argument("size mismatch");
  NC_CHECK(nc_put_vara_uchar(ncid_,varid_,start.data(),count.data(),d));
}

// Template specializations for read operations
template<> void Var::read<float>(const std::vector<size_t>& start,const std::vector<size_t>& count,float* o,size_t n) const{
  if(product(count)!=n)throw std::invalid_argument("size mismatch");
  NC_CHECK(nc_get_vara_float(ncid_,varid_,start.data(),count.data(),o));
}

template<> void Var::read<double>(const std::vector<size_t>& start,const std::vector<size_t>& count,double* o,size_t n) const{
  if(product(count)!=n)throw std::invalid_argument("size mismatch");
  NC_CHECK(nc_get_vara_double(ncid_,varid_,start.data(),count.data(),o));
}

template<> void Var::read<int32_t>(const std::vector<size_t>& start,const std::vector<size_t>& count,int32_t* o,size_t n) const{
  if(product(count)!=n)throw std::invalid_argument("size mismatch");
  NC_CHECK(nc_get_vara_int(ncid_,varid_,start.data(),count.data(),o));
}

template<> void Var::read<int16_t>(const std::vector<size_t>& start,const std::vector<size_t>& count,int16_t* o,size_t n) const{
  if(product(count)!=n)throw std::invalid_argument("size mismatch");
  NC_CHECK(nc_get_vara_short(ncid_,varid_,start.data(),count.data(),o));
}

template<> void Var::read<uint8_t>(const std::vector<size_t>& start,const std::vector<size_t>& count,uint8_t* o,size_t n) const{
  if(product(count)!=n)throw std::invalid_argument("size mismatch");
  NC_CHECK(nc_get_vara_uchar(ncid_,varid_,start.data(),count.data(),o));
}

Var define_2d(File& f,const std::string& name,DType dtype,const Dim& d0,const Dim& d1){int a=ensure_dim(f,d0.name,d0.size);int b=ensure_dim(f,d1.name,d1.size);std::vector<int> dimids={a,b};return ensure_var(f,name,dtype,dimids);}

}
