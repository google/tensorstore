#ifndef EXP_NETCDF_WEEK10_TS_NETCDF_IO_H_
#define EXP_NETCDF_WEEK10_TS_NETCDF_IO_H_

#include <netcdf.h>
#include <cstddef>
#include <string>
#include <vector>
#include <stdexcept>

namespace ncutil {

struct Dim { std::string name; size_t size; };

enum class DType { FLOAT32, FLOAT64, INT32, INT16, UINT8, CHAR8 };

class NcError : public std::runtime_error {
public:
  explicit NcError(int code) : std::runtime_error(nc_strerror(code)), code_(code) {}
  int code() const { return code_; }
private:
  int code_;
};

#define NC_CHECK(expr) do { int _r=(expr); if(_r!=NC_NOERR) throw ncutil::NcError(_r); } while(0)

class File {
public:
  File() = default;
  ~File() { close_noexcept(); }
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  File(File&& o) noexcept : ncid_(o.ncid_) { o.ncid_=-1; }
  File& operator=(File&& o) noexcept { if(this!=&o){ close_noexcept(); ncid_=o.ncid_; o.ncid_=-1;} return *this; }
  static File Create(const std::string& path,bool clobber=true){File f;int mode=clobber?NC_CLOBBER:NC_NOCLOBBER;NC_CHECK(nc_create(path.c_str(),mode,&f.ncid_));return f;}
  static File Open(const std::string& path,bool write=false){File f;int mode=write?NC_WRITE:NC_NOWRITE;NC_CHECK(nc_open(path.c_str(),mode,&f.ncid_));return f;}
  int id() const { return ncid_; }
  void EndDef(){NC_CHECK(nc_enddef(ncid_));}
  void ReDef(){NC_CHECK(nc_redef(ncid_));}
  void Sync(){NC_CHECK(nc_sync(ncid_));}
  void Close(){if(ncid_>=0)NC_CHECK(nc_close(ncid_));ncid_=-1;}
private:
  void close_noexcept(){if(ncid_>=0)nc_close(ncid_);ncid_=-1;}
  int ncid_=-1;
};

class Var {
public:
  Var()=default;
  Var(int ncid,int varid,DType dtype,std::vector<int> dimids):ncid_(ncid),varid_(varid),dtype_(dtype),dimids_(std::move(dimids)){}
  template<class T> void write(const std::vector<size_t>& start,const std::vector<size_t>& count,const T* data,size_t n) const;
  template<class T> void read(const std::vector<size_t>& start,const std::vector<size_t>& count,T* out,size_t n) const;
private:
  int ncid_=-1; int varid_=-1; DType dtype_=DType::FLOAT32; std::vector<int> dimids_;
};

int ensure_dim(File& f,const std::string& name,size_t size);
Var ensure_var(File& f,const std::string& name,DType dtype,const std::vector<int>& dimids);
Var define_2d(File& f,const std::string& name,DType dtype,const Dim& d0,const Dim& d1);
size_t product(const std::vector<size_t>& v);

} 
#endif
