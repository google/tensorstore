#include <iostream>
#include <string>
#include <vector>
#include <type_traits>
extern "C" {
#include <netcdf.h>
}

static void fail(int rc, const char* where) {
  std::cerr << where << ": " << nc_strerror(rc) << "\n";
  std::exit(1);
}
static std::string type_name(nc_type t) {
  switch(t){
    case NC_BYTE: return "NC_BYTE"; case NC_CHAR: return "NC_CHAR";
    case NC_SHORT: return "NC_SHORT"; case NC_INT: return "NC_INT";
    case NC_FLOAT: return "NC_FLOAT"; case NC_DOUBLE: return "NC_DOUBLE";
    case NC_UBYTE: return "NC_UBYTE"; case NC_USHORT: return "NC_USHORT";
    case NC_UINT: return "NC_UINT"; case NC_INT64: return "NC_INT64";
    case NC_UINT64: return "NC_UINT64"; case NC_STRING: return "NC_STRING";
    default: return "NC_NAT";
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <file.nc> <varname>\n";
    return 2;
  }
  std::string path = argv[1], var = argv[2];

  int ncid; int rc = nc_open(path.c_str(), NC_NOWRITE, &ncid);
  if (rc != NC_NOERR) fail(rc, "nc_open");

  int varid; rc = nc_inq_varid(ncid, var.c_str(), &varid);
  if (rc != NC_NOERR) fail(rc, "nc_inq_varid");

  char vname[NC_MAX_NAME+1]={0}; nc_type vtype=NC_NAT; int vndims=0, vnatts=0;
  int dimids[NC_MAX_VAR_DIMS]={0};
  rc = nc_inq_var(ncid, varid, vname, &vtype, &vndims, dimids, &vnatts);
  if (rc != NC_NOERR) fail(rc, "nc_inq_var");

  std::vector<size_t> dims(vndims, 0);
  for (int i=0;i<vndims;++i) {
    char dname[NC_MAX_NAME+1]={0}; size_t dlen=0;
    rc = nc_inq_dim(ncid, dimids[i], dname, &dlen);
    if (rc != NC_NOERR) fail(rc, "nc_inq_dim");
    dims[i]=dlen;
  }

  std::cout << "Variable: " << vname << " type=" << type_name(vtype) << " shape=[";
  for (int i=0;i<vndims;++i) { std::cout << dims[i] << (i+1<vndims?", ":""); }
  std::cout << "]\n";

  size_t total = 1; for (auto d: dims) total *= d ? d : 1;
  size_t want = std::min<size_t>(total, 10);

  std::vector<size_t> start(vndims, 0), count(vndims, 1);
  if (vndims>0) count[vndims-1] = std::min(dims[vndims-1], want);

  auto dump = [&](auto* buf, const char* label){
    std::cout << label << " first " << count.back() << " vals: ";
    for (size_t i=0;i<count.back();++i) std::cout << buf[i] << (i+1<count.back()?", ":"");
    std::cout << "\n";
  };

  int rc2 = NC_NOERR;
  switch (vtype) {
    case NC_FLOAT: {
      std::vector<float> buf(count.back());
      rc2 = nc_get_vara_float(ncid, varid, start.data(), count.data(), buf.data());
      if (rc2 != NC_NOERR) fail(rc2, "nc_get_vara_float");
      dump(buf.data(), "NC_FLOAT");
    } break;
    case NC_DOUBLE: {
      std::vector<double> buf(count.back());
      rc2 = nc_get_vara_double(ncid, varid, start.data(), count.data(), buf.data());
      if (rc2 != NC_NOERR) fail(rc2, "nc_get_vara_double");
      dump(buf.data(), "NC_DOUBLE");
    } break;
    case NC_INT: {
      std::vector<int> buf(count.back());
      rc2 = nc_get_vara_int(ncid, varid, start.data(), count.data(), buf.data());
      if (rc2 != NC_NOERR) fail(rc2, "nc_get_vara_int");
      dump(buf.data(), "NC_INT");
    } break;
    case NC_SHORT: {
      std::vector<short> buf(count.back());
      rc2 = nc_get_vara_short(ncid, varid, start.data(), count.data(), buf.data());
      if (rc2 != NC_NOERR) fail(rc2, "nc_get_vara_short");
      dump(buf.data(), "NC_SHORT");
    } break;
    default:
      std::cout << "Reading preview for type " << type_name(vtype) << " not implemented in demo.\n";
      break;
  }

  rc = nc_close(ncid);
  if (rc != NC_NOERR) fail(rc, "nc_close");
  return 0;
}
