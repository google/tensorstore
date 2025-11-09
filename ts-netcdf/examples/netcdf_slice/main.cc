#include <iostream>
#include <string>
#include <vector>
#include <sstream>
extern "C" {
#include <netcdf.h>
}

static void fail(int rc, const char* where){
  std::cerr << where << ": " << nc_strerror(rc) << "\n"; std::exit(1);
}
static std::vector<size_t> parse_csv_sizes(const std::string& s){
  std::vector<size_t> out; std::stringstream ss(s); std::string item;
  while (std::getline(ss,item,',')) out.push_back(static_cast<size_t>(std::stoull(item)));
  return out;
}
static std::string type_name(nc_type t){
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

int main(int argc, char** argv){
  if (argc < 5){
    std::cerr << "usage: " << argv[0] << " <file.nc> <var> <start_csv> <count_csv>\n"
              << "example: " << argv[0] << " /tmp/demo.nc x 0 5\n"
              << "         " << argv[0] << " file.nc temp 0,0,0 1,1,10\n";
    return 2;
  }
  std::string path=argv[1], var=argv[2];
  std::vector<size_t> start = parse_csv_sizes(argv[3]);
  std::vector<size_t> count = parse_csv_sizes(argv[4]);

  int ncid; int rc = nc_open(path.c_str(), NC_NOWRITE, &ncid);
  if (rc != NC_NOERR) fail(rc, "nc_open");

  int varid; rc = nc_inq_varid(ncid, var.c_str(), &varid);
  if (rc != NC_NOERR) fail(rc, "nc_inq_varid");

  char vname[NC_MAX_NAME+1]={0}; nc_type vtype=NC_NAT; int vndims=0, vnatts=0;
  int dimids[NC_MAX_VAR_DIMS]={0};
  rc = nc_inq_var(ncid, varid, vname, &vtype, &vndims, dimids, &vnatts);
  if (rc != NC_NOERR) fail(rc, "nc_inq_var");

  if ((int)start.size()!=vndims || (int)count.size()!=vndims){
    std::cerr << "rank mismatch: var rank=" << vndims
              << " but got start=" << start.size()
              << " count=" << count.size() << "\n";
    return 3;
  }

  std::cout << "Reading " << var << " type=" << type_name(vtype) << " slice:\n  start=[";
  for (int i=0;i<vndims;++i) std::cout << start[i] << (i+1<vndims?", ":"");
  std::cout << "] count=[";
  for (int i=0;i<vndims;++i) std::cout << count[i] << (i+1<vndims?", ":"");
  std::cout << "]\n";

  size_t linear = 1; for (auto c: count) linear *= (c?c:1);

  int rc2 = NC_NOERR;
  switch (vtype){
    case NC_DOUBLE: {
      std::vector<double> buf(linear);
      rc2 = nc_get_vara_double(ncid, varid, start.data(), count.data(), buf.data());
      if (rc2 != NC_NOERR) fail(rc2, "nc_get_vara_double");
      for (size_t i=0;i<buf.size();++i) std::cout << buf[i] << (i+1<buf.size()?", ":"");
      std::cout << "\n";
    } break;
    case NC_FLOAT: {
      std::vector<float> buf(linear);
      rc2 = nc_get_vara_float(ncid, varid, start.data(), count.data(), buf.data());
      if (rc2 != NC_NOERR) fail(rc2, "nc_get_vara_float");
      for (size_t i=0;i<buf.size();++i) std::cout << buf[i] << (i+1<buf.size()?", ":"");
      std::cout << "\n";
    } break;
    case NC_INT: {
      std::vector<int> buf(linear);
      rc2 = nc_get_vara_int(ncid, varid, start.data(), count.data(), buf.data());
      if (rc2 != NC_NOERR) fail(rc2, "nc_get_vara_int");
      for (size_t i=0;i<buf.size();++i) std::cout << buf[i] << (i+1<buf.size()?", ":"");
      std::cout << "\n";
    } break;
    default:
      std::cout << "Type " << type_name(vtype) << " not implemented for slice demo.\n";
  }

  rc = nc_close(ncid);
  if (rc != NC_NOERR) fail(rc, "nc_close");
  return 0;
}
