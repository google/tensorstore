#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cctype>
extern "C" {
#include <netcdf.h>
}

static void fail(int rc, const char* where){
  std::cerr << where << ": " << nc_strerror(rc) << "\n"; std::exit(1);
}

static std::string strip(std::string s){
  size_t i=0; while(i<s.size() && std::isspace((unsigned char)s[i])) ++i;
  size_t j=s.size(); while(j>i && std::isspace((unsigned char)s[j-1])) --j;
  return s.substr(i, j-i);
}

// super-lightweight JSON getters (demo only)
static bool find_string(const std::string& js, const std::string& key, std::string& out){
  auto k = js.find("\""+key+"\""); if (k==std::string::npos) return false;
  k = js.find(':', k); if (k==std::string::npos) return false;
  k = js.find('"', k); if (k==std::string::npos) return false;
  auto e = js.find('"', k+1); if (e==std::string::npos) return false;
  out = js.substr(k+1, e-(k+1)); return true;
}
static bool find_array(const std::string& js, const std::string& key, std::vector<size_t>& out){
  auto k = js.find("\""+key+"\""); if (k==std::string::npos) return false;
  k = js.find('[', k); if (k==std::string::npos) return false;
  auto e = js.find(']', k); if (e==std::string::npos) return false;
  std::string body = js.substr(k+1, e-(k+1));
  std::stringstream ss(body); std::string item; out.clear();
  while(std::getline(ss,item,',')) { item = strip(item); if(!item.empty()) out.push_back((size_t)std::stoull(item)); }
  return true;
}

int main(int argc, char** argv){
  if (argc < 2){
    std::cerr << "usage: " << argv[0] << " '<json-spec>'\n"
              << "example: " << argv[0] << " '{\"driver\":\"netcdf\",\"path\":\"/tmp/demo.nc\",\"var\":\"x\",\"start\":[0],\"count\":[5]}'\n";
  }
  std::string js = (argc>=2) ? argv[1] : "{}";

  std::string driver, path, var;
  std::vector<size_t> start, count;

  if (!find_string(js, "driver", driver) || driver != "netcdf"){
    std::cerr << "Invalid spec: driver must be \"netcdf\"\n"; return 3;
  }
  if (!find_string(js, "path", path)){ std::cerr << "Spec missing: path\n"; return 3; }
  (void)find_string(js, "var", var);      // optional
  (void)find_array(js, "start", start);   // optional
  (void)find_array(js, "count", count);   // optional

  int ncid; int rc = nc_open(path.c_str(), NC_NOWRITE, &ncid);
  if (rc != NC_NOERR) fail(rc, "nc_open");

  if (var.empty()){
    int ndims=0, nvars=0, ngatts=0, unlim=-1;
    rc = nc_inq(ncid, &ndims, &nvars, &ngatts, &unlim);
    if (rc != NC_NOERR) fail(rc, "nc_inq");
    std::cout << "Opened " << path << " (dims=" << ndims << ", vars=" << nvars << ")\n";
    nc_close(ncid);
    return 0;
  }

  int varid; rc = nc_inq_varid(ncid, var.c_str(), &varid);
  if (rc != NC_NOERR) fail(rc, "nc_inq_varid");

  char vname[NC_MAX_NAME+1]={0}; nc_type vtype=NC_NAT; int vndims=0, vnatts=0;
  int dimids[NC_MAX_VAR_DIMS]={0};
  rc = nc_inq_var(ncid, varid, vname, &vtype, &vndims, dimids, &vnatts);
  if (rc != NC_NOERR) fail(rc, "nc_inq_var");

  // default slice if none provided: last dim up to 10
  if ((int)start.size()!=vndims || (int)count.size()!=vndims){
    start.assign(vndims, 0); count.assign(vndims, 1);
    if (vndims>0){
      char dname[NC_MAX_NAME+1]={0}; size_t dlen=1;
      rc = nc_inq_dim(ncid, dimids[vndims-1], dname, &dlen);
      if (rc != NC_NOERR) fail(rc, "nc_inq_dim");
      count[vndims-1] = std::min<size_t>(dlen, 10);
    }
  }

  size_t linear=1; for (auto c: count) linear *= (c?c:1);

  switch (vtype){
    case NC_DOUBLE: {
      std::vector<double> buf(linear);
      rc = nc_get_vara_double(ncid, varid, start.data(), count.data(), buf.data());
      if (rc != NC_NOERR) fail(rc, "nc_get_vara_double");
      for (size_t i=0;i<buf.size();++i) std::cout << buf[i] << (i+1<buf.size()?", ":"");
      std::cout << "\n";
    } break;
    case NC_FLOAT: {
      std::vector<float> buf(linear);
      rc = nc_get_vara_float(ncid, varid, start.data(), count.data(), buf.data());
      if (rc != NC_NOERR) fail(rc, "nc_get_vara_float");
      for (size_t i=0;i<buf.size();++i) std::cout << buf[i] << (i+1<buf.size()?", ":"");
      std::cout << "\n";
    } break;
    case NC_INT: {
      std::vector<int> buf(linear);
      rc = nc_get_vara_int(ncid, varid, start.data(), count.data(), buf.data());
      if (rc != NC_NOERR) fail(rc, "nc_get_vara_int");
      for (size_t i=0;i<buf.size();++i) std::cout << buf[i] << (i+1<buf.size()?", ":"");
      std::cout << "\n";
    } break;
    default:
      std::cout << "Type not implemented in demo.\n";
  }

  rc = nc_close(ncid);
  if (rc != NC_NOERR) fail(rc, "nc_close");
  return 0;
}
