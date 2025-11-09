#include <netcdf.h>
#include <iostream>
#include <vector>
#include <string>
#include <cctype>
#include <cstddef>   // for ptrdiff_t

static std::string get_str(const std::string& j, const std::string& key){
  auto k="\""+key+"\""; auto p=j.find(k); if(p==std::string::npos) return {};
  p=j.find(':',p); if(p==std::string::npos) return {};
  auto s=j.find('"',p+1); if(s==std::string::npos) return {};
  auto e=j.find('"',s+1); if(e==std::string::npos) return {};
  return j.substr(s+1,e-s-1);
}
static std::vector<size_t> get_arr(const std::string& j, const std::string& key){
  std::vector<size_t> out; auto k="\""+key+"\""; auto p=j.find(k); if(p==std::string::npos) return out;
  p=j.find('[',p); auto e=j.find(']',p); if(p==std::string::npos||e==std::string::npos) return out;
  auto seg=j.substr(p+1,e-p-1);
  size_t i=0; while(i<seg.size()){
    while(i<seg.size() && (seg[i]==' '||seg[i]==',')) ++i;
    size_t j2=i; while(j2<seg.size() && std::isdigit(static_cast<unsigned char>(seg[j2]))) ++j2;
    if(j2>i) out.push_back(static_cast<size_t>(std::stoull(seg.substr(i,j2-i))));
    i=j2+1;
  }
  return out;
}
static void fail(int rc,const char* where){ std::cerr<<where<<": "<<nc_strerror(rc)<<"\n"; std::exit(1); }

int main(int argc, char** argv){
  if(argc<2){ std::cerr<<"usage: "<<argv[0]<<" '{\"path\":\"/tmp/demo2d.nc\",\"var\":\"z\",\"start\":[0,0],\"count\":[1,4],\"stride\":[1,2]}'\n"; return 2; }
  std::string j = argv[1];
  std::string path = get_str(j,"path");
  std::string var  = get_str(j,"var");
  auto start  = get_arr(j,"start");
  auto count  = get_arr(j,"count");
  auto stride_sz = get_arr(j,"stride"); // optional size_t stride

  if(path.empty()||var.empty()||start.empty()||count.empty()){ std::cerr<<"bad json\n"; return 1; }

  int ncid; int rc = nc_open(path.c_str(), NC_NOWRITE, &ncid); if(rc) fail(rc,"nc_open");
  int varid; rc = nc_inq_varid(ncid, var.c_str(), &varid); if(rc) fail(rc,"nc_inq_varid");
  nc_type vtype=NC_NAT; int vnd=0; rc = nc_inq_var(ncid, varid, nullptr, &vtype, &vnd, nullptr, nullptr); if(rc) fail(rc,"nc_inq_var");
  if((int)start.size()!=vnd || (int)count.size()!=vnd){ std::cerr<<"rank mismatch\n"; return 1; }

  size_t n=1; for(auto c:count) n*=c;

  // Convert optional stride (size_t) to ptrdiff_t as required by netCDF *vars* APIs.
  std::vector<ptrdiff_t> stride;
  bool use_stride = false;
  if(!stride_sz.empty()){
    if((int)stride_sz.size()!=vnd){ std::cerr<<"stride rank mismatch\n"; return 1; }
    stride.resize(stride_sz.size());
    for(size_t i=0;i<stride_sz.size();++i) stride[i] = static_cast<ptrdiff_t>(stride_sz[i]);
    use_stride = true;
  }

  if(vtype==NC_DOUBLE){
    std::vector<double> buf(n);
    if(use_stride) rc = nc_get_vars_double(ncid,varid,start.data(),count.data(),stride.data(),buf.data());
    else           rc = nc_get_vara_double(ncid,varid,start.data(),count.data(),buf.data());
    if(rc) fail(rc, use_stride?"nc_get_vars_double":"nc_get_vara_double");
    for(size_t i=0;i<n;i++){ std::cout<<buf[i]<<(i+1<n?", ":""); } std::cout<<"\n";
  } else if(vtype==NC_FLOAT){
    std::vector<float> buf(n);
    if(use_stride) rc = nc_get_vars_float(ncid,varid,start.data(),count.data(),stride.data(),buf.data());
    else           rc = nc_get_vara_float(ncid,varid,start.data(),count.data(),buf.data());
    if(rc) fail(rc, use_stride?"nc_get_vars_float":"nc_get_vara_float");
    for(size_t i=0;i<n;i++){ std::cout<<buf[i]<<(i+1<n?", ":""); } std::cout<<"\n";
  } else if(vtype==NC_INT){
    std::vector<int> buf(n);
    if(use_stride) rc = nc_get_vars_int(ncid,varid,start.data(),count.data(),stride.data(),buf.data());
    else           rc = nc_get_vara_int(ncid,varid,start.data(),count.data(),buf.data());
    if(rc) fail(rc, use_stride?"nc_get_vars_int":"nc_get_vara_int");
    for(size_t i=0;i<n;i++){ std::cout<<buf[i]<<(i+1<n?", ":""); } std::cout<<"\n";
  } else {
    std::cerr<<"type not handled in demo\n"; return 1;
  }
  rc = nc_close(ncid); if(rc) fail(rc,"nc_close");
  return 0;
}
