#include <iostream>
#include <string>
#include <vector>
#include <cctype>
#include "tensorstore/driver/netcdf/minidriver.h"

using ts_netcdf::Slice; using ts_netcdf::Info; using ts_netcdf::DType;

static std::string get_string(const std::string& j, const char* key){
  auto k = "\"" + std::string(key) + "\"";
  auto p = j.find(k); if(p==std::string::npos) return {};
  p = j.find(':', p); if(p==std::string::npos) return {};
  p = j.find('"', p); if(p==std::string::npos) return {};
  auto q = j.find('"', p+1); if(q==std::string::npos) return {};
  return j.substr(p+1, q-p-1);
}
static std::vector<size_t> get_arr(const std::string& j, const char* key){
  std::vector<size_t> v; auto k="\""+std::string(key)+"\"";
  auto p=j.find(k); if(p==std::string::npos) return v;
  p=j.find('[',p); if(p==std::string::npos) return v;
  auto q=j.find(']',p); if(q==std::string::npos) return v;
  std::string s=j.substr(p+1,q-p-1); size_t i=0;
  while(i<s.size()){
    while(i<s.size() && (std::isspace((unsigned char)s[i])||s[i]==',')) ++i;
    if(i>=s.size()) break;
    size_t j2=i; while(j2<s.size() && std::isdigit((unsigned char)s[j2])) ++j2;
    if(j2>i) v.push_back(std::stoull(s.substr(i,j2-i)));
    i=j2;
  }
  return v;
}

int main(int argc, char** argv){
  if(argc<2){ std::cerr<<"usage: <spec_json>\n"; return 2; }
  std::string spec = argv[1];
  std::string path = get_string(spec,"path");
  std::string var  = get_string(spec,"var");
  Slice s; s.start = get_arr(spec,"start"); s.count = get_arr(spec,"count");
  { auto tmp = get_arr(spec,"stride"); s.stride.assign(tmp.begin(), tmp.end()); }

  std::string err; Info info;
  int rc = ts_netcdf::Inspect(path, var, &info, &err);
  if(rc){ std::cerr << "inspect error: " << err << "\n"; return 1; }

  if(s.start.empty()) s.start.assign(info.shape.size(), 0);
  if(s.count.empty()) s.count = info.shape;

  if(info.dtype == DType::kDouble){
    std::vector<double> buf; rc = ts_netcdf::ReadDoubles(path,var,s,&buf,&err);
    if(rc){ std::cerr<<"read error: "<<err<<"\n"; return 1; }
    for(size_t i=0;i<buf.size();++i){ if(i) std::cout<<", "; std::cout<<buf[i]; }
    std::cout<<"\n";
  }else if(info.dtype == DType::kFloat){
    std::vector<float> buf; rc = ts_netcdf::ReadFloats(path,var,s,&buf,&err);
    if(rc){ std::cerr<<"read error: "<<err<<"\n"; return 1; }
    for(size_t i=0;i<buf.size();++i){ if(i) std::cout<<", "; std::cout<<buf[i]; }
    std::cout<<"\n";
  }else if(info.dtype == DType::kInt32){
    std::vector<int> buf; rc = ts_netcdf::ReadInts(path,var,s,&buf,&err);
    if(rc){ std::cerr<<"read error: "<<err<<"\n"; return 1; }
    for(size_t i=0;i<buf.size();++i){ if(i) std::cout<<", "; std::cout<<buf[i]; }
    std::cout<<"\n";
  }else{
    std::cerr<<"dtype not supported in mini-driver\n"; return 3;
  }
  return 0;
}
