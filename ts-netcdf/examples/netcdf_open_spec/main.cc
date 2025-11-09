#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#include "tensorstore/driver/netcdf/reader.h"

static std::string get_field(const std::string& j, const std::string& key){
  auto pos = j.find("\""+key+"\"");
  if (pos==std::string::npos) return "";
  pos = j.find(':', pos);
  if (pos==std::string::npos) return "";
  ++pos;
  while (pos<j.size() && isspace(static_cast<unsigned char>(j[pos]))) ++pos;
  if (pos<j.size() && j[pos]=='"') {
    ++pos; std::string out;
    while (pos<j.size() && j[pos]!='"') out.push_back(j[pos++]);
    return out;
  }
  std::string out;
  while (pos<j.size() && j[pos]!=',' && j[pos]!=']' && j[pos]!='}' && !isspace(static_cast<unsigned char>(j[pos])))
    out.push_back(j[pos++]);
  return out;
}

static std::vector<size_t> get_arr_size_t(const std::string& j, const std::string& key){
  std::vector<size_t> v; auto pos = j.find("\""+key+"\"");
  if (pos==std::string::npos) return v;
  pos = j.find('[', pos);
  if (pos==std::string::npos) return v;
  ++pos; std::string num;
  while (pos<j.size() && j[pos]!=']') {
    char c = j[pos++];
    if (c==',' || c==']') { if(!num.empty()){ v.push_back(static_cast<size_t>(std::stoull(num))); num.clear(); } }
    else if (!isspace(static_cast<unsigned char>(c))) num.push_back(c);
  }
  if(!num.empty()) v.push_back(static_cast<size_t>(std::stoull(num)));
  return v;
}

static std::vector<ptrdiff_t> get_arr_stride(const std::string& j, const std::string& key){
  std::vector<ptrdiff_t> v; auto pos = j.find("\""+key+"\"");
  if (pos==std::string::npos) return v;
  pos = j.find('[', pos);
  if (pos==std::string::npos) return v;
  ++pos; std::string num;
  while (pos<j.size() && j[pos]!=']') {
    char c = j[pos++];
    if (c==',' || c==']') { if(!num.empty()){ v.push_back(static_cast<ptrdiff_t>(std::stoll(num))); num.clear(); } }
    else if (!isspace(static_cast<unsigned char>(c))) num.push_back(c);
  }
  if(!num.empty()) v.push_back(static_cast<ptrdiff_t>(std::stoll(num)));
  return v;
}

int main(int argc, char** argv){
  if(argc != 2){
    std::cerr << "Usage: " << argv[0] << " '{\"driver\":\"netcdf\",\"path\":\"/tmp/demo.nc\",\"var\":\"x\",\"start\":[...],\"count\":[...],\"stride\":[...]}'\n";
    return 2;
  }
  std::string j = argv[1];
  if (get_field(j,"driver") != "netcdf") {
    std::cerr << "Unsupported driver (expecting netcdf)\n"; return 2;
  }
  std::string path = get_field(j,"path");
  std::string var  = get_field(j,"var");
  if (path.empty() || var.empty()){
    std::cerr << "Missing 'path' or 'var' in spec\n"; return 2;
  }

  ts_netcdf::Slice s;
  s.start  = get_arr_size_t(j,"start");
  s.count  = get_arr_size_t(j,"count");
  s.stride = get_arr_stride(j,"stride"); // optional

  if (s.start.empty() || s.count.empty() || s.start.size()!=s.count.size()){
    std::cerr << "start/count must be present and same rank\n"; return 2;
  }
  if (!s.stride.empty() && s.stride.size()!=s.count.size()){
    std::cerr << "stride rank must match start/count\n"; return 2;
  }

  std::vector<double> out;
  std::string err;
  int rc = ts_netcdf::ReadDoubles(path, var, s, &out, &err);
  if (rc) {
    std::cerr << "Read failed: " << err << "\n"; return 1;
  }
  for(size_t i=0;i<out.size();++i){
    std::cout << out[i] << (i+1<out.size()?", ":"");
  }
  std::cout << "\n";
  return 0;
}
