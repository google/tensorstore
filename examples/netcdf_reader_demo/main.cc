#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include "tensorstore/driver/netcdf/reader.h"

static std::vector<size_t> parse_size_ts(const std::string& s){
  std::vector<size_t> v; std::stringstream ss(s); std::string tok;
  while(std::getline(ss, tok, ',')) v.push_back(static_cast<size_t>(std::stoull(tok)));
  return v;
}
static std::vector<ptrdiff_t> parse_stride(const std::string& s){
  std::vector<ptrdiff_t> v; std::stringstream ss(s); std::string tok;
  while(std::getline(ss, tok, ',')) v.push_back(static_cast<ptrdiff_t>(std::stoll(tok)));
  return v;
}

int main(int argc, char** argv){
  if(argc < 5){
    std::cerr << "Usage: " << argv[0] << " <path> <var> <start_csv> <count_csv> [stride_csv]\n";
    return 2;
  }
  std::string path = argv[1];
  std::string var  = argv[2];
  ts_netcdf::Slice s;
  s.start  = parse_size_ts(argv[3]);
  s.count  = parse_size_ts(argv[4]);
  if(argc >= 6) s.stride = parse_stride(argv[5]);

  std::vector<double> out;
  std::string err;
  int rc = ts_netcdf::ReadDoubles(path, var, s, &out, &err);
  if(rc){
    std::cerr << "ReadDoubles failed: " << err << "\n";
    return 1;
  }
  for(size_t i=0;i<out.size();++i){
    std::cout << out[i] << (i+1<out.size()?", ":"");
  }
  std::cout << "\n";
  return 0;
}
