#include <iostream>
#include <vector>
#include "tensorstore/driver/netcdf/driver.h"

int main(int argc, char** argv) {
  std::string spec = (argc > 1)
    ? argv[1]
    : R"({"driver":"netcdf","path":"/tmp/demo.nc","var":"x","start":[0],"count":[5]})";

  ts_netcdf::Spec s;
  if (auto e = ts_netcdf::ParseSpecJson(spec, &s); !e.empty()) {
    std::cerr << "ParseSpecJson error: " << e << "\n"; return 1;
  }
  std::vector<double> out;
  if (auto e = ts_netcdf::ReadAsDouble(s, &out); !e.empty()) {
    std::cerr << "ReadAsDouble error: " << e << "\n"; return 2;
  }
  for (size_t i = 0; i < out.size(); ++i) {
    std::cout << out[i] << (i + 1 < out.size() ? ", " : "\n");
  }
  return 0;
}
