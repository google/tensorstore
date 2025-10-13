#ifndef TENSORSTORE_DRIVER_NETCDF_NETCDF_JSON_H_
#define TENSORSTORE_DRIVER_NETCDF_NETCDF_JSON_H_
#include <string>
#include <utility>

namespace tensorstore {
namespace internal_netcdf {

struct NetcdfSpec {
  std::string driver;   // always "netcdf"
  std::string path;     // dataset path (optional in stub)
};

inline std::pair<bool,std::string> ParseNetcdfSpec(const std::string& json, NetcdfSpec* out) {
  // Super-minimal, demo-only: accept {"driver":"netcdf", "path":"..."}
  // We only need to recognize the driver and optional "path".
  if (json.find("\"driver\"") == std::string::npos ||
      json.find("netcdf") == std::string::npos) {
    return {false, "driver is not 'netcdf'"};
  }
  out->driver = "netcdf";
  auto p = json.find("\"path\"");
  if (p != std::string::npos) {
    auto q = json.find('"', p + 6);
    if (q != std::string::npos) {
      auto r = json.find('"', q + 1);
      if (r != std::string::npos) out->path = json.substr(q + 1, r - q - 1);
    }
  }
  return {true, ""};
}

}  // namespace internal_netcdf
}  // namespace tensorstore
#endif
