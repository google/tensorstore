#include "tensorstore/driver/netcdf/driver.h"

#include <cstdlib>
#include <sstream>
#include <type_traits>

extern "C" {
#include <netcdf.h>
}

#include "tensorstore/driver/netcdf/minidriver.h"  // Inspect/ReadTyped

namespace ts_netcdf {

static inline std::string jerr(const char* m){ return std::string("JSON: ") + m; }

// Minimal helpers to parse arrays/strings from a simple JSON string.
static bool parse_int_array(const std::string& j, const char* key,
                            std::vector<long long>* out_ll,
                            bool required) {
  auto k = std::string("\"") + key + "\"";
  auto p = j.find(k);
  if (p == std::string::npos) return !required;
  p = j.find('[', p);
  if (p == std::string::npos) return false;
  auto q = j.find(']', p);
  if (q == std::string::npos) return false;
  auto a = j.substr(p + 1, q - p - 1);
  std::stringstream ss(a);
  out_ll->clear();
  while (ss.good()) {
    long long v; char c;
    if (!(ss >> v)) break;
    out_ll->push_back(v);
    ss >> c;  // eat comma if any
  }
  return true;
}

static bool parse_string(const std::string& j, const char* key, std::string* out,
                         bool required) {
  auto k = std::string("\"") + key + "\"";
  auto p = j.find(k);
  if (p == std::string::npos) return !required;
  p = j.find('"', p + k.size());
  if (p == std::string::npos) return false;
  auto q = j.find('"', p + 1);
  if (q == std::string::npos) return false;
  *out = j.substr(p + 1, q - p - 1);
  return true;
}

std::string ParseSpecJson(const std::string& json, Spec* out) {
  if (!out) return jerr("null output");
  out->path.clear(); out->var.clear();
  out->start.clear(); out->count.clear(); out->stride.clear();

  std::string driver;
  if (!parse_string(json, "driver", &driver, true)) return jerr("missing driver");
  if (driver != "netcdf") return "driver must be \"netcdf\"";

  if (!parse_string(json, "path", &out->path, true)) return jerr("missing path");
  if (!parse_string(json, "var",  &out->var,  true)) return jerr("missing var");

  std::vector<long long> start_ll, count_ll, stride_ll;
  if (!parse_int_array(json, "start",  &start_ll,  true))  return jerr("bad start");
  if (!parse_int_array(json, "count",  &count_ll,  true))  return jerr("bad count");
  (void)parse_int_array(json, "stride", &stride_ll, false); // optional

  out->start.assign(start_ll.begin(), start_ll.end());
  out->count.assign(count_ll.begin(), count_ll.end());
  out->stride.assign(stride_ll.begin(), stride_ll.end());   // ll -> ptrdiff_t

  return "";
}

std::string ReadAsDouble(const Spec& s, std::vector<double>* out) {
  if (!out) return "output buffer is null";
  ts_netcdf::Info info;
  std::string err;
  int rc = ts_netcdf::Inspect(s.path, s.var, &info, &err);
  if (rc) return err.empty() ? "inspect failed" : err;

  // rank checks
  if (s.start.size() != info.shape.size() || s.count.size() != info.shape.size())
    return "rank mismatch: start/count vs variable rank";
  if (!s.stride.empty() && s.stride.size() != info.shape.size())
    return "stride rank mismatch";

  // number of elements
  size_t n = 1;
  for (auto c : s.count) n *= c;
  out->assign(n, 0.0);

  ts_netcdf::Slice slice;
  slice.start = s.start;
  slice.count = s.count;
  if (!s.stride.empty()) slice.stride = s.stride;

  if (info.dtype == ts_netcdf::DType::kDouble) {
    std::vector<double> tmp(n);
    rc = ts_netcdf::ReadDoubles(s.path, s.var, slice, &tmp, &err);
    if (rc) return err;
    *out = std::move(tmp);
    return "";
  } else if (info.dtype == ts_netcdf::DType::kFloat) {
    std::vector<float> tmp(n);
    rc = ts_netcdf::ReadFloats(s.path, s.var, slice, &tmp, &err);
    if (rc) return err;
    for (size_t i = 0; i < n; ++i) (*out)[i] = tmp[i];
    return "";
  } else if (info.dtype == ts_netcdf::DType::kInt32) {
    std::vector<int> tmp(n);
    rc = ts_netcdf::ReadInts(s.path, s.var, slice, &tmp, &err);
    if (rc) return err;
    for (size_t i = 0; i < n; ++i) (*out)[i] = static_cast<double>(tmp[i]);
    return "";
  }
  return "unsupported dtype for demo driver";
}

}  // namespace ts_netcdf
