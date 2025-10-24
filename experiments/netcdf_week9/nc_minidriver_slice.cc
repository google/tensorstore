#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "tensorstore/driver/netcdf/minidriver.h"

static void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --path=/tmp/week9.nc --var=temp --slices=\"0:3:1, 0:4:1, 0:5:1\"\n";
}

struct SL { long start, stop, stride; };

static bool parse_slices(const std::string& s, std::vector<SL>* out) {
  // parse "a:b:c, d:e:f, ..."
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (i >= s.size()) break;
    char* end1 = nullptr; long a = std::strtol(&s[i], &end1, 10);
    if (end1 == &s[i] || *end1 != ':') return false;
    i = (end1 - s.data()) + 1;
    char* end2 = nullptr; long b = std::strtol(&s[i], &end2, 10);
    if (end2 == &s[i] || *end2 != ':') return false;
    i = (end2 - s.data()) + 1;
    char* end3 = nullptr; long c = std::strtol(&s[i], &end3, 10);
    if (end3 == &s[i]) return false;
    i = (end3 - s.data());
    out->push_back({a,b,c});
    while (i < s.size() && isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (i < s.size() && s[i] == ',') ++i;
  }
  return true;
}

int main(int argc, char** argv) {
  std::string path, var, slices_s;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if      (a.rfind("--path=",   0) == 0) path     = a.substr(7);
    else if (a.rfind("--var=",    0) == 0) var      = a.substr(6);
    else if (a.rfind("--slices=", 0) == 0) slices_s = a.substr(9);
    else { usage(argv[0]); return 1; }
  }
  if (path.empty() || var.empty() || slices_s.empty()) { usage(argv[0]); return 1; }

  ts_netcdf::Info info;
  std::string err;
  if (ts_netcdf::Inspect(path, var, &info, &err) != 0) {
    std::cerr << "Inspect failed: " << err << "\n";
    return 2;
  }

  std::vector<SL> sl;
  if (!parse_slices(slices_s, &sl)) {
    std::cerr << "Bad --slices string\n";
    return 1;
  }
  if (sl.size() != info.shape.size()) {
    std::cerr << "slice rank (" << sl.size()
              << ") must match variable rank (" << info.shape.size() << ")\n";
    return 1;
  }

  // Build ts_netcdf::Slice (uses start/count/stride)
  ts_netcdf::Slice slice;
  slice.start.resize(sl.size());
  slice.count.resize(sl.size());
  slice.stride.resize(sl.size());

  for (size_t i = 0; i < sl.size(); ++i) {
    long start = sl[i].start;
    long stop  = sl[i].stop;    // exclusive
    long st    = sl[i].stride;  // assume >0 for demo

    slice.start[i]  = static_cast<size_t>(start);
    size_t len = (stop <= start) ? 0
               : static_cast<size_t>((stop - start + st - 1) / st);
    slice.count[i]  = len;
    slice.stride[i] = static_cast<ptrdiff_t>(st);
  }

  // Read according to dtype
  int rc = 0;
  if (info.dtype == ts_netcdf::DType::kFloat) {
    std::vector<float> out;
    rc = ts_netcdf::ReadFloats(path, var, slice, &out, &err);
    if (rc != 0) { std::cerr << "Read failed: " << err << "\n"; return 3; }

    std::cout << "Shape:";
    for (auto c : slice.count) std::cout << " " << c;
    std::cout << "\nValues:\n";
    if (!slice.count.empty()) {
      size_t last = slice.count.back();
      for (size_t i = 0; i < out.size(); ++i) {
        std::cout << out[i] << ((last && (i + 1) % last == 0) ? "\n" : " ");
      }
    } else {
      for (auto v : out) std::cout << v << " ";
      std::cout << "\n";
    }
  } else if (info.dtype == ts_netcdf::DType::kDouble) {
    std::vector<double> out;
    rc = ts_netcdf::ReadDoubles(path, var, slice, &out, &err);
    if (rc != 0) { std::cerr << "Read failed: " << err << "\n"; return 3; }
    std::cout << "Read " << out.size() << " doubles\n";
  } else if (info.dtype == ts_netcdf::DType::kInt32) {
    std::vector<int> out;
    rc = ts_netcdf::ReadInts(path, var, slice, &out, &err);
    if (rc != 0) { std::cerr << "Read failed: " << err << "\n"; return 3; }
    std::cout << "Read " << out.size() << " ints\n";
  } else {
    std::cerr << "Unsupported dtype\n";
    return 4;
  }

  return 0;
}
