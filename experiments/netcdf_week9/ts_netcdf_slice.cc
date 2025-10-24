#include "tensorstore/context.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/open.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/tensorstore.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

struct DimSlice { long start, stop, stride; };

static bool StartsWith(const std::string& s, const char* k) {
  return s.rfind(k, 0) == 0;
}

static bool ParseSlices(const std::string& s, std::vector<DimSlice>* out) {
  out->clear();
  size_t i = 0;
  while (i < s.size()) {
    // skip spaces and commas
    while (i < s.size() && (s[i] == ' ' || s[i] == ',')) ++i;
    if (i >= s.size()) break;
    size_t j = s.find(':', i);
    if (j == std::string::npos) return false;
    size_t k = s.find(':', j + 1);
    if (k == std::string::npos) return false;
    size_t end = s.find(',', k + 1);
    std::string a = s.substr(i, j - i);
    std::string b = s.substr(j + 1, k - (j + 1));
    std::string c = s.substr(k + 1, (end == std::string::npos ? s.size() : end) - (k + 1));
    DimSlice ds{std::stol(a), std::stol(b), std::stol(c)};
    out->push_back(ds);
    i = (end == std::string::npos) ? s.size() : end + 1;
  }
  return true;
}

int main(int argc, char** argv) {
  std::string path, var, slices;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (StartsWith(arg, "--path="))   path   = arg.substr(7);
    else if (StartsWith(arg, "--var="))     var    = arg.substr(6);
    else if (StartsWith(arg, "--slices="))  slices = arg.substr(9);
  }

  if (path.empty() || var.empty() || slices.empty()) {
    std::cerr << "Usage: "
              << argv[0]
              << " --path=/tmp/week9.nc --var=temp --slices=\"0:2:1, 0:3:1, 1:5:2\"\n";
    return 2;
  }

  std::vector<DimSlice> sl;
  if (!ParseSlices(slices, &sl)) {
    std::cerr << "Bad --slices. Use start:stop:stride per-dimension, comma-separated.\n";
    return 2;
  }

  // Open via your TensorStore netCDF driver.
  auto spec = tensorstore::Spec::FromJson({
      {"driver","netcdf"},
      {"path", path},
      {"variable", var}
  }).value();

  auto ctx = tensorstore::Context::Default();
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto ts, tensorstore::Open(spec, ctx).result());

  // Apply half-open + stride per dimension in file order.
  using namespace tensorstore;
  auto sliced = ts;
  for (size_t i = 0; i < sl.size(); ++i) {
    
    { auto __res = (sliced | Dims(static_cast<DimensionIndex>(i))
                     .HalfOpenInterval(sl[i].start, sl[i].stop)
                     .Stride(sl[i].stride));
      if (!__res.ok()) { std::cerr << __res.status() << "\n"; return 2; }
      sliced = std::move(__res).value();
    }
  
  }

  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto arr, tensorstore::Read(sliced).result());

  // Print shape
  std::cout << "Shape:";
  for (auto d : arr.shape()) std::cout << " " << d;
  std::cout << "\nValues:\n";

  // Assume float variable for the demo dataset.
  const float* data = static_cast<const float*>(arr.data());
  const size_t n = arr.num_elements();
  size_t last = arr.rank() ? arr.shape().back() : n;

  for (size_t i = 0; i < n; ++i) {
    std::cout << data[i] << ((last && (i + 1) % last == 0) ? "\n" : " ");
  }
  std::cout << "\n";
  return 0;
}
