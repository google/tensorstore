#include <cstdint>
#include <iostream>

#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/index.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
// Ensure N5 driver is linked/registered via all_drivers dep.

namespace ts = tensorstore;

int main() {
  try {
    // N5 spec: uint8, shape [64,64,1], chunks 16x16x1, raw compression.
    auto spec = ts::Spec::FromJson({
      {"driver","n5"},
      {"kvstore", {{"driver","file"}, {"path","data_n5"}}},
      {"metadata", {
          {"dataType","uint8"},
          {"dimensions",{64,64,1}},
          {"blockSize",{16,16,1}},
          {"compression", {{"type","raw"}}}
      }}
    }).value();

    // Open (create if missing). Open returns Future<TensorStore<...>>.
    auto store = ts::Open(
        spec, ts::Context::Default(),
        ts::OpenMode::create | ts::OpenMode::open).result().value();

    // Fill a 64x64x1 ramp.
    auto arr = ts::AllocateArray<std::uint8_t>({64,64,1});
    for (ts::Index i = 0; i < 64; ++i)
      for (ts::Index j = 0; j < 64; ++j)
        arr(i,j,0) = static_cast<std::uint8_t>((i*64 + j) & 0xFF);

    // Write is a free function: Write(value, store).
    auto st = ts::Write(arr, store).result();
    if (!st.ok()) {
      std::cerr << "Write failed: " << st.status() << "\n";
      return 1;
    }

    std::cout << "Seeded N5 dataset at ./data_n5\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
