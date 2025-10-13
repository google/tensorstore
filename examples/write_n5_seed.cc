#include <cstdint>
#include <iostream>
#include <string>
#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/index.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"

namespace ts = tensorstore;

int main(int argc, char** argv) {
  try {
    std::string out_path = (argc > 1) ? argv[1] : std::string("data_n5");

    auto spec = ts::Spec::FromJson({
      {"driver","n5"},
      {"kvstore", {{"driver","file"}, {"path", out_path}}},
      {"metadata", {
          {"dataType","uint8"},
          {"dimensions",{64,64,1}},
          {"blockSize",{16,16,1}},
          {"compression", {{"type","raw"}}}
      }}
    }).value();

    auto store = ts::Open(
        spec, ts::Context::Default(),
        ts::OpenMode::create | ts::OpenMode::open).result().value();

    auto arr = ts::AllocateArray<std::uint8_t>({64,64,1});
    for (ts::Index i = 0; i < 64; ++i)
      for (ts::Index j = 0; j < 64; ++j)
        arr(i,j,0) = static_cast<std::uint8_t>((i*64 + j) & 0xFF);

    auto st = ts::Write(arr, store).result();
    if (!st.ok()) { std::cerr << "Write failed: " << st.status() << "\n"; return 1; }

    std::cout << "Seeded N5 dataset at " << out_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n"; return 1;
  }
}
