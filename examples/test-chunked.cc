#include <unistd.h>

#include <chrono>

#include "tensorstore/context.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/status.h"
#include "tensorstore/virtual_chunked.h"

template <typename Array>
void PrintCSVArray(Array&& data) {
  if (data.rank() == 0) {
    std::cout << data << std::endl;
    return;
  }

  // Iterate over the shape of the data array, which gives us one
  // reference for every element.
  //
  // The builtin streaming operator outputs data in C++ array initialization
  // syntax: {{0, 0}, {1, 0}}, but this routine prefers CSV-formatted output.
  //
  // The output of this function is equivalent to:
  //
  // for (int x = 0; x < data.shape()[0]; x++)
  //  for (int y = 0; y < data.shape()[1]; y++) {
  //     ...
  //       std::cout << data[x][y][...] << "\t";
  //  }
  //
  const auto max = data.shape()[data.rank() - 1] - 1;
  auto element_rep = data.dtype();

  // FIXME: We can't use operator() to get a value reference since that doesn't
  // work for tensorstore::ArrayView<const void, N>. However in the case of
  // printing, rank-0 arrays have been overloaded to print correctly, and so we
  // can do this:
  std::string s;
  tensorstore::IterateOverIndexRange(  //
      data.shape(), [&](tensorstore::span<const tensorstore::Index> idx) {
        element_rep->append_to_string(&s, data[idx].pointer());
        if (*idx.rbegin() == max) {
          std::cout << s << std::endl;
          s.clear();
        } else {
          s.append("\t");
        }
      });
  std::cout << s << std::endl;
}

namespace {

namespace kvstore = tensorstore::kvstore;
using ::tensorstore::KvStore;
using ::tensorstore::StorageGeneration;

KvStore GetStore(std::string root) {
  return kvstore::Open({{"driver", "file"}, {"path", root + "/"}}).value();
}

}  // namespace

// int main(int argc, char** argv) {
//   auto store =
//   GetStore("/Users/hsidky/Code/tensorstore/examples/ts_resources");

//   // Read a byte range.
//   kvstore::ReadOptions kvs_read_options;
//   tensorstore::ByteRange byte_range;
//   byte_range.inclusive_min = 10;
//   byte_range.exclusive_max = 20;
//   kvs_read_options.byte_range = byte_range;

//   auto result =
//       kvstore::Read(store, "testfile.bin", std::move(kvs_read_options))
//           .result()
//           .value()
//           .value;
//   std::cout << "Result size: " << result.size() << std::endl;

//   auto result_flat = result.Flatten();
//   std::vector<uint8_t> decoded(result_flat.size(), 0);
//   for (size_t i = 0; i < result_flat.size(); ++i) {
//     decoded[i] = static_cast<uint8_t>(result_flat[i]);
//   }

//   std::cout << "Decoded data:" << std::endl;
//   for (auto c : decoded) std::cout << +c << " ";
//   std::cout << std::endl;

//   return 0;
// }

using namespace std::chrono;

int main(int argc, char** argv) {
  auto resource_spec = tensorstore::Context::FromJson(
                           {{"cache_pool", {{"total_bytes_limit", 100000000}}},
                            {"data_copy_concurrency", {{"limit", 1}}}})
                           .value();
  tensorstore::DimensionIndex dim = 0;
  tensorstore::ChunkLayout chunk_layout;
  chunk_layout.Set(tensorstore::ChunkLayout::ReadChunkShape({6, 6}));

  auto store =
      tensorstore::VirtualChunked<tensorstore::Index>(
          tensorstore::NonSerializable{
              [dim](tensorstore::OffsetArrayView<tensorstore::Index> output,
                    tensorstore::virtual_chunked::ReadParameters read_params) {
                std::cout << "Data access read triggered." << std::endl;
                std::cout << "Request domain: " << output.domain() << std::endl;
                tensorstore::IterateOverIndexRange(
                    output.domain(),
                    [&](tensorstore::span<const tensorstore::Index> indices) {
                      output(indices) = indices[dim];
                    });
                return tensorstore::TimestampedStorageGeneration{
                    tensorstore::StorageGeneration::FromString(""),
                    absl::InfiniteFuture()};
              }},
          tensorstore::Schema::Shape({10, 10}), chunk_layout, resource_spec)
          .value();
  std::cout << "Store: " << store.schema().value() << std::endl;
  std::cout << "Rank type: " << store.rank() << std::endl;
  std::cout << "dtype: " << store.dtype() << std::endl;
  std::cout << "domain: " << store.domain() << std::endl;
  std::cout << "chunk layout: " << store.chunk_layout().value() << std::endl;

  // Slice data.
  tensorstore::IndexTransform<> transform =
      tensorstore::IdentityTransform(store.domain());

  transform =
      (std::move(transform) | tensorstore::Dims(0).HalfOpenInterval(0, 3) |
       tensorstore::Dims(1).HalfOpenInterval(0, 3))
          .value();

  auto constrained_store = store | transform;
  std::cout << "First read" << std::endl;

  auto start = high_resolution_clock::now();
  auto data = tensorstore::Read(store).result().value();
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);

  std::cout << "total duration: " << duration.count() << std::endl;
  PrintCSVArray(data);

  std::cout << "Second read" << std::endl;
  start = high_resolution_clock::now();
  data = tensorstore::Read(constrained_store).result().value();
  stop = high_resolution_clock::now();
  duration = duration_cast<milliseconds>(stop - start);
  std::cout << "total duration: " << duration.count() << std::endl;
  PrintCSVArray(data);
}