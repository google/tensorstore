// Copyright 2022 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Examples

# Sequential reads, 1G tensorstore, in-memory, n5, 256k chunks, 2G total reads

bazel run -c opt \
  //tensorstore/internal/benchmark:ts_benchmark -- \
  --alsologtostderr       \
  --strategy=sequential   \
  --total_read_bytes=-10  \
  --total_write_bytes=-2  \
  --chunk_bytes=2097152   \
  --repeat_reads=16       \
  --repeat_writes=8


# Random reads, 1G tensorstore, in-memory, n5, 256k chunks, 2G total reads

bazel run -c opt \
  //tensorstore/internal/benchmark:ts_benchmark -- \
  --alsologtostderr       \
  --strategy=random       \
  --total_read_bytes=-10  \
  --total_write_bytes=-2  \
  --chunk_bytes=2097152   \
  --repeat_reads=16       \
  --repeat_writes=8


# As above, with context specified:

bazel run -c opt \
  //tensorstore/internal/benchmark:ts_benchmark -- \
  --alsologtostderr       \
  --strategy=random       \
  --total_read_bytes=-10  \
  --total_write_bytes=-2  \
  --chunk_bytes=2097152   \
  --repeat_reads=16       \
  --repeat_writes=8       \
  --context_spec='{"cache_pool": { "total_bytes_limit": 268435456 }}' \
  --tensorstore_spec='{
      "driver": "n5",
      "kvstore": "memory://abc/",
      "metadata": {
           "compression": {"type": "raw"},
           "dataType": "uint8",
           "blockSize": [64, 64, 64, 1],
           "dimensions": [1024, 1024, 1024, 1]
      }
  }'



# Using a file driver and mis-aligned chunks

bazel run -c opt \
  /third_party/tensorstore/internal/benchmark:ts_benchmark -- \
  --alsologtostderr              \
  --strategy=random              \
  --total_read_bytes=-10         \
  --total_write_bytes=-2         \
  --chunk_shape=100,100,64,1     \
  --repeat_reads=16              \
  --repeat_writes=8              \
  --context_spec='{"cache_pool": { "total_bytes_limit": 268435456 }}' \
  --tensorstore_spec='{
      "driver": "n5",
      "kvstore": "file:///tmp/tensorstore_ts_benchmark",
      "metadata": {
           "compression": {"type": "raw"},
           "dataType": "uint8",
           "blockSize": [64, 64, 64, 1],
           "dimensions": [1024, 1024, 1024, 1]
      }
  }'

# Quick size reference:

16KB   --chunk_bytes=16384
512KB  --chunk_bytes=524288
1MB    --chunk_bytes=1048576
2MB    --chunk_bytes=2097152  (default)
4MB    --chunk_bytes=4194304

256MB  --total_read_bytes=268435456
1GB    --total_read_bytes=1073741824  (default)
4GB    --total_read_bytes=4294967296

*/

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/marshalling.h"
#include "absl/log/absl_log.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/driver_testutil.h"
#include "absl/flags/parse.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/benchmark/metric_utils.h"
#include "tensorstore/spec.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace {

template <typename T>
struct VectorFlag {
  VectorFlag() = default;
  VectorFlag(std::vector<T> e) : elements(std::move(e)) {}
  VectorFlag(T x) : elements({std::move(x)}) {}

  std::vector<T> elements;
};

template <typename T>
std::string AbslUnparseFlag(const VectorFlag<T>& list) {
  auto unparse_element = [](std::string* const out, const T element) {
    absl::StrAppend(out, absl::UnparseFlag(element));
  };
  return absl::StrJoin(list.elements, ",", unparse_element);
}

template <typename T>
bool AbslParseFlag(absl::string_view text, VectorFlag<T>* list,
                   std::string* error) {
  list->elements.clear();
  for (const auto& part : absl::StrSplit(text, ',', absl::SkipWhitespace())) {
    T element;
    // Let flag module parse the element type for us.
    if (!absl::ParseFlag(part, &element, error)) {
      return false;
    }
    list->elements.push_back(element);
  }
  return true;
}

tensorstore::Spec DefaultTensorstore() {
  return tensorstore::Spec::FromJson(
             {
                 {"create", true},
                 {"open", true},
                 {"driver", "n5"},
                 {"kvstore", "memory://abc/"},
                 {"metadata",
                  {
                      {"compression", {{"type", "raw"}}},
                      {"dataType", "uint8"},
                      {"blockSize", {64, 64, 64, 1}},
                      {"dimensions", {1024, 1024, 1024, 1}},
                  }},
             })
      .value();
}

}  // namespace

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Spec>, tensorstore_spec,
          DefaultTensorstore(),
          "TensorStore spec for reading/writing data.  See examples at the "
          "start of the source file.");

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Context::Spec>, context_spec,
          {},
          "Context spec for writing data.  This can be used to control the "
          "number of concurrent write operations of the underlying key-value "
          "store.  See examples at the start of the source file.");

ABSL_FLAG(std::string, strategy, "random",
          "Specifies the strategy to use: 'sequential' or 'random'.");

ABSL_FLAG(VectorFlag<tensorstore::Index>, chunk_shape, {},
          "Read/write chunks of --chunk_shape dimensions.");

ABSL_FLAG(size_t, chunk_bytes, 2 * 1024 * 1024,
          "Read/write chunks of --chunk_bytes size (default 2MB).");

ABSL_FLAG(int64_t, total_read_bytes, -1,
          "Number of bytes to read.  Negative values cause the entire "
          "tensorstore to be read that many times.");

ABSL_FLAG(int64_t, total_write_bytes, 0,
          "Number of bytes to write.  Negative values cause the entire "
          "tensorstore to be written that many times.");

ABSL_FLAG(int64_t, repeat_reads, 1,
          "Number of times to repeat read benchmark.");

ABSL_FLAG(int64_t, repeat_writes, 0,
          "Number of times to repeat write benchmark.");

namespace tensorstore {
namespace {

using ::tensorstore::internal::TestDriverWriteReadChunks;
using ::tensorstore::internal::TestDriverWriteReadChunksOptions;

void DoTsBenchmark() {
  using Options = TestDriverWriteReadChunksOptions;
  Options options;

  if (absl::GetFlag(FLAGS_strategy) == "random") {
    options.strategy = Options::kRandom;
  } else if (absl::GetFlag(FLAGS_strategy) == "sequential") {
    options.strategy = Options::kSequential;
  } else {
    ABSL_LOG(FATAL) << "--strategy must be 'sequential' or 'random'";
  }

  if (const auto& flag = absl::GetFlag(FLAGS_chunk_shape).elements;
      !flag.empty()) {
    options.chunk_shape = flag;
  } else if (size_t bytes = absl::GetFlag(FLAGS_chunk_bytes); bytes > 0) {
    options.chunk_bytes = bytes;
  } else {
    ABSL_LOG(FATAL) << "--chunk_shape or --chunk_bytes must be set.";
  }

  options.context_spec = absl::GetFlag(FLAGS_context_spec).value;
  options.tensorstore_spec = absl::GetFlag(FLAGS_tensorstore_spec).value;
  options.repeat_reads = absl::GetFlag(FLAGS_repeat_reads);
  options.repeat_writes = absl::GetFlag(FLAGS_repeat_writes);
  options.total_write_bytes = absl::GetFlag(FLAGS_total_write_bytes);
  options.total_read_bytes = absl::GetFlag(FLAGS_total_read_bytes);

  if (options.total_write_bytes == 0 && options.total_read_bytes == 0) {
    ABSL_LOG(FATAL)
        << "At least one of --total_read_bytes and --total_write_bytes must "
           "be set";
  }

  absl::InsecureBitGen gen;
  TENSORSTORE_CHECK_OK(TestDriverWriteReadChunks(gen, options));

  internal::DumpMetrics("");
}

}  // namespace
}  // namespace tensorstore

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);  // InitTensorstore
  tensorstore::DoTsBenchmark();
  return 0;
}
