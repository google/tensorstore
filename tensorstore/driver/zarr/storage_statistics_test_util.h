// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_DRIVER_ZARR_STORAGE_STATISTICS_TEST_UTIL_H_
#define TENSORSTORE_DRIVER_ZARR_STORAGE_STATISTICS_TEST_UTIL_H_

// Reusable test suite of `GetStorageStatistics` support for zarr and n5
// drivers.

#include <ostream>

#include <gtest/gtest.h>
#include "tensorstore/context.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/kvstore/mock_kvstore.h"

namespace tensorstore {
namespace internal_zarr {

struct ZarrLikeStorageStatisticsTestParams {
  std::string driver;
  char dimension_separator;
  ::nlohmann::json::object_t extra_metadata = {};

  friend std::ostream& operator<<(
      std::ostream& os, const ZarrLikeStorageStatisticsTestParams& x) {
    return os << x.driver << (x.dimension_separator == '/' ? "Slash" : "Dot");
  }
};

class ZarrLikeStorageStatisticsTest
    : public ::testing::TestWithParam<ZarrLikeStorageStatisticsTestParams> {
 protected:
  Context context = Context::Default();
  // Set up mock kvstore that logs operations and forwards them to
  // `memory_store`.  This is used to check that, in addition to returning the
  // correct statistics, that the correct kvstore operations are also performed,
  // to ensure that the most efficient set of read/list operations are used.
  tensorstore::internal::MockKeyValueStore::MockPtr mock_kvstore =
      *context.GetResource<tensorstore::internal::MockKeyValueStoreResource>()
           .value();
  tensorstore::kvstore::DriverPtr memory_store =
      tensorstore::GetMemoryKeyValueStore();

  std::string driver = GetParam().driver;
  char sep = GetParam().dimension_separator;
  char sep_next = sep + 1;
  ::nlohmann::json json_spec;

 public:
  ZarrLikeStorageStatisticsTest() {
    mock_kvstore->forward_to = memory_store;
    mock_kvstore->log_requests = true;
    json_spec = GetParam().extra_metadata;
    json_spec["driver"] = GetParam().driver;
    json_spec["kvstore"] = {{"driver", "mock_key_value_store"}};
  }
};

}  // namespace internal_zarr
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR_STORAGE_STATISTICS_TEST_UTIL_H_
