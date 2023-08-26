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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/open.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = tensorstore::kvstore;

using ::tensorstore::MatchesStatus;

std::string GetPath() {
  return "/Users/hsidky/Working/tensorstore_development/testfile.bin";
}
::nlohmann::json GetKvstoreSpec() { return {{"driver", "file"}}; }

::nlohmann::json GetSpec() {
  return ::nlohmann::json{
      {"driver", "ometiff"},
      {"dtype", "uint8"},
      {"rank", 2},
      {"schema", {{"domain", {{"shape", {5, 5}}}}}},
      {"kvstore", {{"driver", "file"}, {"path", GetPath()}}},
      {"cache_pool", {{"total_bytes_limit", 100000000}}},
      {"data_copy_concurrency", {{"limit", 2}}}};
}

TEST(OMETiffDriverTest, Basic) {
  auto context = tensorstore::Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto kvs, kvstore::Open(GetKvstoreSpec(), context).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(GetSpec(), context).result());

  std::cout << "Rank type: " << store.rank() << std::endl;
  std::cout << "dtype: " << store.dtype() << std::endl;
  std::cout << "domain: " << store.domain() << std::endl;
  std::cout << "chunk layout: " << store.chunk_layout().value() << std::endl;
  std::cout << "\n\n\n" << std::endl;
  tensorstore::Read(store).result();

  // EXPECT_THAT(tensorstore::Read(store).result(),
  //             MatchesStatus(absl::StatusCode::kNotFound, ""));
}

}  // namespace