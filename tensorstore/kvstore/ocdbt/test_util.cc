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

#include "tensorstore/kvstore/ocdbt/test_util.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/driver.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace tensorstore {
namespace internal_ocdbt {

IoHandle::Ptr GetOcdbtIoHandle(kvstore::Driver& driver) {
  return dynamic_cast<OcdbtDriver&>(driver).io_handle_;
}

Result<std::shared_ptr<const Manifest>> ReadManifest(OcdbtDriver& driver) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto manifest_with_time,
      driver.io_handle_->GetManifest(absl::InfiniteFuture()).result());
  return manifest_with_time.manifest;
}

void TestUnmodifiedNode(const Context& context) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::kvstore::Open(
                      {{"driver", "ocdbt"}, {"base", "memory://"}}, context)
                      .result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "testa", absl::Cord("a")));

  auto& driver = static_cast<OcdbtDriver&>(*store.driver);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto manifest, ReadManifest(driver));
    ASSERT_TRUE(manifest);
    auto& version = manifest->latest_version();
    EXPECT_EQ(2, version.generation_number);
  }

  {
    kvstore::WriteOptions options;
    options.generation_conditions.if_equal = StorageGeneration::NoValue();
    EXPECT_THAT(
        kvstore::Write(store, "testa", absl::Cord("a"), options).result(),
        internal::MatchesTimestampedStorageGeneration(
            StorageGeneration::Unknown()));
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto manifest, ReadManifest(driver));
    ASSERT_TRUE(manifest);
    auto& version = manifest->latest_version();
    EXPECT_EQ(2, version.generation_number);
  }
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
