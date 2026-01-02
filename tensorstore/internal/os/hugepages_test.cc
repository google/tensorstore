// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/internal/os/hugepages.h"

#include <stddef.h>

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/reflection.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::IsOk;
using ::tensorstore::StatusIs;
using ::tensorstore::internal_os::AllocateHugePageRegion;
using ::tensorstore::internal_os::AllocateHugePageRegionWithFallback;
using ::tensorstore::internal_os::kHugepageSize;
using ::tensorstore::internal_os::UseAllocateHugePageRegion;
using ::testing::AnyOf;
using ::testing::Ge;
using ::testing::IsNull;
using ::testing::NotNull;

namespace {

void EnableHugePageFlag() {
  std::string error;
  absl::FindCommandLineFlag("tensorstore_hugepage_threshold")
      ->ParseFrom("1024", &error);
}

TEST(HugePagesTest, AllocateAlignedRegion) {
  EnableHugePageFlag();

  auto region = AllocateHugePageRegionWithFallback(1024, 16 * 1024 * 1024);
  EXPECT_THAT(region.data(), NotNull());
  EXPECT_THAT(region.size(), Ge(16 * 1024 * 1024));
  EXPECT_THAT(region.as_string_view().size(), Ge(16 * 1024 * 1024));

  absl::Cord a = std::move(region).as_cord();
}

#if defined(__linux__)
// This is a linux-only value.
TEST(HugePagesTest, GetTransparentHugePageMode) {
  EXPECT_THAT(::tensorstore::internal_os::GetTransparentHugePageMode(), IsOk());
}
#endif

TEST(HugePagesTest, IsShmemHugePagesEnabled) {
  auto result = UseAllocateHugePageRegion();
  EXPECT_THAT(result,
              testing::AnyOf(IsOk(),
                             // Windows may not have SeLockMemoryPrivilege
                             StatusIs(absl::StatusCode::kPermissionDenied)));
}

TEST(HugePagesTest, AllocateHugePageRegionSizeZero) {
  EnableHugePageFlag();

  auto use_hugepages = UseAllocateHugePageRegion();
  if (!use_hugepages.ok()) {
    GTEST_SKIP() << "Huge pages not enabled on this system.";
    return;
  }

  auto region = AllocateHugePageRegion(4096, 0);
  EXPECT_THAT(region,
              AnyOf(StatusIs(absl::StatusCode::kUnimplemented),
                    StatusIs(absl::StatusCode::kResourceExhausted), IsOk()));
  if (region.ok()) {
    EXPECT_THAT(region, IsOk());
    EXPECT_THAT(region->data(), IsNull());
    EXPECT_THAT(region->size(), 0);
  }
}

TEST(HugePagesTest, AllocateHugePageRegion) {
  EnableHugePageFlag();

  auto use_hugepages = UseAllocateHugePageRegion();
  if (!use_hugepages.ok()) {
    GTEST_SKIP() << "Huge pages not enabled on this system.";
    return;
  }

  if (!*use_hugepages) {
    EXPECT_THAT(AllocateHugePageRegion(4096, 1024),
                AnyOf(StatusIs(absl::StatusCode::kUnimplemented),
                      StatusIs(absl::StatusCode::kResourceExhausted)));
    GTEST_SKIP() << "Huge pages not enabled on this system.";
    return;
  }

  auto region = AllocateHugePageRegion(kHugepageSize, 1024);
  if (!region.ok()) {
    // Allocation may fail even if huge pages are enabled, e.g. if
    // not enough huge pages are available.
    LOG(WARNING) << "Failed to allocate huge page region: " << region.status();
    EXPECT_THAT(region.status(),
                AnyOf(StatusIs(absl::StatusCode::kResourceExhausted),
                      StatusIs(absl::StatusCode::kUnknown),
                      StatusIs(absl::StatusCode::kInternal)));
    return;
  }
  // Size is rounded up to the next huge page size.
  EXPECT_THAT(region->data(), NotNull());
  EXPECT_THAT(region->size(), Ge(kHugepageSize));

  // Try to write to memory.
  region->data()[0] = 1;
  EXPECT_EQ(region->data()[0], 1);
}

}  // namespace
