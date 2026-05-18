// Copyright 2026 The TensorStore Authors
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

#if defined(TENSORSTORE_INTERNAL_TEST_HOOKS)

#include <optional>
#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/internal/os/file_descriptor.h"
#include "tensorstore/internal/os/file_test_hooks.h"
#include "tensorstore/internal/os/open_flags.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/internal/testing/test_hook.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = tensorstore::kvstore;
using ::tensorstore::StatusIs;
using ::tensorstore::internal_os::CloseOpTag;
using ::tensorstore::internal_os::DeleteOpTag;
using ::tensorstore::internal_os::FileDescriptor;
using ::tensorstore::internal_os::FsyncOpTag;
using ::tensorstore::internal_os::OpenFlags;
using ::tensorstore::internal_os::OpenOpTag;
using ::tensorstore::internal_os::ReadOpTag;
using ::tensorstore::internal_os::RenameOpTag;
using ::tensorstore::internal_os::WriteOpTag;
using ::tensorstore::internal_testing::ScopedTemporaryDirectory;
using ::tensorstore::internal_testing::ScopedTestHook;

class FileHookTest : public ::testing::TestWithParam<const char*> {
 protected:
  tensorstore::KvStore OpenStore(const std::string& root) {
    return kvstore::Open({{"driver", "file"},
                          {"path", root + "/"},
                          {"file_io_locking", {{"mode", GetParam()}}}})
        .value();
  }
};

TEST_P(FileHookTest, PutFailsOnClose) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";

  auto store = OpenStore(root);

  bool hook_called = false;
  {
    ScopedTestHook<CloseOpTag> scoped_hook(
        [&](FileDescriptor fd) -> std::optional<absl::Status> {
          hook_called = true;
          return absl::DataLossError("Injected close failure");
        });

    auto result = kvstore::Write(store, "foo", absl::Cord("abc")).result();

    EXPECT_THAT(result, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }

  // After hook is removed, writes should succeed again.
  TENSORSTORE_EXPECT_OK(
      kvstore::Write(store, "bar", absl::Cord("xyz")).result());
}

TEST_P(FileHookTest, PutFailsOnWrite) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";

  auto store = OpenStore(root);

  bool hook_called = false;
  {
    ScopedTestHook<WriteOpTag> scoped_hook(
        [&](FileDescriptor fd) -> std::optional<absl::Status> {
          hook_called = true;
          return absl::DataLossError("Injected write failure");
        });

    auto result = kvstore::Write(store, "foo", absl::Cord("abc")).result();
    EXPECT_THAT(result, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }
}

TEST_P(FileHookTest, ReadFailsOnOpen) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";

  auto store = OpenStore(root);

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "foo", absl::Cord("abc")).result());

  bool hook_called = false;
  {
    ScopedTestHook<OpenOpTag> scoped_hook(
        [&](const std::string& path,
            OpenFlags flags) -> std::optional<absl::Status> {
          hook_called = true;
          return absl::DataLossError("Injected open failure");
        });

    auto result = kvstore::Read(store, "foo").result();
    EXPECT_THAT(result, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }
}

TEST_P(FileHookTest, ReadFailsOnRead) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";

  auto store = OpenStore(root);

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "foo", absl::Cord("abc")).result());

  bool hook_called = false;
  {
    ScopedTestHook<ReadOpTag> scoped_hook(
        [&](FileDescriptor fd) -> std::optional<absl::Status> {
          hook_called = true;
          return absl::DataLossError("Injected read failure");
        });

    auto result = kvstore::Read(store, "foo").result();
    EXPECT_THAT(result, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }
}

TEST_P(FileHookTest, PutFailsOnFsync) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";

  auto store = OpenStore(root);

  bool hook_called = false;
  {
    ScopedTestHook<FsyncOpTag> scoped_hook(
        [&](FileDescriptor fd) -> std::optional<absl::Status> {
          hook_called = true;
          return absl::DataLossError("Injected fsync failure");
        });

    auto result = kvstore::Write(store, "foo", absl::Cord("abc")).result();
    EXPECT_THAT(result, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }
}

TEST_P(FileHookTest, PutFailsOnRename) {
  if (std::string_view(GetParam()) == "non_atomic") {
    GTEST_SKIP() << "Skipping test for non-atomic mode";
  }

  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";

  auto store = OpenStore(root);

  bool hook_called = false;
  {
    ScopedTestHook<RenameOpTag> scoped_hook(
        [&](FileDescriptor fd, const std::string& old_name,
            const std::string& new_name) -> std::optional<absl::Status> {
          hook_called = true;
          return absl::DataLossError("Injected rename failure");
        });

    auto result = kvstore::Write(store, "foo", absl::Cord("abc")).result();
    EXPECT_THAT(result, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }
}

TEST_P(FileHookTest, DeleteFailsOnDelete) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";

  auto store = OpenStore(root);

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "foo", absl::Cord("abc")).result());

  bool hook_called = false;
  {
    ScopedTestHook<DeleteOpTag> scoped_hook(
        [&](FileDescriptor fd,
            const std::string& path) -> std::optional<absl::Status> {
          hook_called = true;
          return absl::DataLossError("Injected delete failure");
        });

    auto result = kvstore::Write(store, "foo", std::nullopt).result();
    EXPECT_THAT(result, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }
}

INSTANTIATE_TEST_SUITE_P(AllModes, FileHookTest,
                         ::testing::Values("os", "lockfile", "none",
                                           "non_atomic"));

}  // namespace

#endif  // TENSORSTORE_INTERNAL_TEST_HOOKS
