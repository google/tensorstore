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

#include "tensorstore/internal/os/file_test_hooks.h"

#if defined(TENSORSTORE_INTERNAL_TEST_HOOKS)

#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/internal/os/file_descriptor.h"
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/internal/testing/test_hook.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::StatusIs;
using ::tensorstore::internal_os::CloseFileDescriptor;
using ::tensorstore::internal_os::CloseOpTag;
using ::tensorstore::internal_os::FileDescriptor;
using ::tensorstore::internal_os::FsyncFile;
using ::tensorstore::internal_os::FsyncOpTag;
using ::tensorstore::internal_os::InvalidFileDescriptor;
using ::tensorstore::internal_os::ReadFromFile;
using ::tensorstore::internal_os::ReadOpTag;
using ::tensorstore::internal_os::RenameOpenFile;
using ::tensorstore::internal_os::RenameOpTag;
using ::tensorstore::internal_os::WriteOpTag;
using ::tensorstore::internal_os::WriteToFile;
using ::tensorstore::internal_testing::ScopedTestHook;

TEST(FileDescriptorTest, CloseHookWorks) {
  bool hook_called = false;
  {
    ScopedTestHook<CloseOpTag> scoped_hook(
        [&](FileDescriptor fd) -> std::optional<absl::Status> {
          hook_called = true;
          return absl::DataLossError("Injected close failure");
        });

    auto status = CloseFileDescriptor(InvalidFileDescriptor());
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }

  auto status = CloseFileDescriptor(InvalidFileDescriptor());
  EXPECT_THAT(status, ::testing::Not(StatusIs(absl::StatusCode::kDataLoss)));
}

TEST(FileDescriptorTest, ReadHookWorks) {
  bool hook_called = false;
  {
    ScopedTestHook<ReadOpTag> scoped_hook(
        [&](FileDescriptor fd) -> std::optional<absl::Status> {
          hook_called = true;
          return absl::DataLossError("Injected read failure");
        });

    char buf[1];
    auto result =
        ReadFromFile(InvalidFileDescriptor(), tensorstore::span(buf, 1));
    EXPECT_THAT(result, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }
}

TEST(FileDescriptorTest, WriteHookWorks) {
  bool hook_called = false;
  {
    ScopedTestHook<WriteOpTag> scoped_hook(
        [&](FileDescriptor fd) -> std::optional<absl::Status> {
          hook_called = true;
          return absl::DataLossError("Injected write failure");
        });

    char buf[1] = {0};
    auto result = WriteToFile(InvalidFileDescriptor(), buf, 1);
    EXPECT_THAT(result, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }
}

TEST(FileDescriptorTest, FsyncHookWorks) {
  bool hook_called = false;
  {
    ScopedTestHook<FsyncOpTag> scoped_hook(
        [&](FileDescriptor fd) -> std::optional<absl::Status> {
          hook_called = true;
          return absl::DataLossError("Injected fsync failure");
        });

    auto status = FsyncFile(InvalidFileDescriptor());
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }
}

TEST(FileDescriptorTest, RenameHookWorks) {
  bool hook_called = false;
  {
    ScopedTestHook<RenameOpTag> scoped_hook(
        [&](FileDescriptor fd, const std::string& old_name,
            const std::string& new_name) -> std::optional<absl::Status> {
          hook_called = true;
          return absl::DataLossError("Injected rename failure");
        });

    auto status = RenameOpenFile(InvalidFileDescriptor(), "old", "new");
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }
}

TEST(FileDescriptorTest, DeleteOpenFileHookWorks) {
  bool hook_called = false;
  {
    ::tensorstore::internal_testing::ScopedTestHook<
        ::tensorstore::internal_os::DeleteOpTag>
        scoped_hook(
            [&](FileDescriptor fd,
                const std::string& path) -> std::optional<absl::Status> {
              hook_called = true;
              return absl::DataLossError("Injected delete failure");
            });

    auto status = ::tensorstore::internal_os::DeleteOpenFile(
        InvalidFileDescriptor(), "path");
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }
}

TEST(FileDescriptorTest, DeleteFileHookWorks) {
  bool hook_called = false;
  {
    ::tensorstore::internal_testing::ScopedTestHook<
        ::tensorstore::internal_os::DeleteOpTag>
        scoped_hook(
            [&](FileDescriptor fd,
                const std::string& path) -> std::optional<absl::Status> {
              hook_called = true;
              EXPECT_EQ(fd, InvalidFileDescriptor());
              return absl::DataLossError("Injected delete failure");
            });

    auto status = ::tensorstore::internal_os::DeleteFile("path");
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss));
    EXPECT_TRUE(hook_called);
  }
}

}  // namespace

#endif  // TENSORSTORE_INTERNAL_TEST_HOOKS
