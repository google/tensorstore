// Copyright 2020 The TensorStore Authors
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

#include <errno.h>
#include <stddef.h>

#include "absl/time/clock.h"
#include "absl/time/time.h"

#ifdef _WIN32
#include <sys/utime.h>
#include <time.h>
#define utimbuf _utimbuf
#define utime _utime
#else
#include <utime.h>
#endif

#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/notification.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/os/filesystem.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status_testutil.h"

// Include system headers last to reduce impact of macros.
#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

namespace kvstore = tensorstore::kvstore;
using ::tensorstore::CompletionNotifyingReceiver;
using ::tensorstore::IsOkAndHolds;
using ::tensorstore::KeyRange;
using ::tensorstore::KvStore;
using ::tensorstore::MatchesStatus;
using ::tensorstore::StorageGeneration;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore::internal::MatchesListEntry;
using ::tensorstore::internal::MatchesTimestampedStorageGeneration;
using ::tensorstore::internal_os::GetDirectoryContents;
using ::tensorstore::internal_testing::ScopedCurrentWorkingDirectory;
using ::tensorstore::internal_testing::ScopedTemporaryDirectory;
using ::testing::HasSubstr;

KvStore GetStore(std::string root) {
  return kvstore::Open({{"driver", "file"}, {"path", root + "/"}}).value();
}

TEST(FileKeyValueStoreTest, Basic) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST(FileKeyValueStoreTest, BasiclockfileLocking) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = kvstore::Open({
                                 {"driver", "file"},
                                 {"path", root + "/"},
                                 {"file_io_locking", {{"mode", "lockfile"}}},
                             })
                   .value();
  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST(FileKeyValueStoreTest, BasicNoLocking) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = kvstore::Open({
                                 {"driver", "file"},
                                 {"path", root + "/"},
                                 {"file_io_locking", {{"mode", "none"}}},
                             })
                   .value();
  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST(FileKeyValueStoreTest, BasicNoSync) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = kvstore::Open({
                                 {"driver", "file"},
                                 {"path", root + "/"},
                                 {"file_io_sync", false},
                             })
                   .value();
  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST(FileKeyValueStoreTest, BasicMemmap) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = kvstore::Open({
                                 {"driver", "file"},
                                 {"path", root + "/"},
                                 {"file_io_memmap", true},
                             })
                   .value();
  tensorstore::internal::TestKeyValueReadWriteOps(store, 256 * 1024);
}

TEST(FileKeyValueStoreTest, InvalidKey) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);

  EXPECT_THAT(kvstore::Read(store, "this_is_a_long_key").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(
      kvstore::Read(store, "").result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      kvstore::Read(store, std::string("\0", 1)).result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      kvstore::Write(store, "", {}).result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      kvstore::Read(store, "/").result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      kvstore::Read(store, ".").result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      kvstore::Read(store, "..").result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      kvstore::Read(store, "a/./b").result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      kvstore::Read(store, "a/../b").result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      kvstore::Read(store, "a/").result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      kvstore::Read(store, "a.__lock").result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      kvstore::Read(store, "a/b.__lock/c").result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      kvstore::Read(store, "///").result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
}

TEST(FileKeyValueStoreTest, LockFiles) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "a/foo", absl::Cord("xyz"),
                     {/*.if_equal=*/StorageGeneration::NoValue()})
          .result());
  EXPECT_THAT(GetDirectoryContents(root),
              ::testing::UnorderedElementsAre("a", "a/foo"));
  EXPECT_THAT(
      kvstore::Write(store, "a/foo", absl::Cord("qqq"),
                     {/*.if_equal=*/StorageGeneration::NoValue()})
          .result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  // Test that no lock files are left around.
  EXPECT_THAT(GetDirectoryContents(root),
              ::testing::UnorderedElementsAre("a", "a/foo"));

  // Create a lock file to simulate a stale lock file left by a process that
  // crashes in the middle of a Write/Delete operation.
  {
    std::string lock_path = root + "/a/foo.__lock";
    std::ofstream x(lock_path);

    // Set the mtime to be stale.
    auto now = absl::Now();
    utimbuf times;
    times.actime = absl::ToTimeT(now - absl::Minutes(60));
    times.modtime = absl::ToTimeT(now - absl::Minutes(60));
    utime(lock_path.c_str(), &times);
  }
  EXPECT_THAT(GetDirectoryContents(root),
              ::testing::UnorderedElementsAre("a", "a/foo", "a/foo.__lock"));

  // Test that the lock file is not included in the `List` result.
  EXPECT_THAT(
      ListFuture(store).result(),
      IsOkAndHolds(::testing::UnorderedElementsAre(MatchesListEntry("a/foo"))));

  // Test that a stale lock file does not interfere with writing.
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "a/foo", absl::Cord("xyz")).result());

  // Recreate the lock file.
  {
    std::string lock_path = root + "/a/foo.__lock";
    std::ofstream x(lock_path);

    // Set the mtime to be stale.
    auto now = absl::Now();
    utimbuf times;
    times.actime = absl::ToTimeT(now - absl::Minutes(60));
    times.modtime = absl::ToTimeT(now - absl::Minutes(60));
    utime(lock_path.c_str(), &times);
  }

  // Test that the "a" prefix can be deleted despite the presence of the lock
  // file.  Only a single key, "a/foo" is removed.  The lock file should not be
  // included in the count.
  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange::Prefix("a/")));
  EXPECT_THAT(GetDirectoryContents(root), ::testing::UnorderedElementsAre("a"));
}

TEST(FileKeyValueStoreTest, NestedDirectories) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/foo", absl::Cord("xyz")));

  TENSORSTORE_EXPECT_OK(
      kvstore::Write(store, "a/ba/ccc/dddd", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(
      kvstore::Write(store, "a/ba/ccc/foo", absl::Cord("xyz")));
  EXPECT_THAT(
      kvstore::Write(store, "a/ba/ccc", absl::Cord("xyz")).result(),
      ::testing::AnyOf(MatchesStatus(absl::StatusCode::kPermissionDenied),
                       MatchesStatus(absl::StatusCode::kFailedPrecondition)));
}

TEST(FileKeyValueStoreTest, ConcurrentWrites) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  tensorstore::internal::TestConcurrentWritesOptions options;
  options.get_store = [&] { return GetStore(root); };
  tensorstore::internal::TestConcurrentWrites(options);
}

TEST(FileKeyValueStoreTest, ConcurrentWritesNoLocks) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  tensorstore::internal::TestConcurrentWritesOptions options;
  options.get_store = [&] {
    return kvstore::Open({
                             {"driver", "file"},
                             {"path", root + "/"},
                             {"file_io_locking", {{"mode", "lockfile"}}},
                         })
        .value();
  };
  tensorstore::internal::TestConcurrentWrites(options);
}

// Tests `FileKeyValueStore` on a directory without write or read/write
// permissions.
#ifndef _WIN32
TEST(FileKeyValueStoreTest, Permissions) {
  // This test fails if our effective user id is root.
  if (::geteuid() == 0) {
    return;
  }

  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "foo", absl::Cord("xyz")).result());

  // Remove write permission on directory.
  ASSERT_EQ(0, ::chmod(root.c_str(), 0500))
      << "Error " << errno << ": " << ::strerror(errno);

  // Ensure we restore write permission so that ScopedTemporaryDirectory can
  // clean up properly.
  struct RestoreWritePermission {
    std::string path;
    ~RestoreWritePermission() {
      EXPECT_EQ(0, ::chmod(path.c_str(), 0700))
          << "Error " << errno << ": " << ::strerror(errno);
    }
  };
  RestoreWritePermission restore{root};

  // Read should still succeed.
  EXPECT_EQ("xyz", kvstore::Read(store, "foo").value().value);

  // Writing an existing key should fail.
  EXPECT_THAT(kvstore::Write(store, "foo", absl::Cord("abc")).result(),
              MatchesStatus(absl::StatusCode::kPermissionDenied));

  // Value should not have changed.
  EXPECT_EQ("xyz", kvstore::Read(store, "foo").value().value);

  // Writing a new key should fail.
  EXPECT_THAT(kvstore::Write(store, "bar", absl::Cord("abc")).result(),
              MatchesStatus(absl::StatusCode::kPermissionDenied));

  // Value should not exist.
  EXPECT_THAT(kvstore::Read(store, "bar").result(),
              MatchesKvsReadResultNotFound());

  // Delete should fail.
  EXPECT_THAT(kvstore::Delete(store, "foo").result(),
              MatchesStatus(absl::StatusCode::kPermissionDenied));

  // Remove read permission on file.
  ASSERT_EQ(0, ::chmod((root + "/foo").c_str(), 0))
      << "Error " << errno << ": " << ::strerror(errno);

  // Read should fail.
  EXPECT_THAT(kvstore::Read(store, "foo").result(),
              MatchesStatus(absl::StatusCode::kPermissionDenied));
}
#endif

TEST(FileKeyValueStoreTest, DeletePrefix) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  tensorstore::internal::TestKeyValueStoreDeletePrefix(store);
}

TEST(FileKeyValueStoreTest, DeleteRange) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  tensorstore::internal::TestKeyValueStoreDeleteRange(store);
}

TEST(FileKeyValueStoreTest, DeleteRangeToEnd) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  tensorstore::internal::TestKeyValueStoreDeleteRangeToEnd(store);
}

TEST(FileKeyValueStoreTest, DeleteRangeFromBeginning) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  tensorstore::internal::TestKeyValueStoreDeleteRangeFromBeginning(store);
}

#if 0
TEST(FileKeyValueStoreTest, CopyRange) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  tensorstore::internal::TestKeyValueStoreCopyRange(store);
}
#endif

TEST(FileKeyValueStoreTest, ListErrors) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);

  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {KeyRange::Prefix("a//")}),
        CompletionNotifyingReceiver{&notification,
                                    tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();
    EXPECT_THAT(log,
                ::testing::ElementsAre(
                    "set_starting",
                    HasSubstr("set_error: INVALID_ARGUMENT: Invalid key: "),
                    "set_stopping"));
  }
}

TEST(FileKeyValueStoreTest, List) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  tensorstore::internal::TestKeyValueStoreList(store, /*match_size=*/false);
}

TEST(FileKeyValueStoreTest, SpecRoundtrip) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.full_spec = {{"driver", "file"}, {"path", root}};
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(FileKeyValueStoreTest, SpecRoundtripSync) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.full_spec = {
      {"driver", "file"},
      {"path", root},
      {"file_io_sync", false},
      {"context",
       {
           {"file_io_concurrency", ::nlohmann::json::object_t()},
           {"file_io_memmap", false},
           {"file_io_locking", {{"mode", "lockfile"}}},
       }},
  };
  options.spec_request_options.Set(tensorstore::retain_context).IgnoreError();
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(FileKeyValueStoreTest, InvalidSpec) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto context = tensorstore::Context::Default();

  // Test with extra key.
  EXPECT_THAT(
      kvstore::Open({{"driver", "file"}, {"path", root}, {"extra", "key"}},
                    context)
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test with invalid `"path"` key.
  EXPECT_THAT(
      kvstore::Open({{"driver", "file"}, {"path", 5}}, context).result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test with invalid `"path"`
  EXPECT_THAT(kvstore::Open({{"driver", "file"}, {"path", "/a/../b/"}}, context)
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*Invalid file path.*"));
}

TEST(FileKeyValueStoreTest, UrlRoundtrip) {
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip({{"driver", "file"}},
                                                       "file://");
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "file"}, {"path", "/abc/"}}, "file:///abc/");
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "file"}, {"path", "/abc def/"}}, "file:///abc%20def/");
}

TEST(FileKeyValueStoreTest, UrlOpen) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   kvstore::Open("file://" + root).result());

  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST(FileKeyValueStoreTest, InvalidUri) {
  EXPECT_THAT(kvstore::Spec::FromUrl("file:///"), tensorstore::IsOk());
  // Currently valid, should it be?
  EXPECT_THAT(kvstore::Spec::FromUrl("file://"), tensorstore::IsOk());

  EXPECT_THAT(kvstore::Spec::FromUrl("file://abc?query"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: Query string not supported"));
  EXPECT_THAT(kvstore::Spec::FromUrl("file://abc#fragment"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: Fragment identifier not supported"));
  EXPECT_THAT(kvstore::Spec::FromUrl("file://abc/../b/"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*Invalid file path.*"));
}

TEST(FileKeyValueStoreTest, RelativePath) {
  ScopedTemporaryDirectory tempdir;
  ScopedCurrentWorkingDirectory scoped_cwd(tempdir.path());
  auto store = GetStore("tmp/dataset");
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "abc", {}).result());
}

TEST(FileKeyValueStoreTest, BatchRead) {
  ScopedTemporaryDirectory tempdir;
  auto store = GetStore(tempdir.path());

  tensorstore::internal::BatchReadGenericCoalescingTestOptions options;
  options.coalescing_options.max_extra_read_bytes = 255;
  options.metric_prefix = "/tensorstore/kvstore/file/";
  options.has_file_open_metric = true;
  tensorstore::internal::TestBatchReadGenericCoalescing(store, options);
}

#if 0
// TODO: Make this test reasonable for mmap cases.
TEST(FileKeyValueStoreTest, BatchReadMemmap) {
  ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = kvstore::Open({
                                 {"driver", "file"},
                                 {"path", root + "/"},
                                 {"file_io_memmap", true},
                             })
                   .value();

  tensorstore::internal::BatchReadGenericCoalescingTestOptions options;
  options.coalescing_options.max_extra_read_bytes = 255;
  options.metric_prefix = "/tensorstore/file/";
  options.has_file_open_metric = true;
  tensorstore::internal::TestBatchReadGenericCoalescing(store, options);
}
#endif

}  // namespace
