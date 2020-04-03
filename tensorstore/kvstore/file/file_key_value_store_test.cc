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
#include <string.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/file_io_concurrency_resource.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/kvstore/key_value_store_testutil.h"
#include "tensorstore/util/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/sender_testutil.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/to_string.h"

// Include this system header last to reduce impact of macros.
#include <sys/stat.h>

namespace {

using tensorstore::CompletionNotifyingReceiver;
using tensorstore::Context;
using tensorstore::KeyValueStore;
using tensorstore::MatchesStatus;
using tensorstore::Status;
using tensorstore::StorageGeneration;
using tensorstore::internal::MatchesKvsReadResultNotFound;
using tensorstore::internal::MatchesTimestampedStorageGeneration;

KeyValueStore::Ptr GetStore(std::string root) {
  return KeyValueStore::Open(Context::Default(),
                             {{"driver", "file"}, {"path", root}})
      .value();
}

TEST(FileKeyValueStoreTest, Basic) {
  tensorstore::internal::ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  tensorstore::internal::TestKeyValueStoreBasicFunctionality(store);
}

TEST(FileKeyValueStoreTest, InvalidKey) {
  tensorstore::internal::ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);

  EXPECT_THAT(store->Read("this_is_a_long_key").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(
      GetStatus(store->Read("").result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      GetStatus(store->Read(std::string("\0", 1)).result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      GetStatus(store->Write("", {}).result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      GetStatus(store->Read("/").result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      GetStatus(store->Read(".").result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      GetStatus(store->Read("..").result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      GetStatus(store->Read("a//b").result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      GetStatus(store->Read("a/./b").result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      GetStatus(store->Read("a/../b").result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      GetStatus(store->Read("a/").result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      GetStatus(store->Read("/a").result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      GetStatus(store->Read("a.__lock").result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      GetStatus(store->Read("a/b.__lock/c").result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
  EXPECT_THAT(
      GetStatus(store->Read("///").result()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid key: .*"));
}

/// Returns the list of relative paths contained within the directory `root`.
std::vector<std::string> GetDirectoryContents(const std::string& root) {
  std::vector<std::string> paths;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(root)) {
    paths.push_back(entry.path().lexically_relative(root).generic_string());
  }
  return paths;
}

TEST(FileKeyValueStoreTest, LockFiles) {
  tensorstore::internal::ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  TENSORSTORE_ASSERT_OK(
      store->Write("a/foo", "xyz", {/*.if_equal=*/StorageGeneration::NoValue()})
          .result());
  EXPECT_THAT(GetDirectoryContents(root),
              ::testing::UnorderedElementsAre("a", "a/foo"));
  EXPECT_THAT(
      store->Write("a/foo", "qqq", {/*.if_equal=*/StorageGeneration::NoValue()})
          .result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  // Test that no lock files are left around.
  EXPECT_THAT(GetDirectoryContents(root),
              ::testing::UnorderedElementsAre("a", "a/foo"));

  // Create a lock file to simulate a stale lock file left by a process that
  // crashes in the middle of a Write/Delete operation.
  std::ofstream(root + "/a/foo.__lock");
  EXPECT_THAT(GetDirectoryContents(root),
              ::testing::UnorderedElementsAre("a", "a/foo", "a/foo.__lock"));

  // Test that the lock file is not included in the `List` result.
  EXPECT_THAT(ListFuture(store.get()).result(),
              ::testing::Optional(::testing::UnorderedElementsAre("a/foo")));

  // Test that a stale lock file does not interfere with writing.
  TENSORSTORE_ASSERT_OK(store->Write("a/foo", "xyz").result());

  // Recreate the lock file.
  std::ofstream(root + "/a/foo.__lock");

  // Test that the "a" prefix can be deleted despite the presence of the lock
  // file.  Only a single key, "a/foo" is removed.  The lock file should not be
  // included in the count.
  EXPECT_THAT(store->DeletePrefix("a/").result(), ::testing::Optional(1));
  EXPECT_THAT(GetDirectoryContents(root), ::testing::UnorderedElementsAre());
}

TEST(FileKeyValueStoreTest, NestedDirectories) {
  tensorstore::internal::ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  EXPECT_EQ(Status(), GetStatus(store->Write("a/foo", "xyz").result()));

  EXPECT_EQ(Status(), GetStatus(store->Write("a/ba/ccc/dddd", "xyz").result()));
  EXPECT_EQ(Status(), GetStatus(store->Write("a/ba/ccc/foo", "xyz").result()));
  EXPECT_THAT(
      GetStatus(store->Write("a/ba/ccc", "xyz").result()),
      ::testing::AnyOf(MatchesStatus(absl::StatusCode::kPermissionDenied),
                       MatchesStatus(absl::StatusCode::kFailedPrecondition)));
}

TEST(FileKeyValueStoreTest, ConcurrentWrites) {
  constexpr std::size_t num_threads = 4;
  std::vector<std::thread> threads;
  tensorstore::internal::ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  std::string key = "test";
  std::string initial_value;
  initial_value.resize(sizeof(std::size_t) * num_threads);
  auto initial_generation =
      store->Write(key, {initial_value}).value().generation;
  constexpr std::size_t num_iterations = 100;
  for (std::size_t thread_i = 0; thread_i < num_threads; ++thread_i) {
    threads.push_back(std::thread([&, thread_i] {
      StorageGeneration generation = initial_generation;
      std::string value = initial_value;
      for (std::size_t i = 0; i < num_iterations; ++i) {
        const std::size_t value_offset = sizeof(std::size_t) * thread_i;
        while (true) {
          std::size_t x;
          std::memcpy(&x, &value[value_offset], sizeof(std::size_t));
          ASSERT_EQ(i, x);
          std::string new_value = value;
          x = i + 1;
          std::memcpy(&new_value[value_offset], &x, sizeof(std::size_t));
          auto write_result =
              store->Write(key, new_value, {generation}).result();
          ASSERT_EQ(Status(), GetStatus(write_result));
          if (!StorageGeneration::IsUnknown(write_result->generation)) {
            generation = write_result->generation;
            value = new_value;
            break;
          }
          auto read_result = store->Read(key).result();
          ASSERT_EQ(Status(), GetStatus(read_result));
          ASSERT_FALSE(read_result->aborted() || read_result->not_found());
          value = *read_result->value;
          ASSERT_EQ(sizeof(std::size_t) * num_threads, value.size());
          generation = read_result->generation.generation;
        }
      }
    }));
  }
  for (auto& t : threads) t.join();
  {
    auto read_result = store->Read(key).result();
    ASSERT_TRUE(read_result);
    std::string expected_value;
    expected_value.resize(sizeof(std::size_t) * num_threads);
    {
      std::vector<std::size_t> expected_nums(num_threads, num_iterations);
      std::memcpy(const_cast<char*>(expected_value.data()),
                  expected_nums.data(), expected_value.size());
    }
    EXPECT_THAT(read_result->value, ::testing::Optional(expected_value));
  }
}

// Tests `FileKeyValueStore` on a directory without write or read/write
// permissions.
#ifndef _WIN32
TEST(FileKeyValueStoreTest, Permissions) {
  tensorstore::internal::ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  EXPECT_EQ(Status(), GetStatus(store->Write("foo", "xyz").result()));

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
  EXPECT_EQ("xyz", store->Read("foo").value().value);

  // Writing an existing key should fail.
  EXPECT_THAT(store->Write("foo", "abc").result(),
              MatchesStatus(absl::StatusCode::kPermissionDenied));

  // Value should not have changed.
  EXPECT_EQ("xyz", store->Read("foo").value().value);

  // Writing a new key should fail.
  EXPECT_THAT(store->Write("bar", "abc").result(),
              MatchesStatus(absl::StatusCode::kPermissionDenied));

  // Value should not exist.
  EXPECT_THAT(store->Read("bar").result(), MatchesKvsReadResultNotFound());

  // Delete should fail.
  EXPECT_THAT(store->Delete("foo").result(),
              MatchesStatus(absl::StatusCode::kPermissionDenied));

  // Remove read permission on file.
  ASSERT_EQ(0, ::chmod((root + "/foo").c_str(), 0))
      << "Error " << errno << ": " << ::strerror(errno);

  // Read should fail.
  EXPECT_THAT(store->Read("foo").result(),
              MatchesStatus(absl::StatusCode::kPermissionDenied));
}
#endif

TEST(FileKeyValueStoreTest, DeletePrefix) {
  tensorstore::internal::ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);
  EXPECT_EQ(Status(), GetStatus(store->Write("a/b", "xyz").result()));
  EXPECT_EQ(Status(), GetStatus(store->Write("a/d", "xyz").result()));
  EXPECT_EQ(Status(), GetStatus(store->Write("a/c/x", "xyz").result()));
  EXPECT_EQ(Status(), GetStatus(store->Write("a/c/y", "xyz").result()));
  EXPECT_EQ(Status(), GetStatus(store->Write("a/c/z/e", "xyz").result()));
  EXPECT_EQ(Status(), GetStatus(store->Write("a/c/z/f", "xyz").result()));

  EXPECT_THAT(store->DeletePrefix("a/c").result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  EXPECT_EQ(4, store->DeletePrefix("a/c/").result());

  EXPECT_EQ("xyz", store->Read("a/b").value().value);
  EXPECT_EQ("xyz", store->Read("a/d").value().value);

  EXPECT_THAT(store->Read("a/c/x").result(), MatchesKvsReadResultNotFound());
  EXPECT_THAT(store->Read("a/c/y").result(), MatchesKvsReadResultNotFound());
  EXPECT_THAT(store->Read("a/c/z/e").result(), MatchesKvsReadResultNotFound());
  EXPECT_THAT(store->Read("a/c/z/f").result(), MatchesKvsReadResultNotFound());
}

TEST(FileKeyValueStoreTest, ListErrors) {
  tensorstore::internal::ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);

  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        store->List({"a"}),
        CompletionNotifyingReceiver{&notification,
                                    tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();
    EXPECT_THAT(log,
                ::testing::ElementsAre("set_error: INVALID_ARGUMENT: "
                                       "Prefix must be empty or end with '/'",
                                       "set_stopping"));
  }
}

TEST(FileKeyValueStoreTest, List) {
  tensorstore::internal::ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto store = GetStore(root);

  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        store->List({""}),
        CompletionNotifyingReceiver{&notification,
                                    tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();
    EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_done",
                                            "set_stopping"));
  }

  EXPECT_EQ(Status(), GetStatus(store->Write("a/b", "xyz").result()));
  EXPECT_EQ(Status(), GetStatus(store->Write("a/d", "xyz").result()));
  EXPECT_EQ(Status(), GetStatus(store->Write("a/c/x", "xyz").result()));
  EXPECT_EQ(Status(), GetStatus(store->Write("a/c/y", "xyz").result()));
  EXPECT_EQ(Status(), GetStatus(store->Write("a/c/z/e", "xyz").result()));
  EXPECT_EQ(Status(), GetStatus(store->Write("a/c/z/f", "xyz").result()));

  // Listing the entire stream works.
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        store->List({""}),
        CompletionNotifyingReceiver{&notification,
                                    tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();

    EXPECT_THAT(
        log, ::testing::UnorderedElementsAre(
                 "set_starting", "set_value: a/d", "set_value: a/c/z/f",
                 "set_value: a/c/y", "set_value: a/c/z/e", "set_value: a/c/x",
                 "set_value: a/b", "set_done", "set_stopping"));
  }

  // Listing a subset of the stream works.
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        store->List({"a/c/"}),
        CompletionNotifyingReceiver{&notification,
                                    tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();

    EXPECT_THAT(log, ::testing::UnorderedElementsAre(
                         "set_starting", "set_value: a/c/z/f",
                         "set_value: a/c/y", "set_value: a/c/z/e",
                         "set_value: a/c/x", "set_done", "set_stopping"));
  }

  // Cancellation immediately after starting yields nothing..
  struct CancelOnStarting : public tensorstore::LoggingReceiver {
    void set_starting(tensorstore::AnyCancelReceiver do_cancel) {
      this->tensorstore::LoggingReceiver::set_starting({});
      do_cancel();
    }
  };

  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        store->List({""}),
        CompletionNotifyingReceiver{&notification, CancelOnStarting{{&log}}});
    notification.WaitForNotification();

    EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_done",
                                            "set_stopping"));
  }

  // Cancellation in the middle of the stream stops the stream.
  struct CancelAfter2 : public tensorstore::LoggingReceiver {
    using Key = tensorstore::KeyValueStore::Key;
    tensorstore::AnyCancelReceiver cancel;

    void set_starting(tensorstore::AnyCancelReceiver do_cancel) {
      this->cancel = std::move(do_cancel);
      this->tensorstore::LoggingReceiver::set_starting({});
    }

    void set_value(Key k) {
      this->tensorstore::LoggingReceiver::set_value(std::move(k));
      if (this->log->size() == 2) {
        this->cancel();
      }
    }
  };

  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        store->List({""}),
        CompletionNotifyingReceiver{&notification, CancelAfter2{{&log}}});
    notification.WaitForNotification();

    EXPECT_THAT(log,
                ::testing::ElementsAre(
                    "set_starting",
                    ::testing::AnyOf("set_value: a/d", "set_value: a/c/z/f",
                                     "set_value: a/c/y", "set_value: a/c/z/e",
                                     "set_value: a/c/x", "set_value: a/b"),
                    "set_done", "set_stopping"));
  }
}

TEST(FileKeyValueStoreTest, SpecRoundtrip) {
  tensorstore::internal::ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(
      {{"driver", "file"}, {"path", root}});
}

TEST(FileKeyValueStoreTest, InvalidSpec) {
  tensorstore::internal::ScopedTemporaryDirectory tempdir;
  std::string root = tempdir.path() + "/root";
  auto context = tensorstore::Context::Default();

  // Test with extra key.
  EXPECT_THAT(
      KeyValueStore::Open(
          context, {{"driver", "file"}, {"path", root}, {"extra", "key"}})
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test with missing `"path"` key.
  EXPECT_THAT(KeyValueStore::Open(context, {{"driver", "file"}}).result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test with invalid `"path"` key.
  EXPECT_THAT(
      KeyValueStore::Open(context, {{"driver", "file"}, {"path", 5}}, {})
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
