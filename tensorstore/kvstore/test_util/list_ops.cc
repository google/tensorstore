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
#include "tensorstore/kvstore/test_util/list_ops.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/functional/function_ref.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "riegeli/base/byte_fill.h"
#include "tensorstore/kvstore/driver.h"  // IWYU pragma: keep
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util/internal.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

void TestKeyValueStoreTransactionalListOps(
    const TransactionalListOpsParameters& p) {
  Cleanup cleanup(p.store, {std::begin(p.keys), std::end(p.keys)});

  using Reference = std::map<std::string, int64_t>;
  Reference reference;

  auto do_write = [&](const KvStore& store, size_t key_idx,
                      bool retain_size) -> absl::Status {
    const auto& key = p.keys[key_idx];
    TENSORSTORE_RETURN_IF_ERROR(
        kvstore::Write(store, key,
                       absl::Cord(riegeli::ByteFill(key_idx + 1, 'X')))
            .status());
    reference[key] =
        retain_size && p.match_size ? static_cast<int64_t>(key_idx + 1) : -1;
    return absl::OkStatus();
  };

  auto do_delete_range = [&](const KvStore& store,
                             const KeyRange& range) -> absl::Status {
    TENSORSTORE_RETURN_IF_ERROR(kvstore::DeleteRange(store, range).status());
    for (Reference::iterator it = reference.lower_bound(range.inclusive_min),
                             next;
         it != reference.end() && Contains(range, it->first); it = next) {
      next = std::next(it);
      reference.erase(it);
    }
    return absl::OkStatus();
  };

  auto get_reference_list =
      [&](const KeyRange& range,
          size_t strip_prefix_length) -> std::vector<kvstore::ListEntry> {
    std::vector<kvstore::ListEntry> vec;
    for (const auto& p : reference) {
      if (!Contains(range, p.first)) continue;
      std::string key = p.first;
      key = key.substr(std::min(key.size(), strip_prefix_length));
      kvstore::ListEntry entry;
      entry.key = std::move(key);
      entry.size = p.second;
      vec.push_back(std::move(entry));
    }
    return vec;
  };

  if (p.write_outside_transaction) {
    TENSORSTORE_ASSERT_OK(do_write(p.store, 0, /*retain_size=*/true));
    TENSORSTORE_ASSERT_OK(do_write(p.store, 2, /*retain_size=*/true));
    TENSORSTORE_ASSERT_OK(do_write(p.store, 4, /*retain_size=*/true));
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto txn_store, p.store | tensorstore::Transaction(p.transaction_mode));

  auto get_list = [&](const KeyRange& range, size_t strip_prefix_length) {
    kvstore::ListOptions options;
    options.range = range;
    options.strip_prefix_length = strip_prefix_length;
    return kvstore::ListFuture(txn_store, std::move(options)).result();
  };

  auto verify = [&](const KeyRange& range, size_t strip_prefix_length) {
    SCOPED_TRACE(tensorstore::StrCat(
        "range=", range, ", strip_prefix_length=", strip_prefix_length));
    EXPECT_THAT(get_list(range, strip_prefix_length),
                ::testing::Optional(::testing::UnorderedElementsAreArray(
                    get_reference_list(range, strip_prefix_length))));
  };

  auto verify_for_multiple_prefix_lengths = [&](const KeyRange& range) {
    for (size_t strip_prefix_length : {0 /*, 1*/}) {
      verify(range, strip_prefix_length);
    }
  };

  auto verify_for_multiple_ranges = [&] {
    verify_for_multiple_prefix_lengths({});
    verify_for_multiple_prefix_lengths(KeyRange({}, p.keys[4]));
    verify_for_multiple_prefix_lengths(KeyRange(p.keys[2], {}));
    verify_for_multiple_prefix_lengths(KeyRange(p.keys[2], p.keys[4]));
  };

  {
    SCOPED_TRACE("Before writing within transaction");
    verify_for_multiple_ranges();
  }

  if (p.get_other_store) {
    p.get_other_store([&](KvStore other) {
      other.transaction = txn_store.transaction;
      TENSORSTORE_ASSERT_OK(do_write(other, 0, /*retain_size=*/true));
      TENSORSTORE_ASSERT_OK(do_write(other, 1, /*retain_size=*/true));
      TENSORSTORE_ASSERT_OK(do_write(other, 3, /*retain_size=*/true));
    });
    {
      SCOPED_TRACE("After writing to other node");
      verify_for_multiple_ranges();
    }
  }

  TENSORSTORE_ASSERT_OK(do_write(txn_store, 0, /*retain_size=*/false));
  TENSORSTORE_ASSERT_OK(do_write(txn_store, 1, /*retain_size=*/false));
  TENSORSTORE_ASSERT_OK(do_write(txn_store, 2, /*retain_size=*/false));

  {
    SCOPED_TRACE("After writing within transaction");
    verify_for_multiple_ranges();
  }

  TENSORSTORE_ASSERT_OK(
      do_delete_range(txn_store, KeyRange{p.keys[2], p.keys[4]}));
  TENSORSTORE_ASSERT_OK(do_write(txn_store, 3, /*retain_size=*/false));

  {
    SCOPED_TRACE("After delete range and write");
    verify_for_multiple_ranges();
  }

  TENSORSTORE_ASSERT_OK(do_delete_range(txn_store, KeyRange{p.keys[3], {}}));

  {
    SCOPED_TRACE("After delete range to end");
    verify_for_multiple_ranges();
  }
}

// Tests List on `store`, which should be empty.
void TestKeyValueStoreList(const KvStore& store, bool match_size) {
  ABSL_LOG(INFO) << "Test list, empty";
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        CompletionNotifyingReceiver{&notification,
                                    tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();
    EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_done",
                                            "set_stopping"));
  }

  const absl::Cord value("xyz");

  ABSL_LOG(INFO) << "Test list: ... write elements ...";
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/b", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/d", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/x", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/y", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/e", value));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/f", value));

  ::testing::Matcher<int64_t> size_matcher = ::testing::_;
  if (match_size) {
    size_matcher = ::testing::Eq(value.size());
  }

  // Listing the entire stream works.
  ABSL_LOG(INFO) << "Test list, entire stream";
  {
    EXPECT_THAT(tensorstore::kvstore::ListFuture(store).value(),
                ::testing::UnorderedElementsAre(
                    MatchesListEntry("a/d", size_matcher),
                    MatchesListEntry("a/c/z/f", size_matcher),
                    MatchesListEntry("a/c/y", size_matcher),
                    MatchesListEntry("a/c/z/e", size_matcher),
                    MatchesListEntry("a/c/x", size_matcher),
                    MatchesListEntry("a/b", size_matcher)));
  }

  // Listing a subset of the stream works.
  ABSL_LOG(INFO) << "Test list, prefix range";
  {
    EXPECT_THAT(
        tensorstore::kvstore::ListFuture(store, {KeyRange::Prefix("a/c/")})
            .value(),
        ::testing::UnorderedElementsAre(
            MatchesListEntry("a/c/z/f", size_matcher),
            MatchesListEntry("a/c/y", size_matcher),
            MatchesListEntry("a/c/z/e", size_matcher),
            MatchesListEntry("a/c/x", size_matcher)));
  }

  // Cancellation immediately after starting yields nothing.
  ABSL_LOG(INFO) << "Test list, cancel on start";
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        CompletionNotifyingReceiver{&notification,
                                    CancelOnStartingReceiver{{&log}}});
    notification.WaitForNotification();

    ASSERT_THAT(log, ::testing::SizeIs(::testing::Ge(3)));

    EXPECT_EQ("set_starting", log[0]);
    EXPECT_EQ("set_done", log[log.size() - 2]);
    EXPECT_EQ("set_stopping", log[log.size() - 1]);
    EXPECT_THAT(span<const std::string>(&log[1], log.size() - 3),
                ::testing::IsSubsetOf({"set_value: a/d", "set_value: a/c/z/f",
                                       "set_value: a/c/y", "set_value: a/c/z/e",
                                       "set_value: a/c/x", "set_value: a/b"}));
  }

  // Cancellation in the middle of the stream may stop the stream.
  ABSL_LOG(INFO) << "Test list, cancel after 2";
  {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        CompletionNotifyingReceiver{&notification,
                                    CancelAfterNReceiver<2>{{&log}}});
    notification.WaitForNotification();

    ASSERT_THAT(log, ::testing::SizeIs(::testing::Gt(3)));

    EXPECT_EQ("set_starting", log[0]);
    EXPECT_EQ("set_done", log[log.size() - 2]);
    EXPECT_EQ("set_stopping", log[log.size() - 1]);
    EXPECT_THAT(span<const std::string>(&log[1], log.size() - 3),
                ::testing::IsSubsetOf({"set_value: a/d", "set_value: a/c/z/f",
                                       "set_value: a/c/y", "set_value: a/c/z/e",
                                       "set_value: a/c/x", "set_value: a/b"}));
  }

  ABSL_LOG(INFO) << "Test list: ... delete elements ...";
  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "a/b"));
  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "a/d"));
  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "a/c/x"));
  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "a/c/y"));
  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "a/c/z/e"));
  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "a/c/z/f"));
}

}  // namespace internal
}  // namespace tensorstore