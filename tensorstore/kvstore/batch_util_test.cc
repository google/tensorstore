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

#include "tensorstore/kvstore/batch_util.h"

#include <stddef.h>
#include <stdint.h>

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/clock.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::ByteRange;
using ::tensorstore::Future;
using ::tensorstore::OptionalByteRangeRequest;
using ::tensorstore::StatusIs;
using ::tensorstore::internal_kvstore_batch::ByteRangeReadRequest;
using ::tensorstore::internal_kvstore_batch::ForEachCoalescedRequest;
using ::tensorstore::internal_kvstore_batch::
    kDefaultRemoteStorageCoalescingOptions;
using ::tensorstore::internal_kvstore_batch::ResolveCoalescedRequests;

struct R {
  OptionalByteRangeRequest byte_range;
};

TEST(ForEachCoalescedRequestTest, SingleRead) {
  // All these requests are coalesced into a single request because
  // the second item is unbounded.
  std::vector<R> requests = {
      R{{0, 100}},
  };

  ForEachCoalescedRequest(
      tensorstore::span(requests), kDefaultRemoteStorageCoalescingOptions,
      [&](OptionalByteRangeRequest coalesced_byte_range,
          tensorstore::span<R> coalesced_requests) {
        EXPECT_THAT(coalesced_byte_range.inclusive_min, ::testing::Eq(0));
        EXPECT_THAT(coalesced_byte_range.exclusive_max, ::testing::Eq(100));
        EXPECT_THAT(coalesced_requests.size(), ::testing::Eq(requests.size()));
      });
}

TEST(ForEachCoalescedRequestTest, Unmerged) {
  std::vector<R> requests = {
      R{{0, 10}},
      R{{90, 100}},
  };

  ForEachCoalescedRequest(
      tensorstore::span(requests),
      [](ByteRange coalesced_byte_range, int64_t next_inclusive_min) {
        return false;
      },
      [&](OptionalByteRangeRequest coalesced_byte_range,
          tensorstore::span<R> coalesced_requests) {
        if (coalesced_byte_range.inclusive_min == 0) {
          EXPECT_THAT(coalesced_byte_range.exclusive_max, ::testing::Eq(10));

        } else {
          EXPECT_THAT(coalesced_byte_range.inclusive_min, ::testing::Eq(90));
          EXPECT_THAT(coalesced_byte_range.exclusive_max, ::testing::Eq(100));
        }
        EXPECT_THAT(coalesced_requests.size(), ::testing::Eq(1));
      });
}

TEST(ForEachCoalescedRequestTest, Adjacent) {
  // Adjacent requests are coalesced.
  std::vector<R> requests = {
      R{{0, 51}},
      R{{50, 100}},
  };

  ForEachCoalescedRequest(
      tensorstore::span(requests),
      [](ByteRange coalesced_byte_range, int64_t next_inclusive_min) {
        return false;
      },
      [&](OptionalByteRangeRequest coalesced_byte_range,
          tensorstore::span<R> coalesced_requests) {
        EXPECT_THAT(coalesced_byte_range.inclusive_min, ::testing::Eq(0));
        EXPECT_THAT(coalesced_byte_range.exclusive_max, ::testing::Eq(100));
        EXPECT_THAT(coalesced_requests.size(), ::testing::Eq(requests.size()));
      });
}

TEST(ForEachCoalescedRequestTest, TruePredicate) {
  // True predicate requests are coalesced.
  std::vector<R> requests = {
      R{{0, 1}},
      R{{99, 100}},
  };

  ForEachCoalescedRequest(
      tensorstore::span(requests),
      [](ByteRange coalesced_byte_range, int64_t next_inclusive_min) {
        return true;
      },
      [&](OptionalByteRangeRequest coalesced_byte_range,
          tensorstore::span<R> coalesced_requests) {
        EXPECT_THAT(coalesced_byte_range.inclusive_min, ::testing::Eq(0));
        EXPECT_THAT(coalesced_byte_range.exclusive_max, ::testing::Eq(100));
        EXPECT_THAT(coalesced_requests.size(), ::testing::Eq(requests.size()));
      });
}

TEST(ForEachCoalescedRequestTest, Unbounded) {
  // Two adjacent requests, second unbounded, coalesces all following requests.
  std::vector<R> requests = {
      R{{0, 10}},
      R{{10, -1}},
      R{{90, 100}},
  };

  ForEachCoalescedRequest(
      tensorstore::span(requests), kDefaultRemoteStorageCoalescingOptions,
      [&](OptionalByteRangeRequest coalesced_byte_range,
          tensorstore::span<R> coalesced_requests) {
        EXPECT_THAT(coalesced_byte_range.inclusive_min, ::testing::Eq(0));
        EXPECT_THAT(coalesced_byte_range.exclusive_max, ::testing::Eq(-1));
        EXPECT_THAT(coalesced_requests.size(), ::testing::Eq(requests.size()));
      });
}

TEST(ForEachCoalescedRequestTest, SuffixLength) {
  // A suffix length request is not coalesced with range requests.
  std::vector<R> requests = {
      R{{-10, 0}},
      R{{90, 100}},
  };

  ForEachCoalescedRequest(
      tensorstore::span(requests), kDefaultRemoteStorageCoalescingOptions,
      [&](OptionalByteRangeRequest coalesced_byte_range,
          tensorstore::span<R> coalesced_requests) {
        EXPECT_THAT(coalesced_requests.size(), ::testing::Eq(1));
      });
}

TEST(ForEachCoalescedRequestTest, FullRequest) {
  // An unbound request will coalesce range and suffix requests.
  std::vector<R> requests = {
      R{{-10, 0}},
      R{{0, 10}},
      R{{90, 100}},
      R{},
  };

  ForEachCoalescedRequest(
      tensorstore::span(requests), kDefaultRemoteStorageCoalescingOptions,
      [&](OptionalByteRangeRequest coalesced_byte_range,
          tensorstore::span<R> coalesced_requests) {
        EXPECT_THAT(coalesced_requests.size(), ::testing::Eq(requests.size()));
      });
}

TEST(ForEachCoalescedRequestTest, DefaultExtraReadBytes) {
  // Two reads that are coalesced.
  constexpr int64_t kSize =
      kDefaultRemoteStorageCoalescingOptions.max_extra_read_bytes;
  std::vector<R> requests = {
      R{{0, 1}},
      R{{1 + kSize, 2 + kSize}},
  };

  ForEachCoalescedRequest(
      tensorstore::span(requests), kDefaultRemoteStorageCoalescingOptions,
      [&](OptionalByteRangeRequest coalesced_byte_range,
          tensorstore::span<R> coalesced_requests) {
        EXPECT_THAT(coalesced_byte_range.inclusive_min, ::testing::Eq(0));
        EXPECT_THAT(coalesced_byte_range.exclusive_max,
                    ::testing::Eq(2 + kSize));
        EXPECT_THAT(coalesced_requests.size(), ::testing::Eq(requests.size()));
      });
}

TEST(ForEachCoalescedRequestTest, DefaultExtraReadBytes_Gap) {
  // Two reads that are not coalesced due to gap between them.
  constexpr int64_t kSize =
      kDefaultRemoteStorageCoalescingOptions.max_extra_read_bytes;
  std::vector<R> requests = {
      R{{0, 1}},
      R{{2 + kSize, 3 + kSize}},
  };

  ForEachCoalescedRequest(
      tensorstore::span(requests), kDefaultRemoteStorageCoalescingOptions,
      [&](OptionalByteRangeRequest coalesced_byte_range,
          tensorstore::span<R> coalesced_requests) {
        EXPECT_THAT(coalesced_requests.size(), ::testing::Eq(1));
      });
}

TEST(ForEachCoalescedRequestTest, DefaultExtraReadBytes_Max) {
  // Two reads that are not coalesced due to size limit.
  constexpr int64_t kSize =
      kDefaultRemoteStorageCoalescingOptions.target_coalesced_size;
  std::vector<R> requests = {
      R{{0, kSize}},
      R{{kSize, kSize + 1}},
  };

  ForEachCoalescedRequest(
      tensorstore::span(requests), kDefaultRemoteStorageCoalescingOptions,
      [&](OptionalByteRangeRequest coalesced_byte_range,
          tensorstore::span<R> coalesced_requests) {
        EXPECT_THAT(coalesced_requests.size(), ::testing::Eq(1));
      });
}

TEST(ResolveCoalescedRequestsTest, Value) {
  std::vector<ByteRangeReadRequest> requests(5);
  std::vector<Future<tensorstore::kvstore::ReadResult>> futures(
      requests.size());

  for (size_t i = 0; i < requests.size(); ++i) {
    auto pair = tensorstore::PromiseFuturePair<
        tensorstore::kvstore::ReadResult>::Make();
    requests[i].promise = std::move(pair.promise);
    futures[i] = std::move(pair.future);
  }

  requests[0].byte_range = OptionalByteRangeRequest::Stat();
  requests[1].byte_range = OptionalByteRangeRequest::SuffixLength(3);
  requests[2].byte_range = OptionalByteRangeRequest::Suffix(8);
  requests[3].byte_range = OptionalByteRangeRequest::Range(1, 5);
  requests[4].byte_range = OptionalByteRangeRequest{};  // Full

  ResolveCoalescedRequests(
      ByteRange{0, 10}, tensorstore::span(requests),
      tensorstore::kvstore::ReadResult::Value(absl::Cord("0123456789"), {}));

  ASSERT_TRUE(futures[0].ready());
  EXPECT_EQ(futures[0].result().value().value, absl::Cord(""));
  ASSERT_TRUE(futures[1].ready());
  EXPECT_EQ(futures[1].result().value().value, absl::Cord("789"));
  ASSERT_TRUE(futures[2].ready());
  EXPECT_EQ(futures[2].result().value().value, absl::Cord("89"));
  ASSERT_TRUE(futures[3].ready());
  EXPECT_EQ(futures[3].result().value().value, absl::Cord("1234"));
  ASSERT_TRUE(futures[4].ready());
  EXPECT_EQ(futures[4].result().value().value, absl::Cord("0123456789"));
}

TEST(ResolveCoalescedRequestsTest, OutOfBoundsValue) {
  std::vector<ByteRangeReadRequest> requests(5);
  std::vector<Future<tensorstore::kvstore::ReadResult>> futures(
      requests.size());

  for (size_t i = 0; i < requests.size(); ++i) {
    auto pair = tensorstore::PromiseFuturePair<
        tensorstore::kvstore::ReadResult>::Make();
    requests[i].promise = std::move(pair.promise);
    futures[i] = std::move(pair.future);
  }

  requests[0].byte_range = OptionalByteRangeRequest::Stat();
  requests[1].byte_range = OptionalByteRangeRequest::SuffixLength(5);
  requests[2].byte_range = OptionalByteRangeRequest::Suffix(5);
  requests[3].byte_range = OptionalByteRangeRequest::Range(2, 5);
  requests[4].byte_range = OptionalByteRangeRequest{};  // Full

  ResolveCoalescedRequests(
      ByteRange{0, 4}, tensorstore::span(requests),
      tensorstore::kvstore::ReadResult::Value(absl::Cord("0123"), {}));

  ASSERT_TRUE(futures[0].ready());
  EXPECT_EQ(futures[0].result().value().value, absl::Cord(""));
  ASSERT_TRUE(futures[1].ready());
  EXPECT_THAT(futures[1].result().status(),
              StatusIs(absl::StatusCode::kOutOfRange));
  ASSERT_TRUE(futures[2].ready());
  EXPECT_THAT(futures[2].result().status(),
              StatusIs(absl::StatusCode::kOutOfRange));
  ASSERT_TRUE(futures[3].ready());
  EXPECT_THAT(futures[3].result().status(),
              StatusIs(absl::StatusCode::kOutOfRange));
  ASSERT_TRUE(futures[4].ready());
  EXPECT_EQ(futures[4].result().value().value, absl::Cord("0123"));
}

TEST(ResolveCoalescedRequestsTest, ExactRead) {
  std::vector<ByteRangeReadRequest> requests(2);
  std::vector<Future<tensorstore::kvstore::ReadResult>> futures(
      requests.size());

  for (size_t i = 0; i < requests.size(); ++i) {
    auto pair = tensorstore::PromiseFuturePair<
        tensorstore::kvstore::ReadResult>::Make();
    requests[i].promise = std::move(pair.promise);
    futures[i] = std::move(pair.future);
  }
  requests[0].byte_range = OptionalByteRangeRequest::Stat();
  requests[1].byte_range = OptionalByteRangeRequest::Range(10, 20);

  ResolveCoalescedRequests(
      ByteRange{10, 20}, tensorstore::span(requests),
      tensorstore::kvstore::ReadResult::Value(absl::Cord("0123456789"), {}));

  ASSERT_TRUE(futures[0].ready());
  EXPECT_EQ(futures[0].result().value().value, absl::Cord());
  ASSERT_TRUE(futures[1].ready());
  EXPECT_EQ(futures[1].result().value().value, absl::Cord("0123456789"));
}

TEST(ResolveCoalescedRequestsTest, ExactReadZeroSize) {
  std::vector<ByteRangeReadRequest> requests(2);
  std::vector<Future<tensorstore::kvstore::ReadResult>> futures(
      requests.size());

  for (size_t i = 0; i < requests.size(); ++i) {
    auto pair = tensorstore::PromiseFuturePair<
        tensorstore::kvstore::ReadResult>::Make();
    requests[i].promise = std::move(pair.promise);
    futures[i] = std::move(pair.future);
  }
  requests[0].byte_range = OptionalByteRangeRequest::Stat();
  requests[1].byte_range = OptionalByteRangeRequest::Range(20, 20);

  ResolveCoalescedRequests(
      ByteRange{20, 20}, tensorstore::span(requests),
      tensorstore::kvstore::ReadResult::Value(absl::Cord(""), {}));

  for (size_t i = 0; i < requests.size(); ++i) {
    ASSERT_TRUE(futures[i].ready());
    EXPECT_EQ(futures[i].result().value().value, absl::Cord(""));
  }
}

TEST(ResolveCoalescedRequestsTest, NonValue) {
  std::vector<ByteRangeReadRequest> requests(5);
  std::vector<Future<tensorstore::kvstore::ReadResult>> futures(5);

  for (size_t i = 0; i < 5; ++i) {
    auto pair = tensorstore::PromiseFuturePair<
        tensorstore::kvstore::ReadResult>::Make();
    requests[i].promise = std::move(pair.promise);
    futures[i] = std::move(pair.future);
  }

  requests[0].byte_range = OptionalByteRangeRequest::Stat();
  requests[1].byte_range = OptionalByteRangeRequest::SuffixLength(3);
  requests[2].byte_range = OptionalByteRangeRequest::Suffix(8);
  requests[3].byte_range = OptionalByteRangeRequest::Range(1, 5);
  requests[4].byte_range = OptionalByteRangeRequest{};  // Full

  ResolveCoalescedRequests(
      ByteRange{0, 10}, tensorstore::span(requests),
      tensorstore::kvstore::ReadResult::Missing(absl::Now()));

  for (size_t i = 0; i < 5; ++i) {
    ASSERT_TRUE(futures[i].ready());
    EXPECT_THAT(futures[i].result().value().state,
                ::testing::Eq(tensorstore::kvstore::ReadResult::kMissing));
  }
}

}  // namespace
