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

#ifndef TENSORSTORE_KVSTORE_READ_RESULT_TESTUTIL_H_
#define TENSORSTORE_KVSTORE_READ_RESULT_TESTUTIL_H_

#include <string>
#include <type_traits>

#include <gmock/gmock.h>
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

/// Returns a GMock matcher for a `kvstore::ReadResult` or
/// `Result<kvstore::ReadResult>`.
template <typename ValueMatcher>
::testing::Matcher<Result<kvstore::ReadResult>> MatchesKvsReadResult(
    ValueMatcher value,
    ::testing::Matcher<StorageGeneration> generation = ::testing::_,
    ::testing::Matcher<absl::Time> time = ::testing::_) {
  using ReadResult = kvstore::ReadResult;
  ::testing::Matcher<kvstore::ReadResult::State> state_matcher;
  ::testing::Matcher<kvstore::Value> value_matcher;
  if constexpr (std::is_convertible_v<ValueMatcher,
                                      ::testing::Matcher<kvstore::Value>>) {
    value_matcher = ::testing::Matcher<kvstore::Value>(value);
    state_matcher = kvstore::ReadResult::kValue;
  } else {
    static_assert(
        std::is_convertible_v<ValueMatcher,
                              ::testing::Matcher<kvstore::ReadResult::State>>);
    value_matcher = absl::Cord();
    state_matcher = ::testing::Matcher<kvstore::ReadResult::State>(value);
  }
  return ::testing::Optional(::testing::AllOf(
      ::testing::Field("state", &ReadResult::state, state_matcher),
      ::testing::Field("value", &ReadResult::value, value_matcher),
      ::testing::Field("stamp", &ReadResult::stamp,
                       MatchesTimestampedStorageGeneration(generation, time))));
}

/// Returns a GMock matcher for a "not found" `kvstore::ReadResult`.
inline ::testing::Matcher<Result<kvstore::ReadResult>>
MatchesKvsReadResultNotFound(
    ::testing::Matcher<absl::Time> time = ::testing::_) {
  return MatchesKvsReadResult(kvstore::ReadResult::kMissing,
                              ::testing::Not(StorageGeneration::Unknown()),
                              time);
}

/// Returns a GMock matcher for an "aborted" `kvstore::ReadResult`.
inline ::testing::Matcher<Result<kvstore::ReadResult>>
MatchesKvsReadResultAborted(
    ::testing::Matcher<absl::Time> time = ::testing::_) {
  return MatchesKvsReadResult(kvstore::ReadResult::kUnspecified, ::testing::_,
                              time);
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_READ_RESULT_TESTUTIL_H_
