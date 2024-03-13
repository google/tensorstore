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

#ifndef TENSORSTORE_KVSTORE_TEST_MATCHERS_H_
#define TENSORSTORE_KVSTORE_TEST_MATCHERS_H_

#include <string>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

/// Returns a GMock matcher for `StorageGeneration`.
inline ::testing::Matcher<StorageGeneration> MatchesStorageGeneration(
    ::testing::Matcher<std::string> value_matcher) {
  return ::testing::Field("value", &StorageGeneration::value, value_matcher);
}

/// GMock matcher for `StorageGeneration` values that correspond to a
/// successfully-written value.
MATCHER(IsRegularStorageGeneration, "") {
  return StorageGeneration::IsClean(arg) &&
         arg != StorageGeneration::Invalid() &&
         !StorageGeneration::IsNoValue(arg);
}

/// Returns a GMock matcher for `TimestampedStorageGeneration`.
template <typename GenerationMatcher>
::testing::Matcher<Result<TimestampedStorageGeneration>>
MatchesTimestampedStorageGeneration(
    GenerationMatcher generation,
    ::testing::Matcher<absl::Time> time = ::testing::_) {
  ::testing::Matcher<StorageGeneration> generation_matcher;
  if constexpr (std::is_convertible_v<GenerationMatcher,
                                      ::testing::Matcher<std::string>>) {
    generation_matcher = MatchesStorageGeneration(generation);
  } else {
    static_assert(std::is_convertible_v<GenerationMatcher,
                                        ::testing::Matcher<StorageGeneration>>);
    generation_matcher = generation;
  }
  return ::testing::Optional(::testing::AllOf(
      ::testing::Field("generation", &TimestampedStorageGeneration::generation,
                       generation_matcher),
      ::testing::Field("time", &TimestampedStorageGeneration::time, time)));
}

/// Returns a GMock matcher for `TimestampedStorageGeneration` that matches any
/// `TimestampedStorageGeneration` where the `generation` is not one of the
/// special `Unknown`, `NoValue`, or `Invalid` values.
inline ::testing::Matcher<Result<TimestampedStorageGeneration>>
MatchesRegularTimestampedStorageGeneration(
    ::testing::Matcher<absl::Time> time = ::testing::_) {
  return MatchesTimestampedStorageGeneration(
      ::testing::Matcher<StorageGeneration>(IsRegularStorageGeneration()),
      time);
}

/// Returns a GMock matcher for `TimestampedStorageGeneration` that matches any
/// `TimestampedStorageGeneration` where the `generation` is not
/// `StorageGeneration::Unknown()`.
inline ::testing::Matcher<Result<TimestampedStorageGeneration>>
MatchesKnownTimestampedStorageGeneration(
    ::testing::Matcher<absl::Time> time = ::testing::_) {
  return MatchesTimestampedStorageGeneration(
      ::testing::Not(StorageGeneration::Unknown()), time);
}

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

/// Returns a GMock matcher for a `ListEntry`.
inline ::testing::Matcher<kvstore::ListEntry> MatchesListEntry(
    ::testing::Matcher<std::string> key_matcher,
    ::testing::Matcher<int64_t> size_matcher = ::testing::_) {
  return ::testing::AllOf(
      ::testing::Field("key", &kvstore::ListEntry::key, key_matcher),
      ::testing::Field("size", &kvstore::ListEntry::size, size_matcher));
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TEST_MATCHERS_H_
