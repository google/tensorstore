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

#ifndef TENSORSTORE_KVSTORE_GENERATION_TESTUTIL_H_
#define TENSORSTORE_KVSTORE_GENERATION_TESTUTIL_H_

#include <string>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

/// Returns the current time as of the start of the call, and waits until that
/// time is no longer the current time.
///
/// This is used to ensure consistent testing.
inline absl::Time UniqueNow() {
  absl::Time t = absl::Now();
  do {
    absl::SleepFor(absl::Milliseconds(1));
  } while (absl::Now() == t);
  return t;
}

/// Returns a GMock matcher for `StorageGeneration`.
inline ::testing::Matcher<StorageGeneration> MatchesStorageGeneration(
    ::testing::Matcher<std::string> value_matcher) {
  return ::testing::Field("value", &StorageGeneration::value, value_matcher);
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
      ::testing::AllOf(::testing::Not(StorageGeneration::Unknown()),
                       ::testing::Not(StorageGeneration::NoValue()),
                       ::testing::Not(StorageGeneration::Invalid())),
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

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GENERATION_TESTUTIL_H_
