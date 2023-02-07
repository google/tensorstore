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

#ifndef TENSORSTORE_KVSTORE_TEST_UTIL_H_
#define TENSORSTORE_KVSTORE_TEST_UTIL_H_

#include <map>
#include <string>
#include <string_view>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/function_ref.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

/// Tests all operations on `store`, which should be empty.
///
/// \param get_key Maps arbitrary strings (which are nonetheless valid file
///     paths) to keys in the format expected by `store`.  For stores that
///     support file paths as keys, `get_key` can simply be the identity
///     function.  This function must ensure that a given input key always maps
///     to the same output key, and distinct input keys always map to distinct
///     output keys.
void TestKeyValueStoreBasicFunctionality(
    const KvStore& store,
    absl::FunctionRef<std::string(std::string key)> get_key);

inline void TestKeyValueStoreBasicFunctionality(const KvStore& store) {
  return TestKeyValueStoreBasicFunctionality(
      store, [](std::string key) { return key; });
}

/// Tests DeleteRange on `store`, which should be empty.
void TestKeyValueStoreDeleteRange(const KvStore& store);

/// Tests DeleteRange on `store`, which should be empty.
void TestKeyValueStoreDeletePrefix(const KvStore& store);

/// Tests DeleteRange on `store`, which should be empty.
void TestKeyValueStoreDeleteRangeToEnd(const KvStore& store);

/// Tests DeleteRange on `store`, which should be empty.
void TestKeyValueStoreDeleteRangeFromBeginning(const KvStore& store);

struct KeyValueStoreSpecRoundtripOptions {
  // Spec that round trips with default options.
  ::nlohmann::json full_spec;

  // Specifies spec for initially creating the kvstore.  Defaults to
  // `full_spec`.
  ::nlohmann::json create_spec = ::nlohmann::json::value_t::discarded;

  // Specifies spec returned when `MinimalSpec{true}` is specified.  Defaults to
  // `full_spec`.
  ::nlohmann::json minimal_spec = ::nlohmann::json::value_t::discarded;
  kvstore::SpecRequestOptions spec_request_options;
  JsonSerializationOptions json_serialization_options;
  // Checks reading and writing.
  bool check_write_read = true;

  // Checks that data persists after re-opening from the returned spec.
  // Requires `check_write_read == true`.
  bool check_data_persists = true;

  std::string roundtrip_key = "mykey";
  absl::Cord roundtrip_value = absl::Cord("myvalue");
};

/// Tests that the KvStore spec round-trips in several ways.
///
/// - Tests that calling `kvstore::Open` with `options.create_spec` returns a
///   `KvStore` whose `spec()` method returns `options.full_spec`.
///
/// - Tests that applying the `MinimalSpec` option to `option.full_spec` results
///   in `options.minimal_spec`.
///
/// - If `options.check_data_persists`, tests that data persists when re-opened
///   using `options.minimal_spec`.
///
/// - If `options.check_data_persists`, tests that data persists when re-opened
///   using `options.full_spec`.
void TestKeyValueStoreSpecRoundtrip(
    const KeyValueStoreSpecRoundtripOptions& options);

/// Tests that the KvStore spec constructed from `json_spec` corresponds to the
/// URL representation `url`.
void TestKeyValueStoreUrlRoundtrip(::nlohmann::json json_spec,
                                   std::string_view url);

/// Tests that `json_spec` round trips to `normalized_json_spec`.
void TestKeyValueStoreSpecRoundtripNormalize(
    ::nlohmann::json json_spec, ::nlohmann::json normalized_json_spec);

/// Returns the contents of `kv_store` as an `std::map`.
Result<std::map<kvstore::Key, kvstore::Value>> GetMap(const KvStore& store);

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

/// Overload that permits an `absl::Cord` matcher to be specified for the
/// `value`.
::testing::Matcher<Result<kvstore::ReadResult>> MatchesKvsReadResult(
    ::testing::Matcher<kvstore::Value> value,
    ::testing::Matcher<StorageGeneration> generation = ::testing::_,
    ::testing::Matcher<absl::Time> time = ::testing::_);

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

#endif  // TENSORSTORE_KVSTORE_TEST_UTIL_H_
