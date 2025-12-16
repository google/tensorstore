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

#include "tensorstore/kvstore/test_util/spec_ops.h"

#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/testing/json_gtest.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/auto_detect.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

void TestKeyValueStoreSpecRoundtrip(
    const KeyValueStoreSpecRoundtripOptions& options) {
  const auto& expected_minimal_spec = options.minimal_spec.is_discarded()
                                          ? options.full_spec
                                          : options.minimal_spec;
  const auto& create_spec = options.create_spec.is_discarded()
                                ? options.full_spec
                                : options.create_spec;
  SCOPED_TRACE(tensorstore::StrCat("full_spec=", options.full_spec.dump()));
  SCOPED_TRACE(tensorstore::StrCat("create_spec=", create_spec.dump()));
  SCOPED_TRACE(
      tensorstore::StrCat("minimal_spec=", expected_minimal_spec.dump()));
  auto context = options.context;

  ASSERT_TRUE(options.check_write_read || !options.check_data_persists);
  ASSERT_TRUE(options.check_write_read ||
              !options.check_data_after_serialization);

  KvStore serialized_store;
  kvstore::Spec serialized_spec;

  ASSERT_TRUE(options.check_store_serialization ||
              !options.check_data_after_serialization);

  if (!options.url.empty()) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_obj,
                                     kvstore::Spec::FromJson(create_spec));
    EXPECT_THAT(spec_obj.ToUrl(), ::testing::Optional(options.url));
  }

  // Open and populate roundtrip_key.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, kvstore::Open(create_spec, context).result());

    if (options.check_write_read) {
      ASSERT_THAT(
          kvstore::Write(store, options.roundtrip_key, options.roundtrip_value)
              .result(),
          MatchesRegularTimestampedStorageGeneration());
      EXPECT_THAT(kvstore::Read(store, options.roundtrip_key).result(),
                  MatchesKvsReadResult(options.roundtrip_value));
    }

    if (options.check_store_serialization) {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          serialized_store, serialization::SerializationRoundTrip(store));
      {
        TENSORSTORE_ASSERT_OK_AND_ASSIGN(
            auto serialized_store_spec,
            serialized_store.spec(
                kvstore::SpecRequestOptions{options.spec_request_options}));
        EXPECT_THAT(
            serialized_store_spec.ToJson(options.json_serialization_options),
            IsOkAndHolds(MatchesJson(options.full_spec)));
      }
    }

    if (options.check_data_after_serialization) {
      EXPECT_THAT(
          kvstore::Read(serialized_store, options.roundtrip_key).result(),
          MatchesKvsReadResult(options.roundtrip_value));
      TENSORSTORE_ASSERT_OK(
          kvstore::Delete(serialized_store, options.roundtrip_key).result());
      EXPECT_THAT(kvstore::Read(store, options.roundtrip_key).result(),
                  MatchesKvsReadResultNotFound());
      EXPECT_THAT(
          kvstore::Read(serialized_store, options.roundtrip_key).result(),
          MatchesKvsReadResultNotFound());
      ASSERT_THAT(kvstore::Write(serialized_store, options.roundtrip_key,
                                 options.roundtrip_value)
                      .result(),
                  MatchesRegularTimestampedStorageGeneration());
      EXPECT_THAT(kvstore::Read(store, options.roundtrip_key).result(),
                  MatchesKvsReadResult(options.roundtrip_value));
      EXPECT_THAT(
          kvstore::Read(serialized_store, options.roundtrip_key).result(),
          MatchesKvsReadResult(options.roundtrip_value));
    }

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto spec,
        store.spec(kvstore::SpecRequestOptions{options.spec_request_options}));
    EXPECT_THAT(spec.ToJson(options.json_serialization_options),
                IsOkAndHolds(MatchesJson(options.full_spec)));

    // Test serialization of spec.
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        serialized_spec, serialization::SerializationRoundTrip(spec));

    EXPECT_THAT(serialized_spec.ToJson(options.json_serialization_options),
                IsOkAndHolds(MatchesJson(options.full_spec)));

    if (options.url.empty()) {
      EXPECT_THAT(spec.ToUrl(), ::testing::Not(IsOk()));
    } else {
      EXPECT_THAT(spec.ToUrl(), ::testing::Optional(options.url));
    }

    auto minimal_spec_obj = spec;
    TENSORSTORE_ASSERT_OK(minimal_spec_obj.Set(tensorstore::MinimalSpec{true}));
    EXPECT_THAT(minimal_spec_obj.ToJson(options.json_serialization_options),
                IsOkAndHolds(MatchesJson(expected_minimal_spec)));

    // Check base
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_base, store.base());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_base, spec.base());
    EXPECT_EQ(store_base.valid(), spec_base.valid());
    if (store_base.valid()) {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_base_spec, store_base.spec());
      EXPECT_THAT(spec_base.ToJson(),
                  IsOkAndHolds(MatchesJson(options.full_base_spec)));
      EXPECT_THAT(store_base_spec.ToJson(),
                  IsOkAndHolds(MatchesJson(options.full_base_spec)));

      // Check that base spec can be opened.
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto store_base_reopened, kvstore::Open(spec_base, context).result());
      EXPECT_EQ(store_base_reopened, store_base);

      if (options.check_auto_detect) {
        EXPECT_THAT(
            internal_kvstore::AutoDetectFormat(InlineExecutor{}, store_base)
                .result(),
            ::testing::Optional(
                ::testing::ElementsAre(internal_kvstore::AutoDetectMatch{
                    std::string(spec.driver->driver_id())})));
      }
    } else {
      EXPECT_THAT(options.full_base_spec,
                  MatchesJson(::nlohmann::json::value_t::discarded));
      ASSERT_FALSE(options.check_auto_detect);
    }
  }

  // Reopen and verify contents.
  if (options.check_data_persists) {
    // Reopen with full_spec
    {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto store, kvstore::Open(options.full_spec, context).result());
      TENSORSTORE_ASSERT_OK(store.spec());
      EXPECT_THAT(kvstore::Read(store, options.roundtrip_key).result(),
                  MatchesKvsReadResult(options.roundtrip_value));
    }
    if (!options.minimal_spec.is_discarded()) {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto store, kvstore::Open(expected_minimal_spec, context).result());
      TENSORSTORE_ASSERT_OK(store.spec());
      EXPECT_THAT(kvstore::Read(store, options.roundtrip_key).result(),
                  MatchesKvsReadResult(options.roundtrip_value));
    }

    // Reopen with serialized spec.
    {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto store, kvstore::Open(serialized_spec, context).result());
      TENSORSTORE_ASSERT_OK(store.spec());
      EXPECT_THAT(kvstore::Read(store, options.roundtrip_key).result(),
                  MatchesKvsReadResult(options.roundtrip_value));
    }

    // Reopen with url
    if (!options.url.empty()) {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto store, kvstore::Open(options.url, context).result());
      TENSORSTORE_ASSERT_OK(store.spec());
      EXPECT_THAT(kvstore::Read(store, options.roundtrip_key).result(),
                  MatchesKvsReadResult(options.roundtrip_value));
    }
  }
}

void TestKeyValueStoreSpecRoundtripNormalize(
    ::nlohmann::json json_spec, ::nlohmann::json normalized_json_spec) {
  SCOPED_TRACE(tensorstore::StrCat("json_spec=", json_spec.dump()));
  SCOPED_TRACE(tensorstore::StrCat("normalized_json_spec=",
                                   normalized_json_spec.dump()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   kvstore::Open(json_spec).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, store.spec());
  EXPECT_THAT(spec.ToJson(), IsOkAndHolds(MatchesJson(normalized_json_spec)));
}

void TestKeyValueStoreUrlRoundtrip(::nlohmann::json json_spec,
                                   std::string_view url) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_from_json,
                                   kvstore::Spec::FromJson(json_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_from_url,
                                   kvstore::Spec::FromUrl(url));
  EXPECT_THAT(spec_from_json.ToUrl(), IsOkAndHolds(url));
  EXPECT_THAT(spec_from_url.ToJson(), IsOkAndHolds(MatchesJson(json_spec)));
}

}  // namespace internal
}  // namespace tensorstore
