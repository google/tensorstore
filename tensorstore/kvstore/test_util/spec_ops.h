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

#ifndef TENSORSTORE_KVSTORE_TEST_UTIL_SPEC_OPS_H_
#define TENSORSTORE_KVSTORE_TEST_UTIL_SPEC_OPS_H_

#include <string>
#include <string_view>

#include "absl/strings/cord.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/kvstore.h"

namespace tensorstore {
namespace internal {

struct KeyValueStoreSpecRoundtripOptions {
  // Spec that round trips with default options.
  ::nlohmann::json full_spec;

  // Specifies spec for initially creating the kvstore.  Defaults to
  // `full_spec`.
  ::nlohmann::json create_spec = ::nlohmann::json::value_t::discarded;

  // Result of calling `base()` on full spec.  The default value of `discarded`
  // means a null spec.
  ::nlohmann::json full_base_spec = ::nlohmann::json::value_t::discarded;

  // Specifies spec returned when `MinimalSpec{true}` is specified.  Defaults to
  // `full_spec`.
  ::nlohmann::json minimal_spec = ::nlohmann::json::value_t::discarded;
  kvstore::SpecRequestOptions spec_request_options;
  JsonSerializationOptions json_serialization_options;

  Context context = Context::Default();

  // Checks reading and writing.
  bool check_write_read = true;

  // Checks that data persists after re-opening from the returned spec.
  // Requires `check_write_read == true`.
  bool check_data_persists = true;

  // Check that the store can be round-tripped through its serialized
  // representation.
  bool check_store_serialization = true;

  // Checks that data can be read/written after round-tripping through
  // serialization.
  //
  // Doesn't work for "memory://".
  bool check_data_after_serialization = true;

  std::string roundtrip_key = "mykey";
  absl::Cord roundtrip_value = absl::Cord("myvalue");

  // Check that store round trips through the specified URL.
  //
  // If not specified, checks that `ToUrl()` returns an error.
  std::string url;

  // Check that kvstore auto-detection on `full_base_spec` results in
  // `minimal_spec`.  Requires `check_data_persists=true`.
  bool check_auto_detect = false;
};

// Tests that the KvStore spec round-trips in several ways.
void TestKeyValueStoreSpecRoundtrip(
    const KeyValueStoreSpecRoundtripOptions& options);

// Tests that the KvStore spec constructed from `json_spec` corresponds to the
// URL representation `url`.
void TestKeyValueStoreUrlRoundtrip(::nlohmann::json json_spec,
                                   std::string_view url);

// Tests that `json_spec` round trips to `normalized_json_spec`.
void TestKeyValueStoreSpecRoundtripNormalize(
    ::nlohmann::json json_spec, ::nlohmann::json normalized_json_spec);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TEST_UTIL_SPEC_OPS_H_
