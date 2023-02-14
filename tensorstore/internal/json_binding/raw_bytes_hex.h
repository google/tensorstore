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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_RAW_BYTES_HEX_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_RAW_BYTES_HEX_H_

#include <cstddef>
#include <type_traits>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/json_serialization_options_base.h"

namespace tensorstore {
namespace internal_json_binding {

namespace raw_bytes_hex_binder {
struct RawBytesHexImpl {
  size_t num_bytes;
  absl::Status operator()(std::true_type is_loading, NoOptions, void* obj,
                          ::nlohmann::json* j) const;
  absl::Status operator()(std::false_type is_loading, NoOptions,
                          const void* obj, ::nlohmann::json* j) const;
};

/// JSON binder for values encoded as a string of hex digits directly specifying
/// the in-memory byte representation.
constexpr auto RawBytesHex = [](auto is_loading, NoOptions options, auto* obj,
                                auto* j) -> absl::Status {
  using T = internal::remove_cvref_t<decltype(*obj)>;
  static_assert(std::is_trivially_destructible_v<T>);
  return RawBytesHexImpl{sizeof(T)}(is_loading, options, obj, j);
};
}  // namespace raw_bytes_hex_binder
using raw_bytes_hex_binder::RawBytesHex;

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_RAW_BYTES_HEX_H_
