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

#ifndef TENSORSTORE_UTIL_SPAN_JSON_H_
#define TENSORSTORE_UTIL_SPAN_JSON_H_

#include <nlohmann/json.hpp>
#include "tensorstore/util/span.h"

namespace tensorstore {

/// Converts a `span` to a JSON array.
template <typename T, std::ptrdiff_t Extent>
void to_json(::nlohmann::json& out,  // NOLINT
             span<T, Extent> s) {
  out = ::nlohmann::json::array_t(s.begin(), s.end());
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_SPAN_JSON_H_
