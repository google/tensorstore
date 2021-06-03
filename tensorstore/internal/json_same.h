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

#ifndef TENSORSTORE_INTERNAL_JSON_SAME_H_
#define TENSORSTORE_INTERNAL_JSON_SAME_H_

#include "tensorstore/internal/json_fwd.h"

namespace tensorstore {
namespace internal_json {

/// Returns `true` if `a` and `b` are equal.
///
/// Unlike `operator==`, the comparison is non-recursive, and is therefore safe
/// from stack overflow even for deeply nested structures.
///
/// Like `operator==`, two int64_t/uint64_t/double values representing exactly
/// the same number are all considered equal even if their types differ.
///
/// Unlike `operator==`, two `discarded` values are considered equal.
bool JsonSame(const ::nlohmann::json& a, const ::nlohmann::json& b);

}  // namespace internal_json
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_SAME_H_
