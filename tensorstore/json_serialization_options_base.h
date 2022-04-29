// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_JSON_SERIALIZATION_OPTIONS_BASE_H_
#define TENSORSTORE_JSON_SERIALIZATION_OPTIONS_BASE_H_

#include "tensorstore/index.h"
#include "tensorstore/rank.h"

namespace tensorstore {
namespace internal_json_binding {

struct NoOptions {
  constexpr NoOptions() = default;
  template <typename T>
  constexpr NoOptions(const T&) {}
};

}  // namespace internal_json_binding

/// Specifies whether members equal to their default values are included when
/// converting to JSON.
///
/// \ingroup json
class IncludeDefaults {
 public:
  constexpr explicit IncludeDefaults(bool include_defaults = false)
      : value_(include_defaults) {}
  constexpr bool include_defaults() const { return value_; }

 private:
  bool value_;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_JSON_SERIALIZATION_OPTIONS_BASE_H_
