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

#ifndef TENSORSTORE_JSON_SERIALIZATION_OPTIONS_H_
#define TENSORSTORE_JSON_SERIALIZATION_OPTIONS_H_

#include <type_traits>

#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/rank.h"

/// \file
/// Defines options for conversion to/from JSON.
///
/// Each individual option is defined by a separate class in order to support a
/// form of named parameters.  For example:
///
///     auto j = x.ToJson(IncludeDefaults{false});
///     auto j = x.ToJson({IncludeDefaults{false}, RankConstraint{3}});

namespace tensorstore {

class JsonSerializationOptions {
 public:
  template <typename T>
  constexpr static bool IsOption = false;

  template <typename... T, typename = std::enable_if_t<(IsOption<T> && ...)>>
  constexpr JsonSerializationOptions(T... option) {
    (Set(option), ...);
  }

  constexpr void Set(JsonSerializationOptions value) { *this = value; }

  constexpr void Set(RankConstraint value) {
    assert(value.rank >= -1 && value.rank <= kMaxRank);
    rank_ = value.rank;
  }

  constexpr void Set(IncludeDefaults value) {
    include_defaults_ = value.include_defaults();
  }

  constexpr void Set(DataType value) { data_type_ = value; }
  template <typename T>
  constexpr void Set(StaticDataType<T> value) {
    data_type_ = value;
  }
  constexpr void Set(internal_json_binding::NoOptions) {}

  constexpr operator RankConstraint() const { return RankConstraint(rank_); }
  constexpr RankConstraint rank() const { return RankConstraint(rank_); }

  constexpr operator DataType() const { return data_type_; }
  constexpr DataType dtype() const { return data_type_; }

  constexpr operator IncludeDefaults() const {
    return IncludeDefaults(include_defaults_);
  }

 private:
  DataType data_type_;
  bool include_defaults_ = false;
  int8_t rank_ = dynamic_rank;

 public:
  // For internal use only: context resource specs that should be immediately
  // re-bound upon deserialization are indicated by a single-element array
  // wrapping the normal context resource spec.  This is used for
  // serialization/pickling only.
  bool preserve_bound_context_resources_ = false;
};

template <>
constexpr bool JsonSerializationOptions::IsOption<JsonSerializationOptions> =
    true;
template <>
constexpr bool JsonSerializationOptions::IsOption<RankConstraint> = true;
template <>
constexpr bool JsonSerializationOptions::IsOption<IncludeDefaults> = true;
template <>
constexpr bool JsonSerializationOptions::IsOption<DataType> = true;
template <typename T>
constexpr bool JsonSerializationOptions::IsOption<StaticDataType<T>> = true;
template <>
constexpr bool
    JsonSerializationOptions::IsOption<internal_json_binding::NoOptions> = true;

}  // namespace tensorstore

#endif  // TENSORSTORE_JSON_SERIALIZATION_OPTIONS_H_
