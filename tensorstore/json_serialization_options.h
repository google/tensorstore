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

#include "tensorstore/index.h"
#include "tensorstore/rank.h"

/// \file
/// Defines options for conversion to/from JSON.
///
/// Each individual option is defined by a separate class in order to support a
/// form of named parameters.  For example:
///
///     auto x = KeyValaueStore::Spec::FromJson(AllowUnregistered{true});
///     auto j = x.ToJson(IncludeDefaults{false});
///     auto j = x.ToJson({IncludeDefaults{false}, IncludeContext{true}});

namespace tensorstore {

/// Specifies whether to defer errors due to unregistered drivers/context
/// resources until such driver/resource is actually used.  This may be useful
/// to allow working with JSON specifications with a mixture of binaries, where
/// some binaries do not have all of the resources/drivers linked in.
class AllowUnregistered {
 public:
  constexpr explicit AllowUnregistered(bool allow_unregistered = true)
      : value_(allow_unregistered) {}
  bool allow_unregistered() const { return value_; }

 private:
  bool value_;
};

/// Specifies whether members equal to their default values are included when
/// converting to JSON.
class IncludeDefaults {
 public:
  constexpr explicit IncludeDefaults(bool include_defaults = true)
      : value_(include_defaults) {}
  bool include_defaults() const { return value_; }

 private:
  bool value_;
};

/// Specifies whether context and context resource specifications are included
/// when converting to JSON.
class IncludeContext {
 public:
  constexpr explicit IncludeContext(bool include_context = true)
      : value_(include_context) {}
  bool include_context() const { return value_; }

 private:
  bool value_;
};

/// Options for converting `Context::Spec` and other types that embed context
/// resources to JSON.
class ContextToJsonOptions : public IncludeDefaults, public IncludeContext {
 public:
  constexpr ContextToJsonOptions(
      IncludeDefaults include_defaults,
      IncludeContext include_context = IncludeContext{true})
      : IncludeDefaults(include_defaults), IncludeContext(include_context) {}
  constexpr ContextToJsonOptions(
      IncludeContext include_context = IncludeContext{true})
      : IncludeContext(include_context) {}
};

/// Options for converting JSON to `Context::Spec` and other types that embed
/// context context resources.
class ContextFromJsonOptions : public AllowUnregistered {
 public:
  constexpr ContextFromJsonOptions(
      AllowUnregistered allow_unregistered = AllowUnregistered{false})
      : AllowUnregistered(allow_unregistered) {}
};

struct RankConstraint {
  DimensionIndex rank = dynamic_rank;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_JSON_SERIALIZATION_OPTIONS_H_
