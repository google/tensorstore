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

#ifndef TENSORSTORE_INTERNAL_ARRAY_CONSTRAINTS_H_
#define TENSORSTORE_INTERNAL_ARRAY_CONSTRAINTS_H_

#include "tensorstore/data_type.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/rank.h"

namespace tensorstore {
namespace internal {

/// Specifies rank and data type information.
struct ArrayConstraints {
  /// Specifies the data type, or equal to `DataType()` if unknown.
  DataType dtype;

  /// Specifies the rank, or equal to `dynamic_rank` if unknown.
  DimensionIndex rank = dynamic_rank;
};

/// Options for loading array-like types from JSON.
///
/// These are used for `internal::DriverSpec` and `CreateSpec`.
struct ArrayFromJsonOptions : public ContextFromJsonOptions,
                              public ArrayConstraints {
  using ContextFromJsonOptions::ContextFromJsonOptions;
  constexpr ArrayFromJsonOptions(const ContextFromJsonOptions& options,
                                 const ArrayConstraints& constraints = {})
      : ContextFromJsonOptions(options), ArrayConstraints(constraints) {}
};

/// Options for converting an array-like type to JSON.
///
/// These are used for `internal::DriverSpec` and `CreateSpec`.
struct ArrayToJsonOptions : public ContextToJsonOptions,
                            public ArrayConstraints {
  using ContextToJsonOptions::ContextToJsonOptions;
  constexpr ArrayToJsonOptions(const ContextToJsonOptions& options,
                               const ArrayConstraints& constraints = {})
      : ContextToJsonOptions(options), ArrayConstraints(constraints) {}
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ARRAY_CONSTRAINTS_H_
