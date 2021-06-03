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

#ifndef TENSORSTORE_SCHEMA_H_
#define TENSORSTORE_SCHEMA_H_

#include <memory>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_spec.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

/// Collection of properties that define the key characteristics of a
/// TensorStore, independent of the specific driver or storage mechanism/format.
class Schema {
 public:
  struct Builder;

  struct FillValue : public SharedArrayView<const void> {
    FillValue() = default;
    explicit FillValue(SharedArrayView<const void> value)
        : SharedArrayView<const void>(std::move(value)) {}
    friend bool operator==(const FillValue& a, const FillValue& b);
    friend bool operator!=(const FillValue& a, const FillValue& b) {
      return !(a == b);
    }
  };

  static Result<Schema> Make(Builder builder);

  DataType dtype() const;
  DimensionIndex rank() const;
  IndexDomain<> domain() const;
  ChunkLayout chunk_layout() const;
  CodecSpec::Ptr codec() const;
  FillValue fill_value() const;

  struct Builder {
    DataType dtype;
    IndexDomain<> domain;
    ChunkLayout chunk_layout;
    CodecSpec::Ptr codec;
    SharedArray<const void> fill_value;
  };

  template <typename Expr>
  friend std::enable_if_t<
      !IsIndexTransform<internal::remove_cvref_t<Expr>>::value, Result<Schema>>
  ApplyIndexTransform(Expr&& expr, Schema schema) {
    if (!schema.impl_) return schema;
    TENSORSTORE_ASSIGN_OR_RETURN(auto identity_transform,
                                 schema.identity_transform());
    if (!identity_transform.valid()) {
      // Only `dtype`, `codec`, or scalar `fill_value` set.
      return schema;
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto transform,
        std::forward<Expr>(expr)(std::move(identity_transform)));
    return ApplyIndexTransform(std::move(transform), std::move(schema));
  }

  friend Result<Schema> ApplyIndexTransform(IndexTransform<> transform,
                                            Schema schema);

  using ToJsonOptions = JsonSerializationOptions;
  using FromJsonOptions = JsonSerializationOptions;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(Schema, FromJsonOptions,
                                          ToJsonOptions)

  /// "Pipeline" operator.
  ///
  /// In the expression  `x | y`, if
  ///   * y is a function having signature `Result<U>(T)`
  ///
  /// Then operator| applies y to the value of x, returning a
  /// Result<U>. See tensorstore::Result operator| for examples.
  template <typename Func>
  friend PipelineResultType<Schema, Func> operator|(Schema schema,
                                                    Func&& func) {
    return std::forward<Func>(func)(std::move(schema));
  }

  friend bool operator==(const Schema& a, const Schema& b);
  friend bool operator!=(const Schema& a, const Schema& b) { return !(a == b); }

  friend std::ostream& operator<<(std::ostream& os, const Schema& schema);

 private:
  /// Returns an identity index transform over the domain/rank to which
  /// dimension expressions can be applied in order to transform this schema.
  ///
  /// If `domain().valid() == true`, returns `IdentityTransform(domain())`.
  ///
  /// Otherwise, if `rank() != dynamic_rank`, returns
  /// `IdentityTransform(rank())`.
  ///
  /// Otherwise, returns a default-constructed (invalid) transform.
  ///
  /// \error `absl::StatusCode::kInvalidArgument` if `rank() == dynamic_rank`
  ///     but `fill_value` is specified and non-scalar, since in that case the
  ///     result depends on the rank, which is unknown.
  Result<IndexTransform<>> identity_transform() const;

 public:
  // Treat as private:

  class Impl;
  friend void intrusive_ptr_increment(Impl* p);
  friend void intrusive_ptr_decrement(Impl* p);
  internal::IntrusivePtr<Impl> impl_;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_CREATE_SPEC_H_
