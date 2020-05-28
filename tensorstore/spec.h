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

#ifndef TENSORSTORE_SPEC_H_
#define TENSORSTORE_SPEC_H_

#include <iosfwd>
#include <type_traits>

#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_spec.h"
#include "tensorstore/rank.h"
#include "tensorstore/spec_impl.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Specifies the parameters necessary to open or create a TensorStore,
/// including the driver identifier, driver parameters, and optionally the data
/// type and a transform or rank constraint.
class Spec {
 public:
  /// Constructs an invalid specification.
  Spec() = default;

  /// Returns the data type.
  ///
  /// If the data type is unknown, returns the invalid data type.
  DataType data_type() const {
    return impl_.driver_spec ? impl_.driver_spec->constraints().data_type
                             : DataType();
  }

  /// Returns the TransformSpec applied on top of the driver.
  const IndexTransformSpec& transform_spec() const& {
    return impl_.transform_spec;
  }

  /// Returns the transform applied on top of the driver.
  const IndexTransform<>& transform() const {
    return impl_.transform_spec.transform();
  }

  /// Returns a new `Spec` with the specified `options` applied.
  Result<Spec> Convert(const SpecRequestOptions& options) const;

  /// Returns the rank of the TensorStore, or `dynamic_rank` if unknown.
  DimensionIndex rank() const { return impl_.transform_spec.input_rank(); }

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(Spec, Context::FromJsonOptions,
                                          Context::ToJsonOptions)

  /// Applies a function that operates on an IndexTransform to a Spec.
  ///
  /// This definition allows DimExpression objects to be applied to Spec
  /// objects.
  ///
  /// \returns The transformed `Spec` on success.
  /// \error `absl::StatusCode::kInvalidArgument` if `spec.transform()` is not
  ///     valid.
  template <typename Expr>
  friend Result<Spec> ApplyIndexTransform(Expr&& expr, Spec spec) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        spec.impl_.transform_spec,
        ApplyIndexTransform(std::forward<Expr>(expr),
                            std::move(spec.impl_.transform_spec)));
    return spec;
  }

  friend std::ostream& operator<<(std::ostream& os, const Spec& spec);

  /// Compares for equality via JSON representation.
  friend bool operator==(const Spec& a, const Spec& b);

  friend bool operator!=(const Spec& a, const Spec& b) { return !(a == b); }

 private:
  friend class internal_spec::SpecAccess;
  internal::TransformedDriverSpec<> impl_;
};

/// Returns an error if `rank_constraint` is not compatible with `actual_rank`.
///
/// This is intended to be used by TensorStore driver implementations to check
/// the `rank` constraint in a `Spec`.
Status ValidateSpecRankConstraint(DimensionIndex actual_rank,
                                  DimensionIndex rank_constraint);

}  // namespace tensorstore

#endif  // TENSORSTORE_SPEC_H_
