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

#ifndef TENSORSTORE_INDEX_SPACE_DIMENSION_UNITS_H_
#define TENSORSTORE_INDEX_SPACE_DIMENSION_UNITS_H_

#include <optional>
#include <vector>

#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/unit.h"

namespace tensorstore {

/// Vector specifying optional units for each dimension of an index space.
///
/// \relates Schema
using DimensionUnitsVector = std::vector<std::optional<Unit>>;

/// Converts a dimension unit vector to a string for use in error messages.
///
/// \relates DimensionUnitsVector
std::string DimensionUnitsToString(span<const std::optional<Unit>> u);

/// Merges new dimension units with existing dimension units.
///
/// The units for each dimension are merged independently.
///
/// 1. If the existing and new unit are same (either both unspecified, or both
///    specified with the same unit), the merged unit is equal to the common
///    unit.
///
/// 2. If the existing unit for a dimension is
///    specified, and the new unit is not, or vice versa, the merged unit is the
///    specified unit.
///
/// 3. It is an error two merge two distinct specified (i.e. not `std::nullopt`)
///    units.
///
/// \relates DimensionUnitsVector
absl::Status MergeDimensionUnits(DimensionUnitsVector& existing_units,
                                 span<const std::optional<Unit>> new_units);

/// Converts dimension units for the input space to dimension units for the
/// output space.
///
/// Output dimensions with `OutputIndexMethod::single_input_dimension` maps are
/// assigned the unit of the corresponding input dimension, if any, divided by
/// the absolute value of the stride.
///
/// The units of output dimensions with `OutputIndexMethod::constant` and
/// `OutputIndexMethod::array` maps are left unspecified.
///
/// If a unit is specified for an input dimension, but no output dimension
/// depends on it via a `OutputIndexMethod::single_input_dimension` map, an
/// error is returned to indicate that this unit would be "lost".
///
/// \param transform Index transform.
/// \param input_units Units for each input dimension of `transform`.
/// \error `absl::StatusCode::kInvalidArgument` if a unit is specified for an
///     input dimension that does not correspond to an output dimension.
/// \relates DimensionUnitsVector
Result<DimensionUnitsVector> TransformInputDimensionUnits(
    IndexTransformView<> transform, DimensionUnitsVector input_units);

/// Converts dimension units for the output space to dimension units for the
/// input space.
///
/// Input dimensions that correspond to exactly one output dimension via a
/// `OutputIndexMethod::single_input_dimension` map are assigned the unit of the
/// corresponding output dimension, if any, multiplied by the absolute value of
/// the stride.
///
/// The units of other input dimensions are left unspecified.
///
/// \param transform Index transform.
/// \param output_units Units for each output dimension of `transform`.
/// \relates DimensionUnitsVector
DimensionUnitsVector TransformOutputDimensionUnits(
    IndexTransformView<> transform, DimensionUnitsVector output_units);

}  // namespace tensorstore

#endif  //  TENSORSTORE_INDEX_SPACE_DIMENSION_UNITS_H_
