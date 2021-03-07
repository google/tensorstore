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

#ifndef TENSORSTORE_DRIVER_CAST_CAST_H_
#define TENSORSTORE_DRIVER_CAST_CAST_H_

#include <cassert>

#include "tensorstore/data_type.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {
Result<Driver::Handle> MakeCastDriver(
    Driver::Handle base, DataType target_dtype,
    ReadWriteMode read_write_mode = ReadWriteMode::dynamic);

/// Determines the compile-time read/write mode that results from a cast
/// operation.
///
/// Fails with an assertion failure (which leads to a compile-time error during
/// constexpr evaluation) if the data types are known at compile time to be
/// incompatible (meaning neither conversion direction is supported).
template <typename SourceElement, typename TargetElement>
constexpr ReadWriteMode GetCastMode(ReadWriteMode existing_mode) {
  if constexpr (std::is_void_v<SourceElement> ||
                std::is_void_v<TargetElement>) {
    // If existing mode is read-write, then resultant mode depends on which
    // conversions are supported.
    //
    // Otherwise, resultant mode is always `existing_mode`, and the conversion
    // fails at run time if the run-time existing mode is incompatible with the
    // supported conversions.
    return (existing_mode == ReadWriteMode::read_write) ? ReadWriteMode::dynamic
                                                        : existing_mode;
  } else if (std::is_same_v<SourceElement, TargetElement>) {
    // No-op conversion.
    return existing_mode;
  } else {
    constexpr auto input_flags =
        DataTypeConversionTraits<SourceElement, TargetElement>::flags;
    constexpr auto output_flags =
        DataTypeConversionTraits<TargetElement, SourceElement>::flags;
    ReadWriteMode mode = ReadWriteMode{};
    if ((input_flags & DataTypeConversionFlags::kSupported) ==
        DataTypeConversionFlags::kSupported) {
      mode = mode | ReadWriteMode::read;
    }
    if ((output_flags & DataTypeConversionFlags::kSupported) ==
        DataTypeConversionFlags::kSupported) {
      mode = mode | ReadWriteMode::write;
    }
    assert(mode != ReadWriteMode() && "Cannot convert data types");
    assert((existing_mode == ReadWriteMode::dynamic) ||
           (((existing_mode & mode) != ReadWriteMode()) &&
            "Supported conversions incompatible with existing mode"));
    // If both read and write conversions are supported, then resultant mode is
    // `existing_mode`.
    //
    // Otherwise, resultant mode is always the statically-known mode `mode`,
    // even if it was previously `dynamic`.  The conversion will fail at run
    // time if the run-time existing mode is incompatible.
    return (mode == ReadWriteMode::read_write) ? existing_mode : mode;
  }
}

struct CastDataTypeConversions {
  /// Conversions used when reading.
  DataTypeConversionLookupResult input;

  /// Conversion used when writing.
  DataTypeConversionLookupResult output;

  /// Supported modes (never `ReadWriteMode::dynamic`).
  ReadWriteMode mode;
};

/// Determines the supported conversions between `source_dtype` and
/// `target_dtype` that are compatible with `existing_mode`.
///
/// \param source_dtype The source data type.
/// \param target_dtype The target data type.
/// \param existing_mode The run-time mode of the existing TensorStore.
/// \param required_mode The mask of required modes.
/// \returns The supported conversions if at least one conversion compatible
///     with `existing_mode` and `required_mode` is supported.
/// \error `absl::StatusCode::kInvalidArgument` if neither read nor write is
///     supported and compatible with `existing_mode` and `required_mode`.
Result<CastDataTypeConversions> GetCastDataTypeConversions(
    DataType source_dtype, DataType target_dtype, ReadWriteMode existing_mode,
    ReadWriteMode required_mode);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_CAST_CAST_H_
