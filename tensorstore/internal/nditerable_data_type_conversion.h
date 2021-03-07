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

#ifndef TENSORSTORE_INTERNAL_NDITERABLE_DATA_TYPE_CONVERSION_H_
#define TENSORSTORE_INTERNAL_NDITERABLE_DATA_TYPE_CONVERSION_H_

/// \file
/// Facilities for applying data type conversions to NDIterable objects.

#include "tensorstore/data_type.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/internal/nditerable.h"

namespace tensorstore {
namespace internal {

/// Returns a read-only NDIterable with a `dtype` of `target_type` using
/// `conversion`.
///
/// \param iterable Readable source iterable.
/// \param target_type Target data type.
/// \param conversion Must equal
///     `GetDataTypeConverter(iterable.dtype(), target_type)`.
NDIterable::Ptr GetConvertedInputNDIterable(
    NDIterable::Ptr iterable, DataType target_type,
    const DataTypeConversionLookupResult& conversion);

/// Returns a write-only NDIterable with a `dtype` of `target_type` using
/// `conversion`.
///
/// \param iterable Writable target iterable.
/// \param source_type Source data type.
/// \param conversion Must equal
///     `GetDataTypeConverter(source_type, iterable.dtype())`.
NDIterable::Ptr GetConvertedOutputNDIterable(
    NDIterable::Ptr iterable, DataType source_type,
    const DataTypeConversionLookupResult& conversion);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_NDITERABLE_DATA_TYPE_CONVERSION_H_
