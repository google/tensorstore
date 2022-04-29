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

#ifndef TENSORSTORE_CAST_H_
#define TENSORSTORE_CAST_H_

/// \file
/// Data type conversion adapter for TensorStore objects.

#include "tensorstore/data_type.h"
#include "tensorstore/driver/cast/cast.h"
#include "tensorstore/tensorstore.h"

namespace tensorstore {

/// Returns a view of a `TensorStore` with a converted data type.
///
/// The returned `TensorStore` supports read/write mode if, and only if, the
/// input `store` supports the same mode and the necessary conversion is
/// defined.
///
/// For example::
///
///     TensorStore<int32_t, 2, ReadWriteMode::read_write> a = ...;
///
///     // type: TensorStore<int64_t, 2, ReadWriteMode::read_write>
///     auto b = Cast<int64_t>(a);
///     // Supports both reading and writing, since `int32 <-> int64` conversion
///     // is supported in both directions.
///
///     // type: TensorStore<std::string, 2, ReadWriteMode::read>
///     auto c = Cast<std::string>(a);
///     // Supports reading only, since `int32 -> string` conversion is
///     // supported but `string -> int32` is not.
///
///     TensorStore<std::string, 2, ReadWriteMode::read_write> d = ...;
///
///     // type: TensorStore<std::string, 2, ReadWriteMode::write>
///     auto d = Cast<int32_t>(d);
///     // Supports writing only, since `int32 -> string` conversion is
///     // supported but `string -> int32` is not.
///
/// \tparam TargetElementType The target element type, must be unqualified.
/// \param store The TensorStore to convert.
/// \param target_dtype May be specified in order to allow `TargetElementType`
///     to be inferred.
/// \error `absl::StatusCode::kInvalidArgument` if neither reading nor writing
///     would be supported by the returned `TensorStore`.
/// \relates TensorStore
template <typename TargetElementType, int&... ExplicitArgumentBarrier,
          typename ElementType, DimensionIndex Rank, ReadWriteMode Mode>
Result<TensorStore<
    TargetElementType, Rank,
    tensorstore::internal::GetCastMode<ElementType, TargetElementType>(Mode)>>
Cast(TensorStore<ElementType, Rank, Mode> store,
     StaticDataType<TargetElementType> target_dtype = {}) {
  return tensorstore::ChainResult(
      internal::MakeCastDriver(
          std::move(internal::TensorStoreAccess::handle(store)), target_dtype),
      internal::TensorStoreAccess::Construct<TensorStore<
          TargetElementType, Rank,
          tensorstore::internal::GetCastMode<ElementType, TargetElementType>(
              Mode)>>);
}
template <int&... ExplicitArgumentBarrier, typename ElementType,
          DimensionIndex Rank, ReadWriteMode Mode>
Result<TensorStore<void, Rank,
                   (Mode == ReadWriteMode::read_write ? ReadWriteMode::dynamic
                                                      : Mode)>>
Cast(TensorStore<ElementType, Rank, Mode> store, DataType target_dtype) {
  return tensorstore::ChainResult(
      internal::MakeCastDriver(
          std::move(internal::TensorStoreAccess::handle(store)), target_dtype),
      internal::TensorStoreAccess::Construct<TensorStore<
          void, Rank,
          (Mode == ReadWriteMode::read_write ? ReadWriteMode::dynamic
                                             : Mode)>>);
}

}  // namespace tensorstore

#endif  // TENSORSTORE_CAST_H_
