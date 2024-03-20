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

#ifndef TENSORSTORE_DRIVER_ARRAY_ARRAY_H_
#define TENSORSTORE_DRIVER_ARRAY_ARRAY_H_

#include <type_traits>

#include "tensorstore/context.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/tensorstore.h"

namespace tensorstore {

/// Alias that evaluates to the TensorStore type corresponding to `Array`.
///
/// \tparam Array The `Array` or `TransformedArray` type.
/// \requires `IsTransformedArrayLike<Array>`.
template <typename Array>
using TensorStoreFromArrayType = TensorStore<
    std::remove_const_t<typename Array::Element>, Array::static_rank,
    (std::is_const_v<typename Array::Element> ? ReadWriteMode::read
                                              : ReadWriteMode::read_write)>;

struct FromArrayOptions {
  Context context;
  DimensionUnitsVector dimension_units;

  template <typename T>
  constexpr static inline bool IsOption = false;

  /// Combines any number of supported options.
  template <typename... T, typename = std::enable_if_t<
                               (IsOption<absl::remove_cvref_t<T>> && ...)>>
  FromArrayOptions(T&&... option) {
    (Set(std::forward<T>(option)), ...);
  }

  void Set(Context value) { context = std::move(value); }

  void Set(DimensionUnitsVector value) { dimension_units = std::move(value); }
};

template <>
constexpr inline bool FromArrayOptions::IsOption<Context> = true;

template <>
constexpr inline bool FromArrayOptions::IsOption<DimensionUnitsVector> = true;

namespace internal {
template <ArrayOriginKind OriginKind>
Result<internal::Driver::Handle> MakeArrayDriver(
    SharedArray<void, dynamic_rank, OriginKind> array,
    FromArrayOptions options = {});
}  // namespace internal

/// Returns a `TensorStore` that holds a given `array`.
///
/// \param array The array held by the `TensorStore`.
/// \param options Options compatible with `FromArrayOptions`.
/// \requires `IsArray<Array>`.
template <typename Array>
std::enable_if_t<(IsArray<Array> && IsShared<typename Array::ElementTag>),
                 Result<TensorStoreFromArrayType<Array>>>
FromArray(const Array& array, FromArrayOptions options) {
  using Store = TensorStoreFromArrayType<Array>;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto handle, internal::MakeArrayDriver<Array::array_origin_kind>(
                       ConstDataTypeCast<typename Store::Element>(array),
                       std::move(options)));
  return internal::TensorStoreAccess::Construct<Store>(std::move(handle));
}
template <typename Array, typename... Options>
std::enable_if_t<(IsArray<Array> && IsShared<typename Array::ElementTag> &&
                  IsCompatibleOptionSequence<FromArrayOptions, Options...>),
                 Result<TensorStoreFromArrayType<Array>>>
FromArray(const Array& array, Options&&... options) {
  using Store = TensorStoreFromArrayType<Array>;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto handle, internal::MakeArrayDriver<Array::array_origin_kind>(
                       ConstDataTypeCast<typename Store::Element>(array),
                       FromArrayOptions(std::forward<Options>(options)...)));
  return internal::TensorStoreAccess::Construct<Store>(std::move(handle));
}

/// Returns a `Spec` for an array driver that holds the given `array`.
///
/// The specified `array` is shared with the returned `Spec` and should not be
/// modified, but opening the returned `Spec` makes a copy of the array.

/// \param array The array to be held by the `Spec`.
/// \param dimension_units Optional dimension units.  If specified, the length
///     must equal `array.rank()`.
Result<tensorstore::Spec> SpecFromArray(
    SharedOffsetArrayView<const void> array,
    DimensionUnitsVector dimension_units = {});

}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ARRAY_ARRAY_H_
