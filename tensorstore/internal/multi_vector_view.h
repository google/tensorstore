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

#ifndef TENSORSTORE_INTERNAL_MULTI_VECTOR_VIEW_H_
#define TENSORSTORE_INTERNAL_MULTI_VECTOR_VIEW_H_

#include <cassert>
#include <type_traits>

#include "tensorstore/index.h"
#include "tensorstore/internal/gdb_scripting.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"

TENSORSTORE_GDB_AUTO_SCRIPT("multi_vector_gdb.py")

namespace tensorstore {
namespace internal {

/// MultiVectorViewStorage is a non-owning view of multiple one-dimensional
/// arrays, all of the same length, one array for each of the `Ts...`.
///
/// Like `tensorstore::span`, MultiVectorViewStorage allows both static and
/// dynamic array lengths via the `Extent` parameter.
///
/// MultiVectorViewStorage provides no public interface. All access to the
/// underlying data occurs through the `MultiVectorAccess` type.

/// The awkward interface exists because MultiVectorViewStorage is intended to
/// be used as a base class for internal tensorstore types which we want to
/// optimize as small as possible. The class is intended to be used as a base
/// class to take advantage of the empty base class optimization.
///
/// Example usage:
///
///     template <DimensionIndex Rank>
///     class StridedNdRangeView
///         : public MultiVectorViewStorage<Rank, const int, const int,
///                                         const int> {
///       using Access_ = MultiVectorAccess<
///         MultiVectorViewStorage<Rank, const int, const int, const int>>;
///      public:
///       using RankType = typename Access_::ExtentType;
///       StridedNdRangeView(span<const int, Rank> start,
///                          span<const int, Rank> stop,
///                          span<const int, Rank> stride) {
///         Access_::Assign(this, start, stop, stride);
///       }
///       constexpr static RankType rank() const {
///         return Access_::GetExtent(*this);
///       }
///       span<const int, Rank> start() const {
///         return Access_::template get<0>(this);
///       }
///       span<const int, Rank> stop() const {
///         return Access_::template get<1>(this);
///       }
///       span<const int, Rank> stride() const {
///         return Access_::template get<2>(this);
///       }
///     };
template <DimensionIndex Extent, typename... Ts>
class MultiVectorViewStorage;

template <typename StorageT>
class MultiVectorAccess;

/// Specialization of `MultiVectorViewStorage` for `Extent != 0` and `Extent !=
/// dynamic_rank`.
////
/// This specialization stores base pointers for each vector but does not store
/// the extent (since it is a compile-time constant).
template <std::ptrdiff_t Extent, typename... Ts>
class MultiVectorViewStorage {
 private:
  friend class MultiVectorAccess<MultiVectorViewStorage>;
  constexpr static StaticRank<Extent> InternalGetExtent() { return {}; }
  void InternalSetExtent(StaticRank<Extent>) {}
  void* InternalGetDataPointer(std::size_t i) const {
    return const_cast<void*>(data_[i]);
  }
  void InternalSetDataPointer(std::size_t i, const void* ptr) {
    data_[i] = ptr;
  }

  /// Stores the pointers to the arrays as a single array of `const void *`
  /// pointers to simplify the implementation.  `MultiVectorAccess` provides a
  /// type-safe interface.
  const void* data_[sizeof...(Ts)]{};
};

/// Specialization of `MultiVectorViewStorage` for `Extent == 0`.
///
/// This specialization stores neither base pointers nor the extent, and is
/// therefore an empty class.
template <typename... Ts>
class MultiVectorViewStorage<0, Ts...> {
 private:
  friend class MultiVectorAccess<MultiVectorViewStorage>;
  constexpr static StaticRank<0> InternalGetExtent() { return {}; }
  void InternalSetExtent(StaticRank<0>) {}
  void* InternalGetDataPointer(std::size_t i) const { return nullptr; }
  void InternalSetDataPointer(std::size_t i, const void* ptr) {}
};

/// Specialization of `MultiVectorViewStorage` for `Extent == dynamic_rank`.
///
/// This specialization stores base pointers for each vector and the extent.
template <typename... Ts>
class MultiVectorViewStorage<dynamic_rank, Ts...> {
 private:
  friend class MultiVectorAccess<MultiVectorViewStorage>;
  std::ptrdiff_t InternalGetExtent() const { return extent_; }
  void InternalSetExtent(std::ptrdiff_t extent) { extent_ = extent; }
  void* InternalGetDataPointer(std::size_t i) const {
    return const_cast<void*>(data_[i]);
  }
  void InternalSetDataPointer(std::size_t i, const void* ptr) {
    data_[i] = ptr;
  }

  /// Stores the pointers to the arrays as a single array of `const void *`
  /// pointers to simplify the implementation.  `MultiVectorAccess` provides a
  /// type-safe interface.
  const void* data_[sizeof...(Ts)]{};
  std::ptrdiff_t extent_ = 0;
};

/// Friend class of MultiVectorViewStorage that provides the public API.
///
/// \tparam StorageT The MultiVectorViewStorage instance.
///
/// Note that multi_vector.h also specializes the `MultiVectorAccess` class
/// template for instances of MultiVector, and provides the same interface as is
/// provided here for `MultiVectorViewStorage`, so that common functionality is
/// available with the same syntax.
template <DimensionIndex Extent, typename... Ts>
class MultiVectorAccess<MultiVectorViewStorage<Extent, Ts...>> {
 public:
  using StorageType = MultiVectorViewStorage<Extent, Ts...>;

  using ExtentType = StaticOrDynamicRank<Extent>;

  constexpr static std::ptrdiff_t static_extent = Extent;
  constexpr static std::size_t num_vectors = sizeof...(Ts);

  template <std::size_t I>
  using ElementType = TypePackElement<I, Ts...>;
  template <std::size_t I>
  using ConstElementType = TypePackElement<I, Ts...>;

  /// Returns the extent of a MultiVectorViewStorage.
  static ExtentType GetExtent(const StorageType& storage) {
    return storage.InternalGetExtent();
  }

  /// Returns the `I`th vector as a `span`.
  template <std::size_t I>
  static span<ElementType<I>, Extent> get(const StorageType* array) noexcept {
    return {static_cast<ElementType<I>*>(array->InternalGetDataPointer(I)),
            array->InternalGetExtent()};
  }

  /// Assigns a MultiVectorViewStorage to refer to the arrays of length `extent`
  /// indicated by `pointers...`.
  static void Assign(StorageType* array, ExtentType extent, Ts*... pointers) {
    array->InternalSetExtent(extent);
    std::size_t i = 0;
    (array->InternalSetDataPointer(i++, pointers), ...);
  }

  /// Assigns a MultiVectorViewStorage to refer to the arrays indicated by
  /// `spans...`.
  ///
  /// \dchecks `spans.size()`... are all the same.
  static void Assign(StorageType* array, span<Ts, Extent>... spans) {
    const ExtentType extent =
        GetFirstArgument(GetStaticOrDynamicExtent(spans)...);
    assert(((spans.size() == extent) && ...));
    Assign(array, extent, spans.data()...);
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_MULTI_VECTOR_VIEW_H_
