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

#ifndef TENSORSTORE_INTERNAL_MULTI_VECTOR_H_
#define TENSORSTORE_INTERNAL_MULTI_VECTOR_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

#include "tensorstore/internal/gdb_scripting.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/internal/multi_vector_impl.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"

TENSORSTORE_GDB_AUTO_SCRIPT("multi_vector_gdb.py")

namespace tensorstore {
namespace internal {

template <std::ptrdiff_t Extent, std::ptrdiff_t InlineSize, typename... Ts>
class MultiVectorStorageImpl;

/// MultiVectorStorage stores multiple one-dimensional arrays, all of the same
/// length, one array for each of the `Ts...`.
///
/// Currently each of the types must be trivial.
///
/// Like `tensorstore::span`, MultiVectorStorage allows both static and dynamic
/// array lengths via the `Extent` parameter.

/// When `Extent = dynamic_rank - n`, for `n >= 0`, the arrays are either stored
/// inline (if the actual extent is `<= n`) or in a single dynamically allocated
/// block.

/// MultiVectorStorage provides no public interface. All access to the
/// underlying data occurs through the `MultiVectorAccess` type.

/// The awkward interface exists because MultiVectorStorage is intended to be
/// used as a base class for internal tensorstore types which we want to
/// optimize as small as possible. The class is intended to be used as a base
/// class to take advantage of the empty base class optimization.
///
/// Example usage:
///
///     template <DimensionIndex Rank>
///     class StridedNdRange
///         : public MultiVectorStorage<Rank, int, int> {
///       using Access_ = MultiVectorAccess<
///         MultiVectorStorage<Rank, int, int, int>>;
///      public:
///       using RankType = typename Access_::ExtentType;
///       constexpr static DimensionIndex static_rank = Access_::static_rank;
///       StridedNdRange(RankType rank = RankType{}) {
///         set_rank(rank);
///       }
///       StridedNdRange(span<int, static_rank> start,
///                      span<int, static_rank> stop,
///                      span<int, static_rank> stride) {
///         Access_::Assign(this, start, stop, stride);
///       }
///       constexpr static RankType rank() const {
///         return Access_::GetExtent(*this);
///       }
///       void set_rank(RankType rank) {
///         Access_::Resize(this, rank);
///       }
///       span<int, static_rank> start() {
///         return Access_::template get<0>(this);
///       }
///       span<const int, static_rank> start() const {
///         return Access_::template get<0>(this);
///       }
///       span<int, static_rank> stop() {
///         return Access_::template get<1>(this);
///       }
///       span<const int, static_rank> stop() const {
///         return Access_::template get<1>(this);
///       }
///       span<int, static_rank> stride() {
///         return Access_::template get<2>(this);
///       }
///       span<const int, static_rank> stride() const {
///         return Access_::template get<2>(this);
///       }
///     };
template <std::ptrdiff_t Extent, typename... Ts>
using MultiVectorStorage =
    MultiVectorStorageImpl<RankConstraint::FromInlineRank(Extent),
                           InlineRankLimit(Extent), Ts...>;

template <typename StorageT>
class MultiVectorAccess;

/// Specialization of `MultiVectorStorageImpl` for a static `Extent > 0`.
template <std::ptrdiff_t Extent, std::ptrdiff_t InlineSize, typename... Ts>
class MultiVectorStorageImpl {
 private:
  static_assert((... && std::is_trivial_v<Ts>),
                "Non-trivial types are not currently supported.");
  static_assert(InlineSize == 0,
                "InlineSize must be 0 if Extent != dynamic_extent.");
  using Offsets = internal_multi_vector::PackStorageOffsets<Ts...>;

  friend class MultiVectorAccess<MultiVectorStorageImpl>;
  void* InternalGetDataPointer(std::size_t array_i) {
    return data_ + Offsets::GetVectorOffset(Extent, array_i);
  }
  constexpr static StaticRank<Extent> InternalGetExtent() { return {}; }

  /// Resizes the vectors.  This is a no op.
  void InternalResize(StaticRank<Extent>) {}

  alignas(Offsets::kAlignment) char data_[Offsets::GetTotalSize(Extent)];
};

/// Specialization of `MultiVectorStorageImpl` for a static `Extent == 0`.
///
/// This is an empty class.
template <std::ptrdiff_t InlineSize, typename... Ts>
class MultiVectorStorageImpl<0, InlineSize, Ts...> {
 private:
  static_assert(InlineSize == 0,
                "InlineSize must be 0 if Extent != dynamic_extent.");
  friend class MultiVectorAccess<MultiVectorStorageImpl>;
  void* InternalGetDataPointer(std::size_t array_i) { return nullptr; }
  constexpr static StaticRank<0> InternalGetExtent() { return {}; }

  /// Resizes the vectors.  This is a no op.
  void InternalResize(StaticRank<0>) {}
};

/// Specialization of `MultiVectorStorageImpl` for a dynamic extent.
template <std::ptrdiff_t InlineSize, typename... Ts>
class MultiVectorStorageImpl<dynamic_rank, InlineSize, Ts...> {
  static_assert((std::is_trivial_v<Ts> && ...),
                "Non-trivial types are not currently supported.");
  static_assert(InlineSize >= 0, "InlineSize must be non-negative.");
  using Offsets = internal_multi_vector::PackStorageOffsets<Ts...>;

 public:
  explicit constexpr MultiVectorStorageImpl() noexcept {}
  MultiVectorStorageImpl(MultiVectorStorageImpl&& other) {
    *this = std::move(other);
  }
  MultiVectorStorageImpl(const MultiVectorStorageImpl& other) { *this = other; }

  MultiVectorStorageImpl& operator=(MultiVectorStorageImpl&& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(extent_, other.extent_);
    return *this;
  }
  MultiVectorStorageImpl& operator=(const MultiVectorStorageImpl& other) {
    if (this == &other) return *this;
    const std::ptrdiff_t extent = other.extent_;
    InternalResize(extent);
    const bool use_inline = InlineSize > 0 && extent <= InlineSize;
    std::memcpy(use_inline ? data_.inline_data : data_.pointer,
                use_inline ? other.data_.inline_data : other.data_.pointer,
                Offsets::GetTotalSize(extent));
    return *this;
  }
  ~MultiVectorStorageImpl() {
    if (extent_ > InlineSize) {
      ::operator delete(data_.pointer);
    }
  }

 private:
  friend class MultiVectorAccess<MultiVectorStorageImpl>;
  std::ptrdiff_t InternalGetExtent() const { return extent_; }
  void* InternalGetDataPointer(std::ptrdiff_t array_i) {
    return (extent_ > InlineSize ? data_.pointer : data_.inline_data) +
           Offsets::GetVectorOffset(extent_, array_i);
  }

  /// Resizes the vectors to the specified size.
  void InternalResize(std::ptrdiff_t new_extent) {
    assert(new_extent >= 0);
    if (extent_ == new_extent) return;
    if (new_extent > InlineSize) {
      void* new_data = ::operator new(Offsets::GetTotalSize(new_extent));
      if (extent_ > InlineSize) ::operator delete(data_.pointer);
      data_.pointer = static_cast<char*>(new_data);
    } else if (extent_ > InlineSize) {
      ::operator delete(data_.pointer);
    }
    extent_ = new_extent;
  }

  constexpr static std::ptrdiff_t kAlignment =
      InlineSize == 0 ? 1 : Offsets::kAlignment;
  constexpr static std::ptrdiff_t kInlineBytes =
      InlineSize == 0 ? 1 : Offsets::GetTotalSize(InlineSize);
  union Data {
    char* pointer;
    alignas(kAlignment) char inline_data[kInlineBytes];
  };
  Data data_;
  std::ptrdiff_t extent_ = 0;
};

/// Friend class of MultiVectorStorage that provides the public API.
///
/// \tparam StorageT The MultiVectorStorage instance.
///
/// Note that multi_vector_view.h also specializes the `MultiVectorAccess` class
/// template for instances of `MultiVectorViewStorage`, so that common
/// functionality is available with the same syntax.  The common interface for
/// `MultiVectorStorage` and `MultiVectorViewStorage` is as follows:
///
///   using StorageType = T;
///   using ExtentType = ...;
///   constexpr static std::ptrdiff_t static_extent = ...;
///
///   template <std::size_t I>
///   using ElementType = ...;
///
///   template <std::size_t I>
///   using ConstElementType = ...;
///
///   static ExtentType GetExtent(const StorageType&);
///
///   template <std::size_t I>
///   static span<ElementType<I>, static_extent> get(StorageType*);
///
///   template <std::size_t I>
///   static span<ConstElementType<I>, static_extent> get(const StorageType*);
///
///   static void Assign(StorageType*, ExtentType extent, Ts*... pointers);
///
///   static void Assign(StorageType*, span<Ts,static_extent>*... pointers);
///
/// The interface provided for `MultiVectorStorage` is a superset of the
/// interface provided for `MultiVectorViewStorage`: for
/// `MultiVectorViewStorage`, there is an additional `Resize` method and the
/// `Assign` methods allow element type conversions.
template <std::ptrdiff_t Extent, std::ptrdiff_t InlineSize, typename... Ts>
class MultiVectorAccess<MultiVectorStorageImpl<Extent, InlineSize, Ts...>> {
 public:
  using StorageType = MultiVectorStorageImpl<Extent, InlineSize, Ts...>;

  using ExtentType = StaticOrDynamicRank<Extent>;

  constexpr static std::ptrdiff_t static_extent = Extent;
  constexpr static std::size_t num_vectors = sizeof...(Ts);

  template <std::size_t I>
  using ElementType = TypePackElement<I, Ts...>;
  template <std::size_t I>
  using ConstElementType = const TypePackElement<I, Ts...>;

  /// Returns the extent of a MultiVectorStorageImpl.
  static ExtentType GetExtent(const StorageType& storage) {
    return storage.InternalGetExtent();
  }

  //// Returns the `I`th vector of a non-const `MultiVectorStorage` as a `span`.
  template <std::size_t I>
  static span<ElementType<I>, Extent> get(StorageType* array) {
    return {static_cast<ElementType<I>*>(array->InternalGetDataPointer(I)),
            GetExtent(*array)};
  }

  /// Returns the `I`th vector of a const `MultiVectorStorage` as a `span`.
  template <std::size_t I>
  static span<ConstElementType<I>, Extent> get(const StorageType* array) {
    return get<I>(const_cast<StorageType*>(array));
  }

  /// Copies the arrays of length `extent` pointed to by `pointers...` to
  /// `*array`, resizing it if necessary.
  ///
  /// The number of pointers must equal the number of `Ts...`, and the types
  /// `Ts...` must be assignable from the types `Us...`.
  template <typename... Us>
  static void Assign(StorageType* array, ExtentType extent, Us*... pointers) {
    static_assert(sizeof...(Us) == sizeof...(Ts));
    array->InternalResize(extent);
    std::size_t vector_i = 0;
    (std::copy_n(pointers, extent,
                 static_cast<Ts*>(array->InternalGetDataPointer(vector_i++))),
     ...);
  }

  /// Copies the arrays indicated by `spans...` to a `MultiVectorStorage`,
  /// resizing it if necessary.
  ///
  /// The number of pointers must equal the number of `Ts...`, and the types
  /// `Ts...` must be assignable from the types `Us...`.
  ///
  /// \dchecks `spans.size()`... are all the same.
  template <typename... Us, std::ptrdiff_t... Extents>
  static void Assign(StorageType* array, span<Us, Extents>... spans) {
    static_assert(sizeof...(Us) == sizeof...(Ts));
    const ExtentType extent =
        GetFirstArgument(GetStaticOrDynamicExtent(spans)...);
    assert(((spans.size() == extent) && ...));
    Assign(array, extent, spans.data()...);
  }

  /// Resizes a MultiVectorStorage.  If the size changes, the new contents are
  /// unspecified.
  static void Resize(StorageType* array, ExtentType new_extent) {
    array->InternalResize(new_extent);
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_MULTI_VECTOR_H_
