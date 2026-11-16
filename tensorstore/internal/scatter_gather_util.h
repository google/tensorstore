// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_SCATTER_GATHER_UTIL_H_
#define TENSORSTORE_INTERNAL_SCATTER_GATHER_UTIL_H_

#include <stddef.h>

#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"

namespace tensorstore {
namespace internal {

/// Forward declaration to avoid a circular dependency on dtype.h
template <typename DataTypeLike>
constexpr const ElementwiseFunction<2, void*>* GetCopyAssignIfNotTrivial(
    DataTypeLike dtype);

/// Unaligned (`alignof == 1`) byte-array wrapper for size-erased trivial
/// element moves without over-alignment or C++ strict-aliasing UB.
template <size_t N>
struct TrivialBytes {
  unsigned char bytes[N];
};
static_assert(sizeof(TrivialBytes<16>) == 16 && alignof(TrivialBytes<16>) == 1);

/// Gathers `count` trivial elements of size `elem_size` bytes (with fast paths
/// for `elem_size` in `{1, 2, 4, 8, 16}`) from offset-indexed positions
/// `base + byte_offsets[i]` into the contiguous buffer `dst` (`dst` and `base`
/// must not overlap).
void GatherIndexedTrivial(const void* base, const Index* byte_offsets,
                          Index count, size_t elem_size, void* dst);

/// Scatters `count` contiguous trivial elements of size `elem_size` bytes (with
/// fast paths for `elem_size` in `{1, 2, 4, 8, 16}`) from `src` into
/// offset-indexed positions `base + byte_offsets[i]` (`src` and `base` must not
/// overlap).
void ScatterIndexedTrivial(const void* src, void* base,
                           const Index* byte_offsets, Index count,
                           size_t elem_size);

/// Gathers `count` trivial elements of size `elem_size` bytes (with fast paths
/// for `elem_size` in `{1, 2, 4, 8, 16}`) from strided positions
/// `row_src + i * byte_stride` into the contiguous buffer `dst` (`dst` and
/// `row_src` must not overlap). Handles `byte_stride == 0` (broadcast),
/// positive/negative strides, and unit strides.
void GatherStridedTrivial(const void* row_src, Index byte_stride, Index count,
                          size_t elem_size, void* dst);

/// Scatters `count` contiguous trivial elements of size `elem_size` bytes (with
/// fast paths for `elem_size` in `{1, 2, 4, 8, 16}`) from `src` into strided
/// positions `row_dst + i * byte_stride` (`src` and `row_dst` must not
/// overlap).
void ScatterStridedTrivial(const void* src, void* row_dst, Index byte_stride,
                           Index count, size_t elem_size);

namespace internal_scatter_gather {

void GatherContiguousRowImpl(Index elem_size,
                             const ElementwiseFunction<2, void*>* copy_assign,
                             IterationBufferKind kind,
                             IterationBufferPointer src, Index row, Index count,
                             void* dst);

const void* EnsureContiguousRowImpl(
    Index elem_size, const ElementwiseFunction<2, void*>* copy_assign,
    IterationBufferKind kind, IterationBufferPointer src, Index row,
    Index count, void* scratch);

void ScatterContiguousRowImpl(Index elem_size,
                              const ElementwiseFunction<2, void*>* copy_assign,
                              IterationBufferKind kind, const void* src,
                              IterationBufferPointer dst, Index row,
                              Index count);

void GatherContiguousBufferImpl(
    Index elem_size, const ElementwiseFunction<2, void*>* copy_assign,
    IterationBufferKind kind, IterationBufferShape shape,
    IterationBufferPointer src, void* dst);

void ScatterContiguousBufferImpl(
    Index elem_size, const ElementwiseFunction<2, void*>* copy_assign,
    IterationBufferKind kind, IterationBufferShape shape, const void* src,
    IterationBufferPointer dst);

}  // namespace internal_scatter_gather

/// Copies `count` elements from outer row `row` of `(kind, src)` into the
/// contiguous buffer `dst` (`dst` and `src` must not overlap).
template <typename DataTypeLike>
inline void GatherContiguousRow(DataTypeLike dtype, IterationBufferKind kind,
                                IterationBufferPointer src, Index row,
                                Index count, void* dst) {
  internal_scatter_gather::GatherContiguousRowImpl(
      dtype.size(), GetCopyAssignIfNotTrivial(dtype), kind, src, row, count,
      dst);
}

/// Returns a pointer to `count` contiguous `dtype` elements for outer row `row`
/// of `(kind, src)`.
///
/// Returns the existing buffer pointer directly (without touching `scratch`)
/// when the row is already contiguous (`kContiguous`, `kStrided` with unit
/// inner byte stride, or `count == 1`); otherwise gathers row `row` into
/// `scratch` and returns `scratch`.
template <typename DataTypeLike>
inline const void* EnsureContiguousRow(DataTypeLike dtype,
                                       IterationBufferKind kind,
                                       IterationBufferPointer src, Index row,
                                       Index count, void* scratch) {
  return internal_scatter_gather::EnsureContiguousRowImpl(
      dtype.size(), GetCopyAssignIfNotTrivial(dtype), kind, src, row, count,
      scratch);
}

/// Copies `count` contiguous elements from `src` into outer row `row` of
/// `(kind, dst)` (`src` and `dst` must not overlap).
template <typename DataTypeLike>
inline void ScatterContiguousRow(DataTypeLike dtype, IterationBufferKind kind,
                                 const void* src, IterationBufferPointer dst,
                                 Index row, Index count) {
  internal_scatter_gather::ScatterContiguousRowImpl(
      dtype.size(), GetCopyAssignIfNotTrivial(dtype), kind, src, dst, row,
      count);
}

/// Copies a 2-d block of shape `shape` from `(kind, src)` into a packed
/// row-major contiguous buffer `dst` (`dst` and `src` must not overlap).
template <typename DataTypeLike>
inline void GatherContiguousBuffer(DataTypeLike dtype, IterationBufferKind kind,
                                   IterationBufferShape shape,
                                   IterationBufferPointer src, void* dst) {
  internal_scatter_gather::GatherContiguousBufferImpl(
      dtype.size(), GetCopyAssignIfNotTrivial(dtype), kind, shape, src, dst);
}

/// Copies a packed row-major contiguous buffer `src` into a 2-d block `(kind,
/// dst)` of shape `shape` (`src` and `dst` must not overlap).
template <typename DataTypeLike>
inline void ScatterContiguousBuffer(DataTypeLike dtype,
                                    IterationBufferKind kind,
                                    IterationBufferShape shape, const void* src,
                                    IterationBufferPointer dst) {
  internal_scatter_gather::ScatterContiguousBufferImpl(
      dtype.size(), GetCopyAssignIfNotTrivial(dtype), kind, shape, src, dst);
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_SCATTER_GATHER_UTIL_H_
