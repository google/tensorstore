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

#include "tensorstore/internal/scatter_gather_util.h"

#include <stddef.h>

#include <cstring>
#include <type_traits>

#include "absl/base/attributes.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/meta/attributes.h"
#include "tensorstore/util/byte_strided_pointer.h"

namespace tensorstore {
namespace internal {
namespace {

template <typename Fn, typename DefaultFn>
void DispatchTrivialSize(size_t elem_size, Fn fn, DefaultFn default_fn) {
  switch (elem_size) {
    case 1:
      fn(std::integral_constant<size_t, 1>{});
      break;
    case 2:
      fn(std::integral_constant<size_t, 2>{});
      break;
    case 4:
      fn(std::integral_constant<size_t, 4>{});
      break;
    case 8:
      fn(std::integral_constant<size_t, 8>{});
      break;
    case 16:
      fn(std::integral_constant<size_t, 16>{});
      break;
    default:
      default_fn();
      break;
  }
}

ABSL_ATTRIBUTE_NOINLINE void GatherIndexedGeneric(
    const char* TENSORSTORE_INTERNAL_RESTRICT base_bytes,
    const Index* TENSORSTORE_INTERNAL_RESTRICT byte_offsets, Index count,
    size_t elem_size, char* TENSORSTORE_INTERNAL_RESTRICT dst_bytes) {
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
  for (Index i = 0; i < count; ++i) {
    std::memcpy(dst_bytes + static_cast<size_t>(i) * elem_size,
                base_bytes + byte_offsets[i], elem_size);
  }
}

ABSL_ATTRIBUTE_NOINLINE void ScatterIndexedGeneric(
    const char* TENSORSTORE_INTERNAL_RESTRICT src_bytes,
    char* TENSORSTORE_INTERNAL_RESTRICT base_bytes,
    const Index* TENSORSTORE_INTERNAL_RESTRICT byte_offsets, Index count,
    size_t elem_size) {
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
  for (Index i = 0; i < count; ++i) {
    std::memcpy(base_bytes + byte_offsets[i],
                src_bytes + static_cast<size_t>(i) * elem_size, elem_size);
  }
}

ABSL_ATTRIBUTE_NOINLINE void GatherStridedGeneric(
    const char* TENSORSTORE_INTERNAL_RESTRICT src_bytes, Index byte_stride,
    Index count, size_t elem_size,
    char* TENSORSTORE_INTERNAL_RESTRICT dst_bytes) {
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
  for (Index i = 0; i < count; ++i) {
    std::memcpy(dst_bytes + static_cast<size_t>(i) * elem_size,
                src_bytes + wrap_on_overflow::Multiply(byte_stride, i),
                elem_size);
  }
}

ABSL_ATTRIBUTE_NOINLINE void ScatterStridedGeneric(
    const char* TENSORSTORE_INTERNAL_RESTRICT src_bytes,
    char* TENSORSTORE_INTERNAL_RESTRICT dst_bytes, Index byte_stride,
    Index count, size_t elem_size) {
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
  for (Index i = 0; i < count; ++i) {
    std::memcpy(dst_bytes + wrap_on_overflow::Multiply(byte_stride, i),
                src_bytes + static_cast<size_t>(i) * elem_size, elem_size);
  }
}

ABSL_ATTRIBUTE_NOINLINE void GatherRowDataTypeOps(
    Index elem_size, const ElementwiseFunction<2, void*>* copy_assign,
    IterationBufferKind kind, IterationBufferPointer src, Index row,
    Index count, void* dst) {
  if (kind == IterationBufferKind::kIndexed) {
    const auto copy_one = (*copy_assign)[IterationBufferKind::kContiguous];
    const Index* TENSORSTORE_INTERNAL_RESTRICT offsets =
        src.byte_offsets +
        wrap_on_overflow::Multiply(src.byte_offsets_outer_stride, row);
    auto* dst_bytes = static_cast<unsigned char*>(dst);
    for (Index i = 0; i < count; ++i) {
      IterationBufferPointer elem_src(src.pointer + offsets[i], 0, elem_size);
      IterationBufferPointer elem_dst(
          dst_bytes + static_cast<size_t>(i) * static_cast<size_t>(elem_size),
          0, elem_size);
      copy_one(nullptr, {1, 1}, elem_src, elem_dst, nullptr);
    }
    return;
  }
  const IterationBufferKind copy_kind =
      (kind == IterationBufferKind::kContiguous ||
       src.inner_byte_stride == elem_size)
          ? IterationBufferKind::kContiguous
          : IterationBufferKind::kStrided;
  IterationBufferPointer row_src(
      src.pointer + wrap_on_overflow::Multiply(src.outer_byte_stride, row), 0,
      copy_kind == IterationBufferKind::kContiguous ? elem_size
                                                    : src.inner_byte_stride);
  IterationBufferPointer row_dst(dst, 0, elem_size);
  (*copy_assign)[copy_kind](nullptr, {1, count}, row_src, row_dst, nullptr);
}

ABSL_ATTRIBUTE_NOINLINE void ScatterRowDataTypeOps(
    Index elem_size, const ElementwiseFunction<2, void*>* copy_assign,
    IterationBufferKind kind, const void* src, IterationBufferPointer dst,
    Index row, Index count) {
  if (kind == IterationBufferKind::kIndexed) {
    const auto copy_one = (*copy_assign)[IterationBufferKind::kContiguous];
    const Index* TENSORSTORE_INTERNAL_RESTRICT offsets =
        dst.byte_offsets +
        wrap_on_overflow::Multiply(dst.byte_offsets_outer_stride, row);
    const auto* src_bytes = static_cast<const unsigned char*>(src);
    for (Index i = 0; i < count; ++i) {
      IterationBufferPointer elem_src(
          const_cast<unsigned char*>(src_bytes) +
              static_cast<size_t>(i) * static_cast<size_t>(elem_size),
          0, elem_size);
      IterationBufferPointer elem_dst(dst.pointer + offsets[i], 0, elem_size);
      copy_one(nullptr, {1, 1}, elem_src, elem_dst, nullptr);
    }
    return;
  }
  ByteStridedPointer<void> dst_row_ptr =
      dst.pointer + wrap_on_overflow::Multiply(dst.outer_byte_stride, row);
  if (kind == IterationBufferKind::kStrided && dst.inner_byte_stride == 0) {
    IterationBufferPointer last_src(
        const_cast<unsigned char*>(static_cast<const unsigned char*>(src)) +
            static_cast<size_t>(count - 1) * static_cast<size_t>(elem_size),
        0, elem_size);
    IterationBufferPointer single_dst(dst_row_ptr, 0, elem_size);
    (*copy_assign)[IterationBufferKind::kContiguous](nullptr, {1, 1}, last_src,
                                                     single_dst, nullptr);
    return;
  }
  const IterationBufferKind copy_kind =
      (kind == IterationBufferKind::kContiguous ||
       dst.inner_byte_stride == elem_size)
          ? IterationBufferKind::kContiguous
          : IterationBufferKind::kStrided;
  IterationBufferPointer row_src(const_cast<void*>(src), 0, elem_size);
  IterationBufferPointer row_dst(dst_row_ptr, 0,
                                 copy_kind == IterationBufferKind::kContiguous
                                     ? elem_size
                                     : dst.inner_byte_stride);
  (*copy_assign)[copy_kind](nullptr, {1, count}, row_src, row_dst, nullptr);
}

bool IsFlatContiguousBlock(IterationBufferKind kind, IterationBufferShape shape,
                           IterationBufferPointer ptr, Index elem_size) {
  if (kind != IterationBufferKind::kContiguous &&
      (kind != IterationBufferKind::kStrided ||
       ptr.inner_byte_stride != elem_size)) {
    return false;
  }
  return shape[0] == 1 || ptr.outer_byte_stride ==
                              wrap_on_overflow::Multiply(shape[1], elem_size);
}

}  // namespace

ABSL_ATTRIBUTE_NOINLINE void GatherIndexedTrivial(
    const void* base, const Index* TENSORSTORE_INTERNAL_RESTRICT byte_offsets,
    Index count, size_t elem_size, void* TENSORSTORE_INTERNAL_RESTRICT dst) {
  if (count <= 0) return;
  const char* TENSORSTORE_INTERNAL_RESTRICT base_bytes =
      static_cast<const char*>(base);
  char* TENSORSTORE_INTERNAL_RESTRICT dst_bytes = static_cast<char*>(dst);
  DispatchTrivialSize(
      elem_size,
      [&](auto elem_bytes) {
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
        for (Index i = 0; i < count; ++i) {
          std::memcpy(dst_bytes + static_cast<size_t>(i) * elem_bytes,
                      base_bytes + byte_offsets[i], elem_bytes);
        }
      },
      [&] {
        GatherIndexedGeneric(base_bytes, byte_offsets, count, elem_size,
                             dst_bytes);
      });
}

ABSL_ATTRIBUTE_NOINLINE void ScatterIndexedTrivial(
    const void* TENSORSTORE_INTERNAL_RESTRICT src, void* base,
    const Index* TENSORSTORE_INTERNAL_RESTRICT byte_offsets, Index count,
    size_t elem_size) {
  if (count <= 0) return;
  const char* TENSORSTORE_INTERNAL_RESTRICT src_bytes =
      static_cast<const char*>(src);
  char* TENSORSTORE_INTERNAL_RESTRICT base_bytes = static_cast<char*>(base);
  DispatchTrivialSize(
      elem_size,
      [&](auto elem_bytes) {
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
        for (Index i = 0; i < count; ++i) {
          std::memcpy(base_bytes + byte_offsets[i],
                      src_bytes + static_cast<size_t>(i) * elem_bytes,
                      elem_bytes);
        }
      },
      [&] {
        ScatterIndexedGeneric(src_bytes, base_bytes, byte_offsets, count,
                              elem_size);
      });
}

ABSL_ATTRIBUTE_NOINLINE void GatherStridedTrivial(
    const void* row_src, Index byte_stride, Index count, size_t elem_size,
    void* TENSORSTORE_INTERNAL_RESTRICT dst) {
  if (count <= 0) return;
  const char* TENSORSTORE_INTERNAL_RESTRICT src_bytes =
      static_cast<const char*>(row_src);
  char* TENSORSTORE_INTERNAL_RESTRICT dst_bytes = static_cast<char*>(dst);
  if (byte_stride == static_cast<Index>(elem_size)) {
    std::memcpy(dst_bytes, src_bytes, static_cast<size_t>(count) * elem_size);
    return;
  }
  if (byte_stride == 0) {
    if (elem_size == 1) {
      std::memset(dst_bytes, static_cast<unsigned char>(*src_bytes),
                  static_cast<size_t>(count));
      return;
    }
    DispatchTrivialSize(
        elem_size,
        [&](auto elem_bytes) {
          constexpr size_t kSize = decltype(elem_bytes)::value;
          TrivialBytes<kSize> val;
          std::memcpy(&val, src_bytes, kSize);
          auto* TENSORSTORE_INTERNAL_RESTRICT out =
              static_cast<TrivialBytes<kSize>*>(dst);
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
          for (Index i = 0; i < count; ++i) {
            out[i] = val;
          }
        },
        [&] {
          GatherStridedGeneric(src_bytes, /*byte_stride=*/0, count, elem_size,
                               dst_bytes);
        });
    return;
  }
  DispatchTrivialSize(
      elem_size,
      [&](auto elem_bytes) {
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
        for (Index i = 0; i < count; ++i) {
          std::memcpy(dst_bytes + static_cast<size_t>(i) * elem_bytes,
                      src_bytes + wrap_on_overflow::Multiply(byte_stride, i),
                      elem_bytes);
        }
      },
      [&] {
        GatherStridedGeneric(src_bytes, byte_stride, count, elem_size,
                             dst_bytes);
      });
}

ABSL_ATTRIBUTE_NOINLINE void ScatterStridedTrivial(
    const void* TENSORSTORE_INTERNAL_RESTRICT src, void* row_dst,
    Index byte_stride, Index count, size_t elem_size) {
  if (count <= 0) return;
  const char* TENSORSTORE_INTERNAL_RESTRICT src_bytes =
      static_cast<const char*>(src);
  char* TENSORSTORE_INTERNAL_RESTRICT dst_bytes = static_cast<char*>(row_dst);
  if (byte_stride == static_cast<Index>(elem_size)) {
    std::memcpy(dst_bytes, src_bytes, static_cast<size_t>(count) * elem_size);
    return;
  }
  if (byte_stride == 0) {
    std::memcpy(dst_bytes,
                src_bytes + static_cast<size_t>(count - 1) * elem_size,
                elem_size);
    return;
  }
  DispatchTrivialSize(
      elem_size,
      [&](auto elem_bytes) {
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
        for (Index i = 0; i < count; ++i) {
          std::memcpy(dst_bytes + wrap_on_overflow::Multiply(byte_stride, i),
                      src_bytes + static_cast<size_t>(i) * elem_bytes,
                      elem_bytes);
        }
      },
      [&] {
        ScatterStridedGeneric(src_bytes, dst_bytes, byte_stride, count,
                              elem_size);
      });
}

namespace internal_scatter_gather {

void GatherContiguousRowImpl(Index elem_size,
                             const ElementwiseFunction<2, void*>* copy_assign,
                             IterationBufferKind kind,
                             IterationBufferPointer src, Index row, Index count,
                             void* dst) {
  if (count <= 0) return;
  if (copy_assign != nullptr) {
    GatherRowDataTypeOps(elem_size, copy_assign, kind, src, row, count, dst);
    return;
  }
  if (kind == IterationBufferKind::kIndexed) {
    const Index* offsets =
        src.byte_offsets +
        wrap_on_overflow::Multiply(src.byte_offsets_outer_stride, row);
    GatherIndexedTrivial(src.pointer.get(), offsets, count,
                         static_cast<size_t>(elem_size), dst);
    return;
  }
  const char* row_src = static_cast<const char*>(src.pointer.get()) +
                        wrap_on_overflow::Multiply(src.outer_byte_stride, row);
  const Index byte_stride = (kind == IterationBufferKind::kContiguous)
                                ? elem_size
                                : src.inner_byte_stride;
  GatherStridedTrivial(row_src, byte_stride, count,
                       static_cast<size_t>(elem_size), dst);
}

const void* EnsureContiguousRowImpl(
    Index elem_size, const ElementwiseFunction<2, void*>* copy_assign,
    IterationBufferKind kind, IterationBufferPointer src, Index row,
    Index count, void* scratch) {
  if (count <= 0) return scratch;
  if (kind == IterationBufferKind::kContiguous ||
      (kind == IterationBufferKind::kStrided &&
       (count == 1 || src.inner_byte_stride == elem_size))) {
    return static_cast<const unsigned char*>(src.pointer.get()) +
           wrap_on_overflow::Multiply(src.outer_byte_stride, row);
  }
  if (kind == IterationBufferKind::kIndexed && count == 1) {
    const Index offset = src.byte_offsets[wrap_on_overflow::Multiply(
        src.byte_offsets_outer_stride, row)];
    return static_cast<const unsigned char*>(src.pointer.get()) + offset;
  }
  GatherContiguousRowImpl(elem_size, copy_assign, kind, src, row, count,
                          scratch);
  return scratch;
}

void ScatterContiguousRowImpl(Index elem_size,
                              const ElementwiseFunction<2, void*>* copy_assign,
                              IterationBufferKind kind, const void* src,
                              IterationBufferPointer dst, Index row,
                              Index count) {
  if (count <= 0) return;
  if (copy_assign != nullptr) {
    ScatterRowDataTypeOps(elem_size, copy_assign, kind, src, dst, row, count);
    return;
  }
  if (kind == IterationBufferKind::kIndexed) {
    const Index* offsets =
        dst.byte_offsets +
        wrap_on_overflow::Multiply(dst.byte_offsets_outer_stride, row);
    ScatterIndexedTrivial(src, dst.pointer.get(), offsets, count,
                          static_cast<size_t>(elem_size));
    return;
  }
  char* row_dst = static_cast<char*>(dst.pointer.get()) +
                  wrap_on_overflow::Multiply(dst.outer_byte_stride, row);
  const Index byte_stride = (kind == IterationBufferKind::kContiguous)
                                ? elem_size
                                : dst.inner_byte_stride;
  ScatterStridedTrivial(src, row_dst, byte_stride, count,
                        static_cast<size_t>(elem_size));
}

void GatherContiguousBufferImpl(
    Index elem_size, const ElementwiseFunction<2, void*>* copy_assign,
    IterationBufferKind kind, IterationBufferShape shape,
    IterationBufferPointer src, void* dst) {
  if (shape[0] <= 0 || shape[1] <= 0) return;
  if (IsFlatContiguousBlock(kind, shape, src, elem_size)) {
    GatherContiguousRowImpl(elem_size, copy_assign,
                            IterationBufferKind::kContiguous, src, /*row=*/0,
                            shape[0] * shape[1], dst);
    return;
  }
  auto* dst_bytes = static_cast<unsigned char*>(dst);
  const size_t row_bytes =
      static_cast<size_t>(shape[1]) * static_cast<size_t>(elem_size);
  for (Index row = 0; row < shape[0]; ++row) {
    GatherContiguousRowImpl(elem_size, copy_assign, kind, src, row, shape[1],
                            dst_bytes + static_cast<size_t>(row) * row_bytes);
  }
}

void ScatterContiguousBufferImpl(
    Index elem_size, const ElementwiseFunction<2, void*>* copy_assign,
    IterationBufferKind kind, IterationBufferShape shape, const void* src,
    IterationBufferPointer dst) {
  if (shape[0] <= 0 || shape[1] <= 0) return;
  if (IsFlatContiguousBlock(kind, shape, dst, elem_size)) {
    ScatterContiguousRowImpl(elem_size, copy_assign,
                             IterationBufferKind::kContiguous, src, dst,
                             /*row=*/0, shape[0] * shape[1]);
    return;
  }
  const auto* src_bytes = static_cast<const unsigned char*>(src);
  const size_t row_bytes =
      static_cast<size_t>(shape[1]) * static_cast<size_t>(elem_size);
  for (Index row = 0; row < shape[0]; ++row) {
    ScatterContiguousRowImpl(elem_size, copy_assign, kind,
                             src_bytes + static_cast<size_t>(row) * row_bytes,
                             dst, row, shape[1]);
  }
}

}  // namespace internal_scatter_gather
}  // namespace internal
}  // namespace tensorstore
