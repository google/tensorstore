// Copyright 2023 The TensorStore Authors
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

#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <type_traits>

#include <benchmark/benchmark.h>
#include "absl/base/attributes.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable_array.h"
#include "tensorstore/internal/nditerable_copy.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

#if defined(__clang__) || defined(__GNUC__)
#define TENSORSTORE_INTERNAL_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define TENSORSTORE_INTERNAL_RESTRICT __restrict
#else
#define TENSORSTORE_INTERNAL_RESTRICT
#endif

namespace {

void DoCopyUnrolled(const uint8_t* TENSORSTORE_INTERNAL_RESTRICT src,
                    uint8_t* TENSORSTORE_INTERNAL_RESTRICT target,
                    int64_t inner_size, int64_t outer_size,
                    ptrdiff_t src_outer_stride, ptrdiff_t target_outer_stride) {
  const auto do_outer_unroll = [&](auto outer_unroll_size) {
    constexpr size_t kNumRows = decltype(outer_unroll_size)::value;
    const uint8_t* src_rows[kNumRows];
    uint8_t* target_rows[kNumRows];
    while (outer_size >= outer_unroll_size) {
      for (size_t row = 0; row < kNumRows; ++row) {
        src_rows[row] = src + src_outer_stride * row;
        target_rows[row] = target + target_outer_stride * row;
      }
      int64_t remaining_inner_size = inner_size;
      ptrdiff_t inner_offset = 0;
      const auto do_inner_unroll = [&](auto inner_unroll_size) {
        for (; remaining_inner_size >= inner_unroll_size;
             remaining_inner_size -= inner_unroll_size,
             inner_offset += inner_unroll_size) {
          for (int row = 0; row < outer_unroll_size; ++row) {
            std::memcpy(target_rows[row] + inner_offset,
                        src_rows[row] + inner_offset, inner_unroll_size);
          }
        }
      };
      do_inner_unroll(std::integral_constant<int64_t, 64>{});
      do_inner_unroll(std::integral_constant<int64_t, 32>{});
      do_inner_unroll(std::integral_constant<int64_t, 16>{});
      do_inner_unroll(std::integral_constant<int64_t, 1>{});
      src += src_outer_stride * outer_unroll_size;
      target += target_outer_stride * outer_unroll_size;
      outer_size -= outer_unroll_size;
    }
  };
  do_outer_unroll(std::integral_constant<int64_t, 8>{});
  do_outer_unroll(std::integral_constant<int64_t, 1>{});
}

void DoCopySimple(const uint8_t* src, uint8_t* target, int64_t inner_size,
                  int64_t outer_size, ptrdiff_t src_outer_stride,
                  ptrdiff_t target_outer_stride) {
  for (int64_t outer_i = 0; outer_i < outer_size; ++outer_i) {
    for (int64_t inner_i = 0; inner_i < inner_size; ++inner_i) {
      target[outer_i * target_outer_stride + inner_i] =
          src[outer_i * src_outer_stride + inner_i];
    }
  }
}

void DoCopySimpleRestrict(const uint8_t* TENSORSTORE_INTERNAL_RESTRICT src,
                          uint8_t* TENSORSTORE_INTERNAL_RESTRICT target,
                          int64_t inner_size, int64_t outer_size,
                          ptrdiff_t src_outer_stride,
                          ptrdiff_t target_outer_stride) {
  for (int64_t outer_i = 0; outer_i < outer_size; ++outer_i) {
    for (int64_t inner_i = 0; inner_i < inner_size; ++inner_i) {
      target[outer_i * target_outer_stride + inner_i] =
          src[outer_i * src_outer_stride + inner_i];
    }
  }
}

void DoCopySimpleRestrictNoBuiltin(
    const uint8_t* TENSORSTORE_INTERNAL_RESTRICT src,
    uint8_t* TENSORSTORE_INTERNAL_RESTRICT target, int64_t inner_size,
    int64_t outer_size, ptrdiff_t src_outer_stride,
    ptrdiff_t target_outer_stride)
#if ABSL_HAVE_ATTRIBUTE(no_builtin)
    __attribute__((no_builtin))
#endif
{
  for (int64_t outer_i = 0; outer_i < outer_size; ++outer_i) {
    for (int64_t inner_i = 0; inner_i < inner_size; ++inner_i) {
      target[outer_i * target_outer_stride + inner_i] =
          src[outer_i * src_outer_stride + inner_i];
    }
  }
}

enum CopyMode {
  kNDIter,
  kUnrolled,
  kSimple,
  kSimpleRestrict,
  kSimpleRestrictNoBuiltin,
  kDataType,
};

template <CopyMode Mode>
void BM_Copy(benchmark::State& state) {
  constexpr int64_t kInnerSizeFactor = 10;
  constexpr int64_t kOuterSizeFactor = 100;

  int outer = state.range(0), inner = state.range(1);
  auto source_array = tensorstore::AllocateArray<uint8_t>(
      {kOuterSizeFactor * kInnerSizeFactor, outer, inner}, tensorstore::c_order,
      tensorstore::value_init);
  auto target_array = tensorstore::AllocateArray<uint8_t>(
      {kOuterSizeFactor, outer, inner * kInnerSizeFactor}, tensorstore::c_order,
      tensorstore::value_init);

  int64_t offset = 0;
  for (auto s : state) {
    tensorstore::internal::Arena arena;
    int64_t outer_offset = offset / kInnerSizeFactor;
    int64_t inner_offset = offset % kInnerSizeFactor;

    auto source_part = tensorstore::SharedSubArray(source_array, {offset});
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto target_part,
        tensorstore::SharedSubArray(target_array, {outer_offset}) |
            tensorstore::Dims(1).TranslateSizedInterval(inner_offset * inner,
                                                        inner) |
            tensorstore::Materialize());
    switch (Mode) {
      case kNDIter: {
        auto source_iterable = GetArrayNDIterable(source_part, &arena);
        auto target_iterable =
            GetTransformedArrayNDIterable(target_part, &arena).value();
        tensorstore::internal::NDIterableCopier copier(
            *source_iterable, *target_iterable, source_part.shape(),
            tensorstore::c_order, &arena);
        TENSORSTORE_CHECK_OK(copier.Copy());
        break;
      }
      case kUnrolled: {
        DoCopyUnrolled(source_part.data(),
                       const_cast<uint8_t*>(target_part.data()), inner, outer,
                       source_part.byte_strides()[0],
                       target_part.byte_strides()[0]);
        break;
      }
      case kSimple: {
        DoCopySimple(source_part.data(),
                     const_cast<uint8_t*>(target_part.data()), inner, outer,
                     source_part.byte_strides()[0],
                     target_part.byte_strides()[0]);
        break;
      }
      case kSimpleRestrict: {
        DoCopySimpleRestrict(source_part.data(),
                             const_cast<uint8_t*>(target_part.data()), inner,
                             outer, source_part.byte_strides()[0],
                             target_part.byte_strides()[0]);
        break;
      }
      case kSimpleRestrictNoBuiltin: {
        DoCopySimpleRestrictNoBuiltin(
            source_part.data(), const_cast<uint8_t*>(target_part.data()), inner,
            outer, source_part.byte_strides()[0],
            target_part.byte_strides()[0]);
        break;
      }
      case kDataType: {
        using ::tensorstore::internal::IterationBufferKind;
        using ::tensorstore::internal::IterationBufferPointer;
        auto input_pointer = IterationBufferPointer(
            source_part.data(), source_part.byte_strides()[0], 1);
        auto output_pointer =
            IterationBufferPointer(const_cast<uint8_t*>(target_part.data()),
                                   target_part.byte_strides()[0], 1);
        tensorstore::dtype_v<uint8_t>->copy_assign
            [IterationBufferKind::kContiguous](nullptr, {outer, inner},
                                               input_pointer, output_pointer,
                                               nullptr);
        break;
      }
    }
    offset = (offset + 1) % (kInnerSizeFactor * kOuterSizeFactor);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * outer *
                          inner);
}

template <typename Bench>
void DefineArgs(Bench* benchmark) {
  benchmark->Args({1, 32 * 1024});
  benchmark->Args({500, 100});
  benchmark->Args({500, 64});
  benchmark->Args({1000, 32});
  benchmark->Args({2000, 16});
}

BENCHMARK(BM_Copy<kNDIter>)->Apply(DefineArgs);
BENCHMARK(BM_Copy<kUnrolled>)->Apply(DefineArgs);
BENCHMARK(BM_Copy<kSimple>)->Apply(DefineArgs);
BENCHMARK(BM_Copy<kSimpleRestrict>)->Apply(DefineArgs);
BENCHMARK(BM_Copy<kSimpleRestrictNoBuiltin>)->Apply(DefineArgs);
BENCHMARK(BM_Copy<kDataType>)->Apply(DefineArgs);

}  // namespace
