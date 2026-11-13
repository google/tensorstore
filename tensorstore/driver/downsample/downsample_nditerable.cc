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

#include "tensorstore/driver/downsample/downsample_nditerable.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include <nlohmann/json.hpp>
#include "tensorstore/box.h"
#include "tensorstore/data_type.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/downsample_method.h"
#include "tensorstore/driver/downsample/downsample_kernels.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/insertion_sort.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_buffer_management.h"
#include "tensorstore/internal/scatter_gather_util.h"
#include "tensorstore/internal/unique_with_intrusive_allocator.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/generic_stringify.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/utf8_string.h"

// Uncomment the line below for debug logging.
// #define TENSORSTORE_INTERNAL_DOWNSAMPLE_DEBUG 1

#ifndef TENSORSTORE_INTERNAL_DOWNSAMPLE_DEBUG
#define TENSORSTORE_INTERNAL_DOWNSAMPLE_DEBUG 0
#endif

namespace tensorstore {
namespace internal_downsample {
namespace {

using ::tensorstore::internal::AddSaturate;
using ::tensorstore::internal::ArenaAllocator;
using ::tensorstore::internal::IterationBufferKind;
using ::tensorstore::internal::IterationBufferPointer;
using ::tensorstore::internal::MulSaturate;
using ::tensorstore::internal::NDIterable;
using ::tensorstore::internal::NDIterator;

// Downsampling is split into four layers (1-d helpers and kernels live in
// `downsample_kernels.h`) to avoid a
// `DownsampleMethod x DataType x IterationBufferKind` template cross-product:
//
// 1. Non-templated 2-d block drivers (`ProcessInputBlock` and
//    `ComputeOutputBlock` on `DownsampledNDIterator`) partition clipped blocks
//    via `DimPartition` and normalize `kStrided`/`kIndexed` buffers via
//    `EnsureContiguousRow` / `ScatterContiguousRow` (`nditerable_util.h`).
// 2. Byte-width `StoreSegmentTrivial` and `dtype->copy_assign`-backed
//    `StoreSegmentNonTrivial` stage `kMedian`/`kMode` windows with zero
//    per-type staging templates.
// 3. `ComputeTypeTraits` promotes low-precision floats to `float32_t` and
//    low-precision integers (`int2_t`, `int4_t`) to `int8_t` via
//    `GetDataTypeConverter`, and aliases layout-identical `char_t` and `byte_t`
//    with zero conversion.
// 4. 1-d contiguous row kernels (`AccumulateRow`, `FinalizeMean*Row`,
//    order-isomorphic unsigned `ComputeMedianRow`/`ComputeModeRow`, and
//    in-place `ShellSort`-backed `ComputeModeGenericRow`) execute only on
//    canonical widths.

// Output elements processed per `GetDataTypeConverter` tile when promoting
// staged `kMedian`/`kMode` windows; bounds `tile_compute_scratch_` to L1 cache
// while amortizing indirect converter dispatch.
constexpr Index kTileWidth = 64;

// Maximum alignment of any trivial accumulator or compute element (covers up to
// 16-byte `absl::{u,}int128` and `complex128_t`).
constexpr size_t kMaxTrivialAlign = 16;

constexpr size_t SaturatingToSizeT(Index n) {
  assert(n >= 0);
  if (static_cast<uint64_t>(n) > std::numeric_limits<size_t>::max()) {
    return std::numeric_limits<size_t>::max();
  }
  return static_cast<size_t>(n);
}

constexpr size_t AlignUp(size_t n, size_t align) {
  return AddSaturate(n, align - 1) & ~(align - 1);
}

/// `bool`-valued metafunction that evaluates to `true` for bool, integer, and
/// floating-point data types.
template <typename Element>
struct IsOrderingSupported {
  static constexpr bool value = false;
};

#define TENSORSTORE_INTERNAL_SPECIALIZE_ORDERING_SUPPORTED(T, ...) \
  template <>                                                      \
  struct IsOrderingSupported<::tensorstore::dtypes::T> {           \
    static constexpr bool value = true;                            \
  };
TENSORSTORE_FOR_EACH_BOOL_DATA_TYPE(
    TENSORSTORE_INTERNAL_SPECIALIZE_ORDERING_SUPPORTED)
TENSORSTORE_FOR_EACH_INT_DATA_TYPE(
    TENSORSTORE_INTERNAL_SPECIALIZE_ORDERING_SUPPORTED)
TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE(
    TENSORSTORE_INTERNAL_SPECIALIZE_ORDERING_SUPPORTED)
#undef TENSORSTORE_INTERNAL_SPECIALIZE_ORDERING_SUPPORTED

template <typename Element>
constexpr bool kIsMeanSupported =
    IsOrderingSupported<Element>::value || IsComplex<Element>;

/// Initialization strategy for `accumulate_buffer_` prior to `GetBlock`
/// accumulation.
enum class InitKind : uint8_t {
  kNone,     // Elements are overwritten before read (`kMedian`, `kMode`).
  kZero,     // Zero-fill (integer `kMean`, and `kMax` on unsigned / `bool`).
  kPattern,  // Fill 1/2/4/8/16-byte identity bit pattern (float/complex
             // `kMean`, `kMin`, signed/float `kMax`).
};

/// Output finalization path in `DownsampledNDIterator::ComputeOutputBlock`.
enum class OutputKind : uint8_t {
  kCopyAccumulate,  // Accumulator already holds final compute values
                    // (`kMin`/`kMax`).
  kFinalizeMean,    // Divide sums by window divisor (`kMean`, `bool`
                    // `kMedian`/`kMode`).
  kReduceStored,    // Reduce staged element windows (`kMedian`, `kMode`).
};

/// Compact descriptor of canonical 1-d kernels and buffer metadata for a
/// `(DownsampleMethod, DataTypeId)` pair.  `accumulate_size == 0` indicates an
/// unsupported combination.  Stores `DataTypeId` + `compute_size` rather than a
/// `DataType` pointer so `kDownsampleFunctions` requires zero dynamic symbol
/// relocations.
struct DownsampleFunctions {
  uint64_t init_bits = 0;
  AccumulateRowFn accumulate_row = nullptr;
  FinalizeMeanRowFn finalize_mean_row = nullptr;
  ReduceStoredRowFn reduce_stored_row = nullptr;
  DataTypeId compute_dtype_id = DataTypeId::custom;
  uint8_t compute_size = 0;
  uint8_t accumulate_size = 0;
  InitKind init_kind = InitKind::kNone;
  OutputKind output_kind = OutputKind::kCopyAccumulate;
  bool store_all_elements = false;
  bool is_promoted = false;
  bool is_trivial = true;
  bool replace_inf_with_nan = false;
};

/// Returns `(InitKind, bit_pattern)` for `kMin` (+inf or type max) and `kMax`
/// (-inf or type lowest) accumulator initialization.
template <typename ComputeElement>
constexpr std::pair<InitKind, uint64_t> GetMinInitSpec() {
  if constexpr (std::is_same_v<ComputeElement, bool>) {
    return {InitKind::kPattern, 1};
  } else if constexpr (std::is_same_v<ComputeElement, float>) {
    static_assert(std::numeric_limits<float>::is_iec559);
    return {InitKind::kPattern, 0x7f800000u};
  } else if constexpr (std::is_same_v<ComputeElement, double>) {
    static_assert(std::numeric_limits<double>::is_iec559);
    return {InitKind::kPattern, 0x7ff0000000000000ULL};
  } else {
    using UInt = std::make_unsigned_t<ComputeElement>;
    return {InitKind::kPattern,
            static_cast<uint64_t>(
                static_cast<UInt>(std::numeric_limits<ComputeElement>::max()))};
  }
}

template <typename ComputeElement>
constexpr std::pair<InitKind, uint64_t> GetMaxInitSpec() {
  if constexpr (std::is_same_v<ComputeElement, bool> ||
                std::is_unsigned_v<ComputeElement>) {
    return {InitKind::kZero, 0};
  } else if constexpr (std::is_same_v<ComputeElement, float>) {
    static_assert(std::numeric_limits<float>::is_iec559);
    return {InitKind::kPattern, 0xff800000u};
  } else if constexpr (std::is_same_v<ComputeElement, double>) {
    static_assert(std::numeric_limits<double>::is_iec559);
    return {InitKind::kPattern, 0xfff0000000000000ULL};
  } else {
    using UInt = std::make_unsigned_t<ComputeElement>;
    return {InitKind::kPattern,
            static_cast<uint64_t>(static_cast<UInt>(
                std::numeric_limits<ComputeElement>::lowest()))};
  }
}

template <typename ComputeElement>
constexpr FinalizeMeanRowFn GetFinalizeMeanFn() {
  using Acc = typename MeanAccumulateType<ComputeElement>::type;
  if constexpr (std::is_same_v<ComputeElement, bool>) {
    return &FinalizeMeanBoolRow;
  } else if constexpr (std::is_same_v<ComputeElement, float>) {
    return &FinalizeMeanFloatRow<float>;
  } else if constexpr (std::is_same_v<ComputeElement, double>) {
    return &FinalizeMeanFloatRow<double>;
  } else if constexpr (std::is_same_v<ComputeElement,
                                      ::tensorstore::dtypes::complex64_t>) {
    return &FinalizeMeanComplexRow<float>;
  } else if constexpr (std::is_same_v<ComputeElement,
                                      ::tensorstore::dtypes::complex128_t>) {
    return &FinalizeMeanComplexRow<double>;
  } else {
    return &FinalizeMeanIntRow<Acc, ComputeElement>;
  }
}

template <DownsampleMethod Method, typename Element>
constexpr DownsampleFunctions MakeDownsampleFunctions() {
  using Traits = ComputeTypeTraits<Element>;
  using ComputeElement = typename Traits::ComputeElement;
  constexpr bool kIsTrivial = IsTrivial<Element>;

  DownsampleFunctions fn{};
  fn.compute_dtype_id = DataTypeIdOf<ComputeElement>;
  fn.compute_size = static_cast<uint8_t>(sizeof(ComputeElement));
  fn.is_promoted = Traits::kIsPromoted;
  fn.is_trivial = kIsTrivial;

  // For `bool`, median and mode equal the rounded mean, avoiding element
  // staging and sorting.
  if constexpr (Method == DownsampleMethod::kMean ||
                (std::is_same_v<Element, bool> &&
                 (Method == DownsampleMethod::kMedian ||
                  Method == DownsampleMethod::kMode))) {
    if constexpr (!kIsMeanSupported<Element>) {
      return {};
    } else {
      using Acc = typename MeanAccumulateType<ComputeElement>::type;
      static_assert(alignof(Acc) <= kMaxTrivialAlign);
      static_assert(std::is_trivially_destructible_v<Acc>);
      fn.accumulate_row = &AccumulateRow<SumOp, Acc, ComputeElement>;
      fn.finalize_mean_row = GetFinalizeMeanFn<ComputeElement>();
      fn.accumulate_size = static_cast<uint8_t>(sizeof(Acc));
      // Initialize floating-point `kMean` accumulators to `-0.0` (the IEEE-754
      // additive identity) so an all-`-0.0` window preserves `-0.0`.
      if constexpr (std::is_same_v<Acc, float>) {
        fn.init_kind = InitKind::kPattern;
        fn.init_bits = 0x80000000u;
      } else if constexpr (std::is_same_v<Acc,
                                          ::tensorstore::dtypes::complex64_t>) {
        fn.init_kind = InitKind::kPattern;
        fn.init_bits = 0x8000000080000000ULL;
      } else if constexpr (std::is_same_v<Acc, double> ||
                           std::is_same_v<
                               Acc, ::tensorstore::dtypes::complex128_t>) {
        fn.init_kind = InitKind::kPattern;
        fn.init_bits = 0x8000000000000000ULL;
      } else {
        fn.init_kind = InitKind::kZero;
      }
      fn.output_kind = OutputKind::kFinalizeMean;
      return fn;
    }
  } else if constexpr (Method == DownsampleMethod::kMin ||
                       Method == DownsampleMethod::kMax) {
    if constexpr (!IsOrderingSupported<Element>::value) {
      return {};
    } else {
      static_assert(alignof(ComputeElement) <= kMaxTrivialAlign);
      constexpr auto init_spec = (Method == DownsampleMethod::kMin)
                                     ? GetMinInitSpec<ComputeElement>()
                                     : GetMaxInitSpec<ComputeElement>();
      using Op =
          std::conditional_t<Method == DownsampleMethod::kMin, MinOp, MaxOp>;
      // Canonical C++ `bool` values are 0/1 bytes, so share `uint8_t` SIMD
      // min/max kernels rather than instantiating separate `bool` loops.
      using AccElement =
          std::conditional_t<std::is_same_v<ComputeElement, bool>, uint8_t,
                             ComputeElement>;
      fn.init_bits = init_spec.second;
      fn.accumulate_row = &AccumulateRow<Op, AccElement, AccElement>;
      fn.accumulate_size = static_cast<uint8_t>(sizeof(ComputeElement));
      fn.init_kind = init_spec.first;
      fn.output_kind = OutputKind::kCopyAccumulate;
      // IEEE float types with `has_infinity` retain their `+inf`/`-inf`
      // accumulator identity when an entire window is `NaN`.  Finite-only
      // `float8_*` types (`!has_infinity`) cannot represent infinity, so any
      // untouched `+/-inf` `float32_t` accumulator is replaced with `+qNaN`
      // before narrowing back to `Element`.
      fn.replace_inf_with_nan = Traits::kIsPromoted &&
                                std::is_same_v<ComputeElement, float> &&
                                !std::numeric_limits<Element>::has_infinity;
      return fn;
    }
  } else if constexpr (Method == DownsampleMethod::kMedian) {
    if constexpr (!IsOrderingSupported<Element>::value) {
      return {};
    } else {
      static_assert(kIsTrivial && alignof(Element) <= kMaxTrivialAlign);
      if constexpr (std::is_unsigned_v<ComputeElement>) {
        fn.reduce_stored_row = &ComputeMedianUnsignedRow<ComputeElement>;
      } else {
        fn.reduce_stored_row = &ComputeMedianRow<ComputeElement>;
      }
      fn.accumulate_size = static_cast<uint8_t>(sizeof(Element));
      fn.output_kind = OutputKind::kReduceStored;
      fn.store_all_elements = true;
      return fn;
    }
  } else {
    static_assert(Method == DownsampleMethod::kMode);
    static_assert(!Traits::kIsPromoted || kIsTrivial);
    if constexpr (std::is_unsigned_v<ComputeElement>) {
      fn.reduce_stored_row = &ComputeModeUnsignedRow<ComputeElement>;
    } else if constexpr (std::is_same_v<ComputeElement, Utf8String>) {
      fn.reduce_stored_row = &ComputeModeRow<std::string>;
    } else {
      fn.reduce_stored_row = &ComputeModeRow<ComputeElement>;
    }
    fn.accumulate_size = static_cast<uint8_t>(sizeof(Element));
    fn.output_kind = OutputKind::kReduceStored;
    fn.store_all_elements = true;
    return fn;
  }
}

/// Index of the supported downsampling method, for use in indexing
/// `kDownsampleFunctions`.  Excludes `DownsampleMethod::kStride`, which does
/// not use this NDIterable-based downsampling.
constexpr size_t DownsampleMethodIndex(DownsampleMethod method) {
  assert(method != DownsampleMethod::kStride);
  return static_cast<size_t>(method) -
         static_cast<size_t>(DownsampleMethod::kMean);
}

/// Number of supported downsampling methods.
constexpr size_t kNumDownsampleMethods =
    static_cast<size_t>(DownsampleMethod::kMax) + 1 -
    static_cast<size_t>(DownsampleMethod::kMean);

template <typename Func>
constexpr std::array<
    std::invoke_result_t<Func, std::integral_constant<DownsampleMethod,
                                                      DownsampleMethod::kMean>>,
    kNumDownsampleMethods>
MapDownsampleMethods(Func func) {
  using M = DownsampleMethod;
  return {{
      func(std::integral_constant<M, M::kMean>{}),
      func(std::integral_constant<M, M::kMedian>{}),
      func(std::integral_constant<M, M::kMode>{}),
      func(std::integral_constant<M, M::kMin>{}),
      func(std::integral_constant<M, M::kMax>{}),
  }};
}

template <DownsampleMethod Method>
struct MakeDownsampleFunctionsForMethod {
  template <typename Dtype>
  constexpr DownsampleFunctions operator()(Dtype) const {
    return MakeDownsampleFunctions<Method, typename Dtype::Element>();
  }
};

struct MakeDownsampleFunctionsTable {
  template <typename MethodConst>
  constexpr std::array<DownsampleFunctions, kNumDataTypeIds> operator()(
      MethodConst) const {
    return MapCanonicalDataTypes(
        MakeDownsampleFunctionsForMethod<MethodConst::value>{});
  }
};

/// Array of type-erased `DownsampleFunctions` indexed by `DownsampleMethod` and
/// `DataTypeId`.
constexpr std::array<std::array<DownsampleFunctions, kNumDataTypeIds>,
                     kNumDownsampleMethods>
    kDownsampleFunctions = MapDownsampleMethods(MakeDownsampleFunctionsTable{});

inline const DownsampleFunctions& GetDownsampleFunctions(
    DownsampleMethod downsample_method, DataType dtype) {
  assert(dtype.id() != DataTypeId::custom);
  assert(downsample_method != DownsampleMethod::kStride);
  return kDownsampleFunctions[DownsampleMethodIndex(downsample_method)]
                             [static_cast<int>(dtype.id())];
}

// Writes `init_bits` via `TrivialBytes` rather than `PatternInt*` so subsequent
// typed `float` / `double` / signed-integer `AccumulateRow` reads do not
// violate C++ strict-aliasing TBAA rules.
template <typename PatternInt>
void FillPattern(void* buffer, Index count, uint64_t init_bits) {
  const PatternInt pattern_int = static_cast<PatternInt>(init_bits);
  TrivialBytes<sizeof(PatternInt)> pattern;
  std::memcpy(&pattern, &pattern_int, sizeof(pattern));
  auto* dst = static_cast<TrivialBytes<sizeof(PatternInt)>*>(buffer);
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
  for (Index i = 0; i < count; ++i) {
    dst[i] = pattern;
  }
}

/// Initializes `count` accumulator elements before accumulating input blocks.
void InitializeAccumulateBuffer(const DownsampleFunctions& fn, void* buffer,
                                Index count) {
  switch (fn.init_kind) {
    case InitKind::kNone:
      return;
    case InitKind::kZero:
      std::memset(buffer, 0, static_cast<size_t>(count) * fn.accumulate_size);
      return;
    case InitKind::kPattern:
      switch (fn.accumulate_size) {
        case 1:
          std::memset(buffer, static_cast<uint8_t>(fn.init_bits),
                      static_cast<size_t>(count));
          return;
        case 2:
          FillPattern<uint16_t>(buffer, count, fn.init_bits);
          return;
        case 4:
          FillPattern<uint32_t>(buffer, count, fn.init_bits);
          return;
        case 8:
          FillPattern<uint64_t>(buffer, count, fn.init_bits);
          return;
        case 16: {
          const uint64_t elem_bits = fn.init_bits;
          TrivialBytes<16> pattern;
          std::memcpy(&pattern, &elem_bits, 8);
          std::memcpy(reinterpret_cast<unsigned char*>(&pattern) + 8,
                      &elem_bits, 8);
          auto* dst = static_cast<TrivialBytes<16>*>(buffer);
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
          for (Index i = 0; i < count; ++i) {
            dst[i] = pattern;
          }
          return;
        }
      }
  }
}

/// `NDIterator` implementation returned by `DownsampledNDIterable`.
///
/// This uses an accumulator buffer equal to the block size.  Its `GetBlock`
/// implementation makes one or more calls to the base iterator's `GetBlock`,
/// corresponding to different positions within the downsample block of size
/// given by `downsample_factors`, and adds the blocks to the accumulator
/// buffer.
///
/// The inner-most iteration dimension may itself be downsampled.  In that case,
/// it has to be added to the accumulator buffer specially.
///
/// The base iterator may have a higher rank (`base_rank`) than this iterator
/// (of rank `downsampled_rank`).  In that case, base iteration dimension `i`,
/// for `i >= (base_rank - downsampled_rank)`, corresponds to downsampled
/// iteration dimension `i - (base_rank - downsampled_rank)`, and base iteration
/// dimension `i`, for `i < (base_rank - downsampled_rank)`, corresponds to an
/// implicit singleton downsampled dimension.
class DownsampledNDIterator : public NDIterator::Base<DownsampledNDIterator> {
  /// Returns a function that returns contiguous buffers from `indices_buffer_`.
  auto IndicesGetter() {
    return
        [next_indices = indices_buffer_](DimensionIndex n) mutable -> Index* {
          auto* ret = next_indices;
          next_indices += n;
          return ret;
        };
  }

 public:
  /// Constructs a downsampled iterator.
  ///
  /// \param downsample_functions Dispatch entry for `(method, dtype)`.
  /// \param base_iterable The iterable over the original (not downsampled
  ///     space) to downsample.
  /// \param downsample_factors Array of length `base_layout.full_rank()`
  ///     specifying the downsample factor for each base dimension.
  /// \param base_origin Array of length `base_layout.full_rank()` specifying
  ///     the origin of `base_iterable` relative to a downsample block boundary.
  ///     For all `i`, `base_origin[i]` is guaranteed to be in the half-open
  ///     interval `[0, downsample_factors[i])`.
  /// \param base_layout Iteration layout to use for `base_iterable` computed
  ///     from `layout` by `DownsampledNDIterable::ComputeBaseLayout`.
  /// \param layout Iteration layout for the downsampled output.
  /// \param allocator Allocator to use.
  DownsampledNDIterator(
      const DownsampleFunctions& downsample_functions,
      const NDIterable& base_iterable, const Index* downsample_factors,
      const Index* base_origin,
      internal::NDIterable::IterationBufferKindLayoutView base_layout,
      internal::NDIterable::IterationBufferKindLayoutView layout,
      ArenaAllocator<> allocator)
      : fn_(downsample_functions),
        elem_dtype_(base_iterable.dtype()),
        base_kind_(base_layout.buffer_kind),
        out_kind_(layout.buffer_kind),
        base_(std::array{&base_iterable}, base_layout, allocator) {
    base_iteration_rank_ = base_layout.iteration_rank();
    if (fn_.is_promoted) {
      const DataType compute_dtype =
          kDataTypes[static_cast<size_t>(fn_.compute_dtype_id)];
      to_compute_ =
          internal::GetDataTypeConverter(elem_dtype_, compute_dtype).closure;
      from_compute_ =
          internal::GetDataTypeConverter(compute_dtype, elem_dtype_).closure;
    }

    const Index max_out_h = layout.block_shape[0];
    const Index max_out_w = layout.block_shape[1];
    const Index max_in_w = base_layout.block_shape[1];

    DimensionIndex num_downsample_dims = 0;
    const auto is_downsample_dim = [&](DimensionIndex base_iter_dim) {
      const DimensionIndex dim =
          base_layout.iteration_dimensions[base_iter_dim];
      return dim != -1 && downsample_factors[dim] != 1 &&
             base_layout.iteration_shape[base_iter_dim] > 1;
    };
    // Count the number of downsampled dimensions (`num_downsample_dims` may be
    // 0 when `base_layout.full_rank() > layout.full_rank()` due to extent-1
    // synthetic dimensions added by `PropagateIndexTransformDownsampling`).
    for (DimensionIndex base_iter_dim = 0;
         base_iter_dim < base_layout.iteration_rank(); ++base_iter_dim) {
      if (is_downsample_dim(base_iter_dim)) ++num_downsample_dims;
    }
    assert(num_downsample_dims > 0 ||
           base_layout.full_rank() > layout.full_rank());
    num_downsample_dims_ = num_downsample_dims;

    // `indices_buffer_` packs together the following constant arrays that are
    // initialized below:
    //
    //   Index downsample_dims[num_downsample_dims];
    //
    //     Specifies the index within the base iteration layout (i.e. the
    //     iteration layout used for iterating over `base_iterable`) of each
    //     downsampled dimension.  The relative order is maintained, which we
    //     rely on to ensure that if dimension
    //     `base_layout.iteration_rank() - 1` is downsampled, then it is stored
    //     in `downsample_dims[num_downsample_dims - 1]`.  Note: This array
    //     logically stores `DimensionIndex` values rather than `Index` values.
    //
    //   Index downsample_factors[num_downsample_dims];
    //
    //     Downsample factor (from `downsample_factors`) corresponding to each
    //     iteration dimension in `downsample_dims`.
    //
    //   Index downsample_dim_iteration_shape[num_downsample_dims];
    //
    //     Iteration shape (from `base_layout.iteration_shape`) corresponding to
    //     each iteration dimension in `downsample_dims`.
    //
    //   Index downsample_dim_origin[num_downsample_dims];
    //
    //     Origin (from `base_origin`) corresponding to each iteration dimension
    //     in `downsample_dims`.
    //
    // `indices_buffer_` also contains the following arrays (after the ones
    // listed above) that are used as temporary space by `GetBlock`:
    //
    //   Index base_downsample_dim_offsets[num_downsample_dims];
    //   Index base_downsample_dim_offsets_bounds[num_downsample_dims];
    //   Index initial_base_indices[base_layout.iteration_rank()];
    //   Index base_indices[base_layout.iteration_rank()];
    indices_count_ = static_cast<size_t>(num_downsample_dims * 6 +
                                         2 * base_layout.iteration_rank());
    indices_buffer_ = allocator.arena()->allocate<Index>(indices_count_);
    auto get_indices = IndicesGetter();
    Index* downsample_dims = get_indices(num_downsample_dims);
    Index* downsample_factors_ = get_indices(num_downsample_dims);
    Index* downsample_dim_iteration_shape = get_indices(num_downsample_dims);
    Index* downsample_dim_origin = get_indices(num_downsample_dims);

    // Store the iteration dimension index, downsample factor, and extent for
    // each downsampled dimension.
    DimensionIndex downsample_dim_i = 0;
    Index max_elements_per_window = 1;
    for (DimensionIndex base_iter_dim = 0;
         base_iter_dim < base_layout.iteration_rank(); ++base_iter_dim) {
      if (!is_downsample_dim(base_iter_dim)) continue;
      const DimensionIndex dim =
          base_layout.iteration_dimensions[base_iter_dim];
      downsample_dims[downsample_dim_i] = base_iter_dim;
      const Index factor = downsample_factors_[downsample_dim_i] =
          downsample_factors[dim];
      assert(base_layout.directions[dim] == 1);
      const Index iter_extent =
          downsample_dim_iteration_shape[downsample_dim_i] =
              base_layout.iteration_shape[base_iter_dim];
      max_elements_per_window =
          MulSaturate(max_elements_per_window, std::min(factor, iter_extent));
      downsample_dim_origin[downsample_dim_i] = base_origin[dim];
      ++downsample_dim_i;
    }

    // Sub-allocate the accumulator and any required 1-d row scratch buffers
    // from a single `Arena` allocation.
    const Index block_elems = MulSaturate(max_out_h, max_out_w);
    const Index acc_count =
        fn_.store_all_elements
            ? MulSaturate(block_elems, max_elements_per_window)
            : block_elems;

    if (!fn_.is_trivial) {
      // Non-trivial types (`string_t`, `ustring_t`, `json_t` in `kMode`) are
      // never promoted, so accumulator and optional row scratch buffers all
      // hold `elem_dtype_` elements and share a single construct/destroy span.
      const Index in_elems =
          (base_kind_ != IterationBufferKind::kContiguous) ? max_in_w : 0;
      const Index out_elems =
          (out_kind_ != IterationBufferKind::kContiguous) ? max_out_w : 0;
      const Index total_elems =
          AddSaturate(AddSaturate(acc_count, in_elems), out_elems);
      const size_t elem_size = static_cast<size_t>(elem_dtype_.size());
      alloc_bytes_ = MulSaturate(SaturatingToSizeT(total_elems), elem_size);
      alloc_align_ = static_cast<size_t>(elem_dtype_.alignment());
      non_trivial_count_ = total_elems;
      unsigned char* buf = allocator.arena()->allocate<unsigned char>(
          alloc_bytes_, alloc_align_);
      elem_dtype_->construct(total_elems, buf);
      alloc_ptr_ = buf;
      accumulate_buffer_ = buf;
      if (in_elems != 0) {
        input_row_scratch_ = buf + SaturatingToSizeT(acc_count) * elem_size;
      }
      if (out_elems != 0) {
        compute_row_scratch_ =
            buf + (SaturatingToSizeT(acc_count) + SaturatingToSizeT(in_elems)) *
                      elem_size;
      }
      return;
    }

    // Trivial types pack 16-byte-aligned raw regions with zero virtual calls:
    //   1. `accumulate_buffer_`: accumulator or staged raw elements.
    //   2. `input_row_scratch_`: 1-d contiguous input row (promoted or
    //      gathered/copied from `kStrided`/`kIndexed`).
    //   3. `compute_row_scratch_`: 1-d contiguous `compute_dtype` output row
    //      before narrowing conversion or non-contiguous scatter.
    //   4. `raw_row_scratch_`: raw `elem_dtype_` staging row for promoted
    //      non-contiguous inputs or `kIndexed` outputs.
    //   5. `tile_compute_scratch_`: `kTileWidth` promoted windows for
    //      `kMedian`/`kMode` on low-precision types.
    const size_t elem_size = static_cast<size_t>(elem_dtype_.size());
    const size_t compute_size = static_cast<size_t>(fn_.compute_size);
    const size_t acc_bytes =
        AlignUp(MulSaturate(SaturatingToSizeT(acc_count),
                            static_cast<size_t>(fn_.accumulate_size)),
                kMaxTrivialAlign);

    size_t in_scratch_bytes = 0;
    if (!fn_.store_all_elements) {
      if (fn_.is_promoted || base_kind_ != IterationBufferKind::kContiguous) {
        in_scratch_bytes =
            AlignUp(MulSaturate(SaturatingToSizeT(max_in_w), compute_size),
                    kMaxTrivialAlign);
      }
    } else if (base_kind_ != IterationBufferKind::kContiguous) {
      in_scratch_bytes =
          AlignUp(MulSaturate(SaturatingToSizeT(max_in_w), elem_size),
                  kMaxTrivialAlign);
    }

    size_t compute_scratch_bytes = 0;
    if (fn_.output_kind != OutputKind::kCopyAccumulate &&
        (fn_.is_promoted || out_kind_ != IterationBufferKind::kContiguous)) {
      compute_scratch_bytes =
          AlignUp(MulSaturate(SaturatingToSizeT(max_out_w), compute_size),
                  kMaxTrivialAlign);
    }

    size_t raw_scratch_bytes = 0;
    if (fn_.is_promoted) {
      Index raw_elems = 0;
      if (!fn_.store_all_elements &&
          base_kind_ != IterationBufferKind::kContiguous) {
        raw_elems = std::max(raw_elems, max_in_w);
      }
      if (out_kind_ == IterationBufferKind::kIndexed) {
        raw_elems = std::max(raw_elems, max_out_w);
      }
      raw_scratch_bytes =
          AlignUp(MulSaturate(SaturatingToSizeT(raw_elems), elem_size),
                  kMaxTrivialAlign);
    }

    size_t tile_scratch_bytes = 0;
    if (fn_.is_promoted && fn_.store_all_elements) {
      const size_t tile_elems =
          MulSaturate(SaturatingToSizeT(std::min(max_out_w, kTileWidth)),
                      SaturatingToSizeT(max_elements_per_window));
      tile_scratch_bytes =
          AlignUp(MulSaturate(tile_elems, compute_size), kMaxTrivialAlign);
    }

    alloc_bytes_ = AddSaturate(
        AddSaturate(AddSaturate(acc_bytes, in_scratch_bytes),
                    AddSaturate(compute_scratch_bytes, raw_scratch_bytes)),
        tile_scratch_bytes);
    alloc_align_ = kMaxTrivialAlign;
    unsigned char* buf =
        allocator.arena()->allocate<unsigned char>(alloc_bytes_, alloc_align_);
    alloc_ptr_ = buf;
    accumulate_buffer_ = buf;
    size_t offset = acc_bytes;
    if (in_scratch_bytes != 0) {
      input_row_scratch_ = buf + offset;
      offset += in_scratch_bytes;
    }
    if (compute_scratch_bytes != 0) {
      compute_row_scratch_ = buf + offset;
      offset += compute_scratch_bytes;
    }
    if (raw_scratch_bytes != 0) {
      raw_row_scratch_ = buf + offset;
      offset += raw_scratch_bytes;
    }
    if (tile_scratch_bytes != 0) {
      tile_compute_scratch_ = buf + offset;
    }
  }

  ~DownsampledNDIterator() override {
    if (non_trivial_count_ != 0) {
      elem_dtype_->destroy(non_trivial_count_, alloc_ptr_);
    }
    internal::Arena* arena = get_allocator().arena();
    arena->deallocate<unsigned char>(alloc_ptr_, alloc_bytes_, alloc_align_);
    arena->deallocate<Index>(indices_buffer_, indices_count_);
  }

  ArenaAllocator<> get_allocator() const override {
    return base_.get_allocator();
  }

  bool GetBlock(span<const Index> indices,
                internal::IterationBufferShape block_shape,
                IterationBufferPointer* pointer,
                absl::Status* status) override {
    const DimensionIndex num_downsample_dims = num_downsample_dims_;
    const DimensionIndex base_iteration_rank = base_iteration_rank_;
    const DimensionIndex base_iter_dim_offset =
        base_iteration_rank - indices.size();

    // Extract index vectors from `indices_buffer_`.
    auto get_indices = IndicesGetter();
    const Index* downsample_dims = get_indices(num_downsample_dims);
    const Index* downsample_factors = get_indices(num_downsample_dims);
    const Index* downsample_dim_iteration_shape =
        get_indices(num_downsample_dims);
    const Index* downsample_dim_origin = get_indices(num_downsample_dims);

    // Vector of length `num_outer_downsample_dims` used in the loop below to
    // specify the offset relative to `indices` of the block within `base_` to
    // process/accumulate.  Initialized to all 0 and stepped over
    // `[0, base_downsample_dim_offsets_bounds)` via `internal::AdvanceIndices`.
    Index* base_downsample_dim_offsets = get_indices(num_downsample_dims);

    // Shape of the outer downsample block for `indices`, clipped on both sides
    // to fit within `base_` (`downsample_dim_origin` and
    // `downsample_dim_iteration_shape`).
    Index* base_downsample_dim_offsets_bounds =
        get_indices(num_downsample_dims);

    // Clamped starting iteration position within `base_` for `indices`, and
    // current iteration position stepped by `base_downsample_dim_offsets`.
    Index* initial_base_indices = get_indices(base_iteration_rank);
    Index* base_indices = get_indices(base_iteration_rank);

    // Compute the bounds of the base (input) region over which we will iterate
    // to generate the output buffer.
    Index outer_divisor = 1;
    std::fill_n(initial_base_indices, base_iter_dim_offset, 0);
    std::copy_n(indices.begin(), indices.size(),
                initial_base_indices + base_iter_dim_offset);
    DimensionIndex num_outer_downsample_dims = num_downsample_dims;
    internal::IterationBufferShape base_block_shape = block_shape;
    internal::IterationBufferShape base_block_offset{0, 0};
    internal::IterationBufferShape inner_downsample_factor{1, 1};
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
    for (DimensionIndex i = 0; i < num_downsample_dims; ++i) {
      const Index downsample_factor = downsample_factors[i];
      const DimensionIndex base_dim = downsample_dims[i];
      const Index origin = downsample_dim_origin[i];
      base_downsample_dim_offsets[i] = 0;
      Index& initial_base_index = initial_base_indices[base_dim];
      const Index orig_index = initial_base_index;
      // Compute `orig_index * factor - origin` and `(orig_index + block_len) *
      // factor - origin` via non-negative terms (`0 <= origin < factor`) so
      // arbitrarily large `downsample_factor` values never overflow signed
      // `Index`.
      const Index first_cell_end = downsample_factor - origin;
      const Index clamped_base_inclusive_min =
          (orig_index == 0)
              ? Index(0)
              : AddSaturate(MulSaturate(orig_index - 1, downsample_factor),
                            first_cell_end);
      initial_base_index = clamped_base_inclusive_min;
      if (base_dim >= base_iteration_rank - 2) {
        // This downsampling factor applies to one of the 2 innermost
        // dimensions.
        const DimensionIndex inner_dim_i = base_dim - (base_iteration_rank - 2);
        --num_outer_downsample_dims;
        const Index block_len = block_shape[inner_dim_i];
        const Index unclamped_max =
            (orig_index == 0)
                ? AddSaturate(MulSaturate(block_len - 1, downsample_factor),
                              first_cell_end)
                : AddSaturate(clamped_base_inclusive_min,
                              MulSaturate(block_len, downsample_factor));
        const Index base_exclusive_max =
            std::min(unclamped_max, downsample_dim_iteration_shape[i]);
        inner_downsample_factor[inner_dim_i] = downsample_factor;
        base_block_offset[inner_dim_i] = (orig_index == 0) ? origin : Index(0);
        base_block_shape[inner_dim_i] =
            base_exclusive_max - clamped_base_inclusive_min;
      } else {
        // This downsampling factor applies to an outer dimension.
        const Index unclamped_max =
            (orig_index == 0)
                ? first_cell_end
                : AddSaturate(clamped_base_inclusive_min, downsample_factor);
        const Index base_exclusive_max =
            std::min(unclamped_max, downsample_dim_iteration_shape[i]);
        const Index bound = base_downsample_dim_offsets_bounds[i] =
            base_exclusive_max - clamped_base_inclusive_min;
        outer_divisor = MulSaturate(outer_divisor, bound);
      }
    }
    std::copy_n(initial_base_indices, base_iteration_rank, base_indices);

    InitializeAccumulateBuffer(fn_, accumulate_buffer_,
                               block_shape[0] * block_shape[1]);

    const DimPartition p0 =
        PartitionDim(block_shape[0], base_block_shape[0], base_block_offset[0],
                     inner_downsample_factor[0]);
    const DimPartition p1 =
        PartitionDim(block_shape[1], base_block_shape[1], base_block_offset[1],
                     inner_downsample_factor[1]);
    const Index max_n_0 =
        std::min(inner_downsample_factor[0], base_block_shape[0]);
    const Index max_n_1 =
        std::min(inner_downsample_factor[1], base_block_shape[1]);
    const Index max_total_elements =
        MulSaturate(outer_divisor, MulSaturate(max_n_0, max_n_1));

    // Request and accumulate all of the required base (input) blocks.
    Index prior_accumulate_calls = 0;
    do {
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
      for (DimensionIndex i = 0; i < num_outer_downsample_dims; ++i) {
        const DimensionIndex dim = downsample_dims[i];
        base_indices[dim] =
            initial_base_indices[dim] + base_downsample_dim_offsets[i];
      }
#if TENSORSTORE_INTERNAL_DOWNSAMPLE_DEBUG
      ABSL_LOG(INFO) << "Output block: " << GenericStringify(indices)
                     << ", block_shape=" << block_shape[0] << ","
                     << block_shape[1] << ": Getting base block: "
                     << GenericStringify(span<const Index>(base_indices,
                                                           base_iteration_rank))
                     << " of shape=" << base_block_shape[0] << ","
                     << base_block_shape[1];
#endif
      if (!base_.GetBlock(span<const Index>(base_indices, base_iteration_rank),
                          base_block_shape, status)) {
        return false;
      }
      ProcessInputBlock(block_shape, base_.block_pointers()[0],
                        base_block_shape, inner_downsample_factor, p0, p1,
                        max_total_elements, prior_accumulate_calls);
      ++prior_accumulate_calls;
    } while (internal::AdvanceIndices(num_outer_downsample_dims,
                                      base_downsample_dim_offsets,
                                      base_downsample_dim_offsets_bounds));

    // Compute the downsampled output from the accumulated state.
    ComputeOutputBlock(block_shape, *pointer, inner_downsample_factor, p0, p1,
                       outer_divisor, max_total_elements);
    return true;
  }

 private:
  /// Returns a contiguous `compute_dtype` pointer for input row `src_y` of
  /// width `in_w`.  Zero-copy when `!is_promoted` and row `src_y` is already
  /// contiguous; otherwise gathers and/or promotes into `input_row_scratch_`.
  const unsigned char* PrepareComputeInputRow(IterationBufferPointer src_ptr,
                                              Index src_y, Index in_w) {
    if (!fn_.is_promoted) {
      return static_cast<const unsigned char*>(internal::EnsureContiguousRow(
          elem_dtype_, base_kind_, src_ptr, src_y, in_w, input_row_scratch_));
    }
    const void* raw_row = internal::EnsureContiguousRow(
        elem_dtype_, base_kind_, src_ptr, src_y, in_w, raw_row_scratch_);
    IterationBufferPointer raw_contig(const_cast<void*>(raw_row), 0,
                                      elem_dtype_.size());
    IterationBufferPointer dst_contig(input_row_scratch_, 0, fn_.compute_size);
    (*to_compute_.function)[IterationBufferKind::kContiguous](
        to_compute_.context, {1, in_w}, raw_contig, dst_contig, nullptr);
    return input_row_scratch_;
  }

  /// Returns a contiguous raw `elem_dtype_` pointer for input row `src_y` when
  /// staging elements for `kMedian`/`kMode`.
  const unsigned char* PrepareRawInputRow(IterationBufferPointer src_ptr,
                                          Index src_y, Index in_w) {
    return static_cast<const unsigned char*>(internal::EnsureContiguousRow(
        elem_dtype_, base_kind_, src_ptr, src_y, in_w, input_row_scratch_));
  }

  /// Writes `out_w` computed values from `compute_row_scratch_` to output row
  /// `out_y` when the kernel could not write directly to `out_ptr`
  /// (`is_promoted || out_kind_ != kContiguous`).
  void WriteComputeRowToOutput(IterationBufferPointer out_ptr, Index out_y,
                               Index out_w) {
    if (!fn_.is_promoted) {
      internal::ScatterContiguousRow(
          elem_dtype_, out_kind_, compute_row_scratch_, out_ptr, out_y, out_w);
      return;
    }
    const Index compute_size = fn_.compute_size;
    IterationBufferPointer src_contig(compute_row_scratch_, 0, compute_size);
    if (out_kind_ != IterationBufferKind::kIndexed) {
      IterationBufferPointer dst_row(
          out_ptr.pointer + out_y * out_ptr.outer_byte_stride, 0,
          out_ptr.inner_byte_stride);
      (*from_compute_.function)[out_kind_](from_compute_.context, {1, out_w},
                                           src_contig, dst_row, nullptr);
    } else {
      IterationBufferPointer raw_contig(raw_row_scratch_, 0,
                                        elem_dtype_.size());
      (*from_compute_.function)[IterationBufferKind::kContiguous](
          from_compute_.context, {1, out_w}, src_contig, raw_contig, nullptr);
      internal::ScatterContiguousRow(elem_dtype_, out_kind_, raw_row_scratch_,
                                     out_ptr, out_y, out_w);
    }
  }

  /// Visits every input row `src_y` in sequential memory order across head,
  /// full, and tail outer windows, invoking `callback(out_y, src_y, n_0, e_0)`
  /// where `n_0` is the clipped outer window height for `out_y` and `e_0` is
  /// the index within `[0, n_0)`.
  template <typename Callback>
  static void ForEachOuterRow(Index out_h, Index fac_0, const DimPartition& p0,
                              Callback callback) {
    for (Index e_0 = 0; e_0 < p0.head_src_len; ++e_0) {
      callback(/*out_y=*/Index(0), /*src_y=*/e_0, /*n_0=*/p0.head_src_len, e_0);
    }
    for (Index i = 0; i < p0.full_out_count; ++i) {
      const Index out_y = p0.full_out_begin + i;
      const Index src_y_base = p0.full_src_begin + i * fac_0;
      for (Index e_0 = 0; e_0 < fac_0; ++e_0) {
        callback(out_y, src_y_base + e_0, fac_0, e_0);
      }
    }
    for (Index e_0 = 0; e_0 < p0.tail_src_len; ++e_0) {
      callback(/*out_y=*/out_h - 1, p0.tail_src_begin + e_0, p0.tail_src_len,
               e_0);
    }
  }

  void StoreInputSegment(unsigned char* acc_row, const unsigned char* row_src,
                         Index out_x_begin, Index out_count, Index src_x_begin,
                         Index seg_factor, Index max_total_elements,
                         Index row_call_idx, size_t elem_size_u) const {
    if (fn_.is_trivial) {
      StoreSegmentTrivial(acc_row, row_src, out_x_begin, out_count, src_x_begin,
                          seg_factor, max_total_elements, row_call_idx,
                          elem_size_u);
    } else {
      StoreSegmentNonTrivial(elem_dtype_, acc_row, row_src, out_x_begin,
                             out_count, src_x_begin, seg_factor,
                             max_total_elements, row_call_idx);
    }
  }

  void ReduceStoredSegment(unsigned char* staged_row,
                           unsigned char* dst_compute, Index out_x_begin,
                           Index seg_count, Index divisor,
                           Index max_total_elements, size_t window_stride_bytes,
                           Index elem_size, Index compute_size) {
    const size_t compute_size_u = static_cast<size_t>(compute_size);
    if (!fn_.is_promoted) {
      fn_.reduce_stored_row(
          staged_row + static_cast<size_t>(out_x_begin) * window_stride_bytes,
          max_total_elements,
          dst_compute + static_cast<size_t>(out_x_begin) * compute_size_u,
          seg_count, divisor);
      return;
    }
    // Promote at most `kTileWidth` windows at a time so the promoted staging
    // buffer stays bounded in L1 cache.
    for (Index pos = 0; pos < seg_count; pos += kTileWidth) {
      const Index tile = std::min(kTileWidth, seg_count - pos);
      IterationBufferPointer tile_src(
          staged_row +
              static_cast<size_t>(out_x_begin + pos) * window_stride_bytes,
          static_cast<Index>(window_stride_bytes), elem_size);
      IterationBufferPointer tile_dst(tile_compute_scratch_,
                                      divisor * compute_size, compute_size);
      // Flatten `{tile, divisor}` when windows are packed (`divisor ==
      // max_total_elements`) so `FastLoop` / `ApplyContiguous` converts the
      // entire tile in a single contiguous batch.
      const internal::IterationBufferShape conv_shape =
          (divisor == max_total_elements)
              ? internal::IterationBufferShape{1, tile * divisor}
              : internal::IterationBufferShape{tile, divisor};
      (*to_compute_.function)[IterationBufferKind::kContiguous](
          to_compute_.context, conv_shape, tile_src, tile_dst, nullptr);
      fn_.reduce_stored_row(
          tile_compute_scratch_, divisor,
          dst_compute + static_cast<size_t>(out_x_begin + pos) * compute_size_u,
          tile, divisor);
    }
  }

  /// Accumulates or stages a 2-d input block across `(head, full, tail)` outer
  /// and inner window segments, flattening full contiguous 2-d blocks when
  /// `fac_0 == 1`.
  ABSL_ATTRIBUTE_NOINLINE void ProcessInputBlock(
      internal::IterationBufferShape out_shape, IterationBufferPointer src_ptr,
      internal::IterationBufferShape in_shape,
      internal::IterationBufferShape factor, const DimPartition& p0,
      const DimPartition& p1, Index max_total_elements, Index prior_calls) {
    const Index out_h = out_shape[0];
    const Index out_w = out_shape[1];
    const Index in_h = in_shape[0];
    const Index in_w = in_shape[1];
    const Index fac_0 = factor[0];
    const Index fac_1 = factor[1];
    const Index elem_size = elem_dtype_.size();

    if (!fn_.store_all_elements) {
      // Flatten full contiguous 2-d blocks when outer factor is 1.
      if (!fn_.is_promoted && base_kind_ == IterationBufferKind::kContiguous &&
          fac_0 == 1 && p1.head_src_len == 0 && p1.tail_src_len == 0 &&
          (in_h == 1 || src_ptr.outer_byte_stride == in_w * elem_size)) {
        fn_.accumulate_row(accumulate_buffer_, src_ptr.pointer.get(),
                           out_h * out_w, fac_1);
        return;
      }

      const size_t acc_size = fn_.accumulate_size;
      const size_t compute_size = fn_.compute_size;
      auto* acc_bytes = static_cast<unsigned char*>(accumulate_buffer_);
      ForEachOuterRow(
          out_h, fac_0, p0,
          [&](Index out_y, Index src_y, Index /*n_0*/, Index /*e_0*/) {
            const unsigned char* row_src =
                PrepareComputeInputRow(src_ptr, src_y, in_w);
            unsigned char* acc_row =
                acc_bytes + static_cast<size_t>(out_y) *
                                static_cast<size_t>(out_w) * acc_size;
            if (p1.head_src_len > 0) {
              fn_.accumulate_row(acc_row, row_src, 1, p1.head_src_len);
            }
            if (p1.full_out_count > 0) {
              fn_.accumulate_row(
                  acc_row + static_cast<size_t>(p1.full_out_begin) * acc_size,
                  row_src +
                      static_cast<size_t>(p1.full_src_begin) * compute_size,
                  p1.full_out_count, fac_1);
            }
            if (p1.tail_src_len > 0) {
              fn_.accumulate_row(
                  acc_row + static_cast<size_t>(out_w - 1) * acc_size,
                  row_src +
                      static_cast<size_t>(p1.tail_src_begin) * compute_size,
                  1, p1.tail_src_len);
            }
          });
      return;
    }

    auto* acc_bytes = static_cast<unsigned char*>(accumulate_buffer_);
    const size_t elem_size_u = static_cast<size_t>(elem_size);
    const size_t row_acc_stride = static_cast<size_t>(out_w) *
                                  static_cast<size_t>(max_total_elements) *
                                  elem_size_u;
    ForEachOuterRow(
        out_h, fac_0, p0, [&](Index out_y, Index src_y, Index n_0, Index e_0) {
          const unsigned char* row_src =
              PrepareRawInputRow(src_ptr, src_y, in_w);
          unsigned char* acc_row =
              acc_bytes + static_cast<size_t>(out_y) * row_acc_stride;
          const Index row_call_idx = prior_calls * n_0 + e_0;
          if (p1.head_src_len > 0) {
            StoreInputSegment(acc_row, row_src, 0, 1, 0, p1.head_src_len,
                              max_total_elements, row_call_idx, elem_size_u);
          }
          if (p1.full_out_count > 0) {
            StoreInputSegment(acc_row, row_src, p1.full_out_begin,
                              p1.full_out_count, p1.full_src_begin, fac_1,
                              max_total_elements, row_call_idx, elem_size_u);
          }
          if (p1.tail_src_len > 0) {
            StoreInputSegment(acc_row, row_src, out_w - 1, 1, p1.tail_src_begin,
                              p1.tail_src_len, max_total_elements, row_call_idx,
                              elem_size_u);
          }
        });
  }

  /// Finalizes `accumulate_buffer_` into `out_ptr`, flattening unclipped
  /// contiguous blocks and promoting `kMedian`/`kMode` windows in `kTileWidth`
  /// tiles when `is_promoted` is true.
  ABSL_ATTRIBUTE_NOINLINE void ComputeOutputBlock(
      internal::IterationBufferShape out_shape, IterationBufferPointer out_ptr,
      internal::IterationBufferShape factor, const DimPartition& p0,
      const DimPartition& p1, Index outer_divisor, Index max_total_elements) {
    const Index out_h = out_shape[0];
    const Index out_w = out_shape[1];
    const Index fac_0 = factor[0];
    const Index fac_1 = factor[1];
    const Index elem_size = elem_dtype_.size();
    const Index compute_size = fn_.compute_size;

    const bool flat_contig_out =
        out_kind_ == IterationBufferKind::kContiguous &&
        (out_h == 1 || out_ptr.outer_byte_stride == out_w * elem_size);

    if (fn_.output_kind == OutputKind::kCopyAccumulate) {
      if (fn_.replace_inf_with_nan) {
        auto* acc_f32 = static_cast<float*>(accumulate_buffer_);
        const Index total = out_h * out_w;
        for (Index i = 0; i < total; ++i) {
          if (ABSL_PREDICT_FALSE(std::isinf(acc_f32[i]))) {
            acc_f32[i] = std::numeric_limits<float>::quiet_NaN();
          }
        }
      }
      if (!fn_.is_promoted) {
        internal::ScatterContiguousBuffer(elem_dtype_, out_kind_, out_shape,
                                          accumulate_buffer_, out_ptr);
      } else if (out_kind_ != IterationBufferKind::kIndexed) {
        const internal::IterationBufferShape copy_shape =
            flat_contig_out ? internal::IterationBufferShape{1, out_h * out_w}
                            : out_shape;
        IterationBufferPointer acc_2d(accumulate_buffer_, out_w * compute_size,
                                      compute_size);
        (*from_compute_.function)[out_kind_](from_compute_.context, copy_shape,
                                             acc_2d, out_ptr, nullptr);
      } else {
        auto* acc_bytes = static_cast<unsigned char*>(accumulate_buffer_);
        IterationBufferPointer raw_contig(raw_row_scratch_, 0, elem_size);
        for (Index out_y = 0; out_y < out_h; ++out_y) {
          IterationBufferPointer row_acc(
              acc_bytes + static_cast<size_t>(out_y) *
                              static_cast<size_t>(out_w) *
                              static_cast<size_t>(compute_size),
              0, compute_size);
          (*from_compute_.function)[IterationBufferKind::kContiguous](
              from_compute_.context, {1, out_w}, row_acc, raw_contig, nullptr);
          internal::ScatterContiguousRow(
              elem_dtype_, out_kind_, raw_row_scratch_, out_ptr, out_y, out_w);
        }
      }
      return;
    }

    const bool all_full = p0.head_src_len == 0 && p0.tail_src_len == 0 &&
                          p1.head_src_len == 0 && p1.tail_src_len == 0;
    const bool direct_contig_out =
        !fn_.is_promoted && out_kind_ == IterationBufferKind::kContiguous;
    if (all_full && !fn_.is_promoted && flat_contig_out) {
      if (fn_.output_kind == OutputKind::kFinalizeMean) {
        fn_.finalize_mean_row(accumulate_buffer_, out_ptr.pointer.get(),
                              out_h * out_w, max_total_elements);
      } else {
        fn_.reduce_stored_row(accumulate_buffer_, max_total_elements,
                              out_ptr.pointer.get(), out_h * out_w,
                              max_total_elements);
      }
      return;
    }

    auto* out_bytes = static_cast<unsigned char*>(out_ptr.pointer.get());
    auto* acc_bytes = static_cast<unsigned char*>(accumulate_buffer_);
    const size_t compute_size_u = static_cast<size_t>(compute_size);
    const size_t window_stride_bytes = static_cast<size_t>(max_total_elements) *
                                       static_cast<size_t>(elem_size);
    for (Index out_y = 0; out_y < out_h; ++out_y) {
      const Index n_0 =
          (out_y == 0 && p0.head_src_len > 0)
              ? p0.head_src_len
              : ((out_y == out_h - 1 && p0.tail_src_len > 0) ? p0.tail_src_len
                                                             : fac_0);
      const Index divisor1 = outer_divisor * n_0;
      unsigned char* dst_compute =
          direct_contig_out ? (out_bytes + out_y * out_ptr.outer_byte_stride)
                            : compute_row_scratch_;

      if (fn_.output_kind == OutputKind::kFinalizeMean) {
        const size_t acc_size = fn_.accumulate_size;
        unsigned char* acc_row = acc_bytes + static_cast<size_t>(out_y) *
                                                 static_cast<size_t>(out_w) *
                                                 acc_size;
        if (p1.head_src_len > 0) {
          fn_.finalize_mean_row(acc_row, dst_compute, 1,
                                divisor1 * p1.head_src_len);
        }
        if (p1.full_out_count > 0) {
          fn_.finalize_mean_row(
              acc_row + static_cast<size_t>(p1.full_out_begin) * acc_size,
              dst_compute +
                  static_cast<size_t>(p1.full_out_begin) * compute_size_u,
              p1.full_out_count, divisor1 * fac_1);
        }
        if (p1.tail_src_len > 0) {
          fn_.finalize_mean_row(
              acc_row + static_cast<size_t>(out_w - 1) * acc_size,
              dst_compute + static_cast<size_t>(out_w - 1) * compute_size_u, 1,
              divisor1 * p1.tail_src_len);
        }
      } else {
        unsigned char* staged_row = acc_bytes + static_cast<size_t>(out_y) *
                                                    static_cast<size_t>(out_w) *
                                                    window_stride_bytes;
        if (p1.head_src_len > 0) {
          ReduceStoredSegment(staged_row, dst_compute, 0, 1,
                              divisor1 * p1.head_src_len, max_total_elements,
                              window_stride_bytes, elem_size, compute_size);
        }
        if (p1.full_out_count > 0) {
          ReduceStoredSegment(staged_row, dst_compute, p1.full_out_begin,
                              p1.full_out_count, divisor1 * fac_1,
                              max_total_elements, window_stride_bytes,
                              elem_size, compute_size);
        }
        if (p1.tail_src_len > 0) {
          ReduceStoredSegment(staged_row, dst_compute, out_w - 1, 1,
                              divisor1 * p1.tail_src_len, max_total_elements,
                              window_stride_bytes, elem_size, compute_size);
        }
      }

      if (!direct_contig_out) {
        WriteComputeRowToOutput(out_ptr, out_y, out_w);
      }
    }
  }

  const DownsampleFunctions& fn_;
  DataType elem_dtype_;
  IterationBufferKind base_kind_;
  IterationBufferKind out_kind_;
  internal::ElementwiseClosure<2, void*> to_compute_{};
  internal::ElementwiseClosure<2, void*> from_compute_{};
  internal::NDIteratorsWithManagedBuffers<1> base_;
  DimensionIndex num_downsample_dims_ = 0;
  DimensionIndex base_iteration_rank_ = 0;
  Index* indices_buffer_ = nullptr;
  size_t indices_count_ = 0;
  unsigned char* alloc_ptr_ = nullptr;
  size_t alloc_bytes_ = 0;
  size_t alloc_align_ = 1;
  Index non_trivial_count_ = 0;
  void* accumulate_buffer_ = nullptr;
  unsigned char* input_row_scratch_ = nullptr;
  unsigned char* compute_row_scratch_ = nullptr;
  unsigned char* raw_row_scratch_ = nullptr;
  unsigned char* tile_compute_scratch_ = nullptr;
};

class DownsampledNDIterable : public NDIterable::Base<DownsampledNDIterable> {
 public:
  explicit DownsampledNDIterable(NDIterable::Ptr base, BoxView<> base_domain,
                                 span<const Index> downsample_factors,
                                 DownsampleMethod downsample_method,
                                 DimensionIndex target_rank,
                                 ArenaAllocator<> allocator)
      : downsample_functions_(
            GetDownsampleFunctions(downsample_method, base->dtype())),
        base_(std::array{std::move(base)}),
        base_rank_(downsample_factors.size()),
        target_rank_(target_rank),
        arena_(allocator.arena()),
        indices_buffer_(
            arena_->allocate<Index>(static_cast<size_t>(base_rank_ * 3))) {
    const Index base_rank = base_rank_;
    assert(base_domain.rank() == base_rank);
    for (Index dim = 0; dim < base_rank; ++dim) {
      Index downsample_factor = downsample_factors[dim];
      Index base_size = base_domain.shape()[dim];
      if (base_size <= 1) downsample_factor = 1;
      Index origin_remainder = base_domain.origin()[dim] % downsample_factor;
      if (origin_remainder < 0) origin_remainder += downsample_factor;
      indices_buffer_[dim] = downsample_factor;
      indices_buffer_[dim + base_rank] = base_size;
      indices_buffer_[dim + 2 * base_rank] = origin_remainder;
    }
  }

  ~DownsampledNDIterable() override {
    arena_->deallocate<Index>(indices_buffer_,
                              static_cast<size_t>(base_rank_ * 3));
  }

  int GetDimensionOrder(DimensionIndex dim_i,
                        DimensionIndex dim_j) const override {
    return base_.GetDimensionOrder(dim_i, dim_j);
  }

  bool CanCombineDimensions(DimensionIndex dim_i, int dir_i,
                            DimensionIndex dim_j, int dir_j,
                            Index size_j) const override {
    const Index* downsample_factors = this->downsample_factors();
    return (downsample_factors[dim_i] == 1 && downsample_factors[dim_j] == 1 &&
            base_.CanCombineDimensions(dim_i, dir_i, dim_j, dir_j, size_j));
  }

  void UpdateDirectionPrefs(DirectionPref* prefs) const override {
    // Since `prefs` points to an array of length `target_rank_` and
    // `base_rank_ >= target_rank_`, use a temporary array to query `base_`.
    const DimensionIndex target_rank = target_rank_;
    DirectionPref base_prefs[kMaxRank];
    // `kCanSkip` imposes no constraints; `base_.UpdateDirectionPrefs` adds
    // constraints.
    for (DimensionIndex i = 0; i < base_rank_; ++i) {
      base_prefs[i] = DirectionPref::kCanSkip;
    }
    base_.UpdateDirectionPrefs(&base_prefs[0]);
    const Index* downsample_factors = this->downsample_factors();
    for (DimensionIndex i = 0; i < target_rank; ++i) {
      if (downsample_factors[i] != 1) {
        base_prefs[i] = DirectionPref::kForwardRequired;
      }
      prefs[i] = CombineDirectionPrefs(prefs[i], base_prefs[i]);
    }
  }

  IterationBufferConstraint GetIterationBufferConstraint(
      IterationLayoutView layout) const override {
    // Since we use a separate accumulation buffer, we don't propagate the
    // constraints of `base_`.
    return NDIterable::IterationBufferConstraint{
        /*.min_buffer_kind=*/IterationBufferKind::kContiguous,
        /*.external=*/true};
  }

  class ComputeBaseLayout {
   public:
    explicit ComputeBaseLayout(const DownsampledNDIterable& iterable,
                               NDIterable::IterationLayoutView layout,
                               NDIterable::IterationLayoutView& base_layout) {
      const Index* downsample_factors = iterable.downsample_factors();
      const Index* base_shape = iterable.base_shape();
      const DimensionIndex target_rank = layout.full_rank();
      const DimensionIndex base_rank = iterable.base_rank_;
      const DimensionIndex base_iteration_rank =
          layout.iteration_rank() + base_rank - iterable.target_rank_;
      for (DimensionIndex inner_dim_i = 0; inner_dim_i < 2; ++inner_dim_i) {
        DimensionIndex inner_dim =
            layout.iteration_dimensions[layout.iteration_dimensions.size() - 2 +
                                        inner_dim_i];
        inner_downsample_factor[inner_dim_i] =
            inner_dim == -1 ? 1 : downsample_factors[inner_dim];
      }
      // Iteration dimension `iter_dim` of `DownsampledNDIterator` corresponds
      // to iteration dimension `iter_dim + base_iter_dim_offset` of the base
      // `NDIterator`.
      const DimensionIndex base_iter_dim_offset = base_rank - target_rank;
      for (DimensionIndex iter_dim = 0; iter_dim < layout.iteration_rank();
           ++iter_dim) {
        const DimensionIndex dim = layout.iteration_dimensions[iter_dim];
        // If `downsample_factors[dim] == 1`, `iter_dim` may correspond to
        // multiple original dimensions (permitted by `CanCombineDimensions`),
        // none of which are downsampled, so the base extent equals the
        // downsampled extent.  Otherwise `iter_dim` corresponds to a single
        // original dimension.
        base_iteration_shape_[base_iter_dim_offset + iter_dim] =
            (dim == -1 || downsample_factors[dim] == 1 || base_shape[dim] == 1)
                ? layout.iteration_shape[iter_dim]
                : base_shape[dim];
      }
      // Copy the `layout.iteration_rank()` iteration dimensions starting at
      // `base_iter_dim_offset`.
      std::copy_n(layout.iteration_dimensions.begin(), layout.iteration_rank(),
                  &base_iteration_dimensions_[base_iter_dim_offset]);
      // Order the first `base_iter_dim_offset` synthetic dimensions (`<=
      // kMaxRank`) via `InsertionSort` rather than `std::sort` to avoid
      // introsort template bloat, without attempting to combine them.
      for (DimensionIndex iter_dim = 0; iter_dim < base_iter_dim_offset;
           ++iter_dim) {
        base_iteration_dimensions_[iter_dim] = target_rank + iter_dim;
      }
      internal::InsertionSort(&base_iteration_dimensions_[0],
                              &base_iteration_dimensions_[base_iter_dim_offset],
                              [&](DimensionIndex dim_i, DimensionIndex dim_j) {
                                return iterable.base_.GetDimensionOrder(
                                           dim_i, dim_j) < 0;
                              });
      // Since none of the synthetic dimensions are combined, their iteration
      // extents match `base_shape`.
      for (DimensionIndex iter_dim = 0; iter_dim < base_iter_dim_offset;
           ++iter_dim) {
        base_iteration_shape_[iter_dim] =
            base_shape[base_iteration_dimensions_[iter_dim]];
      }
      // Copy directions for `target_rank` dimensions; synthetic dimensions
      // added by `PropagateIndexTransformDownsampling` always iterate forward.
      std::copy_n(layout.directions.begin(), target_rank, &base_directions_[0]);
      for (DimensionIndex dim = target_rank; dim < base_rank; ++dim) {
        base_directions_[dim] = 1;
      }
      base_layout.shape = span<const Index>(base_shape, base_rank);
      base_layout.directions = span<const int>(&base_directions_[0], base_rank);
      base_layout.iteration_dimensions = span<const DimensionIndex>(
          &base_iteration_dimensions_[0], base_iteration_rank);
      base_layout.iteration_shape =
          span<const Index>(&base_iteration_shape_[0], base_iteration_rank);
      base_buffer_constraint =
          iterable.base_.GetIterationBufferConstraint(base_layout)
              .min_buffer_kind;
    }
    internal::IterationBufferShape inner_downsample_factor;
    IterationBufferKind base_buffer_constraint;

   private:
    // Only the first `layout.iteration_rank() + iterable.base_rank_ -
    // iterable.target_rank_` elements are used.
    Index base_iteration_shape_[kMaxRank];
    // Only the first `iterable.base_rank_` elements are used.
    int base_directions_[kMaxRank];
    // Only the first `layout.iteration_rank() + iterable.base_rank_ -
    // iterable.target_rank_` elements are used.
    DimensionIndex base_iteration_dimensions_[kMaxRank];
  };

  ptrdiff_t GetWorkingMemoryBytesPerElement(
      NDIterable::IterationLayoutView layout,
      IterationBufferKind buffer_kind) const override {
    NDIterable::IterationLayoutView base_layout;
    ComputeBaseLayout compute_base_layout(*this, layout, base_layout);
    Index max_window_elems = 1;
    if (downsample_functions_.store_all_elements) {
      const Index* factors = downsample_factors();
      const Index* shape = base_shape();
      for (DimensionIndex dim = 0; dim < base_rank_; ++dim) {
        const Index dim_window =
            std::max<Index>(1, std::min(factors[dim], shape[dim]));
        max_window_elems = MulSaturate(max_window_elems, dim_window);
      }
    }
    Index acc_bytes =
        MulSaturate(max_window_elems,
                    static_cast<Index>(downsample_functions_.accumulate_size));
    const Index compute_size = downsample_functions_.compute_size;
    const Index elem_size = dtype().size();
    const Index in_rows_per_out_row = std::max<Index>(
        1,
        std::min(compute_base_layout.inner_downsample_factor[0],
                 base_layout
                     .iteration_shape[base_layout.iteration_shape.size() - 2]));
    const Index in_cols_per_out_col = std::max<Index>(
        1, std::min(compute_base_layout.inner_downsample_factor[1],
                    base_layout.iteration_shape.back()));
    // Account for 1-d per-row scratch (`input_row_scratch_`,
    // `compute_row_scratch_`, and `raw_row_scratch_`).
    Index row_scratch_per_col = 0;
    if (!downsample_functions_.store_all_elements) {
      if (downsample_functions_.is_promoted ||
          compute_base_layout.base_buffer_constraint !=
              IterationBufferKind::kContiguous) {
        row_scratch_per_col =
            AddSaturate(row_scratch_per_col,
                        MulSaturate(in_cols_per_out_col, compute_size));
      }
    } else if (compute_base_layout.base_buffer_constraint !=
               IterationBufferKind::kContiguous) {
      row_scratch_per_col = AddSaturate(
          row_scratch_per_col, MulSaturate(in_cols_per_out_col, elem_size));
    }
    if (downsample_functions_.output_kind != OutputKind::kCopyAccumulate &&
        (downsample_functions_.is_promoted ||
         buffer_kind != IterationBufferKind::kContiguous)) {
      row_scratch_per_col = AddSaturate(row_scratch_per_col, compute_size);
    }
    if (downsample_functions_.is_promoted) {
      Index raw_cols_per_out_col = 0;
      if (!downsample_functions_.store_all_elements &&
          compute_base_layout.base_buffer_constraint !=
              IterationBufferKind::kContiguous) {
        raw_cols_per_out_col =
            std::max(raw_cols_per_out_col, in_cols_per_out_col);
      }
      if (buffer_kind == IterationBufferKind::kIndexed) {
        raw_cols_per_out_col = std::max<Index>(raw_cols_per_out_col, 1);
      }
      row_scratch_per_col = AddSaturate(
          row_scratch_per_col, MulSaturate(raw_cols_per_out_col, elem_size));
    }
    const Index inner_extent = layout.iteration_shape.back();
    if (downsample_functions_.is_promoted &&
        downsample_functions_.store_all_elements) {
      // `tile_compute_scratch_` holds `min(block_shape[1], kTileWidth)`
      // promoted windows; fold its per-column cost in whenever the tile could
      // exceed the headroom between the 24 KiB block target and 32 KiB `Arena`.
      const Index tile_bytes_per_col =
          MulSaturate(max_window_elems, compute_size);
      const Index max_tile_cols = std::min<Index>(inner_extent, kTileWidth);
      if (MulSaturate(max_tile_cols, tile_bytes_per_col) > 4 * 1024) {
        acc_bytes = AddSaturate(acc_bytes, tile_bytes_per_col);
      }
    }
    // `GetNDIterationBlockShape` sizes iteration blocks against a 24 KiB target
    // budget inside the 32 KiB stack `Arena`.  1-d row scratch scales with
    // `block_shape[1]` rather than `block_shape[0] * block_shape[1]`, so fold
    // `row_scratch_per_col` in only when a wide innermost dimension (`1 x W`
    // block) could exceed the 24 KiB target.
    if (row_scratch_per_col > 0 &&
        MulSaturate(inner_extent, AddSaturate(acc_bytes, row_scratch_per_col)) >
            24 * 1024) {
      acc_bytes = AddSaturate(acc_bytes, row_scratch_per_col);
    }
    Index base_bytes = base_.GetWorkingMemoryBytesPerElement(
        base_layout, compute_base_layout.base_buffer_constraint);
    base_bytes = MulSaturate(base_bytes, in_rows_per_out_row);
    base_bytes = MulSaturate(base_bytes, in_cols_per_out_col);
    const Index total = AddSaturate(base_bytes, acc_bytes);
    return static_cast<ptrdiff_t>(
        std::min<Index>(total, std::numeric_limits<ptrdiff_t>::max()));
  }

  ArenaAllocator<> get_allocator() const override {
    return ArenaAllocator<>(arena_);
  }

  DataType dtype() const override { return base_.iterables[0]->dtype(); }

  NDIterator::Ptr GetIterator(
      NDIterable::IterationBufferKindLayoutView layout) const override {
    NDIterable::IterationBufferKindLayoutView base_layout;
    ComputeBaseLayout compute_base_layout(*this, layout, base_layout);
    base_layout.buffer_kind = compute_base_layout.base_buffer_constraint;
    for (DimensionIndex i = 0; i < 2; ++i) {
      base_layout.block_shape[i] = std::min(
          base_layout
              .iteration_shape[base_layout.iteration_shape.size() - 2 + i],
          MulSaturate(compute_base_layout.inner_downsample_factor[i],
                      layout.block_shape[i]));
    }
    return internal::MakeUniqueWithVirtualIntrusiveAllocator<
        DownsampledNDIterator>(get_allocator(), downsample_functions_,
                               *base_.iterables[0], downsample_factors(),
                               base_origin(), base_layout, layout);
  }

  const Index* downsample_factors() const { return indices_buffer_; }
  const Index* base_shape() const { return indices_buffer_ + base_rank_; }
  const Index* base_origin() const { return indices_buffer_ + 2 * base_rank_; }

 private:
  const DownsampleFunctions& downsample_functions_;
  internal::NDIterablesWithManagedBuffers<std::array<NDIterable::Ptr, 1>> base_;
  DimensionIndex base_rank_, target_rank_;
  internal::Arena* arena_;
  Index* indices_buffer_;
};

}  // namespace

NDIterable::Ptr DownsampleNDIterable(NDIterable::Ptr base,
                                     BoxView<> base_domain,
                                     span<const Index> downsample_factors,
                                     DownsampleMethod downsample_method,
                                     DimensionIndex target_rank,
                                     internal::Arena* arena) {
  assert(base_domain.rank() == downsample_factors.size());
  assert(downsample_method != DownsampleMethod::kStride &&
         IsDownsampleMethodSupported(base->dtype(), downsample_method));
  bool has_downsample_dim = false;
  for (DimensionIndex i = 0; i < base_domain.rank(); ++i) {
    if (downsample_factors[i] != 1 && base_domain.shape()[i] > 1) {
      has_downsample_dim = true;
      break;
    }
  }
  if (!has_downsample_dim && target_rank == base_domain.rank()) {
    return base;
  }
  return internal::MakeUniqueWithVirtualIntrusiveAllocator<
      DownsampledNDIterable>(ArenaAllocator<>(arena), std::move(base),
                             base_domain, downsample_factors, downsample_method,
                             target_rank);
}

bool IsDownsampleMethodSupported(DataType dtype, DownsampleMethod method) {
  if (method == DownsampleMethod::kStride) return true;
  if (!dtype.valid() || dtype.id() == DataTypeId::custom) return false;
  return GetDownsampleFunctions(method, dtype).accumulate_size != 0;
}

absl::Status ValidateDownsampleMethod(DataType dtype,
                                      DownsampleMethod downsample_method) {
  if (IsDownsampleMethodSupported(dtype, downsample_method)) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Downsample method \"", downsample_method,
                   "\" does not support data type \"", dtype, "\""));
}

}  // namespace internal_downsample
}  // namespace tensorstore
