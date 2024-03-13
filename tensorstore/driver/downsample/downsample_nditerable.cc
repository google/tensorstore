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
#include <complex>
#include <functional>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/box.h"
#include "tensorstore/data_type.h"
#include "tensorstore/downsample_method.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_buffer_management.h"
#include "tensorstore/internal/unique_with_intrusive_allocator.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

// Uncomment the line below for debug logging.
// #define TENSORSTORE_INTERNAL_DOWNSAMPLE_DEBUG 1

#ifndef TENSORSTORE_INTERNAL_DOWNSAMPLE_DEBUG
#define TENSORSTORE_INTERNAL_DOWNSAMPLE_DEBUG 0
#endif

namespace tensorstore {
namespace internal_downsample {
namespace {

using ::tensorstore::internal::ArenaAllocator;
using ::tensorstore::internal::IterationBufferKind;
using ::tensorstore::internal::IterationBufferPointer;
using ::tensorstore::internal::NDIterable;
using ::tensorstore::internal::NDIterator;

/// Traits type that defines the reduction operations for each downsample
/// method, used by `DownsampleImpl`.
///
/// This trait is implemented for each downsample operation by inheriting from
/// either `AccumulateReductionTraitsBase` or `StoreReductionTraitsBase`.
///
/// \tparam Method The downsample method.
/// \tparam Element The element type (for both the input and output).
/// \tparam SFINAE Extra parameter that may be used to constrain partial
///     specializations.
template <DownsampleMethod Method, typename Element, typename SFINAE = void>
struct ReductionTraits {
  /// Reduction operations make use of a temporary "accumulation buffer", of
  /// type `AccumulationElement`.  This default definition of `void` serves to
  /// indicate an unsupported combination of `Method` and `Element`.
  using AccumulateElement = void;

#if 0   // For documentation only.

  /// Indicates the size of the accumulation buffer required.  If `false`, the
  /// accumulation buffer is a 1-d array where each element corresponds to a
  /// single output output element.  Otherwise the buffer is a 2-d C-order
  /// (row-major) array of shape `[num_output_elements, max_total_elements]` and
  /// merely contains a copy of all of the input elements.
  constexpr static bool kStoreAllElements = false;

  /// Initializes an element of the accumulation buffer.
  ///
  /// This is called once for each element of the accumulation buffer.
  static void Initialize(AccumulateElement& x);

  /// Adds an input element to the accumulation buffer.
  ///
  /// \param acc Pointer to the start of the accumulation buffer.
  /// \param output_index Index of output element to which this input element
  ///     corresponds.
  /// \param x Input element.
  /// \param max_total_elements Maximum number of input elements for any output
  ///     element (does not vary depending on `output_index`).
  /// \param element_offset Index of input element for `output_index`.
  static void Accumulate(AccumulateElement* acc, Index output_index,
                         const Element& x, Index max_total_elements,
                         Index element_offset);

  /// Computes an output value from the accumulation buffer.
  ///
  /// \param output[out] Output value to set.
  /// \param acc Pointer to the start of the accumulation buffer.
  /// \param output_index Index of output element to compute.
  /// \param max_total_elements Maximum number of input elements for any output
  ///     element (does not vary depending on `output_index`).
  /// \param total_elements Number of calls to `Accumulate` for this
  ///     ``output_index`.
  static void Finalize(Element& output, AccumulateElement* acc,
                       Index output_index, Index max_total_elements,
                       Index total_elements);
#endif  // For documentation only.
};

/// Common `ReductionTraits` base functionality for downsample methods that
/// accumulate input values incrementally, i.e. `kMean`, `kMin`, and `kMax`.
///
/// The derived `ReductionTraits` type must define:
///
///     using AccumulateElement = ...;
///
///     void Initialize(AccumulateElement& x);
///
///     void ComputeOutput(Element &output, AccumulateElement &acc);
template <DownsampleMethod Method, typename Element>
struct AccumulateReductionTraitsBase {
  constexpr static bool kStoreAllElements = false;

  template <typename AccumulateElement>
  static void ProcessInput(AccumulateElement* acc, Index output_index,
                           const Element& input, Index max_total_elements,
                           Index element_offset) {
    ReductionTraits<Method, Element>::Accumulate(acc[output_index], input);
  }

  template <typename AccumulateElement>
  static void Finalize(Element& output, AccumulateElement* acc,
                       Index output_index, Index max_total_elements,
                       Index total_elements) {
    ReductionTraits<Method, Element>::ComputeOutput(output, acc[output_index],
                                                    total_elements);
  }
};

/// Metafunction that maps a given data type to the corresponding type used to
/// accumulate the total for `DownsampleMethod::kMean`.
///
/// Types not supported by `DownsampleMethod::kMean` are indicated by
/// `type=void`.
///
/// Floating point (and complex) types are accumulated using the same type.
/// `bool` and integer types smaller than 64 bits are accumulated using a 64-bit
/// accumulator to ensure there is no overflow.  64-bit integer types are
/// accumulated using a 128-bit accumulator.
///
/// TODO(jbms): Consider adding support for using a smaller accumulator type,
/// e.g. 16-bit accumulator for 8-bit element type, for the common case where
/// the product of the downsample factors is less than 256.
template <typename Element>
struct MeanAccumulateElement {
  using type = void;
};

template <>
struct MeanAccumulateElement<::tensorstore::dtypes::float8_e4m3fn_t> {
  using type = ::tensorstore::dtypes::float32_t;
};

template <>
struct MeanAccumulateElement<::tensorstore::dtypes::float8_e4m3fnuz_t> {
  using type = ::tensorstore::dtypes::float32_t;
};

template <>
struct MeanAccumulateElement<::tensorstore::dtypes::float8_e4m3b11fnuz_t> {
  using type = ::tensorstore::dtypes::float32_t;
};

template <>
struct MeanAccumulateElement<::tensorstore::dtypes::float8_e5m2_t> {
  using type = ::tensorstore::dtypes::float32_t;
};

template <>
struct MeanAccumulateElement<::tensorstore::dtypes::float8_e5m2fnuz_t> {
  using type = ::tensorstore::dtypes::float32_t;
};

template <>
struct MeanAccumulateElement<::tensorstore::dtypes::float16_t> {
  using type = ::tensorstore::dtypes::float32_t;
};

template <>
struct MeanAccumulateElement<::tensorstore::dtypes::bfloat16_t> {
  using type = ::tensorstore::dtypes::float32_t;
};

template <>
struct MeanAccumulateElement<::tensorstore::dtypes::float32_t> {
  using type = ::tensorstore::dtypes::float32_t;
};

template <>
struct MeanAccumulateElement<::tensorstore::dtypes::float64_t> {
  using type = ::tensorstore::dtypes::float64_t;
};

template <>
struct MeanAccumulateElement<::tensorstore::dtypes::complex64_t> {
  using type = ::tensorstore::dtypes::complex64_t;
};

template <>
struct MeanAccumulateElement<::tensorstore::dtypes::complex128_t> {
  using type = ::tensorstore::dtypes::complex128_t;
};

template <>
struct MeanAccumulateElement<bool> {
  using type = int64_t;
};

template <>
struct MeanAccumulateElement<::tensorstore::dtypes::int4_t> {
  using type = int64_t;
};

// TODO(summivox): b/295577703: uint4_t

template <>
struct MeanAccumulateElement<int8_t> {
  using type = int64_t;
};

template <>
struct MeanAccumulateElement<uint8_t> {
  using type = uint64_t;
};

template <>
struct MeanAccumulateElement<int16_t> {
  using type = int64_t;
};

template <>
struct MeanAccumulateElement<uint16_t> {
  using type = uint64_t;
};

template <>
struct MeanAccumulateElement<int32_t> {
  using type = int64_t;
};

template <>
struct MeanAccumulateElement<uint32_t> {
  using type = uint64_t;
};

template <>
struct MeanAccumulateElement<int64_t> {
  using type = absl::int128;
};

template <>
struct MeanAccumulateElement<uint64_t> {
  using type = absl::uint128;
};

template <typename Element>
struct ReductionTraits<DownsampleMethod::kMean, Element,
                       std::enable_if_t<!std::is_void_v<
                           typename MeanAccumulateElement<Element>::type>>>
    : public AccumulateReductionTraitsBase<DownsampleMethod::kMean, Element> {
  using AccumulateElement = typename MeanAccumulateElement<Element>::type;

  static void Initialize(AccumulateElement& x) { x = AccumulateElement{}; }

  static void Accumulate(AccumulateElement& acc, const Element& input) {
    acc += input;
  }

  static void ComputeOutput(Element& output, const AccumulateElement& acc,
                            Index total_elements) {
    AccumulateElement acc_value = acc;
    const auto converted_total_elements =
        static_cast<AccumulateElement>(total_elements);
    if constexpr (std::numeric_limits<Element>::is_integer ||
                  std::is_same_v<Element, bool>) {
      // Round integral types to nearest value, and round to even in case of a
      // tie.

      // Note: Optimizing compiler will likely combine the quotient and
      // remainder calculation.
      auto quotient = acc_value / converted_total_elements;
      auto remainder = acc_value % converted_total_elements;

      if (acc_value >= 0) {
        acc_value = quotient +
                    (remainder * 2 + (quotient & 1) > converted_total_elements);
      } else {
        acc_value = quotient - (remainder * 2 - (quotient & 1) <
                                -converted_total_elements);
      }

      // TODO(jbms): Consider optimizing `Element={u,}int64_t` case where the
      // `acc_value` fits in a 64-bit range, since `absl::{u,}int128` modulus
      // and division are fairly slow.
    } else {
      acc_value /= converted_total_elements;
    }
    output = static_cast<Element>(acc_value);
  }
};

/// `bool`-valued metafunction that evaluates to `true` for bool, integer, and
/// floating-point types.
template <typename Element>
struct IsOrderingSupported {
  constexpr static bool value = false;
};

#define TENSORSTORE_INTERNAL_SPECIALIZE_ORDERING_SUPPORTED(T, ...) \
  template <>                                                      \
  struct IsOrderingSupported<::tensorstore::dtypes::T> {           \
    constexpr static bool value = true;                            \
  };                                                               \
  /**/

TENSORSTORE_FOR_EACH_BOOL_DATA_TYPE(
    TENSORSTORE_INTERNAL_SPECIALIZE_ORDERING_SUPPORTED)

TENSORSTORE_FOR_EACH_INT_DATA_TYPE(
    TENSORSTORE_INTERNAL_SPECIALIZE_ORDERING_SUPPORTED)

TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE(
    TENSORSTORE_INTERNAL_SPECIALIZE_ORDERING_SUPPORTED)

#undef TENSORSTORE_INTERNAL_SPECIALIZE_ORDERING_SUPPORTED

template <typename Element>
struct ReductionTraits<DownsampleMethod::kMin, Element,
                       std::enable_if_t<IsOrderingSupported<Element>::value>>
    : public AccumulateReductionTraitsBase<DownsampleMethod::kMin, Element> {
  using AccumulateElement = Element;
  static void Initialize(Element& x) {
    if constexpr (std::numeric_limits<Element>::is_integer ||
                  std::is_same_v<Element, bool>) {
      x = std::numeric_limits<Element>::max();
    } else {
      x = std::numeric_limits<Element>::infinity();
    }
  }
  static void Accumulate(Element& acc, const Element& input) {
    acc = std::min(acc, input);
  }
  static void ComputeOutput(Element& output, const Element& acc,
                            Index total_elements) {
    output = acc;
  }
};

template <typename Element>
struct ReductionTraits<DownsampleMethod::kMax, Element,
                       std::enable_if_t<IsOrderingSupported<Element>::value>>
    : public AccumulateReductionTraitsBase<DownsampleMethod::kMax, Element> {
  using AccumulateElement = Element;
  static void Initialize(Element& x) {
    if constexpr (std::numeric_limits<Element>::is_integer ||
                  std::is_same_v<Element, bool>) {
      x = std::numeric_limits<Element>::min();
    } else {
      x = -std::numeric_limits<Element>::infinity();
    }
  }
  static void Accumulate(Element& acc, const Element& input) {
    acc = std::max(acc, input);
  }
  static void ComputeOutput(Element& output, const Element& acc,
                            Index total_elements) {
    output = acc;
  }
};

/// Common `ReductionTraits` base functionality for downsample methods that
/// store all input values, i.e. `DownsampleMethod::kMedian` and
/// `DownsampleMethod::kMode`.
///
/// The derived `ReductionTraits` type must implement:
///
///     void ComputeOutput(Element &output, span<Element> input);
///
/// which computes the downsampled output from the stored input values.
template <DownsampleMethod Method, typename Element>
struct StoreReductionTraitsBase {
  using AccumulateElement = Element;

  constexpr static bool kStoreAllElements = true;

  static void Initialize(Element& x) {}
  static void ProcessInput(Element* acc, Index output_index,
                           const Element& input, Index max_total_elements,
                           Index element_offset) {
    acc[output_index * max_total_elements + element_offset] = input;
  }
  static void Finalize(Element& output, Element* acc, Index output_index,
                       Index max_total_elements, Index total_elements) {
    acc += output_index * max_total_elements;
    ReductionTraits<Method, Element>::ComputeOutput(
        output, span<Element>(acc, total_elements));
  }
};

template <typename Element>
struct ReductionTraits<DownsampleMethod::kMedian, Element,
                       std::enable_if_t<IsOrderingSupported<Element>::value>>
    : public StoreReductionTraitsBase<DownsampleMethod::kMedian, Element> {
  static void ComputeOutput(Element& output, span<Element> input) {
    auto median_it = input.begin() + (input.size() - 1) / 2;
    std::nth_element(input.begin(), median_it, input.end());
    output = *median_it;
  }
};

/// Comparison function used for computing the mode.
///
/// This yields a total ordering over all canonical data types, even for complex
/// numbers and json for which there is not a canonical ordering.
template <typename Element>
struct CompareForMode : public std::less<Element> {};

/// Order complex numbers lexicographically.
template <typename T>
struct CompareForMode<std::complex<T>> {
  bool operator()(const std::complex<T>& a, std::complex<T>& b) const {
    return std::pair(a.real(), a.imag()) < std::pair(b.real(), b.imag());
  }
};

template <typename Element>
struct ReductionTraits<DownsampleMethod::kMode, Element>
    : public StoreReductionTraitsBase<DownsampleMethod::kMode, Element> {
  static void ComputeOutput(Element& output, span<Element> input) {
    // Sort in order to determine the number of times each distinct value is
    // repeated.
    std::sort(input.begin(), input.end(), CompareForMode<Element>{});
    Index most_frequent_index = 0;
    size_t most_frequent_count = 1;
    size_t cur_count = 1;
    for (ptrdiff_t i = 1; i < input.size(); ++i) {
      if (input[i] == input[i - 1]) {
        ++cur_count;
      } else {
        if (cur_count > most_frequent_count) {
          most_frequent_count = cur_count;
          most_frequent_index = i - 1;
        }
        cur_count = 1;
      }
    }
    if (cur_count > most_frequent_count) {
      most_frequent_index = input.size() - 1;
    }
    output = input[most_frequent_index];
  }
};

// For `bool`, the median and mode are equal to the mean, which can be computed
// more efficiently.
template <>
struct ReductionTraits<DownsampleMethod::kMedian, bool>
    : public ReductionTraits<DownsampleMethod::kMean, bool> {};

template <>
struct ReductionTraits<DownsampleMethod::kMode, bool>
    : public ReductionTraits<DownsampleMethod::kMean, bool> {};

/// Template class that generates the type-specific and method-specific
/// implementation for performing the downsample computation.
///
/// Pointers to these functions are stored in `DownsampleFunctions` for type
/// erasure.
template <DownsampleMethod Method, typename Element>
struct DownsampleImpl {
  using Traits = ReductionTraits<Method, Element>;
  using AccumulateElement = typename Traits::AccumulateElement;

  static void Initialize(void* accumulate_buffer, Index output_block_size) {
    auto* acc = static_cast<AccumulateElement*>(accumulate_buffer);
    for (Index i = 0; i < output_block_size; ++i) {
      Traits::Initialize(acc[i]);
    }
  }

  /// ElementwiseFunction LoopTemplate implementation for accumulating the
  /// total.
  struct ProcessInput {
    template <typename ArrayAccessor>
    static bool Loop(void* accumulate_buffer,
                     internal::IterationBufferShape output_block_shape,
                     IterationBufferPointer source_pointer,
                     internal::IterationBufferShape base_block_shape,
                     internal::IterationBufferShape base_block_offset,
                     internal::IterationBufferShape downsample_factor,
                     Index outer_divisor, Index prior_calls) {
      auto* acc = static_cast<AccumulateElement*>(accumulate_buffer);
      const Index max_total_elements =
          outer_divisor * downsample_factor[0] * downsample_factor[1];

      const auto for_each_source_index = [&](auto inner_dim_i, auto callback) {
        if (downsample_factor[inner_dim_i] == 1) {
          // Source block does not need to be downsampled along `inner_dim_i`.
          for (Index i = 0; i < base_block_shape[inner_dim_i]; ++i) {
            callback(/*output_index=*/i, /*source_index=*/i,
                     /*source_element_offset=*/0, /*num_source_elements=*/1);
          }
        } else {
          // Source block needs to be downsampled along `inner_dim_i`.

          // Handle `output_index=0` specially to account for
          // `base_block_offset[inner_dim_i]`.
          const Index offset0_num_source_elements = std::min(
              downsample_factor[inner_dim_i] - base_block_offset[inner_dim_i],
              base_block_shape[inner_dim_i] + base_block_offset[inner_dim_i]);
          for (Index offset = 0; offset < offset0_num_source_elements;
               ++offset) {
            callback(
                /*output_index=*/0, /*source_index=*/offset,
                /*element_offset=*/offset,
                /*num_source_elements=*/offset0_num_source_elements);
          }

          // Handle `output_index>0`.
          for (Index offset = 0; offset < downsample_factor[inner_dim_i];
               ++offset) {
            for (Index output_index = 1,
                       source_i = offset - base_block_offset[inner_dim_i] +
                                  downsample_factor[inner_dim_i];
                 source_i < base_block_shape[inner_dim_i];
                 ++output_index, source_i += downsample_factor[inner_dim_i]) {
              callback(
                  output_index, source_i, /*element_offset=*/offset,
                  /*num_source_elements=*/
                  std::min(downsample_factor[inner_dim_i],
                           base_block_shape[inner_dim_i] +
                               base_block_offset[inner_dim_i] -
                               output_index * downsample_factor[inner_dim_i]));
            }
          }
        }
      };

      const auto process_input_row = [&](Index output_outer_i,
                                         Index source_outer_i,
                                         Index num_outer_elements,
                                         Index outer_element_offset) {
        for_each_source_index(
            std::integral_constant<Index, 1>{},
            [&](Index output_inner_i, Index source_inner_i, Index element_i,
                Index num_source_elements) {
#if TENSORSTORE_INTERNAL_DOWNSAMPLE_DEBUG
              std::string value_str;
              dtype_v<Element>->append_to_string(
                  &value_str,
                  ArrayAccessor::template GetPointerAtPosition<Element>(
                      source_pointer, source_outer_i, source_inner_i));
              ABSL_LOG(INFO)
                  << "output=" << output_outer_i << "," << output_inner_i
                  << "; source=" << source_outer_i << "," << source_inner_i
                  << "; element_i="
                  << outer_element_offset + element_i * num_outer_elements
                  << "; value=" << value_str;
#endif
              Traits::ProcessInput(
                  acc, output_outer_i * output_block_shape[1] + output_inner_i,
                  *ArrayAccessor::template GetPointerAtPosition<Element>(
                      source_pointer, source_outer_i, source_inner_i),
                  /*max_total_elements=*/max_total_elements,
                  /*element_offset=*/outer_element_offset +
                      element_i * num_outer_elements);
            });
      };
      for_each_source_index(
          std::integral_constant<Index, 0>{},
          [&](Index output_outer_i, Index source_outer_i, Index element_i,
              Index num_source_elements) {
            process_input_row(output_outer_i, source_outer_i,
                              outer_divisor * num_source_elements,
                              prior_calls * num_source_elements + element_i);
          });
      return true;
    }
  };

  /// ElementwiseFunction LoopTemplate implementation for computing the output
  /// from the accumulated values.
  struct ComputeOutput {
    template <typename ArrayAccessor>
    static bool Loop(void* accumulate_buffer,
                     internal::IterationBufferShape output_block_shape,
                     IterationBufferPointer output_pointer,
                     internal::IterationBufferShape base_block_shape,
                     internal::IterationBufferShape base_block_offset,
                     internal::IterationBufferShape downsample_factor,
                     Index outer_divisor) {
      const Index max_total_elements =
          outer_divisor * downsample_factor[0] * downsample_factor[1];
      auto* acc = static_cast<AccumulateElement*>(accumulate_buffer);
      for (Index outer_i = 0; outer_i < output_block_shape[0]; ++outer_i) {
        Index divisor1 =
            outer_divisor *
            std::min(downsample_factor[0],
                     outer_i == 0
                         ? std::min(base_block_shape[0], (downsample_factor[0] -
                                                          base_block_offset[0]))
                         : (base_block_shape[0] + base_block_offset[0] -
                            outer_i * downsample_factor[0]));
        const Index full_divisor = divisor1 * downsample_factor[1];
        const auto compute_and_store_output_value = [&](Index i,
                                                        Index divisor) {
          Traits::Finalize(
              *ArrayAccessor::template GetPointerAtPosition<Element>(
                  output_pointer, outer_i, i),
              acc, outer_i * output_block_shape[1] + i, max_total_elements,
              divisor);
#if TENSORSTORE_INTERNAL_DOWNSAMPLE_DEBUG
          std::string value_str;
          dtype_v<Element>->append_to_string(
              &value_str, ArrayAccessor::template GetPointerAtPosition<Element>(
                              output_pointer, outer_i, i));
          ABSL_LOG(INFO) << "Finalize: output=" << outer_i << "," << i
                         << ", full_divisor=" << max_total_elements
                         << ", divisor=" << divisor << ", value=" << value_str;
#endif
        };
        Index full_divisor_begin_offset = 0;
        Index full_divisor_end_offset = output_block_shape[1];
        if (base_block_offset[1] != 0) {
          ++full_divisor_begin_offset;
          compute_and_store_output_value(
              0, divisor1 *
                     std::min(base_block_shape[1],
                              (downsample_factor[1] - base_block_offset[1])));
        }
        if (output_block_shape[1] * downsample_factor[1] !=
                base_block_shape[1] + base_block_offset[1] &&
            full_divisor_begin_offset != output_block_shape[1]) {
          --full_divisor_end_offset;
          compute_and_store_output_value(
              full_divisor_end_offset,
              divisor1 * (base_block_shape[1] + base_block_offset[1] +
                          downsample_factor[1] -
                          output_block_shape[1] * downsample_factor[1]));
        }
        for (Index i = full_divisor_begin_offset; i < full_divisor_end_offset;
             ++i) {
          compute_and_store_output_value(i, full_divisor);
        }
      }
      return true;
    }
  };
};

struct DownsampleFunctions {
  using AllocateAccumulateBuffer = void* (*)(Index n,
                                             ArenaAllocator<> allocator);
  using DeallocateAccumulateBuffer = void (*)(void* p, Index n,
                                              ArenaAllocator<> allocator);

  using Initialize = void (*)(void* p, Index n);

  using ProcessInput = internal::ElementwiseFunction<
      1, /*base_block_shape*/ internal::IterationBufferShape,
      /*base_block_offset*/ internal::IterationBufferShape,
      /*downsample_factor*/ internal::IterationBufferShape,
      /*outer_divisor*/ Index,
      /*prior_calls*/ Index>;
  using ComputeOutput = internal::ElementwiseFunction<
      1, /*base_block_shape*/ internal::IterationBufferShape,
      /*base_block_offset*/ internal::IterationBufferShape,
      /*downsample_factor*/ internal::IterationBufferShape,
      /*outer_divisor*/ Index>;

  AllocateAccumulateBuffer allocate_accumulate_buffer;
  DeallocateAccumulateBuffer deallocate_accumulate_buffer;
  Initialize initialize;
  ProcessInput process_input;
  ComputeOutput compute_output;
  DataType accumulate_data_type;
  bool store_all_elements;
};

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

/// Type-specific operations for allocating and deallocating an accumulator
/// buffer using an `ArenaAllocator`.
///
/// Pointers to these functions are stored in `DownsampleFunctions` for type
/// erasure.
template <typename AccumulateElement>
struct AccumulateBufferImpl {
  static void* Allocate(Index n, ArenaAllocator<> allocator) {
    auto* buf = ArenaAllocator<AccumulateElement>(allocator).allocate(n);
    std::uninitialized_default_construct_n(buf, n);
    return buf;
  }

  static void Deallocate(void* p, Index n, ArenaAllocator<> allocator) {
    auto* acc = static_cast<AccumulateElement*>(p);
    std::destroy_n(acc, n);
    ArenaAllocator<AccumulateElement>(allocator).deallocate(acc, n);
  }
};

/// Array of type-erased `DownsampleFunctions` indexed by `DownsampleMethod` and
/// `DataTypeId`.  These combine the specific implementations of allocate,
/// deallocate, accumulate, etc. into a single collection of operations used by
/// `DownsampledNDIterator`.
constexpr std::array<std::array<DownsampleFunctions, kNumDataTypeIds>,
                     kNumDownsampleMethods>
    kDownsampleFunctions = MapDownsampleMethods([](auto method) {
      return MapCanonicalDataTypes([](auto dtype) -> DownsampleFunctions {
        using Element = typename decltype(dtype)::Element;
        constexpr DownsampleMethod downsample_method = decltype(method)::value;
        using Traits = ReductionTraits<downsample_method, Element>;
        using AccumulateElement = typename Traits::AccumulateElement;
        if constexpr (!std::is_void_v<AccumulateElement>) {
          using Impl = DownsampleImpl<downsample_method, Element>;
          using AccImpl = AccumulateBufferImpl<AccumulateElement>;
          return {
              &AccImpl::Allocate,
              &AccImpl::Deallocate,
              &Impl::Initialize,
              DownsampleFunctions::ProcessInput(typename Impl::ProcessInput{}),
              DownsampleFunctions::ComputeOutput(
                  typename Impl::ComputeOutput{}),
              dtype_v<AccumulateElement>,
              Traits::kStoreAllElements,
          };
        } else {
          return {};
        }
      });
    });

inline const DownsampleFunctions& GetDownsampleFunctions(
    DownsampleMethod downsample_method, DataType dtype) {
  assert(dtype.id() != DataTypeId::custom);
  assert(downsample_method != DownsampleMethod::kStride);
  return kDownsampleFunctions[DownsampleMethodIndex(downsample_method)]
                             [static_cast<int>(dtype.id())];
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
/// iteration dimension `i + (base_rank - downsampled_rank)`, and base iteration
/// dimension `i`, for `i < (base_rank - downsampled_rank)`, corresponds to an
/// implicit singleton downsampled dimension.
class DownsampledNDIterator : public NDIterator::Base<DownsampledNDIterator> {
  /// Returns a function that returns contiguous buffers from `indices_buffer_`.
  auto IndicesGetter() {
    return [next_indices =
                indices_buffer_.data()](DimensionIndex n) mutable -> Index* {
      auto* ret = next_indices;
      next_indices += n;
      return ret;
    };
  }

 public:
  /// Constructs a downsampled iterator.
  ///
  /// \param downsample_method The downsample method to use.
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
  /// \param allocator Allocator to use.
  DownsampledNDIterator(
      const DownsampleFunctions& downsample_functions,
      const NDIterable& base_iterable, const Index* downsample_factors,
      const Index* base_origin,
      internal::NDIterable::IterationBufferKindLayoutView base_layout,
      internal::NDIterable::IterationBufferKindLayoutView layout,
      ArenaAllocator<> allocator)
      : base_(std::array{&base_iterable}, base_layout, allocator),
        indices_buffer_(allocator) {
    base_iteration_rank_ = base_layout.iteration_rank();
    initialize_accumulate_buffer_ = downsample_functions.initialize;
    process_input_ =
        downsample_functions.process_input[base_layout.buffer_kind];
    compute_output_ = downsample_functions.compute_output[layout.buffer_kind];
    deallocate_accumulate_buffer_ =
        downsample_functions.deallocate_accumulate_buffer;
    const Index block_size = layout.block_shape[0] * layout.block_shape[1];
    DimensionIndex num_downsample_dims = 0;
    const auto is_downsample_dim = [&](DimensionIndex base_iter_dim) {
      const DimensionIndex dim =
          base_layout.iteration_dimensions[base_iter_dim];
      return dim != -1 && downsample_factors[dim] != 1 &&
             base_layout.iteration_shape[base_iter_dim] > 1;
    };
    // Count the number of downsampled dimensions.
    for (DimensionIndex base_iter_dim = 0;
         base_iter_dim < base_layout.iteration_rank(); ++base_iter_dim) {
      if (is_downsample_dim(base_iter_dim)) ++num_downsample_dims;
    }
    assert(num_downsample_dims > 0);
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
    //     in `downsample_dims[num_downsample_dims-1]`.  Note: This array
    //     logically stores `DimensionIndex` values rather than `Index` values.
    //
    //   Index downsample_factors[num_downsample_dims];
    //
    //     Downsample factor (from `downsample_factors`) corresponding to each
    //     iteration dimension in `downsample_dims`.
    //
    //   Index downsample_dim_iteration_shape[num_downsample_dims];
    //
    //     Iteration shape (from `base_layout.iteration_shape` corresponding to
    //     each iteration dimension in `downsample_dims`.
    //
    //   Index downsample_dim_origin[num_downsample_dims];
    //
    //     Origin (from `base_origin`) corresponding to each iteration dimension
    //     in `downsample_dims`.
    //
    // `indices_buffer` also contains the following arrays (after the ones
    // listed above) that are used as temporary space by `GetBlock`.
    //
    //   Index base_downsample_dim_offsets[num_downsample_dims];
    //   Index base_downsample_dim_offsets_bounds[num_downsample_dims];
    //   Index initial_base_indices[base_layout.iteration_rank()];
    //   Index initial_base_indices[base_layout.iteration_rank()];
    indices_buffer_.resize(num_downsample_dims * 6 +
                           2 * base_layout.iteration_rank());
    auto get_indices = IndicesGetter();
    Index* downsample_dims = get_indices(num_downsample_dims);
    Index* downsample_factors_ = get_indices(num_downsample_dims);
    Index* downsample_dim_iteration_shape = get_indices(num_downsample_dims);
    Index* downsample_dim_origin = get_indices(num_downsample_dims);

    // Store the iteration dimension index, downsample factor and extent for
    // each downsampled dimension.
    DimensionIndex downsample_dim_i = 0;
    Index product_of_downsample_factors = 1;
    for (DimensionIndex base_iter_dim = 0;
         base_iter_dim < base_layout.iteration_rank(); ++base_iter_dim) {
      if (!is_downsample_dim(base_iter_dim)) continue;
      const DimensionIndex dim =
          base_layout.iteration_dimensions[base_iter_dim];
      downsample_dims[downsample_dim_i] = base_iter_dim;
      const Index factor = downsample_factors_[downsample_dim_i] =
          downsample_factors[dim];
      product_of_downsample_factors *= factor;
      assert(base_layout.directions[dim] == 1);
      downsample_dim_iteration_shape[downsample_dim_i] =
          base_layout.iteration_shape[base_iter_dim];
      downsample_dim_origin[downsample_dim_i] = base_origin[dim];
      ++downsample_dim_i;
    }
    accumulate_buffer_size_ =
        block_size * (downsample_functions.store_all_elements
                          ? product_of_downsample_factors
                          : 1);
    accumulate_buffer_ = downsample_functions.allocate_accumulate_buffer(
        accumulate_buffer_size_, allocator);
  }

  ~DownsampledNDIterator() {
    deallocate_accumulate_buffer_(accumulate_buffer_, accumulate_buffer_size_,
                                  get_allocator());
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
    // process/accumulate.  This is initialized to all 0 and then
    // `internal::AdvanceIndices` is called repeatedly to iterate over the
    // `num_outer_downsample_dims`-rank range
    // `[0, base_downsample_dim_offsets_bounds)`.  Note: the length is always
    // `num_downsample_dims`, but if the inner dimension is downsampled, then
    // the last component is unused.
    Index* base_downsample_dim_offsets = get_indices(num_downsample_dims);

    // Vector of length `num_downsample_dims` specifying the shape of the
    // downsample block to use for the current `indices`.  The shape of the
    // downsample block is by default given by `downsample_factors`, but is
    // clipped to fit within the domain of `base_` (specified by
    // `downsample_dim_origin` and `downsample_dim_iteration_shape`).  Note that
    // `base_downsample_dim_offsets` always starts at `0`; these bounds reflect
    // clipping on both sides.
    Index* base_downsample_dim_offsets_bounds =
        get_indices(num_downsample_dims);

    // Vector of length `base_iteration_rank` specifying the iteration position
    // within `base_` corresponding to `indices`.
    Index* initial_base_indices = get_indices(base_iteration_rank);

    // Vector of length `base_iteration_rank` specifying the iteration position
    // within `base_` for each value of `base_downsample_dim_offset` in the loop
    // below.  This is equal to `initial_base_indices` adjusted by
    // `base_downsample_dim_offsets`.
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
    for (DimensionIndex i = 0; i < num_downsample_dims; ++i) {
      const Index downsample_factor = downsample_factors[i];
      const DimensionIndex base_dim = downsample_dims[i];
      base_downsample_dim_offsets[i] = 0;
      Index& initial_base_index = initial_base_indices[base_dim];
      Index base_inclusive_min =
          initial_base_index * downsample_factor - downsample_dim_origin[i];
      const Index clamped_base_inclusive_min =
          std::max(Index(0), base_inclusive_min);
      initial_base_index = clamped_base_inclusive_min;
      if (base_dim >= base_iteration_rank - 2) {
        // This downsampling factor applies to one of the 2 innermost
        // dimensions.
        const DimensionIndex inner_dim_i = base_dim - (base_iteration_rank - 2);
        --num_outer_downsample_dims;

        const Index base_exclusive_max = std::min(
            base_inclusive_min + block_shape[inner_dim_i] * downsample_factor,
            downsample_dim_iteration_shape[i]);
        inner_downsample_factor[inner_dim_i] = downsample_factor;
        base_block_offset[inner_dim_i] =
            clamped_base_inclusive_min - base_inclusive_min;
        base_block_shape[inner_dim_i] =
            base_exclusive_max - clamped_base_inclusive_min;
      } else {
        // This downsampling factor applies to an outer dimension.
        Index base_exclusive_max =
            std::min(base_inclusive_min + downsample_factor,
                     downsample_dim_iteration_shape[i]);
        const Index bound = base_downsample_dim_offsets_bounds[i] =
            base_exclusive_max - clamped_base_inclusive_min;
        outer_divisor *= bound;
      }
    }
    std::copy_n(initial_base_indices, base_iteration_rank, base_indices);

    // Request and accumulate all of the required base (input) blocks.
    auto process_input = process_input_;
    initialize_accumulate_buffer_(accumulate_buffer_, accumulate_buffer_size_);
    Index prior_accumulate_calls = 0;
    do {
      for (DimensionIndex i = 0; i < num_outer_downsample_dims; ++i) {
        const DimensionIndex dim = downsample_dims[i];
        base_indices[dim] =
            initial_base_indices[dim] + base_downsample_dim_offsets[i];
      }
      ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_DOWNSAMPLE_DEBUG)
          << "Output block: " << indices << ", block_shape=" << block_shape[0]
          << "," << block_shape[1] << ": Getting base block: "
          << span<const Index>(base_indices, base_iteration_rank)
          << " of shape=" << base_block_shape[0] << "," << base_block_shape[1];
      if (!base_.GetBlock(span<const Index>(base_indices, base_iteration_rank),
                          base_block_shape, status)) {
        return false;
      }
      process_input(accumulate_buffer_, block_shape, base_.block_pointers()[0],
                    base_block_shape, base_block_offset,
                    inner_downsample_factor, outer_divisor,
                    prior_accumulate_calls);
      ++prior_accumulate_calls;
    } while (internal::AdvanceIndices(num_outer_downsample_dims,
                                      base_downsample_dim_offsets,
                                      base_downsample_dim_offsets_bounds));

    // Compute the downsampled output from the accumulated state.
    compute_output_(accumulate_buffer_, block_shape, *pointer, base_block_shape,
                    base_block_offset, inner_downsample_factor, outer_divisor);
    return true;
  }

 private:
  internal::NDIteratorsWithManagedBuffers<1> base_;
  Index accumulate_buffer_size_;
  DimensionIndex num_downsample_dims_;
  DimensionIndex base_iteration_rank_;
  std::vector<Index, ArenaAllocator<Index>> indices_buffer_;
  void* accumulate_buffer_;
  DownsampleFunctions::Initialize initialize_accumulate_buffer_;
  DownsampleFunctions::ProcessInput::SpecializedFunctionPointer process_input_;
  DownsampleFunctions::ComputeOutput::SpecializedFunctionPointer
      compute_output_;
  DownsampleFunctions::DeallocateAccumulateBuffer deallocate_accumulate_buffer_;
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
        indices_buffer_(base_rank_ * 3, allocator) {
    const Index base_rank = base_rank_;
    assert(base_domain.rank() == base_rank);
    for (Index dim = 0; dim < base_rank; ++dim) {
      Index downsample_factor = downsample_factors[dim];
      Index base_size = base_domain.shape()[dim];
      if (base_size == 1) downsample_factor = 1;
      Index origin_remainder = base_domain.origin()[dim] % downsample_factor;
      if (origin_remainder < 0) origin_remainder += downsample_factor;
      indices_buffer_[dim] = downsample_factor;
      indices_buffer_[dim + base_rank] = base_size;
      indices_buffer_[dim + 2 * base_rank] = origin_remainder;
    }
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
    // `base_rank_ > target_rank_`, we need to use a temporary array to call
    // `base_.UpdateDirectionPrefs`.
    const DimensionIndex target_rank = target_rank_;
    DirectionPref base_prefs[kMaxRank];

    // kCanSkip imposes no constraints.  The call to
    // `base_.UpdateDirectionPrefs` adds constraints.
    std::fill(&base_prefs[0], &base_prefs[base_rank_], DirectionPref::kCanSkip);
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
        /*.buffer_kind=*/IterationBufferKind::kContiguous, /*.external=*/true};
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
      // Iteration dimension `iter_dim` of the `DownsampledNDIterator`
      // corresponds to iteration dimension `iter_dim + base_iter_dim_offset` of
      // the base `NDIterator`.
      const DimensionIndex base_iter_dim_offset = base_rank - target_rank;
      for (DimensionIndex iter_dim = 0; iter_dim < layout.iteration_rank();
           ++iter_dim) {
        const DimensionIndex dim = layout.iteration_dimensions[iter_dim];
        // If `downsample_factors[dim] == 1`, `iter_dim` may correspond to
        // multiple original dimensions, if permitted by `CanCombineDimensions`,
        // but none of those dimensions may be downsampled; therefore, the base
        // extent for `iter_dim` is guaranteed to equal the downsampled extent.
        // If `downsample_factors[dim] != 1`, `iter_dim` corresponds to a single
        // original dimension.
        base_iteration_shape_[base_iter_dim_offset + iter_dim] =
            (dim == -1 || downsample_factors[dim] == 1 || base_shape[dim] == 1)
                ? layout.iteration_shape[iter_dim]
                : base_shape[dim];
      }
      // Copy the first `layout.iteration_rank()` iteration dimensions starting
      // at `base_iter_dim_offset`.
      std::copy_n(layout.iteration_dimensions.begin(), layout.iteration_rank(),
                  &base_iteration_dimensions_[base_iter_dim_offset]);
      // Choose iteration order for the first `base_iter_dim_offset` iteration
      // dimensions.  We do not attempt to combine any, because
      // `DownsampledNDIterator` does not support combining downsampled
      // dimensions.
      std::iota(&base_iteration_dimensions_[0],
                &base_iteration_dimensions_[base_iter_dim_offset], target_rank);
      std::sort(&base_iteration_dimensions_[0],
                &base_iteration_dimensions_[base_iter_dim_offset],
                [&](DimensionIndex dim_i, DimensionIndex dim_j) {
                  return iterable.base_.GetDimensionOrder(dim_i, dim_j) < 0;
                });
      // Set `base_iteration_shape_` for the first `base_iter_dim_offset`
      // iteration dimensions.  Since none of these dimensions are combined, the
      // iteration extent simply matches the base extent.
      for (DimensionIndex iter_dim = 0; iter_dim < base_iter_dim_offset;
           ++iter_dim) {
        base_iteration_shape_[iter_dim] =
            base_shape[base_iteration_dimensions_[iter_dim]];
      }
      // Copy direction for dimensions included in `target_rank`.
      std::copy_n(layout.directions.begin(), target_rank, &base_directions_[0]);
      // Use forward direction for synthetic dimensions added by
      // `PropagateIndexTransformDownsampling`.
      std::fill_n(&base_directions_[target_rank], base_rank - target_rank, 1);
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
    const Index accumulate_elements_per_output_element =
        downsample_functions_.store_all_elements
            ? ProductOfExtents(
                  span<const Index>(downsample_factors(), base_rank_))
            : 1;
    return base_.GetWorkingMemoryBytesPerElement(
               base_layout, compute_base_layout.base_buffer_constraint) *
               compute_base_layout.inner_downsample_factor[0] *
               compute_base_layout.inner_downsample_factor[1] +
           accumulate_elements_per_output_element *
               downsample_functions_.accumulate_data_type.size();
  }

  ArenaAllocator<> get_allocator() const override {
    return indices_buffer_.get_allocator();
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
          compute_base_layout.inner_downsample_factor[i] *
              layout.block_shape[i]);
    }
    return internal::MakeUniqueWithVirtualIntrusiveAllocator<
        DownsampledNDIterator>(get_allocator(), downsample_functions_,
                               *base_.iterables[0], downsample_factors(),
                               base_origin(), base_layout, layout);
  }

  const Index* downsample_factors() const { return indices_buffer_.data(); }
  const Index* base_shape() const {
    return indices_buffer_.data() + base_rank_;
  }
  const Index* base_origin() const {
    return indices_buffer_.data() + 2 * base_rank_;
  }

 private:
  const DownsampleFunctions& downsample_functions_;
  internal::NDIterablesWithManagedBuffers<std::array<NDIterable::Ptr, 1>> base_;
  DimensionIndex base_rank_, target_rank_;
  std::vector<Index, ArenaAllocator<Index>> indices_buffer_;
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
  if (!has_downsample_dim) {
    assert(target_rank == base_domain.rank());
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
  return GetDownsampleFunctions(method, dtype).accumulate_data_type.valid();
}

absl::Status ValidateDownsampleMethod(DataType dtype,
                                      DownsampleMethod downsample_method) {
  if (IsDownsampleMethodSupported(dtype, downsample_method)) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      tensorstore::StrCat("Downsample method \"", downsample_method,
                          "\" does not support data type \"", dtype, "\""));
}

}  // namespace internal_downsample
}  // namespace tensorstore
