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

#include "tensorstore/internal/data_type_random_generator.h"

#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <limits>
#include <string>

#include "absl/random/random.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/util/assert_macros.h"

namespace tensorstore {
namespace internal {

namespace {
template <typename T>
struct SampleRandomValue {
  T operator()(RandomGeneratorRef gen) const {
    if constexpr (std::is_integral_v<T>) {
      return absl::Uniform(absl::IntervalClosedClosed, gen,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    } else {
      return static_cast<T>(
          absl::Uniform(absl::IntervalClosedClosed, gen, 0.0, 1.0));
    }
  }
};

template <typename T>
struct SampleRandomValue<std::complex<T>> {
  std::complex<T> operator()(RandomGeneratorRef gen) const {
    SampleRandomValue<T> sample;
    return std::complex<T>(sample(gen), sample(gen));
  }
};

template <>
struct SampleRandomValue<bool> {
  bool operator()(RandomGeneratorRef gen) const {
    return absl::Bernoulli(gen, 0.5);
  }
};

template <>
struct SampleRandomValue<std::byte> {
  std::byte operator()(RandomGeneratorRef gen) const {
    return static_cast<std::byte>(absl::Uniform<unsigned char>(gen));
  }
};

template <>
struct SampleRandomValue<std::string> {
  std::string operator()(RandomGeneratorRef gen) const {
    std::string out;
    out.resize(
        absl::Uniform<std::size_t>(absl::IntervalClosedClosed, gen, 0, 50));
    for (auto& x : out) {
      x = static_cast<char>(absl::Uniform<unsigned char>(
          absl::IntervalClosedClosed, gen, static_cast<unsigned char>('a'),
          static_cast<unsigned char>('z')));
    }
    return out;
  }
};

template <>
struct SampleRandomValue<ustring_t> {
  ustring_t operator()(RandomGeneratorRef gen) const {
    return {SampleRandomValue<std::string>()(gen)};
  }
};

template <>
struct SampleRandomValue<json_t> {
  json_t operator()(RandomGeneratorRef gen) const {
    switch (absl::Uniform(absl::IntervalClosedClosed, gen, 0, 7)) {
      case 0:
        return nullptr;
      case 1:
        return SampleRandomValue<bool>()(gen);
      case 2:
        return SampleRandomValue<std::uint64_t>()(gen);
      case 3:
        return SampleRandomValue<std::int64_t>()(gen);
      case 4:
        return SampleRandomValue<double>()(gen);
      case 5:
        return SampleRandomValue<std::string>()(gen);
      case 6: {
        json_t::array_t out;
        out.resize(
            absl::Uniform<std::size_t>(absl::IntervalClosedClosed, gen, 0, 3));
        for (auto& x : out) {
          x = (*this)(gen);
        }
        return out;
      }
      case 7: {
        json_t::object_t out;
        const auto n =
            absl::Uniform<std::size_t>(absl::IntervalClosedClosed, gen, 0, 3);
        for (size_t i = 0; i < n; ++i) {
          out.emplace(SampleRandomValue<std::string>()(gen),
                      SampleRandomValue<json_t>()(gen));
        }
        return out;
      }
      default:
        TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
    }
  }
};

}  // namespace

const std::array<ElementwiseFunction<1, RandomGeneratorRef>, kNumDataTypeIds>
    kDataTypeRandomGenerationFunctions = MapCanonicalDataTypes(
        [](auto d) -> ElementwiseFunction<1, RandomGeneratorRef> {
          using T = typename decltype(d)::Element;
          constexpr auto sample = [](T* out, RandomGeneratorRef gen) {
            *out = SampleRandomValue<T>()(gen);
          };
          return SimpleElementwiseFunction<decltype(sample)(T),
                                           RandomGeneratorRef>();
        });

SharedOffsetArray<const void> MakeRandomArray(RandomGeneratorRef gen,
                                              BoxView<> domain,
                                              DataType data_type,
                                              ContiguousLayoutOrder order) {
  assert(data_type.id() != DataTypeId::custom);
  auto array = AllocateArray(domain, order, default_init, data_type);
  kDataTypeRandomGenerationFunctions[static_cast<std::size_t>(
      data_type.id())][IterationBufferKind::kContiguous](
      nullptr, array.num_elements(),
      IterationBufferPointer{array.byte_strided_origin_pointer(),
                             data_type.size()},
      gen);
  return array;
}

}  // namespace internal
}  // namespace tensorstore
