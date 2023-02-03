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
#include <random>
#include <string>

#include "absl/base/optimization.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/elementwise_function.h"

namespace tensorstore {
namespace internal {

namespace {
template <typename T>
struct SampleRandomValue {
  T operator()(absl::BitGenRef gen) const {
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
  std::complex<T> operator()(absl::BitGenRef gen) const {
    SampleRandomValue<T> sample;
    return std::complex<T>(sample(gen), sample(gen));
  }
};

template <>
struct SampleRandomValue<bool> {
  bool operator()(absl::BitGenRef gen) const {
    return absl::Bernoulli(gen, 0.5);
  }
};

template <>
struct SampleRandomValue<std::byte> {
  std::byte operator()(absl::BitGenRef gen) const {
    return static_cast<std::byte>(absl::Uniform<unsigned char>(gen));
  }
};

template <>
struct SampleRandomValue<std::string> {
  std::string operator()(absl::BitGenRef gen) const {
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
  ustring_t operator()(absl::BitGenRef gen) const {
    return {SampleRandomValue<std::string>()(gen)};
  }
};

template <>
struct SampleRandomValue<json_t> {
  json_t operator()(absl::BitGenRef gen) const {
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
        ABSL_UNREACHABLE();  // COV_NF_LINE
    }
  }
};

}  // namespace

const std::array<ElementwiseFunction<1, absl::BitGenRef>, kNumDataTypeIds>
    kDataTypeRandomGenerationFunctions = MapCanonicalDataTypes(
        [](auto d) -> ElementwiseFunction<1, absl::BitGenRef> {
          using T = typename decltype(d)::Element;
          constexpr auto sample = [](T* out, absl::BitGenRef gen) {
            *out = SampleRandomValue<T>()(gen);
          };
          return SimpleElementwiseFunction<decltype(sample)(T),
                                           absl::BitGenRef>();
        });

SharedOffsetArray<const void> MakeRandomArray(absl::BitGenRef gen,
                                              BoxView<> domain, DataType dtype,
                                              ContiguousLayoutOrder order) {
  assert(dtype.id() != DataTypeId::custom);
  auto array = AllocateArray(domain, order, default_init, dtype);
  kDataTypeRandomGenerationFunctions[static_cast<std::size_t>(
      dtype.id())][IterationBufferKind::kContiguous](
      nullptr, array.num_elements(),
      IterationBufferPointer{array.byte_strided_origin_pointer(), dtype.size()},
      gen);
  return array;
}

}  // namespace internal
}  // namespace tensorstore
