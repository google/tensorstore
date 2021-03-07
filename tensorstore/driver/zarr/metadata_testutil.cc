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

#include "tensorstore/driver/zarr/metadata_testutil.h"

#include <tuple>

#include <gtest/gtest.h>
#include "tensorstore/internal/type_traits.h"

namespace tensorstore {
namespace internal_zarr {

namespace {
const auto GetBaseTuple = [](const ZarrDType::BaseDType& a) {
  return std::tie(a.encoded_dtype, a.dtype, a.endian);
};

const auto GetFieldTuple = [](const ZarrDType::Field& a) {
  return std::tie(static_cast<const ZarrDType::BaseDType&>(a), a.outer_shape,
                  a.name, a.field_shape, a.num_inner_elements, a.byte_offset,
                  a.num_bytes);
};

const auto GetDTypeTuple = [](const ZarrDType& a) {
  return std::tie(a.has_fields, a.fields, a.bytes_per_outer_element);
};

template <typename T>
struct remove_cvref_recursive {
  using type = internal::remove_cvref_t<T>;
};

template <typename... T>
struct remove_cvref_recursive<std::tuple<T...>> {
  using type = std::tuple<typename remove_cvref_recursive<T>::type...>;
};

// GoogleTest prints tuples containing references in an extremely verbose way.
// This works around that problem by recursively converting to a regular tuple.
template <typename X>
typename remove_cvref_recursive<X>::type RemoveRefs(const X& x) {
  return x;
}

}  // namespace

bool operator==(const ZarrDType::BaseDType& a, const ZarrDType::BaseDType& b) {
  return GetBaseTuple(a) == GetBaseTuple(b);
}

void PrintTo(const ZarrDType::BaseDType& x, std::ostream* os) {
  *os << ::testing::PrintToString(RemoveRefs(GetBaseTuple(x)));
}

bool operator==(const ZarrDType::Field& a, const ZarrDType::Field& b) {
  return GetFieldTuple(a) == GetFieldTuple(b);
}

void PrintTo(const ZarrDType::Field& x, std::ostream* os) {
  *os << ::testing::PrintToString(RemoveRefs(GetFieldTuple(x)));
}

bool operator==(const ZarrDType& a, const ZarrDType& b) {
  return GetDTypeTuple(a) == GetDTypeTuple(b);
}
void PrintTo(const ZarrDType& x, std::ostream* os) {
  *os << ::testing::PrintToString(RemoveRefs(GetDTypeTuple(x)));
}

}  // namespace internal_zarr
}  // namespace tensorstore
