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

#include "tensorstore/data_type.h"

#include <stddef.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include <nlohmann/json.hpp>
#include "tensorstore/index.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/json/same.h"
#include "tensorstore/internal/meta/attributes.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/division.h"

namespace tensorstore {
namespace {

// Invokes operator delete with alignment if alignment is greater than
// __STDCPP_DEFAULT_NEW_ALIGNMENT__.
struct AlignedDeleter {
  std::align_val_t alignment;
  void operator()(void* p) const {
    if (static_cast<size_t>(alignment) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
      ::operator delete(p, alignment);
    } else {
      ::operator delete(p);
    }
  }
};

}  // namespace

// On all common platforms, these assumptions holds.
//
// If TensorStore needs to support a platform where they do not hold, additional
// specializations of CanonicalElementType can be defined.
static_assert(std::is_same_v<signed char, ::tensorstore::dtypes::int8_t>);
static_assert(std::is_same_v<short, ::tensorstore::dtypes::int16_t>);  // NOLINT
static_assert(std::is_same_v<int, ::tensorstore::dtypes::int32_t>);
static_assert(std::is_same_v<unsigned char, ::tensorstore::dtypes::uint8_t>);
static_assert(
    std::is_same_v<unsigned short, ::tensorstore::dtypes::uint16_t>);  // NOLINT
static_assert(std::is_same_v<unsigned int, ::tensorstore::dtypes::uint32_t>);

std::ostream& operator<<(std::ostream& os, DataType r) {
  return os << absl::StreamFormat("%v", r);
}

TENSORSTORE_NODISCARD void* AllocateAndConstruct(
    ptrdiff_t n, ElementInitialization initialization, DataType r) {
  assert(n >= 0);
  assert(n <= kMaxFiniteSize);
  // AlignedDeleter is only used when the constructor throws an exception.
  size_t alignment =
      RoundUpTo(static_cast<size_t>(r->alignment), sizeof(void*));
  ptrdiff_t signed_bytes;
  size_t total_size;
  if (internal::MulOverflow(r->size, n, &signed_bytes) ||
      (total_size = RoundUpTo(static_cast<size_t>(signed_bytes), alignment)) >
          static_cast<size_t>(kMaxFiniteSize)) {
#if ABSL_HAVE_EXCEPTIONS
    throw std::bad_alloc();
#else
    std::abort();
#endif
  }
  std::unique_ptr<void, AlignedDeleter> ptr(
      (alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
          ? ::operator new(total_size, std::align_val_t(alignment))
          : ::operator new(total_size),
      AlignedDeleter{std::align_val_t(alignment)});
  if (initialization == value_init) {
    // For simplicity, we just implement value initialization by zero
    // initializing prior to default construction.  For some types, like
    // std::string, this is unnecessary, but there is no C++ type trait that
    // allows us to detect whether there is a user-provided default
    // constructor.
    std::memset(ptr.get(), 0, total_size);
  }
  r->construct(n, ptr.get());
  return ptr.release();
}

void DestroyAndFree(ptrdiff_t n, DataType r, void* ptr) {
  r->destroy(n, ptr);
  size_t alignment =
      RoundUpTo(static_cast<size_t>(r->alignment), sizeof(void*));
  AlignedDeleter deleter{std::align_val_t(alignment)};
  deleter(ptr);
}

template <>
std::shared_ptr<void> AllocateAndConstructShared<void>(
    ptrdiff_t n, ElementInitialization initialization, DataType r) {
  if (void* ptr = AllocateAndConstruct(n, initialization, r)) {
    return std::shared_ptr<void>(ptr,
                                 [r, n](void* x) { DestroyAndFree(n, r, x); });
  }
  return nullptr;
}

std::string StaticCastTraits<DataType>::Describe(DataType dtype) {
  if (!dtype.valid()) return "dynamic data type";
  return absl::StrFormat("data type of %v", dtype);
}

namespace internal_data_type {
#define TENSORSTORE_INTERNAL_DO_INSTANTIATION(X, ...) \
  TENSORSTORE_DATA_TYPE_EXPLICIT_INSTANTIATION(X)     \
  /**/
TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_INSTANTIATION)
#undef TENSORSTORE_INTERNAL_DEFINE_ELEMENT_REPRESENTATION_INSTANTIATION
}  // namespace internal_data_type

DataType GetDataType(std::string_view id) {
#define TENSORSTORE_INTERNAL_MATCH_TYPE(X, ...)     \
  if (id == std::string_view(#X, sizeof(#X) - 3)) { \
    return dtype_v<::tensorstore::dtypes::X>;       \
  }                                                 \
  /**/
  TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_MATCH_TYPE)
#undef TENSORSTORE_INTERNAL_MATCH_TYPE
  return DataType();
}

namespace internal {
absl::Status NonSerializableDataTypeError(DataType dtype) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "Cannot serialize custom data type: %s", dtype->type.name()));
}
}  // namespace internal

namespace internal_data_type {

template <>
bool CompareIdentical<::tensorstore::dtypes::json_t>(
    const ::tensorstore::dtypes::json_t& a,
    const ::tensorstore::dtypes::json_t& b) {
  return internal_json::JsonSame(a, b);
}

}  // namespace internal_data_type

namespace serialization {
bool Serializer<DataType>::Encode(EncodeSink& sink, const DataType& value) {
  if (!value.valid()) {
    return serialization::Encode(sink, std::string_view());
  }
  if (value.id() == DataTypeId::custom) {
    sink.Fail(internal::NonSerializableDataTypeError(value));
    return false;
  }
  return serialization::Encode(sink, value.name());
}

bool Serializer<DataType>::Decode(DecodeSource& source, DataType& value) {
  std::string_view name;
  if (!serialization::Decode(source, name)) return false;
  if (name.empty()) {
    value = DataType();
    return true;
  }
  value = GetDataType(name);
  if (!value.valid()) {
    source.Fail(absl::InvalidArgumentError(
        absl::StrFormat("Invalid data type: %s", name)));
    return false;
  }
  return true;
}
}  // namespace serialization

}  // namespace tensorstore
