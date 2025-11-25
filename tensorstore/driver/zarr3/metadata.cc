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

#include "tensorstore/driver/zarr3/metadata.h"

#include <stddef.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <cstring>

#include "absl/algorithm/container.h"
#include "absl/strings/escaping.h"
#include "absl/base/casts.h"
#include "absl/base/optimization.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/sharding_indexed.h"
#include "tensorstore/driver/zarr3/default_nan.h"
#include "tensorstore/driver/zarr3/dtype.h"
#include "tensorstore/driver/zarr3/name_configuration_json_binder.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/dimension_labels.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/data_type.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/json_binding/unit.h"
#include "tensorstore/internal/json_metadata_matching.h"
#include "tensorstore/internal/meta/integer_types.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_zarr3 {

namespace jb = ::tensorstore::internal_json_binding;

namespace {

#define TENSORSTORE_ZARR3_FOR_EACH_DATA_TYPE(X, ...)       \
  TENSORSTORE_FOR_EACH_BOOL_DATA_TYPE(X, ##__VA_ARGS__)    \
  TENSORSTORE_FOR_EACH_INT_DATA_TYPE(X, ##__VA_ARGS__)     \
  TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE(X, ##__VA_ARGS__)   \
  TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE(X, ##__VA_ARGS__) \
  /**/

constexpr std::array kSupportedDataTypes{
#define TENSORSTORE_INTERNAL_DO_DEF(T, ...) DataTypeId::T, /**/
    TENSORSTORE_ZARR3_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_DEF)
#undef TENSORSTORE_INTERNAL_DO_DEF
};

std::string GetSupportedDataTypes() {
  return absl::StrJoin(
      kSupportedDataTypes, ", ", [](std::string* out, DataTypeId id) {
        absl::StrAppend(out, kDataTypes[static_cast<int>(id)].name());
      });
}

}  // namespace

absl::Status ValidateDataType(DataType dtype) {
  if (!absl::c_linear_search(kSupportedDataTypes, dtype.id())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        dtype, " data type is not one of the supported data types: ",
        GetSupportedDataTypes()));
  }
  return absl::OkStatus();
}

namespace {

template <typename T>
constexpr auto FloatFillValueJsonBinder() {
  return [](auto is_loading, jb::NoOptions options, auto* obj,
            auto* j) -> absl::Status {
    using IntType = typename internal::uint_type<sizeof(T) * 8>::type;
    if constexpr (is_loading) {
      if (auto* s = j->template get_ptr<const std::string*>()) {
        if (*s == "Infinity") {
          *obj = std::numeric_limits<T>::infinity();
          return absl::OkStatus();
        }
        if (*s == "-Infinity") {
          *obj = -std::numeric_limits<T>::infinity();
          return absl::OkStatus();
        }
        if (*s == "NaN") {
          *obj = GetDefaultNaN<T>();
          return absl::OkStatus();
        }
        IntType raw_value;
        std::from_chars_result r;
        if (s->size() < 3 || s->size() > 2 + sizeof(T) * 2 || (*s)[0] != '0' ||
            (*s)[1] != 'x' ||
            (r = std::from_chars(s->data() + 2, s->data() + s->size(),
                                 raw_value, 16))
                    .ptr != s->data() + s->size() ||
            r.ec != std::errc{}) {
          return internal_json::ExpectedError(
              *j, "\"Infinity\", \"-Infinity\", \"NaN\", or hex string");
        }
        *obj = absl::bit_cast<T>(raw_value);
        return absl::OkStatus();
      } else if (j->is_number()) {
        *obj = static_cast<T>(j->template get<double>());
        return absl::OkStatus();
      } else {
        return internal_json::ExpectedError(*j, "floating-point number");
      }
    } else {
      using std::isfinite;
      if (isfinite(*obj)) {
        *j = static_cast<double>(*obj);
        return absl::OkStatus();
      }
      if (*obj == std::numeric_limits<T>::infinity()) {
        *j = "Infinity";
      } else if (*obj == -std::numeric_limits<T>::infinity()) {
        *j = "-Infinity";
      } else if (absl::bit_cast<IntType>(*obj) ==
                 absl::bit_cast<IntType>(GetDefaultNaN<T>())) {
        *j = "NaN";
      } else {
        *j = absl::StrFormat("0x%0*x", sizeof(T) * 2,
                             absl::bit_cast<IntType>(*obj));
      }
      return absl::OkStatus();
    }
  };
}

template <typename T>
constexpr auto ComplexFillValueJsonBinder() {
  return [](auto is_loading, const auto& options, auto* obj,
            auto* j) -> absl::Status {
    using FloatType = typename T::value_type;
    using QualifiedFloatType =
        std::conditional_t<is_loading, FloatType, const FloatType>;
    span<QualifiedFloatType, 2> components(
        reinterpret_cast<QualifiedFloatType*>(obj), 2);
    return jb::FixedSizeArray(FloatFillValueJsonBinder<FloatType>())(
        is_loading, options, &components, j);
  };
}

template <typename T>
constexpr auto SpecializedFillValueJsonBinder() {
  return jb::DefaultBinder<T>;
}

#define TENSORSTORE_INTERNAL_DO_DEFINE_SPECIALIZED_FILL_VALUE_PARSER(         \
    T, SPECIALIZATION)                                                        \
  template <>                                                                 \
  constexpr auto SpecializedFillValueJsonBinder<::tensorstore::dtypes::T>() { \
    return SPECIALIZATION<::tensorstore::dtypes::T>();                        \
  }                                                                           \
  /**/

TENSORSTORE_FOR_EACH_FLOAT_DATA_TYPE(
    TENSORSTORE_INTERNAL_DO_DEFINE_SPECIALIZED_FILL_VALUE_PARSER,
    FloatFillValueJsonBinder)

TENSORSTORE_FOR_EACH_COMPLEX_DATA_TYPE(
    TENSORSTORE_INTERNAL_DO_DEFINE_SPECIALIZED_FILL_VALUE_PARSER,
    ComplexFillValueJsonBinder)

#undef TENSORSTORE_INTERNAL_DO_DEFINE_SPECIALIZED_FILL_VALUE_PARSER

struct FillValueDataTypeFunctions {
  using Encode = absl::Status (*)(const void* value, ::nlohmann::json& j);
  Encode encode;
  using Decode = absl::Status (*)(void* value, ::nlohmann::json& j);
  Decode decode;
  template <typename T>
  static constexpr FillValueDataTypeFunctions Make() {
    FillValueDataTypeFunctions fs = {};
    fs.encode = +[](const void* value, ::nlohmann::json& j) -> absl::Status {
      return SpecializedFillValueJsonBinder<T>()(
          std::false_type{}, jb::NoOptions{}, static_cast<const T*>(value), &j);
    };
    fs.decode = +[](void* value, ::nlohmann::json& j) -> absl::Status {
      return SpecializedFillValueJsonBinder<T>()(
          std::true_type{}, jb::NoOptions{}, static_cast<T*>(value), &j);
    };
    return fs;
  }
};

constexpr std::array<FillValueDataTypeFunctions, kNumDataTypeIds>
    kFillValueDataTypeFunctions = [] {
      std::array<FillValueDataTypeFunctions, kNumDataTypeIds> functions = {};
#define TENSORSTORE_INTERNAL_DO_DEF(T, ...)                         \
  functions[static_cast<size_t>(DataTypeId::T)] =                   \
      FillValueDataTypeFunctions::Make<::tensorstore::dtypes::T>(); \
  /**/
      TENSORSTORE_ZARR3_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_DEF)
      // Add char_t support for string data types
      functions[static_cast<size_t>(DataTypeId::char_t)] =
          FillValueDataTypeFunctions::Make<::tensorstore::dtypes::char_t>();
      // byte_t is handled specially to use uint8_t functions
#undef TENSORSTORE_INTERNAL_DO_DEF
      return functions;
    }();

}  // namespace

FillValueJsonBinder::FillValueJsonBinder(ZarrDType dtype,
                                         bool allow_missing_dtype)
    : dtype(std::move(dtype)), allow_missing_dtype(allow_missing_dtype) {}

FillValueJsonBinder::FillValueJsonBinder(DataType data_type,
                                         bool allow_missing_dtype)
    : allow_missing_dtype(allow_missing_dtype) {
  dtype.has_fields = false;
  dtype.fields.resize(1);
  auto& field = dtype.fields[0];
  field.name.clear();
  field.outer_shape.clear();
  field.flexible_shape.clear();
  field.field_shape.clear();
  field.num_inner_elements = 1;
  field.byte_offset = 0;
  field.num_bytes = data_type->size;
  field.dtype = data_type;
  field.encoded_dtype = std::string(data_type.name());
}

absl::Status FillValueJsonBinder::operator()(
    std::true_type is_loading, internal_json_binding::NoOptions,
    std::vector<SharedArray<const void>>* obj, ::nlohmann::json* j) const {
  obj->resize(dtype.fields.size());
  if (dtype.fields.size() == 1) {
    // Special case: raw_bytes (single field with byte_t and flexible shape)
    if (dtype.fields[0].dtype.id() == DataTypeId::byte_t &&
        !dtype.fields[0].flexible_shape.empty()) {
      // Handle base64-encoded fill value for raw_bytes
      if (!j->is_string()) {
        return absl::InvalidArgumentError(
            "Expected base64-encoded string for raw_bytes fill_value");
      }
      std::string b64_decoded;
      if (!absl::Base64Unescape(j->get<std::string>(), &b64_decoded)) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Expected valid base64-encoded fill value, but received: ",
            j->dump()));
      }
      // Verify size matches expected byte array size
      Index expected_size = dtype.fields[0].num_inner_elements;
      if (static_cast<Index>(b64_decoded.size()) != expected_size) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Expected ", expected_size,
            " base64-encoded bytes for fill_value, but received ",
            b64_decoded.size(), " bytes"));
      }
      // Create fill value array
      auto fill_arr = AllocateArray(dtype.fields[0].field_shape, c_order,
                                   default_init, dtype.fields[0].dtype);
      std::memcpy(fill_arr.data(), b64_decoded.data(), b64_decoded.size());
      std::cout << "[DEBUG] Raw bytes fill_value parsed: shape=" << fill_arr.shape()
                << ", dtype=" << dtype.fields[0].dtype << std::endl;
      (*obj)[0] = std::move(fill_arr);
    } else {
      TENSORSTORE_RETURN_IF_ERROR(
          DecodeSingle(*j, dtype.fields[0].dtype, (*obj)[0]));
    }
  } else {
    // For structured types, handle both array format and base64-encoded string
    if (j->is_string()) {
      // Decode base64-encoded fill value for entire struct
      std::string b64_decoded;
      if (!absl::Base64Unescape(j->get<std::string>(), &b64_decoded)) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Expected valid base64-encoded fill value, but received: ",
            j->dump()));
      }
      // Verify size matches expected struct size
      if (static_cast<Index>(b64_decoded.size()) !=
          dtype.bytes_per_outer_element) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Expected ", dtype.bytes_per_outer_element,
            " base64-encoded bytes for fill_value, but received ",
            b64_decoded.size(), " bytes"));
      }
      // Extract per-field fill values from decoded bytes
      for (size_t i = 0; i < dtype.fields.size(); ++i) {
        const auto& field = dtype.fields[i];
        auto arr = AllocateArray(span<const Index, 0>{}, c_order, default_init,
                                 field.dtype);
        std::memcpy(arr.data(), b64_decoded.data() + field.byte_offset,
                    field.dtype->size);
        (*obj)[i] = std::move(arr);
      }
    } else if (j->is_array()) {
      if (j->size() != dtype.fields.size()) {
        return internal_json::ExpectedError(
            *j, tensorstore::StrCat("array of size ", dtype.fields.size()));
      }
      for (size_t i = 0; i < dtype.fields.size(); ++i) {
        TENSORSTORE_RETURN_IF_ERROR(
            DecodeSingle((*j)[i], dtype.fields[i].dtype, (*obj)[i]));
      }
    } else {
      return internal_json::ExpectedError(*j,
                                          "array or base64-encoded string");
    }
  }
  return absl::OkStatus();
}

absl::Status FillValueJsonBinder::operator()(
    std::false_type is_loading, internal_json_binding::NoOptions,
    const std::vector<SharedArray<const void>>* obj,
    ::nlohmann::json* j) const {
  if (dtype.fields.size() == 1) {
    return EncodeSingle((*obj)[0], dtype.fields[0].dtype, *j);
  }
  // Structured fill value
  *j = ::nlohmann::json::array();
  for (size_t i = 0; i < dtype.fields.size(); ++i) {
    ::nlohmann::json item;
    TENSORSTORE_RETURN_IF_ERROR(
        EncodeSingle((*obj)[i], dtype.fields[i].dtype, item));
    j->push_back(std::move(item));
  }
  return absl::OkStatus();
}

absl::Status FillValueJsonBinder::DecodeSingle(::nlohmann::json& j,
                                               DataType data_type,
                                               SharedArray<const void>& out) const {
  if (!data_type.valid()) {
    if (allow_missing_dtype) {
      out = SharedArray<const void>();
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError(
        "data_type must be specified before fill_value");
  }
  auto arr =
      AllocateArray(span<const Index, 0>{}, c_order, default_init, data_type);
  void* data = arr.data();
  out = std::move(arr);
  // Special handling for byte_t: use uint8_t functions since they're binary compatible
  auto type_id = data_type.id();
  if (type_id == DataTypeId::byte_t) {
    type_id = DataTypeId::uint8_t;
  }

  const auto& functions =
      kFillValueDataTypeFunctions[static_cast<size_t>(type_id)];
  if (!functions.decode) {
    if (allow_missing_dtype) {
      out = SharedArray<const void>();
      return absl::OkStatus();
    }
    return absl::FailedPreconditionError(
        "fill_value unsupported for specified data_type");
  }
  return functions.decode(data, j);
}

absl::Status FillValueJsonBinder::EncodeSingle(
    const SharedArray<const void>& arr, DataType data_type,
    ::nlohmann::json& j) const {
  if (!data_type.valid()) {
    return absl::InvalidArgumentError(
        "data_type must be specified before fill_value");
  }
  // Special handling for byte_t: use uint8_t functions since they're binary compatible
  auto type_id = data_type.id();
  if (type_id == DataTypeId::byte_t) {
    type_id = DataTypeId::uint8_t;
  }

  const auto& functions =
      kFillValueDataTypeFunctions[static_cast<size_t>(type_id)];
  if (!functions.encode) {
    return absl::FailedPreconditionError(
        "fill_value unsupported for specified data_type");
  }
  return functions.encode(arr.data(), j);
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(ChunkKeyEncoding, [](auto is_loading,
                                                            const auto& options,
                                                            auto* obj,
                                                            auto* j) {
  using Self = ChunkKeyEncoding;
  return jb::Object(NameConfigurationJsonBinder(
      jb::Projection<&Self::kind>(jb::Enum<Self::Kind, std::string_view>({
          {Self::kDefault, "default"},
          {Self::kV2, "v2"},
      })),
      jb::Object(jb::Member("separator",
                            jb::Projection<&Self::separator>(jb::DefaultValue(
                                [outer = obj](auto* obj) -> absl::Status {
                                  switch (outer->kind) {
                                    case Self::kDefault:
                                      *obj = '/';
                                      break;
                                    case Self::kV2:
                                      *obj = '.';
                                      break;
                                    default:
                                      ABSL_UNREACHABLE();
                                  }
                                  return absl::OkStatus();
                                },
                                jb::Enum<char, std::string_view>({
                                    {'.', "."},
                                    {'/', "/"},
                                }))))))  //
                    )(is_loading, options, obj, j);
})

namespace {

constexpr auto UnknownExtensionAttributesJsonBinder =
    jb::Validate([](const auto& options, auto* obj) {
      for (const auto& [key, value] : *obj) {
        if (value.is_object()) {
          const auto& obj =
              value.template get_ref<const ::nlohmann::json::object_t&>();
          auto it = obj.find("must_understand");
          if (it != obj.end() && it->second == false) {
            continue;
          }
        }
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Unsupported metadata field ", tensorstore::QuoteString(key),
            " is not marked {\"must_understand\": false}"));
      }
      return absl::OkStatus();
    });

template <bool Constraints, bool CompatibilityOnly = false>
constexpr auto MetadataJsonBinder = [] {
  constexpr auto maybe_optional = [](auto binder) {
    if constexpr (Constraints) {
      return jb::Optional(binder);
    } else {
      return binder;
    }
  };

  constexpr auto maybe_optional_member = [](auto name, auto binder) {
    if constexpr (Constraints) {
      return jb::OptionalMember(name, std::move(binder));
    } else {
      return jb::Member(name, std::move(binder));
    }
  };

  constexpr auto non_compatibility_field = [](auto binder) {
    if constexpr (CompatibilityOnly) {
      return jb::Sequence();
    } else {
      return binder;
    }
  };

  return [=](auto is_loading, const auto& options, auto* obj, auto* j) {
    using Self = absl::remove_cvref_t<decltype(*obj)>;
    DimensionIndex* rank = nullptr;
    if constexpr (is_loading) {
      rank = &obj->rank;
    }

    auto ensure_data_type = [&]() -> Result<ZarrDType> {
      if constexpr (std::is_same_v<Self, ZarrMetadata>) {
        return obj->data_type;
      }
      if constexpr (std::is_same_v<Self, ZarrMetadataConstraints>) {
        // data_type is wrapped in std::optional<>
        if (obj->data_type) {
          return *obj->data_type;
        }
      }
      // The return here works around a gcc flow analysis bug.
      return absl::InvalidArgumentError(
          "must be specified in conjunction with \"data_type\"");
    };

    return jb::Object(
        jb::Member("zarr_format", jb::Projection<&Self::zarr_format>(
                                      maybe_optional(jb::Integer<int>(3, 3)))),
        maybe_optional_member("node_type",
                              jb::Constant([] { return "array"; })),
        jb::Member("data_type",
                   jb::Projection<&Self::data_type>(maybe_optional(
                       jb::DefaultBinder<>))),
        jb::Member(
            "fill_value",
            jb::Projection<&Self::fill_value>(maybe_optional(
                [&](auto is_loading, const auto& options, auto* obj, auto* j) {
                  TENSORSTORE_ASSIGN_OR_RETURN(auto data_type,
                                               ensure_data_type());
                  constexpr bool allow_missing_dtype =
                      std::is_same_v<Self, ZarrMetadata>;
                  return FillValueJsonBinder{data_type, allow_missing_dtype}(
                      is_loading, options, obj, j);
                }))),
        non_compatibility_field(
            jb::Member("shape", jb::Projection<&Self::shape>(
                                    maybe_optional(jb::ShapeVector(rank))))),
        jb::Member("dimension_names",
                   jb::Projection<&Self::dimension_names>(maybe_optional(
                       [rank](auto is_loading, const auto& options, auto* obj,
                              auto* j) {
                         if constexpr (is_loading) {
                           if (j->is_discarded() && *rank != dynamic_rank) {
                             obj->resize(*rank);
                             return absl::OkStatus();
                           }
                         } else {
                           if (std::none_of(obj->begin(), obj->end(),
                                            [](const auto& x) {
                                              return x.has_value();
                                            })) {
                             return absl::OkStatus();
                           }
                         }
                         return jb::DimensionIndexedVector(
                             rank, jb::OptionalWithNull())(is_loading, options,
                                                           obj, j);
                       }))),
        jb::Member("chunk_key_encoding",
                   jb::Projection<&Self::chunk_key_encoding>(
                       maybe_optional(jb::DefaultBinder<>))),
        jb::Member(
            "chunk_grid",
            jb::Projection<&Self::chunk_shape>(maybe_optional(jb::Object(
                jb::Member("name", jb::Constant([] { return "regular"; })),
                jb::Member(
                    "configuration",
                    jb::Object(jb::Member("chunk_shape",
                                          jb::ChunkShapeVector(rank))))  //
                )))),                                                    //
        jb::Member("codecs", jb::Projection<&Self::codec_specs>(maybe_optional(
                                 ZarrCodecChainJsonBinder<Constraints>))),
        // Allow empty storage_transformers list.
        jb::LoadSave(jb::OptionalMember(
            "storage_transformers",
            jb::Compose<::nlohmann::json::array_t>(
                [](auto is_loading, const auto& options, auto* obj, auto* j) {
                  if (!j->empty()) {
                    return absl::InvalidArgumentError(
                        "No storage transformers supported");
                  }
                  return absl::OkStatus();
                }))),
        non_compatibility_field(jb::Member(
            "attributes",
            jb::OptionalObject(jb::Sequence(
                jb::Member("dimension_units",
                           jb::Projection<&Self::dimension_units>(
                               jb::Optional(jb::DimensionIndexedVector(
                                   rank, jb::OptionalWithNull(
                                             jb::StringOnlyUnitJsonBinder))))),
                jb::Projection<&Self::user_attributes>())))),
        non_compatibility_field(
            jb::Projection<&Self::unknown_extension_attributes>(
                UnknownExtensionAttributesJsonBinder))  //
        )(is_loading, options, obj, j);
  };
};

}  // namespace

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    ZarrMetadata, jb::Validate([](const auto& options,
                                  auto* obj) { return ValidateMetadata(*obj); },
                               MetadataJsonBinder</*Constraints=*/false>()))

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    ZarrMetadataConstraints, MetadataJsonBinder</*Constraints=*/true>())

std::string ZarrMetadata::GetCompatibilityKey() const {
  return jb::ToJson(*this, MetadataJsonBinder</*Constraints=*/false,
                                              /*CompatibilityOnly=*/true>())
      .value()
      .dump();
}

absl::Status ValidateMetadata(ZarrMetadata& metadata) {
  // Determine if this is a structured type with multiple fields
  const bool is_structured =
      metadata.data_type.fields.size() > 1 ||
      (metadata.data_type.fields.size() == 1 &&
       !metadata.data_type.fields[0].outer_shape.empty());

  // Build the codec shape - for structured types, include bytes dimension
  std::vector<Index> codec_shape(metadata.chunk_shape.begin(),
                                 metadata.chunk_shape.end());
  if (is_structured) {
    codec_shape.push_back(metadata.data_type.bytes_per_outer_element);
  }

  if (!metadata.codecs) {
    ArrayCodecResolveParameters decoded;
    if (!is_structured) {
      decoded.dtype = metadata.data_type.fields[0].dtype;
      decoded.rank = metadata.rank;
    } else {
      // For structured types, use byte dtype with extra dimension
      decoded.dtype = dtype_v<std::byte>;
      decoded.rank = metadata.rank + 1;
    }
    // Fill value for codec resolve might be complex.
    // For structured types, create a byte fill value
    if (metadata.fill_value.size() == 1 && !is_structured) {
      decoded.fill_value = metadata.fill_value[0];
    }

    BytesCodecResolveParameters encoded;
    TENSORSTORE_ASSIGN_OR_RETURN(
        metadata.codecs,
        metadata.codec_specs.Resolve(std::move(decoded), encoded));
  }

  // Get codec chunk layout info.
  ArrayDataTypeAndShapeInfo array_info;
  if (!is_structured) {
    array_info.dtype = metadata.data_type.fields[0].dtype;
    array_info.rank = metadata.rank;
    std::copy_n(metadata.chunk_shape.begin(), metadata.rank,
                array_info.shape.emplace().begin());
  } else {
    array_info.dtype = dtype_v<std::byte>;
    array_info.rank = metadata.rank + 1;
    auto& shape = array_info.shape.emplace();
    std::copy_n(metadata.chunk_shape.begin(), metadata.rank, shape.begin());
    shape[metadata.rank] = metadata.data_type.bytes_per_outer_element;
  }

  ArrayCodecChunkLayoutInfo layout_info;
  TENSORSTORE_RETURN_IF_ERROR(
      metadata.codec_specs.GetDecodedChunkLayout(array_info, layout_info));
  if (layout_info.inner_order) {
    std::copy_n(layout_info.inner_order->begin(), metadata.rank,
                metadata.inner_order.begin());
  } else {
    std::iota(metadata.inner_order.begin(),
              metadata.inner_order.begin() + metadata.rank,
              static_cast<DimensionIndex>(0));
  }

  TENSORSTORE_ASSIGN_OR_RETURN(metadata.codec_state,
                               metadata.codecs->Prepare(codec_shape));
  return absl::OkStatus();
}

absl::Status ValidateMetadata(const ZarrMetadata& metadata,
                              const ZarrMetadataConstraints& constraints) {
  using internal::MetadataMismatchError;
  if (constraints.data_type) {
    // Compare ZarrDType
    if (::nlohmann::json(*constraints.data_type) !=
        ::nlohmann::json(metadata.data_type)) {
      return MetadataMismatchError(
          "data_type", ::nlohmann::json(*constraints.data_type).dump(),
          ::nlohmann::json(metadata.data_type).dump());
    }
  }
  if (constraints.fill_value) {
    // Compare vector of arrays
    if (constraints.fill_value->size() != metadata.fill_value.size()) {
      return MetadataMismatchError("fill_value size",
                                   constraints.fill_value->size(),
                                   metadata.fill_value.size());
    }
    for (size_t i = 0; i < metadata.fill_value.size(); ++i) {
      if (!AreArraysIdenticallyEqual((*constraints.fill_value)[i],
                                     metadata.fill_value[i])) {
        auto binder = FillValueJsonBinder{metadata.data_type};
        auto constraint_json =
            jb::ToJson(*constraints.fill_value, binder).value();
        auto metadata_json =
            jb::ToJson(metadata.fill_value, binder).value();
        return MetadataMismatchError("fill_value", constraint_json,
                                     metadata_json);
      }
    }
  }
  if (constraints.shape && *constraints.shape != metadata.shape) {
    return MetadataMismatchError("shape", *constraints.shape, metadata.shape);
  }
  if (constraints.chunk_shape &&
      *constraints.chunk_shape != metadata.chunk_shape) {
    return MetadataMismatchError("chunk_shape", *constraints.chunk_shape,
                                 metadata.chunk_shape);
  }
  if (constraints.chunk_key_encoding &&
      *constraints.chunk_key_encoding != metadata.chunk_key_encoding) {
    return MetadataMismatchError("chunk_key_encoding",
                                 *constraints.chunk_key_encoding,
                                 metadata.chunk_key_encoding);
  }
  if (constraints.codec_specs) {
    ZarrCodecChainSpec codecs_copy = metadata.codec_specs;
    TENSORSTORE_RETURN_IF_ERROR(
        codecs_copy.MergeFrom(*constraints.codec_specs, /*strict=*/true),
        tensorstore::MaybeAnnotateStatus(_, "Mismatch in \"codecs\""));
  }
  if (constraints.dimension_names &&
      *constraints.dimension_names != metadata.dimension_names) {
    return MetadataMismatchError(
        "dimension_names", jb::ToJson(*constraints.dimension_names).value(),
        jb::ToJson(metadata.dimension_names).value());
  }
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateMetadataSubset(constraints.user_attributes,
                                       metadata.user_attributes),
      tensorstore::MaybeAnnotateStatus(_, "Mismatch in \"attributes\""));
  if (constraints.dimension_units) {
    for (DimensionIndex i = 0, rank = metadata.rank; i < rank; ++i) {
      const auto& constraint_unit = (*constraints.dimension_units)[i];
      if (!constraint_unit) continue;
      if (!metadata.dimension_units ||
          (*metadata.dimension_units)[i] != *constraint_unit) {
        const auto binder = jb::Optional(
            jb::Array(jb::OptionalWithNull(jb::StringOnlyUnitJsonBinder)));
        return MetadataMismatchError(
            "dimension_units",
            jb::ToJson(constraints.dimension_units, binder).value(),
            jb::ToJson(metadata.dimension_units, binder).value());
      }
    }
  }
  return internal::ValidateMetadataSubset(
      constraints.unknown_extension_attributes,
      metadata.unknown_extension_attributes);
}

namespace {
std::string GetFieldNames(const ZarrDType& dtype) {
  std::vector<std::string> field_names;
  for (const auto& field : dtype.fields) {
    field_names.push_back(field.name);
  }
  return ::nlohmann::json(field_names).dump();
}
}  // namespace

constexpr size_t kVoidFieldIndex = size_t(-1);

Result<size_t> GetFieldIndex(const ZarrDType& dtype,
                             std::string_view selected_field) {
  // Special case: "<void>" requests raw byte access (works for any dtype)
  if (selected_field == "<void>") {
    if (dtype.fields.empty()) {
      return absl::FailedPreconditionError(
          "Requested field \"<void>\" but dtype has no fields");
    }
    return kVoidFieldIndex;
  }

  if (selected_field.empty()) {
    if (dtype.fields.size() != 1) {
      return absl::FailedPreconditionError(tensorstore::StrCat(
          "Must specify a \"field\" that is one of: ", GetFieldNames(dtype)));
    }
    return 0;
  }
  if (!dtype.has_fields) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("Requested field ", QuoteString(selected_field),
                            " but dtype does not have named fields"));
  }
  for (size_t field_index = 0; field_index < dtype.fields.size();
       ++field_index) {
    if (dtype.fields[field_index].name == selected_field) return field_index;
  }
  return absl::FailedPreconditionError(
      tensorstore::StrCat("Requested field ", QuoteString(selected_field),
                          " is not one of: ", GetFieldNames(dtype)));
}

SpecRankAndFieldInfo GetSpecRankAndFieldInfo(const ZarrMetadata& metadata,
                                             size_t field_index) {
  SpecRankAndFieldInfo info;
  info.chunked_rank = metadata.rank;
  info.field = &metadata.data_type.fields[field_index];
  if (!info.field->field_shape.empty()) {
    info.chunked_rank += info.field->field_shape.size();
  }
  return info;
}

Result<IndexDomain<>> GetEffectiveDomain(
    const SpecRankAndFieldInfo& info,
    std::optional<tensorstore::span<const Index>> metadata_shape,
    std::optional<span<const std::optional<std::string>>> dimension_names,
    const Schema& schema, bool* dimension_names_used) {
  const DimensionIndex rank = info.chunked_rank;
  if (dimension_names_used) *dimension_names_used = false;
  auto domain = schema.domain();
  if (!metadata_shape && !dimension_names && !domain.valid()) {
    if (schema.rank() == 0) return {std::in_place, 0};
    return {std::in_place};
  }

  assert(RankConstraint::EqualOrUnspecified(schema.rank(), rank));
  IndexDomainBuilder builder(std::max(schema.rank().rank, rank));
  if (metadata_shape) {
    if (static_cast<DimensionIndex>(metadata_shape->size()) < rank &&
        info.field && !info.field->field_shape.empty() &&
        static_cast<DimensionIndex>(metadata_shape->size() +
                                    info.field->field_shape.size()) == rank) {
      std::vector<Index> full_shape(metadata_shape->begin(),
                                    metadata_shape->end());
      full_shape.insert(full_shape.end(), info.field->field_shape.begin(),
                        info.field->field_shape.end());
      builder.shape(full_shape);
      DimensionSet implicit_upper_bounds(false);
      for (size_t i = 0; i < metadata_shape->size(); ++i) {
        implicit_upper_bounds[i] = true;
      }
      builder.implicit_upper_bounds(implicit_upper_bounds);
    } else {
      builder.shape(*metadata_shape);
      builder.implicit_upper_bounds(true);
    }
  } else {
    builder.origin(GetConstantVector<Index, 0>(builder.rank()));
  }
  if (dimension_names) {
    std::string_view normalized_dimension_names[kMaxRank];
    for (DimensionIndex i = 0; i < rank; ++i) {
      if (const auto& name = (*dimension_names)[i]; name.has_value()) {
        normalized_dimension_names[i] = *name;
      }
    }
    if (internal::ValidateDimensionLabelsAreUnique(
            span<const std::string_view>(&normalized_dimension_names[0], rank))
            .ok()) {
      builder.labels(
          span<const std::string_view>(&normalized_dimension_names[0], rank));
      if (dimension_names_used) *dimension_names_used = true;
    }
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto domain_from_metadata, builder.Finalize());
  TENSORSTORE_ASSIGN_OR_RETURN(
      domain, MergeIndexDomains(domain, domain_from_metadata),
      internal::ConvertInvalidArgumentToFailedPrecondition(
          tensorstore::MaybeAnnotateStatus(
              _, "Mismatch between metadata and schema")));
  return WithImplicitDimensions(domain, false, true);
}

Result<IndexDomain<>> GetEffectiveDomain(
    const ZarrMetadataConstraints& metadata_constraints, const Schema& schema,
    bool* dimension_names_used) {
  SpecRankAndFieldInfo info;
  info.chunked_rank = metadata_constraints.rank;
  if (info.chunked_rank == dynamic_rank && metadata_constraints.shape) {
    info.chunked_rank = metadata_constraints.shape->size();
  }

  std::optional<span<const Index>> shape_span;
  if (metadata_constraints.shape) {
    shape_span.emplace(metadata_constraints.shape->data(),
                       metadata_constraints.shape->size());
  }
  std::optional<span<const std::optional<std::string>>> names_span;
  if (metadata_constraints.dimension_names) {
    names_span.emplace(metadata_constraints.dimension_names->data(),
                       metadata_constraints.dimension_names->size());
  }

  return GetEffectiveDomain(info, shape_span, names_span, schema,
                            dimension_names_used);
}

absl::Status SetChunkLayoutFromMetadata(
    const SpecRankAndFieldInfo& info,
    std::optional<span<const Index>> chunk_shape,
    const ZarrCodecChainSpec* codecs, ChunkLayout& chunk_layout) {
  const DimensionIndex rank = info.chunked_rank;
  if (rank == dynamic_rank) {
    return absl::OkStatus();
  }
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(RankConstraint(rank)));
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(
      ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(rank))));

  if (chunk_shape) {
    assert(chunk_shape->size() == rank);
    TENSORSTORE_RETURN_IF_ERROR(
        chunk_layout.Set(ChunkLayout::WriteChunkShape(*chunk_shape)));
  }

  if (codecs) {
    ArrayDataTypeAndShapeInfo array_info;
    array_info.dtype = info.field ? info.field->dtype : dtype_v<std::byte>;
    array_info.rank = rank;
    if (chunk_shape) {
      std::copy_n(chunk_shape->begin(), rank,
                  array_info.shape.emplace().begin());
    }
    ArrayCodecChunkLayoutInfo layout_info;
    TENSORSTORE_RETURN_IF_ERROR(
        codecs->GetDecodedChunkLayout(array_info, layout_info));
    if (layout_info.inner_order) {
      TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(ChunkLayout::InnerOrder(
          span<const DimensionIndex>(layout_info.inner_order->data(), rank))));
    }
    if (layout_info.read_chunk_shape) {
      TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(ChunkLayout::ReadChunkShape(
          span<const Index>(layout_info.read_chunk_shape->data(), rank))));
    }
    if (layout_info.codec_chunk_shape) {
      TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(ChunkLayout::CodecChunkShape(
          span<const Index>(layout_info.codec_chunk_shape->data(), rank))));
    }
  }

  return absl::OkStatus();
}

absl::Status SetChunkLayoutFromMetadata(
    DataType dtype, DimensionIndex rank,
    std::optional<span<const Index>> chunk_shape,
    const ZarrCodecChainSpec* codecs, ChunkLayout& chunk_layout) {
  SpecRankAndFieldInfo info;
  info.chunked_rank = rank;
  info.field = nullptr;
  return SetChunkLayoutFromMetadata(info, chunk_shape, codecs, chunk_layout);
}

Result<ChunkLayout> GetEffectiveChunkLayout(
    const ZarrMetadataConstraints& metadata_constraints, const Schema& schema) {
  // Approximation: assume whole array access or simple array
  SpecRankAndFieldInfo info;
  info.chunked_rank = std::max(metadata_constraints.rank, schema.rank().rank);
  if (info.chunked_rank == dynamic_rank && metadata_constraints.shape) {
    info.chunked_rank = metadata_constraints.shape->size();
  }
  if (info.chunked_rank == dynamic_rank && metadata_constraints.chunk_shape) {
    info.chunked_rank = metadata_constraints.chunk_shape->size();
  }
  // We can't easily know field info from constraints unless we parse data_type.
  // If data_type is present and has 1 field, we can check it.
  // For now, basic implementation.

  ChunkLayout chunk_layout = schema.chunk_layout();
  std::optional<span<const Index>> chunk_shape_span;
  if (metadata_constraints.chunk_shape) {
    chunk_shape_span.emplace(metadata_constraints.chunk_shape->data(),
                             metadata_constraints.chunk_shape->size());
  }
  TENSORSTORE_RETURN_IF_ERROR(SetChunkLayoutFromMetadata(
      info, chunk_shape_span,
      metadata_constraints.codec_specs ? &*metadata_constraints.codec_specs
                                       : nullptr,
      chunk_layout));
  return chunk_layout;
}

Result<DimensionUnitsVector> GetDimensionUnits(
    DimensionIndex rank,
    const std::optional<DimensionUnitsVector>& dimension_units_constraints) {
  if (dimension_units_constraints) return *dimension_units_constraints;
  if (rank == dynamic_rank) return {std::in_place};
  return DimensionUnitsVector(rank);
}

Result<DimensionUnitsVector> GetEffectiveDimensionUnits(
    DimensionIndex rank,
    const std::optional<DimensionUnitsVector>& dimension_units_constraints,
    Schema::DimensionUnits schema_units) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto dimension_units,
      GetDimensionUnits(rank, dimension_units_constraints));
  if (schema_units.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(
        tensorstore::MergeDimensionUnits(dimension_units, schema_units));
  }
  return dimension_units;
}

Result<internal::CodecDriverSpec::PtrT<TensorStoreCodecSpec>> GetEffectiveCodec(
    const ZarrMetadataConstraints& metadata_constraints, const Schema& schema) {
  auto codec_spec = internal::CodecDriverSpec::Make<TensorStoreCodecSpec>();
  codec_spec->codecs = metadata_constraints.codec_specs;
  TENSORSTORE_RETURN_IF_ERROR(codec_spec->MergeFrom(schema.codec()));
  return codec_spec;
}

CodecSpec GetCodecFromMetadata(const ZarrMetadata& metadata) {
  auto codec = internal::CodecDriverSpec::Make<TensorStoreCodecSpec>();
  codec->codecs = metadata.codec_specs;
  return codec;
}

absl::Status ValidateMetadataSchema(const ZarrMetadata& metadata,
                                    size_t field_index, const Schema& schema) {
  auto info = GetSpecRankAndFieldInfo(metadata, field_index);
  const auto& field = metadata.data_type.fields[field_index];

  if (!RankConstraint::EqualOrUnspecified(schema.rank(), info.chunked_rank)) {
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Rank specified by schema (", schema.rank(),
        ") does not match rank specified by metadata (", info.chunked_rank,
        ")"));
  }

  if (schema.domain().valid()) {
    std::optional<span<const Index>> metadata_shape_span;
    metadata_shape_span.emplace(metadata.shape.data(), metadata.shape.size());
    std::optional<span<const std::optional<std::string>>> dimension_names_span;
    dimension_names_span.emplace(metadata.dimension_names.data(),
                                 metadata.dimension_names.size());
    TENSORSTORE_RETURN_IF_ERROR(GetEffectiveDomain(
        info, metadata_shape_span, dimension_names_span, schema,
        /*dimension_names_used=*/nullptr));
  }

  if (auto dtype = schema.dtype();
      !IsPossiblySameDataType(field.dtype, dtype)) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("data_type from metadata (", field.dtype,
                            ") does not match dtype in schema (", dtype, ")"));
  }

  if (schema.chunk_layout().rank() != dynamic_rank) {
    ChunkLayout chunk_layout = schema.chunk_layout();
    std::optional<span<const Index>> chunk_shape_span;
    chunk_shape_span.emplace(metadata.chunk_shape.data(),
                             metadata.chunk_shape.size());
    TENSORSTORE_RETURN_IF_ERROR(SetChunkLayoutFromMetadata(
        info, chunk_shape_span, &metadata.codec_specs, chunk_layout));
    if (chunk_layout.codec_chunk_shape().hard_constraint) {
      return absl::InvalidArgumentError("codec_chunk_shape not supported");
    }
  }

  if (auto schema_fill_value = schema.fill_value(); schema_fill_value.valid()) {
    const auto& fill_value = metadata.fill_value[field_index];
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto broadcast_fill_value,
        tensorstore::BroadcastArray(schema_fill_value, span<const Index>{}));
    TENSORSTORE_ASSIGN_OR_RETURN(
        SharedArray<const void> converted_fill_value,
        tensorstore::MakeCopy(std::move(broadcast_fill_value),
                              skip_repeated_elements, field.dtype));
    if (!AreArraysIdenticallyEqual(converted_fill_value, fill_value)) {
      auto binder = FillValueJsonBinder{metadata.data_type};
      // Error message generation might be tricky with binder
      return absl::FailedPreconditionError(tensorstore::StrCat(
          "Invalid fill_value: schema requires fill value of ",
          schema_fill_value, ", but metadata specifies fill value of ",
          fill_value));
    }
  }

  if (auto schema_codec = schema.codec(); schema_codec.valid()) {
    auto codec = GetCodecFromMetadata(metadata);
    TENSORSTORE_RETURN_IF_ERROR(
        codec.MergeFrom(schema_codec),
        tensorstore::MaybeAnnotateStatus(
            _, "codec from metadata does not match codec in schema"));
  }

  if (auto schema_dimension_units = schema.dimension_units();
      schema_dimension_units.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(
        GetEffectiveDimensionUnits(metadata.rank, metadata.dimension_units,
                                   schema_dimension_units),
        tensorstore::MaybeAnnotateStatus(
            internal::ConvertInvalidArgumentToFailedPrecondition(_),
            "dimension_units from metadata does not match dimension_units in "
            "schema"));
  }

  return absl::OkStatus();
}

absl::Status ValidateMetadataSchema(const ZarrMetadata& metadata,
                                    const Schema& schema) {
  return ValidateMetadataSchema(metadata, /*field_index=*/0, schema);
}

Result<std::shared_ptr<const ZarrMetadata>> GetNewMetadata(
    const ZarrMetadataConstraints& metadata_constraints, const Schema& schema,
    std::string_view selected_field) {
  auto metadata = std::make_shared<ZarrMetadata>();

  metadata->zarr_format = metadata_constraints.zarr_format.value_or(3);
  metadata->chunk_key_encoding =
      metadata_constraints.chunk_key_encoding.value_or(ChunkKeyEncoding{
          /*.kind=*/ChunkKeyEncoding::kDefault, /*.separator=*/'/'});

  // Determine data type first
  if (metadata_constraints.data_type) {
    metadata->data_type = *metadata_constraints.data_type;
  } else if (!selected_field.empty()) {
    return absl::InvalidArgumentError(
        "\"dtype\" must be specified in \"metadata\" if \"field\" is "
        "specified");
  } else if (auto dtype = schema.dtype(); dtype.valid()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        static_cast<ZarrDType::BaseDType&>(
            metadata->data_type.fields.emplace_back()),
        ChooseBaseDType(dtype));
    metadata->data_type.has_fields = false;
    TENSORSTORE_RETURN_IF_ERROR(ValidateDType(metadata->data_type));
  } else {
    return absl::InvalidArgumentError("dtype must be specified");
  }

  TENSORSTORE_ASSIGN_OR_RETURN(
      size_t field_index, GetFieldIndex(metadata->data_type, selected_field));
  SpecRankAndFieldInfo info;
  info.field = &metadata->data_type.fields[field_index];
  info.chunked_rank = metadata_constraints.rank;
  if (info.chunked_rank == dynamic_rank && metadata_constraints.shape) {
    info.chunked_rank = metadata_constraints.shape->size();
  }
  if (info.chunked_rank == dynamic_rank &&
      schema.rank().rank != dynamic_rank) {
    info.chunked_rank = schema.rank().rank;
  }

  // Set domain
  bool dimension_names_used = false;
  std::optional<span<const Index>> constraint_shape_span;
  if (metadata_constraints.shape) {
    constraint_shape_span.emplace(metadata_constraints.shape->data(),
                                  metadata_constraints.shape->size());
  }
  std::optional<span<const std::optional<std::string>>> constraint_names_span;
  if (metadata_constraints.dimension_names) {
    constraint_names_span.emplace(
        metadata_constraints.dimension_names->data(),
        metadata_constraints.dimension_names->size());
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto domain, GetEffectiveDomain(info, constraint_shape_span,
                                      constraint_names_span, schema,
                                      &dimension_names_used));
  if (!domain.valid() || !IsFinite(domain.box())) {
    return absl::InvalidArgumentError("domain must be specified");
  }
  const DimensionIndex rank = domain.rank();
  metadata->rank = rank;
  info.chunked_rank = rank;
  metadata->shape.assign(domain.shape().begin(),
                         domain.shape().begin() + rank);
  metadata->dimension_names.assign(domain.labels().begin(),
                                   domain.labels().begin() + rank);

  for (DimensionIndex i = 0; i < rank; ++i) {
    auto& name = metadata->dimension_names[i];
    if (!name || !name->empty()) continue;
    if (dimension_names_used && metadata_constraints.dimension_names &&
        (*metadata_constraints.dimension_names)[i]) {
      assert((*metadata_constraints.dimension_names)[i]->empty());
      continue;
    }
    name = std::nullopt;
  }

  if (metadata_constraints.fill_value) {
    metadata->fill_value = *metadata_constraints.fill_value;
  } else if (auto fill_value = schema.fill_value(); fill_value.valid()) {
    // Assuming single field if setting from schema
    if (metadata->data_type.fields.size() != 1) {
      return absl::InvalidArgumentError(
          "Cannot specify fill_value through schema for structured zarr data "
          "type");
    }
    const auto status = [&] {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto broadcast_fill_value,
          tensorstore::BroadcastArray(fill_value, span<const Index>{}));
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto converted_fill_value,
          tensorstore::MakeCopy(std::move(broadcast_fill_value),
                                skip_repeated_elements,
                                metadata->data_type.fields[0].dtype));
      metadata->fill_value.push_back(std::move(converted_fill_value));
      return absl::OkStatus();
    }();
    TENSORSTORE_RETURN_IF_ERROR(
        status, tensorstore::MaybeAnnotateStatus(_, "Invalid fill_value"));
  } else {
    metadata->fill_value.resize(metadata->data_type.fields.size());
    for (size_t i = 0; i < metadata->fill_value.size(); ++i) {
      metadata->fill_value[i] = tensorstore::AllocateArray(
          /*shape=*/span<const Index>(), c_order, value_init,
          metadata->data_type.fields[i].dtype);
    }
  }

  metadata->user_attributes = metadata_constraints.user_attributes;
  metadata->unknown_extension_attributes =
      metadata_constraints.unknown_extension_attributes;

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto dimension_units,
      GetEffectiveDimensionUnits(rank, metadata_constraints.dimension_units,
                                 schema.dimension_units()),
      tensorstore::MaybeAnnotateStatus(_, "Invalid dimension_units"));
  if (std::any_of(dimension_units.begin(), dimension_units.end(),
                  [](const auto& unit) { return unit.has_value(); })) {
    metadata->dimension_units = std::move(dimension_units);
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto codec_spec,
                               GetEffectiveCodec(metadata_constraints, schema));

  ArrayCodecResolveParameters decoded;
  if (metadata->data_type.fields.size() == 1 &&
      metadata->data_type.fields[0].outer_shape.empty()) {
    decoded.dtype = metadata->data_type.fields[0].dtype;
  } else {
    decoded.dtype = dtype_v<std::byte>;
  }
  decoded.rank = metadata->rank;
  if (metadata->fill_value.size() == 1)
    decoded.fill_value = metadata->fill_value[0];

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto chunk_layout, GetEffectiveChunkLayout(metadata_constraints, schema));
  metadata->chunk_shape.resize(rank);

  if (auto inner_order = chunk_layout.inner_order(); inner_order.valid()) {
    std::copy(inner_order.begin(), inner_order.end(),
              decoded.inner_order.emplace().begin());
  }

  span<Index> read_chunk_shape(decoded.read_chunk_shape.emplace().data(),
                               metadata->rank);

  TENSORSTORE_RETURN_IF_ERROR(internal::ChooseReadWriteChunkShapes(
      chunk_layout.read_chunk(), chunk_layout.write_chunk(), domain.box(),
      read_chunk_shape, metadata->chunk_shape));

  if (!internal::RangesEqual(span<const Index>(metadata->chunk_shape),
                             span<const Index>(read_chunk_shape))) {
    if (!codec_spec->codecs || codec_spec->codecs->sharding_height() == 0) {
      auto sharding_codec =
          internal::MakeIntrusivePtr<ShardingIndexedCodecSpec>(
              ShardingIndexedCodecSpec::Options{
                  std::vector<Index>(read_chunk_shape.begin(),
                                     read_chunk_shape.end()),
                  std::nullopt, codec_spec->codecs});
      auto& codec_chain_spec = codec_spec->codecs.emplace();
      codec_chain_spec.array_to_bytes = std::move(sharding_codec);
    }
  }

  const auto set_up_codecs =
      [&](const ZarrCodecChainSpec& codec_specs) -> absl::Status {
    BytesCodecResolveParameters encoded;
    TENSORSTORE_ASSIGN_OR_RETURN(
        metadata->codecs, codec_specs.Resolve(std::move(decoded), encoded,
                                              &metadata->codec_specs));
    return absl::OkStatus();
  };
  TENSORSTORE_RETURN_IF_ERROR(set_up_codecs(
      codec_spec->codecs ? *codec_spec->codecs : ZarrCodecChainSpec{}));
  TENSORSTORE_RETURN_IF_ERROR(ValidateMetadata(*metadata));
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateMetadataSchema(*metadata, field_index, schema));
  return metadata;
}

ZarrMetadataConstraints::ZarrMetadataConstraints(const ZarrMetadata& metadata)
    : rank(metadata.rank),
      zarr_format(metadata.zarr_format),
      shape(metadata.shape),
      data_type(metadata.data_type),
      user_attributes(metadata.user_attributes),
      dimension_units(metadata.dimension_units),
      dimension_names(metadata.dimension_names),
      chunk_key_encoding(metadata.chunk_key_encoding),
      chunk_shape(metadata.chunk_shape),
      codec_specs(metadata.codec_specs),
      fill_value(metadata.fill_value),
      unknown_extension_attributes(metadata.unknown_extension_attributes) {}

}  // namespace internal_zarr3
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_zarr3::ZarrMetadataConstraints,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_zarr3::ZarrMetadataConstraints>())
