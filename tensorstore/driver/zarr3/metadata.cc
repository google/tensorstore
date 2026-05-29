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
#include <cstring>
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

#include <nlohmann/json.hpp>
#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/bytes.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/sharding_indexed.h"
#include "tensorstore/internal/data_type_endian_conversion.h"
#include "tensorstore/util/endian.h"
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
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

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
    return absl::InvalidArgumentError(absl::StrFormat(
        "%v data type is not one of the supported data types: %s", dtype,
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
#undef TENSORSTORE_INTERNAL_DO_DEF
      // Add char_t support for string data types
      functions[static_cast<size_t>(DataTypeId::char_t)] =
          FillValueDataTypeFunctions::Make<::tensorstore::dtypes::char_t>();
      // byte_t is handled specially to use uint8_t functions
      return functions;
    }();

}  // namespace

FillValueJsonBinder::FillValueJsonBinder(ZarrDType zarr_dtype,
                                         bool allow_missing_dtype)
    : zarr_dtype(std::move(zarr_dtype)),
      allow_missing_dtype(allow_missing_dtype) {}

FillValueJsonBinder::FillValueJsonBinder(DataType data_type,
                                         bool allow_missing_dtype)
    : allow_missing_dtype(allow_missing_dtype) {
  zarr_dtype.has_fields = false;
  zarr_dtype.fields.resize(1);
  auto& field = zarr_dtype.fields[0];
  field.name.clear();
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
  obj->resize(zarr_dtype.fields.size());
  if (zarr_dtype.fields.size() == 1) {
    if (zarr_dtype.fields[0].dtype.id() == DataTypeId::byte_t &&
        !zarr_dtype.fields[0].flexible_shape.empty()) {
      if (!j->is_string()) {
        return absl::InvalidArgumentError(
            "Expected base64-encoded string for raw_bytes fill_value");
      }
      std::string b64_decoded;
      if (!absl::Base64Unescape(j->get<std::string>(), &b64_decoded)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Expected valid base64-encoded fill value, but received: %s",
            j->dump()));
      }
      Index expected_size = zarr_dtype.fields[0].num_inner_elements;
      if (static_cast<Index>(b64_decoded.size()) != expected_size) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Expected %d base64-encoded bytes for fill_value, but received "
            "%d bytes",
            expected_size, b64_decoded.size()));
      }
      auto fill_arr =
          AllocateArray(zarr_dtype.fields[0].field_shape, c_order, default_init,
                        zarr_dtype.fields[0].dtype);
      std::memcpy(fill_arr.data(), b64_decoded.data(), b64_decoded.size());
      (*obj)[0] = std::move(fill_arr);
    } else {
      TENSORSTORE_RETURN_IF_ERROR(
          DecodeSingle(*j, zarr_dtype.fields[0].dtype, (*obj)[0]));
    }
  } else {
    if (j->is_object()) {
      for (size_t i = 0; i < zarr_dtype.fields.size(); ++i) {
        const auto& field_name = zarr_dtype.fields[i].name;
        if (j->contains(field_name)) {
          TENSORSTORE_RETURN_IF_ERROR(DecodeSingle(
              (*j)[field_name], zarr_dtype.fields[i].dtype, (*obj)[i]));
        } else {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Missing required field \"%s\" in fill_value object", field_name));
        }
      }
    } else if (zarr_dtype.is_legacy_structured && j->is_string()) {
      // Base64 / array fills are legacy "structured" affordances; modern
      // "struct" falls through to the trailing `ExpectedError("object")`.
      std::string b64_decoded;
      if (!absl::Base64Unescape(j->get<std::string>(), &b64_decoded)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Expected valid base64-encoded fill value, but received: %s",
            j->dump()));
      }
      if (static_cast<Index>(b64_decoded.size()) !=
          zarr_dtype.bytes_per_outer_element) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Expected %d base64-encoded bytes for fill_value, but received "
            "%d bytes",
            zarr_dtype.bytes_per_outer_element, b64_decoded.size()));
      }
      for (size_t i = 0; i < zarr_dtype.fields.size(); ++i) {
        const auto& field = zarr_dtype.fields[i];
        auto arr = AllocateArray(span<const Index, 0>{}, c_order, default_init,
                                 field.dtype);
        ArrayView<const void> src_view(
            {static_cast<const void*>(b64_decoded.data() + field.byte_offset),
             field.dtype},
            arr.layout());
        internal::DecodeArray(src_view, fill_endian, arr);
        (*obj)[i] = std::move(arr);
      }
    } else if (zarr_dtype.is_legacy_structured && j->is_array()) {
      if (j->size() != zarr_dtype.fields.size()) {
        return internal_json::ExpectedError(
            *j, absl::StrFormat("array of size %d", zarr_dtype.fields.size()));
      }
      for (size_t i = 0; i < zarr_dtype.fields.size(); ++i) {
        TENSORSTORE_RETURN_IF_ERROR(
            DecodeSingle((*j)[i], zarr_dtype.fields[i].dtype, (*obj)[i]));
      }
    } else {
      return internal_json::ExpectedError(
          *j, zarr_dtype.is_legacy_structured
                  ? "object, array, or base64-encoded string"
                  : "object");
    }
  }
  return absl::OkStatus();
}

absl::Status FillValueJsonBinder::operator()(
    std::false_type is_loading, internal_json_binding::NoOptions,
    const std::vector<SharedArray<const void>>* obj,
    ::nlohmann::json* j) const {
  if (zarr_dtype.fields.size() == 1) {
    return EncodeSingle((*obj)[0], zarr_dtype.fields[0].dtype, *j);
  }
  *j = ::nlohmann::json::object();
  for (size_t i = 0; i < zarr_dtype.fields.size(); ++i) {
    ::nlohmann::json item;
    TENSORSTORE_RETURN_IF_ERROR(
        EncodeSingle((*obj)[i], zarr_dtype.fields[i].dtype, item));
    (*j)[zarr_dtype.fields[i].name] = std::move(item);
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
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported metadata field %v is not marked "
                            "{\"must_understand\": false}",
                            QuoteString(key)));
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

    auto ensure_zarr_dtype = [&]() -> Result<ZarrDType> {
      if constexpr (std::is_same_v<Self, ZarrMetadata>) {
        return obj->zarr_dtype;
      }
      if constexpr (std::is_same_v<Self, ZarrMetadataConstraints>) {
        // zarr_dtype is wrapped in std::optional<>
        if (obj->zarr_dtype) {
          return *obj->zarr_dtype;
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
                   jb::Projection<&Self::zarr_dtype>(maybe_optional(
                       jb::DefaultBinder<>))),
        jb::Member(
            "fill_value",
            [&](auto is_loading, const auto& options, auto* obj,
                auto* j) -> absl::Status {
              if constexpr (std::is_same_v<Self, ZarrMetadata>) {
                TENSORSTORE_ASSIGN_OR_RETURN(auto zarr_dtype,
                                             ensure_zarr_dtype());
                // Defer base64 decoding to `ValidateMetadata`: the bytes
                // codec's `endian` (per
                // https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/structured)
                // isn't resolved yet here.
                if constexpr (is_loading) {
                  if (zarr_dtype.is_legacy_structured &&
                      zarr_dtype.fields.size() > 1 && j->is_string()) {
                    obj->deferred_base64_fill_value = *j;
                    obj->fill_value.assign(zarr_dtype.fields.size(),
                                           SharedArray<const void>());
                    return absl::OkStatus();
                  }
                } else {
                  ABSL_DCHECK(!obj->deferred_base64_fill_value.has_value())
                      << "Call ValidateMetadata before serializing";
                }
                return FillValueJsonBinder{std::move(zarr_dtype),
                                           /*allow_missing_dtype=*/true}(
                    is_loading, options, &obj->fill_value, j);
              } else {
                return jb::Optional([&](auto is_loading, const auto& options,
                                        auto* obj, auto* j) {
                  TENSORSTORE_ASSIGN_OR_RETURN(auto zarr_dtype,
                                               ensure_zarr_dtype());
                  return FillValueJsonBinder{std::move(zarr_dtype)}(
                      is_loading, options, obj, j);
                })(is_loading, options, &obj->fill_value, j);
              }
            }),
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

endian GetBytesCodecEndian(const ZarrCodecChainSpec& codec_specs) {
  const BytesCodecSpec* leaf = GetLeafBytesCodec(codec_specs);
  if (leaf == nullptr) return endian::native;
  return leaf->options.endianness.value_or(endian::native);
}

SharedArray<const void> MakeVoidFillValue(
    const ZarrDType& zarr_dtype, const ZarrCodecChainSpec& codec_specs,
    span<const SharedArray<const void>> per_field_fill) {
  const Index nbytes = zarr_dtype.bytes_per_outer_element;
  auto byte_fill = AllocateArray(span<const Index, 1>({nbytes}), c_order,
                                 value_init,
                                 dtype_v<tensorstore::dtypes::byte_t>);
  auto* dst = static_cast<std::byte*>(byte_fill.data());
  // Match `ZarrLeafChunkCache::EncodeChunk`: every multi-byte field on
  // disk is in the bytes codec's `endian`.
  const endian target_endian = GetBytesCodecEndian(codec_specs);
  const size_t num_fields = std::min<size_t>(
      zarr_dtype.fields.size(), static_cast<size_t>(per_field_fill.size()));
  for (size_t i = 0; i < num_fields; ++i) {
    const auto& field = zarr_dtype.fields[i];
    const auto& fill = per_field_fill[i];
    if (!fill.valid()) continue;
    ArrayView<void> dst_view(
        ElementPointer<void>{static_cast<void*>(dst + field.byte_offset),
                             field.dtype},
        fill.layout());
    internal::EncodeArray(fill, dst_view, target_endian);
  }
  return byte_fill;
}

ZarrDType MakeVoidDType(Index bytes_per_outer_element) {
  return ZarrDType{
      /*.has_fields=*/false,
      /*.fields=*/{ZarrDType::Field{
          ZarrDType::BaseDType{"", dtype_v<tensorstore::dtypes::byte_t>,
                               {bytes_per_outer_element}},
          /*.name=*/"",
          /*.field_shape=*/{bytes_per_outer_element},
          /*.num_inner_elements=*/bytes_per_outer_element,
          /*.byte_offset=*/0,
          /*.num_bytes=*/bytes_per_outer_element}},
      /*.bytes_per_outer_element=*/bytes_per_outer_element};
}

absl::Status ValidateVoidCodecChain(const ZarrCodecChainSpec& codec_specs) {
  // open_as_void requires the leaf array-to-bytes codec to be `bytes`;
  // other codecs would alter the on-disk byte layout.
  const ZarrCodecChainSpec* leaf = GetLeafChainSpec(codec_specs);
  if (leaf == nullptr) {
    return absl::InvalidArgumentError(
        "open_as_void: nested sharding_indexed codec is missing its "
        "sub-chunk codec specification");
  }
  if (dynamic_cast<const BytesCodecSpec*>(leaf->array_to_bytes.get()) ==
      nullptr) {
    return absl::InvalidArgumentError(
        "open_as_void requires the innermost array-to-bytes codec to be the "
        "`bytes` codec (after unwrapping any sharding_indexed layers).  "
        "Codecs that alter the byte representation of the chunk (e.g. "
        "blosc-direct, future bitround) are not supported under "
        "open_as_void.");
  }
  return absl::OkStatus();
}

Result<std::shared_ptr<const ZarrMetadata>> GetVoidMetadata(
    const ZarrMetadata& metadata) {
  // Per the zarr v3 open_as_void spec, raw byte access is supported for any
  // data type; the only structural precondition is that the codec chain is
  // re-resolvable under the substituted `byte` data type.
  TENSORSTORE_RETURN_IF_ERROR(ValidateVoidCodecChain(metadata.codec_specs));

  auto void_metadata = std::make_shared<ZarrMetadata>(metadata);

  // Replace the data type with the synthetic single-field byte view.  This is
  // what every downstream consumer (chunk cache, codec resolution, schema
  // validation) sees.  The persisted on-disk metadata is untouched: this view
  // exists only in memory.
  void_metadata->zarr_dtype =
      MakeVoidDType(metadata.zarr_dtype.bytes_per_outer_element);

  // Pack the per-field fill values into a single byte array following the
  // struct's byte_offset layout, mirroring how a chunk is laid out on disk.
  // Endian conversion (when applicable) is handled by `MakeVoidFillValue`.
  void_metadata->fill_value = {
      MakeVoidFillValue(metadata.zarr_dtype, metadata.codec_specs,
                        metadata.fill_value)};

  // Cleared so that `ValidateMetadata` rederives `field_shape` from the
  // newly-substituted dtype rather than carrying over the natural metadata's
  // value.  In every supported case this rederives to the same numeric value
  // (the per-field field_shape on the synthetic byte field), but staging it
  // through the standard derivation keeps the void path uniform with parse-
  // time metadata.
  void_metadata->field_shape.clear();

  // `MakeVoidFillValue` already produced the typed fill above; drop any
  // deferred base64 JSON so `ValidateMetadata` doesn't re-decode it
  // against the synthetic byte dtype.
  void_metadata->deferred_base64_fill_value.reset();

  // Force re-resolution of the codec chain against the byte data type.  The
  // codec chain spec itself (endianness selections, sharding parameters, etc.)
  // is unchanged; only the resolved codec instances and prepared state are
  // recomputed.
  void_metadata->codecs.reset();
  void_metadata->codec_state.reset();
  TENSORSTORE_RETURN_IF_ERROR(ValidateMetadata(*void_metadata));
  return void_metadata;
}

namespace {
// Populate `metadata.field_shape` (the codec-view inner trailing dimensions)
// from the data type, unless the caller pre-populated it (e.g.
// `GetVoidMetadata` pins it to `{bytes_per_outer_element}` after substituting
// the dtype).  For single-field rN-style dtypes we hoist the per-field
// `field_shape`; for multi-field structs we use the single flat byte trailing
// dim.  Single-field scalar dtypes contribute no inner codec dimensions.
//
// This is the single source of truth for `field_shape`.  Call sites that need
// it (both at parse time via `ValidateMetadata` and at create time via
// `GetNewMetadata`) defer to this helper rather than recomputing locally.
void DeriveFieldShape(ZarrMetadata& metadata) {
  if (!metadata.field_shape.empty()) return;
  const auto& zarr_dtype = metadata.zarr_dtype;
  if (zarr_dtype.fields.size() == 1 &&
      !zarr_dtype.fields[0].field_shape.empty()) {
    metadata.field_shape.assign(zarr_dtype.fields[0].field_shape.begin(),
                                zarr_dtype.fields[0].field_shape.end());
  } else if (zarr_dtype.fields.size() > 1) {
    metadata.field_shape.push_back(zarr_dtype.bytes_per_outer_element);
  }
}

}  // namespace

absl::Status ValidateStructEndianness(const ZarrDType& dtype,
                                      ZarrCodecChainSpec& codec_specs) {
  if (!dtype.has_fields) return absl::OkStatus();

  const bool has_multi_byte_field = std::any_of(
      dtype.fields.begin(), dtype.fields.end(), [](const auto& field) {
        return field.dtype.valid() && field.dtype->size > 1;
      });
  if (!has_multi_byte_field) return absl::OkStatus();

  const BytesCodecSpec* leaf = GetLeafBytesCodec(codec_specs);
  if (leaf == nullptr) return absl::OkStatus();
  if (leaf->options.endianness.has_value()) return absl::OkStatus();

  if (dtype.is_legacy_structured) {
    return SetLeafBytesCodecEndian(codec_specs, endian::little);
  }
  return absl::InvalidArgumentError(
      "Zarr v3 \"struct\" data types with multi-byte fields require the "
      "innermost `bytes` codec to specify an explicit `endian` (\"little\" "
      "or \"big\").");
}

absl::Status ValidateMetadata(ZarrMetadata& metadata) {
  DeriveFieldShape(metadata);

  // Skip when codecs are already resolved: the create path runs the
  // pre-Resolve check itself in `GetNewMetadata`, and post-Resolve the
  // leaf bytes spec has its `endian` stripped.
  if (!metadata.codecs) {
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateStructEndianness(metadata.zarr_dtype, metadata.codec_specs));
  }

  // Resolve any base64 fill deferred from JSON parse (see
  // `MetadataJsonBinder`); the bytes codec's `endian` is now known.
  if (metadata.deferred_base64_fill_value.has_value()) {
    FillValueJsonBinder fill_binder{metadata.zarr_dtype};
    fill_binder.fill_endian = GetBytesCodecEndian(metadata.codec_specs);
    TENSORSTORE_RETURN_IF_ERROR(fill_binder(
        std::true_type{}, jb::NoOptions{}, &metadata.fill_value,
        &*metadata.deferred_base64_fill_value));
    metadata.deferred_base64_fill_value.reset();
  }

  // The codec chain is resolved at the *chunked* rank only.  Inner trailing
  // dimensions contributed by the dtype's field_shape (multi-field structs,
  // `rN` raw byte fields, `open_as_void`'s byte substitution) travel via
  // `decoded.inner_shape`: array-to-array codecs propagate this field
  // unchanged and cannot operate on it; the leaf array-to-bytes codec
  // consumes it for byte-stream sizing.
  //
  // At runtime the chunk cache still hands extended-rank arrays to the
  // codec chain, so each codec's runtime state is built at runtime rank
  // (chunked + inner) inside its `Resolve` step.
  const bool has_field_shape = !metadata.field_shape.empty();

  std::vector<Index> runtime_shape(metadata.chunk_shape.begin(),
                                   metadata.chunk_shape.end());
  runtime_shape.insert(runtime_shape.end(), metadata.field_shape.begin(),
                       metadata.field_shape.end());

  if (!metadata.codecs) {
    ArrayCodecResolveParameters decoded;
    decoded.dtype = has_field_shape
                        ? dtype_v<std::byte>
                        : metadata.zarr_dtype.fields[0].dtype;
    decoded.rank = metadata.rank;
    decoded.inner_shape = metadata.field_shape;
    // `read_chunk_shape` is at the chunked rank only; sharding_indexed's
    // internal `sub_chunk_shape` is at this same rank now too.
    {
      auto& read_chunk_shape = decoded.read_chunk_shape.emplace();
      std::copy_n(metadata.chunk_shape.begin(), metadata.rank,
                  read_chunk_shape.begin());
    }
    if (metadata.fill_value.size() == 1 && !has_field_shape) {
      decoded.fill_value = metadata.fill_value[0];
    }

    BytesCodecResolveParameters encoded;
    TENSORSTORE_ASSIGN_OR_RETURN(
        metadata.codecs,
        metadata.codec_specs.Resolve(std::move(decoded), encoded));
  }

  // Get codec chunk layout info at the chunked rank.
  ArrayDataTypeAndShapeInfo array_info;
  array_info.dtype = has_field_shape
                         ? dtype_v<std::byte>
                         : metadata.zarr_dtype.fields[0].dtype;
  array_info.rank = metadata.rank;
  array_info.inner_shape = metadata.field_shape;
  std::copy_n(metadata.chunk_shape.begin(), metadata.rank,
              array_info.shape.emplace().begin());

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
                               metadata.codecs->Prepare(runtime_shape));
  return absl::OkStatus();
}

absl::Status ValidateMetadata(const ZarrMetadata& metadata,
                              const ZarrMetadataConstraints& constraints) {
  using internal::MetadataMismatchError;
  if (constraints.zarr_dtype) {
    if (::nlohmann::json(*constraints.zarr_dtype) !=
        ::nlohmann::json(metadata.zarr_dtype)) {
      return MetadataMismatchError(
          "data_type", ::nlohmann::json(*constraints.zarr_dtype).dump(),
          ::nlohmann::json(metadata.zarr_dtype).dump());
    }
  }
  if (constraints.fill_value) {
    if (constraints.fill_value->size() != metadata.fill_value.size()) {
      return MetadataMismatchError("fill_value size",
                                   constraints.fill_value->size(),
                                   metadata.fill_value.size());
    }
    for (size_t i = 0; i < metadata.fill_value.size(); ++i) {
      if (!AreArraysIdenticallyEqual((*constraints.fill_value)[i],
                                     metadata.fill_value[i])) {
        auto binder = FillValueJsonBinder{metadata.zarr_dtype};
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
        codecs_copy.MergeFrom(*constraints.codec_specs, /*strict=*/true))
        .Format("Mismatch in \"codecs\"");
  }
  if (constraints.dimension_names &&
      *constraints.dimension_names != metadata.dimension_names) {
    return MetadataMismatchError(
        "dimension_names", jb::ToJson(*constraints.dimension_names).value(),
        jb::ToJson(metadata.dimension_names).value());
  }
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateMetadataSubset(constraints.user_attributes,
                                       metadata.user_attributes))
      .Format("Mismatch in \"attributes\"");
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
std::string GetFieldNames(const ZarrDType& zarr_dtype) {
  std::vector<std::string> field_names;
  for (const auto& field : zarr_dtype.fields) {
    field_names.push_back(field.name);
  }
  return ::nlohmann::json(field_names).dump();
}
}  // namespace

Result<size_t> GetFieldIndex(const ZarrDType& zarr_dtype,
                             std::string_view selected_field) {
  if (selected_field.empty()) {
    if (zarr_dtype.fields.size() != 1) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Must specify a \"field\" that is one of: %s", GetFieldNames(zarr_dtype)));
    }
    return 0;
  }
  if (!zarr_dtype.has_fields) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Requested field %v but dtype does not have named fields",
        QuoteString(selected_field)));
  }
  for (size_t field_index = 0; field_index < zarr_dtype.fields.size();
       ++field_index) {
    if (zarr_dtype.fields[field_index].name == selected_field) return field_index;
  }
  return absl::FailedPreconditionError(absl::StrFormat(
      "Requested field %v is not one of: %s", QuoteString(selected_field),
      GetFieldNames(zarr_dtype)));
}

SpecRankAndFieldInfo GetSpecRankAndFieldInfo(const ZarrMetadata& metadata,
                                             size_t field_index) {
  SpecRankAndFieldInfo info;
  info.chunked_rank = metadata.rank;
  info.field = &metadata.zarr_dtype.fields[field_index];
  info.field_rank = info.field->field_shape.size();
  info.full_rank = info.chunked_rank + info.field_rank;
  return info;
}

absl::Status ValidateSpecRankAndFieldInfo(SpecRankAndFieldInfo& info) {
  if (info.field) {
    info.field_rank = info.field->field_shape.size();
  }

  if (info.full_rank == dynamic_rank) {
    info.full_rank = RankConstraint::Add(info.chunked_rank, info.field_rank);
    if (info.full_rank != dynamic_rank) {
      TENSORSTORE_RETURN_IF_ERROR(ValidateRank(info.full_rank));
    }
  }

  if (!RankConstraint::LessEqualOrUnspecified(info.chunked_rank,
                                              info.full_rank) ||
      !RankConstraint::LessEqualOrUnspecified(info.field_rank,
                                              info.full_rank) ||
      !RankConstraint::EqualOrUnspecified(
          info.full_rank,
          RankConstraint::Add(info.chunked_rank, info.field_rank))) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Rank specified by schema (%d) is not compatible with metadata",
        info.full_rank));
  }

  if (info.chunked_rank == dynamic_rank) {
    info.chunked_rank =
        RankConstraint::Subtract(info.full_rank, info.field_rank);
  }
  if (info.field_rank == dynamic_rank) {
    info.field_rank =
        RankConstraint::Subtract(info.full_rank, info.chunked_rank);
  }

  return absl::OkStatus();
}

Result<SpecRankAndFieldInfo> GetSpecRankAndFieldInfo(
    const ZarrMetadataConstraints& metadata, std::string_view selected_field,
    const Schema& schema, bool open_as_void) {
  if (open_as_void && !selected_field.empty()) {
    return absl::InvalidArgumentError(
        "\"field\" and \"open_as_void\" are mutually exclusive");
  }

  SpecRankAndFieldInfo info;
  info.full_rank = schema.rank();
  info.chunked_rank = metadata.rank;

  if (open_as_void) {
    info.field_rank = 1;  // bytes dimension
  } else if (metadata.zarr_dtype) {
    const ZarrDType& zarr_dtype = *metadata.zarr_dtype;
    if (!selected_field.empty()) {
      for (const auto& field : zarr_dtype.fields) {
        if (field.name == selected_field) {
          info.field = &field;
          break;
        }
      }
    } else if (zarr_dtype.fields.size() == 1) {
      info.field = &zarr_dtype.fields[0];
    }
  }

  TENSORSTORE_RETURN_IF_ERROR(ValidateSpecRankAndFieldInfo(info));
  return info;
}

absl::Status TrySetMetadataConstraintsOnSchema(
    const ZarrMetadataConstraints& metadata_constraints,
    std::string_view selected_field, bool open_as_void, Schema& schema) {
  // Get rank/field info to determine schema rank and the selected field.
  // This also validates that selected_field and open_as_void aren't both set.
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto info,
      GetSpecRankAndFieldInfo(metadata_constraints, selected_field, schema,
                              open_as_void));

  // Set schema dtype from metadata constraints.
  if (metadata_constraints.zarr_dtype) {
    const auto& zarr_dtype = *metadata_constraints.zarr_dtype;
    if (info.field) {
      // A single field is resolved (either the sole field of an unstructured
      // dtype or a selected field of a structured dtype).  The schema dtype
      // need not be unspecified; it just needs to match the field's dtype,
      // which `Schema::Set` validates.
      TENSORSTORE_RETURN_IF_ERROR(schema.Set(info.field->dtype));
    } else if (!zarr_dtype.has_fields && !zarr_dtype.fields.empty()) {
      TENSORSTORE_RETURN_IF_ERROR(schema.Set(zarr_dtype.fields[0].dtype));
    } else if (schema.dtype().valid()) {
      return absl::InvalidArgumentError(
          "schema dtype must be unspecified for structured zarr3 data types "
          "unless a field is selected");
    }
  }

  if (info.full_rank != dynamic_rank) {
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(RankConstraint{info.full_rank}));
  }

  return absl::OkStatus();
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
      _.Format("Mismatch between metadata and schema")
          .With(internal::ConvertInvalidArgumentToFailedPrecondition));
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
  SpecRankAndFieldInfo info;
  info.chunked_rank = metadata_constraints.rank;
  if (info.chunked_rank == dynamic_rank) {
    info.chunked_rank = schema.rank().rank;
  }
  if (info.chunked_rank == dynamic_rank && metadata_constraints.shape) {
    info.chunked_rank = metadata_constraints.shape->size();
  }
  if (info.chunked_rank == dynamic_rank && metadata_constraints.chunk_shape) {
    info.chunked_rank = metadata_constraints.chunk_shape->size();
  }

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

  if (!RankConstraint::EqualOrUnspecified(schema.rank(), info.full_rank)) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Rank specified by schema (%v) does not match rank specified by "
        "metadata (%v)",
        schema.rank(), info.full_rank));
  }

  if (auto dtype = schema.dtype();
      !IsPossiblySameDataType(info.field->dtype, dtype)) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "data_type from metadata (%v) does not match dtype in schema (%v)",
        info.field->dtype, dtype));
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
                              skip_repeated_elements, info.field->dtype));
    if (!AreArraysIdenticallyEqual(converted_fill_value, fill_value)) {
      auto binder = FillValueJsonBinder{metadata.zarr_dtype};
      std::vector<SharedArray<const void>> schema_fill_vec{converted_fill_value};
      std::vector<SharedArray<const void>> metadata_fill_vec{fill_value};
      auto schema_json = jb::ToJson(schema_fill_vec, binder).value();
      auto metadata_json = jb::ToJson(metadata_fill_vec, binder).value();
      return absl::FailedPreconditionError(absl::StrFormat(
          "Invalid fill_value: schema requires fill value of %s, but metadata "
          "specifies fill value of %s",
          schema_json.dump(), metadata_json.dump()));
    }
  }

  if (auto schema_codec = schema.codec(); schema_codec.valid()) {
    auto codec = GetCodecFromMetadata(metadata);
    TENSORSTORE_RETURN_IF_ERROR(codec.MergeFrom(schema_codec))
        .Format("codec from metadata does not match codec in schema");
  }

  if (auto schema_dimension_units = schema.dimension_units();
      schema_dimension_units.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(
        GetEffectiveDimensionUnits(metadata.rank, metadata.dimension_units,
                                   schema_dimension_units))
        .Format(
            "dimension_units from metadata does not match dimension_units in "
            "schema")
        .With(internal::ConvertInvalidArgumentToFailedPrecondition);
  }

  return absl::OkStatus();
}

absl::Status ValidateMetadataSchema(const ZarrMetadata& metadata,
                                    const Schema& schema) {
  return ValidateMetadataSchema(metadata, /*field_index=*/0, schema);
}

Result<std::shared_ptr<const ZarrMetadata>> GetNewMetadata(
    const ZarrMetadataConstraints& metadata_constraints, const Schema& schema,
    std::string_view selected_field, bool open_as_void) {
  auto metadata = std::make_shared<ZarrMetadata>();

  metadata->zarr_format = metadata_constraints.zarr_format.value_or(3);
  metadata->chunk_key_encoding =
      metadata_constraints.chunk_key_encoding.value_or(ChunkKeyEncoding{
          /*.kind=*/ChunkKeyEncoding::kDefault, /*.separator=*/'/'});

  // Determine data type first
  if (metadata_constraints.zarr_dtype) {
    metadata->zarr_dtype = *metadata_constraints.zarr_dtype;
  } else if (!selected_field.empty()) {
    return absl::InvalidArgumentError(
        "\"dtype\" must be specified in \"metadata\" if \"field\" is "
        "specified");
  } else if (auto dtype = schema.dtype(); dtype.valid()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        static_cast<ZarrDType::BaseDType&>(
            metadata->zarr_dtype.fields.emplace_back()),
        ChooseBaseDType(dtype));
    metadata->zarr_dtype.has_fields = false;
    TENSORSTORE_RETURN_IF_ERROR(ValidateDType(metadata->zarr_dtype));
  } else {
    return absl::InvalidArgumentError("dtype must be specified");
  }

  // Resolve the selected field: void mode does not pick a "real" field (the
  // void substitution happens later, via `GetVoidMetadata`), but we still need
  // a placeholder field index/info to drive shape/rank computation here.
  size_t field_index;
  if (open_as_void) {
    if (!selected_field.empty()) {
      return absl::InvalidArgumentError(
          "\"field\" and \"open_as_void\" are mutually exclusive");
    }
    field_index = 0;
  } else {
    TENSORSTORE_ASSIGN_OR_RETURN(
        field_index, GetFieldIndex(metadata->zarr_dtype, selected_field));
  }
  SpecRankAndFieldInfo info;
  info.field = &metadata->zarr_dtype.fields[field_index];
  info.chunked_rank = metadata_constraints.rank;
  if (info.chunked_rank == dynamic_rank && metadata_constraints.shape) {
    info.chunked_rank = metadata_constraints.shape->size();
  }
  if (info.chunked_rank == dynamic_rank &&
      schema.rank().rank != dynamic_rank) {
    info.chunked_rank = schema.rank().rank;
  }
  // Number of synthetic trailing dimensions contributed by either an explicit
  // field_shape (rN / struct field) or by `open_as_void` (the bytes
  // dimension).  Both are field_shape in the laramiel sense.
  const DimensionIndex extra_field_dims =
      open_as_void ? 1
                   : (info.field ? info.field->field_shape.size() : 0);
  if (extra_field_dims != 0 && info.chunked_rank != dynamic_rank) {
    info.chunked_rank += extra_field_dims;
  }

  // Set domain.  When the field contributes trailing field_shape dimensions
  // (rN, struct field, or the synthetic void byte dimension), extend the
  // user-provided chunked shape with those dimensions before merging with the
  // schema, so that the schema and metadata describe the same full rank.
  bool dimension_names_used = false;
  std::vector<Index> extended_shape;
  std::optional<span<const Index>> constraint_shape_span;
  if (metadata_constraints.shape) {
    if (open_as_void) {
      // For void access, extend the shape to include the bytes dimension
      extended_shape.assign(metadata_constraints.shape->begin(),
                           metadata_constraints.shape->end());
      extended_shape.push_back(metadata->zarr_dtype.bytes_per_outer_element);
      constraint_shape_span.emplace(extended_shape.data(), extended_shape.size());
    } else if (info.field && !info.field->field_shape.empty()) {
      // For fields with field_shape, extend the shape to include field dimensions
      extended_shape.assign(metadata_constraints.shape->begin(),
                           metadata_constraints.shape->end());
      extended_shape.insert(extended_shape.end(),
                           info.field->field_shape.begin(),
                           info.field->field_shape.end());
      constraint_shape_span.emplace(extended_shape.data(), extended_shape.size());
    } else {
      constraint_shape_span.emplace(metadata_constraints.shape->data(),
                                    metadata_constraints.shape->size());
    }
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
  // The user-visible domain may include trailing dimensions contributed by
  // either an explicit field_shape (rN / struct field) or by `open_as_void`
  // (the bytes dimension), but the persisted metadata stores only the
  // logical chunked dimensions.
  const DimensionIndex logical_rank = domain.rank() - extra_field_dims;
  metadata->rank = logical_rank;
  info.chunked_rank = domain.rank();  // Keep extended rank for codec processing
  metadata->shape.assign(domain.shape().begin(),
                         domain.shape().begin() + logical_rank);
  metadata->dimension_names.assign(domain.labels().begin(),
                                   domain.labels().begin() + logical_rank);

  for (DimensionIndex i = 0; i < logical_rank; ++i) {
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
    if (metadata->zarr_dtype.fields.size() != 1) {
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
                                metadata->zarr_dtype.fields[0].dtype));
      metadata->fill_value.push_back(std::move(converted_fill_value));
      return absl::OkStatus();
    }();
    TENSORSTORE_RETURN_IF_ERROR(status).Format("Invalid fill_value");
  } else {
    metadata->fill_value.resize(metadata->zarr_dtype.fields.size());
    for (size_t i = 0; i < metadata->fill_value.size(); ++i) {
      metadata->fill_value[i] = tensorstore::AllocateArray(
          /*shape=*/span<const Index>(), c_order, value_init,
          metadata->zarr_dtype.fields[i].dtype);
    }
  }

  metadata->user_attributes = metadata_constraints.user_attributes;
  metadata->unknown_extension_attributes =
      metadata_constraints.unknown_extension_attributes;

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto dimension_units,
      GetEffectiveDimensionUnits(metadata->rank, metadata_constraints.dimension_units,
                                 schema.dimension_units()),
      _.Format("Invalid dimension_units"));
  if (std::any_of(dimension_units.begin(), dimension_units.end(),
                  [](const auto& unit) { return unit.has_value(); })) {
    metadata->dimension_units = std::move(dimension_units);
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto codec_spec,
                               GetEffectiveCodec(metadata_constraints, schema));

  // Derive `metadata->field_shape` up front so the rest of this function (and
  // the eventual `ValidateMetadata` re-check) drives off a single source of
  // truth, identical to the parse-time code path.
  DeriveFieldShape(*metadata);
  const bool has_field_shape = !metadata->field_shape.empty();

  // Codec resolution is at the chunked rank; inner (`field_shape`) dims are
  // carried via `decoded.inner_shape` and never appear in any rank/shape
  // field that array-to-array codecs see.
  ArrayCodecResolveParameters decoded;
  decoded.dtype = has_field_shape
                      ? dtype_v<std::byte>
                      : metadata->zarr_dtype.fields[0].dtype;
  decoded.rank = metadata->rank;
  decoded.inner_shape = metadata->field_shape;
  if (!has_field_shape) {
    if (metadata->fill_value.size() == 1) {
      decoded.fill_value = metadata->fill_value[0];
    }
  } else {
    // Zero-filled scalar byte fill_value (broadcast to chunk shape at use).
    decoded.fill_value = AllocateArray(
        span<const Index, 0>{}, c_order, value_init, dtype_v<std::byte>);
  }

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto chunk_layout, GetEffectiveChunkLayout(metadata_constraints, schema));
  metadata->chunk_shape.resize(metadata->rank);

  if (auto inner_order = chunk_layout.inner_order(); inner_order.valid()) {
    auto& dest = decoded.inner_order.emplace();
    std::copy(inner_order.begin(), inner_order.end(), dest.begin());
  }

  // `read_chunk_shape` is at the chunked rank only.
  span<Index> read_chunk_shape(decoded.read_chunk_shape.emplace().data(),
                               metadata->rank);

  TENSORSTORE_RETURN_IF_ERROR(internal::ChooseReadWriteChunkShapes(
      chunk_layout.read_chunk(), chunk_layout.write_chunk(),
      SubBoxView(domain.box(), 0, metadata->rank), read_chunk_shape,
      metadata->chunk_shape));

  if (!internal::RangesEqual(span<const Index>(metadata->chunk_shape),
                             read_chunk_shape)) {
    if (!codec_spec->codecs || codec_spec->codecs->sharding_height() == 0) {
      // sub_chunk_shape is at the chunked rank now -- same as the
      // user-facing form and the on-disk zarr.json representation.
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

  // Must run before `Resolve` strips `endian` under the byte-substituted
  // inner dtype.
  if (codec_spec->codecs) {
    TENSORSTORE_RETURN_IF_ERROR(ValidateStructEndianness(metadata->zarr_dtype,
                                                          *codec_spec->codecs));
  }

  const auto set_up_codecs =
      [&](const ZarrCodecChainSpec& codec_specs) -> absl::Status {
    BytesCodecResolveParameters encoded;
    TENSORSTORE_ASSIGN_OR_RETURN(
        metadata->codecs, codec_specs.Resolve(std::move(decoded), encoded,
                                              &metadata->codec_specs));
    // `Resolve` strips `endian` from the leaf bytes spec under the
    // byte-substituted inner dtype; re-inject the validated user value.
    if (metadata->zarr_dtype.has_fields) {
      const BytesCodecSpec* resolved_leaf =
          GetLeafBytesCodec(metadata->codec_specs);
      if (resolved_leaf != nullptr &&
          !resolved_leaf->options.endianness.has_value()) {
        TENSORSTORE_RETURN_IF_ERROR(SetLeafBytesCodecEndian(
            metadata->codec_specs, GetBytesCodecEndian(codec_specs)));
      }
    }
    return absl::OkStatus();
  };
  TENSORSTORE_RETURN_IF_ERROR(set_up_codecs(
      codec_spec->codecs ? *codec_spec->codecs : ZarrCodecChainSpec{}));
  TENSORSTORE_RETURN_IF_ERROR(ValidateMetadata(*metadata));
  if (open_as_void) {
    // The user-supplied schema dtype is `byte` and rank is chunked + 1, neither
    // of which match the (natural) `*metadata`.  The driver re-validates against
    // the void-substituted view via `GetVoidMetadata` after this returns, so we
    // skip the non-applicable per-field schema check here.
    TENSORSTORE_RETURN_IF_ERROR(ValidateVoidCodecChain(metadata->codec_specs));
  } else {
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateMetadataSchema(*metadata, field_index, schema));
  }
  return metadata;
}

ZarrMetadataConstraints::ZarrMetadataConstraints(const ZarrMetadata& metadata)
    : rank(metadata.rank),
      zarr_format(metadata.zarr_format),
      shape(metadata.shape),
      zarr_dtype(metadata.zarr_dtype),
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
