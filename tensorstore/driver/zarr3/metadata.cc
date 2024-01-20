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
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/optimization.h"
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
#include "tensorstore/driver/zarr3/name_configuration_json_binder.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/dimension_labels.h"
#include "tensorstore/internal/integer_types.h"
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
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/float8.h"
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
#undef TENSORSTORE_INTERNAL_DO_DEF
      return functions;
    }();

}  // namespace

absl::Status FillValueJsonBinder::operator()(std::true_type is_loading,
                                             internal_json_binding::NoOptions,
                                             SharedArray<const void>* obj,
                                             ::nlohmann::json* j) const {
  auto arr =
      AllocateArray(span<const Index, 0>{}, c_order, default_init, data_type);
  void* data = arr.data();
  *obj = std::move(arr);
  return kFillValueDataTypeFunctions[static_cast<size_t>(data_type.id())]
      .decode(data, *j);
}

absl::Status FillValueJsonBinder::operator()(std::false_type is_loading,
                                             internal_json_binding::NoOptions,
                                             const SharedArray<const void>* obj,
                                             ::nlohmann::json* j) const {
  return kFillValueDataTypeFunctions[static_cast<size_t>(data_type.id())]
      .encode(obj->data(), *j);
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
    using Self = internal::remove_cvref_t<decltype(*obj)>;
    DimensionIndex* rank = nullptr;
    if constexpr (is_loading) {
      rank = &obj->rank;
    }

    auto ensure_data_type = [&]() -> Result<DataType> {
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
                   jb::Projection<&Self::data_type>(maybe_optional(jb::Validate(
                       [](const auto& options, auto* obj) {
                         return ValidateDataType(*obj);
                       },
                       jb::DataTypeJsonBinder)))),
        jb::Member(
            "fill_value",
            jb::Projection<&Self::fill_value>(maybe_optional(
                [&](auto is_loading, const auto& options, auto* obj, auto* j) {
                  TENSORSTORE_ASSIGN_OR_RETURN(auto data_type,
                                               ensure_data_type());
                  return FillValueJsonBinder{data_type}(is_loading, options,
                                                        obj, j);
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
  if (!metadata.codecs) {
    ArrayCodecResolveParameters decoded;
    decoded.dtype = metadata.data_type;
    decoded.rank = metadata.rank;
    decoded.fill_value = metadata.fill_value;
    BytesCodecResolveParameters encoded;
    TENSORSTORE_ASSIGN_OR_RETURN(
        metadata.codecs,
        metadata.codec_specs.Resolve(std::move(decoded), encoded));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(metadata.codec_state,
                               metadata.codecs->Prepare(metadata.chunk_shape));
  return absl::OkStatus();
}

absl::Status ValidateMetadata(const ZarrMetadata& metadata,
                              const ZarrMetadataConstraints& constraints) {
  using internal::MetadataMismatchError;
  if (constraints.data_type && *constraints.data_type != metadata.data_type) {
    return MetadataMismatchError("data_type", constraints.data_type->name(),
                                 metadata.data_type.name());
  }
  if (constraints.fill_value &&
      !AreArraysIdenticallyEqual(*constraints.fill_value,
                                 metadata.fill_value)) {
    auto binder = FillValueJsonBinder{metadata.data_type};
    auto constraint_json = jb::ToJson(*constraints.fill_value, binder).value();
    auto metadata_json = jb::ToJson(metadata.fill_value, binder).value();
    return MetadataMismatchError("fill_value", constraint_json, metadata_json);
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

Result<IndexDomain<>> GetEffectiveDomain(
    DimensionIndex rank, std::optional<span<const Index>> shape,
    std::optional<span<const std::optional<std::string>>> dimension_names,
    const Schema& schema, bool* dimension_names_used = nullptr) {
  if (dimension_names_used) *dimension_names_used = false;
  auto domain = schema.domain();
  if (!shape && !dimension_names && !domain.valid()) {
    if (schema.rank() == 0) return {std::in_place, 0};
    // No information about the domain available.
    return {std::in_place};
  }

  // Rank is already validated by caller.
  assert(RankConstraint::EqualOrUnspecified(schema.rank(), rank));
  IndexDomainBuilder builder(std::max(schema.rank().rank, rank));
  if (shape) {
    builder.shape(*shape);
    builder.implicit_upper_bounds(true);
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
    // Use dimension_names as labels if they are valid.
    if (internal::ValidateDimensionLabelsAreUnique(normalized_dimension_names)
            .ok()) {
      if (dimension_names_used) *dimension_names_used = true;
      builder.labels(
          span<const std::string_view>(&normalized_dimension_names[0], rank));
    }
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto domain_from_metadata, builder.Finalize());
  TENSORSTORE_ASSIGN_OR_RETURN(
      domain, MergeIndexDomains(domain, domain_from_metadata),
      internal::ConvertInvalidArgumentToFailedPrecondition(
          tensorstore::MaybeAnnotateStatus(
              _, "Mismatch between metadata and schema")));
  return WithImplicitDimensions(domain, false, true);
  return domain;
}

Result<IndexDomain<>> GetEffectiveDomain(
    const ZarrMetadataConstraints& metadata_constraints, const Schema& schema,
    bool* dimension_names_used) {
  return GetEffectiveDomain(
      metadata_constraints.rank, metadata_constraints.shape,
      metadata_constraints.dimension_names, schema, dimension_names_used);
}

absl::Status SetChunkLayoutFromMetadata(
    DataType dtype, DimensionIndex rank,
    std::optional<span<const Index>> chunk_shape,
    const ZarrCodecChainSpec* codecs, ChunkLayout& chunk_layout) {
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(RankConstraint{rank}));
  rank = chunk_layout.rank();
  if (rank == dynamic_rank) return absl::OkStatus();

  if (chunk_shape) {
    assert(chunk_shape->size() == rank);
    TENSORSTORE_RETURN_IF_ERROR(
        chunk_layout.Set(ChunkLayout::WriteChunkShape(*chunk_shape)));
  }
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(
      ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(rank))));

  if (codecs) {
    ArrayDataTypeAndShapeInfo array_info;
    array_info.dtype = dtype;
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

Result<ChunkLayout> GetEffectiveChunkLayout(
    DataType dtype, DimensionIndex rank,
    std::optional<span<const Index>> chunk_shape,
    const ZarrCodecChainSpec* codecs, const Schema& schema) {
  auto chunk_layout = schema.chunk_layout();
  TENSORSTORE_RETURN_IF_ERROR(SetChunkLayoutFromMetadata(
      dtype, rank, chunk_shape, codecs, chunk_layout));
  return chunk_layout;
}

Result<ChunkLayout> GetEffectiveChunkLayout(
    const ZarrMetadataConstraints& metadata_constraints, const Schema& schema) {
  assert(RankConstraint::EqualOrUnspecified(metadata_constraints.rank,
                                            schema.rank()));
  return GetEffectiveChunkLayout(
      metadata_constraints.data_type.value_or(DataType{}),
      std::max(metadata_constraints.rank, schema.rank().rank),
      metadata_constraints.chunk_shape,
      metadata_constraints.codec_specs ? &*metadata_constraints.codec_specs
                                       : nullptr,
      schema);
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
                                    const Schema& schema) {
  if (!RankConstraint::EqualOrUnspecified(metadata.rank, schema.rank())) {
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Rank specified by schema (", schema.rank(),
        ") does not match rank specified by metadata (", metadata.rank, ")"));
  }

  if (schema.domain().valid()) {
    TENSORSTORE_RETURN_IF_ERROR(GetEffectiveDomain(
        metadata.rank, metadata.shape, metadata.dimension_names, schema));
  }

  if (auto dtype = schema.dtype();
      !IsPossiblySameDataType(metadata.data_type, dtype)) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("data_type from metadata (", metadata.data_type,
                            ") does not match dtype in schema (", dtype, ")"));
  }

  if (schema.chunk_layout().rank() != dynamic_rank) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto chunk_layout,
        GetEffectiveChunkLayout(metadata.data_type, metadata.rank,
                                metadata.chunk_shape, &metadata.codec_specs,
                                schema));
    if (chunk_layout.codec_chunk_shape().hard_constraint) {
      return absl::InvalidArgumentError("codec_chunk_shape not supported");
    }
  }

  if (auto schema_fill_value = schema.fill_value(); schema_fill_value.valid()) {
    const auto& fill_value = metadata.fill_value;
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto broadcast_fill_value,
        tensorstore::BroadcastArray(schema_fill_value, span<const Index>{}));
    TENSORSTORE_ASSIGN_OR_RETURN(
        SharedArray<const void> converted_fill_value,
        tensorstore::MakeCopy(std::move(broadcast_fill_value),
                              skip_repeated_elements, metadata.data_type));
    if (!AreArraysIdenticallyEqual(converted_fill_value, fill_value)) {
      auto binder = FillValueJsonBinder{metadata.data_type};
      auto schema_json = jb::ToJson(converted_fill_value, binder).value();
      auto metadata_json = jb::ToJson(metadata.fill_value, binder).value();
      return absl::FailedPreconditionError(tensorstore::StrCat(
          "Invalid fill_value: schema requires fill value of ",
          schema_json.dump(), ", but metadata specifies fill value of ",
          metadata_json.dump()));
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

Result<std::shared_ptr<const ZarrMetadata>> GetNewMetadata(
    const ZarrMetadataConstraints& metadata_constraints, const Schema& schema) {
  auto metadata = std::make_shared<ZarrMetadata>();

  metadata->zarr_format = metadata_constraints.zarr_format.value_or(3);
  metadata->chunk_key_encoding =
      metadata_constraints.chunk_key_encoding.value_or(ChunkKeyEncoding{
          /*.kind=*/ChunkKeyEncoding::kDefault, /*.separator=*/'/'});

  // Set domain
  bool dimension_names_used;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto domain,
      GetEffectiveDomain(metadata_constraints, schema, &dimension_names_used));
  if (!domain.valid() || !IsFinite(domain.box())) {
    return absl::InvalidArgumentError("domain must be specified");
  }
  const DimensionIndex rank = metadata->rank = domain.rank();
  metadata->shape.assign(domain.shape().begin(), domain.shape().end());
  metadata->dimension_names.assign(domain.labels().begin(),
                                   domain.labels().end());
  // Normalize empty string dimension names to `std::nullopt`.  This is more
  // consistent with the zarr v3 dimension name semantics, and ensures that the
  // `dimension_names` metadata field will be excluded entirely if all dimension
  // names are the empty string.
  //
  // However, if empty string dimension names were specified explicitly in
  // `metadata_constraints`, leave them exactly as specified.
  for (DimensionIndex i = 0; i < rank; ++i) {
    auto& name = metadata->dimension_names[i];
    if (!name || !name->empty()) continue;
    // Dimension name equals the empty string.
    if (dimension_names_used && (*metadata_constraints.dimension_names)[i]) {
      // Empty dimension name was explicitly specified in
      // `metadata_constraints`, leave it as is.
      assert((*metadata_constraints.dimension_names)[i]->empty());
      continue;
    }
    // Name was not explicitly specified in `metadata_constraints` as an empty
    // string.  Normalize it to `std::nullopt`.
    name = std::nullopt;
  }

  // Set dtype
  auto dtype = schema.dtype();
  if (!dtype.valid()) {
    return absl::InvalidArgumentError("dtype must be specified");
  }
  TENSORSTORE_RETURN_IF_ERROR(ValidateDataType(dtype));
  metadata->data_type = dtype;

  if (metadata_constraints.fill_value) {
    metadata->fill_value = *metadata_constraints.fill_value;
  } else if (auto fill_value = schema.fill_value(); fill_value.valid()) {
    const auto status = [&] {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto broadcast_fill_value,
          tensorstore::BroadcastArray(fill_value, span<const Index>{}));
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto converted_fill_value,
          tensorstore::MakeCopy(std::move(broadcast_fill_value),
                                skip_repeated_elements, metadata->data_type));
      metadata->fill_value = std::move(converted_fill_value);
      return absl::OkStatus();
    }();
    TENSORSTORE_RETURN_IF_ERROR(
        status, tensorstore::MaybeAnnotateStatus(_, "Invalid fill_value"));
  } else {
    metadata->fill_value = tensorstore::AllocateArray(
        /*shape=*/span<const Index>(), c_order, value_init,
        metadata->data_type);
  }

  metadata->user_attributes = metadata_constraints.user_attributes;
  metadata->unknown_extension_attributes =
      metadata_constraints.unknown_extension_attributes;

  // Set dimension units
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

  // Set chunk shape

  ArrayCodecResolveParameters decoded;
  decoded.dtype = metadata->data_type;
  decoded.rank = metadata->rank;
  decoded.fill_value = metadata->fill_value;

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
    // Read chunk and write chunk shapes differ.  Insert sharding codec if there
    // is not already one.
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
  TENSORSTORE_RETURN_IF_ERROR(ValidateMetadataSchema(*metadata, schema));
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
