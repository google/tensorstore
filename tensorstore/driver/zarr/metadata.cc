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

#include "tensorstore/driver/zarr/metadata.h"

#include <cassert>
#include <memory>
#include <string>

#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/escaping.h"
#include "tensorstore/driver/zarr/compressor.h"
#include "tensorstore/internal/data_type_endian_conversion.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_zarr {

namespace jb = internal_json_binding;
TENSORSTORE_DEFINE_JSON_BINDER(
    OrderJsonBinder,
    jb::Enum<ContiguousLayoutOrder, std::string_view>({{c_order, "C"},
                                                       {fortran_order, "F"}}))

TENSORSTORE_DEFINE_JSON_BINDER(DimensionSeparatorJsonBinder,
                               jb::Enum<DimensionSeparator, std::string_view>(
                                   {{DimensionSeparator::kDotSeparated, "."},
                                    {DimensionSeparator::kSlashSeparated,
                                     "/"}}))

void to_json(::nlohmann::json& out, DimensionSeparator value) {
  DimensionSeparatorJsonBinder(/*is_loading=*/std::false_type{},
                               /*options=*/internal_json_binding::NoOptions{},
                               &value, &out)
      .IgnoreError();
}

namespace {

Result<double> DecodeFloat(const nlohmann::json& j) {
  double value;
  if (j == "NaN") {
    value = std::numeric_limits<double>::quiet_NaN();
  } else if (j == "Infinity") {
    value = std::numeric_limits<double>::infinity();
  } else if (j == "-Infinity") {
    value = -std::numeric_limits<double>::infinity();
  } else if (j.is_number()) {
    value = j.get<double>();
  } else {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Invalid floating-point value: ", j.dump()));
  }
  return value;
}

::nlohmann::json EncodeFloat(double value) {
  if (std::isnan(value)) return "NaN";
  if (value == std::numeric_limits<double>::infinity()) return "Infinity";
  if (value == -std::numeric_limits<double>::infinity()) return "-Infinity";
  return value;
}

}  // namespace

Result<std::vector<SharedArray<const void>>> ParseFillValue(
    const nlohmann::json& j, const ZarrDType& dtype) {
  std::vector<SharedArray<const void>> fill_values;
  fill_values.resize(dtype.fields.size());
  if (j.is_null()) return fill_values;
  if (!dtype.has_fields) {
    assert(dtype.fields.size() == 1);
    auto& field = dtype.fields[0];
    const char type_indicator = field.encoded_dtype[1];
    switch (type_indicator) {
      case 'f': {
        TENSORSTORE_ASSIGN_OR_RETURN(double value, DecodeFloat(j));
        fill_values[0] =
            MakeCopy(MakeScalarArrayView(value), c_order, field.dtype).value();
        return fill_values;
      }
      case 'i': {
        std::int64_t value;
        const std::size_t num_bits = 8 * field.dtype->size - 1;
        const std::uint64_t max_value = static_cast<std::int64_t>(
            (static_cast<std::uint64_t>(1) << num_bits) - 1);
        const std::int64_t min_value = static_cast<std::int64_t>(-1)
                                       << num_bits;
        TENSORSTORE_RETURN_IF_ERROR(internal_json::JsonRequireInteger(
            j, &value, /*strict=*/true, min_value, max_value));
        fill_values[0] =
            MakeCopy(MakeScalarArrayView(value), c_order, field.dtype).value();
        return fill_values;
      }
      case 'u': {
        std::uint64_t value;
        const std::size_t num_bits = 8 * field.dtype->size;
        const std::uint64_t max_value =
            (static_cast<std::uint64_t>(2) << (num_bits - 1)) - 1;
        TENSORSTORE_RETURN_IF_ERROR(internal_json::JsonRequireInteger(
            j, &value, /*strict=*/true, 0, max_value));
        fill_values[0] =
            MakeCopy(MakeScalarArrayView(value), c_order, field.dtype).value();
        return fill_values;
      }
      case 'b': {
        bool value;
        TENSORSTORE_RETURN_IF_ERROR(
            internal_json::JsonRequireValueAs(j, &value, /*strict=*/true));
        fill_values[0] = MakeScalarArray<bool>(value);
        return fill_values;
      }
      case 'c': {
        double values[2];
        if (!j.is_array()) {
          // Fallthrough to allow base64 encoding.
          break;
        }
        TENSORSTORE_RETURN_IF_ERROR(internal_json::JsonParseArray(
            j,
            [](std::ptrdiff_t size) {
              return internal_json::JsonValidateArrayLength(size, 2);
            },
            [&](const ::nlohmann::json& v, std::ptrdiff_t i) {
              TENSORSTORE_ASSIGN_OR_RETURN(values[i], DecodeFloat(v));
              return absl::OkStatus();
            }));
        fill_values[0] =
            MakeCopy(MakeScalarArrayView(complex128_t(values[0], values[1])),
                     c_order, field.dtype)
                .value();
        return fill_values;
      }
    }
  }
  // Decode as Base64
  std::string b64_decoded;
  if (!j.is_string() ||
      !absl::Base64Unescape(j.get<std::string>(), &b64_decoded) ||
      static_cast<Index>(b64_decoded.size()) != dtype.bytes_per_outer_element) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Expected ", dtype.bytes_per_outer_element,
                            " base64-encoded bytes, but received: ", j.dump()));
  }
  for (size_t field_i = 0; field_i < dtype.fields.size(); ++field_i) {
    auto& field = dtype.fields[field_i];
    DataType r = field.dtype;
    auto fill_value = AllocateArray(field.field_shape, ContiguousLayoutOrder::c,
                                    default_init, r);
    internal::DecodeArray(
        ArrayView<const void>(
            {static_cast<const void*>(b64_decoded.data() + field.byte_offset),
             r},
            fill_value.layout()),
        field.endian, fill_value);
    fill_values[field_i] = std::move(fill_value);
  }
  return fill_values;
}

::nlohmann::json EncodeFillValue(
    const ZarrDType& dtype, span<const SharedArray<const void>> fill_values) {
  assert(dtype.fields.size() == static_cast<size_t>(fill_values.size()));
  if (!dtype.has_fields) {
    assert(dtype.fields.size() == 1);
    const auto& field = dtype.fields[0];
    const auto& fill_value = fill_values[0];
    if (!fill_value.valid()) return nullptr;
    const char type_indicator = field.encoded_dtype[1];
    switch (type_indicator) {
      case 'f': {
        double value;
        TENSORSTORE_CHECK_OK(
            CopyConvertedArray(fill_value, MakeScalarArrayView(value)));
        return EncodeFloat(value);
      }
      case 'c': {
        complex128_t value;
        TENSORSTORE_CHECK_OK(
            CopyConvertedArray(fill_value, MakeScalarArrayView(value)));
        return ::nlohmann::json::array_t{EncodeFloat(value.real()),
                                         EncodeFloat(value.imag())};
      }
      case 'i':
      case 'u':
      case 'b': {
        ::nlohmann::json value;
        TENSORSTORE_CHECK_OK(
            CopyConvertedArray(fill_value, MakeScalarArrayView(value)));
        return value;
      }
    }
  }
  // Compute base-64 encoding of fill values.
  std::vector<unsigned char> buffer(dtype.bytes_per_outer_element);
  for (size_t field_i = 0; field_i < dtype.fields.size(); ++field_i) {
    const auto& field = dtype.fields[field_i];
    const auto& fill_value = fill_values[field_i];
    if (!fill_value.valid()) return nullptr;
    DataType r = field.dtype;
    Array<void> encoded_fill_value(
        {static_cast<void*>(buffer.data() + field.byte_offset), r},
        field.field_shape);
    internal::EncodeArray(fill_value, encoded_fill_value, field.endian);
  }
  std::string b64_encoded;
  absl::Base64Escape(
      std::string_view(reinterpret_cast<const char*>(buffer.data()),
                       buffer.size()),
      &b64_encoded);
  return b64_encoded;
}

Result<ZarrChunkLayout> ComputeChunkLayout(const ZarrDType& dtype,
                                           ContiguousLayoutOrder order,
                                           span<const Index> chunk_shape) {
  ZarrChunkLayout layout;
  layout.fields.resize(dtype.fields.size());
  layout.num_outer_elements = ProductOfExtents(chunk_shape);
  if (layout.num_outer_elements == std::numeric_limits<Index>::max()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Product of chunk dimensions ", chunk_shape, " is too large"));
  }
  if (internal::MulOverflow(dtype.bytes_per_outer_element,
                            layout.num_outer_elements,
                            &layout.bytes_per_chunk)) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Total number of bytes per chunk is too large"));
  }

  for (std::size_t field_i = 0; field_i < dtype.fields.size(); ++field_i) {
    auto& field = dtype.fields[field_i];
    auto& field_layout = layout.fields[field_i];
    const DimensionIndex inner_rank = field.field_shape.size();
    const DimensionIndex total_rank = chunk_shape.size() + inner_rank;
    TENSORSTORE_RETURN_IF_ERROR(ValidateRank(total_rank));
    const auto initialize_layout = [&](StridedLayout<>* strided_layout,
                                       Index outer_element_stride) {
      strided_layout->set_rank(total_rank);
      std::copy(field.field_shape.begin(), field.field_shape.end(),
                std::copy(chunk_shape.begin(), chunk_shape.end(),
                          strided_layout->shape().begin()));
      // Compute strides for inner field dimensions.
      ComputeStrides(ContiguousLayoutOrder::c, field.dtype->size,
                     strided_layout->shape().last(inner_rank),
                     strided_layout->byte_strides().last(inner_rank));
      // Compute strides for outer dimensions.
      ComputeStrides(order, outer_element_stride, chunk_shape,
                     strided_layout->byte_strides().first(chunk_shape.size()));
    };
    initialize_layout(&field_layout.decoded_chunk_layout, field.num_bytes);
    initialize_layout(&field_layout.encoded_chunk_layout,
                      dtype.bytes_per_outer_element);
  }
  return layout;
}

constexpr auto MetadataJsonBinder = [](auto maybe_optional) {
  return [=](auto is_loading, const auto& options, auto* obj, auto* j) {
    using T = internal::remove_cvref_t<decltype(*obj)>;
    DimensionIndex* rank = nullptr;
    if constexpr (is_loading) {
      rank = &obj->rank;
    }
    auto ensure_dtype = [&]() -> Result<const ZarrDType*> {
      if constexpr (std::is_same_v<T, ZarrMetadata>) {
        return &obj->dtype;
      } else if constexpr (std::is_same_v<T, ZarrPartialMetadata>) {
        /// dtype is wrapped std::optional<>
        if (!obj->dtype) {
          return absl::InvalidArgumentError(
              "must be specified in conjunction with \"dtype\"");
        }
        return &*obj->dtype;
      }
      ABSL_UNREACHABLE();  // COV_NF_LINE
    };

    return jb::Object(
        jb::Member("zarr_format",
                   jb::Projection(&T::zarr_format,
                                  maybe_optional(jb::Integer<int>(2, 2)))),
        jb::Member(
            "shape",
            jb::Projection(&T::shape, maybe_optional(jb::ShapeVector(rank)))),
        jb::Member("chunks",
                   jb::Projection(&T::chunks,
                                  maybe_optional(jb::ChunkShapeVector(rank)))),
        jb::Member("dtype", jb::Projection(&T::dtype)),
        jb::Member("compressor", jb::Projection(&T::compressor)),
        jb::Member("fill_value",
                   jb::Projection(
                       &T::fill_value,
                       maybe_optional([&](auto is_loading, const auto& options,
                                          auto* obj, auto* j) {
                         TENSORSTORE_ASSIGN_OR_RETURN(auto* dtype,
                                                      ensure_dtype());
                         return FillValueJsonBinder(*dtype)(is_loading, options,
                                                            obj, j);
                       }))),
        jb::Member("order",
                   jb::Projection(&T::order, maybe_optional(OrderJsonBinder))),
        jb::Member("filters", jb::Projection(&T::filters)),
        jb::Member("dimension_separator",
                   jb::Projection(&T::dimension_separator,
                                  jb::Optional(DimensionSeparatorJsonBinder))),
        [](auto is_loading, const auto& options, auto* obj, auto* j) {
          if constexpr (std::is_same_v<T, ZarrMetadata>) {
            return jb::DefaultBinder<>(is_loading, options, &obj->extra_members,
                                       j);
          } else {
            return absl::OkStatus();
          }
        })(is_loading, options, obj, j);
  };
};

absl::Status ValidateMetadata(ZarrMetadata& metadata) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      metadata.chunk_layout,
      ComputeChunkLayout(metadata.dtype, metadata.order, metadata.chunks));
  return absl::OkStatus();
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    ZarrMetadata, jb::Validate([](const auto& options,
                                  auto* obj) { return ValidateMetadata(*obj); },
                               MetadataJsonBinder(internal::identity{})))

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(ZarrPartialMetadata,
                                       MetadataJsonBinder([](auto binder) {
                                         return jb::Optional(binder);
                                       }))

// Two decoding strategies:  raw decoder and custom decoder.  Initially we will
// only support raw decoder.

Result<absl::InlinedVector<SharedArrayView<const void>, 1>> DecodeChunk(
    const ZarrMetadata& metadata, absl::Cord buffer) {
  const size_t num_fields = metadata.dtype.fields.size();
  if (metadata.compressor) {
    absl::Cord decoded;
    TENSORSTORE_RETURN_IF_ERROR(metadata.compressor->Decode(
        buffer, &decoded, metadata.dtype.bytes_per_outer_element));
    buffer = std::move(decoded);
  }
  if (static_cast<Index>(buffer.size()) !=
      metadata.chunk_layout.bytes_per_chunk) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Uncompressed chunk is ", buffer.size(), " bytes, but should be ",
        metadata.chunk_layout.bytes_per_chunk, " bytes"));
  }
  absl::InlinedVector<SharedArrayView<const void>, 1> field_arrays(num_fields);

  bool must_copy = false;
  // First, attempt to create arrays that reference the cord without copying.
  for (size_t field_i = 0; field_i < num_fields; ++field_i) {
    const auto& field = metadata.dtype.fields[field_i];
    const auto& field_layout = metadata.chunk_layout.fields[field_i];
    field_arrays[field_i] = internal::TryViewCordAsArray(
        buffer, field.byte_offset, field.dtype, field.endian,
        field_layout.encoded_chunk_layout);
    if (!field_arrays[field_i].valid()) {
      must_copy = true;
      break;
    }
  }
  if (must_copy) {
    auto flat_buffer = buffer.Flatten();
    for (size_t field_i = 0; field_i < num_fields; ++field_i) {
      const auto& field = metadata.dtype.fields[field_i];
      const auto& field_layout = metadata.chunk_layout.fields[field_i];
      ArrayView<const void> source_array{
          ElementPointer<const void>(
              static_cast<const void*>(flat_buffer.data() + field.byte_offset),
              field.dtype),
          field_layout.encoded_chunk_layout};
      field_arrays[field_i] = internal::CopyAndDecodeArray(
          source_array, field.endian, field_layout.decoded_chunk_layout);
    }
  }
  return field_arrays;
}

namespace {
bool SingleArrayMatchesEncodedRepresentation(
    const ZarrMetadata& metadata,
    const SharedArrayView<const void>& component) {
  auto& field = metadata.dtype.fields[0];
  if (field.endian != endian::native) return false;
  return internal::RangesEqual(
      component.byte_strides(),
      metadata.chunk_layout.fields[0].encoded_chunk_layout.byte_strides());
}

absl::Cord MakeCordFromSharedPtr(std::shared_ptr<const void> ptr, size_t size) {
  std::string_view s(static_cast<const char*>(ptr.get()), size);
  return absl::MakeCordFromExternal(
      s, [ptr = std::move(ptr)](std::string_view s) mutable { ptr.reset(); });
}

absl::Cord MakeCordFromContiguousArray(
    const SharedArrayView<const void>& array) {
  return MakeCordFromSharedPtr(array.pointer(),
                               array.num_elements() * array.dtype().size());
}

absl::Cord CopyComponentsToEncodedLayout(
    const ZarrMetadata& metadata,
    span<const SharedArrayView<const void>> components) {
  internal::FlatCordBuilder output_builder(
      metadata.chunk_layout.bytes_per_chunk);
  ByteStridedPointer<void> data_ptr = output_builder.data();
  for (size_t field_i = 0; field_i < components.size(); ++field_i) {
    const auto& field = metadata.dtype.fields[field_i];
    const auto& field_layout = metadata.chunk_layout.fields[field_i];
    ArrayView<void> encoded_array{{data_ptr + field.byte_offset, field.dtype},
                                  field_layout.encoded_chunk_layout};
    internal::EncodeArray(components[field_i], encoded_array, field.endian);
  }
  return std::move(output_builder).Build();
}
}  // namespace

Result<absl::Cord> EncodeChunk(
    const ZarrMetadata& metadata,
    span<const SharedArrayView<const void>> components) {
  absl::Cord output;
  if (components.size() == 1 &&
      SingleArrayMatchesEncodedRepresentation(metadata, components[0])) {
    output = MakeCordFromContiguousArray(components[0]);
  } else {
    output = CopyComponentsToEncodedLayout(metadata, components);
  }
  if (metadata.compressor) {
    absl::Cord encoded;
    TENSORSTORE_RETURN_IF_ERROR(metadata.compressor->Encode(
        output, &encoded, metadata.dtype.bytes_per_outer_element));
    return encoded;
  }
  return output;
}

bool IsMetadataCompatible(const ZarrMetadata& a, const ZarrMetadata& b) {
  // Rank must be the same.
  if (a.shape.size() != b.shape.size()) return false;

  auto a_json = ::nlohmann::json(a);
  auto b_json = ::nlohmann::json(b);

  // Shape is allowed to differ.
  a_json.erase("shape");
  b_json.erase("shape");

  // Extra members are allowed to differ.
  for (const auto& [key, value] : a.extra_members) {
    a_json.erase(key);
  }
  for (const auto& [key, value] : b.extra_members) {
    b_json.erase(key);
  }

  return a_json == b_json;
}

void EncodeCacheKeyAdl(std::string* out, const ZarrMetadata& metadata) {
  auto json = ::nlohmann::json(metadata);
  json["shape"] = metadata.shape.size();
  out->append(json.dump());
}

}  // namespace internal_zarr
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_zarr::ZarrPartialMetadata,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_zarr::ZarrPartialMetadata>())
