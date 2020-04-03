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

#include "tensorstore/driver/zarr/dtype.h"

#include "tensorstore/internal/json.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace internal_zarr {

Result<ZarrDType::BaseDType> ParseBaseDType(absl::string_view dtype) {
  using D = ZarrDType::BaseDType;
  if (dtype.size() < 3) goto error;
  {
    const char endian_indicator = dtype[0];
    const char type_indicator = dtype[1];
    const absl::string_view suffix = dtype.substr(2);
    endian endian_value;
    switch (type_indicator) {
      case 'b':
        if (suffix != "1") goto error;
        ABSL_FALLTHROUGH_INTENDED;
      case 'S':
      case 'V':
        // Single byte types must have endian indicator of '|'.
        if (endian_indicator != '|') goto error;
        endian_value = endian::native;
        break;
      case 'i':
      case 'u':
        if (endian_indicator == '|') {
          // Endian indicator of '|' is only valid if size is 1 byte.
          if (suffix != "1") goto error;
          endian_value = endian::native;
          break;
        } else if (suffix == "1") {
          goto error;
        }
        ABSL_FALLTHROUGH_INTENDED;
      case 'f':
      case 'c':
      case 'm':
      case 'M':
        // Endian indicator must be '<' or '>'.
        if (endian_indicator == '<') {
          endian_value = endian::little;
        } else if (endian_indicator == '>') {
          endian_value = endian::big;
        } else {
          goto error;
        }
        break;
    }
    switch (type_indicator) {
      case 'b':
        return D{std::string(dtype), DataTypeOf<bool>(), endian::native};
      case 'i':
        if (suffix == "1") {
          return D{std::string(dtype), DataTypeOf<int8_t>(), endian_value};
        }
        if (suffix == "2") {
          return D{std::string(dtype), DataTypeOf<int16_t>(), endian_value};
        }
        if (suffix == "4") {
          return D{std::string(dtype), DataTypeOf<int32_t>(), endian_value};
        }
        if (suffix == "8") {
          return D{std::string(dtype), DataTypeOf<int64_t>(), endian_value};
        }
        goto error;
      case 'u':
        if (suffix == "1") {
          return D{std::string(dtype), DataTypeOf<uint8_t>(), endian_value};
        }
        if (suffix == "2") {
          return D{std::string(dtype), DataTypeOf<uint16_t>(), endian_value};
        }
        if (suffix == "4") {
          return D{std::string(dtype), DataTypeOf<uint32_t>(), endian_value};
        }
        if (suffix == "8") {
          return D{std::string(dtype), DataTypeOf<uint64_t>(), endian_value};
        }
        goto error;
      case 'f':
        if (suffix == "2") {
          return D{std::string(dtype), DataTypeOf<float16_t>(), endian_value};
        }
        if (suffix == "4") {
          return D{std::string(dtype), DataTypeOf<float32_t>(), endian_value};
        }
        if (suffix == "8") {
          return D{std::string(dtype), DataTypeOf<float64_t>(), endian_value};
        }
        goto error;
      case 'c':
        if (suffix == "8") {
          return D{std::string(dtype), DataTypeOf<complex64_t>(), endian_value};
        }
        if (suffix == "16") {
          return D{std::string(dtype), DataTypeOf<complex128_t>(),
                   endian_value};
        }
        goto error;
      case 'S':
      case 'V': {
        // TODO(jbms): Support 'U' ("unicode")
        // Parse suffix as number.
        Index num_elements = 0;
        for (char c : suffix) {
          if (internal::MulOverflow(num_elements, Index(10), &num_elements))
            goto error;
          if (c < '0' || c > '9') goto error;
          if (internal::AddOverflow(num_elements, Index(c - '0'),
                                    &num_elements))
            goto error;
        }
        return D{std::string(dtype),
                 (type_indicator == 'S') ? DataType(DataTypeOf<char_t>())
                                         : DataType(DataTypeOf<byte_t>()),
                 endian::native,
                 {num_elements}};
      }
    }
  }
error:
  return absl::InvalidArgumentError(
      StrCat("Unsupported zarr dtype: ", QuoteString(dtype)));
}

namespace {

/// Parses a zarr metadata "dtype" JSON specification, but does not compute any
/// derived values, and does not check for duplicate field names.
///
/// This is called by `ParseDType`.
///
/// \param value The zarr metadata "dtype" JSON specification.
/// \param out[out] Must be non-null.  Filled with the parsed dtype on success.
/// \error `absl::StatusCode::kInvalidArgument' if `value` is invalid.
Result<ZarrDType> ParseDTypeNoDerived(const nlohmann::json& value) {
  ZarrDType out;
  if (value.is_string()) {
    // Single field.
    out.has_fields = false;
    out.fields.resize(1);
    TENSORSTORE_ASSIGN_OR_RETURN(
        static_cast<ZarrDType::BaseDType&>(out.fields[0]),
        ParseBaseDType(value.get<std::string>()));
    return out;
  }
  out.has_fields = true;
  auto parse_result = internal::JsonParseArray(
      value,
      [&](std::ptrdiff_t size) {
        out.fields.resize(size);
        return absl::OkStatus();
      },
      [&](const ::nlohmann::json& x, std::ptrdiff_t field_i) {
        auto& field = out.fields[field_i];
        return internal::JsonParseArray(
            x,
            [&](std::ptrdiff_t size) {
              if (size < 2 || size > 3) {
                return absl::InvalidArgumentError(StrCat(
                    "Expected array of size 2 or 3, but received: ", x.dump()));
              }
              return absl::OkStatus();
            },
            [&](const ::nlohmann::json& v, std::ptrdiff_t i) {
              switch (i) {
                case 0:
                  if (internal::JsonRequireValueAs(v, &field.name).ok()) {
                    if (!field.name.empty()) return absl::OkStatus();
                  }
                  return absl::InvalidArgumentError(StrCat(
                      "Expected non-empty string, but received: ", v.dump()));
                case 1: {
                  std::string dtype_string;
                  TENSORSTORE_RETURN_IF_ERROR(
                      internal::JsonRequireValueAs(v, &dtype_string));
                  TENSORSTORE_ASSIGN_OR_RETURN(
                      static_cast<ZarrDType::BaseDType&>(field),
                      ParseBaseDType(dtype_string));
                  return absl::OkStatus();
                }
                case 2: {
                  return internal::JsonParseArray(
                      v,
                      [&](std::ptrdiff_t size) {
                        field.outer_shape.resize(size);
                        return absl::OkStatus();
                      },
                      [&](const ::nlohmann::json& x, std::ptrdiff_t j) {
                        return internal::JsonRequireInteger(
                            x, &field.outer_shape[j], /*strict=*/true, 1,
                            kInfIndex);
                      });
                }
                default:
                  TENSORSTORE_UNREACHABLE;
              }
            });
      });
  if (!parse_result.ok()) return parse_result;
  return out;
}

/// Validates a parsed `ZarrDType` and computes the derived values.
///
/// This is called by `ParseDType`.
///
/// \error `absl::StatusCode::kInvalidArgument` if two fields have the same
///     name.
/// \error `absl::StatusCode::kInvalidArgument` if the field size is too
///     large.
Status ComputeDerivedDTypeValues(ZarrDType* dtype) {
  // Compute field byte offsets and other derived data.
  dtype->bytes_per_outer_element = 0;
  for (std::size_t field_i = 0; field_i < dtype->fields.size(); ++field_i) {
    auto& field = dtype->fields[field_i];
    if (std::any_of(
            dtype->fields.begin(), dtype->fields.begin() + field_i,
            [&](const ZarrDType::Field& f) { return f.name == field.name; })) {
      return absl::InvalidArgumentError(StrCat(
          "Field name ", QuoteString(field.name), " occurs more than once"));
    }
    field.field_shape.resize(field.flexible_shape.size() +
                             field.outer_shape.size());
    std::copy(field.flexible_shape.begin(), field.flexible_shape.end(),
              std::copy(field.outer_shape.begin(), field.outer_shape.end(),
                        field.field_shape.begin()));

    field.num_inner_elements = ProductOfExtents(span(field.field_shape));
    if (field.num_inner_elements == std::numeric_limits<Index>::max()) {
      return absl::InvalidArgumentError(StrCat(
          "Product of dimensions ", span(field.field_shape), " is too large"));
    }
    if (internal::MulOverflow(field.num_inner_elements,
                              static_cast<Index>(field.data_type->size),
                              &field.num_bytes)) {
      return absl::InvalidArgumentError("Field size in bytes is too large");
    }
    field.byte_offset = dtype->bytes_per_outer_element;
    if (internal::AddOverflow(dtype->bytes_per_outer_element, field.num_bytes,
                              &dtype->bytes_per_outer_element)) {
      return absl::InvalidArgumentError(
          "Total number of bytes per outer array element is too large");
    }
  }
  return absl::OkStatus();
}

}  // namespace

Result<ZarrDType> ParseDType(const nlohmann::json& value) {
  TENSORSTORE_ASSIGN_OR_RETURN(ZarrDType dtype, ParseDTypeNoDerived(value));
  TENSORSTORE_RETURN_IF_ERROR(ComputeDerivedDTypeValues(&dtype));
  return dtype;
}

void to_json(::nlohmann::json& out, const ZarrDType::Field& field) {
  using array_t = ::nlohmann::json::array_t;
  if (field.outer_shape.empty()) {
    out = array_t{field.name, field.encoded_dtype};
  } else {
    out = array_t{field.name, field.encoded_dtype, field.outer_shape};
  }
}

void to_json(::nlohmann::json& out,  // NOLINT
             const ZarrDType& dtype) {
  if (!dtype.has_fields) {
    out = dtype.fields[0].encoded_dtype;
  } else {
    out = dtype.fields;
  }
}

}  // namespace internal_zarr
}  // namespace tensorstore
