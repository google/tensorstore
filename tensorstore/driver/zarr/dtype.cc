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

#include "absl/base/optimization.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_zarr {

Result<ZarrDType::BaseDType> ParseBaseDType(std::string_view dtype) {
  using D = ZarrDType::BaseDType;
  if (dtype == "bfloat16") {
    // Support `bfloat16` as an extension.  This is inconsistent with the normal
    // NumPy typestr syntax and does not provide a way to indicate the byte
    // order, but has the advantage of working with the official Zarr Python
    // library provided that a `"bfloat16"` data type is registered in
    // `numpy.typeDict`, since zarr invokes `numpy.dtype` to parse data types.
    return D{std::string(dtype), dtype_v<bfloat16_t>, endian::little};
  }
  if (dtype.size() < 3) goto error;
  {
    const char endian_indicator = dtype[0];
    const char type_indicator = dtype[1];
    const std::string_view suffix = dtype.substr(2);
    endian endian_value;
    switch (endian_indicator) {
      case '<':
        endian_value = endian::little;
        break;
      case '>':
        endian_value = endian::big;
        break;
      case '|':
        endian_value = endian::native;
        break;
      default:
        goto error;
    }
    switch (type_indicator) {
      case 'b':
        if (suffix != "1") goto error;
        ABSL_FALLTHROUGH_INTENDED;
      case 'S':
      case 'V':
        // Single byte types ignore the endian indicator.
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
          endian_value = endian::native;
          break;
        }
        // Fallthrough if size is greater than 1 byte.
        [[fallthrough]];
      case 'f':
      case 'c':
      case 'm':
      case 'M':
        // Endian indicator must be '<' or '>'.
        if (endian_indicator == '|') {
          goto error;
        }
        break;
    }
    switch (type_indicator) {
      case 'b':
        return D{std::string(dtype), dtype_v<bool>, endian::native};
      case 'i':
        if (suffix == "1") {
          return D{std::string(dtype), dtype_v<int8_t>, endian_value};
        }
        if (suffix == "2") {
          return D{std::string(dtype), dtype_v<int16_t>, endian_value};
        }
        if (suffix == "4") {
          return D{std::string(dtype), dtype_v<int32_t>, endian_value};
        }
        if (suffix == "8") {
          return D{std::string(dtype), dtype_v<int64_t>, endian_value};
        }
        goto error;
      case 'u':
        if (suffix == "1") {
          return D{std::string(dtype), dtype_v<uint8_t>, endian_value};
        }
        if (suffix == "2") {
          return D{std::string(dtype), dtype_v<uint16_t>, endian_value};
        }
        if (suffix == "4") {
          return D{std::string(dtype), dtype_v<uint32_t>, endian_value};
        }
        if (suffix == "8") {
          return D{std::string(dtype), dtype_v<uint64_t>, endian_value};
        }
        goto error;
      case 'f':
        if (suffix == "2") {
          return D{std::string(dtype), dtype_v<float16_t>, endian_value};
        }
        if (suffix == "4") {
          return D{std::string(dtype), dtype_v<float32_t>, endian_value};
        }
        if (suffix == "8") {
          return D{std::string(dtype), dtype_v<float64_t>, endian_value};
        }
        goto error;
      case 'c':
        if (suffix == "8") {
          return D{std::string(dtype), dtype_v<complex64_t>, endian_value};
        }
        if (suffix == "16") {
          return D{std::string(dtype), dtype_v<complex128_t>, endian_value};
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
                 (type_indicator == 'S') ? DataType(dtype_v<char_t>)
                                         : DataType(dtype_v<byte_t>),
                 endian::native,
                 {num_elements}};
      }
    }
  }
error:
  return absl::InvalidArgumentError(
      tensorstore::StrCat("Unsupported zarr dtype: ", QuoteString(dtype)));
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
  auto parse_result = internal_json::JsonParseArray(
      value,
      [&](std::ptrdiff_t size) {
        out.fields.resize(size);
        return absl::OkStatus();
      },
      [&](const ::nlohmann::json& x, std::ptrdiff_t field_i) {
        auto& field = out.fields[field_i];
        return internal_json::JsonParseArray(
            x,
            [&](std::ptrdiff_t size) {
              if (size < 2 || size > 3) {
                return absl::InvalidArgumentError(tensorstore::StrCat(
                    "Expected array of size 2 or 3, but received: ", x.dump()));
              }
              return absl::OkStatus();
            },
            [&](const ::nlohmann::json& v, std::ptrdiff_t i) {
              switch (i) {
                case 0:
                  if (internal_json::JsonRequireValueAs(v, &field.name).ok()) {
                    if (!field.name.empty()) return absl::OkStatus();
                  }
                  return absl::InvalidArgumentError(tensorstore::StrCat(
                      "Expected non-empty string, but received: ", v.dump()));
                case 1: {
                  std::string dtype_string;
                  TENSORSTORE_RETURN_IF_ERROR(
                      internal_json::JsonRequireValueAs(v, &dtype_string));
                  TENSORSTORE_ASSIGN_OR_RETURN(
                      static_cast<ZarrDType::BaseDType&>(field),
                      ParseBaseDType(dtype_string));
                  return absl::OkStatus();
                }
                case 2: {
                  return internal_json::JsonParseArray(
                      v,
                      [&](std::ptrdiff_t size) {
                        field.outer_shape.resize(size);
                        return absl::OkStatus();
                      },
                      [&](const ::nlohmann::json& x, std::ptrdiff_t j) {
                        return internal_json::JsonRequireInteger(
                            x, &field.outer_shape[j], /*strict=*/true, 1,
                            kInfIndex);
                      });
                }
                default:
                  ABSL_UNREACHABLE();  // COV_NF_LINE
              }
            });
      });
  if (!parse_result.ok()) return parse_result;
  return out;
}

}  // namespace

absl::Status ValidateDType(ZarrDType& dtype) {
  dtype.bytes_per_outer_element = 0;
  for (std::size_t field_i = 0; field_i < dtype.fields.size(); ++field_i) {
    auto& field = dtype.fields[field_i];
    if (std::any_of(
            dtype.fields.begin(), dtype.fields.begin() + field_i,
            [&](const ZarrDType::Field& f) { return f.name == field.name; })) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Field name ", QuoteString(field.name), " occurs more than once"));
    }
    field.field_shape.resize(field.flexible_shape.size() +
                             field.outer_shape.size());
    std::copy(field.flexible_shape.begin(), field.flexible_shape.end(),
              std::copy(field.outer_shape.begin(), field.outer_shape.end(),
                        field.field_shape.begin()));

    field.num_inner_elements = ProductOfExtents(span(field.field_shape));
    if (field.num_inner_elements == std::numeric_limits<Index>::max()) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Product of dimensions ", span(field.field_shape), " is too large"));
    }
    if (internal::MulOverflow(field.num_inner_elements,
                              static_cast<Index>(field.dtype->size),
                              &field.num_bytes)) {
      return absl::InvalidArgumentError("Field size in bytes is too large");
    }
    field.byte_offset = dtype.bytes_per_outer_element;
    if (internal::AddOverflow(dtype.bytes_per_outer_element, field.num_bytes,
                              &dtype.bytes_per_outer_element)) {
      return absl::InvalidArgumentError(
          "Total number of bytes per outer array element is too large");
    }
  }
  return absl::OkStatus();
}

Result<ZarrDType> ParseDType(const nlohmann::json& value) {
  TENSORSTORE_ASSIGN_OR_RETURN(ZarrDType dtype, ParseDTypeNoDerived(value));
  TENSORSTORE_RETURN_IF_ERROR(ValidateDType(dtype));
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

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(ZarrDType, [](auto is_loading,
                                                     const auto& options,
                                                     auto* obj, auto* j) {
  if constexpr (is_loading) {
    TENSORSTORE_ASSIGN_OR_RETURN(*obj, ParseDType(*j));
  } else {
    to_json(*j, *obj);
  }
  return absl::OkStatus();
})

char EndianIndicator(tensorstore::endian e) {
  return e == tensorstore::endian::little ? '<' : '>';
}

Result<ZarrDType::BaseDType> ChooseBaseDType(DataType dtype) {
  ZarrDType::BaseDType base_dtype;
  base_dtype.endian = endian::native;
  base_dtype.dtype = dtype;
  const auto set_typestr = [&](std::string_view typestr, int size) {
    if (size > 1) {
      base_dtype.encoded_dtype = tensorstore::StrCat(
          EndianIndicator(base_dtype.endian), typestr, size);
    } else {
      base_dtype.encoded_dtype = tensorstore::StrCat("|", typestr, size);
    }
  };
  switch (dtype.id()) {
    case DataTypeId::bool_t:
      set_typestr("b", 1);
      break;
    case DataTypeId::uint8_t:
      set_typestr("u", 1);
      break;
    case DataTypeId::uint16_t:
      set_typestr("u", 2);
      break;
    case DataTypeId::uint32_t:
      set_typestr("u", 4);
      break;
    case DataTypeId::uint64_t:
      set_typestr("u", 8);
      break;
    case DataTypeId::int8_t:
      set_typestr("i", 1);
      break;
    case DataTypeId::int16_t:
      set_typestr("i", 2);
      break;
    case DataTypeId::int32_t:
      set_typestr("i", 4);
      break;
    case DataTypeId::int64_t:
      set_typestr("i", 8);
      break;
    case DataTypeId::float16_t:
      set_typestr("f", 2);
      break;
    case DataTypeId::bfloat16_t:
      base_dtype.endian = endian::little;
      base_dtype.encoded_dtype = "bfloat16";
      break;
    case DataTypeId::float32_t:
      set_typestr("f", 4);
      break;
    case DataTypeId::float64_t:
      set_typestr("f", 8);
      break;
    case DataTypeId::complex64_t:
      set_typestr("c", 8);
      break;
    case DataTypeId::complex128_t:
      set_typestr("c", 16);
      break;
    default:
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Data type not supported: ", dtype));
  }
  return base_dtype;
}

}  // namespace internal_zarr
}  // namespace tensorstore
