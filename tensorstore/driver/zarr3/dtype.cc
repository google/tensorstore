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

#include "tensorstore/driver/zarr3/dtype.h"

#include <stddef.h>

#include <string>

#include "absl/base/optimization.h"
#include "absl/strings/ascii.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_zarr3 {

Result<ZarrDType::BaseDType> ParseBaseDType(std::string_view dtype) {
  using D = ZarrDType::BaseDType;
  const auto make_dtype = [&](DataType result_dtype) -> Result<D> {
    return D{std::string(dtype), result_dtype, {}};
  };

  if (dtype == "bool") return make_dtype(dtype_v<bool>);
  if (dtype == "uint8") return make_dtype(dtype_v<uint8_t>);
  if (dtype == "uint16") return make_dtype(dtype_v<uint16_t>);
  if (dtype == "uint32") return make_dtype(dtype_v<uint32_t>);
  if (dtype == "uint64") return make_dtype(dtype_v<uint64_t>);
  if (dtype == "int8") return make_dtype(dtype_v<int8_t>);
  if (dtype == "int16") return make_dtype(dtype_v<int16_t>);
  if (dtype == "int32") return make_dtype(dtype_v<int32_t>);
  if (dtype == "int64") return make_dtype(dtype_v<int64_t>);
  if (dtype == "bfloat16")
    return make_dtype(dtype_v<::tensorstore::dtypes::bfloat16_t>);
  if (dtype == "float16")
    return make_dtype(dtype_v<::tensorstore::dtypes::float16_t>);
  if (dtype == "float32")
    return make_dtype(dtype_v<::tensorstore::dtypes::float32_t>);
  if (dtype == "float64")
    return make_dtype(dtype_v<::tensorstore::dtypes::float64_t>);
  if (dtype == "complex64")
    return make_dtype(dtype_v<::tensorstore::dtypes::complex64_t>);
  if (dtype == "complex128")
    return make_dtype(dtype_v<::tensorstore::dtypes::complex128_t>);

  // Handle r<N> raw bits type where N is number of bits (must be multiple of 8)
  if (dtype.size() > 1 && dtype[0] == 'r' && absl::ascii_isdigit(dtype[1])) {
    std::string_view suffix = dtype.substr(1);
    Index num_bits = 0;
    if (!absl::SimpleAtoi(suffix, &num_bits) ||
        num_bits == 0 ||
        num_bits % 8 != 0) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          dtype, " data type is invalid; expected r<N> where N is a positive "
                 "multiple of 8"));
    }
    Index num_bytes = num_bits / 8;
    return ZarrDType::BaseDType{std::string(dtype),
                                 dtype_v<::tensorstore::dtypes::byte_t>,
                                 {num_bytes}};
  }

  constexpr std::string_view kSupported =
      "bool, uint8, uint16, uint32, uint64, int8, int16, int32, int64, "
      "bfloat16, float16, float32, float64, complex64, complex128, r<N>";
  return absl::InvalidArgumentError(
      tensorstore::StrCat(dtype, " data type is not one of the supported "
                                 "data types: ",
                          kSupported));
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
// Helper to parse fields array (used by both array format and object format)
absl::Status ParseFieldsArray(const nlohmann::json& fields_json,
                               ZarrDType& out) {
  out.has_fields = true;
  return internal_json::JsonParseArray(
      fields_json,
      [&](ptrdiff_t size) {
        out.fields.resize(size);
        return absl::OkStatus();
      },
      [&](const ::nlohmann::json& x, ptrdiff_t field_i) {
        auto& field = out.fields[field_i];
        return internal_json::JsonParseArray(
            x,
            [&](ptrdiff_t size) {
              if (size < 2 || size > 3) {
                return absl::InvalidArgumentError(tensorstore::StrCat(
                    "Expected array of size 2 or 3, but received: ", x.dump()));
              }
              return absl::OkStatus();
            },
            [&](const ::nlohmann::json& v, ptrdiff_t i) {
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
                      [&](ptrdiff_t size) {
                        field.outer_shape.resize(size);
                        return absl::OkStatus();
                      },
                      [&](const ::nlohmann::json& x, ptrdiff_t j) {
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
}

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
  // Handle extended object format:
  // {"name": "structured", "configuration": {"fields": [...]}}
  if (value.is_object()) {
    if (value.contains("name") && value.contains("configuration")) {
      std::string type_name;
      TENSORSTORE_RETURN_IF_ERROR(
          internal_json::JsonRequireValueAs(value["name"], &type_name));
      if (type_name == "structured") {
        const auto& config = value["configuration"];
        if (!config.is_object() || !config.contains("fields")) {
          return absl::InvalidArgumentError(
              "Structured data type requires 'configuration' object with "
              "'fields' array");
        }
        TENSORSTORE_RETURN_IF_ERROR(ParseFieldsArray(config["fields"], out));
        return out;
      }
      if (type_name == "raw_bytes") {
        const auto& config = value["configuration"];
        if (!config.is_object() || !config.contains("length_bytes")) {
          return absl::InvalidArgumentError(
              "raw_bytes data type requires 'configuration' object with "
              "'length_bytes' field");
        }
        Index length_bytes;
        TENSORSTORE_RETURN_IF_ERROR(
            internal_json::JsonRequireValueAs(config["length_bytes"], &length_bytes));
        if (length_bytes <= 0) {
          return absl::InvalidArgumentError(
              "raw_bytes length_bytes must be positive");
        }
        out.has_fields = false;
        out.fields.resize(1);
        out.fields[0].encoded_dtype = "raw_bytes";
        out.fields[0].dtype = dtype_v<tensorstore::dtypes::byte_t>;
        out.fields[0].flexible_shape = {length_bytes};
        out.fields[0].outer_shape = {};
        out.fields[0].name = "";
        out.fields[0].field_shape = {length_bytes};
        out.fields[0].num_inner_elements = length_bytes;
        out.fields[0].byte_offset = 0;
        out.fields[0].num_bytes = length_bytes;
        out.bytes_per_outer_element = length_bytes;
        return out;
      }
      // For other named types, try to parse as a base dtype
      out.has_fields = false;
      out.fields.resize(1);
      TENSORSTORE_ASSIGN_OR_RETURN(
          static_cast<ZarrDType::BaseDType&>(out.fields[0]),
          ParseBaseDType(type_name));
      return out;
    }
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Expected string, array, or object with 'name' and 'configuration', "
        "but received: ",
        value.dump()));
  }
  // Handle array format: [["field1", "type1"], ["field2", "type2"], ...]
  TENSORSTORE_RETURN_IF_ERROR(ParseFieldsArray(value, out));
  return out;
}

}  // namespace

absl::Status ValidateDType(ZarrDType& dtype) {
  dtype.bytes_per_outer_element = 0;
  for (size_t field_i = 0; field_i < dtype.fields.size(); ++field_i) {
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

std::optional<DataType> GetScalarDataType(const ZarrDType& dtype) {
  if (!dtype.has_fields && !dtype.fields.empty()) {
    return dtype.fields[0].dtype;
  }
  return std::nullopt;
}

Result<ZarrDType> ParseDType(const nlohmann::json& value) {
  TENSORSTORE_ASSIGN_OR_RETURN(ZarrDType dtype, ParseDTypeNoDerived(value));
  TENSORSTORE_RETURN_IF_ERROR(ValidateDType(dtype));
  return dtype;
}

bool operator==(const ZarrDType::BaseDType& a,
                const ZarrDType::BaseDType& b) {
  return a.encoded_dtype == b.encoded_dtype && a.dtype == b.dtype &&
         a.flexible_shape == b.flexible_shape;
}

bool operator!=(const ZarrDType::BaseDType& a,
                const ZarrDType::BaseDType& b) {
  return !(a == b);
}

bool operator==(const ZarrDType::Field& a, const ZarrDType::Field& b) {
  return static_cast<const ZarrDType::BaseDType&>(a) ==
             static_cast<const ZarrDType::BaseDType&>(b) &&
         a.outer_shape == b.outer_shape && a.name == b.name &&
         a.field_shape == b.field_shape &&
         a.num_inner_elements == b.num_inner_elements &&
         a.byte_offset == b.byte_offset && a.num_bytes == b.num_bytes;
}

bool operator!=(const ZarrDType::Field& a, const ZarrDType::Field& b) {
  return !(a == b);
}

bool operator==(const ZarrDType& a, const ZarrDType& b) {
  return a.has_fields == b.has_fields &&
         a.bytes_per_outer_element == b.bytes_per_outer_element &&
         a.fields == b.fields;
}

bool operator!=(const ZarrDType& a, const ZarrDType& b) { return !(a == b); }

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

namespace {

Result<ZarrDType::BaseDType> MakeBaseDType(std::string_view name,
                                           DataType dtype) {
  ZarrDType::BaseDType base_dtype;
  base_dtype.dtype = dtype;
  base_dtype.encoded_dtype = std::string(name);
  return base_dtype;
}

}  // namespace

Result<ZarrDType::BaseDType> ChooseBaseDType(DataType dtype) {
  if (dtype == dtype_v<bool>) return MakeBaseDType("bool", dtype);
  if (dtype == dtype_v<uint8_t>) return MakeBaseDType("uint8", dtype);
  if (dtype == dtype_v<uint16_t>) return MakeBaseDType("uint16", dtype);
  if (dtype == dtype_v<uint32_t>) return MakeBaseDType("uint32", dtype);
  if (dtype == dtype_v<uint64_t>) return MakeBaseDType("uint64", dtype);
  if (dtype == dtype_v<int8_t>) return MakeBaseDType("int8", dtype);
  if (dtype == dtype_v<int16_t>) return MakeBaseDType("int16", dtype);
  if (dtype == dtype_v<int32_t>) return MakeBaseDType("int32", dtype);
  if (dtype == dtype_v<int64_t>) return MakeBaseDType("int64", dtype);
  if (dtype == dtype_v<::tensorstore::dtypes::bfloat16_t>)
    return MakeBaseDType("bfloat16", dtype);
  if (dtype == dtype_v<::tensorstore::dtypes::float16_t>)
    return MakeBaseDType("float16", dtype);
  if (dtype == dtype_v<::tensorstore::dtypes::float32_t>)
    return MakeBaseDType("float32", dtype);
  if (dtype == dtype_v<::tensorstore::dtypes::float64_t>)
    return MakeBaseDType("float64", dtype);
  if (dtype == dtype_v<::tensorstore::dtypes::complex64_t>)
    return MakeBaseDType("complex64", dtype);
  if (dtype == dtype_v<::tensorstore::dtypes::complex128_t>)
    return MakeBaseDType("complex128", dtype);
  if (dtype == dtype_v<::tensorstore::dtypes::byte_t>)
    return MakeBaseDType("r8", dtype);
  if (dtype == dtype_v<::tensorstore::dtypes::char_t>)
    return MakeBaseDType("r8", dtype);
  return absl::InvalidArgumentError(
      tensorstore::StrCat("Data type not supported: ", dtype));
}

}  // namespace internal_zarr3
}  // namespace tensorstore
