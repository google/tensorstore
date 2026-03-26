// Copyright 2025 The TensorStore Authors
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
#include <stdint.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

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

  // Handle r<N> raw bits type where N is number of bits (must be multiple of 8).
  // Parse N as uint64_t so values above 2^31-1 (e.g. r8589934592) are accepted;
  // (std::numeric_limits<uint64_t>::max() / 8) < std::numeric_limits<Index>::max(),
  // so num_bits / 8 always fits in Index.
  if (!dtype.empty() && dtype[0] == 'r' && dtype.size() > 1 &&
      absl::ascii_isdigit(dtype[1])) {
    std::string_view suffix = dtype.substr(1);
    uint64_t num_bits = 0;
    if (!absl::SimpleAtoi(suffix, &num_bits) || num_bits == 0 ||
        num_bits % 8 != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s data type is invalid; expected r<N> where N is a positive "
          "multiple of 8",
          dtype));
    }
    Index num_bytes = static_cast<Index>(num_bits / 8);
    return ZarrDType::BaseDType{std::string(dtype),
                                 dtype_v<::tensorstore::dtypes::byte_t>,
                                 {num_bytes}};
  }

  // Handle bare "r" - must have a number after it
  if (!dtype.empty() && dtype[0] == 'r') {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s data type is invalid; expected r<N> where N is a positive "
        "multiple of 8",
        dtype));
  }

  constexpr std::string_view kSupported =
      "bool, uint8, uint16, uint32, uint64, int8, int16, int32, int64, "
      "bfloat16, float16, float32, float64, complex64, complex128, r<N>";
  return absl::InvalidArgumentError(absl::StrFormat(
      "%s data type is not one of the supported data types: %s", dtype,
      kSupported));
}

namespace {

/// Validates that a fields array contains at least one field.
///
/// Per the Zarr v3 struct extension, the "fields" array MUST contain at least one field.
///
/// \param size The number of fields in the array.
/// \param type_name The data type name for error messages ("struct" or
///     "structured").
/// \error `absl::StatusCode::kInvalidArgument` if `size < 1`.
absl::Status ValidateFieldsArrayNotEmpty(const ptrdiff_t size,
                                         const std::string_view type_name) {
  if (size < 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s data type requires at least one field", type_name));
  }
  return absl::OkStatus();
}

/// Parses a single struct field.
///
/// Expected format: {"name": "field_name", "data_type": "float32"}
///
/// Note: Nested struct types and extension data types with configuration
/// (e.g., numpy.datetime64) are valid per the Zarr v3 spec but are not
/// currently supported by TensorStore.
///
/// \param field_json The JSON object representing a single field.
/// \param field[out] Filled with the parsed field on success.
/// \error `absl::StatusCode::kInvalidArgument` if `field_json` is not valid
absl::Status ParseObjectField(const nlohmann::json& field_json,
                              ZarrDType::Field& field) {
  if (!field_json.is_object()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "struct dtype requires fields as objects, but received: %s",
        field_json.dump()));
  }
  if (!field_json.contains("name") || !field_json.contains("data_type")) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Field object must contain 'name' and 'data_type', but received: %s",
        field_json.dump()));
  }
  TENSORSTORE_RETURN_IF_ERROR(
      internal_json::JsonRequireValueAs(field_json["name"], &field.name));
  if (field.name.empty()) {
    return absl::InvalidArgumentError("Field 'name' must be non-empty");
  }
  const auto& data_type_json = field_json["data_type"];
  if (data_type_json.is_object()) {
    return absl::InvalidArgumentError(
        "Nested struct types and extension data types with configuration are "
        "not supported by TensorStore. Field 'data_type' must be a string.");
  }
  std::string dtype_string;
  TENSORSTORE_RETURN_IF_ERROR(
      internal_json::JsonRequireValueAs(data_type_json, &dtype_string));
  TENSORSTORE_ASSIGN_OR_RETURN(static_cast<ZarrDType::BaseDType&>(field),
                               ParseBaseDType(dtype_string));
  return absl::OkStatus();
}

/// Parses a field in the legacy tuple format.
///
/// Expected format: ["name", "dtype"]
/// Used by "structured" (legacy format) and bare array format.
///
/// \param field_json The JSON array representing a single field.
/// \param field[out] Filled with the parsed field on success.
/// \error `absl::StatusCode::kInvalidArgument` if `field_json` is not a valid
///     field tuple.
absl::Status ParseTupleField(const nlohmann::json& field_json,
                             ZarrDType::Field& field) {
  if (!field_json.is_array()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "structured dtype requires fields as arrays, but received: %s",
        field_json.dump()));
  }
  return internal_json::JsonParseArray(
      field_json,
      [&](ptrdiff_t size) {
        if (size != 2) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Expected array of size 2, but received: %s", field_json.dump()));
        }
        return absl::OkStatus();
      },
      [&](const ::nlohmann::json& element, ptrdiff_t i) -> absl::Status {
        switch (i) {
          case 0:
            if (internal_json::JsonRequireValueAs(element, &field.name).ok()) {
              if (!field.name.empty()) return absl::OkStatus();
            }
            return absl::InvalidArgumentError(absl::StrFormat(
                "Expected non-empty string, but received: %s", element.dump()));
          case 1: {
            std::string dtype_string;
            TENSORSTORE_RETURN_IF_ERROR(
                internal_json::JsonRequireValueAs(element, &dtype_string));
            TENSORSTORE_ASSIGN_OR_RETURN(
                static_cast<ZarrDType::BaseDType&>(field),
                ParseBaseDType(dtype_string));
            return absl::OkStatus();
          }
          default:
            ABSL_UNREACHABLE();  // COV_NF_LINE
        }
      });
}

/// Parses the fields array for "struct" dtype.
///
/// Each field must be an object with "name" and "data_type" keys.
///
/// \param fields_json The JSON array of field objects.
/// \param out[out] Filled with the parsed fields on success.
/// \error `absl::StatusCode::kInvalidArgument` if the fields array is empty or
///     contains invalid field objects.
absl::Status ParseStructFieldsArray(const nlohmann::json& fields_json,
                                    ZarrDType& out) {
  out.has_fields = true;
  return internal_json::JsonParseArray(
      fields_json,
      [&](ptrdiff_t size) -> absl::Status {
        TENSORSTORE_RETURN_IF_ERROR(ValidateFieldsArrayNotEmpty(size, "struct"));
        out.fields.resize(size);
        return absl::OkStatus();
      },
      [&](const ::nlohmann::json& field_json, ptrdiff_t field_i) {
        return ParseObjectField(field_json, out.fields[field_i]);
      });
}

/// Parses the fields array for "structured" dtype or bare array format (legacy).
///
/// Each field must be a tuple: ["name", "dtype"].
///
/// \param fields_json The JSON array of field tuples.
/// \param out[out] Filled with the parsed fields on success.
/// \error `absl::StatusCode::kInvalidArgument` if the fields array is empty or
///     contains invalid field tuples.
absl::Status ParseStructuredFieldsArray(const nlohmann::json& fields_json,
                                        ZarrDType& out) {
  out.has_fields = true;
  return internal_json::JsonParseArray(
      fields_json,
      [&](ptrdiff_t size) -> absl::Status {
        TENSORSTORE_RETURN_IF_ERROR(
            ValidateFieldsArrayNotEmpty(size, "structured"));
        out.fields.resize(size);
        return absl::OkStatus();
      },
      [&](const ::nlohmann::json& field_json, ptrdiff_t field_i) {
        return ParseTupleField(field_json, out.fields[field_i]);
      });
}

/// Parses a zarr metadata "dtype" JSON specification, but does not compute any
/// derived values, and does not check for duplicate field names.
///
/// This is called by `ParseDType`.
///
/// \param value The zarr metadata "dtype" JSON specification.
/// \returns The parsed ZarrDType on success.
/// \error `absl::StatusCode::kInvalidArgument` if `value` is invalid.
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
      if (type_name == "struct") {
        // Zarr v3 spec format: fields must be objects
        const auto& config = value["configuration"];
        if (!config.is_object() || !config.contains("fields")) {
          return absl::InvalidArgumentError(
              "struct data type requires 'configuration' object with "
              "'fields' array");
        }
        TENSORSTORE_RETURN_IF_ERROR(ParseStructFieldsArray(config["fields"], out));
        return out;
      }
      if (type_name == "structured") {
        // Legacy format: fields must be tuples
        const auto& config = value["configuration"];
        if (!config.is_object() || !config.contains("fields")) {
          return absl::InvalidArgumentError(
              "structured data type requires 'configuration' object with "
              "'fields' array");
        }
        TENSORSTORE_RETURN_IF_ERROR(ParseStructuredFieldsArray(config["fields"], out));
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
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected string, array, or object with 'name' and 'configuration', "
        "but received: %s",
        value.dump()));
  }
  // Handle bare array format: [["field1", "type1"], ["field2", "type2"], ...]
  // This is the legacy format, so fields must be tuples
  TENSORSTORE_RETURN_IF_ERROR(ParseStructuredFieldsArray(value, out));
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
      return absl::InvalidArgumentError(absl::StrFormat(
          "Field name %v occurs more than once", QuoteString(field.name)));
    }
    field.field_shape = field.flexible_shape;

    field.num_inner_elements = ProductOfExtents(span(field.field_shape));
    if (field.num_inner_elements == std::numeric_limits<Index>::max()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Product of dimensions [%s] is too large",
          absl::StrJoin(field.field_shape, ", ")));
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
  // Zarr v3 struct extension format: {"name": "x", "data_type": "float32"}
  out = ::nlohmann::json::object();
  out["name"] = field.name;
  out["data_type"] = field.encoded_dtype;
}

void to_json(::nlohmann::json& out,  // NOLINT
             const ZarrDType& dtype) {
  if (!dtype.has_fields) {
    out = dtype.fields[0].encoded_dtype;
  } else {
    // Zarr v3 struct extension format: {"name": "struct", "configuration": {"fields": [...]}}
    out = ::nlohmann::json::object();
    out["name"] = "struct";
    out["configuration"] = ::nlohmann::json::object();
    out["configuration"]["fields"] = dtype.fields;
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

Result<ZarrDType::BaseDType> ChooseBaseDType(DataType dtype) {
  using D = ZarrDType::BaseDType;
  const auto make_dtype = [&](std::string_view name) -> Result<D> {
    return D{std::string(name), dtype, {}};
  };

  if (dtype == dtype_v<bool>) return make_dtype("bool");
  if (dtype == dtype_v<uint8_t>) return make_dtype("uint8");
  if (dtype == dtype_v<uint16_t>) return make_dtype("uint16");
  if (dtype == dtype_v<uint32_t>) return make_dtype("uint32");
  if (dtype == dtype_v<uint64_t>) return make_dtype("uint64");
  if (dtype == dtype_v<int8_t>) return make_dtype("int8");
  if (dtype == dtype_v<int16_t>) return make_dtype("int16");
  if (dtype == dtype_v<int32_t>) return make_dtype("int32");
  if (dtype == dtype_v<int64_t>) return make_dtype("int64");
  if (dtype == dtype_v<::tensorstore::dtypes::bfloat16_t>)
    return make_dtype("bfloat16");
  if (dtype == dtype_v<::tensorstore::dtypes::float16_t>)
    return make_dtype("float16");
  if (dtype == dtype_v<::tensorstore::dtypes::float32_t>)
    return make_dtype("float32");
  if (dtype == dtype_v<::tensorstore::dtypes::float64_t>)
    return make_dtype("float64");
  if (dtype == dtype_v<::tensorstore::dtypes::complex64_t>)
    return make_dtype("complex64");
  if (dtype == dtype_v<::tensorstore::dtypes::complex128_t>)
    return make_dtype("complex128");
  if (dtype == dtype_v<::tensorstore::dtypes::byte_t>) {
    ZarrDType::BaseDType base_dtype;
    base_dtype.dtype = dtype;
    base_dtype.encoded_dtype = "r8";
    base_dtype.flexible_shape = {1};
    return base_dtype;
  }
  if (dtype == dtype_v<::tensorstore::dtypes::char_t>) {
    // char_t encodes as r8, which parses back to byte_t
    ZarrDType::BaseDType base_dtype;
    base_dtype.dtype = dtype_v<::tensorstore::dtypes::byte_t>;
    base_dtype.encoded_dtype = "r8";
    base_dtype.flexible_shape = {1};
    return base_dtype;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Data type not supported: %v", dtype));
}

}  // namespace internal_zarr3
}  // namespace tensorstore
