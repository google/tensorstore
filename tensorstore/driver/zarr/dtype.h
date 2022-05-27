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

#ifndef TENSORSTORE_DRIVER_ZARR_DTYPE_H_
#define TENSORSTORE_DRIVER_ZARR_DTYPE_H_

/// \file
/// Support for encoding/decoding zarr "dtype" specifications.
/// See: https://zarr.readthedocs.io/en/stable/spec/v2.html

#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_zarr {

/// Decoded representation of a zarr "dtype" specification.
///
/// A zarr "dtype" is a JSON value that is either:
///
/// 1. A string, which specifies a single NumPy typestr value (see
///    https://docs.scipy.org/doc/numpy/reference/arrays.interface.html).  In
///    this case, the zarr array is considered to have a single, unnamed field.
///
/// 2. An array, where each element of the array is of the form:
///    `[name, typestr]` or `[name, typestr, shape]`, where `name` is a JSON
///    string specifying the unique, non-empty field name, `typestr` is a NumPy
///    typestr value, and `shape` is an optional "inner" array shape (specified
///    as a JSON array of non-negative integers) which defaults to the rank-0
///    shape `[]` if not specified.
///
/// Each field is encoded according to `typestr` into a fixed-size sequence of
/// bytes.  If the optional "inner" array `shape` is specified, the individual
/// elements are encoded in C order.  The encoding of each multi-field array
/// element is simply the concatenation of the encodings of each field.
///
/// The zarr "dtype" is a JSON representation of a NumPy data type (see
/// https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html), but
/// only the subset of possible NumPy dtype specifications, described above, is
/// supported.
///
/// For example:
///
///  - given a dtype of "<u2", the value 0x0456 is encoded as `0x56 0x04`.
///
///  - given a dtype of `[["a", "<u2"], ["b", "|u1", [2,3]]]`, the value
///    `{"a": 0x0456, "b": {{1, 2, 3}, {4, 5, 6}}}` is encoded as
///    `0x56 0x04 0x01 0x02 0x03 0x04 0x05 0x06`.
struct ZarrDType {
  /// Decoded representation of single NumPy typestr value.
  struct BaseDType {
    /// NumPy typestr value.
    std::string encoded_dtype;

    /// Corresponding DataType used for in-memory representation.
    DataType dtype;

    /// Endianness.
    tensorstore::endian endian;

    /// For "flexible" data types that are themselves arrays, this specifies the
    /// shape.  For regular data types, this is empty.
    std::vector<Index> flexible_shape;
  };

  /// Decoded representation of a single field.
  struct Field : public BaseDType {
    /// Optional `shape` dimensions specified by a zarr "dtype" field specified
    /// as a JSON array.  If the zarr dtype was specified as a single `typestr`
    /// value, or as a two-element array, this is empty.
    std::vector<Index> outer_shape;

    /// Field name.  Must be non-empty and unique if the zarr "dtype" was
    /// specified as an array.  Otherwise, is empty.
    std::string name;

    /// The inner array dimensions of this field, equal to the concatenation of
    ///  `outer_shape` and `flexible_shape` (derived value).
    std::vector<Index> field_shape;

    /// Product of `field_shape` dimensions (derived value).
    Index num_inner_elements;

    /// Byte offset of this field within an "outer" element (derived value).
    Index byte_offset;

    /// Number of bytes occupied by this field within an "outer" element
    /// (derived value).
    Index num_bytes;
  };

  /// Equal to `true` if the zarr "dtype" was specified as an array, in which
  /// case all fields must have a unique, non-empty `name`.  If `false`, there
  /// must be a single field with an empty `name`.
  bool has_fields;

  /// Decoded representation of the fields.
  std::vector<Field> fields;

  /// Bytes per "outer" element (derived value).
  Index bytes_per_outer_element;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ZarrDType,
                                          internal_json_binding::NoOptions)

  friend void to_json(::nlohmann::json& out,  // NOLINT
                      const ZarrDType& dtype);
};

/// Parses a zarr metadata "dtype" JSON specification.
///
/// \error `absl::StatusCode::kInvalidArgument` if `value` is not valid.
Result<ZarrDType> ParseDType(const ::nlohmann::json& value);

/// Validates `dtype and computes derived values.
///
/// \error `absl::StatusCode::kInvalidArgument` if two fields have the same
///     name.
/// \error `absl::StatusCode::kInvalidArgument` if the field size is too large.
absl::Status ValidateDType(ZarrDType& dtype);

/// Parses a NumPy typestr, which is used in the zarr "dtype" specification.
///
/// \error `absl::StatusCode::kInvalidArgument` if `dtype` is not valid.
Result<ZarrDType::BaseDType> ParseBaseDType(std::string_view dtype);

/// Chooses a zarr data type corresponding to `dtype`.
///
/// Always chooses native endian.
Result<ZarrDType::BaseDType> ChooseBaseDType(DataType dtype);

}  // namespace internal_zarr
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR_DTYPE_H_
