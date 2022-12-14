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

#include "tensorstore/internal/json/array.h"

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json {

::nlohmann::json JsonEncodeNestedArrayImpl(
    ArrayView<const void, dynamic_rank, offset_origin> array,
    absl::FunctionRef<::nlohmann::json(const void*)> encode_element) {
  // To avoid the possibility of stack overflow, this implementation is
  // non-recursive.

  // Special case rank-0 arrays, because we assume below that there is always a
  // parent array to which elements are added.
  if (array.rank() == 0) {
    return encode_element(array.data());
  }

  // Pointer to next array element to encode.
  ByteStridedPointer<const void> pointer = array.byte_strided_origin_pointer();

  // `path[0], ..., path[level]` specify the ancestor JSON arrays of the next
  // encoded element.  `path[0]` is the root, `path[level]` is the immediate
  // parent that will contain the next encoded element.
  using array_t = ::nlohmann::json::array_t;
  absl::FixedArray<array_t*, internal::kNumInlinedDims> path(array.rank());
  DimensionIndex level = 0;

  // Nested array result value.
  array_t j_root;
  j_root.reserve(array.shape()[0]);
  path[0] = &j_root;

  if (array.shape()[0] == 0) {
    return j_root;
  }

  while (true) {
    // Encode the next element of the current level, either as a terminal
    // element (if `level == array.rank() - 1`) as a nested array (if
    // `level < array.rank() - 1`).
    array_t* j_parent = path[level];
    if (level == array.rank() - 1) {
      j_parent->push_back(encode_element(pointer.get()));
    } else {
      // We are not at the last array dimension.  Create a new JSON array and
      // recurse into it.
      const Index size = array.shape()[level + 1];
      array_t next_array;
      next_array.reserve(size);
      j_parent->emplace_back(std::move(next_array));
      j_parent = j_parent->back().get_ptr<array_t*>();
      if (size != 0) {
        path[++level] = j_parent;
        // Recurse into next nesting level.
        continue;
      }
      // Since this dimension has size 0, we don't recurse further.
    }

    // Advance to the next element to encode.
    while (true) {
      array_t* j_array = path[level];
      const Index i = j_array->size();
      const Index size = array.shape()[level];
      const Index byte_stride = array.byte_strides()[level];
      // Advance to the next element at the current nesting level.
      pointer += byte_stride;
      if (i != size) break;
      // We reached the end of the current nesting level: return to the parent
      // level.
      pointer -= i * byte_stride;
      if (level-- == 0) {
        // We reached the end of the first level.  No more elements to encode.
        return j_root;
      }
    }
  }
}

Result<SharedArray<void>> JsonParseNestedArrayImpl(
    const ::nlohmann::json& j_root, DataType dtype,
    absl::FunctionRef<absl::Status(const ::nlohmann::json& v, void* out)>
        decode_element) {
  assert(dtype.valid());
  // To avoid the possibility of stack overflow, this implementation is
  // non-recursive.
  using array_t = ::nlohmann::json::array_t;

  // The result array, allocated once the shape has been determined.
  SharedArray<void> array;

  // Pointer to the next element in the result array to assign.
  ByteStridedPointer<void> pointer;
  const Index byte_stride = dtype->size;

  // Prior to `array` being allocated, stores the shape up to the current
  // nesting level of `path.size()`.  After `array` is allocated,
  // `shape_or_position[0:path.size()]` specifies the current position up to the
  // current nesting level.
  absl::InlinedVector<Index, internal::kNumInlinedDims> shape_or_position;

  // The stack of JSON arrays corresponding to the current nesting level of
  // `path.size()`.
  absl::InlinedVector<const array_t*, internal::kNumInlinedDims> path;

  // The new JSON value for the `deeper_level` code to process (not previously
  // seen).
  const ::nlohmann::json* j = &j_root;

  const auto allocate_array = [&] {
    array = AllocateArray(shape_or_position, c_order, default_init, dtype);
    pointer = array.byte_strided_origin_pointer();
    // Convert shape vector to position vector.
    std::fill(shape_or_position.begin(), shape_or_position.end(), 0);
  };

  while (true) {
    // Control transferred here from a shallower nesting level (or the start).
    // Process a new element `j` that has not been seen before.

    const array_t* j_array = j->get_ptr<const ::nlohmann::json::array_t*>();
    if (!j_array) {
      // The new element is not an array: handle leaf case.
      if (!array.data()) allocate_array();
      if (path.size() != static_cast<std::size_t>(array.rank())) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Expected rank-", shape_or_position.size(),
            " array, but found non-array element ", j->dump(), " at position ",
            span(shape_or_position.data(), path.size()), "."));
      }
      TENSORSTORE_RETURN_IF_ERROR(
          decode_element(*j, pointer.get()),
          MaybeAnnotateStatus(
              _, tensorstore::StrCat("Error parsing array element at position ",
                                     span(shape_or_position))));
      pointer += byte_stride;
    } else {
      // The new element is an array: handle another nesting level.
      path.push_back(j_array);
      const Index size = j_array->size();
      if (!array.data()) {
        shape_or_position.push_back(size);
        if (shape_or_position.size() > kMaxRank) {
          return absl::InvalidArgumentError(tensorstore::StrCat(
              "Nesting level exceeds maximum rank of ", kMaxRank));
        }
        if (size == 0) {
          // Allocate zero-element array.
          allocate_array();
          return array;
        }
      } else if (path.size() > static_cast<size_t>(array.rank())) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Expected rank-", array.rank(), " array, but found array element ",
            j->dump(), " at position ", span(shape_or_position), "."));
      } else if (array.shape()[path.size() - 1] != size) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Expected array of shape ", array.shape(),
            ", but found array element ", j->dump(), " of length ", size,
            " at position ", span(shape_or_position.data(), path.size() - 1),
            "."));
      }

      // Process first element of the array.
      j = &(*j_array)[0];
      continue;
    }

    // Advance to the next element to decode.
    while (true) {
      if (path.empty()) {
        // No more elements left.
        return array;
      }
      const array_t* j_array = path.back();
      const Index size = j_array->size();

      // Increment position at current nesting level.
      const Index i = ++shape_or_position[path.size() - 1];
      if (i != size) {
        // Process next element of the array.
        j = &(*j_array)[i];
        break;
      }

      // Reached the end of the array at the current nesting level, return to
      // the next lower level.
      shape_or_position[path.size() - 1] = 0;
      path.pop_back();
    }
  }
}

Result<::nlohmann::json> JsonEncodeNestedArray(ArrayView<const void> array) {
  auto convert = internal::GetDataTypeConverter(array.dtype(), dtype_v<json_t>);
  if (!(convert.flags & DataTypeConversionFlags::kSupported)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Conversion from ", array.dtype(), " to JSON is not implemented"));
  }
  bool error = false;
  absl::Status status;
  ::nlohmann::json j = JsonEncodeNestedArrayImpl(
      array, [&](const void* ptr) -> ::nlohmann::json {
        if ((convert.flags & DataTypeConversionFlags::kCanReinterpretCast) ==
            DataTypeConversionFlags::kCanReinterpretCast) {
          return *reinterpret_cast<const json_t*>(ptr);
        }
        ::nlohmann::json value;
        if ((*convert.closure
                  .function)[internal::IterationBufferKind::kContiguous](
                convert.closure.context, 1,
                internal::IterationBufferPointer(const_cast<void*>(ptr),
                                                 Index(0)),
                internal::IterationBufferPointer(&value, Index(0)),
                &status) != 1) {
          error = true;
          return nullptr;
        }
        return value;
      });
  if (error) return internal::GetElementCopyErrorStatus(std::move(status));
  return j;
}

Result<SharedArray<void>> JsonParseNestedArray(const ::nlohmann::json& j,
                                               DataType dtype,
                                               DimensionIndex rank_constraint) {
  auto convert = internal::GetDataTypeConverter(dtype_v<json_t>, dtype);
  if (!(convert.flags & DataTypeConversionFlags::kSupported)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Conversion from JSON to ", dtype, " is not implemented"));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto array,
      JsonParseNestedArrayImpl(
          j, dtype, [&](const ::nlohmann::json& v, void* out) -> absl::Status {
            if ((convert.flags &
                 DataTypeConversionFlags::kCanReinterpretCast) ==
                DataTypeConversionFlags::kCanReinterpretCast) {
              *reinterpret_cast<json_t*>(out) = v;
              return absl::OkStatus();
            } else {
              absl::Status status;
              if ((*convert.closure
                        .function)[internal::IterationBufferKind::kContiguous](
                      convert.closure.context, 1,
                      internal::IterationBufferPointer(
                          const_cast<::nlohmann::json*>(&v), Index(0)),
                      internal::IterationBufferPointer(out, Index(0)),
                      &status) != 1) {
                return internal::GetElementCopyErrorStatus(std::move(status));
              }
              return absl::OkStatus();
            }
          }));
  if (rank_constraint != dynamic_rank && array.rank() != rank_constraint) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Array rank (", array.rank(), ") does not match expected rank (",
        rank_constraint, ")"));
  }
  return array;
}

}  // namespace internal_json
}  // namespace tensorstore
