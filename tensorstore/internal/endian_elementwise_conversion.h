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

#ifndef TENSORSTORE_INTERNAL_ENDIAN_ELEMENTWISE_CONVERSION_H_
#define TENSORSTORE_INTERNAL_ENDIAN_ELEMENTWISE_CONVERSION_H_

#include <stddef.h>

#include <algorithm>
#include <array>
#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/riegeli/delimited.h"
#include "tensorstore/internal/riegeli/json_input.h"
#include "tensorstore/internal/riegeli/json_output.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/util/utf8_string.h"

namespace tensorstore {
namespace internal {

template <size_t SubElementSize, size_t NumSubElements>
struct SwapEndianUnalignedLoopImpl {
  // Type used as a placeholder for a value of size
  // `SubElementSize*NumElements` without an alignment requirement.  To avoid
  // running afoul of C++ strict aliasing rules, this type should not actually
  // be used to read or write data.
  using UnalignedValue =
      std::array<unsigned char, SubElementSize * NumSubElements>;
  static_assert(sizeof(UnalignedValue) == SubElementSize * NumSubElements);
  static_assert(alignof(UnalignedValue) == 1);

  void operator()(UnalignedValue* value, void* arg) const {
    SwapEndianUnalignedInplace<SubElementSize, NumSubElements>(value);
  }

  void operator()(const UnalignedValue* source, UnalignedValue* target,
                  void* arg) const {
    SwapEndianUnaligned<SubElementSize, NumSubElements>(source, target);
  }

  using InplaceLoopImpl = internal_elementwise_function::SimpleLoopTemplate<
      SwapEndianUnalignedLoopImpl<SubElementSize, NumSubElements>(
          UnalignedValue),
      void*>;

  using LoopImpl = internal_elementwise_function::SimpleLoopTemplate<
      SwapEndianUnalignedLoopImpl<SubElementSize, NumSubElements>(
          UnalignedValue, UnalignedValue),
      void*>;
};

/// Swaps endianness of elements of an array in place.
///
/// Each element of the array has a total size of
/// `SubElementSize*NumSubElements` bytes.  The byte swapping is performed
/// independently on each sub-element of `SubElementSize` bytes.
///
/// \param SubElementSize Size in bytes of each sub-element.
/// \param NumSubElements Number of sub-elements in each element.
template <size_t SubElementSize, size_t NumSubElements = 1>
using SwapEndianUnalignedInplaceLoopTemplate =
    typename SwapEndianUnalignedLoopImpl<SubElementSize,
                                         NumSubElements>::InplaceLoopImpl;

/// Copies unaligned elements from one array to another, optionally swapping
/// endianness.
///
/// Each element of the array has a total size of
/// `SubElementSize*NumSubElements` bytes.  The byte swapping is performed
/// independently on each sub-element of `SubElementSize` bytes.
///
/// If `SubElementSize > 1`, the byte order is swapped.  If
/// `SubElementSize == 1`, elements are just copied without swapping the byte
/// order (since there would be nothing to swap).  For example:
///
/// - To copy an array of `std::complex<float32_t>` elements
///   with byte swapping, specify `SubElementSize=4` and `NumSubElements=2`.
///
/// - To copy an array of `uint32_t` elements with byte swapping, specify
///   `SubElementSize=4` and `NumSubElements=1`.
///
/// - To copy an array of `uint32_t` elements without byte swapping, specify
///   `SubElementSize=1` and `NumSubElements=4`.
///
/// \param SubElementSize Size in bytes of each sub-element.
/// \param NumSubElements Number of sub-elements in each element.
template <size_t SubElementSize, size_t NumSubElements = 1>
using SwapEndianUnalignedLoopTemplate =
    typename SwapEndianUnalignedLoopImpl<SubElementSize,
                                         NumSubElements>::LoopImpl;

/// Specialized Serializer used to read/write non-trivial data types from/to
/// `riegeli::Reader` and `riegeli::Writer`, respectively.
template <typename Element>
struct NonTrivialDataTypeSerializer;

/// Specialized serializer for `std::string`.
///
/// Each element is written as a delimited string.
template <>
struct NonTrivialDataTypeSerializer<std::string> {
  [[nodiscard]] static bool Write(riegeli::Writer& writer,
                                  const std::string& value) {
    return serialization::WriteDelimited(writer, value);
  }
  [[nodiscard]] static bool Read(riegeli::Reader& reader, std::string& value) {
    return serialization::ReadDelimited(reader, value);
  }
};

/// Specialized serializer for `Utf8String`.
///
/// Each element is written as a delimited string, and validated when read.
template <>
struct NonTrivialDataTypeSerializer<Utf8String> {
  [[nodiscard]] static bool Write(riegeli::Writer& writer,
                                  const Utf8String& value) {
    return serialization::WriteDelimited(writer, value.utf8);
  }
  [[nodiscard]] static bool Read(riegeli::Reader& reader, Utf8String& value) {
    return serialization::ReadDelimitedUtf8(reader, value.utf8);
  }
};

/// Specialized serializer for `::nlohmann::json`.
///
/// Each element is written in CBOR format, which is self-delimited.
template <>
struct NonTrivialDataTypeSerializer<::nlohmann::json> {
  [[nodiscard]] static bool Write(riegeli::Writer& writer,
                                  const ::nlohmann::json& value) {
    return internal::WriteCbor(writer, value);
  }
  [[nodiscard]] static bool Read(riegeli::Reader& reader,
                                 ::nlohmann::json& value) {
    return internal::ReadCbor(reader, value, /*strict=*/false);
  }
};

/// Writes an array of non-trivial elements to a `riegeli::Writer`, using
/// `NonTrivialDataTypeSerializer<Element>::Write`.
template <typename Element>
struct WriteNonTrivialLoopImpl {
  bool operator()(riegeli::Writer& writer, const Element* element,
                  void*) const {
    return NonTrivialDataTypeSerializer<Element>::Write(writer, *element);
  }
};
template <typename Element>
using WriteNonTrivialLoopTemplate =
    internal_elementwise_function::SimpleLoopTemplate<
        internal_elementwise_function::Stateless<
            riegeli::Writer, WriteNonTrivialLoopImpl<Element>>(const Element),
        void*>;

/// Reads an array of non-trivial elements from a `riegeli::Reader`, using
/// `NonTrivialDataTypeSerializer<Element>::Read`.
template <typename Element>
struct ReadNonTrivialLoopImpl {
  bool operator()(riegeli::Reader& reader, Element* element, void*) const {
    return NonTrivialDataTypeSerializer<Element>::Read(reader, *element);
  }
};
template <typename Element>
using ReadNonTrivialLoopTemplate =
    internal_elementwise_function::SimpleLoopTemplate<
        internal_elementwise_function::Stateless<
            riegeli::Reader, ReadNonTrivialLoopImpl<Element>>(Element),
        void*>;

/// Writes an array of trivial elements to a `riegeli::Writer`, optionally
/// swapping byte order.
///
/// Each element of the array has a total size of
/// `SubElementSize*NumSubElements` bytes.  The byte swapping is performed
/// independently on each sub-element of `SubElementSize` bytes.
///
/// If `SubElementSize > 1`, the byte order is swapped.  If
/// `SubElementSize == 1`, elements are just copied without swapping the byte
/// order.
///
/// \param SubElementSize Size in bytes of each sub-element.
/// \param NumSubElements Number of sub-elements in each element.
template <size_t SubElementSize, size_t NumSubElements>
struct WriteSwapEndianLoopTemplate {
  using Element = std::array<unsigned char, SubElementSize * NumSubElements>;

  using ElementwiseFunctionType = ElementwiseFunction<1, void*>;
  template <typename ArrayAccessor>
  static bool Loop(void* context, internal::IterationBufferShape shape,
                   IterationBufferPointer source, void* /*arg*/) {
    auto& writer = *reinterpret_cast<riegeli::Writer*>(context);
    for (Index outer_i = 0; outer_i < shape[0]; ++outer_i) {
      if constexpr (SubElementSize == 1 &&
                    ArrayAccessor::buffer_kind ==
                        internal::IterationBufferKind::kContiguous) {
        // Fast path: source array is contiguous and byte swapping is not
        // required.
        if (!writer.Write(std::string_view(
                reinterpret_cast<const char*>(
                    ArrayAccessor::template GetPointerAtPosition<Element>(
                        source, outer_i, 0)),
                shape[1] * sizeof(Element)))) {
          return false;
        }
      } else {
        Index element_i = 0;
        while (element_i < shape[1]) {
          const size_t remaining_bytes =
              (shape[1] - element_i) * sizeof(Element);
          if (!writer.Push(/*min_length=*/sizeof(Element),
                           /*recommended_length=*/remaining_bytes)) {
            return false;
          }
          const Index end_element_i = std::min(
              shape[1], static_cast<Index>(element_i + (writer.available() /
                                                        sizeof(Element))));
          char* cursor = writer.cursor();
          for (; element_i < end_element_i; ++element_i) {
            SwapEndianUnaligned<SubElementSize, NumSubElements>(
                ArrayAccessor::template GetPointerAtPosition<Element>(
                    source, outer_i, element_i),
                cursor);
            cursor += sizeof(Element);
          }
          element_i = end_element_i;
          writer.set_cursor(cursor);
        }
      }
    }
    return true;
  }
};

/// Reads an array of trivial elements from a `riegeli::Reader`, optionally
/// swapping byte order.
///
/// Each element of the array has a total size of
/// `SubElementSize*NumSubElements` bytes.  The byte swapping is performed
/// independently on each sub-element of `SubElementSize` bytes.
///
/// If `SubElementSize > 1`, the byte order is swapped.  If
/// `SubElementSize == 1`, elements are just copied without swapping the byte
/// order.
///
/// \param SubElementSize Size in bytes of each sub-element.
/// \param NumSubElements Number of sub-elements in each element.
/// \param IsBool Indicates that the element type is `bool`.  If `true`, ensures
///     that decoded values are `0` or `1` (as required by `bool`).
template <size_t SubElementSize, size_t NumSubElements, bool IsBool = false>
struct ReadSwapEndianLoopTemplate {
  static_assert(!IsBool || (SubElementSize == 1 && NumSubElements == 1));
  using Element = std::array<unsigned char, SubElementSize * NumSubElements>;

  using ElementwiseFunctionType = ElementwiseFunction<1, void*>;
  template <typename ArrayAccessor>
  static bool Loop(void* context, internal::IterationBufferShape shape,
                   IterationBufferPointer source, void* /*arg*/) {
    auto& reader = *reinterpret_cast<riegeli::Reader*>(context);
    for (Index outer_i = 0; outer_i < shape[0]; ++outer_i) {
      if constexpr (SubElementSize == 1 &&
                    ArrayAccessor::buffer_kind ==
                        internal::IterationBufferKind::kContiguous &&
                    !IsBool) {
        // Fast path: destination array is contiguous and byte swapping is not
        // required.
        if (!reader.Read(
                shape[1] * sizeof(Element),
                reinterpret_cast<char*>(
                    ArrayAccessor::template GetPointerAtPosition<Element>(
                        source, outer_i, 0)))) {
          return false;
        }
      } else {
        Index element_i = 0;
        while (element_i < shape[1]) {
          const size_t remaining_bytes =
              (shape[1] - element_i) * sizeof(Element);
          if (!reader.Pull(/*min_length=*/sizeof(Element),
                           /*recommended_length=*/remaining_bytes)) {
            return false;
          }
          const Index end_element_i = std::min(
              shape[1], static_cast<Index>(element_i + (reader.available() /
                                                        sizeof(Element))));
          const char* cursor = reader.cursor();
          for (; element_i < end_element_i; ++element_i) {
            if constexpr (IsBool) {
              unsigned char val = static_cast<unsigned char>(*cursor);
              if (val & ~static_cast<unsigned char>(1)) {
                reader.set_cursor(cursor);
                reader.Fail(absl::InvalidArgumentError(
                    tensorstore::StrCat("Invalid bool value: ",
                                        static_cast<unsigned int>(*cursor))));
                return false;
              }
              *ArrayAccessor::template GetPointerAtPosition<bool>(
                  source, outer_i, element_i) = static_cast<bool>(val);
            } else {
              SwapEndianUnaligned<SubElementSize, NumSubElements>(
                  cursor, ArrayAccessor::template GetPointerAtPosition<Element>(
                              source, outer_i, element_i));
            }
            cursor += sizeof(Element);
          }
          element_i = end_element_i;
          reader.set_cursor(cursor);
        }
      }
    }
    return true;
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ENDIAN_ELEMENTWISE_CONVERSION_H_
