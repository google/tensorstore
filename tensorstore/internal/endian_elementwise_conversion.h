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

#include <array>
#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/riegeli_json_input.h"
#include "tensorstore/internal/riegeli_json_output.h"
#include "tensorstore/serialization/riegeli_delimited.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/utf8_string.h"

namespace tensorstore {
namespace internal {

/// Swaps endianness of elements of an array in place.
///
/// Each element of the array has a total size of
/// `SubElementSize*NumSubElements` bytes.  The byte swapping is performed
/// independently on each sub-element of `SubElementSize` bytes.
///
/// \param SubElementSize Size in bytes of each sub-element.
/// \param NumSubElements Number of sub-elements in each element.
template <size_t SubElementSize, size_t NumSubElements = 1>
struct SwapEndianUnalignedInplaceLoopTemplate {
  using ElementwiseFunctionType = ElementwiseFunction<1, absl::Status*>;
  template <typename ArrayAccessor>
  static Index Loop(void* context, Index count, IterationBufferPointer pointer,
                    absl::Status* /*status*/) {
    // Type used as a placeholder for a value of size
    // `SubElementSize*NumElements` without an alignment requirement.  To avoid
    // running afoul of C++ strict aliasing rules, this type should not actually
    // be used to read or write data.
    using UnalignedValue =
        std::array<unsigned char, SubElementSize * NumSubElements>;
    static_assert(sizeof(UnalignedValue) == SubElementSize * NumSubElements);
    static_assert(alignof(UnalignedValue) == 1);
    // TODO(jbms): check if this loop gets auto-vectorized properly, and if not,
    // consider manually creating a vectorized implementation.
    for (Index i = 0; i < count; ++i) {
      SwapEndianUnalignedInplace<SubElementSize, NumSubElements>(
          ArrayAccessor::template GetPointerAtOffset<UnalignedValue>(pointer,
                                                                     i));
    }
    return count;
  }
};

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
struct SwapEndianUnalignedLoopTemplate {
  using ElementwiseFunctionType = ElementwiseFunction<2, absl::Status*>;
  template <typename ArrayAccessor>
  static Index Loop(void* context, Index count, IterationBufferPointer source,
                    IterationBufferPointer dest, absl::Status* /*status*/) {
    // Type used as a placeholder for a value of size
    // `SubElementSize*NumSubElements` without an alignment requirement.  To
    // avoid running afoul of C++ strict aliasing rules, this type should not
    // actually be used to read or write data.
    using UnalignedValue =
        std::array<unsigned char, SubElementSize * NumSubElements>;
    static_assert(sizeof(UnalignedValue) == SubElementSize * NumSubElements);
    static_assert(alignof(UnalignedValue) == 1);
    // TODO(jbms): check if this loop gets auto-vectorized properly, and if not,
    // consider manually creating a vectorized implementation.
    for (Index i = 0; i < count; ++i) {
      SwapEndianUnaligned<SubElementSize, NumSubElements>(
          ArrayAccessor::template GetPointerAtOffset<UnalignedValue>(source, i),
          ArrayAccessor::template GetPointerAtOffset<UnalignedValue>(dest, i));
    }
    return count;
  }
};

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
/// `NonTrivialDataTypeSerializer<Element>::Read`.
template <typename Element>
struct WriteNonTrivialLoopTemplate {
  using ElementwiseFunctionType = ElementwiseFunction<1, absl::Status*>;
  template <typename ArrayAccessor>
  static Index Loop(void* context, Index count, IterationBufferPointer source,
                    absl::Status* /*status*/) {
    auto& writer = *reinterpret_cast<riegeli::Writer*>(context);
    for (Index i = 0; i < count; ++i) {
      if (!NonTrivialDataTypeSerializer<Element>::Write(
              writer, *ArrayAccessor::template GetPointerAtOffset<Element>(
                          source, i))) {
        return i;
      }
    }
    return count;
  }
};

/// Reads an array of non-trivial elements from a `riegeli::Reader`, using
/// `NonTrivialDataTypeSerializer<Element>::Write`.
template <typename Element>
struct ReadNonTrivialLoopTemplate {
  using ElementwiseFunctionType = ElementwiseFunction<1, absl::Status*>;
  template <typename ArrayAccessor>
  static Index Loop(void* context, Index count, IterationBufferPointer source,
                    absl::Status* /*status*/) {
    auto& reader = *reinterpret_cast<riegeli::Reader*>(context);
    for (Index i = 0; i < count; ++i) {
      if (!NonTrivialDataTypeSerializer<Element>::Read(
              reader, *ArrayAccessor::template GetPointerAtOffset<Element>(
                          source, i))) {
        return i;
      }
    }
    return count;
  }
};

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

  using ElementwiseFunctionType = ElementwiseFunction<1, absl::Status*>;
  template <typename ArrayAccessor>
  static Index Loop(void* context, Index count, IterationBufferPointer source,
                    absl::Status* /*status*/) {
    auto& writer = *reinterpret_cast<riegeli::Writer*>(context);
    if constexpr (SubElementSize == 1 &&
                  ArrayAccessor::buffer_kind ==
                      internal::IterationBufferKind::kContiguous) {
      // Fast path: source array is contiguous and byte swapping is not
      // required.
      if (!writer.Write(std::string_view(
              reinterpret_cast<const char*>(source.pointer.get()),
              count * sizeof(Element)))) {
        return 0;
      }
    } else {
      Index element_i = 0;
      while (element_i < count) {
        const size_t remaining_bytes = (count - element_i) * sizeof(Element);
        if (!writer.Push(/*min_length=*/sizeof(Element),
                         /*recommended_length=*/remaining_bytes)) {
          return element_i;
        }
        const Index end_element_i = std::min(
            count, static_cast<Index>(element_i +
                                      (writer.available() / sizeof(Element))));
        char* cursor = writer.cursor();
        for (; element_i < end_element_i; ++element_i) {
          SwapEndianUnaligned<SubElementSize, NumSubElements>(
              ArrayAccessor::template GetPointerAtOffset<Element>(source,
                                                                  element_i),
              cursor);
          cursor += sizeof(Element);
        }
        element_i = end_element_i;
        writer.set_cursor(cursor);
      }
    }
    return count;
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

  using ElementwiseFunctionType = ElementwiseFunction<1, absl::Status*>;
  template <typename ArrayAccessor>
  static Index Loop(void* context, Index count, IterationBufferPointer source,
                    absl::Status* /*status*/) {
    auto& reader = *reinterpret_cast<riegeli::Reader*>(context);
    if constexpr (SubElementSize == 1 &&
                  ArrayAccessor::buffer_kind ==
                      internal::IterationBufferKind::kContiguous &&
                  !IsBool) {
      // Fast path: destination array is contiguous and byte swapping is not
      // required.
      if (!reader.Read(count * sizeof(Element),
                       reinterpret_cast<char*>(source.pointer.get()))) {
        return 0;
      }
    } else {
      Index element_i = 0;
      while (element_i < count) {
        const size_t remaining_bytes = (count - element_i) * sizeof(Element);
        if (!reader.Pull(/*min_length=*/sizeof(Element),
                         /*recommended_length=*/remaining_bytes)) {
          return element_i;
        }
        const Index end_element_i = std::min(
            count, static_cast<Index>(element_i +
                                      (reader.available() / sizeof(Element))));
        const char* cursor = reader.cursor();
        for (; element_i < end_element_i; ++element_i) {
          if constexpr (IsBool) {
            // Ensure that the result is exactly 0 or 1.
            *ArrayAccessor::template GetPointerAtOffset<bool>(
                source, element_i) = static_cast<bool>(*cursor);
          } else {
            SwapEndianUnaligned<SubElementSize, NumSubElements>(
                cursor, ArrayAccessor::template GetPointerAtOffset<Element>(
                            source, element_i));
          }
          cursor += sizeof(Element);
        }
        element_i = end_element_i;
        reader.set_cursor(cursor);
      }
    }
    return count;
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ENDIAN_ELEMENTWISE_CONVERSION_H_
