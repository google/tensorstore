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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_NEUROGLANCER_COMPRESSED_SEGMENTATION_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_NEUROGLANCER_COMPRESSED_SEGMENTATION_H_

/// \file
/// Implements encoding into the compressed segmentation format described at
/// https://github.com/google/neuroglancer/tree/master/src/neuroglancer/sliceview/compressed_segmentation.
///
/// Only uint32 and uint64 data types are supported.

/// Compresses a 3-D label array by splitting in a grid of fixed-size blocks,
/// and encoding each block using a per-block table of label values.  The number
/// of bits used to encode the value within each block depends on the size of
/// the table, i.e. the number of distinct uint64 values within that block.  The
/// number of BITS is required to be either 0, or a power of 2, i.e. 0, 1, 2, 4,
/// 8, 16.
///
/// The format consists of a block index containing a block header for each
/// block, followed by the encoded block values, followed by the table that maps
/// encoded indices to uint32 or uint64 label values.  Blocks are numbered as:
///   x + grid_size.x() * (y + grid_size.y() * z).
///
/// Overall file format:
///
///   [block header] * <number of blocks>
///   [encoded values]
///   [value table]
///
/// The format of each block header is:
///
///   table_base_offset : 24-bit LE integer
///   encoding_bits : 8-bit unsigned integer
///
///   encoded_value_base_offset : 24-bit LE integer
///   padding : 8 bits
///
///
/// The encoded_value_base_offset specifies the offset in 32-bit units from the
/// start of the file to the first 32-bit unit containing encoded values for the
/// block.
///
/// The table_base_offset specifies the offset in 32-bit units from the start of
/// the file to the first table entry for the block.
///
/// If multiple blocks have exactly the same set of encoded values, the same
/// value table will be shared by both blocks.

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace tensorstore {
namespace neuroglancer_compressed_segmentation {

template <class Label>
using EncodedValueCache = absl::flat_hash_map<std::vector<Label>, uint32_t>;

/// Encodes a single block.
///
/// \tparam Label Must be `std::uint32_t` or `std::uint64_t`.
/// \param input Pointer to input array.  If `input_shape` is smaller than
///     `block_shape`, the input block is effectively padded up to
///     `block_shape` with the lowest numerical label value contained within
///     it.
/// \param input_shape Dimensions of input array.
/// \param input_byte_strides Stride in bytes between consecutive elements
/// along
///     each dimension of the input array.
/// \param base_offset Starting byte offset into `*output` relative to which
///     table offsets will be specified.
/// \param encoded_bits_output[out] Will be set to the number of bits used to
///     encode each value.  Must be non-null.
/// \param table_offset_output[out] Will be set to the offset of either the
///     existing or newly written value table used for this block.  Must be
///     non-null.
/// \param cache Cache of existing tables written and their corresponding
///     offsets.  Must be non-null.
/// \param output[out] String to which encoded output will be appended.
/// \pre `input_shape[i] >= 0 && input_shape[i] <= block_shape[i]`
template <typename Label>
void EncodeBlock(const Label* input, const std::ptrdiff_t input_shape[3],
                 const std::ptrdiff_t input_byte_strides[3],
                 const std::ptrdiff_t block_shape[3], size_t base_offset,
                 size_t* encoded_bits_output, size_t* table_offset_output,
                 EncodedValueCache<Label>* cache, std::string* output);

/// Encodes a single channel.
///
/// \tparam Label Must be `std::uint32_t` or `std::uint64_t`.
/// \param input Pointer to input array.
/// \param input_shape Dimensions of input array.
/// \param input_byte_strides Stride in bytes between consecutive elements
/// along
///     each dimension of the input array.
/// \param block_shape Block shape to use for encoding.
/// \param output[out] String to which encoded output will be appended.
template <typename Label>
void EncodeChannel(const Label* input, const std::ptrdiff_t input_shape[3],
                   const std::ptrdiff_t input_byte_strides[3],
                   const std::ptrdiff_t block_shape[3], std::string* output);

/// Encodes multiple channels.
///
/// Each channel is encoded independently.
///
/// The output starts with `num_channels = input_shape[0]` uint32le values
/// specifying the starting offset of the encoding of each channel in units of
/// 4 bytes (the first offset will always equal `num_channels`).
///
/// \tparam Label Must be `std::uint32_t` or `std::uint64_t`.
/// \param input Pointer to the input array.
/// \param input_shape Dimensions of input array.  The first dimension
///     corresponds to the channel.
/// \param input_byte_strides Stride in bytes between consecutive elements
/// along
///     each dimension of the input array.
/// \param block_shape Block shape to use for encoding.
/// \param output[out] String to which encoded output will be appended.
template <typename Label>
void EncodeChannels(const Label* input, const std::ptrdiff_t input_shape[3 + 1],
                    const std::ptrdiff_t input_byte_strides[3 + 1],
                    const std::ptrdiff_t block_shape[3], std::string* output);

/// Decodes a single block.
///
/// \tparam Label Must be `std::uint32_t` or `std::uint64_t`.
/// \param encoded_bits Number of bits used to encode each label (i.e. to
/// encode
///     index into the table of labels).
/// \param encoded_input Pointer to encoded block values.
/// \param table_input Pointer to table of labels.
/// \param table_size Number of labels in table, used for bounds checking.
/// \param block_shape Block shape used for encoding.
/// \param output_shape Shape of output array.
/// \param output_byte_strides Byte strides for each dimension of the output
///     array.
/// \param output[out] Pointer to output array.
/// \pre `output_shape[i] >= 0 && output_shape[i] <= block_shape[i]` for
///     `0 <= i < 3`.
/// \returns `true` on success, or `false` if the input is corrupt.
template <typename Label>
bool DecodeBlock(size_t encoded_bits, const char* encoded_input,
                 const char* table_input, size_t table_size,
                 const std::ptrdiff_t block_shape[3],
                 const std::ptrdiff_t output_shape[3],
                 const std::ptrdiff_t output_byte_strides[3], Label* output);

/// Decodes a single channel.
///
/// \tparam Label Must be `std::uint32_t` or `std::uint64_t`.
/// \param input Encoded input data.
/// \param block_shape Block shape used for encoding.
/// \param output_shape Shape of output array.
/// \param output_byte_strides Byte strides for each dimension of the output
///     array.
/// \param output[out] Pointer to output array.
/// \returns `true` on success, or `false` if the input is corrupt.
template <typename Label>
bool DecodeChannel(absl::string_view input, const std::ptrdiff_t block_shape[3],
                   const std::ptrdiff_t output_shape[3],
                   const std::ptrdiff_t output_byte_strides[3], Label* output);

/// Decodes multiple channel.
///
/// \tparam Label Must be `std::uint32_t` or `std::uint64_t`.
/// \param input Encoded input data.
/// \param block_shape Block shape used for encoding.
/// \param output_shape Shape of output array.  The first dimension
/// corresponds
///     to the channel.
/// \param output_byte_strides Byte strides for each dimension of the output
///     array.
/// \param output[out] Pointer to output array.
/// \returns `true` on success, or `false` if the input is corrupt.
template <typename Label>
bool DecodeChannels(absl::string_view input,
                    const std::ptrdiff_t block_shape[3],
                    const std::ptrdiff_t output_shape[3 + 1],
                    const std::ptrdiff_t output_byte_strides[3 + 1],
                    Label* output);

}  // namespace neuroglancer_compressed_segmentation
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_NEUROGLANCER_COMPRESSED_SEGMENTATION_H_
