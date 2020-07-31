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

#include "tensorstore/driver/neuroglancer_precomputed/chunk_encoding.h"

#include "absl/algorithm/container.h"
#include "tensorstore/internal/compression/jpeg.h"
#include "tensorstore/internal/compression/neuroglancer_compressed_segmentation.h"
#include "tensorstore/internal/container_to_shared.h"
#include "tensorstore/internal/data_type_endian_conversion.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/util/endian.h"

namespace tensorstore {
namespace internal_neuroglancer_precomputed {

Result<SharedArrayView<const void>> DecodeRawChunk(
    DataType data_type, span<const Index, 4> shape,
    StridedLayoutView<4> chunk_layout, absl::Cord buffer) {
  const Index expected_bytes = ProductOfExtents(shape) * data_type.size();
  if (expected_bytes != static_cast<Index>(buffer.size())) {
    return absl::InvalidArgumentError(StrCat("Expected chunk length to be ",
                                             expected_bytes, ", but received ",
                                             buffer.size(), " bytes"));
  }
  auto flat_buffer = buffer.Flatten();
  if (absl::c_equal(shape, chunk_layout.shape())) {
    // Chunk is full size.  Attempt to decode in place.  Transfer ownership of
    // the existing `buffer` string into `decoded_array`.
    auto decoded_array = internal::TryViewCordAsArray(
        buffer, /*offset=*/0, data_type, endian::little, chunk_layout);
    if (decoded_array.valid()) return decoded_array;
  }
  // Partial chunk, must copy.  It is safe to default initialize because the
  // out-of-bounds positions will never be read, but we use value initialization
  // for simplicity in case resize is supported later.
  Array<const void, 4> source(
      {static_cast<const void*>(flat_buffer.data()), data_type}, shape);
  SharedArrayView<void> full_decoded_array(
      internal::AllocateAndConstructSharedElements(chunk_layout.num_elements(),
                                                   value_init, data_type),
      chunk_layout);
  ArrayView<void> partial_decoded_array(
      full_decoded_array.element_pointer(),
      StridedLayoutView<>{shape, chunk_layout.byte_strides()});
  internal::DecodeArray(source, endian::little, partial_decoded_array);
  return full_decoded_array;
}

Result<SharedArrayView<const void>> DecodeJpegChunk(
    DataType data_type, span<const Index, 4> partial_shape,
    StridedLayoutView<4> chunk_layout, absl::Cord encoded_input) {
  // `array` will contain decoded jpeg with C-order `(z, y, x, channel)` layout.
  //
  // If number of channels is 1, then this is equivalent to the
  // `(channel, z, y, x)` layout in `chunk_layout`.
  auto array = AllocateArray(
      {partial_shape[1], partial_shape[2], partial_shape[3], partial_shape[0]},
      c_order, default_init, data_type);
  TENSORSTORE_RETURN_IF_ERROR(jpeg::Decode(
      encoded_input,
      [&](size_t width, size_t height,
          size_t num_components) -> Result<unsigned char*> {
        size_t total_pixels;
        const Index num_elements = ProductOfExtents(partial_shape.subspan<1>());
        if (internal::MulOverflow(width, height, &total_pixels) ||
            num_elements == std::numeric_limits<Index>::max() ||
            static_cast<Index>(total_pixels) != num_elements ||
            static_cast<Index>(num_components) != partial_shape[0]) {
          return absl::InvalidArgumentError(StrCat(
              "Image dimensions (", width, ", ", height, ", ", num_components,
              ") are not compatible with expected chunk shape ",
              partial_shape));
        }
        return reinterpret_cast<unsigned char*>(array.data());
      }));
  if (partial_shape[0] == 1 &&
      absl::c_equal(partial_shape, chunk_layout.shape())) {
    // `array` already has correct layout.
    return SharedArrayView<const void>(array.element_pointer(), chunk_layout);
  }
  // Partial chunk, or number of channels is not 1.  Must copy to obtain the
  // expected `chunk_layout`.
  //
  // It is safe to value initialize because the out-of-bounds positions will
  // never be read.  If resize is supported, this must change, however.
  SharedArrayView<void> full_decoded_array(
      internal::AllocateAndConstructSharedElements(chunk_layout.num_elements(),
                                                   default_init, data_type),
      chunk_layout);
  Array<void, 4> partial_decoded_array(
      full_decoded_array.element_pointer(),
      StridedLayout<4>(
          {partial_shape[1], partial_shape[2], partial_shape[3],
           partial_shape[0]},
          {chunk_layout.byte_strides()[1], chunk_layout.byte_strides()[2],
           chunk_layout.byte_strides()[3], chunk_layout.byte_strides()[0]}));
  CopyArray(array, partial_decoded_array);
  return full_decoded_array;
}

Result<SharedArrayView<const void>> DecodeCompressedSegmentationChunk(
    DataType data_type, span<const Index, 4> shape,
    StridedLayoutView<4> chunk_layout, std::array<Index, 3> block_size,
    absl::Cord buffer) {
  auto flat_buffer = buffer.Flatten();
  SharedArrayView<void> full_decoded_array(
      internal::AllocateAndConstructSharedElements(chunk_layout.num_elements(),
                                                   default_init, data_type),
      chunk_layout);
  std::ptrdiff_t output_shape_ptrdiff_t[4] = {shape[0], shape[1], shape[2],
                                              shape[3]};
  std::ptrdiff_t block_shape_ptrdiff_t[3] = {block_size[2], block_size[1],
                                             block_size[0]};
  std::ptrdiff_t output_byte_strides[4] = {
      chunk_layout.byte_strides()[0], chunk_layout.byte_strides()[1],
      chunk_layout.byte_strides()[2], chunk_layout.byte_strides()[3]};
  bool success = false;
  switch (data_type.id()) {
    case DataTypeId::uint32_t:
      success = neuroglancer_compressed_segmentation::DecodeChannels(
          flat_buffer, block_shape_ptrdiff_t, output_shape_ptrdiff_t,
          output_byte_strides,
          static_cast<std::uint32_t*>(full_decoded_array.data()));
      break;
    case DataTypeId::uint64_t:
      success = neuroglancer_compressed_segmentation::DecodeChannels(
          flat_buffer, block_shape_ptrdiff_t, output_shape_ptrdiff_t,
          output_byte_strides,
          static_cast<std::uint64_t*>(full_decoded_array.data()));
      break;
    default:
      TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
  }
  if (!success) {
    return absl::InvalidArgumentError(
        "Corrupted Neuroglancer compressed segmentation");
  }
  return full_decoded_array;
}

/// Computes the partial chunk shape (the size of the intersection of the full
/// chunk bounds with the volume dimensions).
///
/// \param chunk_indices The chunk grid cell indices in `xyz` order.
/// \param metadata The volume metadata.
/// \param scale_index The scale index within `metadata`.
/// \param full_chunk_shape The full chunk shape in `cxyx` order.
/// \param partial_chunk_shape[out] The partial chunk shape in `cxyx` order.
void GetChunkShape(span<const Index> chunk_indices,
                   const MultiscaleMetadata& metadata, std::size_t scale_index,
                   span<const Index, 4> full_chunk_shape,
                   span<Index, 4> partial_chunk_shape) {
  const auto& scale = metadata.scales[scale_index];
  partial_chunk_shape[0] = full_chunk_shape[0];
  for (int i = 0; i < 3; ++i) {
    const Index full_size = full_chunk_shape[3 - i];
    partial_chunk_shape[3 - i] = std::min(
        scale.box.shape()[i] - chunk_indices[i] * full_size, full_size);
  }
}

Result<SharedArrayView<const void>> DecodeChunk(
    span<const Index> chunk_indices, const MultiscaleMetadata& metadata,
    std::size_t scale_index, StridedLayoutView<4> chunk_layout,
    absl::Cord buffer) {
  const auto& scale_metadata = metadata.scales[scale_index];
  std::array<Index, 4> chunk_shape;
  GetChunkShape(chunk_indices, metadata, scale_index, chunk_layout.shape(),
                chunk_shape);
  switch (scale_metadata.encoding) {
    case ScaleMetadata::Encoding::raw:
      return DecodeRawChunk(metadata.data_type, chunk_shape, chunk_layout,
                            std::move(buffer));
    case ScaleMetadata::Encoding::jpeg:
      return DecodeJpegChunk(metadata.data_type, chunk_shape, chunk_layout,
                             std::move(buffer));
    case ScaleMetadata::Encoding::compressed_segmentation:
      return DecodeCompressedSegmentationChunk(
          metadata.data_type, chunk_shape, chunk_layout,
          scale_metadata.compressed_segmentation_block_size, std::move(buffer));
  }
  TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
}

absl::Cord EncodeRawChunk(DataType data_type, span<const Index, 4> shape,
                          ArrayView<const void> array) {
  ArrayView<const void> partial_source(
      array.element_pointer(),
      StridedLayoutView<>(shape, array.byte_strides()));
  internal::FlatCordBuilder buffer(ProductOfExtents(shape) * data_type.size());
  Array<void, 4> encoded_array({static_cast<void*>(buffer.data()), data_type},
                               shape);
  internal::EncodeArray(partial_source, encoded_array, endian::little);
  return std::move(buffer).Build();
}

Result<absl::Cord> EncodeJpegChunk(DataType data_type, int quality,
                                   span<const Index, 4> shape,
                                   ArrayView<const void> array) {
  Array<const void, 4> partial_source(
      array.element_pointer(),
      StridedLayout<4>({shape[1], shape[2], shape[3], shape[0]},
                       {array.byte_strides()[1], array.byte_strides()[2],
                        array.byte_strides()[3], array.byte_strides()[0]}));
  auto contiguous_array = MakeCopy(partial_source, c_order);
  jpeg::EncodeOptions options;
  options.quality = quality;
  absl::Cord buffer;
  TENSORSTORE_RETURN_IF_ERROR(jpeg::Encode(
      reinterpret_cast<const unsigned char*>(contiguous_array.data()),
      shape[1] * shape[2], shape[3], shape[0], options, &buffer));
  return buffer;
}

Result<absl::Cord> EncodeCompressedSegmentationChunk(
    DataType data_type, span<const Index, 4> shape, ArrayView<const void> array,
    std::array<Index, 3> block_size) {
  std::ptrdiff_t input_shape_ptrdiff_t[4] = {shape[0], shape[1], shape[2],
                                             shape[3]};
  std::ptrdiff_t block_shape_ptrdiff_t[3] = {block_size[2], block_size[1],
                                             block_size[0]};
  std::string out;
  std::ptrdiff_t input_byte_strides[4] = {
      array.byte_strides()[0], array.byte_strides()[1], array.byte_strides()[2],
      array.byte_strides()[3]};
  switch (data_type.id()) {
    case DataTypeId::uint32_t:
      neuroglancer_compressed_segmentation::EncodeChannels(
          static_cast<const std::uint32_t*>(array.data()),
          input_shape_ptrdiff_t, input_byte_strides, block_shape_ptrdiff_t,
          &out);
      break;
    case DataTypeId::uint64_t:
      neuroglancer_compressed_segmentation::EncodeChannels(
          static_cast<const std::uint64_t*>(array.data()),
          input_shape_ptrdiff_t, input_byte_strides, block_shape_ptrdiff_t,
          &out);
      break;
    default:
      TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
  }
  return absl::Cord(std::move(out));
}

Result<absl::Cord> EncodeChunk(span<const Index> chunk_indices,
                               const MultiscaleMetadata& metadata,
                               std::size_t scale_index,
                               ArrayView<const void> array) {
  const auto& scale_metadata = metadata.scales[scale_index];
  std::array<Index, 4> partial_chunk_shape;
  GetChunkShape(chunk_indices, metadata, scale_index,
                span<const Index, 4>(array.shape().data(), 4),
                partial_chunk_shape);
  switch (scale_metadata.encoding) {
    case ScaleMetadata::Encoding::raw:
      return EncodeRawChunk(metadata.data_type, partial_chunk_shape, array);
    case ScaleMetadata::Encoding::jpeg:
      return EncodeJpegChunk(metadata.data_type, scale_metadata.jpeg_quality,
                             partial_chunk_shape, array);
    case ScaleMetadata::Encoding::compressed_segmentation:
      return EncodeCompressedSegmentationChunk(
          metadata.data_type, partial_chunk_shape, array,
          scale_metadata.compressed_segmentation_block_size);
  }
  TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
}

}  // namespace internal_neuroglancer_precomputed
}  // namespace tensorstore
