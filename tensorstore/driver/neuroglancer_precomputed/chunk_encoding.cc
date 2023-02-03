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
#include "absl/base/optimization.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "tensorstore/internal/compression/neuroglancer_compressed_segmentation.h"
#include "tensorstore/internal/data_type_endian_conversion.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/jpeg_reader.h"
#include "tensorstore/internal/image/jpeg_writer.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_neuroglancer_precomputed {

using ::tensorstore::internal_image::ImageInfo;
using ::tensorstore::internal_image::JpegReader;
using ::tensorstore::internal_image::JpegWriter;
using ::tensorstore::internal_image::JpegWriterOptions;

Result<SharedArrayView<const void>> DecodeRawChunk(
    DataType dtype, span<const Index, 4> shape,
    StridedLayoutView<4> chunk_layout, absl::Cord buffer) {
  const Index expected_bytes = ProductOfExtents(shape) * dtype.size();
  if (expected_bytes != static_cast<Index>(buffer.size())) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Expected chunk length to be ", expected_bytes,
                            ", but received ", buffer.size(), " bytes"));
  }
  auto flat_buffer = buffer.Flatten();
  if (absl::c_equal(shape, chunk_layout.shape())) {
    // Chunk is full size.  Attempt to decode in place.  Transfer ownership of
    // the existing `buffer` string into `decoded_array`.
    auto decoded_array = internal::TryViewCordAsArray(
        buffer, /*offset=*/0, dtype, endian::little, chunk_layout);
    if (decoded_array.valid()) return decoded_array;
  }
  // Partial chunk, must copy.  It is safe to default initialize because the
  // out-of-bounds positions will never be read, but we use value initialization
  // for simplicity in case resize is supported later.
  Array<const void, 4> source(
      {static_cast<const void*>(flat_buffer.data()), dtype}, shape);
  SharedArrayView<void> full_decoded_array(
      internal::AllocateAndConstructSharedElements(chunk_layout.num_elements(),
                                                   value_init, dtype),
      chunk_layout);
  ArrayView<void> partial_decoded_array(
      full_decoded_array.element_pointer(),
      StridedLayoutView<>{shape, chunk_layout.byte_strides()});
  internal::DecodeArray(source, endian::little, partial_decoded_array);
  return full_decoded_array;
}

Result<SharedArrayView<const void>> DecodeJpegChunk(
    DataType dtype, span<const Index, 4> partial_shape,
    StridedLayoutView<4> chunk_layout, absl::Cord encoded_input) {
  // `array` will contain decoded jpeg with C-order `(z, y, x, channel)` layout.
  //
  // If number of channels is 1, then this is equivalent to the
  // `(channel, z, y, x)` layout in `chunk_layout`.
  auto array = AllocateArray(
      {partial_shape[1], partial_shape[2], partial_shape[3], partial_shape[0]},
      c_order, default_init, dtype);

  {
    riegeli::CordReader<> cord_reader(&encoded_input);
    JpegReader reader;
    TENSORSTORE_RETURN_IF_ERROR(reader.Initialize(&cord_reader));
    auto info = reader.GetImageInfo();

    // Check constraints.
    const Index num_elements = ProductOfExtents(partial_shape.subspan<1>());
    size_t total_pixels;
    if (internal::MulOverflow(static_cast<size_t>(info.width),
                              static_cast<size_t>(info.height),
                              &total_pixels) ||
        num_elements == std::numeric_limits<Index>::max() ||
        static_cast<Index>(total_pixels) != num_elements ||
        static_cast<Index>(info.num_components) != partial_shape[0]) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Image dimensions (", info.width, ", ", info.height, ", ",
          info.num_components,
          ") are not compatible with expected chunk shape ", partial_shape));
    }

    TENSORSTORE_RETURN_IF_ERROR(reader.Decode(
        tensorstore::span(reinterpret_cast<unsigned char*>(array.data()),
                          ImageRequiredBytes(info))));
    if (!cord_reader.Close()) {
      return cord_reader.status();
    }
  }

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
                                                   default_init, dtype),
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
    DataType dtype, span<const Index, 4> shape,
    StridedLayoutView<4> chunk_layout, std::array<Index, 3> block_size,
    absl::Cord buffer) {
  auto flat_buffer = buffer.Flatten();
  SharedArrayView<void> full_decoded_array(
      internal::AllocateAndConstructSharedElements(chunk_layout.num_elements(),
                                                   default_init, dtype),
      chunk_layout);
  std::ptrdiff_t output_shape_ptrdiff_t[4] = {shape[0], shape[1], shape[2],
                                              shape[3]};
  std::ptrdiff_t block_shape_ptrdiff_t[3] = {block_size[2], block_size[1],
                                             block_size[0]};
  std::ptrdiff_t output_byte_strides[4] = {
      chunk_layout.byte_strides()[0], chunk_layout.byte_strides()[1],
      chunk_layout.byte_strides()[2], chunk_layout.byte_strides()[3]};
  bool success = false;
  switch (dtype.id()) {
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
      ABSL_UNREACHABLE();  // COV_NF_LINE
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
      return DecodeRawChunk(metadata.dtype, chunk_shape, chunk_layout,
                            std::move(buffer));
    case ScaleMetadata::Encoding::jpeg:
      return DecodeJpegChunk(metadata.dtype, chunk_shape, chunk_layout,
                             std::move(buffer));
    case ScaleMetadata::Encoding::compressed_segmentation:
      return DecodeCompressedSegmentationChunk(
          metadata.dtype, chunk_shape, chunk_layout,
          scale_metadata.compressed_segmentation_block_size, std::move(buffer));
  }
  ABSL_UNREACHABLE();  // COV_NF_LINE
}

absl::Cord EncodeRawChunk(DataType dtype, span<const Index, 4> shape,
                          const SharedArrayView<const void>& array) {
  ArrayView<const void> partial_source(
      array.element_pointer(),
      StridedLayoutView<>(shape, array.byte_strides()));
  internal::FlatCordBuilder buffer(ProductOfExtents(shape) * dtype.size());
  Array<void, 4> encoded_array({static_cast<void*>(buffer.data()), dtype},
                               shape);
  internal::EncodeArray(partial_source, encoded_array, endian::little);
  return std::move(buffer).Build();
}

Result<absl::Cord> EncodeJpegChunk(DataType dtype, int quality,
                                   span<const Index, 4> shape,
                                   ArrayView<const void> array) {
  Array<const void, 4> partial_source(
      array.element_pointer(),
      StridedLayout<4>({shape[1], shape[2], shape[3], shape[0]},
                       {array.byte_strides()[1], array.byte_strides()[2],
                        array.byte_strides()[3], array.byte_strides()[0]}));
  auto contiguous_array = MakeCopy(partial_source, c_order);

  absl::Cord buffer;
  {
    JpegWriterOptions options;
    options.quality = quality;
    JpegWriter writer;
    riegeli::CordWriter<> cord_writer(&buffer);
    TENSORSTORE_RETURN_IF_ERROR(writer.Initialize(&cord_writer, options));
    ImageInfo info{/*.height =*/static_cast<int32_t>(shape[3]),
                   /*.width =*/static_cast<int32_t>(shape[1] * shape[2]),
                   /*.num_components =*/static_cast<int32_t>(shape[0])};
    TENSORSTORE_RETURN_IF_ERROR(writer.Encode(
        info, tensorstore::span(reinterpret_cast<const unsigned char*>(
                                    contiguous_array.data()),
                                contiguous_array.num_elements() *
                                    contiguous_array.dtype().size())));
    TENSORSTORE_RETURN_IF_ERROR(writer.Done());
  }
  return buffer;
}

Result<absl::Cord> EncodeCompressedSegmentationChunk(
    DataType dtype, span<const Index, 4> shape, ArrayView<const void> array,
    std::array<Index, 3> block_size) {
  std::ptrdiff_t input_shape_ptrdiff_t[4] = {shape[0], shape[1], shape[2],
                                             shape[3]};
  std::ptrdiff_t block_shape_ptrdiff_t[3] = {block_size[2], block_size[1],
                                             block_size[0]};
  std::string out;
  std::ptrdiff_t input_byte_strides[4] = {
      array.byte_strides()[0], array.byte_strides()[1], array.byte_strides()[2],
      array.byte_strides()[3]};
  switch (dtype.id()) {
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
      ABSL_UNREACHABLE();  // COV_NF_LINE
  }
  return absl::Cord(std::move(out));
}

Result<absl::Cord> EncodeChunk(span<const Index> chunk_indices,
                               const MultiscaleMetadata& metadata,
                               std::size_t scale_index,
                               const SharedArrayView<const void>& array) {
  const auto& scale_metadata = metadata.scales[scale_index];
  std::array<Index, 4> partial_chunk_shape;
  GetChunkShape(chunk_indices, metadata, scale_index,
                span<const Index, 4>(array.shape().data(), 4),
                partial_chunk_shape);
  switch (scale_metadata.encoding) {
    case ScaleMetadata::Encoding::raw:
      return EncodeRawChunk(metadata.dtype, partial_chunk_shape, array);
    case ScaleMetadata::Encoding::jpeg:
      return EncodeJpegChunk(metadata.dtype, scale_metadata.jpeg_quality,
                             partial_chunk_shape, array);
    case ScaleMetadata::Encoding::compressed_segmentation:
      return EncodeCompressedSegmentationChunk(
          metadata.dtype, partial_chunk_shape, array,
          scale_metadata.compressed_segmentation_block_size);
  }
  ABSL_UNREACHABLE();  // COV_NF_LINE
}

}  // namespace internal_neuroglancer_precomputed
}  // namespace tensorstore
