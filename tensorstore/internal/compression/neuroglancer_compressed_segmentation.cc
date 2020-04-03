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

#include "tensorstore/internal/compression/neuroglancer_compressed_segmentation.h"

#include <cstddef>

#include "absl/base/internal/endian.h"

namespace tensorstore {
namespace neuroglancer_compressed_segmentation {

constexpr size_t kBlockHeaderSize = 2;

void WriteBlockHeader(size_t encoded_value_base_offset,
                      size_t table_base_offset, size_t encoding_bits,
                      void* output) {
  absl::little_endian::Store32(output,
                               table_base_offset | (encoding_bits << 24));
  absl::little_endian::Store32(static_cast<char*>(output) + 4,
                               encoded_value_base_offset);
}

template <typename Label>
void EncodeBlock(const Label* input, const std::ptrdiff_t input_shape[3],
                 const std::ptrdiff_t input_byte_strides[3],
                 const std::ptrdiff_t block_shape[3], size_t base_offset,
                 size_t* encoded_bits_output, size_t* table_offset_output,
                 EncodedValueCache<Label>* cache, std::string* output) {
  if (input_shape[0] == 0 && input_shape[1] == 0 && input_shape[2] == 0) {
    *encoded_bits_output = 0;
    *table_offset_output = 0;
    return;
  }

  constexpr size_t num_32bit_words_per_label = sizeof(Label) / 4;

  // TODO(jbms): allow the memory allocated for these to be shared across
  // multiple calls to EncodeBlock.
  absl::flat_hash_map<Label, std::uint32_t> seen_values;
  std::vector<Label> seen_values_inv;

  // Calls `func(z, y, x, label)` for each `label` value within the block, in C
  // order.
  const auto ForEachElement = [&](auto func) {
    auto* input_z = reinterpret_cast<const char*>(input);
    for (std::ptrdiff_t z = 0; z < input_shape[0]; ++z) {
      auto* input_y = input_z;
      for (std::ptrdiff_t y = 0; y < input_shape[1]; ++y) {
        auto* input_x = input_y;
        for (std::ptrdiff_t x = 0; x < input_shape[2]; ++x) {
          func(z, y, x, *reinterpret_cast<const Label*>(input_x));
          input_x += input_byte_strides[2];
        }
        input_y += input_byte_strides[1];
      }
      input_z += input_byte_strides[0];
    }
  };

  // First determine the distinct values.

  // Initialize previous_value such that it is guaranteed not to equal to the
  // first value.
  Label previous_value = input[0] + 1;
  ForEachElement([&](size_t z, size_t y, size_t x, Label value) {
    // If this value matches the previous value, we can skip the more
    // expensive hash table lookup.
    if (value != previous_value) {
      previous_value = value;
      if (seen_values.emplace(value, 0).second) {
        seen_values_inv.push_back(value);
      }
    }
  });

  std::sort(seen_values_inv.begin(), seen_values_inv.end());
  for (size_t i = 0; i < seen_values_inv.size(); ++i) {
    seen_values[seen_values_inv[i]] = static_cast<uint32_t>(i);
  }

  // Determine number of bits with which to encode each index.
  size_t encoded_bits = 0;
  if (seen_values.size() != 1) {
    encoded_bits = 1;
    while ((size_t(1) << encoded_bits) < seen_values.size()) {
      encoded_bits *= 2;
    }
  }
  *encoded_bits_output = encoded_bits;
  const size_t encoded_size_32bits =
      (encoded_bits * block_shape[0] * block_shape[1] * block_shape[2] + 31) /
      32;

  const size_t encoded_value_base_offset = output->size();
  assert((encoded_value_base_offset - base_offset) % 4 == 0);
  size_t elements_to_write = encoded_size_32bits;

  bool write_table;
  {
    auto it = cache->find(seen_values_inv);
    if (it == cache->end()) {
      write_table = true;
      elements_to_write += seen_values.size() * num_32bit_words_per_label;
      *table_offset_output =
          (encoded_value_base_offset - base_offset) / 4 + encoded_size_32bits;
    } else {
      write_table = false;
      *table_offset_output = it->second;
    }
  }

  output->resize(encoded_value_base_offset + elements_to_write * 4);
  char* output_ptr = output->data() + encoded_value_base_offset;
  // Write encoded representation.
  ForEachElement([&](size_t z, size_t y, size_t x, Label value) {
    uint32_t index = seen_values.at(value);
    size_t output_offset = x + block_shape[2] * (y + block_shape[1] * z);
    void* cur_ptr = output_ptr + output_offset * encoded_bits / 32 * 4;
    absl::little_endian::Store32(
        cur_ptr, absl::little_endian::Load32(cur_ptr) |
                     (index << (output_offset * encoded_bits % 32)));
  });

  // Write table
  if (write_table) {
    output_ptr =
        output->data() + encoded_value_base_offset + encoded_size_32bits * 4;
    for (auto value : seen_values_inv) {
      for (size_t word_i = 0; word_i < num_32bit_words_per_label; ++word_i) {
        absl::little_endian::Store32(
            output_ptr + word_i * 4,
            static_cast<uint32_t>(value >> (32 * word_i)));
      }
      output_ptr += num_32bit_words_per_label * 4;
    }
    cache->emplace(seen_values_inv,
                   static_cast<std::uint32_t>(*table_offset_output));
  }
}

template <class Label>
void EncodeChannel(const Label* input, const std::ptrdiff_t input_shape[3],
                   const std::ptrdiff_t input_byte_strides[3],
                   const std::ptrdiff_t block_shape[3], std::string* output) {
  EncodedValueCache<Label> cache;
  const size_t base_offset = output->size();
  ptrdiff_t grid_shape[3];
  size_t block_index_size = kBlockHeaderSize;
  for (size_t i = 0; i < 3; ++i) {
    grid_shape[i] = (input_shape[i] + block_shape[i] - 1) / block_shape[i];
    block_index_size *= grid_shape[i];
  }
  output->resize(base_offset + block_index_size * 4);
  ptrdiff_t block[3];
  for (block[0] = 0; block[0] < grid_shape[0]; ++block[0]) {
    for (block[1] = 0; block[1] < grid_shape[1]; ++block[1]) {
      for (block[2] = 0; block[2] < grid_shape[2]; ++block[2]) {
        const size_t block_offset =
            block[2] + grid_shape[2] * (block[1] + grid_shape[1] * block[0]);
        ptrdiff_t input_block_shape[3];
        ptrdiff_t input_offset = 0;
        for (size_t i = 0; i < 3; ++i) {
          auto pos = block[i] * block_shape[i];
          input_block_shape[i] = std::min(block_shape[i], input_shape[i] - pos);
          input_offset += pos * input_byte_strides[i];
        }
        const size_t encoded_value_base_offset =
            (output->size() - base_offset) / 4;
        size_t encoded_bits, table_offset;
        EncodeBlock(reinterpret_cast<const Label*>(
                        reinterpret_cast<const char*>(input) + input_offset),
                    input_block_shape, input_byte_strides, block_shape,
                    base_offset, &encoded_bits, &table_offset, &cache, output);
        WriteBlockHeader(
            encoded_value_base_offset, table_offset, encoded_bits,
            output->data() + base_offset + block_offset * kBlockHeaderSize * 4);
      }
    }
  }
}

template <class Label>
void EncodeChannels(const Label* input, const std::ptrdiff_t input_shape[3 + 1],
                    const std::ptrdiff_t input_byte_strides[3 + 1],
                    const std::ptrdiff_t block_shape[3], std::string* output) {
  const size_t base_offset = output->size();
  output->resize(base_offset + input_shape[0] * 4);
  for (std::ptrdiff_t channel_i = 0; channel_i < input_shape[0]; ++channel_i) {
    absl::little_endian::Store32(output->data() + base_offset + channel_i * 4,
                                 (output->size() - base_offset) / 4);
    EncodeChannel(
        reinterpret_cast<const Label*>(reinterpret_cast<const char*>(input) +
                                       input_byte_strides[0] * channel_i),
        input_shape + 1, input_byte_strides + 1, block_shape, output);
  }
}

void ReadBlockHeader(const void* header, size_t* encoded_value_base_offset,
                     size_t* table_base_offset, size_t* encoding_bits) {
  auto h = absl::little_endian::Load64(header);
  *table_base_offset = h & 0xffffff;
  *encoding_bits = (h >> 24) & 0xff;
  *encoded_value_base_offset = (h >> 32) & 0xffffff;
}

template <typename Label>
bool DecodeBlock(size_t encoded_bits, const char* encoded_input,
                 const char* table_input, size_t table_size,
                 const std::ptrdiff_t block_shape[3],
                 const std::ptrdiff_t output_shape[3],
                 const std::ptrdiff_t output_byte_strides[3], Label* output) {
  // TODO(jbms): Consider specializing this function for the value of
  // `encoded_bits` and whether `table_size` is `< 2**encoded_bits`.  If
  // `table_size >= 2**encoded_bits`, there is no need to check below that
  // `index <= table_size`.
  const std::uint32_t encoded_value_mask =
      (std::uint32_t(1) << encoded_bits) - 1;
  auto* output_z = reinterpret_cast<char*>(output);
  for (std::ptrdiff_t z = 0; z < output_shape[0]; ++z) {
    auto* output_y = output_z;
    for (std::ptrdiff_t y = 0; y < output_shape[1]; ++y) {
      auto* output_x = output_y;
      for (std::ptrdiff_t x = 0; x < output_shape[2]; ++x) {
        size_t encoded_offset = x + block_shape[2] * (y + block_shape[1] * z);
        auto index =
            absl::little_endian::Load32(
                encoded_input + encoded_offset * encoded_bits / 32 * 4) >>
                (encoded_offset * encoded_bits % 32) &
            encoded_value_mask;
        if (index >= table_size) return false;
        auto& label = *reinterpret_cast<Label*>(output_x);
        if constexpr (sizeof(Label) == 4) {
          label =
              absl::little_endian::Load32(table_input + index * sizeof(Label));
        } else {
          label =
              absl::little_endian::Load64(table_input + index * sizeof(Label));
        }
        output_x += output_byte_strides[2];
      }
      output_y += output_byte_strides[1];
    }
    output_z += output_byte_strides[0];
  }
  return true;
}

template <typename Label>
bool DecodeChannel(absl::string_view input, const std::ptrdiff_t block_shape[3],
                   const std::ptrdiff_t output_shape[3],
                   const std::ptrdiff_t output_byte_strides[3], Label* output) {
  if ((input.size() % 4) != 0) return false;
  ptrdiff_t grid_shape[3];
  size_t block_index_size = kBlockHeaderSize;
  for (size_t i = 0; i < 3; ++i) {
    grid_shape[i] = (output_shape[i] + block_shape[i] - 1) / block_shape[i];
    block_index_size *= grid_shape[i];
  }
  if (input.size() / 4 < block_index_size) {
    // `input` is too short to contain block headers
    return false;
  }
  ptrdiff_t block[3];
  for (block[0] = 0; block[0] < grid_shape[0]; ++block[0]) {
    for (block[1] = 0; block[1] < grid_shape[1]; ++block[1]) {
      for (block[2] = 0; block[2] < grid_shape[2]; ++block[2]) {
        const size_t block_offset =
            block[2] + grid_shape[2] * (block[1] + grid_shape[1] * block[0]);
        ptrdiff_t output_block_shape[3];
        ptrdiff_t output_offset = 0;
        for (size_t i = 0; i < 3; ++i) {
          auto pos = block[i] * block_shape[i];
          output_block_shape[i] =
              std::min(block_shape[i], output_shape[i] - pos);
          output_offset += pos * output_byte_strides[i];
        }
        size_t encoded_value_base_offset;
        size_t encoded_bits, table_offset;
        ReadBlockHeader(input.data() + block_offset * kBlockHeaderSize * 4,
                        &encoded_value_base_offset, &table_offset,
                        &encoded_bits);
        if (encoded_bits > 32 || (encoded_bits & (encoded_bits - 1)) != 0) {
          // encoded bits is not a power of 2 <= 32.
          return false;
        }
        if (encoded_value_base_offset > input.size() / 4 ||
            table_offset > input.size() / 4) {
          return false;
        }
        const size_t encoded_size_32bits =
            (encoded_bits * block_shape[0] * block_shape[1] * block_shape[2] +
             31) /
            32;
        if ((encoded_value_base_offset + encoded_size_32bits) * 4 >
            input.size()) {
          return false;
        }
        auto* block_output = reinterpret_cast<Label*>(
            reinterpret_cast<char*>(output) + output_offset);
        const char* encoded_input =
            input.data() + encoded_value_base_offset * 4;
        const char* table_input = input.data() + table_offset * 4;
        const size_t table_size =
            (input.size() - table_offset * 4) / sizeof(Label);
        if (!DecodeBlock(encoded_bits, encoded_input, table_input, table_size,
                         block_shape, output_block_shape, output_byte_strides,
                         block_output)) {
          return false;
        }
      }
    }
  }
  return true;
}

template <typename Label>
bool DecodeChannels(absl::string_view input,
                    const std::ptrdiff_t block_shape[3],
                    const std::ptrdiff_t output_shape[3 + 1],
                    const std::ptrdiff_t output_byte_strides[3 + 1],
                    Label* output) {
  if ((input.size() % 4) != 0) return false;
  if (input.size() / 4 < static_cast<std::size_t>(output_shape[0])) {
    // `input` is too short to contain channel offsets
    return false;
  }
  for (std::ptrdiff_t channel_i = 0; channel_i < output_shape[0]; ++channel_i) {
    const size_t offset =
        absl::little_endian::Load32(input.data() + channel_i * 4);
    if (offset > input.size() / 4) {
      // channel offset is invalid
      return false;
    }
    if (!DecodeChannel(
            input.substr(offset * 4), block_shape, output_shape + 1,
            output_byte_strides + 1,
            reinterpret_cast<Label*>(reinterpret_cast<char*>(output) +
                                     output_byte_strides[0] * channel_i))) {
      // Error decoding channel
      return false;
    }
  }
  return true;
}

#define DO_INSTANTIATE(Label)                                                  \
  template void EncodeBlock<Label>(                                            \
      const Label* input, const std::ptrdiff_t input_shape[3],                 \
      const std::ptrdiff_t input_byte_strides[3],                              \
      const std::ptrdiff_t block_shape[3], size_t base_offset,                 \
      size_t* encoded_bits_output, size_t* table_offset_output,                \
      EncodedValueCache<Label>* cache, std::string* output);                   \
  template void EncodeChannel<Label>(                                          \
      const Label* input, const std::ptrdiff_t input_shape[3],                 \
      const std::ptrdiff_t input_byte_strides[3],                              \
      const std::ptrdiff_t block_shape[3], std::string* output);               \
  template void EncodeChannels<Label>(                                         \
      const Label* input, const std::ptrdiff_t input_shape[3 + 1],             \
      const std::ptrdiff_t input_byte_strides[3 + 1],                          \
      const std::ptrdiff_t block_shape[3], std::string* output);               \
  template bool DecodeBlock(                                                   \
      size_t encoded_bits, const char* encoded_input, const char* table_input, \
      size_t table_size, const std::ptrdiff_t block_shape[3],                  \
      const std::ptrdiff_t output_shape[3],                                    \
      const std::ptrdiff_t output_byte_strides[3], Label* output);             \
  template bool DecodeChannel<Label>(                                          \
      absl::string_view input, const std::ptrdiff_t block_shape[3],            \
      const std::ptrdiff_t output_shape[3],                                    \
      const std::ptrdiff_t output_byte_strides[3], Label* output);             \
  template bool DecodeChannels(                                                \
      absl::string_view input, const std::ptrdiff_t block_shape[3],            \
      const std::ptrdiff_t output_shape[3 + 1],                                \
      const std::ptrdiff_t output_byte_strides[3 + 1], Label* output);         \
  /**/

DO_INSTANTIATE(std::uint32_t)
DO_INSTANTIATE(std::uint64_t)

#undef DO_INSTANTIATE

}  // namespace neuroglancer_compressed_segmentation
}  // namespace tensorstore
