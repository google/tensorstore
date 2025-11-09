// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/driver/zarr3/codec/codec.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_zarr3 {

ZarrArrayToArrayCodec::~ZarrArrayToArrayCodec() = default;
ZarrArrayToBytesCodec::~ZarrArrayToBytesCodec() = default;
ZarrBytesToBytesCodec::~ZarrBytesToBytesCodec() = default;

ZarrArrayToArrayCodec::PreparedState::~PreparedState() = default;
ZarrArrayToBytesCodec::PreparedState::~PreparedState() = default;
ZarrBytesToBytesCodec::PreparedState::~PreparedState() = default;

int64_t ZarrArrayToBytesCodec::PreparedState::encoded_size() const {
  return -1;
}

int64_t ZarrBytesToBytesCodec::PreparedState::encoded_size() const {
  return -1;
}

bool ZarrShardingCodec::is_sharding_codec() const { return true; }

absl::Status ZarrCodecChain::PreparedState::EncodeArray(
    SharedArrayView<const void> decoded, riegeli::Writer& writer) const {
  StridedLayout<> encoded_layout_storage;
  // Compute the transformed array.
  for (const auto& codec : array_to_array) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto encoded,
                                 codec->EncodeArray(std::move(decoded)));
    encoded_layout_storage = std::move(encoded.layout());
    decoded = SharedArrayView<const void>(
        std::move(encoded.element_pointer()),
        StridedLayoutView<>(encoded_layout_storage));
  }

  // Compose the bytes -> bytes writers.
  absl::InlinedVector<std::unique_ptr<riegeli::Writer>, 8> writers;
  writers.reserve(bytes_to_bytes.size());
  riegeli::Writer* outer_writer = &writer;
  for (size_t i = bytes_to_bytes.size(); i--;) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_writer, bytes_to_bytes[i]->GetEncodeWriter(*outer_writer));
    const int64_t encoded_size = (i == 0)
                                     ? array_to_bytes->encoded_size()
                                     : bytes_to_bytes[i - 1]->encoded_size();
    if (encoded_size != -1) {
      new_writer->SetWriteSizeHint(encoded_size);
    }
    outer_writer = new_writer.get();
    writers.push_back(std::move(new_writer));
  }

  // Encode the array.
  TENSORSTORE_RETURN_IF_ERROR(
      array_to_bytes->EncodeArray(std::move(decoded), *outer_writer));

  for (size_t i = writers.size(); i--;) {
    auto& w = *writers[i];
    if (!w.Close()) {
      return w.status();
    }
  }

  if (!writer.Close()) {
    return writer.status();
  }
  return absl::OkStatus();
}

Result<SharedArray<const void>> ZarrCodecChain::PreparedState::DecodeArray(
    span<const Index> decoded_shape, riegeli::Reader& reader) const {
  constexpr size_t kNumInlineCodecs = 8;
  // Compose the bytes -> bytes readers.
  absl::InlinedVector<std::unique_ptr<riegeli::Reader>, kNumInlineCodecs>
      readers;
  readers.reserve(bytes_to_bytes.size());
  reader.SetReadAllHint(true);
  riegeli::Reader* outer_reader = &reader;
  for (size_t i = bytes_to_bytes.size(); i--;) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_reader, bytes_to_bytes[i]->GetDecodeReader(*outer_reader));
    new_reader->SetReadAllHint(true);
    outer_reader = new_reader.get();
    readers.push_back(std::move(new_reader));
  }

  // Decode from composed `outer_reader` using array -> bytes codec.
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto array,
      array_to_bytes->DecodeArray(array_to_array.empty()
                                      ? decoded_shape
                                      : array_to_array.back()->encoded_shape(),
                                  *outer_reader));

  for (size_t i = readers.size(); i--;) {
    auto& r = *readers[i];
    if (!r.VerifyEndAndClose()) {
      return r.status();
    }
  }

  if (!reader.Close()) {
    return reader.status();
  }

  // Decode using array -> array codecs.
  for (size_t i = array_to_array.size(); i--;) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        array,
        array_to_array[i]->DecodeArray(
            std::move(array),
            i == 0 ? decoded_shape : array_to_array[i - 1]->encoded_shape()));
  }
  return array;
}

Result<ZarrCodecChain::PreparedState::Ptr> ZarrCodecChain::Prepare(
    span<const Index> decoded_shape) const {
  auto state = internal::MakeIntrusivePtr<PreparedState>();
  for (const auto& codec : array_to_array) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto codec_state,
                                 codec->Prepare(decoded_shape));
    decoded_shape = codec_state->encoded_shape();
    state->array_to_array.push_back(std::move(codec_state));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(state->array_to_bytes,
                               array_to_bytes->Prepare(decoded_shape));
  int64_t encoded_size = state->array_to_bytes->encoded_size();
  for (const auto& codec : bytes_to_bytes) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto codec_state,
                                 codec->Prepare(encoded_size));
    encoded_size = codec_state->encoded_size();
    state->bytes_to_bytes.push_back(std::move(codec_state));
  }
  state->encoded_size_ = encoded_size;
  return state;
}

Result<absl::Cord> ZarrCodecChain::PreparedState::EncodeArray(
    SharedArrayView<const void> decoded) const {
  absl::Cord cord;
  riegeli::CordWriter writer{&cord};
  TENSORSTORE_RETURN_IF_ERROR(this->EncodeArray(std::move(decoded), writer));
  return cord;
}

Result<SharedArray<const void>> ZarrCodecChain::PreparedState::DecodeArray(
    span<const Index> decoded_shape, absl::Cord cord) const {
  riegeli::CordReader reader{&cord};
  return this->DecodeArray(decoded_shape, reader);
}

}  // namespace internal_zarr3
}  // namespace tensorstore
