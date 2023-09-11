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

#ifndef TENSORSTORE_DRIVER_ZARR3_CODEC_CODEC_H_
#define TENSORSTORE_DRIVER_ZARR3_CODEC_CODEC_H_

// Defines the `ZarrArrayToArrayCodec`, `ZarrArrayToBytesCodec`, and
// `ZarrBytesToBytesCodec` interfaces along with `ZarrCodecChain` that composes
// them.
//
// See codec_spec.h for the corresponding JSON-serializable interfaces.
//
// Typical usage of `Zarr{ArrayToArray,ArrayToBytes,BytesToBytes}Codec` is as
// follows:
//
// 1. The codec object is constructed from a codec spec (via
//    `ZarrCodecChainSpec`), and stored in a `ZarrCodecChain`.
//
// 2. The codec object cannot be used directly to actually perform
//    encoding/decoding.  Instead, a
//    `Zarr{ArrayToArray,ArrayToBytes,BytesToBytes}Codec::PreparedState` must be
//    created for a specific input array shape or input byte sequence length (if
//    fixed-size).  This allows any size-specific setup and validation to be
//    done once and then shared for multiple inputs.  For a sequence of codecs,
//    the prepared states are generated sequentially; each subsequent codec is
//    prepared based on the encoded size information returned from the prior
//    prepared state.  Note: The `PreparedState` must not outlive the
//    ``Zarr{ArrayToArray,ArrayToBytes,BytesToBytes}Codec` from which it was
//    created.
//
// 3. Encode/decode operations for the prepared input shape are performed using
//    the sequence of prepared states.

#include <stdint.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/chunk_grid_specification.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/internal/storage_statistics.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_zarr3 {

// "Array -> array" codec interface.
class ZarrArrayToArrayCodec
    : public internal::AtomicReferenceCount<ZarrArrayToArrayCodec> {
 public:
  using Ptr = internal::IntrusivePtr<const ZarrArrayToArrayCodec>;

  class PreparedState : public internal::AtomicReferenceCount<PreparedState> {
   public:
    using Ptr = internal::IntrusivePtr<const PreparedState>;

    virtual span<const Index> encoded_shape() const = 0;

    // Encodes a complete array.
    //
    // This is used only when this codec is followed by a non-sharding "array ->
    // bytes" codec.
    //
    // \pre `decoded.shape()` equals the `decoded_shape` passed when creating
    //     this prepared state.
    virtual Result<SharedArray<const void>> EncodeArray(
        SharedArrayView<const void> decoded) const = 0;

    // Decodes a complete array.
    //
    // This is used only when this codec is followed by a non-sharding "array ->
    // bytes" codec.
    //
    // \pre `decoded.shape()` equals the `decoded_shape` passed when creating
    //     this prepared state.
    virtual Result<SharedArray<const void>> DecodeArray(
        SharedArrayView<const void> encoded,
        span<const Index> decoded_shape) const = 0;

    // TODO(jbms): Add NDIterable or similar encode/decode interface.

    using NextReader =
        std::function<void(IndexTransform<> transform,
                           AnyFlowReceiver<absl::Status, internal::ReadChunk,
                                           IndexTransform<>>&& receiver)>;

    // Performs a (partial) read.
    //
    // This is used only when this codec is followed by a sharding "array ->
    // bytes" codec.
    //
    // This should apply any necessary composition to `transform` and `receiver`
    // and then forward read requests to `next`.
    //
    // \pre `decoded_shape` equals the `decoded_shape` passed when creating this
    //     prepared state.
    virtual void Read(const NextReader& next, span<const Index> decoded_shape,
                      IndexTransform<> transform,
                      AnyFlowReceiver<absl::Status, internal::ReadChunk,
                                      IndexTransform<>>&& receiver) const = 0;

    using NextWriter =
        std::function<void(IndexTransform<> transform,
                           AnyFlowReceiver<absl::Status, internal::WriteChunk,
                                           IndexTransform<>>&& receiver)>;

    // Performs a (partial) write.
    //
    // This is used only when this codec is followed by a sharding "array ->
    // bytes" codec.
    //
    // This should apply any necessary composition to `transform` and `receiver`
    // and then forward write requests to `next`.
    //
    // \pre `decoded_shape` equals the `decoded_shape` passed when creating this
    //     prepared state.
    virtual void Write(const NextWriter& next, span<const Index> decoded_shape,
                       IndexTransform<> transform,
                       AnyFlowReceiver<absl::Status, internal::WriteChunk,
                                       IndexTransform<>>&& receiver) const = 0;

    using NextGetStorageStatistics = std::function<void(
        IndexTransform<> transform,
        internal::IntrusivePtr<
            internal::GetStorageStatisticsAsyncOperationState>
            state)>;

    virtual void GetStorageStatistics(
        const NextGetStorageStatistics& next, span<const Index> decoded_shape,
        IndexTransform<> transform,
        internal::IntrusivePtr<
            internal::GetStorageStatisticsAsyncOperationState>
            state) const = 0;

    virtual ~PreparedState();
  };

  virtual ~ZarrArrayToArrayCodec();

  // Returns a prepared state that may be used to decode arrays of the specified
  // shape.
  virtual Result<PreparedState::Ptr> Prepare(
      span<const Index> decoded_shape) const = 0;
};

// "Array -> bytes" codec interface.
class ZarrArrayToBytesCodec
    : public internal::AtomicReferenceCount<ZarrArrayToBytesCodec> {
 public:
  using Ptr = internal::IntrusivePtr<const ZarrArrayToBytesCodec>;

  virtual ~ZarrArrayToBytesCodec();

  class PreparedState : public internal::AtomicReferenceCount<PreparedState> {
   public:
    using Ptr = internal::IntrusivePtr<const PreparedState>;

    // Returns the encoded size in bytes, or `-1` if it may vary.
    //
    // The default implementation returns `-1`.
    virtual int64_t encoded_size() const;

    // Encodes a complete array.
    //
    // This is not called for sharding codecs.
    virtual absl::Status EncodeArray(SharedArrayView<const void> decoded,
                                     riegeli::Writer& writer) const = 0;

    // Decodes a complete array.
    //
    // This is not called for sharding codecs.
    virtual Result<SharedArray<const void>> DecodeArray(
        span<const Index> decoded_shape, riegeli::Reader& reader) const = 0;

    // Note: For sharding codecs, the methods defined by
    // `ZarrShardingCodec::PreparedState` are used instead.

    // TODO(jbms): Consider inverting the control flow, and providing a
    // `riegeli::Reader` for encoding and a `riegeli::Writer` for decoding.

    virtual ~PreparedState();
  };

  // Indicates if this is a sharding codec.
  virtual bool is_sharding_codec() const { return false; }

  virtual Result<PreparedState::Ptr> Prepare(
      span<const Index> decoded_shape) const = 0;
};

// "Bytes -> bytes" codec interface.
class ZarrBytesToBytesCodec
    : public internal::AtomicReferenceCount<ZarrBytesToBytesCodec> {
 public:
  using Ptr = internal::IntrusivePtr<const ZarrBytesToBytesCodec>;

  virtual ~ZarrBytesToBytesCodec();

  class PreparedState : public internal::AtomicReferenceCount<PreparedState> {
   public:
    using Ptr = internal::IntrusivePtr<const PreparedState>;

    // Returns the encoded size in bytes, or `-1` if it may vary.
    //
    // The default implementation returns `-1`.
    virtual int64_t encoded_size() const;

    // Returns a writer that receives the decoded representation as input and
    // writes the encoded representation to `encoded_writer`.
    //
    // The caller is responsible for closing `encoded_writer` once this writer
    // is destroyed.
    virtual Result<std::unique_ptr<riegeli::Writer>> GetEncodeWriter(
        riegeli::Writer& encoded_writer) const = 0;

    // Returns a reader that must return the decoded representation and reads
    // the encoded representation from `encoded_reader`.
    //
    // The caller is responsible for closing
    // `encoded_reader.VerifyEndAndClose()` once this reader is destroyed.  It
    // is an error if all data is not read from `encoded_reader`.
    virtual Result<std::unique_ptr<riegeli::Reader>> GetDecodeReader(
        riegeli::Reader& encoded_reader) const = 0;

    virtual ~PreparedState();
  };

  virtual Result<PreparedState::Ptr> Prepare(int64_t decoded_size) const = 0;
};

// Composes zero or more "array -> array" codecs, one "array -> bytes" codec,
// and zero or more "bytes -> bytes" codecs.
class ZarrCodecChain : public internal::AtomicReferenceCount<ZarrCodecChain> {
 public:
  using Ptr = internal::IntrusivePtr<const ZarrCodecChain>;

  class PreparedState : public ZarrArrayToBytesCodec::PreparedState {
   public:
    using Ptr = internal::IntrusivePtr<const PreparedState>;
    int64_t encoded_size() const final { return encoded_size_; }

    // Encodes a complete array.
    //
    // This is not used if `array_to_bytes` is a sharding codec.
    //
    // This is a convenience interface to the `riegeli::Writer&` overload
    // defined below.
    Result<absl::Cord> EncodeArray(SharedArrayView<const void> decoded) const;

    // Decodes a complete array.
    //
    // This is not used if `array_to_bytes` is a sharding codec.
    //
    // This is a convenience interface to the `riegeli::Reader&` overload
    // defined below.
    Result<SharedArray<const void>> DecodeArray(span<const Index> decoded_shape,
                                                absl::Cord cord) const;

    // Encodes a complete array.
    //
    // This is not used if `array_to_bytes` is a sharding codec.
    absl::Status EncodeArray(SharedArrayView<const void> decoded,
                             riegeli::Writer& writer) const final;

    // Decodes a complete array.
    //
    // This is not used if `array_to_bytes` is a sharding codec.
    Result<SharedArray<const void>> DecodeArray(
        span<const Index> decoded_shape, riegeli::Reader& reader) const final;

    std::vector<ZarrArrayToArrayCodec::PreparedState::Ptr> array_to_array;
    ZarrArrayToBytesCodec::PreparedState::Ptr array_to_bytes;
    std::vector<ZarrBytesToBytesCodec::PreparedState::Ptr> bytes_to_bytes;

   private:
    friend class ZarrCodecChain;
    int64_t encoded_size_;
  };

  Result<PreparedState::Ptr> Prepare(span<const Index> decoded_shape) const;

  std::vector<ZarrArrayToArrayCodec::Ptr> array_to_array;
  ZarrArrayToBytesCodec::Ptr array_to_bytes;
  std::vector<ZarrBytesToBytesCodec::Ptr> bytes_to_bytes;
};

// Special subtype of "array -> bytes" codecs that perform sharding.
class ZarrShardingCodec : public ZarrArrayToBytesCodec {
 public:
  using ZarrArrayToBytesCodec::ZarrArrayToBytesCodec;

  class PreparedState : public ZarrArrayToBytesCodec::PreparedState {
   public:
    using Ptr = internal::IntrusivePtr<const PreparedState>;

    // Returns a KvStore adapter for reading and writing sub-chunks.
    virtual kvstore::DriverPtr GetSubChunkKvstore(
        kvstore::DriverPtr parent, std::string parent_key,
        const Executor& executor,
        internal::CachePool::WeakPtr cache_pool) const = 0;

    // Returns the key formatter/parser for use with the kvstore returned by
    // `GetSubChunkKvstore`.
    virtual const internal::LexicographicalGridIndexKeyParser&
    GetSubChunkStorageKeyParser() const = 0;

    // Specifies the sub-chunk grid.
    //
    // Must be set to a non-null value.
    const internal::ChunkGridSpecification* sub_chunk_grid;

    // Specifies the codec chain for encoding/decoding sub-chunks.
    //
    // Must be set to a non-null value.
    const ZarrCodecChain* sub_chunk_codec_chain;

    // Specifies the prepared state for encoding/decoding sub-chunks.
    //
    // Must be set to a non-null value.
    const ZarrCodecChain::PreparedState* sub_chunk_codec_state;
  };

  bool is_sharding_codec() const override;
};

}  // namespace internal_zarr3
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR3_CODEC_CODEC_H_
