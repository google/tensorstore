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

#ifndef TENSORSTORE_DRIVER_ZARR3_CODEC_CODEC_CHAIN_SPEC_H_
#define TENSORSTORE_DRIVER_ZARR3_CODEC_CODEC_CHAIN_SPEC_H_

#include <stddef.h>

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/cache_key/fwd.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_zarr3 {

// Specifies a precise configuration of codecs (when loading from stored zarr v3
// metadata), or constraints on a sequence of codecs (when parsed from a
// TensorStore spec).
//
// To encode an array as a byte sequence, the "array -> array" codecs are
// applied in sequence, then the "array -> bytes" codec, and then the "bytes ->
// bytes" codecs.
struct ZarrCodecChainSpec {
  // Specifies zero or more "array -> array" codecs.
  std::vector<ZarrArrayToArrayCodecSpec::Ptr> array_to_array;

  // Specifies the "array -> bytes" codec.  May be nullptr (to indicate no
  // constraints) if this merely specifies constraints.  If this specifies a
  // precise codec chain configuration, must not be `nullptr`.
  ZarrArrayToBytesCodecSpec::Ptr array_to_bytes;

  // Specifies zero or more "bytes -> bytes" codecs.
  std::vector<ZarrBytesToBytesCodecSpec::Ptr> bytes_to_bytes;

  // Indicates the nested sharding codec height.
  //
  // Equal to 0 if the "array -> bytes" codec is unspecified (`nullptr`).
  // Otherwise, equals `array_to_bytes->sharding_height()`.
  size_t sharding_height() const;

  // Merges this spec with `other`, returning an error if any constraints are
  // incompatible.
  //
  // If `*this` was loaded as a precise configuration from stored zarr v3
  // metadata (or obtained from `Resolve`), then merging with a constraints spec
  // serves to validate that the constraints all hold.
  //
  // Args:
  //
  //   strict: Indicates whether merging should use the "strict" rules
  //     applicable for the codecs from the `metadata` member in the TensorStore
  //     spec, rather than "relaxed" rules applicable for the codecs specified
  //     in the schema.
  //
  //     - In strict merging mode, the `array_to_array` and `bytes_to_bytes`
  //       lists must exactly correspond, and the `array_to_bytes` codecs, if
  //       both specified, must also exactly correspond.
  //
  //     - In relaxed merging mode, the one of the `array_to_array` sequences
  //       may contain a leading "transpose" codec that is not present in the -
  //       other sequence.  Additionally, if one of the two specs has a greater
  //       `sharding_height` and empty `array_to_array` and `bytes_to_bytes`
  //       lists, then an attempt is made to merge the other spec with its
  //       sub-chunk codecs.
  //
  //     The relaxed merging mode is needed to allow the user to specify just
  //     the inner codecs, and allow an extra "transpose" or sharding codec to
  //     be inserted automatically based on the `ChunkLayout`.
  absl::Status MergeFrom(const ZarrCodecChainSpec& other, bool strict);

  // Computes the chunk layout, to the extent that it can be determined.
  //
  // This is used to compute partial chunk layout information for a zarr v3
  // spec.
  //
  // It also serves to further constrain the chunk layout when creating a new
  // array.
  //
  // Args:
  //   array_info: Information about the decoded chunk.
  //   decoded[out]: Chunk layout information to be set.
  absl::Status GetDecodedChunkLayout(
      const ArrayDataTypeAndShapeInfo& array_info,
      ArrayCodecChunkLayoutInfo& decoded) const;

  // Generates a resolved codec chain (that may be used for encode/decode
  // operations).
  //
  // - Any codec options that were not fully specified are chosen automatically
  //   based on the `decoded` parameters.
  //
  // - If an "array -> bytes" codec is not specified, one is chosen
  //   automatically.
  //
  // - If necessary to satisfy `decoded.inner_order`, a "transpose" codec may be
  //   inserted automatically just before the inner-most "array -> bytes" codec.
  //
  // Args:
  //   decoded: Information about the decoded chunk.
  //   encoded[out]: Information about the resultant encoded byte sequence to be
  //     set.
  //   resolved_spec[out]: If not `nullptr`, set to the precise codec
  //     configuration chosen.
  Result<internal::IntrusivePtr<const ZarrCodecChain>> Resolve(
      ArrayCodecResolveParameters&& decoded,
      BytesCodecResolveParameters& encoded,
      ZarrCodecChainSpec* resolved_spec = nullptr) const;

  using FromJsonOptions = ZarrCodecSpec::FromJsonOptions;
  using ToJsonOptions = ZarrCodecSpec::ToJsonOptions;

  // Default JSON binder.
  //
  // Note that the `ZarrCodecSpec::FromJsonOptions::constraints` parameter must
  // be set.
  //
  // When used from within another JSON binder that does not use
  // `ZarrCodecSpec::FromJsonOptions`, `ZarrCodecChainJsonBinder` defined below
  // may be more convenient.
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ZarrCodecChainSpec, FromJsonOptions,
                                          ToJsonOptions);
};

// Codec spec for a "sharding codec" that further divides the chunk into a grid,
// supports reading just individual sub-chunks.
class ZarrShardingCodecSpec : public ZarrArrayToBytesCodecSpec {
 public:
  // Always returns `true` since this is not called for sharding codecs.
  bool SupportsInnerOrder(
      const ArrayCodecResolveParameters& decoded,
      span<DimensionIndex> preferred_inner_order) const override;

  // Merges with the sub-chunk codecs.
  virtual absl::Status MergeSubChunkCodecsFrom(const ZarrCodecChainSpec& other,
                                               bool strict) = 0;

  // Returns the sub-chunk codec spec, or `nullptr` if unspecified.
  virtual const ZarrCodecChainSpec* GetSubChunkCodecs() const = 0;

  // Returns the sharding height.
  //
  // This does not need to be overridden by subclasses.
  size_t sharding_height() const override;
};

// JSON binder for `ZarrCodecChain` where the
// `ZarrCodecSpec::FromJsonOptions::constraints` parameter is fixed at
// compile-time.
template <bool Constraints>
constexpr auto ZarrCodecChainJsonBinder =
    [](auto is_loading, const auto& orig_options, auto* obj, auto* j) {
      using CodecOptions = std::conditional_t<decltype(is_loading)::value,
                                              ZarrCodecSpec::FromJsonOptions,
                                              ZarrCodecSpec::ToJsonOptions>;

      CodecOptions codec_options;
      codec_options.constraints = Constraints;
      if constexpr (!is_loading) {
        static_cast<IncludeDefaults&>(codec_options) = orig_options;
      }
      return ZarrCodecChainSpec::default_json_binder(is_loading, codec_options,
                                                     obj, j);
    };

absl::Status MergeZarrCodecSpecs(
    std::optional<ZarrCodecChainSpec>& target,
    const std::optional<ZarrCodecChainSpec>& source, bool strict);

class TensorStoreCodecSpec : public internal::CodecDriverSpec {
 public:
  constexpr static char id[] = "zarr3";

  CodecSpec Clone() const final;
  absl::Status DoMergeFrom(const internal::CodecDriverSpec& other_base) final;

  std::optional<ZarrCodecChainSpec> codecs;
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(TensorStoreCodecSpec, FromJsonOptions,
                                          ToJsonOptions,
                                          ::nlohmann::json::object_t)
};

}  // namespace internal_zarr3
namespace internal {
template <>
struct CacheKeyEncoder<internal_zarr3::ZarrCodecChainSpec> {
  static void Encode(std::string* out,
                     const internal_zarr3::ZarrCodecChainSpec& value);
};
}  // namespace internal
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_zarr3::ZarrCodecChainSpec)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_zarr3::ZarrCodecChainSpec)

#endif  // TENSORSTORE_DRIVER_ZARR3_CODEC_CODEC_CHAIN_SPEC_H_
