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

#ifndef TENSORSTORE_DRIVER_ZARR3_CODEC_CODEC_SPEC_H_
#define TENSORSTORE_DRIVER_ZARR3_CODEC_CODEC_SPEC_H_

// Defines `ZarrCodecSpec` and the `ZarrCodecKind`-specific subclasses.
//
// A `ZarrCodecSpec` actually serves two separate but related purposes:
//
// 1. Representing the actual codec configuration as specified in the stored
//    zarr metadata.
//
// 2. Specifying *constraints* on the codec configuration, but possibly leaving
//    some options unspecified/unconstrained, for use in a TensorStore spec.
//    When opening an existing array and validating the constraints, any option
//    is allowed to match an unspecified option.  When creating a new array,
//    default values are chosen automatically for any unspecified options.
//
// When parsing the JSON representation, the
// `ZarrCodecSpec::FromJsonOptions::constraints` option is used to distinguish
// between these two purposes:
//
// - when parsing the actual stored metadata, leaving any required configuration
//   option unspecified is an error, and any optional configuration options are
//   resolved to a fixed default value (which then serves as a hard constraint).
//
// - when parsing constraints, unspecified options may be allowed by the codec
//   implementation, and remain unconstrained.
//
// Typical usage of `ZarrCodecSpec` is as follows:
//
// 1. A sequence of `ZarrCodecSpec` objects are parsed from a JSON array (via
//    `ZarrCodecChainSpec`).
//
// 2. Based on their `ZarrCodecSpec::kind()` values, the codec specs are
//    validated.
//
// 3. Additional constraints may be merged in by calling
//    `ZarrCodecSpec::MergeFrom`.
//
// 4. If chunk layout information is needed, the array data type, rank, and
//    shape are first propagated forward through the "array -> array" codecs by
//    calling `ZarrArrayToArrayCodecSpec::PropagateDataTypeAndShape`.  Then the
//    chunk layout information is propagated backwards through the "array ->
//    array" codecs by calling
//    `ZarrArrayToBytesCodecSpec::GetDecodedChunkLayout` and
//    `ZarrArrayToArrayCodecSpec::GetDecodedChunkLayout`.
//
// 5. To convert the codec specs into actual codecs, the
//    `Zarr{ArrayToArray,ArrayToBytes,BytesToBytes}CodecSpec::Resolve` methods
//    is called, propagating parameters forward from one codec to the next.  The
//    `Resolve` methods also optionally return a resolved codec spec, where any
//    unspecified options have been filled in with the actual values chosen.

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <optional>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_zarr3 {

enum class ZarrCodecKind {
  kArrayToArray,
  kArrayToBytes,
  kBytesToBytes,
};

// Base class for codec specs, may refer to an "array -> array", "array ->
// bytes", or "bytes -> bytes" codec.
class ZarrCodecSpec : public internal::AtomicReferenceCount<ZarrCodecSpec> {
 public:
  using Ptr = internal::IntrusivePtr<const ZarrCodecSpec>;
  virtual ~ZarrCodecSpec();
  virtual ZarrCodecKind kind() const = 0;

  // Merge the constraints specified by `*this` with those of `other`.
  //
  // It is guaranteed that `typeid(other) == typeid(*this)`.
  //
  // The `strict` flag should be forwarded to any nested calls to
  // `ZarrCodecChainSpec::MergeFrom`.
  virtual absl::Status MergeFrom(const ZarrCodecSpec& other, bool strict) = 0;

  // Returns a new copy of `*this`, with the same derived type.
  //
  // This is used to implement copy-on-write for `MergeFrom`.
  virtual Ptr Clone() const = 0;

  struct FromJsonOptions {
    // Indicates that the `ZarrCodecSpec` will only specify constraints on a
    // codec, and may be missing options required in the actual metadata.  This
    // must be set to `false` when parsing the actual stored metadata, and
    // should be set to `true` when parsing the metadata constraints in the
    // TensorStore spec.
    //
    // If `constraints == false`, individual codec implementations must ensure
    // that all options are fully specified after parsing.
    bool constraints = false;
  };

  struct ToJsonOptions : public IncludeDefaults {
    constexpr ToJsonOptions() = default;
    constexpr ToJsonOptions(IncludeDefaults include_defaults)
        : IncludeDefaults(include_defaults) {}
    bool constraints = false;
  };
};

struct ArrayDataTypeAndShapeInfo {
  // Specifies the data type of the array on which the codec will operate.
  DataType dtype;

  // Specifies the rank of the array on which the codec will operate.
  DimensionIndex rank = dynamic_rank;

  // Specifies the shape of the array on which the codec will operate.
  std::optional<std::array<Index, kMaxRank>> shape;
};

// Specifies information about the chunk layout that must be propagated through
// the "array -> array" and "array -> bytes" codecs when calling
// `CodecChainSpec::GetDecodedChunkLayout`.
struct ArrayCodecChunkLayoutInfo {
  // Specifies the preferred storage layout, where the first dimension is the
  // "outer most" dimension, and the last dimension is the "inner most"
  // dimension.
  std::optional<std::array<DimensionIndex, kMaxRank>> inner_order;

  // Specifies requested read chunk shape.
  std::optional<std::array<Index, kMaxRank>> read_chunk_shape;

  // Specifies requested codec chunk shape.
  std::optional<std::array<Index, kMaxRank>> codec_chunk_shape;
};

// Specifies information about the chunk that must be propagated through the
// "array -> array" and "array -> bytes" codecs when calling
// `CodecChainSpec::Resolve`.
struct ArrayCodecResolveParameters {
  // Specifies the data type of the array on which the codec will operate.
  DataType dtype;

  // Specifies the rank of the array on which the codec will operate.
  DimensionIndex rank;

  // Specifies the fill value.
  SharedArray<const void> fill_value;

  // Specifies requested read chunk shape.
  std::optional<std::array<Index, kMaxRank>> read_chunk_shape;

  // Specifies requested codec chunk shape.
  std::optional<std::array<Index, kMaxRank>> codec_chunk_shape;

  // Specifies required inner order.
  std::optional<std::array<DimensionIndex, kMaxRank>> inner_order;
};

// Spec for an "array -> array" codec.
//
// After parsing
class ZarrArrayToArrayCodecSpec : public ZarrCodecSpec {
 public:
  using Ptr = internal::IntrusivePtr<const ZarrArrayToArrayCodecSpec>;
  ZarrCodecKind kind() const final;

  // Computes information about the "encoded" representation given information
  // about the "decoded" representation.
  virtual absl::Status PropagateDataTypeAndShape(
      const ArrayDataTypeAndShapeInfo& decoded,
      ArrayDataTypeAndShapeInfo& encoded) const = 0;

  // Computes chunk layout information about the "decoded" representation given
  // chunk layout information about the "encoded" representation.
  //
  // Args:
  //
  //   encoded_info: Information about the "encoded" representation, computed
  //     from `decoded_info` by a prior call to
  //     `PropagatedDataTypeAndShapeInfo`.
  //   encoded: Chunk layout information for the "encoded" representation, to
  //     propagate to the "decoded" representation.
  //   decoded_info: Information about the "decoded" representation.
  //   decoded[out]: Chunk layout information for the "decoded" representation,
  //     to be set.
  virtual absl::Status GetDecodedChunkLayout(
      const ArrayDataTypeAndShapeInfo& encoded_info,
      const ArrayCodecChunkLayoutInfo& encoded,
      const ArrayDataTypeAndShapeInfo& decoded_info,
      ArrayCodecChunkLayoutInfo& decoded) const = 0;

  // Computes the resolved codec.
  //
  // Args:
  //   decoded: Information about the decoded chunk.
  //   encoded[out]: Information about the resultant encoded chunk to be set;
  //     these constraints are passed to the next "array -> array" or "array ->
  //     bytes" codec.
  //   resolved_spec[out]: If not `nullptr`, set to the precise codec
  //     configuration chosen.
  virtual Result<internal::IntrusivePtr<const ZarrArrayToArrayCodec>> Resolve(
      ArrayCodecResolveParameters&& decoded,
      ArrayCodecResolveParameters& encoded,
      ZarrArrayToArrayCodecSpec::Ptr* resolved_spec) const = 0;
};

// Specifies information about an encoded byte sequence that must be propagated
// through the "array -> bytes" and "bytes -> bytes" codecs when calling
// `CodecChainSpec::Resolve`.
struct BytesCodecResolveParameters {
  // If the byte sequence is actually a sequence of fixed-size items, this
  // specifies the item size in bits.
  //
  // For example:
  //
  // - If the byte sequence is actually a sequence of uint32le values, this
  //   should be set to 32.
  //
  // - If the byte sequence is a sequence of packed int4 values, this should be
  //   set to 4.
  //
  // This is used by the "blosc" codec to choose shuffle parameters
  // automatically.
  int64_t item_bits = -1;
};

// Spec for an "array -> bytes" codec.
class ZarrArrayToBytesCodecSpec : public ZarrCodecSpec {
 public:
  using Ptr = internal::IntrusivePtr<const ZarrArrayToBytesCodecSpec>;

  ZarrCodecKind kind() const final;

  // Computes chunk layout information about the "decoded" representation given
  // the decoded chunk information.
  //
  // Args:
  //   array_info: Information about the "decoded" representation.
  //   decoded[out]: Chunk layout information for the "decoded" representation,
  //     to be set.
  virtual absl::Status GetDecodedChunkLayout(
      const ArrayDataTypeAndShapeInfo& array_info,
      ArrayCodecChunkLayoutInfo& decoded) const = 0;

  // Indicates if the specified inner order is supported by `Resolve`.
  //
  // Not called for sharding codecs.
  //
  // If this returns `false`, `preferred_inner_order` must be filled in with a
  // preferred dimension order that will be supported.
  //
  // Args:
  //
  //   decoded: Requirements on the "decoded" representation.  Normally only
  //     `decoded.inner_order` will be relevant but other parameters may impact
  //     the result.
  //   preferred_inner_order[out]: In the case that `false` is returned, must be
  //     filled with the inner order that will be supported by `Resolve`.
  //
  // Returns:
  //   `true` if `Resolve` won't fail due to `decoded.inner_order`, or `false`
  //   if a different `inner_order` is required.
  //
  //   It is expected that if `GetDecodedChunkLayout` returns an inner order,
  //   that this function returns `true` when passed the same order.
  virtual bool SupportsInnerOrder(
      const ArrayCodecResolveParameters& decoded,
      span<DimensionIndex> preferred_inner_order) const = 0;

  // Computes the resolved codec.
  //
  // Args:
  //   decoded: Information about the decoded chunk.
  //   encoded[out]: Information about the resultant encoded byte sequence to be
  //     set; these constraints are passed to the next "bytes -> bytes" codec.
  //   resolved_spec[out]: If not `nullptr`, set to the precise codec
  //     configuration chosen.
  virtual Result<internal::IntrusivePtr<const ZarrArrayToBytesCodec>> Resolve(
      ArrayCodecResolveParameters&& decoded,
      BytesCodecResolveParameters& encoded,
      ZarrArrayToBytesCodecSpec::Ptr* resolved_spec) const = 0;

  // Equal to 0 if this is a non-sharding codec.
  //
  // Otherwise, equal to 1 + the sharding height of the inner "array -> bytes"
  // codec.  For example, with a single level of sharding,
  // `sharding_height() == 1` for the sharding codec.  With two levels of
  // sharding, `sharding_height() == 2` for the outer sharding codec and
  // `sharding_height() == 1` for the inner sharding codec.
  virtual size_t sharding_height() const;
};

// Spec for a "bytes -> bytes" codec.
class ZarrBytesToBytesCodecSpec : public ZarrCodecSpec {
 public:
  using Ptr = internal::IntrusivePtr<const ZarrBytesToBytesCodecSpec>;
  ZarrCodecKind kind() const final;

  // Computes the resolved codec.
  //
  // Args:
  //   decoded: Information about the decoded byte sequence.
  //   encoded[out]: Information about the resultant encoded byte sequence to be
  //     set; these constraints are passed to the next "bytes -> bytes" codec.
  //   resolved_spec[out]: If not `nullptr`, set to the precise codec
  //     configuration chosen.
  virtual Result<internal::IntrusivePtr<const ZarrBytesToBytesCodec>> Resolve(
      BytesCodecResolveParameters&& decoded,
      BytesCodecResolveParameters& encoded,
      ZarrBytesToBytesCodecSpec::Ptr* resolved_spec) const = 0;
};

}  // namespace internal_zarr3
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR3_CODEC_CODEC_SPEC_H_
