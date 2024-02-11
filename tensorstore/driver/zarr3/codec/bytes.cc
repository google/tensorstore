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

#include "tensorstore/driver/zarr3/codec/bytes.h"

#include <assert.h>
#include <stdint.h>

#include <optional>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_permutation.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/riegeli/array_endian_codec.h"
#include "tensorstore/internal/unaligned_data_type_functions.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_zarr3 {

namespace {
absl::Status InvalidDataTypeError(DataType dtype) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Data type ", dtype, " not compatible with \"bytes\" codec"));
}

class BytesCodec : public ZarrArrayToBytesCodec {
 public:
  explicit BytesCodec(DataType decoded_dtype, endian endianness)
      : dtype_(decoded_dtype), endianness_(endianness) {}

  Result<PreparedState::Ptr> Prepare(
      span<const Index> decoded_shape) const final;

 private:
  DataType dtype_;
  endian endianness_;
};
}  // namespace

absl::Status BytesCodecSpec::GetDecodedChunkLayout(
    const ArrayDataTypeAndShapeInfo& array_info,
    ArrayCodecChunkLayoutInfo& decoded) const {
  if (array_info.dtype.valid() &&
      !internal::IsTrivialDataType(array_info.dtype)) {
    return InvalidDataTypeError(array_info.dtype);
  }
  const DimensionIndex rank = array_info.rank;
  if (rank != dynamic_rank) {
    auto& inner_order = decoded.inner_order.emplace();
    for (DimensionIndex i = 0; i < rank; ++i) {
      inner_order[i] = i;
    }
  }

  if (array_info.shape) {
    auto& shape = *array_info.shape;
    auto& read_chunk_shape = decoded.read_chunk_shape.emplace();
    for (DimensionIndex i = 0; i < rank; ++i) {
      read_chunk_shape[i] = shape[i];
    }
  }
  // `decoded.codec_chunk_shape` is unspecified.
  return absl::OkStatus();
}

bool BytesCodecSpec::SupportsInnerOrder(
    const ArrayCodecResolveParameters& decoded,
    span<DimensionIndex> preferred_inner_order) const {
  if (!decoded.inner_order) return true;
  if (PermutationMatchesOrder(span(decoded.inner_order->data(), decoded.rank),
                              c_order)) {
    return true;
  }
  SetPermutation(c_order, preferred_inner_order);
  return false;
}

Result<ZarrArrayToBytesCodec::Ptr> BytesCodecSpec::Resolve(
    ArrayCodecResolveParameters&& decoded, BytesCodecResolveParameters& encoded,
    ZarrArrayToBytesCodecSpec::Ptr* resolved_spec) const {
  assert(decoded.dtype.valid());
  if (!internal::IsTrivialDataType(decoded.dtype)) {
    return InvalidDataTypeError(decoded.dtype);
  }
  const bool is_endian_invariant =
      internal::IsEndianInvariantDataType(decoded.dtype);
  if (!options.constraints && !is_endian_invariant && !options.endianness) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("\"bytes\" codec requires that \"endian\" option "
                            "is specified for data type ",
                            decoded.dtype));
  }
  encoded.item_bits = decoded.dtype.size() * 8;
  DimensionIndex rank = decoded.rank;
  if (decoded.codec_chunk_shape) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "\"bytes\" codec does not support codec_chunk_shape (",
        span<const Index>(decoded.codec_chunk_shape->data(), rank),
        " was specified"));
  }
  if (decoded.inner_order) {
    auto& decoded_inner_order = *decoded.inner_order;
    for (DimensionIndex i = 0; i < rank; ++i) {
      if (decoded_inner_order[i] != i) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "\"bytes\" codec does not support inner_order of ",
            span<const DimensionIndex>(decoded_inner_order.data(), rank)));
      }
    }
  }
  endian resolved_endianness = options.endianness.value_or(endian::native);
  if (resolved_spec) {
    resolved_spec->reset(new BytesCodecSpec(Options{
        is_endian_invariant ? std::optional<endian>()
                            : std::optional<endian>(resolved_endianness)}));
  }
  return internal::MakeIntrusivePtr<BytesCodec>(decoded.dtype,
                                                resolved_endianness);
}

namespace {
namespace jb = ::tensorstore::internal_json_binding;
constexpr auto EndiannessBinder() {
  return jb::Enum<endian, std::string_view>({
      {endian::little, "little"},
      {endian::big, "big"},
  });
}
}  // namespace

absl::Status BytesCodecSpec::MergeFrom(const ZarrCodecSpec& other,
                                       bool strict) {
  using Self = BytesCodecSpec;
  const auto& other_options = static_cast<const Self&>(other).options;
  TENSORSTORE_RETURN_IF_ERROR(MergeConstraint<&Options::endianness>(
      "endian", options, other_options, EndiannessBinder()));
  return absl::OkStatus();
}

ZarrCodecSpec::Ptr BytesCodecSpec::Clone() const {
  return internal::MakeIntrusivePtr<BytesCodecSpec>(*this);
}

namespace {
class BytesCodecPreparedState : public ZarrArrayToBytesCodec::PreparedState {
 public:
  int64_t encoded_size() const final { return encoded_size_; }

  absl::Status EncodeArray(SharedArrayView<const void> decoded,
                           riegeli::Writer& writer) const final {
    if (internal::EncodeArrayEndian(std::move(decoded), endianness_, c_order,
                                    writer)) {
      return absl::OkStatus();
    }
    assert(!writer.ok());
    return writer.status();
  }

  Result<SharedArray<const void>> DecodeArray(
      span<const Index> decoded_shape, riegeli::Reader& reader) const final {
    return internal::DecodeArrayEndian(reader, dtype_, decoded_shape,
                                       endianness_, c_order);
  }

  DataType dtype_;
  endian endianness_;
  int64_t encoded_size_;
};
}  // namespace

Result<ZarrArrayToBytesCodec::PreparedState::Ptr> BytesCodec::Prepare(
    span<const Index> decoded_shape) const {
  int64_t bytes = dtype_.size();
  for (auto size : decoded_shape) {
    if (internal::MulOverflow(size, bytes, &bytes)) {
      return absl::OutOfRangeError(tensorstore::StrCat(
          "Integer overflow computing encoded size of array of shape ",
          decoded_shape));
    }
  }
  auto state = internal::MakeIntrusivePtr<BytesCodecPreparedState>();
  state->dtype_ = dtype_;
  state->endianness_ = endianness_;
  state->encoded_size_ = bytes;
  return state;
}

internal::IntrusivePtr<const BytesCodecSpec> DefaultBytesCodec() {
  return internal::MakeIntrusivePtr<BytesCodecSpec>(
      BytesCodecSpec::Options{endian::native});
}

TENSORSTORE_GLOBAL_INITIALIZER {
  using Self = BytesCodecSpec;
  using Options = Self::Options;
  RegisterCodec<Self>(
      "bytes",
      jb::Projection<&Self::options>(jb::Sequence(  //
          [](auto is_loading, const auto& options, auto* obj, auto* j) {
            if constexpr (is_loading) {
              obj->constraints = options.constraints;
            }
            return absl::OkStatus();
          },
          jb::Member("endian",
                     jb::Projection<&Options::endianness>(
                         jb::Optional(EndiannessBinder())))  //
          )));
}

}  // namespace internal_zarr3
}  // namespace tensorstore
