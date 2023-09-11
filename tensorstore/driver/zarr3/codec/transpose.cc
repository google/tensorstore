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

#include "tensorstore/driver/zarr3/codec/transpose.h"

#include <array>
#include <cassert>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_permutation.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/storage_statistics.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_zarr3 {

namespace {
absl::Status InvalidPermutationError(span<const DimensionIndex> order,
                                     DimensionIndex rank) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      order, " is not a valid dimension permutation for a rank ", rank,
      " array"));
}
}  // namespace

absl::Status TransposeCodecSpec::MergeFrom(const ZarrCodecSpec& other,
                                           bool strict) {
  using Self = TransposeCodecSpec;
  const auto& other_options = static_cast<const Self&>(other).options;
  return MergeConstraint<&Options::order>("order", options, other_options);
}

ZarrCodecSpec::Ptr TransposeCodecSpec::Clone() const {
  return internal::MakeIntrusivePtr<TransposeCodecSpec>(*this);
}

namespace {
class TransposeCodec : public ZarrArrayToArrayCodec {
 public:
  explicit TransposeCodec(std::vector<DimensionIndex> inverse_order)
      : inverse_order_(std::move(inverse_order)) {}

  class State : public ZarrArrayToArrayCodec::PreparedState {
   public:
    span<const Index> encoded_shape() const final { return encoded_shape_; }

    Result<SharedArray<const void>> EncodeArray(
        SharedArrayView<const void> decoded) const final {
      span<const DimensionIndex> inverse_order = codec_->inverse_order_;
      assert(decoded.rank() == inverse_order.size());
      SharedArray<const void> encoded;
      encoded.layout().set_rank(inverse_order.size());
      encoded.element_pointer() = std::move(decoded.element_pointer());
      for (DimensionIndex decoded_dim = 0; decoded_dim < encoded.rank();
           ++decoded_dim) {
        const DimensionIndex encoded_dim = inverse_order[decoded_dim];
        encoded.shape()[encoded_dim] = decoded.shape()[decoded_dim];
        encoded.byte_strides()[encoded_dim] =
            decoded.byte_strides()[decoded_dim];
      }
      return encoded;
    }

    Result<SharedArray<const void>> DecodeArray(
        SharedArrayView<const void> encoded,
        span<const Index> decoded_shape) const final {
      span<const DimensionIndex> inverse_order = codec_->inverse_order_;
      assert(encoded.rank() == inverse_order.size());
      SharedArray<const void> decoded;
      decoded.layout().set_rank(inverse_order.size());
      decoded.element_pointer() = std::move(encoded.element_pointer());
      for (DimensionIndex decoded_dim = 0; decoded_dim < encoded.rank();
           ++decoded_dim) {
        const DimensionIndex encoded_dim = inverse_order[decoded_dim];
        decoded.shape()[decoded_dim] = encoded.shape()[encoded_dim];
        decoded.byte_strides()[decoded_dim] =
            encoded.byte_strides()[encoded_dim];
      }
      assert(internal::RangesEqual(decoded_shape, decoded.shape()));
      return decoded;
    }

    void Read(const NextReader& next, span<const Index> decoded_shape,
              IndexTransform<> transform,
              AnyFlowReceiver<absl::Status, internal::ReadChunk,
                              IndexTransform<>>&& receiver) const final {
      next(std::move(transform).TransposeOutput(codec_->inverse_order_),
           std::move(receiver));
    }

    void Write(const NextWriter& next, span<const Index> decoded_shape,
               IndexTransform<> transform,
               AnyFlowReceiver<absl::Status, internal::WriteChunk,
                               IndexTransform<>>&& receiver) const final {
      next(std::move(transform).TransposeOutput(codec_->inverse_order_),
           std::move(receiver));
    }

    void GetStorageStatistics(
        const NextGetStorageStatistics& next, span<const Index> decoded_shape,
        IndexTransform<> transform,
        internal::IntrusivePtr<
            internal::GetStorageStatisticsAsyncOperationState>
            state) const final {
      next(std::move(transform).TransposeOutput(codec_->inverse_order_),
           std::move(state));
    }

    const TransposeCodec* codec_;
    std::vector<Index> encoded_shape_;
  };

  Result<PreparedState::Ptr> Prepare(
      span<const Index> decoded_shape) const final {
    if (decoded_shape.size() != inverse_order_.size()) {
      std::vector<DimensionIndex> order(inverse_order_.size());
      InvertPermutation(order.size(), inverse_order_.data(), order.data());
      return InvalidPermutationError(order, decoded_shape.size());
    }
    auto state = internal::MakeIntrusivePtr<State>();
    state->codec_ = this;
    state->encoded_shape_.resize(decoded_shape.size());
    for (DimensionIndex decoded_dim = 0; decoded_dim < decoded_shape.size();
         ++decoded_dim) {
      const DimensionIndex encoded_dim = inverse_order_[decoded_dim];
      state->encoded_shape_[encoded_dim] = decoded_shape[decoded_dim];
    }
    return state;
  }

 private:
  std::vector<DimensionIndex> inverse_order_;
};
}  // namespace

absl::Status TransposeCodecSpec::PropagateDataTypeAndShape(
    const ArrayDataTypeAndShapeInfo& decoded,
    ArrayDataTypeAndShapeInfo& encoded) const {
  span<const DimensionIndex> order = options.order;
  if (!RankConstraint::Implies(order.size(), decoded.rank)) {
    return InvalidPermutationError(order, decoded.rank);
  }
  encoded.dtype = decoded.dtype;
  encoded.rank = order.size();
  if (decoded.shape) {
    auto& encoded_shape = encoded.shape.emplace();
    const auto& decoded_shape = *decoded.shape;
    for (DimensionIndex encoded_dim = 0; encoded_dim < order.size();
         ++encoded_dim) {
      const DimensionIndex decoded_dim = order[encoded_dim];
      encoded_shape[encoded_dim] = decoded_shape[decoded_dim];
    }
  }
  return absl::OkStatus();
}

namespace {
void PropagateInnerOrderToDecoded(
    span<const DimensionIndex> order,
    const std::optional<std::array<DimensionIndex, kMaxRank>>&
        encoded_inner_order,
    std::optional<std::array<DimensionIndex, kMaxRank>>& decoded_inner_order) {
  if (!encoded_inner_order) return;
  auto& encoded = *encoded_inner_order;
  auto& decoded = decoded_inner_order.emplace();
  for (DimensionIndex i = 0; i < order.size(); ++i) {
    decoded[i] = order[encoded[i]];
  }
}

void PropagateShapeToDecoded(
    span<const DimensionIndex> order,
    const std::optional<std::array<Index, kMaxRank>>& encoded_shape,
    std::optional<std::array<Index, kMaxRank>>& decoded_shape) {
  if (!encoded_shape) return;
  auto& encoded = *encoded_shape;
  auto& decoded = decoded_shape.emplace();
  for (DimensionIndex encoded_dim = 0; encoded_dim < order.size();
       ++encoded_dim) {
    const DimensionIndex decoded_dim = order[encoded_dim];
    decoded[decoded_dim] = encoded[encoded_dim];
  }
}
}  // namespace

absl::Status TransposeCodecSpec::GetDecodedChunkLayout(
    const ArrayDataTypeAndShapeInfo& encoded_info,
    const ArrayCodecChunkLayoutInfo& encoded,
    const ArrayDataTypeAndShapeInfo& decoded_info,
    ArrayCodecChunkLayoutInfo& decoded) const {
  span<const DimensionIndex> order = options.order;
  assert(encoded_info.rank == order.size());
  assert(decoded_info.rank == order.size());
  PropagateInnerOrderToDecoded(order, encoded.inner_order, decoded.inner_order);
  PropagateShapeToDecoded(order, encoded.read_chunk_shape,
                          decoded.read_chunk_shape);
  PropagateShapeToDecoded(order, encoded.codec_chunk_shape,
                          decoded.codec_chunk_shape);
  return absl::OkStatus();
}

Result<ZarrArrayToArrayCodec::Ptr> TransposeCodecSpec::Resolve(
    ArrayCodecResolveParameters&& decoded, ArrayCodecResolveParameters& encoded,
    ZarrArrayToArrayCodecSpec::Ptr* resolved_spec) const {
  span<const DimensionIndex> order = options.order;
  if (!RankConstraint::Implies(order.size(), decoded.rank)) {
    return InvalidPermutationError(order, decoded.rank);
  }
  encoded.dtype = decoded.dtype;
  encoded.rank = decoded.rank;
  assert(decoded.fill_value.rank() == 0);
  encoded.fill_value = std::move(decoded.fill_value);
  std::vector<DimensionIndex> inverse_order(order.size());
  InvertPermutation(order.size(), order.data(), inverse_order.data());
  PropagateInnerOrderToDecoded(inverse_order, decoded.inner_order,
                               encoded.inner_order);
  PropagateShapeToDecoded(inverse_order, decoded.read_chunk_shape,
                          encoded.read_chunk_shape);
  PropagateShapeToDecoded(inverse_order, decoded.codec_chunk_shape,
                          encoded.codec_chunk_shape);
  if (resolved_spec) resolved_spec->reset(this);
  return internal::MakeIntrusivePtr<TransposeCodec>(std::move(inverse_order));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  using Self = TransposeCodecSpec;
  using Options = Self::Options;
  namespace jb = ::tensorstore::internal_json_binding;
  RegisterCodec<Self>(
      "transpose",
      jb::Projection<&Self::options>(jb::Sequence(jb::Member(
          "order",
          jb::Projection<&Options::order>(jb::Validate(
              [](const auto& options, auto* obj) {
                if (!IsValidPermutation(*obj)) {
                  return absl::InvalidArgumentError(
                      tensorstore::StrCat(span<const DimensionIndex>(*obj),
                                          " is not a valid permutation"));
                }
                return absl::OkStatus();
              },
              jb::DimensionIndexedVector(
                  nullptr, jb::Integer<DimensionIndex>(0, kMaxRank - 1))))))));
}

}  // namespace internal_zarr3
}  // namespace tensorstore
