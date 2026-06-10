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

#include "tensorstore/driver/zarr3/chunk_cache.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/array.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/batch.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/chunk_receiver_utils.h"
#include "tensorstore/driver/read_request.h"
#include "tensorstore/driver/write_request.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/dtype.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/async_write_array.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/chunk_cache.h"
#include "tensorstore/internal/cache/kvs_backed_chunk_cache.h"
#include "tensorstore/internal/chunk_grid_specification.h"
#include "tensorstore/internal/data_type_endian_conversion.h"
#include "tensorstore/internal/grid_partition.h"
#include "tensorstore/internal/grid_partition_iterator.h"
#include "tensorstore/internal/grid_storage_statistics.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/internal/meta/type_traits.h"
#include "tensorstore/internal/regular_grid.h"
#include "tensorstore/internal/storage_statistics.h"
#include "tensorstore/internal/unaligned_data_type_functions.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/flow_sender_operation_state.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_zarr3 {

internal::ChunkGridSpecification CreateFieldGridSpecification(
    span<const Index> chunk_shape, const ZarrDType& zarr_dtype,
    span<const DimensionIndex> inner_order,
    const std::vector<SharedArray<const void>>* fill_values) {
  const DimensionIndex chunked_rank = chunk_shape.size();
  internal::ChunkGridSpecification::ComponentList components;
  components.reserve(zarr_dtype.fields.size());

  for (size_t field_i = 0; field_i < zarr_dtype.fields.size(); ++field_i) {
    const auto& field = zarr_dtype.fields[field_i];
    const size_t field_rank = field.field_shape.size();
    const DimensionIndex total_rank = chunked_rank + field_rank;

    SharedArray<const void> fill_value;
    if (fill_values && field_i < fill_values->size()) {
      fill_value = (*fill_values)[field_i];
    }
    if (!fill_value.valid()) {
      fill_value = AllocateArray(span<const Index, 0>{}, c_order, value_init,
                                 field.dtype);
    }

    std::vector<Index> target_shape(chunked_rank, kInfIndex);
    target_shape.insert(target_shape.end(), field.field_shape.begin(),
                        field.field_shape.end());

    auto chunk_fill_value =
        BroadcastArray(fill_value, BoxView<>(target_shape)).value();

    std::vector<Index> component_chunk_shape(chunk_shape.begin(),
                                             chunk_shape.end());
    component_chunk_shape.insert(component_chunk_shape.end(),
                                 field.field_shape.begin(),
                                 field.field_shape.end());

    std::vector<DimensionIndex> component_permutation(total_rank);
    if (!inner_order.empty()) {
      assert(inner_order.size() == chunked_rank);
      std::copy_n(inner_order.begin(), chunked_rank,
                  component_permutation.begin());
    } else {
      std::iota(component_permutation.begin(),
                component_permutation.begin() + chunked_rank, 0);
    }
    std::iota(component_permutation.begin() + chunked_rank,
              component_permutation.end(), chunked_rank);

    Box<> valid_data_bounds(total_rank);
    for (size_t i = 0; i < field_rank; ++i) {
      valid_data_bounds[chunked_rank + i] =
          IndexInterval::UncheckedSized(0, field.field_shape[i]);
    }

    std::vector<DimensionIndex> chunked_to_cell(chunked_rank);
    std::iota(chunked_to_cell.begin(), chunked_to_cell.end(), 0);

    auto& component = components.emplace_back(
        internal::AsyncWriteArray::Spec{
            std::move(chunk_fill_value), std::move(valid_data_bounds),
            ContiguousLayoutPermutation<>(component_permutation)},
        component_chunk_shape, std::move(chunked_to_cell));
    component.array_spec.fill_value_comparison_kind =
        EqualityComparisonKind::identical;
  }

  return internal::ChunkGridSpecification(std::move(components));
}

std::string FieldKeyParserWrapper::FormatKey(
    span<const Index> grid_indices) const {
  std::vector<Index> padded(grid_indices.begin(), grid_indices.end());
  while (static_cast<DimensionIndex>(padded.size()) < full_rank_) {
    padded.push_back(0);
  }
  return inner_.FormatKey(padded);
}

Index FieldKeyParserWrapper::MinGridIndexForLexicographicalOrder(
    DimensionIndex dim, IndexInterval grid_interval) const {
  return inner_.MinGridIndexForLexicographicalOrder(dim, grid_interval);
}

bool FieldKeyParserWrapper::ParseKey(std::string_view key,
                                     span<Index> grid_indices) const {
  std::vector<Index> full_indices(full_rank_);
  if (!inner_.ParseKey(key, full_indices)) return false;
  const ptrdiff_t n = std::min(grid_indices.size(),
                               static_cast<ptrdiff_t>(full_indices.size()));
  std::copy_n(full_indices.begin(), n, grid_indices.begin());
  std::fill(grid_indices.begin() + n, grid_indices.end(), 0);
  return true;
}

ZarrChunkCache::~ZarrChunkCache() = default;

ZarrLeafChunkCache::ZarrLeafChunkCache(
    kvstore::DriverPtr store, ZarrCodecChain::PreparedState::Ptr codec_state,
    ZarrDType zarr_dtype, std::vector<Index> field_shape,
    std::vector<DimensionIndex> inner_order,
    std::vector<SharedArray<const void>> fill_value, endian codec_endian,
    internal::CachePool::WeakPtr /*data_cache_pool*/)
    : Base(std::move(store)),
      codec_state_(std::move(codec_state)),
      zarr_dtype_(std::move(zarr_dtype)),
      field_shape_(std::move(field_shape)),
      inner_order_(std::move(inner_order)),
      fill_value_(std::move(fill_value)),
      codec_endian_(codec_endian) {}

void ZarrLeafChunkCache::Read(ZarrChunkCache::ReadRequest request,
                              AnyFlowReceiver<absl::Status, internal::ReadChunk,
                                              IndexTransform<>>&& receiver) {
  return internal::ChunkCache::Read(
      {static_cast<internal::DriverReadRequest&&>(request),
       request.component_index, request.staleness_bound,
       request.fill_missing_data_reads},
      std::move(receiver));
}

void ZarrLeafChunkCache::Write(
    ZarrChunkCache::WriteRequest request,
    AnyFlowReceiver<absl::Status, internal::WriteChunk, IndexTransform<>>&&
        receiver) {
  return internal::ChunkCache::Write(
      {static_cast<internal::DriverWriteRequest&&>(request),
       request.component_index, request.store_data_equal_to_fill_value},
      std::move(receiver));
}

struct GridStorageStatisticsChunkHandlerBase
    : public internal::GridStorageStatisticsChunkHandler {
  internal::CachePtr<ZarrChunkCache> cache;

  static void Start(
      internal::IntrusivePtr<GridStorageStatisticsChunkHandlerBase> handler,
      ZarrChunkCache& cache,
      internal::IntrusivePtr<internal::GetStorageStatisticsAsyncOperationState>
          state,
      ZarrChunkCache::GetStorageStatisticsRequest request) {
    handler->state = std::move(state);
    handler->cache.reset(&cache);
    const auto& grid = cache.grid();
    const auto& component = grid.components[0];
    handler->grid_output_dimensions = component.chunked_to_cell_dimensions;
    handler->key_formatter = &cache.GetChunkStorageKeyParser();
    const DimensionIndex rank = component.rank();
    assert(rank == request.shape.size());
    span<const Index> chunk_shape = grid.chunk_shape;
    Box<dynamic_rank(kMaxRank)> grid_bounds(rank);
    for (DimensionIndex i = 0; i < rank; ++i) {
      const Index grid_size = CeilOfRatio(request.shape[i], chunk_shape[i]);
      grid_bounds[i] = IndexInterval::UncheckedSized(0, grid_size);
    }
    handler->chunk_shape = chunk_shape;
    handler->full_transform = std::move(request.transform);
    internal::GetStorageStatisticsForRegularGridWithSemiLexicographicalKeys(
        std::move(handler),
        kvstore::KvStore{kvstore::DriverPtr(cache.GetKvStoreDriver()),
                         internal::TransactionState::ToTransaction(
                             std::move(request.transaction))},
        grid_bounds, request.staleness_bound);
  }
};

void ZarrLeafChunkCache::GetStorageStatistics(
    internal::IntrusivePtr<internal::GetStorageStatisticsAsyncOperationState>
        state,
    ZarrChunkCache::GetStorageStatisticsRequest request) {
  auto handler =
      internal::MakeIntrusivePtr<GridStorageStatisticsChunkHandlerBase>();
  GridStorageStatisticsChunkHandlerBase::Start(
      std::move(handler), *this, std::move(state), std::move(request));
}

std::string ZarrLeafChunkCache::GetChunkStorageKey(
    span<const Index> cell_indices) {
  return GetChunkStorageKeyParser().FormatKey(cell_indices);
}

namespace {

// Attempts to reference `field` within the decoded `byte_array` directly,
// without copying, returning a null array if that is not possible.
//
// This mirrors `internal::TryViewCordAsArray` used by the zarr v2 driver: an
// alias-only view is possible only when no endian conversion is required and
// the field data is suitably aligned for its data type.  `field_byte_strides`
// is the strided layout (over `byte_array`) of the field, including the
// preferred inner order inherited from `byte_array`.
SharedArray<const void> TryViewFieldArray(
    const SharedArray<const void>& byte_array, const ZarrDType::Field& field,
    endian codec_endian, span<const Index> field_shape,
    span<const Index> field_byte_strides) {
  const auto& functions =
      internal::kUnalignedDataTypeFunctions[static_cast<size_t>(
          field.dtype.id())];
  assert(functions.copy != nullptr);  // fail on non-trivial types
  if (codec_endian != endian::native && functions.swap_endian_inplace) {
    // Field data requires endian conversion.
    return {};
  }
  auto field_pointer =
      AddByteOffset(byte_array.element_pointer(), field.byte_offset);
  const size_t alignment = field.dtype->alignment;
  if ((reinterpret_cast<uintptr_t>(field_pointer.data()) % alignment) != 0 ||
      !std::all_of(
          field_byte_strides.begin(), field_byte_strides.end(),
          [&](Index byte_stride) { return (byte_stride % alignment) == 0; })) {
    // Field data is not suitably aligned for its data type.
    return {};
  }
  return SharedArray<const void>(
      SharedElementPointer<const void>(std::move(field_pointer).pointer(),
                                       field.dtype),
      StridedLayout<>(field_shape, field_byte_strides));
}

}  // namespace

Result<absl::InlinedVector<SharedArray<const void>, 1>>
ZarrLeafChunkCache::DecodeChunk(span<const Index> chunk_indices,
                                absl::Cord data) {
  const size_t num_fields = zarr_dtype_.fields.size();
  absl::InlinedVector<SharedArray<const void>, 1> field_arrays(num_fields);

  if (field_shape_.empty()) {
    assert(num_fields == 1);
    TENSORSTORE_ASSIGN_OR_RETURN(
        field_arrays[0], codec_state_->DecodeArray(grid().components[0].shape(),
                                                   std::move(data)));
    return field_arrays;
  }

  const auto& chunk_shape = grid().chunk_shape;
  absl::InlinedVector<Index, kMaxRank> decode_shape(chunk_shape.begin(),
                                                    chunk_shape.end());
  decode_shape.insert(decode_shape.end(), field_shape_.begin(),
                      field_shape_.end());

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto byte_array,
      codec_state_->DecodeArray(decode_shape, std::move(data)));

  for (size_t field_i = 0; field_i < num_fields; ++field_i) {
    const auto& field = zarr_dtype_.fields[field_i];
    const auto& component = grid().components[field_i];

    absl::InlinedVector<Index, kMaxRank> view_shape(chunk_shape.begin(),
                                                    chunk_shape.end());
    view_shape.insert(view_shape.end(), field.field_shape.begin(),
                      field.field_shape.end());

    // The outer (chunked) dimensions inherit the byte strides of the decoded
    // array, which already reflect the preferred inner order; the inner field
    // dimensions are contiguous within each outer element.
    absl::InlinedVector<Index, kMaxRank> src_byte_strides(view_shape.size());
    std::copy_n(byte_array.byte_strides().begin(), chunk_shape.size(),
                src_byte_strides.begin());
    if (!field.field_shape.empty()) {
      ComputeStrides(
          c_order, static_cast<Index>(field.dtype.size()), field.field_shape,
          tensorstore::span(src_byte_strides.data() + chunk_shape.size(),
                            field.field_shape.size()));
    }

    // Reference the decoded bytes directly when possible, as the zarr v2 driver
    // does, falling back to allocating a copy in the preferred inner order.
    if (auto field_array = TryViewFieldArray(byte_array, field, codec_endian_,
                                             view_shape, src_byte_strides);
        field_array.valid()) {
      field_arrays[field_i] = std::move(field_array);
    } else {
      auto result_array =
          AllocateArray(component.shape(), component.array_spec.layout_order(),
                        default_init, field.dtype);
      ArrayView<const void> src_field_view(
          {static_cast<const void*>(
               static_cast<const std::byte*>(byte_array.data()) +
               field.byte_offset),
           field.dtype},
          StridedLayoutView<>(view_shape, src_byte_strides));
      internal::DecodeArray(src_field_view, codec_endian_, result_array);
      field_arrays[field_i] = std::move(result_array);
    }
  }

  return field_arrays;
}

Result<absl::Cord> ZarrLeafChunkCache::EncodeChunk(
    span<const Index> chunk_indices,
    span<const SharedArray<const void>> component_arrays) {
  const size_t num_fields = zarr_dtype_.fields.size();

  if (field_shape_.empty()) {
    assert(num_fields == 1);
    assert(component_arrays.size() == 1);
    return codec_state_->EncodeArray(component_arrays[0]);
  }

  assert(component_arrays.size() == num_fields);

  const auto& chunk_shape = grid().chunk_shape;
  absl::InlinedVector<Index, kMaxRank> encode_shape(chunk_shape.begin(),
                                                    chunk_shape.end());
  encode_shape.insert(encode_shape.end(), field_shape_.begin(),
                      field_shape_.end());

  auto byte_array = AllocateArray<std::byte>(encode_shape, c_order, value_init);

  for (size_t field_i = 0; field_i < num_fields; ++field_i) {
    const auto& field = zarr_dtype_.fields[field_i];
    const auto& field_array = component_arrays[field_i];

    absl::InlinedVector<Index, kMaxRank> view_shape(chunk_shape.begin(),
                                                    chunk_shape.end());
    view_shape.insert(view_shape.end(), field.field_shape.begin(),
                      field.field_shape.end());

    absl::InlinedVector<Index, kMaxRank> dest_byte_strides(view_shape.size());
    ComputeStrides(
        c_order, zarr_dtype_.bytes_per_outer_element, chunk_shape,
        tensorstore::span(dest_byte_strides.data(), chunk_shape.size()));
    if (!field.field_shape.empty()) {
      ComputeStrides(
          c_order, static_cast<Index>(field.dtype.size()), field.field_shape,
          tensorstore::span(dest_byte_strides.data() + chunk_shape.size(),
                            field.field_shape.size()));
    }

    ArrayView<void> dest_field_view(
        {static_cast<void*>(byte_array.data() + field.byte_offset),
         field.dtype},
        StridedLayoutView<>(view_shape, dest_byte_strides));

    internal::EncodeArray(field_array, dest_field_view, codec_endian_);
  }

  return codec_state_->EncodeArray(byte_array);
}

kvstore::Driver* ZarrLeafChunkCache::GetKvStoreDriver() {
  return this->internal::KvsBackedChunkCache::kvstore_driver();
}

ZarrShardedChunkCache::ZarrShardedChunkCache(
    kvstore::DriverPtr store, ZarrCodecChain::PreparedState::Ptr codec_state,
    ZarrDType zarr_dtype, std::vector<Index> field_shape,
    std::vector<DimensionIndex> inner_order,
    std::vector<SharedArray<const void>> fill_value, endian codec_endian,
    internal::CachePool::WeakPtr data_cache_pool)
    : base_kvstore_(std::move(store)),
      codec_state_(std::move(codec_state)),
      zarr_dtype_(std::move(zarr_dtype)),
      field_shape_(std::move(field_shape)),
      inner_order_(std::move(inner_order)),
      fill_value_(std::move(fill_value)),
      codec_endian_(codec_endian),
      data_cache_pool_(std::move(data_cache_pool)) {}

Result<IndexTransform<>> TranslateCellToSourceTransformForShard(
    IndexTransform<> transform, span<const Index> grid_cell_indices,
    const internal::ChunkGridSpecification& grid) {
  span<const Index> chunk_shape = grid.chunk_shape;
  const auto& component_spec = grid.components[0];
  span<const DimensionIndex> chunked_to_cell_dimensions =
      component_spec.chunked_to_cell_dimensions;
  Index offsets[kMaxRank];
  const DimensionIndex output_rank = transform.output_rank();
  std::fill_n(offsets, output_rank, Index(0));
  for (DimensionIndex grid_dim = 0; grid_dim < grid_cell_indices.size();
       ++grid_dim) {
    offsets[chunked_to_cell_dimensions[grid_dim]] =
        -grid_cell_indices[grid_dim] * chunk_shape[grid_dim];
  }
  return TranslateOutputDimensionsBy(std::move(transform),
                                     span(&offsets[0], output_rank));
}

template <typename Receiver, typename BaseFunc, typename CodecFunc>
void ShardedInvokeWithArrayToArrayCodecs(
    ZarrShardedChunkCache& self, BaseFunc base_func, CodecFunc codec_func,
    IndexTransform<> transform,
    internal::type_identity_t<Receiver>&& receiver) {
  const auto& grid = self.grid();
  span<const Index> chunk_shape = grid.chunk_shape;
  const span<const ZarrArrayToArrayCodec::PreparedState::Ptr>
      array_to_array_codec_states = self.codec_state_->array_to_array;
  if (array_to_array_codec_states.empty()) {
    base_func(chunk_shape, std::move(transform),
              std::forward<Receiver>(receiver));
    return;
  }

  span<const Index> transformed_chunk_shape =
      array_to_array_codec_states.empty()
          ? chunk_shape
          : array_to_array_codec_states.back()->encoded_shape();
  // Define the inner-most `next` function that operates on the actual
  // cached chunk.
  std::function<void(IndexTransform<> transform, Receiver receiver)> next =
      [=, base_func = std::move(base_func)](IndexTransform<> transform,
                                            Receiver&& receiver) {
#ifndef NDEBUG
        // Debug sanity check: Validate that the output range of
        // `transform` is restricted to the shard bounds.
        TENSORSTORE_CHECK_OK_AND_ASSIGN(
            transform,
            PropagateExplicitBoundsToTransform(
                BoxView<>(transformed_chunk_shape), std::move(transform)));
#endif
        base_func(transformed_chunk_shape, std::move(transform),
                  std::move(receiver));
      };
  // Apply the "array -> array" codecs to `next` from innermost to
  // outermost.
  for (size_t codec_i = array_to_array_codec_states.size(); codec_i--;) {
    auto* codec = array_to_array_codec_states[codec_i].get();
    span<const Index> cur_decoded_shape =
        codec_i == 0
            ? chunk_shape
            : array_to_array_codec_states[codec_i - 1]->encoded_shape();
    next = [next = std::move(next), codec, cur_decoded_shape, codec_func](
               IndexTransform<> transform, Receiver receiver) {
#ifndef NDEBUG
      // Debug sanity check: Validate that the output range of
      // `transform` is restricted to the expected decoded shape for
      // this codec.
      TENSORSTORE_CHECK_OK_AND_ASSIGN(
          transform, PropagateExplicitBoundsToTransform(
                         BoxView<>(cur_decoded_shape), std::move(transform)));
#endif
      codec_func(*codec, next, cur_decoded_shape, std::move(transform),
                 std::move(receiver));
    };
  }
  next(std::move(transform), std::move(receiver));
}

template <typename ChunkType, auto CodecMethod, typename GetBaseFunc>
void ShardedReadOrWrite(
    ZarrShardedChunkCache& self, IndexTransform<> transform,
    AnyFlowReceiver<absl::Status, ChunkType, IndexTransform<>> receiver,
    GetBaseFunc get_base_func) {
  const auto& grid = self.grid();
  const auto& component_spec = grid.components[0];

  using State = internal::FlowSenderOperationState<ChunkType, IndexTransform<>>;
  using ForwardingReceiver =
      internal::ForwardingChunkOperationReceiver<ChunkType, State>;
  span<const Index> chunk_shape = grid.chunk_shape;
  span<const DimensionIndex> chunked_to_cell_dimensions =
      component_spec.chunked_to_cell_dimensions;
  auto state = internal::MakeIntrusivePtr<State>(std::move(receiver));
  assert(chunked_to_cell_dimensions.size() == chunk_shape.size());

  auto status = [&]() -> absl::Status {
    internal_grid_partition::RegularGridRef regular_grid{chunk_shape};
    internal_grid_partition::PartitionIndexTransformIterator iterator(
        chunked_to_cell_dimensions, regular_grid, transform);
    TENSORSTORE_RETURN_IF_ERROR(iterator.Init());

    while (!iterator.AtEnd()) {
      if (state->cancelled()) {
        return absl::CancelledError("");
      }
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto cell_to_source,
          ComposeTransforms(transform, iterator.cell_transform()));
      TENSORSTORE_ASSIGN_OR_RETURN(
          cell_to_source, TranslateCellToSourceTransformForShard(
                              std::move(cell_to_source),
                              iterator.output_grid_cell_indices(), grid));
      auto entry =
          GetEntryForGridCell(self, iterator.output_grid_cell_indices());
      if (!entry->sharding_error.ok()) {
        return entry->sharding_error;
      }
      using Receiver =
          AnyFlowReceiver<absl::Status, ChunkType, IndexTransform<>>;
      ShardedInvokeWithArrayToArrayCodecs<Receiver&&>(
          self,
          /*base_func=*/get_base_func(std::move(entry)),
          /*codec_func=*/
          [](const ZarrArrayToArrayCodec::PreparedState& codec_state,
             const std::function<void(IndexTransform<>, Receiver&&)>& next,
             span<const Index> decoded_shape, IndexTransform<> transform,
             Receiver&& receiver) {
            (codec_state.*CodecMethod)(
                next, decoded_shape, std::move(transform), std::move(receiver));
          },
          std::move(cell_to_source),
          ForwardingReceiver{state, iterator.cell_transform()});
      iterator.Advance();
    }
    return absl::OkStatus();
  }();
  if (!status.ok()) {
    state->SetError(status);
  }
}

void ZarrShardedChunkCache::Read(
    ZarrChunkCache::ReadRequest request,
    AnyFlowReceiver<absl::Status, internal::ReadChunk, IndexTransform<>>&&
        receiver) {
  ShardedReadOrWrite<internal::ReadChunk,
                     &ZarrArrayToArrayCodec::PreparedState::Read>(
      *this, std::move(request.transform), std::move(receiver),
      [transaction = std::move(request.transaction),
       batch = std::move(request.batch),
       staleness_bound = request.staleness_bound,
       fill_missing_data_reads = request.fill_missing_data_reads,
       component_index = request.component_index](auto entry) {
        Batch shard_batch = batch;
        if (!shard_batch) {
          shard_batch = Batch::New();
        }
        return
            [=, shard_batch = std::move(shard_batch), entry = std::move(entry)](
                span<const Index> decoded_shape, IndexTransform<> transform,
                AnyFlowReceiver<absl::Status, internal::ReadChunk,
                                IndexTransform<>>&& receiver) {
              entry->sub_chunk_cache.get()->Read(
                  {{transaction, std::move(transform), shard_batch},
                   component_index,
                   staleness_bound,
                   fill_missing_data_reads},
                  std::move(receiver));
            };
      });
}

void ZarrShardedChunkCache::Write(
    ZarrChunkCache::WriteRequest request,
    AnyFlowReceiver<absl::Status, internal::WriteChunk, IndexTransform<>>&&
        receiver) {
  ShardedReadOrWrite<internal::WriteChunk,
                     &ZarrArrayToArrayCodec::PreparedState::Write>(
      *this, std::move(request.transform), std::move(receiver),
      [transaction = std::move(request.transaction),
       store_data_equal_to_fill_value = request.store_data_equal_to_fill_value,
       component_index = request.component_index](auto entry) {
        internal::OpenTransactionPtr shard_transaction = transaction;
        if (!shard_transaction) {
          shard_transaction = internal::TransactionState::MakeImplicit();
          shard_transaction->RequestCommit();
        }
        return [=, entry = std::move(entry)](
                   span<const Index> decoded_shape, IndexTransform<> transform,
                   AnyFlowReceiver<absl::Status, internal::WriteChunk,
                                   IndexTransform<>>&& receiver) {
          entry->sub_chunk_cache.get()->Write(
              {{shard_transaction, std::move(transform)},
               component_index,
               store_data_equal_to_fill_value},
              std::move(receiver));
        };
      });
}

struct ShardedGridStorageStatisticsChunkHandler
    : public GridStorageStatisticsChunkHandlerBase {
  internal::OpenTransactionPtr transaction;
  absl::Time staleness_bound;
  void ChunkPresent(span<const Index> grid_indices) final {
    auto cell_transform = this->grid_partition.GetCellTransform(
        this->full_transform, grid_indices, this->grid_output_dimensions,
        [&](DimensionIndex grid_dim, Index grid_cell_index) -> IndexInterval {
          return internal_grid_partition::RegularGridRef{this->chunk_shape}
              .GetCellOutputInterval(grid_dim, grid_cell_index);
        });
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto cell_to_source,
        ComposeTransforms(this->full_transform, std::move(cell_transform)),
        state->SetError(_));
    const auto& grid = cache->grid();
    TENSORSTORE_ASSIGN_OR_RETURN(
        cell_to_source,
        TranslateCellToSourceTransformForShard(std::move(cell_to_source),
                                               grid_indices, grid),
        state->SetError(_));
    Box<dynamic_rank(kMaxRank)> output_range(cell_to_source.output_rank());
    TENSORSTORE_ASSIGN_OR_RETURN(bool output_range_exact,
                                 GetOutputRange(cell_to_source, output_range),
                                 state->SetError(_));
    span<const Index> cell_shape = grid.components[0].shape();
    if (output_range_exact && Contains(output_range, BoxView<>(cell_shape)) &&
        !(state->options.mask & ArrayStorageStatistics::query_fully_stored)) {
      // No need to query sub-chunks.
      state->IncrementChunksPresent();
      return;
    }

    // Query sub-chunks.
    auto entry = GetEntryForGridCell(
        static_cast<ZarrShardedChunkCache&>(*cache), grid_indices);
    if (!entry->sharding_error.ok()) {
      state->SetError(entry->sharding_error);
      return;
    }

    using StatePtr = internal::IntrusivePtr<
        internal::GetStorageStatisticsAsyncOperationState>;
    ShardedInvokeWithArrayToArrayCodecs<StatePtr>(
        static_cast<ZarrShardedChunkCache&>(*cache),
        /*base_func=*/
        [=, entry = std::move(entry)](span<const Index> decoded_shape,
                                      IndexTransform<> transform,
                                      StatePtr state) {
          entry->sub_chunk_cache->GetStorageStatistics(
              std::move(state), {transaction, decoded_shape,
                                 std::move(transform), staleness_bound});
        },
        /*codec_func=*/
        [](const ZarrArrayToArrayCodec::PreparedState& codec_state,
           const std::function<void(IndexTransform<>, StatePtr)>& next,
           span<const Index> decoded_shape, IndexTransform<> transform,
           StatePtr state) {
          codec_state.GetStorageStatistics(
              next, decoded_shape, std::move(transform), std::move(state));
        },
        std::move(cell_to_source), StatePtr(state));

    state->total_chunks -= 1;
  }
};

void ZarrShardedChunkCache::GetStorageStatistics(
    internal::IntrusivePtr<internal::GetStorageStatisticsAsyncOperationState>
        state,
    ZarrChunkCache::GetStorageStatisticsRequest request) {
  auto handler =
      internal::MakeIntrusivePtr<ShardedGridStorageStatisticsChunkHandler>();
  handler->transaction = request.transaction;
  handler->staleness_bound = request.staleness_bound;
  GridStorageStatisticsChunkHandlerBase::Start(
      std::move(handler), *this, std::move(state), std::move(request));
}

void ZarrShardedChunkCache::Entry::DoInitialize() {
  auto& cache = GetOwningCache(*this);
  if (cache.parent_chunk_) {
    parent_chunk = cache.parent_chunk_->AcquireWeakReference();
  }
  const auto& sharding_state = cache.sharding_codec_state();

  auto sharding_kvstore = sharding_state.GetSubChunkKvstore(
      cache.base_kvstore_,
      cache.GetChunkStorageKeyParser().FormatKey(cell_indices()),
      cache.executor(), internal::CachePool::WeakPtr(cache.pool()));
  ZarrChunkCache* zarr_chunk_cache;
  internal::GetCache<internal::Cache>(
      // `cache.pool()` is the metadata cache pool, which may or may not be
      // equal to the data cache pool. If the sub-chunk cache is a leaf cache
      // (no further sharding), then create the sub-chunk cache in the data
      // cache pool. Otherwise, create the sub-chunk cache in the metadata cache
      // pool.
      sharding_state.sub_chunk_codec_chain->is_sharding_chain()
          ? cache.pool()
          : cache.data_cache_pool_.get(),
      "",
      [&]() -> std::unique_ptr<internal::Cache> {
        auto new_cache =
            internal_zarr3::MakeZarrChunkCache<ZarrChunkCache,
                                               ZarrShardSubChunkCache>(
                *sharding_state.sub_chunk_codec_chain,
                std::move(sharding_kvstore), cache.executor(),
                ZarrShardingCodec::PreparedState::Ptr(&sharding_state),
                cache.zarr_dtype_, cache.field_shape_, cache.inner_order_,
                cache.fill_value_, cache.codec_endian_, cache.data_cache_pool_);
        zarr_chunk_cache = new_cache.release();
        return std::unique_ptr<internal::Cache>(&zarr_chunk_cache->cache());
      })
      .release();
  sub_chunk_cache =
      ZarrChunkCache::Ptr(zarr_chunk_cache, internal::adopt_object_ref);
  sub_chunk_cache->parent_chunk_ = this;
}

kvstore::Driver* ZarrShardedChunkCache::GetKvStoreDriver() {
  return this->base_kvstore_.get();
}

Future<const void> ZarrShardedChunkCache::DeleteCell(
    span<const Index> grid_cell_indices,
    internal::OpenTransactionPtr transaction) {
  auto entry = GetEntryForGridCell(*this, grid_cell_indices);
  return kvstore::DeleteRange(entry->sub_chunk_cache->GetKvStoreDriver(),
                              transaction, KeyRange{});
}

}  // namespace internal_zarr3
}  // namespace tensorstore
