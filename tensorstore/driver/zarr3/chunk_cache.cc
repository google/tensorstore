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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/array.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/batch.h"
#include "tensorstore/box.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/chunk_receiver_utils.h"
#include "tensorstore/driver/read_request.h"
#include "tensorstore/driver/write_request.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/chunk_cache.h"
#include "tensorstore/internal/cache/kvs_backed_chunk_cache.h"
#include "tensorstore/internal/chunk_grid_specification.h"
#include "tensorstore/internal/grid_partition.h"
#include "tensorstore/internal/grid_partition_iterator.h"
#include "tensorstore/internal/grid_storage_statistics.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/internal/meta/type_traits.h"
#include "tensorstore/internal/regular_grid.h"
#include "tensorstore/internal/storage_statistics.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/rank.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/flow_sender_operation_state.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_zarr3 {

ZarrChunkCache::~ZarrChunkCache() = default;

ZarrLeafChunkCache::ZarrLeafChunkCache(
    kvstore::DriverPtr store, ZarrCodecChain::PreparedState::Ptr codec_state,
    ZarrDType dtype, internal::CachePool::WeakPtr /*data_cache_pool*/,
    bool open_as_void, bool original_is_structured, DataType original_dtype)
    : Base(std::move(store)),
      codec_state_(std::move(codec_state)),
      dtype_(std::move(dtype)),
      open_as_void_(open_as_void),
      original_is_structured_(original_is_structured),
      original_dtype_(original_dtype) {}

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

Result<absl::InlinedVector<SharedArray<const void>, 1>>
ZarrLeafChunkCache::DecodeChunk(span<const Index> chunk_indices,
                                absl::Cord data) {
  const size_t num_fields = dtype_.fields.size();
  absl::InlinedVector<SharedArray<const void>, 1> field_arrays(num_fields);

  // Special case: void access - decode and return as bytes.
  //
  // For non-structured types: codec was prepared for [chunk_shape] with
  // original dtype. We decode to that shape then reinterpret as bytes.
  //
  // For structured types: codec was already prepared for
  // [chunk_shape, bytes_per_elem] with byte dtype. Just decode directly.
  if (open_as_void_) {
    assert(num_fields == 1);  // Void access uses a single synthesized field
    const auto& void_component_shape = grid().components[0].shape();

    if (original_is_structured_) {
      // Structured types: codec already expects bytes with extra dimension.
      // Just decode directly to the void component shape.
      TENSORSTORE_ASSIGN_OR_RETURN(
          field_arrays[0],
          codec_state_->DecodeArray(void_component_shape, std::move(data)));
      return field_arrays;
    }

    // Non-structured types: codec expects original dtype without extra
    // dimension. Decode, then reinterpret as bytes.
    const auto& void_chunk_shape = grid().chunk_shape;
    std::vector<Index> original_chunk_shape(
        void_chunk_shape.begin(),
        void_chunk_shape.end() - 1);  // Strip bytes dimension

    // Decode using original codec shape
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto decoded_array,
        codec_state_->DecodeArray(original_chunk_shape, std::move(data)));

    // Reinterpret the decoded array's bytes as [chunk_shape..., bytes_per_elem]
    auto byte_array = AllocateArray(
        void_component_shape, c_order, default_init,
        dtype_v<tensorstore::dtypes::byte_t>);

    // Copy decoded data to byte array (handles potential layout differences)
    std::memcpy(byte_array.data(), decoded_array.data(),
                decoded_array.num_elements() *
                    decoded_array.dtype().size());

    field_arrays[0] = std::move(byte_array);
    return field_arrays;
  }

  // For single non-structured field, decode directly
  if (num_fields == 1 && dtype_.fields[0].outer_shape.empty()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        field_arrays[0], codec_state_->DecodeArray(grid().components[0].shape(),
                                                   std::move(data)));
    return field_arrays;
  }

  // For structured types, decode byte array then extract fields
  // Build decode shape: [chunk_dims..., bytes_per_outer_element]
  const auto& chunk_shape = grid().chunk_shape;
  std::vector<Index> decode_shape(chunk_shape.begin(), chunk_shape.end());
  decode_shape.push_back(dtype_.bytes_per_outer_element);

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto byte_array, codec_state_->DecodeArray(decode_shape, std::move(data)));

  // Extract each field from the byte array
  const Index num_elements = byte_array.num_elements() /
                             dtype_.bytes_per_outer_element;
  const auto* src_bytes = static_cast<const std::byte*>(byte_array.data());

  for (size_t field_i = 0; field_i < num_fields; ++field_i) {
    const auto& field = dtype_.fields[field_i];
    // Use the component's shape (from the grid) for the result array
    const auto& component_shape = grid().components[field_i].shape();
    auto result_array =
        AllocateArray(component_shape, c_order, default_init, field.dtype);
    auto* dst = static_cast<std::byte*>(result_array.data());
    const Index field_size = field.dtype->size;

    // Copy field data from each struct element
    for (Index i = 0; i < num_elements; ++i) {
      std::memcpy(dst + i * field_size,
                  src_bytes + i * dtype_.bytes_per_outer_element +
                      field.byte_offset,
                  field_size);
    }
    field_arrays[field_i] = std::move(result_array);
  }

  return field_arrays;
}

Result<absl::Cord> ZarrLeafChunkCache::EncodeChunk(
    span<const Index> chunk_indices,
    span<const SharedArray<const void>> component_arrays) {
  const size_t num_fields = dtype_.fields.size();

  // Special case: void access - encode bytes back to original format.
  if (open_as_void_) {
    assert(component_arrays.size() == 1);

    if (original_is_structured_) {
      // Structured types: codec already expects bytes with extra dimension.
      return codec_state_->EncodeArray(component_arrays[0]);
    }

    // Non-structured types: reinterpret bytes as original dtype/shape.
    const auto& byte_array = component_arrays[0];
    const Index bytes_per_element = dtype_.bytes_per_outer_element;

    // Build original chunk shape by stripping the bytes dimension
    const auto& void_shape = byte_array.shape();
    std::vector<Index> original_shape(void_shape.begin(), void_shape.end() - 1);

    // Use the original dtype (stored during cache creation) for encoding.
    // Create a view over the byte data with original dtype and layout.
    // Use the aliasing constructor to share ownership with byte_array but
    // interpret the data with the original dtype.
    SharedArray<const void> encoded_array;
    auto aliased_ptr = std::shared_ptr<const void>(
        byte_array.pointer(),  // Share ownership with byte_array
        byte_array.data());    // But point to the raw data
    encoded_array.element_pointer() = SharedElementPointer<const void>(
        std::move(aliased_ptr), original_dtype_);
    encoded_array.layout() = StridedLayout<>(c_order, bytes_per_element,
                                             original_shape);

    return codec_state_->EncodeArray(encoded_array);
  }

  // For single non-structured field, encode directly
  if (num_fields == 1 && dtype_.fields[0].outer_shape.empty()) {
    assert(component_arrays.size() == 1);
    return codec_state_->EncodeArray(component_arrays[0]);
  }

  // For structured types, combine multiple field arrays into a single byte array
  assert(component_arrays.size() == num_fields);

  // Build encode shape: [chunk_dims..., bytes_per_outer_element]
  const auto& chunk_shape = grid().chunk_shape;
  std::vector<Index> encode_shape(chunk_shape.begin(), chunk_shape.end());
  encode_shape.push_back(dtype_.bytes_per_outer_element);

  // Calculate number of outer elements
  Index num_elements = 1;
  for (size_t i = 0; i < chunk_shape.size(); ++i) {
    num_elements *= chunk_shape[i];
  }

  // Allocate byte array for combined fields
  auto byte_array = AllocateArray<std::byte>(encode_shape, c_order, value_init);
  auto* dst_bytes = byte_array.data();

  // Copy each field's data into the byte array at their respective offsets
  for (size_t field_i = 0; field_i < num_fields; ++field_i) {
    const auto& field = dtype_.fields[field_i];
    const auto& field_array = component_arrays[field_i];
    const auto* src = static_cast<const std::byte*>(field_array.data());
    const Index field_size = field.dtype->size;

    // Copy field data to each struct element
    for (Index i = 0; i < num_elements; ++i) {
      std::memcpy(dst_bytes + i * dtype_.bytes_per_outer_element +
                      field.byte_offset,
                  src + i * field_size,
                  field_size);
    }
  }

  return codec_state_->EncodeArray(byte_array);
}

kvstore::Driver* ZarrLeafChunkCache::GetKvStoreDriver() {
  return this->internal::KvsBackedChunkCache::kvstore_driver();
}

ZarrShardedChunkCache::ZarrShardedChunkCache(
    kvstore::DriverPtr store, ZarrCodecChain::PreparedState::Ptr codec_state,
    ZarrDType dtype, internal::CachePool::WeakPtr data_cache_pool,
    bool open_as_void, bool original_is_structured, DataType original_dtype)
    : base_kvstore_(std::move(store)),
      codec_state_(std::move(codec_state)),
      dtype_(std::move(dtype)),
      open_as_void_(open_as_void),
      original_is_structured_(original_is_structured),
      original_dtype_(original_dtype),
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
       component_index = request.component_index,
       staleness_bound = request.staleness_bound,
       fill_missing_data_reads = request.fill_missing_data_reads](auto entry) {
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
                   component_index, staleness_bound, fill_missing_data_reads},
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
       component_index = request.component_index,
       store_data_equal_to_fill_value =
           request.store_data_equal_to_fill_value](auto entry) {
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
              {{shard_transaction, std::move(transform)}, component_index,
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
                cache.dtype_, cache.data_cache_pool_, cache.open_as_void_,
                cache.original_is_structured_, cache.original_dtype_);
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
