// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/virtual_chunked.h"

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/box.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/cache/chunk_cache.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/serialization/absl_time.h"
#include "tensorstore/serialization/std_optional.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/garbage_collection/std_optional.h"

namespace tensorstore {
namespace virtual_chunked {

namespace {

class VirtualChunkedCache : public internal::ChunkCache {
 public:
  using internal::ChunkCache::ChunkCache;

  /// Common implementation used by `Entry::DoRead` and
  /// `TransactionNode::DoRead`.
  template <typename EntryOrNode>
  void DoRead(EntryOrNode& node, absl::Time staleness_bound);

  class Entry : public internal::ChunkCache::Entry {
   public:
    using OwningCache = VirtualChunkedCache;
    using internal::ChunkCache::Entry::Entry;
    void DoRead(absl::Time staleness_bound) override {
      GetOwningCache(*this).DoRead(*this, staleness_bound);
    }
  };
  class TransactionNode : public internal::ChunkCache::TransactionNode {
   public:
    using OwningCache = VirtualChunkedCache;
    using internal::ChunkCache::TransactionNode::TransactionNode;

    std::atomic<bool> marked_as_terminal_{false};

    absl::Status DoInitialize(
        internal::OpenTransactionPtr& transaction) override {
      SetReadsCommitted();
      return internal::ChunkCache::TransactionNode::DoInitialize(transaction);
    }

    absl::Status OnModified() override {
      // Ensure a transaction in atomic mode cannot be used to write to more
      // than one chunk.
      if (!marked_as_terminal_.exchange(true, std::memory_order_acq_rel)) {
        return this->MarkAsTerminal();
      }
      return absl::OkStatus();
    }

    std::string Describe() override;

    void DoRead(absl::Time staleness_bound) override {
      GetOwningCache(*this).DoRead(*this, staleness_bound);
    }

    void Commit() override;

    /// Attempts or re-attempts writeback.
    ///
    /// Integrates changes with existing data that is no older than
    /// `staleness_bound`.  For the first call `staleness_bound` should be
    /// `absl::InfinitePast()` to use any cached data, and then if writeback
    /// fails due to a mismatch, this should be re-invoked with a newer
    /// `staleness_bound`.
    void InitiateWriteback(absl::Time staleness_bound);
  };
  Entry* DoAllocateEntry() final { return new Entry; }
  std::size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  TransactionNode* DoAllocateTransactionNode(
      internal::AsyncCache::Entry& entry) final {
    return new TransactionNode(static_cast<Entry&>(entry));
  }

  // In order to support `inner_order`, given that `ChunkCache` always uses
  // lexicographic (C-order) with respect to the specified component dimensions,
  // we permute the component dimensions, and correct for that in the index
  // transform returned from `MakeDriver`.
  //
  // The dimension indices used by the ChunkCache are referred to as "component"
  // dimensions, while indices into the user-specified domain are "external"
  // dimensions (they correspond to the input space of the index transform
  // returned by `MakeDriver`).

  // Indexed by `external_dim`.
  std::vector<Index> grid_origin_for_read_function_;

  // Indexed by `component_dim`.
  DimensionUnitsVector dimension_units_;

  // Indexed by `component_dim`.
  std::vector<DimensionIndex> inner_order_;

  ReadFunction read_function_;

  WriteFunction write_function_;

  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency_;
  Context::Resource<internal::CachePoolResource> cache_pool_;
};

/// Sets `partial_array` to refer to the portion of `full_array` (translated to
/// the chunk origin) that is within bounds for the chunk corresponding to
/// `entry`.  Also permutes the dimensions according to
/// `VirtualChunkedCache::inner_order_`.
///
/// \param entry Entry corresponding to the chunk.
/// \param full_array Array of shape equal to the component chunk shape.
/// \param partial_array[out] Set to the portion of `full_array` corresponding
///     to `entry`, indexed by "external" dimensions.
bool GetPermutedPartialArray(
    VirtualChunkedCache::Entry& entry, ArrayView<const void> full_array,
    Array<const void, dynamic_rank, offset_origin>& partial_array) {
  auto& cache = static_cast<VirtualChunkedCache&>(GetOwningCache(entry));
  const auto& component_spec = cache.grid().components.front();
  const DimensionIndex rank = component_spec.rank();
  span<const Index> cell_shape = component_spec.shape();
  span<const Index> cell_indices = entry.cell_indices();
  span<const DimensionIndex> inner_order = cache.inner_order_;
  span<const Index> grid_origin_for_read_function =
      cache.grid_origin_for_read_function_;
  BoxView<> domain_bounds = component_spec.component_bounds;
  partial_array.layout().set_rank(rank);
  ByteStridedPointer<const void> data = full_array.byte_strided_pointer();
  for (DimensionIndex component_dim = 0; component_dim < rank;
       ++component_dim) {
    const DimensionIndex external_dim = inner_order[component_dim];
    const Index byte_stride = full_array.byte_strides()[component_dim];
    partial_array.byte_strides()[external_dim] = byte_stride;
    Index grid_origin_value = grid_origin_for_read_function[external_dim];
    Index chunk_start = cell_indices[component_dim] * cell_shape[component_dim];
    Index chunk_end = chunk_start + cell_shape[component_dim];
    Index request_start =
        std::max(chunk_start, domain_bounds.origin()[component_dim]);
    Index request_end =
        std::min(chunk_end, domain_bounds[component_dim].exclusive_max());
    if (request_start >= request_end) {
      // Chunk is entirely outside the domain.  This should not normally
      // happen.  No data needs to be filled in this case.
      return false;
    }
    partial_array.origin()[external_dim] = request_start + grid_origin_value;
    partial_array.shape()[external_dim] = request_end - request_start;
    data -= internal::wrap_on_overflow::Multiply(
        byte_stride, chunk_start + grid_origin_value);
  }
  partial_array.element_pointer() =
      ElementPointer<const void>(data, full_array.dtype());
  return true;
}

template <typename EntryOrNode>
void VirtualChunkedCache::DoRead(EntryOrNode& node,
                                 absl::Time staleness_bound) {
  auto& cache = GetOwningCache(node);
  if (!cache.read_function_) {
    // Normally happens only in the case of a partial chunk write.
    node.ReadError(absl::InvalidArgumentError(
        "Write-only virtual chunked view requires chunk-aligned writes"));
    return;
  }
  auto& executor = cache.executor();
  // `node` is guaranteed to remain valid until `ReadSuccess` or `ReadError`
  // is called.  Therefore we don't need to separately hold a reference.
  executor([&node, staleness_bound] {
    auto& entry = GetOwningEntry(node);
    auto& cache = GetOwningCache(entry);
    const auto& component_spec = cache.grid().components.front();
    span<const Index> cell_shape = component_spec.shape();
    // Always allocate the full chunk size, since that is what `ChunkCache`
    // requires.
    auto full_array = AllocateArray(cell_shape, c_order, default_init,
                                    component_spec.dtype());
    // Sub-region of `full_array` that intersects the domain.  The
    // user-specified `read_function` is called with `partial_array`.  The
    // portion of `full_array` that is outside the domain remains
    // uninitialized and is never read.
    Array<const void, dynamic_rank, offset_origin> partial_array;
    auto read_data =
        tensorstore::internal::make_shared_for_overwrite<ReadData[]>(1);
    if (!GetPermutedPartialArray(entry, full_array, partial_array)) {
      node.ReadSuccess(
          {std::move(read_data),
           {StorageGeneration::NoValue(), absl::InfiniteFuture()}});
      return;
    }
    read_data.get()[0] = SharedArrayView<void>(
        std::move(full_array.element_pointer()), component_spec.write_layout());
    ReadParameters read_params;
    read_params.executor_ = cache.executor();
    {
      ReadLock<ReadData> lock{node};
      read_params.if_not_equal_ = lock.stamp().generation;
    }
    read_params.staleness_bound_ = staleness_bound;
    auto read_future =
        cache.read_function_(ConstDataTypeCast<void>(std::move(partial_array)),
                             std::move(read_params));
    read_future.Force();
    read_future.ExecuteWhenReady(
        [&node, read_data = std::move(read_data)](
            ReadyFuture<TimestampedStorageGeneration> future) mutable {
          auto& r = future.result();
          if (!r.ok()) {
            node.ReadError(std::move(r).status());
            return;
          }
          if (StorageGeneration::IsUnknown(r->generation)) {
            // Ignore read_data
            ReadState read_state;
            {
              ReadLock<ReadData> lock{node};
              read_state = lock.read_state();
            }
            read_state.stamp.time = r->time;
            node.ReadSuccess(std::move(read_state));
            return;
          }
          node.ReadSuccess({std::move(read_data), std::move(*r)});
          return;
        });
  });
}

std::string VirtualChunkedCache::TransactionNode::Describe() {
  auto& entry = GetOwningEntry(*this);
  auto& cache = GetOwningCache(entry);
  auto& component_spec = cache.grid().components[0];
  Array<const void, dynamic_rank, offset_origin> partial_array;
  if (!GetPermutedPartialArray(entry, component_spec.fill_value,
                               partial_array)) {
    return {};
  }
  return tensorstore::StrCat("write to virtual chunk ", partial_array.domain());
}

void VirtualChunkedCache::TransactionNode::Commit() {
  if (!GetOwningCache(*this).write_function_) {
    // Should have been prevented by ReadWriteMode.
    SetError(absl::InternalError(
        "No write function specified to virtual_chunked driver"));
    this->WritebackError();
    return;
  }
  InitiateWriteback(absl::InfinitePast());
  internal::ChunkCache::TransactionNode::Commit();
}

void VirtualChunkedCache::TransactionNode::InitiateWriteback(
    absl::Time staleness_bound) {
  struct ApplyReceiver {
    TransactionNode& self;
    void set_value(AsyncCache::ReadState update) {
      GetOwningCache(self).executor()(
          [node = &self, update = std::move(update)] {
            auto* read_data = static_cast<const ReadData*>(update.data.get());
            SharedArrayView<const void> full_array;
            if (read_data && read_data[0].valid()) {
              full_array = read_data[0];
            } else {
              full_array = node->component_specs()[0].fill_value;
            }

            auto& entry = GetOwningEntry(*node);
            auto& cache = GetOwningCache(*node);

            Array<const void, dynamic_rank, offset_origin> partial_array;
            if (!GetPermutedPartialArray(entry, full_array, partial_array)) {
              node->WritebackSuccess(
                  {std::move(update.data),
                   {StorageGeneration::NoValue(), absl::InfiniteFuture()}});
              return;
            }
            WriteParameters write_params;
            write_params.if_equal_ =
                StorageGeneration::Clean(update.stamp.generation);
            write_params.executor_ = cache.executor();
            auto write_future = cache.write_function_(std::move(partial_array),
                                                      std::move(write_params));
            write_future.Force();
            write_future.ExecuteWhenReady(
                [node = node, update = std::move(update),
                 full_array = std::move(full_array)](
                    ReadyFuture<TimestampedStorageGeneration> future) mutable {
                  auto& r = future.result();
                  if (!r.ok()) {
                    node->SetError(std::move(r).status());
                    node->WritebackError();
                    return;
                  }
                  if (StorageGeneration::IsUnknown(r->generation)) {
                    // Generation mismatch.
                    node->InitiateWriteback(r->time);
                    return;
                  }
                  update.stamp = std::move(*r);
                  node->WritebackSuccess(std::move(update));
                });
          });
    }
    void set_error(absl::Status error) {
      self.SetError(std::move(error));
      self.WritebackError();
    }
    void set_cancel() { ABSL_UNREACHABLE(); }  // COV_NF_LINE
  };
  AsyncCache::TransactionNode::ApplyOptions apply_options;
  apply_options.staleness_bound = staleness_bound;
  this->DoApply(std::move(apply_options), ApplyReceiver{*this});
}

class VirtualChunkedDriverSpec
    : public internal::RegisteredDriverSpec<VirtualChunkedDriverSpec,
                                            internal::DriverSpec> {
 public:
  constexpr static const char id[] = "virtual_chunked";

  std::optional<ReadFunction> read_function;
  std::optional<WriteFunction> write_function;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency;
  Context::Resource<internal::CachePoolResource> cache_pool;
  StalenessBound data_staleness;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(internal::BaseCast<internal::DriverSpec>(x), x.read_function,
             x.write_function, x.data_copy_concurrency, x.cache_pool,
             x.data_staleness);
  };

  Future<internal::Driver::Handle> Open(
      internal::OpenTransactionPtr transaction,
      ReadWriteMode read_write_mode) const override;

  absl::Status ApplyOptions(SpecOptions&& options) override {
    if (options.kvstore.valid()) {
      return absl::InvalidArgumentError(
          "virtual_chunked driver does not support a kvstore");
    }

    if (options.recheck_cached_data.specified()) {
      data_staleness = StalenessBound(options.recheck_cached_data);
    }

    if (options.recheck_cached_metadata.specified()) {
      return absl::InvalidArgumentError(
          "virtual_chunked driver does not support recheck_cached_metadata");
    }

    return schema.Set(static_cast<Schema&&>(options));
  }
};

class VirtualChunkedDriver
    : public internal::RegisteredDriver<VirtualChunkedDriver,
                                        internal::ChunkCacheDriver> {
  using Base = internal::RegisteredDriver<VirtualChunkedDriver,
                                          internal::ChunkCacheDriver>;

 public:
  using Base::Base;

  Result<internal::TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override;

  static Result<internal::Driver::Handle> OpenFromSpecData(
      Transaction transaction, const VirtualChunkedDriverSpec& spec);

  VirtualChunkedCache* cache() const {
    return static_cast<VirtualChunkedCache*>(
        internal::ChunkCacheDriver::cache());
  }
  Result<CodecSpec> GetCodec() override { return CodecSpec{}; }

  Result<DimensionUnitsVector> GetDimensionUnits() override {
    return cache()->dimension_units_;
  }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) override {
    return {std::in_place};
  }
};

Result<internal::TransformedDriverSpec> VirtualChunkedDriver::GetBoundSpec(
    internal::OpenTransactionPtr transaction, IndexTransformView<> transform) {
  auto driver_spec = internal::DriverSpec::Make<VirtualChunkedDriverSpec>();
  driver_spec->context_binding_state_ = ContextBindingState::bound;
  auto& cache = *this->cache();
  if (cache.read_function_) {
    driver_spec->read_function = cache.read_function_;
  }
  if (cache.write_function_) {
    driver_spec->write_function = cache.write_function_;
  }
  driver_spec->data_copy_concurrency = cache.data_copy_concurrency_;
  driver_spec->cache_pool = cache.cache_pool_;
  driver_spec->data_staleness = this->data_staleness_bound();
  const DimensionIndex rank = this->rank();
  TENSORSTORE_RETURN_IF_ERROR(driver_spec->schema.Set(RankConstraint{rank}));
  TENSORSTORE_RETURN_IF_ERROR(driver_spec->schema.Set(dtype()));
  TENSORSTORE_RETURN_IF_ERROR(
      driver_spec->schema.Set(Schema::DimensionUnits(cache.dimension_units_)));
  TENSORSTORE_RETURN_IF_ERROR(
      driver_spec->schema.Set(ChunkLayout::InnerOrder(cache.inner_order_)));
  TENSORSTORE_RETURN_IF_ERROR(driver_spec->schema.Set(
      ChunkLayout::GridOrigin(cache.grid_origin_for_read_function_)));

  span<const DimensionIndex> inner_order = cache.inner_order_;
  span<const Index> grid_origin_for_read_function =
      cache.grid_origin_for_read_function_;

  const auto& component_spec = cache.grid().components[component_index()];

  // Additional transform to left-compose with `transform` in order to obtain
  // a transform from the "external" output space.
  IndexTransformBuilder external_to_output_transform_builder(rank, rank);
  IndexDomainBuilder external_domain_builder(rank);
  Index chunk_shape[kMaxRank];
  for (DimensionIndex component_dim = 0; component_dim < rank;
       ++component_dim) {
    const DimensionIndex external_dim = inner_order[component_dim];

    const Index offset = grid_origin_for_read_function[external_dim];

    chunk_shape[external_dim] = component_spec.shape()[component_dim];

    // Output dimension `component_dim` of `transform` has a grid origin of 0.

    // The corresponding output dimension `external_dim` of `new_transform`
    // should have a grid origin of `offset`.
    external_to_output_transform_builder.output_single_input_dimension(
        external_dim, offset, 1, component_dim);

    TENSORSTORE_ASSIGN_OR_RETURN(
        external_domain_builder.bounds()[external_dim],
        ShiftInterval(component_spec.component_bounds[component_dim], offset));
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto external_to_output_transform,
                               external_to_output_transform_builder.Finalize());

  TENSORSTORE_ASSIGN_OR_RETURN(auto external_domain,
                               external_domain_builder.Finalize());

  TENSORSTORE_RETURN_IF_ERROR(driver_spec->schema.Set(
      ChunkLayout::ChunkShape(span<const Index>(&chunk_shape[0], rank))));

  TENSORSTORE_RETURN_IF_ERROR(
      driver_spec->schema.Set(std::move(external_domain)));

  internal::TransformedDriverSpec spec;
  TENSORSTORE_ASSIGN_OR_RETURN(
      spec.transform,
      ComposeTransforms(external_to_output_transform, transform));
  spec.driver_spec = std::move(driver_spec);
  return spec;
}

Result<internal::Driver::Handle> VirtualChunkedDriver::OpenFromSpecData(
    Transaction transaction, const VirtualChunkedDriverSpec& spec) {
  const DimensionIndex rank = spec.schema.rank();
  if (rank == dynamic_rank) {
    return absl::InvalidArgumentError("rank must be specified");
  }

  DataType dtype = spec.schema.dtype();
  if (!dtype.valid()) {
    return absl::InvalidArgumentError("dtype must be specified");
  }

  IndexDomain<> domain = spec.schema.domain();
  if (!domain.valid()) {
    domain = IndexDomain<>(rank);
  }
  domain = WithImplicitDimensions(std::move(domain),
                                  /*implicit_lower_bounds=*/false,
                                  /*implicit_upper_bounds=*/false);

  Box<> chunk_template(rank);
  std::vector<DimensionIndex> inner_order(rank);
  {
    ChunkLayout chunk_layout = spec.schema.chunk_layout();
    if (chunk_layout.codec_chunk_shape().hard_constraint) {
      return absl::InvalidArgumentError("codec_chunk_shape not supported");
    }
    if (spec.schema.fill_value().valid()) {
      return absl::InvalidArgumentError("fill_value not supported");
    }
    TENSORSTORE_RETURN_IF_ERROR(
        internal::ChooseReadWriteChunkGrid(chunk_layout, domain.box(),
                                           chunk_template),
        tensorstore::MaybeAnnotateStatus(_, "Failed to compute chunk grid"));
    if (auto requested_inner_order = chunk_layout.inner_order();
        requested_inner_order.valid()) {
      std::copy_n(requested_inner_order.begin(), rank, inner_order.begin());
    } else {
      std::iota(inner_order.begin(), inner_order.end(), DimensionIndex(0));
    }
  }

  auto external_dimension_units = spec.schema.dimension_units();

  Box<> adjusted_component_domain(rank);
  DimensionUnitsVector component_units(rank);
  for (DimensionIndex component_dim = 0; component_dim < rank;
       ++component_dim) {
    const DimensionIndex external_dim = inner_order[component_dim];
    TENSORSTORE_ASSIGN_OR_RETURN(
        adjusted_component_domain[component_dim],
        ShiftIntervalBackward(domain[external_dim],
                              chunk_template.origin()[external_dim]));
    if (external_dimension_units.valid()) {
      component_units[component_dim] = external_dimension_units[external_dim];
    }
  }

  internal::Driver::Handle handle;
  handle.transaction = std::move(transaction);

  // inner_order[i] is the dimension of the user-requested external space that
  // corresponds to dimension i of the chunk grid component.
  //
  // For example, if the inner order is: {2, 0, 1}, i.e. "z", "x", "y".  Then
  // "y" is the contiguous dimension, and component dimension:
  //
  //     0 -> external dimension "z" (2)
  //     1 -> external dimension "x" (0)
  //     2 -> external dimension "y" (1)

  {
    IndexTransformBuilder transform_builder(rank, rank);
    transform_builder.input_domain(domain);
    for (DimensionIndex component_dim = 0; component_dim < rank;
         ++component_dim) {
      const DimensionIndex external_dim = inner_order[component_dim];
      transform_builder.output_single_input_dimension(
          component_dim, -chunk_template.origin()[external_dim], 1,
          external_dim);
    }
    TENSORSTORE_ASSIGN_OR_RETURN(handle.transform,
                                 transform_builder.Finalize());
  }

  // Cache key of "" means a distinct cache on each call to `GetCache`.
  auto cache = (*spec.cache_pool)->GetCache<VirtualChunkedCache>("", [&] {
    // Create the fill value array, which is just a single value-initialized
    // element broadcast to have a shape equal to the component chunk shape.
    // The fill value is not user-configurable and doesn't have any user-visible
    // effect for this driver, but the chunk cache requires one.
    SharedArray<const void> fill_value;
    fill_value.layout().set_rank(rank);
    std::fill_n(fill_value.byte_strides().begin(), rank, 0);
    for (DimensionIndex component_dim = 0; component_dim < rank;
         ++component_dim) {
      const DimensionIndex external_dim = inner_order[component_dim];
      fill_value.shape()[component_dim] = chunk_template.shape()[external_dim];
    }
    fill_value.element_pointer() = internal::AllocateAndConstructSharedElements(
        1, value_init, spec.schema.dtype());
    internal::ChunkGridSpecification::Components components;
    components.emplace_back(std::move(fill_value),
                            std::move(adjusted_component_domain));
    auto cache = std::make_unique<VirtualChunkedCache>(
        internal::ChunkGridSpecification(std::move(components)),
        spec.data_copy_concurrency->executor);
    cache->dimension_units_ = std::move(component_units);
    if (spec.read_function) {
      cache->read_function_ = *spec.read_function;
    }
    if (spec.write_function) {
      cache->write_function_ = *spec.write_function;
    }
    cache->inner_order_ = std::move(inner_order);
    cache->grid_origin_for_read_function_.assign(
        chunk_template.origin().begin(), chunk_template.origin().end());
    cache->cache_pool_ = spec.cache_pool;
    cache->data_copy_concurrency_ = spec.data_copy_concurrency;
    return cache;
  });
  ReadWriteMode read_write_mode =
      (cache->read_function_ ? ReadWriteMode::read : ReadWriteMode{}) |
      (cache->write_function_ ? ReadWriteMode::write : ReadWriteMode{});
  handle.driver = internal::MakeReadWritePtr<VirtualChunkedDriver>(
      read_write_mode, std::move(cache), /*component_index=*/0,
      spec.data_staleness.BoundAtOpen(absl::Now()));

  return handle;
}

Future<internal::Driver::Handle> VirtualChunkedDriverSpec::Open(
    internal::OpenTransactionPtr transaction,
    ReadWriteMode read_write_mode) const {
  if ((read_write_mode & ReadWriteMode::read) == ReadWriteMode::read &&
      !read_function) {
    return absl::InvalidArgumentError("Reading not supported");
  }
  if ((read_write_mode & ReadWriteMode::write) == ReadWriteMode::write &&
      !write_function) {
    return absl::InvalidArgumentError("Writing not supported");
  }
  if (read_write_mode == ReadWriteMode::dynamic) {
    read_write_mode = (read_function ? ReadWriteMode::read : ReadWriteMode{}) |
                      (write_function ? ReadWriteMode::write : ReadWriteMode{});
  }
  return VirtualChunkedDriver::OpenFromSpecData(
      internal::TransactionState::ToTransaction(std::move(transaction)), *this);
}

}  // namespace

namespace internal_virtual_chunked {
Result<internal::Driver::Handle> MakeDriver(
    virtual_chunked::ReadFunction read_function,
    virtual_chunked::WriteFunction write_function, OpenOptions&& options) {
  VirtualChunkedDriverSpec spec;
  if (read_function) {
    spec.read_function = std::move(read_function);
  }
  if (write_function) {
    spec.write_function = std::move(write_function);
  }
  spec.schema = static_cast<Schema&&>(options);

  if (!options.context) {
    options.context = Context::Default();
  }

  TENSORSTORE_ASSIGN_OR_RETURN(
      spec.cache_pool,
      options.context.GetResource<internal::CachePoolResource>());

  TENSORSTORE_ASSIGN_OR_RETURN(
      spec.data_copy_concurrency,
      options.context.GetResource<internal::DataCopyConcurrencyResource>());

  if (options.recheck_cached_data.specified()) {
    spec.data_staleness = StalenessBound(options.recheck_cached_data);
  }

  return VirtualChunkedDriver::OpenFromSpecData(std::move(options.transaction),
                                                spec);
}
}  // namespace internal_virtual_chunked
}  // namespace virtual_chunked

namespace garbage_collection {
template <>
struct GarbageCollection<virtual_chunked::VirtualChunkedDriver> {
  static void Visit(GarbageCollectionVisitor& visitor,
                    const virtual_chunked::VirtualChunkedDriver& value) {
    garbage_collection::GarbageCollectionVisit(visitor,
                                               value.cache()->read_function_);
    garbage_collection::GarbageCollectionVisit(visitor,
                                               value.cache()->write_function_);
  }
};
}  // namespace garbage_collection
}  // namespace tensorstore

namespace {
const tensorstore::internal::SerializationOnlyDriverRegistration<
    tensorstore::virtual_chunked::VirtualChunkedDriverSpec>
    driver_registration;
}  // namespace
