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

#include "tensorstore/kvstore/driver.h"

#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/chunk_cache_driver.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/cache/async_initialized_cache_mixin.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/cache/chunk_cache.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/serialization/absl_time.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/tensorstore.h"

namespace tensorstore {
namespace ometiff {

namespace {

namespace jb = tensorstore::internal_json_binding;

class DataCache : public internal::ConcreteChunkCache,
                  public internal::AsyncInitializedCacheMixin {
  using Base = internal::ConcreteChunkCache;

 public:
  /// Constructs a `DataCache`.
  template <typename... U>
  explicit DataCache(std::string key, U&&... args)
      : Base(std::forward<U>(args)...),
        key_(std::move(key)),
        kvstore_driver_(kvstore::DriverPtr()) {}

  /// Common implementation used by `Entry::DoRead` and
  /// `TransactionNode::DoRead`.
  template <typename EntryOrNode>
  void DoRead(EntryOrNode& node, absl::Time staleness_bound);

  class Entry : public internal::ChunkCache::Entry {
   public:
    using OwningCache = DataCache;
    using internal::ChunkCache::Entry::Entry;

    void DoRead(absl::Time staleness_bound) override {
      GetOwningCache(*this).DoRead(*this, staleness_bound);
    }
  };

  std::string GetKeyValueStoreKey() { return key_; }

  Entry* DoAllocateEntry() final { return new Entry; }
  std::size_t DoGetSizeofEntry() final { return sizeof(Entry); }

  TransactionNode* DoAllocateTransactionNode(
      internal::AsyncCache::Entry& entry) {
    std::cerr << "We shoudln't be here!" << std::endl;
    return nullptr;
  }
  // Indexed by `external_dim`.
  std::vector<Index> grid_origin_for_read_function_;

  // Indexed by `component_dim`.
  DimensionUnitsVector dimension_units_;

  // Indexed by `component_dim`.
  std::vector<DimensionIndex> inner_order_;

  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency_;
  Context::Resource<internal::CachePoolResource> cache_pool_;

  /// Returns the associated `kvstore::Driver`.
  kvstore::Driver* kvstore_driver() { return kvstore_driver_.get(); }

  /// Sets the `kvstore::Driver`.  The caller is responsible for ensuring there
  /// are no concurrent read or write operations.
  void SetKvStoreDriver(kvstore::DriverPtr driver) {
    kvstore_driver_ = std::move(driver);
  }

  std::string key_;
  kvstore::DriverPtr kvstore_driver_;
};

/// Sets `partial_array` to refer to the portion of `full_array` (translated to
/// the chunk origin) that is within bounds for the chunk corresponding to
/// `entry`.  Also permutes the dimensions according to
/// `DataCache::inner_order_`.
///
/// \param entry Entry corresponding to the chunk.
/// \param full_array Array of shape equal to the component chunk shape.
/// \param partial_array[out] Set to the portion of `full_array` corresponding
///     to `entry`, indexed by "external" dimensions.
bool GetPermutedPartialArray(
    DataCache::Entry& entry, ArrayView<const void> full_array,
    Array<const void, dynamic_rank, offset_origin>& partial_array) {
  auto& cache = static_cast<DataCache&>(GetOwningCache(entry));
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
void DataCache::DoRead(EntryOrNode& node, absl::Time staleness_bound) {
  GetOwningCache(node).executor()([&node, staleness_bound] {
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

    kvstore::ReadOptions options;
    {
      ReadLock<ReadData> lock{node};
      options.if_not_equal = lock.stamp().generation;
    }
    options.staleness_bound = staleness_bound;
    std::cout << "Key " << cache.GetKeyValueStoreKey() << std::endl;
    auto read_future = cache.kvstore_driver_->Read(cache.GetKeyValueStoreKey(),
                                                   std::move(options));
    std::move(read_future)
        .ExecuteWhenReady([&node, read_data = std::move(read_data)](
                              ReadyFuture<kvstore::ReadResult> future) mutable {
          auto& r = future.result();
          if (r->aborted()) {  // Revisit
            node.ReadSuccess({std::move(read_data), std::move(r->stamp)});
            return;
          }
          if (r->not_found()) {
            node.ReadError(absl::NotFoundError(""));
            return;
          }
          auto& value = r->value;
          std::cout << "Data size: " << value.size() << std::endl;

          // Right now no data is getting copied.
          node.ReadSuccess({std::move(read_data), std::move(r->stamp)});
          return;
        });
  });
}

class OMETiffDriverSpec
    : public internal::RegisteredDriverSpec<OMETiffDriverSpec,
                                            internal::DriverSpec> {
 public:
  constexpr static const char id[] = "ometiff";

  kvstore::Spec store;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency;
  Context::Resource<internal::CachePoolResource> cache_pool;
  StalenessBound data_staleness;
  std::vector<Index> shape;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<internal::DriverSpec>(x), x.store,
             x.data_copy_concurrency, x.cache_pool, x.data_staleness);
  };

  OpenMode open_mode() const override {
    // Since opening has no side effects, we return `open` even though `create`
    // might also be considered correct.
    return OpenMode::open;
  }

  static absl::Status ValidateSchema(Schema& schema) {
    if (schema.codec().valid()) {
      return absl::InvalidArgumentError(
          "codec not supported by ometiff driver");
    }
    if (schema.fill_value().valid()) {
      return absl::InvalidArgumentError(
          "fill_value not supported by ometiff driver");
    }
    return absl::OkStatus();
  }

  constexpr static auto default_json_binder = jb::Sequence(
      jb::Initialize([](auto* obj) -> absl::Status {
        return ValidateSchema(obj->schema);
      }),
      // jb::Member("shape", jb::Projection<&OMETiffDriverSpec::shape>()),
      jb::Member(internal::DataCopyConcurrencyResource::id,
                 jb::Projection<&OMETiffDriverSpec::data_copy_concurrency>()),
      jb::Member(internal::CachePoolResource::id,
                 jb::Projection<&OMETiffDriverSpec::cache_pool>()),
      jb::Projection<&OMETiffDriverSpec::store>(
          jb::KvStoreSpecAndPathJsonBinder));

  absl::Status ApplyOptions(SpecOptions&& options) override {
    if (options.recheck_cached_data.specified()) {
      data_staleness = StalenessBound(options.recheck_cached_data);
    }
    if (options.recheck_cached_metadata.specified()) {
      StalenessBound bound(options.recheck_cached_metadata);
      if (!options.recheck_cached_data.specified() ||
          bound.time > data_staleness.time) {
        data_staleness = std::move(bound);
      }
    }
    if (options.kvstore.valid()) {
      if (store.valid()) {
        return absl::InvalidArgumentError("\"kvstore\" is already specified");
      }
      store = std::move(options.kvstore);
    }
    return ValidateSchema(options);
  }

  kvstore::Spec GetKvstore() const override { return store; }

  Future<internal::Driver::Handle> Open(
      internal::OpenTransactionPtr transaction,
      ReadWriteMode read_write_mode) const override;
};

class OMETiffDriver;
using OMETiffDriverBase = internal::RegisteredDriver<
    OMETiffDriver, internal::ChunkGridSpecificationDriver<
                       DataCache, internal::ChunkCacheReadWriteDriverMixin<
                                      OMETiffDriver, internal::Driver>>>;

class OMETiffDriver : public OMETiffDriverBase {
  using Base = OMETiffDriverBase;

 public:
  using Base::Base;

  Result<internal::TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override;

  static Result<internal::Driver::Handle> OpenFromSpecData(
      Transaction transaction, const OMETiffDriverSpec& spec);

  Result<CodecSpec> GetCodec() override { return CodecSpec{}; }

  Result<DimensionUnitsVector> GetDimensionUnits() override {
    return cache()->dimension_units_;
  }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) override {
    return {std::in_place};
  }

  Result<ChunkLayout> GetChunkLayout(IndexTransformView<> transform) override {
    return internal::GetChunkLayoutFromGrid(cache()->grid().components[0]) |
           transform;
  }

  StalenessBound data_staleness_;
};

Result<internal::TransformedDriverSpec> OMETiffDriver::GetBoundSpec(
    internal::OpenTransactionPtr transaction, IndexTransformView<> transform) {
  std::cerr << "Getboundspec" << std::endl;
  auto driver_spec = internal::DriverSpec::Make<OMETiffDriverSpec>();
  driver_spec->context_binding_state_ = ContextBindingState::bound;
  auto& cache = *this->cache();
  TENSORSTORE_ASSIGN_OR_RETURN(driver_spec->store.driver,
                               cache.kvstore_driver()->GetBoundSpec());
  // driver_spec->store.path = cache.key();
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

Result<internal::Driver::Handle> OMETiffDriver::OpenFromSpecData(
    Transaction transaction, const OMETiffDriverSpec& spec) {
  const DimensionIndex rank = spec.schema.rank();
  if (rank == dynamic_rank) {
    return absl::InvalidArgumentError("rank must be specified");
  }

  DataType dtype = spec.schema.dtype();
  if (!dtype.valid()) {
    return absl::InvalidArgumentError("dtype must be specified");
  }

  IndexDomain<> domain = spec.schema.domain();
  std::cerr << "Domain: " << domain << std::endl;
  std::cerr << "Rank: " << rank << std::endl;
  if (!domain.valid()) {
    domain = IndexDomain<>(rank);
  }
  domain = WithImplicitDimensions(std::move(domain),
                                  /*implicit_lower_bounds=*/false,
                                  /*implicit_upper_bounds=*/false);
  std::cerr << "Updated domain: " << domain << std::endl;
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

  auto cache = internal::GetOrCreateAsyncInitializedCache<DataCache>(
      **spec.cache_pool, "",
      [&] {
        std::cerr << "Creating cache" << std::endl;
        SharedArray<const void> fill_value;
        fill_value.layout().set_rank(rank);
        std::fill_n(fill_value.byte_strides().begin(), rank, 0);
        for (DimensionIndex component_dim = 0; component_dim < rank;
             ++component_dim) {
          const DimensionIndex external_dim = inner_order[component_dim];
          fill_value.shape()[component_dim] =
              chunk_template.shape()[external_dim];
        }
        fill_value.element_pointer() =
            internal::AllocateAndConstructSharedElements(1, value_init,
                                                         spec.schema.dtype());
        internal::ChunkGridSpecification::ComponentList components;
        components.emplace_back(std::move(fill_value),
                                std::move(adjusted_component_domain));
        auto cache = std::make_unique<DataCache>(
            spec.store.path,
            internal::ChunkGridSpecification(std::move(components)),
            spec.data_copy_concurrency->executor);
        cache->dimension_units_ = std::move(component_units);
        cache->inner_order_ = std::move(inner_order);
        cache->grid_origin_for_read_function_.assign(
            chunk_template.origin().begin(), chunk_template.origin().end());
        cache->cache_pool_ = spec.cache_pool;
        cache->data_copy_concurrency_ = spec.data_copy_concurrency;
        return cache;
      },
      [&](Promise<void> initialize_promise,
          internal::CachePtr<DataCache> cache) {
        LinkValue(
            [cache = std::move(cache)](Promise<void> cache_promise,
                                       ReadyFuture<kvstore::DriverPtr> future) {
              auto kv = std::move(*future.result());
              cache->SetKvStoreDriver(std::move(kv));
            },
            initialize_promise, kvstore::Open(spec.store.driver));
      });

  // Cache key of "" means a distinct cache on each call to `GetCache`.
  ReadWriteMode read_write_mode = ReadWriteMode::read;

  handle.driver = internal::MakeReadWritePtr<OMETiffDriver>(
      read_write_mode,
      OMETiffDriver::Initializer{std::move(cache), /*component_index=*/0,
                                 spec.data_staleness.BoundAtOpen(absl::Now())});
  // handle.driver->cache_entry_ = GetCacheEntry(cache, store.path);
  return handle;
}

Future<internal::Driver::Handle> OMETiffDriverSpec::Open(
    internal::OpenTransactionPtr transaction,
    ReadWriteMode read_write_mode) const {
  if ((read_write_mode & ReadWriteMode::write) == ReadWriteMode::write) {
    return absl::InvalidArgumentError("Writing not supported");
  }
  if (read_write_mode == ReadWriteMode::dynamic) {
    // No writing for now.
    read_write_mode = ReadWriteMode::read;
  }
  if (!store.valid()) {
    return absl::InvalidArgumentError("\"kvstore\" must be specified");
  }

  return OMETiffDriver::OpenFromSpecData(
      internal::TransactionState::ToTransaction(std::move(transaction)), *this);
}

}  // namespace

}  // namespace ometiff

namespace garbage_collection {
template <>
struct GarbageCollection<ometiff::OMETiffDriver> {
  static void Visit(GarbageCollectionVisitor& visitor,
                    const ometiff::OMETiffDriver& value) {
    return garbage_collection::GarbageCollectionVisit(
        visitor, value.cache()->kvstore_driver());
  }
};
}  // namespace garbage_collection
}  // namespace tensorstore

namespace {
const tensorstore::internal::DriverRegistration<
    tensorstore::ometiff::OMETiffDriverSpec>
    driver_registration;
}  // namespace
