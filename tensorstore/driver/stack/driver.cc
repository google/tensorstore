// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/driver/driver.h"

#include <assert.h>
#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/propagate_bounds.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/grid_partition.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/irregular_grid.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/staleness_bound.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/schema.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

/// Support for ApplyMembers protocols
#include "tensorstore/internal/context_binding_vector.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/util/garbage_collection/fwd.h"  // IWYU pragma: keep
#include "tensorstore/util/garbage_collection/std_vector.h"  // IWYU pragma: keep

namespace tensorstore {
namespace internal_stack {
namespace {

using ::tensorstore::internal::DataCopyConcurrencyResource;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::IrregularGrid;
using ::tensorstore::internal::MakeIntrusivePtr;
using ::tensorstore::internal::OpenTransactionPtr;
using ::tensorstore::internal::ReadChunk;
using ::tensorstore::internal::TransformedDriverSpec;
using ::tensorstore::internal::WriteChunk;

// NOTES:
//
// The stack driver would benefit from being able to hold a weak pointer to a
// driver. For instance, a stack driver is likely to be based on a series of
// ImageDriver objects which have images loaded. When such a driver transitions
// from a strong reference to a weak reference, then it could release the pinned
// entry.  When converted back to a strong reference, then the driver would need
// to reacquire the pinned entry.
//
// To implement fill_value, we may need to do the following:
// 1. In the unspecified case, convert the datatype in GetFillValue() from
//    `json` to the dtype of the driver.
// 2. Clean the options when propagating them to each layer. Some layers error
//    when fill_value is set.
// 3. Handle missing cells on read (see UnmappedReadOp) to submit chunks for
//    those cells which map to the individual transforms.
//

namespace jb = tensorstore::internal_json_binding;

absl::Status TransactionError() {
  return absl::UnimplementedError(
      "\"stack\" driver does not support transactions");
}

/// Used to index individual cells
struct Cell {
  std::vector<Index> points;

  Cell(std::vector<Index> data) : points(std::move(data)) {}
  Cell(span<const Index> data) : points(data.begin(), data.end()) {}

  span<const Index> as_span() const { return span<const Index>(points); }
};

struct CellHash {
  using is_transparent = void;

  size_t operator()(span<const Index> v) const {
    return absl::Hash<span<const Index>>{}(v);
  }
  size_t operator()(const Cell& v) const {
    return absl::Hash<span<const Index>>{}(v.as_span());
  }
};

struct CellEq {
  using is_transparent = void;

  bool operator()(const Cell& a, span<const Index> b) const {
    return std::equal(a.as_span().begin(), a.as_span().end(), b.begin(),
                      b.end());
  }
  bool operator()(span<const Index> a, const Cell& b) const {
    return std::equal(a.begin(), a.end(), b.as_span().begin(),
                      b.as_span().end());
  }
  bool operator()(const Cell& a, const Cell& b) const {
    return std::equal(a.as_span().begin(), a.as_span().end(),
                      b.as_span().begin(), b.as_span().end());
  }
  bool operator()(span<const Index> a, span<const Index> b) const {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
  }
};

class StackDriverSpec
    : public internal::RegisteredDriverSpec<StackDriverSpec,
                                            /*Parent=*/internal::DriverSpec> {
 public:
  constexpr static char id[] = "stack";

  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;
  std::vector<TransformedDriverSpec> layers;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(internal::BaseCast<internal::DriverSpec>(x),
             x.data_copy_concurrency, x.layers);
  };

  absl::Status InitializeLayerRankAndDtype() {
    if (layers.empty()) {
      return absl::InvalidArgumentError("\"stack\" driver spec has no layers");
    }
    /// Set the schema rank and dtype based on the layers.
    for (auto& d : layers) {
      TENSORSTORE_RETURN_IF_ERROR(
          this->schema.Set(RankConstraint{internal::GetRank(d)}));
      TENSORSTORE_RETURN_IF_ERROR(
          this->schema.Set(d.driver_spec->schema.dtype()));
    }
    return absl::OkStatus();
  }

  constexpr static auto default_json_binder = jb::Sequence(
      jb::Member(DataCopyConcurrencyResource::id,
                 jb::Projection<&StackDriverSpec::data_copy_concurrency>()),
      jb::Member("layers", jb::Projection<&StackDriverSpec::layers>()),
      jb::Initialize([](auto* obj) {
        TENSORSTORE_RETURN_IF_ERROR(obj->InitializeLayerRankAndDtype());
        SpecOptions base_options;
        static_cast<Schema&>(base_options) = std::exchange(obj->schema, {});
        return obj->ApplyOptions(std::move(base_options));
      }));

  absl::Status ApplyOptions(SpecOptions&& options) override {
    if (options.codec().valid()) {
      return absl::InvalidArgumentError(
          "\"codec\" not supported by \"stack\" driver");
    }
    if (options.fill_value().valid()) {
      return absl::InvalidArgumentError(
          "\"fill_value\" not supported by \"stack\" driver");
    }
    for (auto& d : layers) {
      // Filter the options to only those that we wish to pass on to the
      // layers, otherwise we may be passing on nonsensical settings for a
      // layer.
      SpecOptions o;
      o.open_mode = options.open_mode;
      o.recheck_cached_data = options.recheck_cached_data;
      o.recheck_cached_metadata = options.recheck_cached_metadata;
      TENSORSTORE_RETURN_IF_ERROR(static_cast<Schema&>(o).Set(schema.dtype()));
      TENSORSTORE_RETURN_IF_ERROR(static_cast<Schema&>(o).Set(schema.rank()));
      TENSORSTORE_RETURN_IF_ERROR(
          internal::TransformAndApplyOptions(d, std::move(o)));
    }
    return schema.Set(static_cast<Schema&&>(options));
  }

  Result<std::vector<IndexDomain<>>> GetEffectiveDomainsForLayers() const {
    assert(!layers.empty());
    std::vector<IndexDomain<>> domains;
    domains.reserve(layers.size());
    DimensionIndex rank;
    for (size_t i = 0; i < layers.size(); i++) {
      TENSORSTORE_ASSIGN_OR_RETURN(auto effective_domain,
                                   internal::GetEffectiveDomain(layers[i]));
      if (!effective_domain.valid()) {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("layer[", i, "] domain is unspecified"));
      }
      domains.emplace_back(std::move(effective_domain));
      // validate rank.
      if (i == 0) {
        rank = domains.back().rank();
      } else if (domains.back().rank() != rank) {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("layer[", i, "] rank mismatch"));
      }
    }
    return domains;
  }

  Result<IndexDomain<>> GetDomain() const override {
    // Each layer is expected to have an effective domain so that each layer
    // can be queried when resolving chunks.
    // The overall domain is Hull(layer.domain...)
    IndexDomain<> domain;
    {
      TENSORSTORE_ASSIGN_OR_RETURN(auto domains,
                                   GetEffectiveDomainsForLayers());
      for (auto& d : domains) {
        TENSORSTORE_ASSIGN_OR_RETURN(domain, HullIndexDomains(domain, d));
      }
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        domain, ConstrainIndexDomain(schema.domain(), std::move(domain)));
    // stack disallows resize, so mark all dimensions as explicit.
    return WithImplicitDimensions(std::move(domain), false, false);
  }

  Result<DimensionUnitsVector> GetDimensionUnits() const override {
    /// Retrieve dimension units from schema. These take precedence
    /// over computed dimensions, so disallow further assignment.
    DimensionUnitsVector dimension_units(schema.dimension_units());
    DimensionSet allow_assignment(true);
    for (size_t i = 0; i < dimension_units.size(); i++) {
      if (dimension_units[i].has_value()) {
        allow_assignment[i] = false;
      }
    }
    /// Merge dimension units from the layers. If there are conflicting
    /// values, clear and disallow further assignment.
    for (const auto& d : layers) {
      TENSORSTORE_ASSIGN_OR_RETURN(auto du, GetEffectiveDimensionUnits(d));
      if (du.size() > dimension_units.size()) {
        dimension_units.resize(du.size());
      }
      for (size_t i = 0; i < du.size(); i++) {
        if (!allow_assignment[i]) continue;
        if (!du[i].has_value()) continue;
        if (!dimension_units[i].has_value()) {
          dimension_units[i] = du[i];
        } else if (dimension_units[i].value() != du[i].value()) {
          // mismatch; clear and disallow future assignment.
          dimension_units[i] = std::nullopt;
          allow_assignment[i] = false;
        }
      }
    }
    return dimension_units;
  }

  Future<internal::Driver::Handle> Open(
      internal::OpenTransactionPtr transaction,
      ReadWriteMode read_write_mode) const override;
};

class StackDriver
    : public internal::RegisteredDriver<StackDriver,
                                        /*Parent=*/internal::Driver> {
 public:
  explicit StackDriver(StackDriverSpec bound_spec)
      : bound_spec_(std::move(bound_spec)) {
    assert(bound_spec_.context_binding_state_ == ContextBindingState::bound);
    dimension_units_ = bound_spec_.GetDimensionUnits().value_or(
        DimensionUnitsVector(bound_spec_.schema.rank()));
  }

  DataType dtype() override { return bound_spec_.schema.dtype(); }
  DimensionIndex rank() override { return bound_spec_.schema.rank(); }

  Executor data_copy_executor() override {
    return bound_spec_.data_copy_concurrency->executor;
  }

  Result<DimensionUnitsVector> GetDimensionUnits() override {
    return dimension_units_;
  }

  Result<TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override;

  Future<IndexTransform<>> ResolveBounds(OpenTransactionPtr transaction,
                                         IndexTransform<> transform,
                                         ResolveBoundsOptions options) override;

  void Read(OpenTransactionPtr transaction, IndexTransform<> transform,
            AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver)
      override;

  void Write(OpenTransactionPtr transaction, IndexTransform<> transform,
             WriteChunkReceiver receiver) override;

  absl::Status InitializeGridIndices(const std::vector<IndexDomain<>>& domains);

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    // Exclude `context_binding_state_` because it is handled specially.
    return f(x.bound_spec_);
  };

  StackDriverSpec bound_spec_;
  DimensionUnitsVector dimension_units_;
  IndexDomain<> layer_domain_;
  IrregularGrid grid_;

  absl::flat_hash_map<Cell, size_t, CellHash, CellEq> grid_to_layer_;
};

Future<internal::Driver::Handle> StackDriverSpec::Open(
    internal::OpenTransactionPtr transaction,
    ReadWriteMode read_write_mode) const {
  if (transaction) return TransactionError();
  if (read_write_mode == ReadWriteMode::dynamic) {
    read_write_mode = ReadWriteMode::read_write;
  }
  if (!schema.dtype().valid()) {
    return absl::InvalidArgumentError(
        "Unable to infer \"dtype\" in \"stack\" driver");
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto domains, GetEffectiveDomainsForLayers());

  auto driver_ptr =
      internal::MakeReadWritePtr<StackDriver>(read_write_mode, *this);
  TENSORSTORE_ASSIGN_OR_RETURN(driver_ptr->layer_domain_, GetDomain());
  TENSORSTORE_RETURN_IF_ERROR(driver_ptr->InitializeGridIndices(domains));

  auto transform = tensorstore::IdentityTransform(driver_ptr->layer_domain_);
  return internal::Driver::Handle{std::move(driver_ptr), std::move(transform)};
}

/// The mechanism used here is to construct an irregular grid based on
/// each layer's effective transform, and then, for each of those grid
/// points, store the layer associated with the bounding rectangle.
///
/// If this becomes a problem, we may want to consider constructing
/// an R-tree variant using the layer MBRs to restrict the query space,
/// then applying a similar gridding decomposition to the read transforms.
absl::Status StackDriver::InitializeGridIndices(
    const std::vector<IndexDomain<>>& domains) {
  assert(domains.size() == bound_spec_.layers.size());
  grid_ = IrregularGrid::Make(span(domains));

  absl::InlinedVector<Index, internal::kNumInlinedDims> start(grid_.rank());
  absl::InlinedVector<Index, internal::kNumInlinedDims> end(grid_.rank());
  for (size_t i = 0; i < domains.size(); i++) {
    auto& d = domains[i];
    for (size_t j = 0; j < grid_.rank(); j++) {
      start[j] = grid_(j, d[j].inclusive_min(), nullptr);
      end[j] = 1 + grid_(j, d[j].inclusive_max(), nullptr);
    }
    // Set the mapping for all irregular grid cell covered by this layer
    // to point to this layer.
    IterateOverIndexRange<>(
        span(start), span(end),
        [i, this](span<const Index> key) { grid_to_layer_[key] = i; });
  }

#if !defined(NDEBUG)
  // Log the missing cells.
  std::fill(start.begin(), start.end(), 0);
  IterateOverIndexRange<>(
      span<const Index>(start), grid_.shape(), [this](span<const Index> key) {
        if (auto it = grid_to_layer_.find(key); it == grid_to_layer_.end()) {
          ABSL_LOG(INFO) << "\"stack\" driver missing grid cell: " << key;
        }
      });
#endif

  return absl::OkStatus();
}

Result<TransformedDriverSpec> StackDriver::GetBoundSpec(
    internal::OpenTransactionPtr transaction, IndexTransformView<> transform) {
  if (transaction) return TransactionError();
  auto driver_spec = internal::DriverSpec::Make<StackDriverSpec>();
  *driver_spec = bound_spec_;

  // When constructing the bound spec, set the dimension_units_ and
  // the current domain on the schema.
  driver_spec->schema.Set(Schema::DimensionUnits(dimension_units_))
      .IgnoreError();
  driver_spec->schema.Set(layer_domain_).IgnoreError();

  TransformedDriverSpec spec;
  spec.driver_spec = std::move(driver_spec);
  spec.transform = transform;
  return spec;
}

Future<IndexTransform<>> StackDriver::ResolveBounds(
    OpenTransactionPtr transaction, IndexTransform<> transform,
    ResolveBoundsOptions options) {
  if (transaction) return TransactionError();
  // All layer bounds are required to be specified in the spec, and
  // may not be modified later, so here we propagate the composed bounds
  // to the index transform.
  using internal_index_space::PropagateExplicitBoundsToTransform;
  using internal_index_space::TransformAccess;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto transform_ptr,
      PropagateExplicitBoundsToTransform(
          layer_domain_.box(), TransformAccess::rep_ptr(std::move(transform))));
  return TransformAccess::Make<IndexTransform<>>(std::move(transform_ptr));
}

// Commits Promise<void> when the last reference is released.
// Typically this is already done by Promise<>, however state->receiver
// holds a reference to promise,so the promise would otherwise not be
// marked ready unless the operation was cancelled.
template <typename StateType>
struct SetPromiseOnRelease
    : public internal::AtomicReferenceCount<SetPromiseOnRelease<StateType>> {
  IntrusivePtr<StateType> state;
  Promise<void> promise;

  SetPromiseOnRelease(IntrusivePtr<StateType> s, Promise<void> p)
      : state(std::move(s)), promise(std::move(p)) {}
  ~SetPromiseOnRelease() { promise.SetReady(); }

  inline void SetError(absl::Status error) {
    SetDeferredResult(promise, std::move(error));
  }
};

// Forwarding receiver which satisfies `Driver::ReadChunkReceiver` or
// `Driver::WriteChunkReceiver`.
// The starting/stopping/error/done parts of the protocol are handled
// by the future, so this only forwards set_error and set_value calls.
template <typename StateType>
struct ForwardingLayerReceiver {
  using ChunkType = typename StateType::ChunkType;
  IntrusivePtr<SetPromiseOnRelease<StateType>> set_promise;
  IndexTransform<> cell_transform;
  FutureCallbackRegistration cancel_registration;

  void set_starting(AnyCancelReceiver cancel) {
    cancel_registration =
        set_promise->promise.ExecuteWhenNotNeeded(std::move(cancel));
  }
  void set_stopping() { cancel_registration(); }
  void set_done() {}
  void set_error(absl::Status error) {
    set_promise->SetError(std::move(error));
  }
  void set_value(ChunkType chunk, IndexTransform<> composed_transform) {
    auto c_transform = ComposeTransforms(cell_transform, composed_transform);
    if (!c_transform.ok()) {
      set_promise->SetError(std::move(c_transform).status());
    } else {
      execution::set_value(set_promise->state->receiver, std::move(chunk),
                           std::move(c_transform).value());
    }
  }
};

/// Starts reads for each cell against a provided driver handle.
template <typename StateType>
struct AfterOpenOp {
  IntrusivePtr<SetPromiseOnRelease<StateType>> set_promise;
  size_t layer_id;
  std::vector<IndexTransform<>> cells;

  absl::Status ComposeAndDispatch(ReadyFuture<internal::Driver::Handle> f) {
    if (!f.result().ok()) {
      return f.result().status();
    }
    // After opening the layer, issue reads to each of the the grid cells.
    // The read transform is: Compose(outer_request, Compose(driver, cell)).
    for (auto& cell_transform : cells) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto a_transform,
          ComposeTransforms(set_promise->state->orig_transform,
                            cell_transform));
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto b_transform,
          ComposeTransforms(f.result()->transform, std::move(a_transform)));

      set_promise->state->Dispatch(*f.result(), std::move(b_transform),
                                   set_promise, cell_transform);
    }
    return absl::OkStatus();
  }

  void operator()(Promise<void>, ReadyFuture<internal::Driver::Handle> f) {
    absl::Status status = ComposeAndDispatch(std::move(f));
    if (!status.ok()) {
      set_promise->SetError(MaybeAnnotateStatus(
          std::move(status),
          tensorstore::StrCat("While opening layer ", layer_id)));
    }
  }
};

// OpenLayerOp partitons the transform by layer, invoking UnmappedOp for each
// grid cell which is not backed by a layer, and then opens each layer and
// and initiates OpType (one of LayerReadOp/LayerWriteOp) for each layer's
// cells.
template <typename StateType, typename UnmappedOpType>
struct OpenLayerOp {
  IntrusivePtr<SetPromiseOnRelease<StateType>> set_promise;

  void operator()() {
    auto& self = set_promise->state->self;
    // Partition the initial transform over irregular grid space,
    // which results in a set of transforms for each layer, and collate them
    // by layer.
    std::vector<DimensionIndex> dimension_order(self->grid_.rank());
    std::iota(dimension_order.begin(), dimension_order.end(),
              DimensionIndex{0});

    UnmappedOpType unmapped{set_promise->state->self.get()};
    absl::flat_hash_map<size_t, std::vector<IndexTransform<>>> layers_to_load;
    auto status = tensorstore::internal::PartitionIndexTransformOverGrid(
        dimension_order, self->grid_, set_promise->state->orig_transform,
        [&](span<const Index> grid_cell_indices,
            IndexTransformView<> cell_transform) {
          auto it = self->grid_to_layer_.find(grid_cell_indices);
          if (it != self->grid_to_layer_.end()) {
            layers_to_load[it->second].emplace_back(cell_transform);
            return absl::OkStatus();
          } else {
            return unmapped(grid_cell_indices, cell_transform);
          }
        });
    if (!status.ok()) {
      set_promise->SetError(std::move(status));
      return;
    }
    if (layers_to_load.empty()) {
      return;
    }

    // Open each layer and invoke OpType for all corresponding cell transforms.
    for (auto& kv : layers_to_load) {
      Link(WithExecutor(self->data_copy_executor(),
                        AfterOpenOp<StateType>{set_promise, kv.first,
                                               std::move(kv.second)}),
           set_promise->promise,
           internal::OpenDriver(set_promise->state->transaction,
                                self->bound_spec_.layers[kv.first],
                                StateType::kMode));
    }
  }
};

// Asynchronous state for StackDriver::Read maintains reference
// counts while the read operation is in progress.
struct ReadState : public internal::AtomicReferenceCount<ReadState> {
  static constexpr ReadWriteMode kMode = ReadWriteMode::read;
  using Receiver = ForwardingLayerReceiver<ReadState>;
  using ChunkType = ReadChunk;

  IntrusivePtr<StackDriver> self;
  OpenTransactionPtr transaction;
  AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver;
  IndexTransform<> orig_transform;

  // Initiate the read of an individual transform; dispatched by AfterOpenOp
  void Dispatch(internal::Driver::Handle& h,
                IndexTransform<> composed_transform,
                IntrusivePtr<SetPromiseOnRelease<ReadState>> set_promise,
                IndexTransform<> cell_transform) {
    h.driver->Read(
        transaction, std::move(composed_transform),
        ReadState::Receiver{std::move(set_promise), std::move(cell_transform)});
  }
};

struct UnmappedReadOp {
  StackDriver* self;
  absl::Status operator()(span<const Index> grid_cell_indices,
                          IndexTransformView<> cell_transform) {
    auto origin = self->grid_.cell_origin(grid_cell_indices);
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Read cell origin=", span(origin),
                            " missing layer mapping in \"stack\" driver"));
  }
};

void StackDriver::Read(
    OpenTransactionPtr transaction, IndexTransform<> transform,
    AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver) {
  auto state = MakeIntrusivePtr<ReadState>();
  state->self = IntrusivePtr<StackDriver>(this);
  state->receiver = std::move(receiver);
  state->transaction = std::move(transaction);
  state->orig_transform = std::move(transform);
  auto op = PromiseFuturePair<void>::Make(absl::OkStatus());
  execution::set_starting(state->receiver, [p = op.promise] {
    SetDeferredResult(p, absl::CancelledError(""));
  });
  op.future.ExecuteWhenReady([state](ReadyFuture<void> f) {
    if (f.status().ok() || absl::IsCancelled(f.status())) {
      execution::set_done(state->receiver);
    } else {
      execution::set_error(state->receiver, f.status());
    }
    execution::set_stopping(state->receiver);
  });

  data_copy_executor()(OpenLayerOp<ReadState, UnmappedReadOp>{
      MakeIntrusivePtr<SetPromiseOnRelease<ReadState>>(std::move(state),
                                                       std::move(op.promise))});
}

// Asynchronous state for StackDriver::Write maintains reference
// counts while the read operation is in progress.
struct WriteState : public internal::AtomicReferenceCount<WriteState> {
  static constexpr ReadWriteMode kMode = ReadWriteMode::write;
  using Receiver = ForwardingLayerReceiver<WriteState>;
  using ChunkType = WriteChunk;

  IntrusivePtr<StackDriver> self;
  OpenTransactionPtr transaction;
  AnyFlowReceiver<absl::Status, WriteChunk, IndexTransform<>> receiver;
  IndexTransform<> orig_transform;

  // Initiate the write of an individual transform; dispatched by AfterOpenOp
  void Dispatch(internal::Driver::Handle& h,
                IndexTransform<> composed_transform,
                IntrusivePtr<SetPromiseOnRelease<WriteState>> set_promise,
                IndexTransform<> cell_transform) {
    h.driver->Write(transaction, std::move(composed_transform),
                    WriteState::Receiver{std::move(set_promise),
                                         std::move(cell_transform)});
  }
};

struct UnmappedWriteOp {
  StackDriver* self;
  absl::Status operator()(span<const Index> grid_cell_indices,
                          IndexTransformView<> cell_transform) {
    auto origin = self->grid_.cell_origin(grid_cell_indices);
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Write cell origin=", span(origin),
                            " missing layer mapping in \"stack\" driver"));
  }
};

void StackDriver::Write(OpenTransactionPtr transaction,
                        IndexTransform<> transform,
                        WriteChunkReceiver receiver) {
  auto state = MakeIntrusivePtr<WriteState>();
  state->self = IntrusivePtr<StackDriver>(this);
  state->receiver = std::move(receiver);
  state->transaction = std::move(transaction);
  state->orig_transform = std::move(transform);
  auto op = PromiseFuturePair<void>::Make(absl::OkStatus());

  execution::set_starting(state->receiver, [p = op.promise] {
    SetDeferredResult(p, absl::CancelledError(""));
  });
  op.future.ExecuteWhenReady([state](ReadyFuture<void> f) {
    if (f.status().ok() || absl::IsCancelled(f.status())) {
      execution::set_done(state->receiver);
    } else {
      execution::set_error(state->receiver, f.status());
    }
    execution::set_stopping(state->receiver);
  });

  data_copy_executor()(OpenLayerOp<WriteState, UnmappedWriteOp>{
      MakeIntrusivePtr<SetPromiseOnRelease<WriteState>>(
          std::move(state), std::move(op.promise))});
}

}  // namespace
}  // namespace internal_stack
}  // namespace tensorstore

namespace {
const tensorstore::internal::DriverRegistration<
    tensorstore::internal_stack::StackDriverSpec>
    driver_registration;
}  // namespace
