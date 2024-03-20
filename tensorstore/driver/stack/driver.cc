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
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorstore/box.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/chunk_receiver_utils.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/driver/stack/driver.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/propagate_bounds.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/grid_partition.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/irregular_grid.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/staleness_bound.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/open_options.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

/// Support for ApplyMembers protocols
#include "tensorstore/internal/context_binding_vector.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/util/garbage_collection/fwd.h"  // IWYU pragma: keep
#include "tensorstore/util/garbage_collection/std_optional.h"  // IWYU pragma: keep
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

// Specifies a stack layer as either a `DriverSpec` (to be opened on demand) or
// an open `Driver`, along with a transform.
struct StackLayer {
  // Index transform, must not be null.
  IndexTransform<> transform;

  // Driver spec, if layer is to be opened on demand.
  internal::DriverSpecPtr driver_spec;

  // Driver, if layer is already open.
  internal::ReadWritePtr<internal::Driver> driver;

  bool is_open() const { return static_cast<bool>(driver); }

  internal::DriverHandle GetDriverHandle(Transaction transaction) const {
    assert(is_open());
    return {driver, transform, transaction};
  }

  internal::DriverHandle GetDriverHandle(
      internal::OpenTransactionPtr& transaction) const {
    return GetDriverHandle(
        internal::TransactionState::ToTransaction(transaction));
  }

  internal::TransformedDriverSpec GetTransformedDriverSpec() const {
    assert(!is_open());
    return internal::TransformedDriverSpec{driver_spec, transform};
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.transform, x.driver_spec, x.driver);
  };
};

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

// Certain operations are applied to either a sequence of
// `internal::TransformedDriverSpec` used by the `StackDriverSpec` to represent
// layers, or to a sequence of `StackLayer` used by the open `StackDriver` to
// represent layers.
template <typename T>
constexpr bool IsStackLayerLike = false;

template <>
constexpr bool IsStackLayerLike<internal::TransformedDriverSpec> = true;

template <>
constexpr bool IsStackLayerLike<StackLayer> = true;

template <typename Callback>
absl::Status ForEachLayer(size_t num_layers, Callback&& callback) {
  for (size_t layer_i = 0; layer_i < num_layers; ++layer_i) {
    absl::Status status = callback(layer_i);
    if (!status.ok()) {
      return tensorstore::MaybeAnnotateStatus(
          status, absl::StrFormat("Layer %d", layer_i));
    }
  }
  return absl::OkStatus();
}

using internal::GetEffectiveDomain;

Result<IndexDomain<>> GetEffectiveDomain(const StackLayer& layer) {
  return layer.is_open()
             ? layer.transform.domain()
             : internal::GetEffectiveDomain(layer.GetTransformedDriverSpec());
}

template <typename Layer>
Result<std::vector<IndexDomain<>>> GetEffectiveDomainsForLayers(
    span<const Layer> layers) {
  static_assert(IsStackLayerLike<Layer>);
  std::vector<IndexDomain<>> domains;
  domains.reserve(layers.size());
  DimensionIndex rank;
  auto status = ForEachLayer(layers.size(), [&](size_t layer_i) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto effective_domain,
        internal_stack::GetEffectiveDomain(layers[layer_i]));
    if (!effective_domain.valid()) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Domain must be specified"));
    }
    domains.emplace_back(std::move(effective_domain));
    // validate rank.
    if (layer_i == 0) {
      rank = domains.back().rank();
    } else if (domains.back().rank() != rank) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Layer domain ", domains.back(), " of rank ", domains.back().rank(),
          " does not match layer 0 rank of ", rank));
    }
    return absl::OkStatus();
  });
  if (!status.ok()) return status;
  return domains;
}

Result<IndexDomain<>> GetCombinedDomain(
    const Schema& schema, span<const IndexDomain<>> layer_domains) {
  // Each layer is expected to have an effective domain so that each layer
  // can be queried when resolving chunks.
  // The overall domain is Hull(layer.domain...)
  IndexDomain<> domain;
  auto status = ForEachLayer(layer_domains.size(), [&](size_t layer_i) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        domain, HullIndexDomains(domain, layer_domains[layer_i]));
    return absl::OkStatus();
  });
  if (!status.ok()) return status;
  TENSORSTORE_ASSIGN_OR_RETURN(
      domain, ConstrainIndexDomain(schema.domain(), std::move(domain)));
  // stack disallows resize, so mark all dimensions as explicit.
  return WithImplicitDimensions(std::move(domain), false, false);
}

using internal::GetEffectiveDimensionUnits;

Result<DimensionUnitsVector> GetEffectiveDimensionUnits(
    const StackLayer& layer) {
  if (layer.is_open()) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto units, layer.driver->GetDimensionUnits());
    return tensorstore::TransformOutputDimensionUnits(layer.transform,
                                                      std::move(units));
  } else {
    return internal::GetEffectiveDimensionUnits(
        layer.GetTransformedDriverSpec());
  }
}

template <typename Layer>
Result<DimensionUnitsVector> GetDimensionUnits(const Schema& schema,
                                               span<const Layer> layers) {
  static_assert(IsStackLayerLike<Layer>);
  // Retrieve dimension units from schema. These take precedence over computed
  // dimensions, so disallow further assignment.
  DimensionUnitsVector dimension_units(schema.dimension_units());
  DimensionSet allow_assignment(true);
  for (size_t i = 0; i < dimension_units.size(); i++) {
    if (dimension_units[i].has_value()) {
      allow_assignment[i] = false;
    }
  }
  // Merge dimension units from the layers. If there are conflicting values,
  // clear and disallow further assignment.
  auto status = ForEachLayer(layers.size(), [&](size_t layer_i) {
    const auto& d = layers[layer_i];
    TENSORSTORE_ASSIGN_OR_RETURN(auto du,
                                 internal_stack::GetEffectiveDimensionUnits(d));
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
    return absl::OkStatus();
  });
  if (!status.ok()) return status;
  return dimension_units;
}

using internal::TransformAndApplyOptions;

absl::Status TransformAndApplyOptions(StackLayer& layer,
                                      SpecOptions&& options) {
  assert(!layer.is_open());
  return internal::TransformAndApplyOptions(layer.driver_spec, layer.transform,
                                            std::move(options));
}

template <typename Layer>
absl::Status ApplyLayerOptions(span<Layer> layers, Schema& schema,
                               const SpecOptions& options) {
  if (&schema != &options) {
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(options.rank()));
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(options.dtype()));
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(options.domain()));
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(options.dimension_units()));
  }
  if (options.codec().valid()) {
    return absl::InvalidArgumentError(
        "codec option not supported by \"stack\" driver");
  }
  if (options.fill_value().valid()) {
    return absl::InvalidArgumentError(
        "fill value option not supported by \"stack\" driver");
  }
  if (options.kvstore.valid()) {
    return absl::InvalidArgumentError(
        "kvstore option not supported by \"stack\" driver");
  }
  if (options.chunk_layout().HasHardConstraints()) {
    return absl::InvalidArgumentError(
        "chunk layout option not supported by \"stack\" driver");
  }
  return ForEachLayer(layers.size(), [&](size_t layer_i) {
    auto& layer = layers[layer_i];
    if constexpr (std::is_same_v<Layer, StackLayer>) {
      if (layer.is_open()) {
        if (options.open_mode != OpenMode{} &&
            !(options.open_mode & OpenMode::open)) {
          return absl::InvalidArgumentError(tensorstore::StrCat(
              "Open mode of ", options.open_mode,
              " is not compatible with already-open layer"));
        }
        if (options.recheck_cached_data.specified() ||
            options.recheck_cached_metadata.specified()) {
          return absl::InvalidArgumentError(
              "Cannot specify cache rechecking options with already-open "
              "layer");
        }
        return absl::OkStatus();
      }
    }
    // Filter the options to only those that we wish to pass on to the
    // layers, otherwise we may be passing on nonsensical settings for a
    // layer.
    SpecOptions layer_options;
    layer_options.open_mode = options.open_mode;
    layer_options.recheck_cached_data = options.recheck_cached_data;
    layer_options.recheck_cached_metadata = options.recheck_cached_metadata;
    layer_options.minimal_spec = options.minimal_spec;
    TENSORSTORE_RETURN_IF_ERROR(
        static_cast<Schema&>(layer_options).Set(schema.dtype()));
    TENSORSTORE_RETURN_IF_ERROR(
        static_cast<Schema&>(layer_options).Set(schema.rank()));
    return internal_stack::TransformAndApplyOptions(layers[layer_i],
                                                    std::move(layer_options));
  });
}

class StackDriverSpec
    : public internal::RegisteredDriverSpec<StackDriverSpec,
                                            /*Parent=*/internal::DriverSpec> {
 public:
  constexpr static char id[] = "stack";

  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;
  std::vector<internal::TransformedDriverSpec> layers;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(internal::BaseCast<internal::DriverSpec>(x),
             x.data_copy_concurrency, x.layers);
  };

  absl::Status InitializeLayerRankAndDtype() {
    return ForEachLayer(layers.size(), [&](size_t layer_i) {
      auto& layer = layers[layer_i];
      DimensionIndex layer_rank = internal::GetRank(layer);
      if (schema.rank() != dynamic_rank && layer_rank != schema.rank()) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Rank of ", layer_rank, " does not match existing rank of ",
            schema.rank()));
      }
      schema.Set(RankConstraint{layer_rank}).IgnoreError();
      TENSORSTORE_RETURN_IF_ERROR(
          schema.Set(layer.driver_spec->schema.dtype()));
      return absl::OkStatus();
    });
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
    return internal_stack::ApplyLayerOptions<internal::TransformedDriverSpec>(
        layers, schema, std::move(options));
  }

  OpenMode open_mode() const override {
    if (layers.empty()) return OpenMode::unknown;
    OpenMode prev_mode;
    for (size_t i = 0; i < layers.size(); ++i) {
      auto mode = internal::GetOpenMode(layers[i]);
      if (i != 0 && mode != prev_mode) return OpenMode::unknown;
      prev_mode = mode;
    }
    return prev_mode;
  }

  Result<IndexDomain<>> GetDomain() const override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto layer_domains,
                                 internal_stack::GetEffectiveDomainsForLayers<
                                     internal::TransformedDriverSpec>(layers));
    return internal_stack::GetCombinedDomain(schema, layer_domains);
  }

  Result<DimensionUnitsVector> GetDimensionUnits() const override {
    return internal_stack::GetDimensionUnits<internal::TransformedDriverSpec>(
        schema, layers);
  }

  Future<internal::Driver::Handle> Open(
      internal::DriverOpenRequest request) const override;
};

class StackDriver
    : public internal::RegisteredDriver<StackDriver,
                                        /*Parent=*/internal::Driver> {
 public:
  DataType dtype() override { return dtype_; }
  DimensionIndex rank() override { return layer_domain_.rank(); }

  Executor data_copy_executor() override {
    return data_copy_concurrency_->executor;
  }

  Result<DimensionUnitsVector> GetDimensionUnits() override {
    return dimension_units_;
  }

  Result<TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override;

  Future<IndexTransform<>> ResolveBounds(ResolveBoundsRequest request) override;

  void Read(ReadRequest request,
            AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver)
      override;

  void Write(WriteRequest request, WriteChunkReceiver receiver) override;

  absl::Status InitializeGridIndices(span<const IndexDomain<>> domains);

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    // Exclude `context_binding_state_` because it is handled specially.
    return f(x.dtype_, x.data_copy_concurrency_, x.layers_, x.dimension_units_,
             x.layer_domain_);
  };

  DataType dtype_;
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency_;
  std::vector<StackLayer> layers_;
  DimensionUnitsVector dimension_units_;
  IndexDomain<> layer_domain_;
  IrregularGrid grid_;

  absl::flat_hash_map<Cell, size_t, CellHash, CellEq> grid_to_layer_;
};

Result<internal::Driver::Handle> MakeStackDriverHandle(
    internal::ReadWritePtr<StackDriver> driver,
    span<const IndexDomain<>> layer_domains, Transaction transaction,
    const Schema& schema) {
  driver->dtype_ = schema.dtype();
  TENSORSTORE_ASSIGN_OR_RETURN(
      driver->layer_domain_,
      internal_stack::GetCombinedDomain(schema, layer_domains));
  TENSORSTORE_RETURN_IF_ERROR(driver->InitializeGridIndices(layer_domains));
  auto transform = IdentityTransform(driver->layer_domain_);
  driver->dimension_units_ =
      internal_stack::GetDimensionUnits<StackLayer>(schema, driver->layers_)
          .value_or(DimensionUnitsVector(transform.input_rank()));
  return internal::DriverHandle{std::move(driver), std::move(transform),
                                std::move(transaction)};
}

Future<internal::Driver::Handle> StackDriverSpec::Open(
    internal::DriverOpenRequest request) const {
  if (!schema.dtype().valid()) {
    return absl::InvalidArgumentError("dtype must be specified");
  }
  if (request.read_write_mode == ReadWriteMode::dynamic) {
    request.read_write_mode = ReadWriteMode::read_write;
  }
  auto driver =
      internal::MakeReadWritePtr<StackDriver>(request.read_write_mode);
  driver->data_copy_concurrency_ = data_copy_concurrency;
  const size_t num_layers = layers.size();
  driver->layers_.resize(num_layers);
  for (size_t layer_i = 0; layer_i < num_layers; ++layer_i) {
    auto& layer_spec = layers[layer_i];
    auto& layer = driver->layers_[layer_i];
    layer.transform = layer_spec.transform;
    layer.driver_spec = layer_spec.driver_spec;
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto layer_domains,
      internal_stack::GetEffectiveDomainsForLayers<StackLayer>(
          driver->layers_));
  return internal_stack::MakeStackDriverHandle(
      std::move(driver), layer_domains,
      internal::TransactionState::ToTransaction(std::move(request.transaction)),
      schema);
}

/// The mechanism used here is to construct an irregular grid based on
/// each layer's effective transform, and then, for each of those grid
/// points, store the layer associated with the bounding rectangle.
///
/// If this becomes a problem, we may want to consider constructing
/// an R-tree variant using the layer MBRs to restrict the query space,
/// then applying a similar gridding decomposition to the read transforms.
absl::Status StackDriver::InitializeGridIndices(
    span<const IndexDomain<>> domains) {
  assert(domains.size() == layers_.size());
  grid_ = IrregularGrid::Make(layers_.empty() ? span(&layer_domain_, 1)
                                              : span(domains));

  Index start[kMaxRank];
  Index shape[kMaxRank];
  const DimensionIndex rank = grid_.rank();
  for (size_t layer_i = 0; layer_i < domains.size(); layer_i++) {
    auto& d = domains[layer_i];
    for (DimensionIndex dim = 0; dim < rank; dim++) {
      start[dim] = grid_(dim, d[dim].inclusive_min(), nullptr);
      shape[dim] = 1 + grid_(dim, d[dim].inclusive_max(), nullptr) - start[dim];
    }
    // Set the mapping for all irregular grid cell covered by this layer
    // to point to this layer.
    IterateOverIndexRange<>(BoxView<>(rank, start, shape),
                            [layer_i, this](span<const Index> key) {
                              grid_to_layer_[key] = layer_i;
                            });
  }

#if !defined(NDEBUG)
  // Log the missing cells.
  IterateOverIndexRange<>(grid_.shape(), [this](span<const Index> key) {
    if (auto it = grid_to_layer_.find(key); it == grid_to_layer_.end()) {
      ABSL_LOG(INFO) << "\"stack\" driver missing grid cell: " << key;
    }
  });
#endif

  return absl::OkStatus();
}

Result<TransformedDriverSpec> StackDriver::GetBoundSpec(
    internal::OpenTransactionPtr transaction, IndexTransformView<> transform) {
  auto driver_spec = internal::DriverSpec::Make<StackDriverSpec>();
  driver_spec->data_copy_concurrency = data_copy_concurrency_;
  driver_spec->schema.Set(dtype_).IgnoreError();
  driver_spec->schema.Set(RankConstraint{rank()}).IgnoreError();
  // When constructing the bound spec, set the dimension_units_ and
  // the current domain on the schema.
  driver_spec->schema.Set(Schema::DimensionUnits(dimension_units_))
      .IgnoreError();
  driver_spec->schema.Set(layer_domain_).IgnoreError();

  const size_t num_layers = layers_.size();
  driver_spec->layers.resize(num_layers);
  auto status = ForEachLayer(num_layers, [&](size_t layer_i) {
    auto& layer_spec = driver_spec->layers[layer_i];
    const auto& layer = layers_[layer_i];
    if (layer.driver_spec) {
      layer_spec.transform = layer.transform;
      layer_spec.driver_spec = layer.driver_spec;
    } else {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto driver_spec,
          GetTransformedDriverSpec(layer.GetDriverHandle(transaction),
                                   SpecRequestOptions()));
      layer_spec.transform = std::move(driver_spec.transform);
      layer_spec.driver_spec = std::move(driver_spec.driver_spec);
    }
    return absl::OkStatus();
  });
  if (!status.ok()) return status;
  TransformedDriverSpec spec;
  spec.driver_spec = std::move(driver_spec);
  spec.transform = transform;
  return spec;
}

Future<IndexTransform<>> StackDriver::ResolveBounds(
    ResolveBoundsRequest request) {
  // All layer bounds are required to be specified in the spec, and
  // may not be modified later, so here we propagate the composed bounds
  // to the index transform.
  using internal_index_space::PropagateExplicitBoundsToTransform;
  using internal_index_space::TransformAccess;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto transform_ptr,
      PropagateExplicitBoundsToTransform(
          layer_domain_.box(),
          TransformAccess::rep_ptr(std::move(request.transform))));
  return TransformAccess::Make<IndexTransform<>>(std::move(transform_ptr));
}

template <typename StateType>
absl::Status ComposeAndDispatchOperation(
    StateType& state, const internal::DriverHandle& driver_handle,
    IndexTransform<> cell_transform) {
  TENSORSTORE_RETURN_IF_ERROR(internal::ValidateSupportsModes(
      driver_handle.driver.read_write_mode(), StateType::kMode));
  // The transform for the layer to use is:
  //    `Compose(outer_request, Compose(driver, cell))`
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto a_transform,
      ComposeTransforms(state.request.transform, cell_transform));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto b_transform,
      ComposeTransforms(driver_handle.transform, std::move(a_transform)));

  state.Dispatch(driver_handle, std::move(b_transform),
                 std::move(cell_transform));
  return absl::OkStatus();
}

/// Starts reads for each cell against a provided driver handle.
template <typename StateType>
struct AfterOpenOp {
  IntrusivePtr<StateType> state;
  size_t layer_id;
  std::vector<IndexTransform<>> cells;

  absl::Status ComposeAndDispatch(ReadyFuture<internal::Driver::Handle> f) {
    if (!f.result().ok()) {
      return f.result().status();
    }
    // After opening the layer, issue reads to each of the the grid cells.
    for (auto& cell_transform : cells) {
      TENSORSTORE_RETURN_IF_ERROR(ComposeAndDispatchOperation(
          *state, f.value(), std::move(cell_transform)));
    }
    return absl::OkStatus();
  }

  void operator()(Promise<void>, ReadyFuture<internal::Driver::Handle> f) {
    absl::Status status = ComposeAndDispatch(std::move(f));
    if (!status.ok()) {
      state->SetError(MaybeAnnotateStatus(
          std::move(status),
          tensorstore::StrCat("While opening layer ", layer_id)));
    }
  }
};

// OpenLayerOp partitions the transform by layer, invoking UnmappedOp for each
// grid cell which is not backed by a layer, and then opens each layer and
// and initiates OpType (one of LayerReadOp/LayerWriteOp) for each layer's
// cells.
template <typename StateType, typename UnmappedOpType>
struct OpenLayerOp {
  IntrusivePtr<StateType> state;

  void operator()() {
    auto* self = state->self.get();
    // Partition the initial transform over irregular grid space,
    // which results in a set of transforms for each layer, and collate them
    // by layer.
    std::vector<DimensionIndex> dimension_order(self->grid_.rank());
    std::iota(dimension_order.begin(), dimension_order.end(),
              DimensionIndex{0});

    UnmappedOpType unmapped{self};
    absl::flat_hash_map<size_t, std::vector<IndexTransform<>>> layers_to_load;
    auto status = tensorstore::internal::PartitionIndexTransformOverGrid(
        dimension_order, self->grid_, state->request.transform,
        [&](span<const Index> grid_cell_indices,
            IndexTransformView<> cell_transform) {
          auto it = self->grid_to_layer_.find(grid_cell_indices);
          if (it != self->grid_to_layer_.end()) {
            const size_t layer_i = it->second;
            const auto& layer = self->layers_[layer_i];
            if (layer.driver) {
              // Layer is already open, dispatch operation directly.
              TENSORSTORE_RETURN_IF_ERROR(
                  ComposeAndDispatchOperation(
                      *state, layer.GetDriverHandle(state->request.transaction),
                      std::move(cell_transform)),
                  tensorstore::MaybeAnnotateStatus(
                      _, absl::StrFormat("Layer %d", layer_i)));
            } else {
              layers_to_load[it->second].emplace_back(cell_transform);
            }
            return absl::OkStatus();
          } else {
            return unmapped(grid_cell_indices, cell_transform);
          }
        });
    if (!status.ok()) {
      state->SetError(std::move(status));
      return;
    }
    if (layers_to_load.empty()) {
      return;
    }

    // Open each layer and invoke OpType for all corresponding cell
    // transforms.
    for (auto& kv : layers_to_load) {
      const size_t layer_i = kv.first;
      const auto& layer = self->layers_[layer_i];
      internal::DriverOpenRequest request;
      request.transaction = state->request.transaction;
      request.read_write_mode = StateType::kMode;
      Link(WithExecutor(
               self->data_copy_executor(),
               AfterOpenOp<StateType>{state, layer_i, std::move(kv.second)}),
           state->promise,
           internal::OpenDriver(layer.GetTransformedDriverSpec(),
                                std::move(request)));
    }
  }
};

struct UnmappedOp {
  StackDriver* self;
  absl::Status operator()(span<const Index> grid_cell_indices,
                          IndexTransformView<> cell_transform) {
    auto origin = self->grid_.cell_origin(grid_cell_indices);
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Cell with origin=", span(origin),
                            " missing layer mapping in \"stack\" driver"));
  }
};

// Asynchronous state for StackDriver::{Read,Write} that maintains reference
// counts while the read/write operation is in progress.
template <typename ChunkType>
struct ReadOrWriteState : public internal::ChunkOperationState<ChunkType> {
  static constexpr ReadWriteMode kMode = std::is_same_v<ChunkType, ReadChunk>
                                             ? ReadWriteMode::read
                                             : ReadWriteMode::write;
  using RequestType = std::conditional_t<std::is_same_v<ChunkType, ReadChunk>,
                                         internal::Driver::ReadRequest,
                                         internal::Driver::WriteRequest>;
  using Base = internal::ChunkOperationState<ChunkType>;
  using State = ReadOrWriteState<ChunkType>;
  using ForwardingReceiver = internal::ForwardingChunkOperationReceiver<State>;

  using Base::Base;

  IntrusivePtr<StackDriver> self;
  RequestType request;

  // Initiate the read of an individual transform; dispatched by AfterOpenOp
  void Dispatch(const internal::Driver::Handle& h,
                IndexTransform<> composed_transform,
                IndexTransform<> cell_transform) {
    auto sub_request = this->request;
    sub_request.transform = std::move(composed_transform);

    constexpr auto method = [] {
      if constexpr (std::is_same_v<ChunkType, ReadChunk>) {
        return &internal::Driver::Read;
      } else {
        return &internal::Driver::Write;
      }
    }();

    (h.driver.get()->*method)(std::move(sub_request),
                              ForwardingReceiver{IntrusivePtr<State>(this),
                                                 std::move(cell_transform)});
  }

  static void Start(
      StackDriver& driver, RequestType&& request,
      AnyFlowReceiver<absl::Status, ChunkType, IndexTransform<>>&& receiver) {
    auto state = MakeIntrusivePtr<State>(std::move(receiver));
    const auto& executor = driver.data_copy_executor();
    state->self = IntrusivePtr<StackDriver>(&driver);
    state->request = std::move(request);
    executor(OpenLayerOp<State, UnmappedOp>{std::move(state)});
  }
};

void StackDriver::Read(
    ReadRequest request,
    AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver) {
  ReadOrWriteState<ReadChunk>::Start(*this, std::move(request),
                                     std::move(receiver));
}

void StackDriver::Write(WriteRequest request, WriteChunkReceiver receiver) {
  ReadOrWriteState<WriteChunk>::Start(*this, std::move(request),
                                      std::move(receiver));
}

Result<internal::ReadWritePtr<StackDriver>> MakeDriverFromLayerSpecs(
    span<const StackLayerSpec> layer_specs, StackOpenOptions& options,
    DimensionIndex& rank) {
  auto driver =
      internal::MakeReadWritePtr<StackDriver>(ReadWriteMode::read_write);
  auto& dtype = driver->dtype_;
  dtype = options.dtype();
  auto& transaction = options.transaction;
  auto& context = options.context;
  auto& read_write_mode = options.read_write_mode;
  driver->layers_.resize(layer_specs.size());
  if (!context) context = Context::Default();
  Transaction common_transaction{no_transaction};
  ReadWriteMode common_read_write_mode = ReadWriteMode::dynamic;
  DimensionIndex common_rank = dynamic_rank;
  auto status = ForEachLayer(layer_specs.size(), [&](size_t layer_i) {
    auto& layer = driver->layers_[layer_i];
    const auto& layer_spec = layer_specs[layer_i];
    const auto& layer_transaction =
        layer_spec.is_open() ? layer_spec.transaction : transaction;
    if (layer_i == 0) {
      common_transaction = layer_transaction;
    } else if (layer_transaction != common_transaction) {
      return absl::InvalidArgumentError("Transaction mismatch");
    }
    layer.transform = layer_spec.transform;
    layer.driver_spec = layer_spec.driver_spec;
    layer.driver = layer_spec.driver;
    DataType layer_dtype;
    if (layer_spec.is_open()) {
      common_read_write_mode |= layer_spec.driver.read_write_mode();
      layer_dtype = layer_spec.driver->dtype();
    } else {
      common_read_write_mode = ReadWriteMode::read_write;
      TENSORSTORE_RETURN_IF_ERROR(
          DriverSpecBindContext(layer.driver_spec, context));
      layer_dtype = layer_spec.driver_spec->schema.dtype();
      if (!layer.transform.valid()) {
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto domain,
            internal::GetEffectiveDomain(layer.GetTransformedDriverSpec()));
        if (!domain.valid()) {
          return absl::InvalidArgumentError(
              tensorstore::StrCat("Domain must be specified"));
        }
        layer.transform = IdentityTransform(domain);
      }
    }
    if (layer_dtype.valid()) {
      if (!dtype.valid()) {
        dtype = layer_dtype;
      } else if (dtype != layer_dtype) {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("Layer dtype of ", layer_dtype,
                                " does not match existing dtype of ", dtype));
      }
    }
    DimensionIndex layer_rank = layer.transform.input_rank();
    if (common_rank == dynamic_rank) {
      common_rank = layer_rank;
    } else if (common_rank != layer_rank) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Layer domain ", layer.transform.domain(), " of rank ", layer_rank,
          " does not match layer 0 rank of ", common_rank));
    }
    return absl::OkStatus();
  });
  if (!status.ok()) return status;

  if (common_read_write_mode == ReadWriteMode::dynamic) {
    common_read_write_mode = ReadWriteMode::read_write;
  }

  if (read_write_mode != ReadWriteMode::dynamic) {
    TENSORSTORE_RETURN_IF_ERROR(internal::ValidateSupportsModes(
        common_read_write_mode, read_write_mode));
  } else {
    read_write_mode = common_read_write_mode;
  }

  if (transaction != no_transaction) {
    if (common_transaction != no_transaction &&
        common_transaction != transaction) {
      return absl::InvalidArgumentError("Transaction mismatch");
    }
  } else {
    transaction = std::move(common_transaction);
  }

  if (!dtype.valid()) {
    return absl::InvalidArgumentError("dtype must be specified");
  }
  rank = common_rank;
  driver.set_read_write_mode(options.read_write_mode);
  options.Set(driver->dtype_).IgnoreError();
  return driver;
}

Result<internal::DriverHandle> FinalizeStackHandle(
    internal::ReadWritePtr<StackDriver> driver, StackOpenOptions&& options) {
  Schema& schema = options;
  TENSORSTORE_RETURN_IF_ERROR(internal_stack::ApplyLayerOptions<StackLayer>(
      driver->layers_, schema, options));
  TENSORSTORE_ASSIGN_OR_RETURN(
      driver->data_copy_concurrency_,
      options.context.GetResource<internal::DataCopyConcurrencyResource>());
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto layer_domains,
      internal_stack::GetEffectiveDomainsForLayers<StackLayer>(
          driver->layers_));
  return internal_stack::MakeStackDriverHandle(
      std::move(driver), layer_domains, std::move(options.transaction), schema);
}

}  // namespace

Result<internal::DriverHandle> Overlay(span<const StackLayerSpec> layer_specs,
                                       StackOpenOptions&& options) {
  DimensionIndex rank;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto driver,
      internal_stack::MakeDriverFromLayerSpecs(layer_specs, options, rank));
  TENSORSTORE_RETURN_IF_ERROR(options.Set(RankConstraint{rank}));
  return internal_stack::FinalizeStackHandle(std::move(driver),
                                             std::move(options));
}

Result<internal::DriverHandle> Stack(span<const StackLayerSpec> layer_specs,
                                     DimensionIndex stack_dimension,
                                     StackOpenOptions&& options) {
  if (layer_specs.empty()) {
    return absl::InvalidArgumentError(
        "At least one layer must be specified for stack");
  }
  DimensionIndex orig_rank;
  TENSORSTORE_ASSIGN_OR_RETURN(auto driver,
                               internal_stack::MakeDriverFromLayerSpecs(
                                   layer_specs, options, orig_rank));
  if (orig_rank == kMaxRank) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("stack would exceed maximum rank of ", kMaxRank));
  }
  const DimensionIndex new_rank = orig_rank + 1;
  TENSORSTORE_RETURN_IF_ERROR(options.Set(RankConstraint{new_rank}));
  TENSORSTORE_ASSIGN_OR_RETURN(
      stack_dimension,
      tensorstore::NormalizeDimensionIndex(stack_dimension, new_rank));
  if (auto status = ForEachLayer(
          layer_specs.size(),
          [&](size_t layer_i) {
            auto& transform = driver->layers_[layer_i].transform;
            TENSORSTORE_ASSIGN_OR_RETURN(
                transform,
                std::move(transform) |
                    Dims(stack_dimension)
                        .AddNew()
                        .SizedInterval(static_cast<Index>(layer_i), 1));
            return absl::OkStatus();
          });
      !status.ok()) {
    return status;
  }
  return internal_stack::FinalizeStackHandle(std::move(driver),
                                             std::move(options));
}

Result<internal::DriverHandle> Concat(span<const StackLayerSpec> layer_specs,
                                      DimensionIdentifier concat_dimension,
                                      StackOpenOptions&& options) {
  if (layer_specs.empty()) {
    return absl::InvalidArgumentError(
        "At least one layer must be specified for concat");
  }
  DimensionIndex rank;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto driver,
      internal_stack::MakeDriverFromLayerSpecs(layer_specs, options, rank));
  TENSORSTORE_RETURN_IF_ERROR(options.Set(RankConstraint{rank}));
  DimensionIndex concat_dimension_index;
  if (concat_dimension.label().data()) {
    // concat_dimension specified by label must be resolved to an index.
    std::string_view labels[kMaxRank];
    if (auto domain = options.domain(); domain.valid()) {
      std::copy(domain.labels().begin(), domain.labels().end(), labels);
    }
    if (auto status = ForEachLayer(
            driver->layers_.size(),
            [&](size_t layer_i) {
              auto layer_labels =
                  driver->layers_[layer_i].transform.domain().labels();
              for (DimensionIndex i = 0; i < rank; ++i) {
                TENSORSTORE_ASSIGN_OR_RETURN(
                    labels[i], MergeDimensionLabels(labels[i], layer_labels[i]),
                    tensorstore::MaybeAnnotateStatus(
                        _, absl::StrFormat("Mismatch in dimension %d", i)));
              }
              return absl::OkStatus();
            });
        !status.ok()) {
      return status;
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        concat_dimension_index,
        tensorstore::NormalizeDimensionLabel(
            concat_dimension.label(),
            span<const std::string_view>(&labels[0], rank)));
  } else {
    TENSORSTORE_ASSIGN_OR_RETURN(
        concat_dimension_index,
        tensorstore::NormalizeDimensionIndex(concat_dimension.index(), rank));
  }
  Index offset;
  if (auto status = ForEachLayer(
          layer_specs.size(),
          [&](size_t layer_i) {
            auto& transform = driver->layers_[layer_i].transform;
            if (layer_i != 0) {
              TENSORSTORE_ASSIGN_OR_RETURN(
                  transform,
                  std::move(transform) |
                      Dims(concat_dimension_index).TranslateTo(offset));
            }
            offset = transform.domain()[concat_dimension_index].exclusive_max();
            return absl::OkStatus();
          });
      !status.ok()) {
    return status;
  }
  return internal_stack::FinalizeStackHandle(std::move(driver),
                                             std::move(options));
}

}  // namespace internal_stack
}  // namespace tensorstore

namespace {
const tensorstore::internal::DriverRegistration<
    tensorstore::internal_stack::StackDriverSpec>
    driver_registration;

}  // namespace
