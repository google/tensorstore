// Copyright 2020 The TensorStore Authors
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

#include "tensorstore/driver/downsample/downsample.h"

#include <mutex>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "tensorstore/driver/downsample/downsample_array.h"
#include "tensorstore/driver/downsample/downsample_method_json_binder.h"
#include "tensorstore/driver/downsample/downsample_nditerable.h"
#include "tensorstore/driver/downsample/downsample_util.h"
#include "tensorstore/driver/downsample/grid_occupancy_map.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/driver/read.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/serialization/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/spec.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/sender_util.h"
#include "tensorstore/util/garbage_collection/std_vector.h"  // IWYU pragma: keep

namespace tensorstore {
namespace internal_downsample {
namespace {

using ::tensorstore::internal::DriverPtr;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::LockCollection;
using ::tensorstore::internal::NDIterable;
using ::tensorstore::internal::OpenTransactionPtr;
using ::tensorstore::internal::ReadChunk;
using ::tensorstore::internal::TransformedDriverSpec;

namespace jb = tensorstore::internal_json_binding;

/// Obtains constraints on the base domain from constraints on the downsampled
/// domain.
///
/// - The labels are copied.
///
/// - For dimensions `i` where `downsampled_factors[i] == 1`, the bounds and
///   implicit indicators are copied.
///
/// - For dimensions `i` where `downsampled_factors[i] != 1`, the bounds are
///   infinite and implicit.
///
/// \dchecks `downsampled_domain.valid()`
/// \dchecks `downsampled_domain.rank() == downsample_factors.size()`
Result<IndexDomain<>> GetBaseDomainConstraintFromDownsampledDomain(
    IndexDomain<> downsampled_domain, span<const Index> downsample_factors) {
  assert(downsampled_domain.valid());
  const DimensionIndex rank = downsampled_domain.rank();
  assert(rank == downsample_factors.size());
  IndexDomainBuilder builder(rank);
  builder.labels(downsampled_domain.labels());
  auto& implicit_lower_bounds = builder.implicit_lower_bounds();
  auto& implicit_upper_bounds = builder.implicit_upper_bounds();
  auto origin = builder.origin();
  auto shape = builder.shape();
  for (DimensionIndex i = 0; i < rank; ++i) {
    if (downsample_factors[i] != 1) {
      implicit_lower_bounds[i] = true;
      implicit_upper_bounds[i] = true;
      origin[i] = -kInfIndex;
      shape[i] = kInfSize;
    } else {
      implicit_lower_bounds[i] = downsampled_domain.implicit_lower_bounds()[i];
      implicit_upper_bounds[i] = downsampled_domain.implicit_upper_bounds()[i];
      origin[i] = downsampled_domain.origin()[i];
      shape[i] = downsampled_domain.shape()[i];
    }
  }
  return builder.Finalize();
}

class DownsampleDriverSpec
    : public internal::RegisteredDriverSpec<DownsampleDriverSpec,
                                            /*Parent=*/internal::DriverSpec> {
 public:
  constexpr static char id[] = "downsample";

  TransformedDriverSpec base;
  std::vector<Index> downsample_factors;
  DownsampleMethod downsample_method;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<internal::DriverSpec>(x), x.base,
             x.downsample_factors, x.downsample_method);
  };

  absl::Status InitializeFromBase() {
    TENSORSTORE_RETURN_IF_ERROR(
        this->schema.Set(RankConstraint{internal::GetRank(this->base)}));
    TENSORSTORE_RETURN_IF_ERROR(
        this->schema.Set(this->base.driver_spec->schema.dtype()));
    return absl::OkStatus();
  }

  absl::Status ValidateDownsampleFactors() {
    TENSORSTORE_RETURN_IF_ERROR(
        this->schema.Set(RankConstraint(this->downsample_factors.size())));
    return absl::OkStatus();
  }

  absl::Status ValidateDownsampleMethod() {
    auto dtype = this->schema.dtype();
    if (!dtype.valid()) return absl::OkStatus();
    return internal_downsample::ValidateDownsampleMethod(
        dtype, this->downsample_method);
  }

  absl::Status ApplyOptions(SpecOptions&& options) override {
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(options.dtype()));
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(options.rank()));
    auto transform = base.transform;
    if (!transform.valid()) {
      transform = tensorstore::IdentityTransform(downsample_factors.size());
    }
    if (options.domain().valid()) {
      // The original domain serves as a constraint.  Additionally, the labels
      // of all dimensions, and the bounds of non-downsampled dimensions, are
      // propagated as constraints on `base`.  The bounds of downsampled
      // dimensions cannot be propagated, since these precise bounds of `base`
      // are under-constrained.
      TENSORSTORE_RETURN_IF_ERROR(schema.Set(options.domain()));
      TENSORSTORE_ASSIGN_OR_RETURN(auto base_domain,
                                   GetBaseDomainConstraintFromDownsampledDomain(
                                       options.domain(), downsample_factors));
      TENSORSTORE_RETURN_IF_ERROR(options.Override(std::move(base_domain)));
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform, transform | AllDims().Stride(downsample_factors));
    TENSORSTORE_RETURN_IF_ERROR(options.TransformInputSpaceSchema(transform));
    return internal::TransformAndApplyOptions(base, std::move(options));
  }

  constexpr static auto default_json_binder = jb::Object(
      jb::Member("base",
                 [](auto is_loading, const auto& options, auto* obj, auto* j) {
                   return jb::Projection<&DownsampleDriverSpec::base>()(
                       is_loading,
                       JsonSerializationOptions(options, obj->schema.dtype(),
                                                obj->schema.rank()),
                       obj, j);
                 }),
      jb::Initialize([](auto* obj) { return obj->InitializeFromBase(); }),
      jb::Member("downsample_factors",
                 jb::Validate(
                     [](const auto& options, auto* obj) {
                       return obj->ValidateDownsampleFactors();
                     },
                     jb::Projection<&DownsampleDriverSpec::downsample_factors>(
                         jb::Array(jb::Integer<Index>(1))))),
      jb::Member(
          "downsample_method",
          jb::Validate(
              [](const auto& options, auto* obj) {
                return obj->ValidateDownsampleMethod();
              },
              jb::Projection<&DownsampleDriverSpec::downsample_method>())),
      jb::Initialize([](auto* obj) {
        SpecOptions base_options;
        static_cast<Schema&>(base_options) = std::exchange(obj->schema, {});
        return obj->ApplyOptions(std::move(base_options));
      }));

  Result<IndexDomain<>> GetDomain() const override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto domain,
                                 internal::GetEffectiveDomain(base));
    if (!domain.valid()) {
      return schema.domain();
    }
    if (domain.rank() != downsample_factors.size()) {
      // Should have already been validated.
      return absl::InternalError(tensorstore::StrCat(
          "Domain of base TensorStore has rank (", domain.rank(),
          ") but expected ", downsample_factors.size()));
    }
    auto downsampled_domain = internal_downsample::DownsampleDomain(
        domain, downsample_factors, downsample_method);
    return MergeIndexDomains(std::move(downsampled_domain), schema.domain());
  }

  Result<ChunkLayout> GetChunkLayout() const override {
    return internal::GetEffectiveChunkLayout(base) |
           AllDims().Stride(downsample_factors);
  }

  Result<CodecSpec> GetCodec() const override {
    return internal::GetEffectiveCodec(base);
  }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) const override {
    return {std::in_place};
  }

  Result<DimensionUnitsVector> GetDimensionUnits() const override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto dimension_units,
                                 internal::GetEffectiveDimensionUnits(base));
    if (!dimension_units.empty()) {
      span<const Index> downsample_factors = this->downsample_factors;
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto transform,
          tensorstore::IdentityTransform(downsample_factors.size()) |
              tensorstore::AllDims().Stride(downsample_factors));
      dimension_units =
          TransformOutputDimensionUnits(transform, std::move(dimension_units));
    }
    return dimension_units;
  }

  kvstore::Spec GetKvstore() const override {
    return base.driver_spec->GetKvstore();
  }

  Future<internal::Driver::Handle> Open(
      internal::OpenTransactionPtr transaction,
      ReadWriteMode read_write_mode) const override {
    if (!!(read_write_mode & ReadWriteMode::write)) {
      return absl::InvalidArgumentError("only reading is supported");
    }
    return MapFutureValue(
        InlineExecutor{},
        [spec = internal::DriverSpec::PtrT<const DownsampleDriverSpec>(this)](
            internal::Driver::Handle handle)
            -> Result<internal::Driver::Handle> {
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto downsampled_handle,
              MakeDownsampleDriver(std::move(handle), spec->downsample_factors,
                                   spec->downsample_method));
          // Validate the domain constraint specified by the schema, if any.
          // All other schema constraints are propagated to the base driver, and
          // therefore aren't checked here.
          if (auto domain = spec->schema.domain(); domain.valid()) {
            TENSORSTORE_RETURN_IF_ERROR(
                MergeIndexDomains(domain,
                                  downsampled_handle.transform.domain()),
                tensorstore::MaybeAnnotateStatus(
                    _, "downsampled domain does not match domain in schema"));
          }
          return downsampled_handle;
        },
        internal::OpenDriver(std::move(transaction), base,
                             ReadWriteMode::read));
  }
};

class DownsampleDriver
    : public internal::RegisteredDriver<DownsampleDriver,
                                        /*Parent=*/internal::Driver> {
 public:
  Result<TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override {
    auto driver_spec = internal::DriverSpec::Make<DownsampleDriverSpec>();
    driver_spec->context_binding_state_ = ContextBindingState::bound;
    TENSORSTORE_ASSIGN_OR_RETURN(
        driver_spec->base,
        base_driver_->GetBoundSpec(std::move(transaction), base_transform_));
    driver_spec->downsample_factors = downsample_factors_;
    driver_spec->downsample_method = downsample_method_;
    TENSORSTORE_RETURN_IF_ERROR(driver_spec->InitializeFromBase());
    TransformedDriverSpec spec;
    spec.transform = transform;
    spec.driver_spec = std::move(driver_spec);
    return spec;
  }

  Result<ChunkLayout> GetChunkLayout(IndexTransformView<> transform) override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto strided_base_transform,
                                 GetStridedBaseTransform());
    return base_driver_->GetChunkLayout(strided_base_transform) | transform;
  }

  Result<CodecSpec> GetCodec() override { return base_driver_->GetCodec(); }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) override {
    if (downsample_method_ == DownsampleMethod::kStride) {
      // Stride-based downsampling just relies on the normal `IndexTransform`
      // machinery.
      TENSORSTORE_ASSIGN_OR_RETURN(auto strided_transform,
                                   GetStridedBaseTransform() | transform);
      return base_driver_->GetFillValue(strided_transform);
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto propagated_transform,
        internal_downsample::PropagateIndexTransformDownsampling(
            transform, base_transform_.domain().box(), downsample_factors_));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto propagated_base_transform,
        ComposeTransforms(base_transform_, propagated_transform.transform));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto fill_value, base_driver_->GetFillValue(propagated_base_transform));
    if (!fill_value.valid()) return {std::in_place};
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto broadcast_fill_value,
        BroadcastArray(std::move(fill_value),
                       propagated_base_transform.domain().box()));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto downsampled_fill_value,
        internal_downsample::DownsampleArray(
            broadcast_fill_value, propagated_transform.input_downsample_factors,
            downsample_method_));
    return UnbroadcastArray(downsampled_fill_value);
  }

  Result<DimensionUnitsVector> GetDimensionUnits() override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto dimension_units,
                                 base_driver_->GetDimensionUnits());
    TENSORSTORE_ASSIGN_OR_RETURN(auto strided_base_transform,
                                 GetStridedBaseTransform());
    return TransformOutputDimensionUnits(strided_base_transform,
                                         std::move(dimension_units));
  }

  KvStore GetKvstore() override { return base_driver_->GetKvstore(); }

  Result<IndexTransform<>> GetStridedBaseTransform() {
    return base_transform_ | tensorstore::AllDims().Stride(downsample_factors_);
  }

  explicit DownsampleDriver(DriverPtr base, IndexTransform<> base_transform,
                            span<const Index> downsample_factors,
                            DownsampleMethod downsample_method)
      : base_driver_(std::move(base)),
        base_transform_(std::move(base_transform)),
        downsample_factors_(downsample_factors.begin(),
                            downsample_factors.end()),
        downsample_method_(downsample_method) {}

  DataType dtype() override { return base_driver_->dtype(); }
  DimensionIndex rank() override { return base_transform_.input_rank(); }

  Executor data_copy_executor() override {
    return base_driver_->data_copy_executor();
  }

  void Read(OpenTransactionPtr transaction, IndexTransform<> transform,
            AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver)
      override;

  Future<IndexTransform<>> ResolveBounds(OpenTransactionPtr transaction,
                                         IndexTransform<> transform,
                                         ResolveBoundsOptions options) override;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.base_driver_, x.base_transform_, x.downsample_factors_,
             x.downsample_method_);
  };

  DriverPtr base_driver_;
  IndexTransform<> base_transform_;
  std::vector<Index> downsample_factors_;
  DownsampleMethod downsample_method_;
};

Future<IndexTransform<>> DownsampleDriver::ResolveBounds(
    OpenTransactionPtr transaction, IndexTransform<> transform,
    ResolveBoundsOptions options) {
  return MapFutureValue(
      InlineExecutor{},
      [self = IntrusivePtr<DownsampleDriver>(this),
       transform = std::move(transform)](
          IndexTransform<> base_transform) -> Result<IndexTransform<>> {
        Box<dynamic_rank(internal::kNumInlinedDims)> downsampled_bounds(
            base_transform.input_rank());
        internal_downsample::DownsampleBounds(
            base_transform.domain().box(), downsampled_bounds,
            self->downsample_factors_, self->downsample_method_);
        return tensorstore::PropagateBoundsToTransform(
            downsampled_bounds, base_transform.implicit_lower_bounds(),
            base_transform.implicit_upper_bounds(), std::move(transform));
      },
      base_driver_->ResolveBounds(std::move(transaction), base_transform_,
                                  std::move(options)));
}

/// Asynchronous operation state for `DownsampleDriver::Read`.
///
/// Reading proceeds as follows:
///
/// 1. Call `base_driver_->ResolveBounds` to obtain an updated domain for the
///    base driver.
///
/// 2. Compute the `base_request_transform` for the base driver by calling
///    `PropagateIndexTransformDownsampling` using the updated base domain and
///    composing on to the existing `base_transform_`.
///
/// 3. Issue a read request to the base driver with the
///    `base_request_transform`.
///
/// 4. Each `ReadChunk` returned by the base driver is handled in one of two
///    ways:
///
///    4a. If the chunk can be independently downsampled, then a downsampled
///        view is directly emitted without any additional buffering.  This
///        allows incremental data processing and is expected to be a common
///        case.  For example, when downsampling a chunked array where the
///        chunks are aligned to downsample block boundaries, this will hold.
///        All such chunks are recorded in the `independently_emitted_chunks_`
///        tracker in case not all chunks can be independently downsampled.
///
///    4b. If the chunk cannot be independently downsampled, `data_buffer_` is
///        allocated with a domain equal to the domain of the
///        `base_request_transform` if it has not already been, and the chunk is
///        copied to it (without downsampling yet).
///
///    Ideally, we would either emit all chunks independently (4a) or copy all
///    chunks to `data_buffer_` (4b).  Unfortunately, we receive chunks from the
///    base driver as a stream and cannot know in advance whether it will be
///    possible to emit all of them independently.  For that reason, it is
///    necessary to record the bounds of all chunks emitted independently.
///
/// 5. Once all chunks have been received from the `base_driver_`, if
///    `data_buffer_` has been allocated:
///
///    5a. If no chunks have been emitted independently, just emit a single
///        downsampled view of the entire `data_buffer_`.
///
///    5b. If some chunks have been emitted independently, they need to be
///        excluded.  To do that, given that there is no guarantee that the
///        independently emitted chunks are in a regular grid, we divide the
///        domain of `data_buffer_` into a non-regular grid, adding grid lines
///        to each dimension as needed to include all chunk boundaries, and
///        compute a `bool` array indicating which grid cells are covered by
///        independently-emitted chunks.  Then for each non-covered grid cell,
///        we emit a separate chunk that provides a downsampled view of that
///        cell of `data_buffer_`.
struct ReadState : public internal::AtomicReferenceCount<ReadState> {
  IntrusivePtr<DownsampleDriver> self_;

  /// Receiver of downsampled chunks.
  AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver_;

  /// Protects access to most other members.
  absl::Mutex mutex_;

  /// Array with domain `base_transform_domain_.box()` with data type
  /// `self_->base_driver_->dtype()`.  The layout and data pointer must not
  /// be changed except while holding `mutex_`.  However, disjoint portions of
  /// the array itself, once allocated, may be written concurrently by multiple
  /// threads.  This is only allocated once the first chunk that cannot be
  /// emitted independently is received.  In many cases it will not need to be
  /// allocated at all.
  SharedOffsetArray<void> data_buffer_;

  /// Number of elements out of `base_transform_domain_.num_elements()` not yet
  /// emitted as independent chunks or copied to `data_buffer_`.  Once this
  /// reaches 0, any chunks from `data_buffer_` will be emitted.
  Index remaining_elements_;

  /// List of chunks that were emitted independently (not copied to
  /// `data_buffer_`).
  internal_downsample::GridOccupancyTracker independently_emitted_chunks_;

  /// Downsample factors for each dimension of `base_transform_domain_`.
  /// Constant.
  absl::InlinedVector<Index, internal::kNumInlinedDims> downsample_factors_;

  /// The first `original_input_rank_` dimensions of `base_transform_domain_`
  /// correspond to the requested domain; the remaining dimensions are synthetic
  /// dimensions added by `PropagateIndexTransformDownsampling` that will be
  /// downsampled to a single element.
  DimensionIndex original_input_rank_;

  /// The domain of the base request transform computed by
  /// `PropagateIndexTransformDownsampling` from the read request on the
  /// downsampled domain.
  IndexDomain<> base_transform_domain_;

  /// Cancellation function provided by the base driver, if it has been set and
  /// not yet called.
  AnyCancelReceiver on_cancel_;

  /// Error that has occurred.
  absl::Status error_;

  /// Indicates that `set_done` has been received from the base driver.
  bool done_signal_received_ = false;

  /// Indicates that either `set_done` or `set_error` has been called on
  /// `receiver_`.
  bool done_sent_ = false;

  /// Indicates that cancellation was requested, either explicitly by
  /// `receiver_` or implicitly due to `error_` being set.
  bool canceled_ = false;

  /// Number of chunks currently being processed.  While this is non-zero, we
  /// don't update `done_sent_`.
  size_t chunks_in_progress_ = 0;

  void Cancel() {
    std::lock_guard<ReadState> guard(*this);
    canceled_ = true;
    // `unlock()` takes care of calling `on_cancel_`, if applicable.
  }

  /// Locks `mutex_`.
  void lock() ABSL_NO_THREAD_SAFETY_ANALYSIS { mutex_.Lock(); }

  /// Unlocks `mutex_`, and then sends any deferred notifications.
  void unlock() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    bool has_error = !error_.ok();
    bool send_done = !done_sent_ && chunks_in_progress_ == 0 &&
                     (done_signal_received_ || has_error);
    if (send_done) done_sent_ = true;
    AnyCancelReceiver on_cancel;
    if (canceled_ && on_cancel_) {
      on_cancel = std::move(on_cancel_);
    }
    mutex_.Unlock();
    if (on_cancel) on_cancel();
    if (!send_done) return;
    if (has_error) {
      execution::set_error(receiver_, error_);
    } else {
      execution::set_done(receiver_);
    }
    execution::set_stopping(receiver_);
  }

  void SetError(absl::Status error, size_t decrement_chunks_in_progress = 0) {
    std::lock_guard<ReadState> guard(*this);
    chunks_in_progress_ -= decrement_chunks_in_progress;
    if (!error_.ok()) return;
    error_ = std::move(error);
    canceled_ = true;
  }

  /// Emits a `ReadChunk` containing a downsample view of the `base_domain`
  /// region of `data_buffer_`.
  ///
  /// The caller must own a `chunks_in_progress_` reference.
  void EmitBufferedChunkForBox(BoxView<> base_domain);

  /// Emits read chunks containing downsampled views of the portions of
  /// `data_buffer_` that have been written (i.e. not emitted as independent
  /// chunks).
  ///
  /// The caller implicitly transfers ownership of a `chunks_in_progress_`
  /// reference.
  void EmitBufferedChunks();
};

/// Implementation of the `internal::ReadChunk::Impl` Poly interface that
/// provides a downsampled view of `ReadState::data_buffer_`.
struct BufferedReadChunkImpl {
  internal::IntrusivePtr<ReadState> state_;

  absl::Status operator()(LockCollection& lock_collection) const {
    // No locks required, since `data_buffer_` is immutable by the time this
    // chunk is emitted.
    return absl::OkStatus();
  }

  Result<NDIterable::Ptr> operator()(internal::ReadChunk::BeginRead,
                                     IndexTransform<> chunk_transform,
                                     internal::Arena* arena) const {
    // `chunk_transform` maps from a copy space to the *downsampled* bounds of
    // `state_->base_transform_domain_`.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto propagated,
        internal_downsample::PropagateIndexTransformDownsampling(
            chunk_transform, state_->data_buffer_.domain(),
            state_->downsample_factors_));
    // The domain of `propagated.transform`, when downsampled by
    // `propagated.input_downsample_factors`, matches
    // `chunk_transform.domain()`.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto transformed_array,
        MakeTransformedArray(state_->data_buffer_,
                             std::move(propagated.transform)));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto base_nditerable,
        GetTransformedArrayNDIterable(transformed_array, arena));
    // Return a downsampled view of `state_->base_buffer_`.  Note that
    // `propagated.transform` may have additional synthetic input dimensions
    // beyond `chunk_transform.input_rank()`, but those are truncated by
    // `DownsampleNDIterable`.
    return internal_downsample::DownsampleNDIterable(
        std::move(base_nditerable), transformed_array.domain().box(),
        propagated.input_downsample_factors, state_->self_->downsample_method_,
        chunk_transform.input_rank(), arena);
  }
};

/// Returns an identity transform from `base_domain.rank()` to `request_rank`,
/// where the domain is equal to `base_domain` downsampled by
/// `downsample_factors`.
///
/// If `request_rank < base_domain.rank()`, the extra input dimensions are
/// assumed to be singleton dimensions and are not used by any output index
/// maps.
IndexTransform<> GetDownsampledRequestIdentityTransform(
    BoxView<> base_domain, span<const Index> downsample_factors,
    DownsampleMethod downsample_method, DimensionIndex request_rank) {
  // Construct the `request_transform` to send to the receiver, which is simply
  // an identity transform over the first `request_rank` dimensions.  The
  // remaining input dimensions are synthetic dimensions added by
  // `PropagateIndexTransformDownsampling` with an extent of 1 since they are
  // downsampled to a single element.
  assert(base_domain.rank() == downsample_factors.size());
  assert(request_rank <= base_domain.rank());
  IndexTransformBuilder builder(base_domain.rank(), request_rank);
  internal_downsample::DownsampleBounds(base_domain, builder.input_bounds(),
                                        downsample_factors, downsample_method);
  builder.output_identity_transform();
  return builder.Finalize().value();
}

void ReadState::EmitBufferedChunkForBox(BoxView<> base_domain) {
  auto request_transform = GetDownsampledRequestIdentityTransform(
      base_domain, downsample_factors_, self_->downsample_method_,
      original_input_rank_);
  ReadChunk downsampled_chunk;
  downsampled_chunk.transform =
      IdentityTransform(request_transform.domain().box());
  downsampled_chunk.impl = BufferedReadChunkImpl{IntrusivePtr<ReadState>(this)};
  execution::set_value(receiver_, std::move(downsampled_chunk),
                       std::move(request_transform));
}

void ReadState::EmitBufferedChunks() {
  if (independently_emitted_chunks_.occupied_chunks.empty()) {
    // No independently-emitted chunks, can just emit a single chunk for the
    // entire `data_buffer_`.
    EmitBufferedChunkForBox(base_transform_domain_.box());
  } else {
    // Need to partition domain to skip chunks that have already been emitted
    // (and aren't present in `state.data_buffer_`).
    internal_downsample::GridOccupancyMap emitted_chunk_map(
        std::move(independently_emitted_chunks_), base_transform_domain_.box());

    // Iterate over grid cells that haven't been independently emitted.
    const DimensionIndex rank = emitted_chunk_map.rank();
    absl::FixedArray<Index, internal::kNumInlinedDims> grid_cell(rank);
    Box<dynamic_rank(internal::kNumInlinedDims)> grid_cell_domain;
    grid_cell_domain.set_rank(rank);
    emitted_chunk_map.InitializeCellIterator(grid_cell);
    do {
      if (!emitted_chunk_map.GetGridCellDomain(grid_cell, grid_cell_domain)) {
        continue;
      }
      EmitBufferedChunkForBox(grid_cell_domain);
    } while (emitted_chunk_map.AdvanceCellIterator(grid_cell));
  }
  {
    std::lock_guard<ReadState> guard(*this);
    --chunks_in_progress_;
  }
}

/// Implementation of the `internal::ReadChunk::Impl` Poly interface that
/// provides a downsampled view of another `ReadChunk`.
struct IndependentReadChunkImpl {
  internal::IntrusivePtr<ReadState> state_;
  /// Base chunk to downsample.  The domain of `base_chunk_.transform` is a
  /// sub-region of `state_.base_transform_domain_`.  Note that the
  /// `base_driver_` did not necessarily provide `base_chunk_.transform` in this
  /// form, but we only use `IndependentReadChunkImpl` with chunks that can be
  /// converted to that.  Otherwise, the chunk is copied to
  /// `state_.data_buffer_`.
  internal::ReadChunk base_chunk_;

  absl::Status operator()(LockCollection& lock_collection) {
    // Register any locks required by `base_chunk_`.
    return base_chunk_.impl(lock_collection);
  }

  Result<NDIterable::Ptr> operator()(internal::ReadChunk::BeginRead,
                                     IndexTransform<> chunk_transform,
                                     internal::Arena* arena) {
    // `chunk_transform` maps from a copy space to the *downsampled* bounds of
    // `state_->base_transform_domain_`.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto propagated,
        internal_downsample::PropagateIndexTransformDownsampling(
            chunk_transform, state_->base_transform_domain_.box(),
            state_->downsample_factors_));
    // The domain of `propagated.transform`, when downsampled by
    // `propagated.input_downsample_factors`, matches
    // `chunk_transform.domain()`.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto base_transform,
        ComposeTransforms(base_chunk_.transform, propagated.transform));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto base_nditerable,
        base_chunk_.impl(internal::ReadChunk::BeginRead{},
                         std::move(base_transform), arena));
    // Return downsampled view of `base_nditerable`.  Note that
    // `propagated.transform` may have additional synthetic input dimensions
    // beyond `chunk_transform.input_rank()`, but those are truncated by
    // `DownsampleNDIterable`.
    return internal_downsample::DownsampleNDIterable(
        std::move(base_nditerable), propagated.transform.domain().box(),
        propagated.input_downsample_factors, state_->self_->downsample_method_,
        chunk_transform.input_rank(), arena);
  }
};

/// Attempts to emit a `ReadChunk` from the base driver independently.
///
/// The caller must own a `state.chunks_in_progress_` reference.  If `false` is
/// returned, the caller retains that reference.  If `true` is returned, the
/// caller has given up ownership of that reference.
bool MaybeEmitIndependentReadChunk(
    ReadState& state, ReadChunk& base_chunk,
    IndexTransformView<> base_request_transform) {
  if (!internal_downsample::CanDownsampleIndexTransform(
          base_request_transform, state.base_transform_domain_.box(),
          state.downsample_factors_)) {
    // Chunk cannot be independently emitted due to an incompatible transform.
    return false;
  }

  // Normalize `base_chunk.transform` such that its input domain is a sub-region
  // of `state.base_transform_domain_`, and the corresponding request transform
  // is an identity transform.  While in general this normalization would not be
  // possible, we are guaranteed to be able to do this because
  // `CanDownsampleIndexTransform` returned `true`.

  TENSORSTORE_ASSIGN_OR_RETURN(auto inverse_request_transform,
                               InverseTransform(base_request_transform), false);
  TENSORSTORE_ASSIGN_OR_RETURN(
      base_chunk.transform,
      ComposeTransforms(base_chunk.transform, inverse_request_transform),
      false);
  const Index num_elements = base_chunk.transform.domain().num_elements();
  bool emit_buffered_chunk;
  {
    absl::MutexLock lock(&state.mutex_);
    bool has_data_buffer =
        state.data_buffer_.byte_strided_origin_pointer() != nullptr;
    bool remaining_data = (state.remaining_elements_ -= num_elements) != 0;
    emit_buffered_chunk = (!remaining_data && has_data_buffer);
    // If there may potentially be non-independently-emitted chunks copied to
    // `state.data_buffer_`, we need to record this chunk in
    // `independently_emitted_chunks_`.
    if (has_data_buffer || remaining_data) {
      state.independently_emitted_chunks_.MarkOccupied(
          base_chunk.transform.domain().box());
    }
  }

  internal::ReadChunk downsampled_chunk;
  auto request_transform = GetDownsampledRequestIdentityTransform(
      base_chunk.transform.domain().box(), state.downsample_factors_,
      state.self_->downsample_method_, state.original_input_rank_);
  downsampled_chunk.impl = IndependentReadChunkImpl{
      internal::IntrusivePtr<ReadState>(&state), std::move(base_chunk)};
  downsampled_chunk.transform =
      IdentityTransform(request_transform.domain().box());
  execution::set_value(state.receiver_, std::move(downsampled_chunk),
                       request_transform);
  if (emit_buffered_chunk) {
    // This method is not called from the `data_copy_executor`.  Because it may
    // involve a significant amount of computation to exclude the
    // independently-emitted chunks, we ensure `EmitBufferedChunks` is run on
    // the executor.  We implicitly transfer ownership of a
    // `chunks_in_progress_` reference.
    state.self_->data_copy_executor()(
        [state = internal::IntrusivePtr<ReadState>(&state)] {
          state->EmitBufferedChunks();
        });
  } else {
    std::lock_guard<ReadState> guard(state);
    --state.chunks_in_progress_;
  }
  return true;
}

/// `Driver::ReadChunkReceiver` implementation passed to `base_driver_->Read`
/// that handles downsampling the stream of chunks from the base driver.
struct ReadReceiverImpl {
  internal::IntrusivePtr<ReadState> state_;

  void set_starting(AnyCancelReceiver on_cancel) {
    {
      absl::MutexLock lock(&state_->mutex_);
      if (!state_->canceled_) {
        state_->on_cancel_ = std::move(on_cancel);
        return;
      }
    }
    if (on_cancel) on_cancel();
  }

  void set_value(ReadChunk chunk, IndexTransform<> cell_transform) {
    if (cell_transform.domain().box().is_empty()) return;
    {
      absl::MutexLock lock(&state_->mutex_);
      if (state_->canceled_) return;
      ++state_->chunks_in_progress_;
    }
    // Check whether this chunk can be downsampled independently.  That doesn't
    // involve any significant data copying, so don't bother deferring the work
    // to an executor.
    if (MaybeEmitIndependentReadChunk(*state_, chunk, cell_transform)) return;
    state_->self_->data_copy_executor()([state = state_,
                                         chunk = std::move(chunk),
                                         cell_transform = std::move(
                                             cell_transform)]() mutable {
      const Index num_elements = cell_transform.domain().num_elements();
      {
        std::lock_guard<ReadState> guard(*state);
        if (state->canceled_) {
          --state->chunks_in_progress_;
          return;
        }
        if (state->data_buffer_.byte_strided_origin_pointer() == nullptr) {
          state->data_buffer_ =
              AllocateArray(state->base_transform_domain_.box(), c_order,
                            default_init, state->self_->base_driver_->dtype());
        }
      }
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto transformed_data_buffer,
          MakeTransformedArray(state->data_buffer_, std::move(cell_transform)),
          state->SetError(_, 1));
      TENSORSTORE_RETURN_IF_ERROR(
          internal::CopyReadChunk(chunk.impl, chunk.transform,
                                  transformed_data_buffer),
          state->SetError(_, 1));
      {
        std::lock_guard<ReadState> guard(*state);
        bool elements_done = (state->remaining_elements_ -= num_elements) == 0;
        if (state->canceled_ || !elements_done) {
          --state->chunks_in_progress_;
          return;
        }
      }
      state->EmitBufferedChunks();
    });
  }

  void set_error(absl::Status status) { state_->SetError(std::move(status)); }

  void set_done() {
    std::lock_guard<ReadState> guard(*state_);
    state_->done_signal_received_ = true;
  }

  void set_stopping() {
    absl::MutexLock lock(&state_->mutex_);
    state_->on_cancel_ = {};
  }
};

void DownsampleDriver::Read(
    OpenTransactionPtr transaction, IndexTransform<> transform,
    AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver) {
  if (downsample_method_ == DownsampleMethod::kStride) {
    // Stride-based downsampling just relies on the normal `IndexTransform`
    // machinery.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto strided_transform, GetStridedBaseTransform() | transform,
        execution::set_error(FlowSingleReceiver{std::move(receiver)}, _));
    base_driver_->Read(std::move(transaction), std::move(strided_transform),
                       std::move(receiver));
    return;
  }
  auto base_resolve_future = base_driver_->ResolveBounds(
      transaction, base_transform_, {fix_resizable_bounds});
  auto state = internal::MakeIntrusivePtr<ReadState>();
  state->self_.reset(this);
  state->original_input_rank_ = transform.input_rank();
  state->receiver_ = std::move(receiver);
  execution::set_starting(state->receiver_,
                          [state = state.get()] { state->Cancel(); });
  std::move(base_resolve_future)
      .ExecuteWhenReady(
          [state = std::move(state), transaction = std::move(transaction),
           transform =
               std::move(transform)](ReadyFuture<IndexTransform<>> future) {
            auto& r = future.result();
            if (!r.ok()) {
              state->SetError(std::move(r.status()));
              return;
            }
            IndexTransform<> base_transform = std::move(*r);
            TENSORSTORE_ASSIGN_OR_RETURN(
                auto propagated,
                internal_downsample::PropagateIndexTransformDownsampling(
                    transform, base_transform.domain().box(),
                    state->self_->downsample_factors_),
                state->SetError(_));
            // The domain of `propagated.transform`, when downsampled by
            // `propagated.input_downsample_factors`, matches
            // `transform.domain()`.

            // Compute the read request for `base_driver_`.
            TENSORSTORE_ASSIGN_OR_RETURN(
                propagated.transform,
                ComposeTransforms(state->self_->base_transform_,
                                  propagated.transform),
                state->SetError(_));
            state->remaining_elements_ =
                propagated.transform.domain().num_elements();
            state->downsample_factors_ =
                std::move(propagated.input_downsample_factors);
            state->base_transform_domain_ = propagated.transform.domain();
            auto* state_ptr = state.get();
            state_ptr->self_->base_driver_->Read(
                std::move(transaction), std::move(propagated.transform),
                ReadReceiverImpl{std::move(state)});
          });
}

const internal::DriverRegistration<DownsampleDriverSpec> driver_registration;

}  // namespace
}  // namespace internal_downsample

namespace internal {

Result<Driver::Handle> MakeDownsampleDriver(
    Driver::Handle base, span<const Index> downsample_factors,
    DownsampleMethod downsample_method) {
  if (downsample_factors.size() != base.transform.input_rank()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Number of downsample factors (", downsample_factors.size(),
        ") does not match TensorStore rank (", base.transform.input_rank(),
        ")"));
  }
  if (!(base.driver.read_write_mode() & ReadWriteMode::read)) {
    return absl::InvalidArgumentError(
        "Cannot downsample write-only TensorStore");
  }
  if (std::any_of(downsample_factors.begin(), downsample_factors.end(),
                  [](Index factor) { return factor < 1; })) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Downsample factors ", downsample_factors, " are not all positive"));
  }
  TENSORSTORE_RETURN_IF_ERROR(internal_downsample::ValidateDownsampleMethod(
      base.driver->dtype(), downsample_method));
  auto downsampled_domain =
      internal_downsample::GetDownsampledDomainIdentityTransform(
          base.transform.domain(), downsample_factors, downsample_method);
  base.driver =
      internal::MakeReadWritePtr<internal_downsample::DownsampleDriver>(
          ReadWriteMode::read, std::move(base.driver),
          std::move(base.transform), downsample_factors, downsample_method);
  base.transform = std::move(downsampled_domain);
  return base;
}

}  // namespace internal

Result<Spec> Downsample(const Spec& base_spec,
                        span<const Index> downsample_factors,
                        DownsampleMethod downsample_method) {
  using internal_spec::SpecAccess;
  Spec downsampled_spec;
  auto& impl = SpecAccess::impl(downsampled_spec);
  auto driver_spec =
      internal::DriverSpec::Make<internal_downsample::DownsampleDriverSpec>();
  driver_spec->context_binding_state_ = base_spec.context_binding_state();
  driver_spec->base = SpecAccess::impl(base_spec);
  TENSORSTORE_RETURN_IF_ERROR(driver_spec->InitializeFromBase());
  driver_spec->downsample_factors.assign(downsample_factors.begin(),
                                         downsample_factors.end());
  driver_spec->downsample_method = downsample_method;
  TENSORSTORE_RETURN_IF_ERROR(driver_spec->ValidateDownsampleFactors());
  TENSORSTORE_RETURN_IF_ERROR(driver_spec->ValidateDownsampleMethod());
  impl.driver_spec = std::move(driver_spec);
  if (base_spec.transform().valid()) {
    impl.transform = internal_downsample::GetDownsampledDomainIdentityTransform(
        base_spec.transform().domain(), downsample_factors, downsample_method);
  }
  return downsampled_spec;
}

}  // namespace tensorstore
