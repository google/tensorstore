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

#include "tensorstore/driver/cast/cast.h"

#include <cassert>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/lock_collection.h"
#include "tensorstore/internal/nditerable_data_type_conversion.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/open_options.h"
#include "tensorstore/rank.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/schema.h"
#include "tensorstore/spec.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_cast_driver {
namespace {

using ::tensorstore::internal::Arena;
using ::tensorstore::internal::DataTypeConversionLookupResult;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::NDIterable;
using ::tensorstore::internal::OpenTransactionPtr;
using ::tensorstore::internal::ReadChunk;
using ::tensorstore::internal::TransformedDriverSpec;
using ::tensorstore::internal::WriteChunk;

namespace jb = tensorstore::internal_json_binding;

class CastDriverSpec
    : public internal::RegisteredDriverSpec<CastDriverSpec,
                                            /*Parent=*/internal::DriverSpec> {
 public:
  constexpr static const char id[] = "cast";

  TransformedDriverSpec base;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(internal::BaseCast<internal::DriverSpec>(x), x.base);
  };

  OpenMode open_mode() const override { return base.driver_spec->open_mode(); }

  absl::Status ApplyOptions(SpecOptions&& options) override {
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(options.dtype()));
    options.Override(DataType()).IgnoreError();
    return internal::TransformAndApplyOptions(base, std::move(options));
  }

  constexpr static auto default_json_binder = jb::Object(
      jb::Member("base",
                 [](auto is_loading, const auto& options, auto* obj, auto* j) {
                   return jb::Projection<&CastDriverSpec::base>()(
                       is_loading,
                       JsonSerializationOptions(options, DataType(),
                                                obj->schema.rank()),
                       obj, j);
                 }),
      jb::Initialize([](auto* obj) -> absl::Status {
        if (obj->base.transform.valid()) {
          TENSORSTORE_RETURN_IF_ERROR(obj->schema.Set(
              RankConstraint{obj->base.transform.input_rank()}));
        }
        DataType dtype = obj->schema.dtype();
        DimensionIndex rank = obj->schema.rank();
        SpecOptions base_options;
        static_cast<Schema&>(base_options) = std::exchange(obj->schema, {});
        obj->schema.Set(dtype).IgnoreError();
        obj->schema.Set(RankConstraint{rank}).IgnoreError();
        return obj->ApplyOptions(std::move(base_options));
      }));

  Result<IndexDomain<>> GetDomain() const override {
    return internal::GetEffectiveDomain(base);
  }

  Result<ChunkLayout> GetChunkLayout() const override {
    return internal::GetEffectiveChunkLayout(base);
  }

  Result<CodecSpec> GetCodec() const override {
    return internal::GetEffectiveCodec(base);
  }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto adjusted_transform,
        tensorstore::ComposeOptionalTransforms(base.transform, transform));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto fill_value, base.driver_spec->GetFillValue(adjusted_transform));
    if (!fill_value.valid()) return {std::in_place};
    auto dtype = schema.dtype();
    if (dtype == fill_value.dtype()) return fill_value;
    // Check if we can convert.
    auto converter = internal::GetDataTypeConverter(fill_value.dtype(), dtype);
    if (!(converter.flags & DataTypeConversionFlags::kSupported)) {
      return {std::in_place};
    }
    return MakeCopy(fill_value, skip_repeated_elements, dtype);
  }

  Result<DimensionUnitsVector> GetDimensionUnits() const override {
    return internal::GetEffectiveDimensionUnits(base);
  }

  kvstore::Spec GetKvstore() const override {
    return base.driver_spec->GetKvstore();
  }

  Result<TransformedDriverSpec> GetBase(
      IndexTransformView<> transform) const override {
    TransformedDriverSpec new_base;
    TENSORSTORE_ASSIGN_OR_RETURN(
        new_base.transform,
        ComposeOptionalTransforms(base.transform, transform));
    new_base.driver_spec = base.driver_spec;
    return new_base;
  }

  Future<internal::Driver::Handle> Open(
      internal::DriverOpenRequest request) const override {
    DataType target_dtype = schema.dtype();
    if (!target_dtype.valid()) {
      return absl::InvalidArgumentError("dtype must be specified");
    }
    auto read_write_mode = request.read_write_mode;
    return MapFutureValue(
        InlineExecutor{},
        [target_dtype, read_write_mode](internal::Driver::Handle handle)
            -> Result<internal::Driver::Handle> {
          return MakeCastDriver(std::move(handle), target_dtype,
                                read_write_mode);
        },
        internal::OpenDriver(base, std::move(request)));
  }
};

class CastDriver
    : public internal::RegisteredDriver<CastDriver,
                                        /*Parent=*/internal::Driver> {
 public:
  Result<TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override {
    auto driver_spec = internal::DriverSpec::Make<CastDriverSpec>();
    driver_spec->context_binding_state_ = ContextBindingState::bound;
    TENSORSTORE_ASSIGN_OR_RETURN(
        driver_spec->base,
        base_driver_->GetBoundSpec(std::move(transaction), transform));
    driver_spec->schema.Set(target_dtype_).IgnoreError();
    const DimensionIndex base_rank = base_driver_->rank();
    driver_spec->schema.Set(RankConstraint{base_rank}).IgnoreError();
    TransformedDriverSpec spec;
    spec.transform = std::exchange(driver_spec->base.transform, {});
    spec.driver_spec = std::move(driver_spec);
    return spec;
  }

  Result<ChunkLayout> GetChunkLayout(IndexTransformView<> transform) override {
    return base_driver_->GetChunkLayout(transform);
  }

  Result<CodecSpec> GetCodec() override { return base_driver_->GetCodec(); }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) override {
    if (!(input_conversion_.flags & DataTypeConversionFlags::kSupported)) {
      // The conversion from `base_driver_->dtype()` to `target_dtype_` is not
      // supported (the driver is in write-only mode).  Therefore, just return
      // an unknown fill value.
      return {std::in_place};
    }

    TENSORSTORE_ASSIGN_OR_RETURN(auto base_fill_value,
                                 base_driver_->GetFillValue(transform));
    if (!base_fill_value.valid()) return {std::in_place};
    if (base_fill_value.dtype() == target_dtype_) {
      return base_fill_value;
    }
    return tensorstore::MakeCopy(base_fill_value, skip_repeated_elements,
                                 target_dtype_);
  }

  Result<DimensionUnitsVector> GetDimensionUnits() override {
    return base_driver_->GetDimensionUnits();
  }

  KvStore GetKvstore(const Transaction& transaction) override {
    return base_driver_->GetKvstore(transaction);
  }

  Result<internal::DriverHandle> GetBase(
      ReadWriteMode read_write_mode, IndexTransformView<> transform,
      const Transaction& transaction) override {
    internal::DriverHandle base_handle;
    base_handle.driver = base_driver_;
    base_handle.driver.set_read_write_mode(read_write_mode);
    base_handle.transform = transform;
    base_handle.transaction = transaction;
    return base_handle;
  }

  Future<ArrayStorageStatistics> GetStorageStatistics(
      GetStorageStatisticsRequest request) override {
    return base_driver_->GetStorageStatistics(std::move(request));
  }

  explicit CastDriver(internal::DriverPtr base, DataType target_dtype,
                      DataTypeConversionLookupResult input_conversion,
                      DataTypeConversionLookupResult output_conversion)
      : base_driver_(std::move(base)),
        target_dtype_(target_dtype),
        input_conversion_(input_conversion),
        output_conversion_(output_conversion) {}

  DataType dtype() override { return target_dtype_; }
  DimensionIndex rank() override { return base_driver_->rank(); }

  Executor data_copy_executor() override {
    return base_driver_->data_copy_executor();
  }

  void Read(ReadRequest request,
            AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver)
      override;

  void Write(WriteRequest request,
             AnyFlowReceiver<absl::Status, WriteChunk, IndexTransform<>>
                 receiver) override;

  Future<IndexTransform<>> ResolveBounds(
      ResolveBoundsRequest request) override {
    return base_driver_->ResolveBounds(std::move(request));
  }

  Future<IndexTransform<>> Resize(ResizeRequest request) override {
    return base_driver_->Resize(std::move(request));
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.base_driver_, x.target_dtype_, x.input_conversion_,
             x.output_conversion_);
  };

  internal::DriverPtr base_driver_;
  DataType target_dtype_;
  DataTypeConversionLookupResult input_conversion_;
  DataTypeConversionLookupResult output_conversion_;
};

// Implementation of `tensorstore::internal::ReadChunk::Impl` Poly
// interface.
struct ReadChunkImpl {
  IntrusivePtr<CastDriver> self;
  ReadChunk::Impl base;

  absl::Status operator()(internal::LockCollection& lock_collection) {
    return base(lock_collection);
  }

  Result<NDIterable::Ptr> operator()(ReadChunk::BeginRead,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto iterable,
        base(ReadChunk::BeginRead{}, std::move(chunk_transform), arena));
    return GetConvertedInputNDIterable(std::move(iterable), self->target_dtype_,
                                       self->input_conversion_);
  }
};

// Implementation of `tensorstore::internal::WriteChunk::Impl` Poly
// interface.
struct WriteChunkImpl {
  IntrusivePtr<CastDriver> self;
  WriteChunk::Impl base;

  absl::Status operator()(internal::LockCollection& lock_collection) {
    return base(lock_collection);
  }

  Result<NDIterable::Ptr> operator()(WriteChunk::BeginWrite,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto iterable,
        base(WriteChunk::BeginWrite{}, std::move(chunk_transform), arena));
    return GetConvertedOutputNDIterable(
        std::move(iterable), self->target_dtype_, self->output_conversion_);
  }

  WriteChunk::EndWriteResult operator()(WriteChunk::EndWrite,
                                        IndexTransformView<> chunk_transform,
                                        bool success, Arena* arena) {
    return base(WriteChunk::EndWrite{}, chunk_transform, success, arena);
  }
};

template <typename Chunk, typename ChunkImpl>
struct ChunkReceiverAdapter {
  IntrusivePtr<CastDriver> self;
  AnyFlowReceiver<absl::Status, Chunk, IndexTransform<>> base;
  template <typename CancelReceiver>
  void set_starting(CancelReceiver receiver) {
    tensorstore::execution::set_starting(base, std::move(receiver));
  }

  void set_value(Chunk chunk, IndexTransform<> transform) {
    tensorstore::execution::set_value(
        base,
        Chunk{ChunkImpl{self, std::move(chunk.impl)},
              std::move(chunk.transform)},
        std::move(transform));
  }

  void set_done() { tensorstore::execution::set_done(base); }

  void set_error(absl::Status status) {
    tensorstore::execution::set_error(base, std::move(status));
  }

  void set_stopping() { tensorstore::execution::set_stopping(base); }
};

void CastDriver::Read(
    ReadRequest request,
    AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver) {
  base_driver_->Read(std::move(request),
                     ChunkReceiverAdapter<ReadChunk, ReadChunkImpl>{
                         IntrusivePtr<CastDriver>(this), std::move(receiver)});
}

void CastDriver::Write(
    WriteRequest request,
    AnyFlowReceiver<absl::Status, WriteChunk, IndexTransform<>> receiver) {
  base_driver_->Write(std::move(request),
                      ChunkReceiverAdapter<WriteChunk, WriteChunkImpl>{
                          IntrusivePtr<CastDriver>(this), std::move(receiver)});
}

const internal::DriverRegistration<CastDriverSpec> driver_registration;

}  // namespace

}  // namespace internal_cast_driver

namespace internal {

Result<CastDataTypeConversions> GetCastDataTypeConversions(
    DataType source_dtype, DataType target_dtype, ReadWriteMode existing_mode,
    ReadWriteMode required_mode) {
  // `required_mode` must be a subset of `existing_mode`
  assert((existing_mode & required_mode) == required_mode);
  CastDataTypeConversions result = {};
  if (required_mode == ReadWriteMode::dynamic &&
      existing_mode != ReadWriteMode::read_write) {
    required_mode = existing_mode;
  }
  const ReadWriteMode requested_mode =
      required_mode == ReadWriteMode::dynamic ? existing_mode : required_mode;
  result.mode = requested_mode;
  if ((requested_mode & ReadWriteMode::read) == ReadWriteMode::read) {
    result.input = GetDataTypeConverter(source_dtype, target_dtype);
    if (!(result.input.flags & DataTypeConversionFlags::kSupported)) {
      if ((required_mode & ReadWriteMode::read) == ReadWriteMode::read) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Read access requires unsupported ", source_dtype, " -> ",
            target_dtype, " conversion"));
      }
      result.mode &= ~ReadWriteMode::read;
    }
  }
  if ((requested_mode & ReadWriteMode::write) == ReadWriteMode::write) {
    result.output = GetDataTypeConverter(target_dtype, source_dtype);
    if (!(result.output.flags & DataTypeConversionFlags::kSupported)) {
      if ((required_mode & ReadWriteMode::write) == ReadWriteMode::write) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Write access requires unsupported ", target_dtype, " -> ",
            source_dtype, " conversion"));
      }
      result.mode &= ~ReadWriteMode::write;
    }
  }
  if (result.mode == ReadWriteMode{}) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Cannot convert ", source_dtype, " <-> ", target_dtype));
  }
  return result;
}

Result<Driver::Handle> MakeCastDriver(Driver::Handle base,
                                      DataType target_dtype,
                                      ReadWriteMode read_write_mode) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto conversions, GetCastDataTypeConversions(
                            base.driver->dtype(), target_dtype,
                            base.driver.read_write_mode(), read_write_mode));
  base.driver = internal::MakeReadWritePtr<internal_cast_driver::CastDriver>(
      conversions.mode, std::move(base.driver), target_dtype, conversions.input,
      conversions.output);
  return base;
}

Result<TransformedDriverSpec> MakeCastDriverSpec(TransformedDriverSpec base,
                                                 DataType target_dtype) {
  if (!base.driver_spec) return {std::in_place};
  DataType source_dtype = base.driver_spec->schema.dtype();
  if (source_dtype.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(GetCastDataTypeConversions(
        source_dtype, target_dtype, ReadWriteMode::read_write,
        ReadWriteMode::dynamic));
  }
  auto driver_spec =
      internal::DriverSpec::Make<internal_cast_driver::CastDriverSpec>();
  driver_spec->schema
      .Set(base.transform.valid() ? RankConstraint{base.transform.output_rank()}
                                  : base.driver_spec->schema.rank())
      .IgnoreError();
  driver_spec->schema.Set(target_dtype).IgnoreError();
  driver_spec->context_binding_state_ = base.context_binding_state();
  driver_spec->base.driver_spec = std::move(base.driver_spec);
  base.driver_spec = std::move(driver_spec);
  return base;
}

}  // namespace internal

Result<Spec> Cast(const Spec& base_spec, DataType target_dtype) {
  Spec spec;
  auto& base_impl = internal_spec::SpecAccess::impl(base_spec);
  auto& impl = internal_spec::SpecAccess::impl(spec);
  TENSORSTORE_ASSIGN_OR_RETURN(
      impl, internal::MakeCastDriverSpec(base_impl, target_dtype));
  return spec;
}

}  // namespace tensorstore
