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

#include "absl/status/status.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/nditerable_data_type_conversion.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/spec.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

namespace internal {
namespace {

namespace jb = tensorstore::internal_json_binding;

class CastDriver
    : public RegisteredDriver<CastDriver, /*Parent=*/internal::Driver> {
 public:
  constexpr static char id[] = "cast";

  template <template <typename> class MaybeBound>
  struct SpecT : public DriverSpecCommonData {
    MaybeBound<TransformedDriverSpec<>> base;

    constexpr static auto ApplyMembers = [](auto& x, auto f) {
      return f(internal::BaseCast<internal::DriverSpecCommonData>(x), x.base);
    };
  };

  using SpecData = SpecT<internal::ContextUnbound>;
  using BoundSpecData = SpecT<internal::ContextBound>;

  using Ptr = Driver::PtrT<CastDriver>;

  static Status ApplyOptions(SpecData& spec, SpecOptions&& options) {
    TENSORSTORE_RETURN_IF_ERROR(spec.schema.Set(options.dtype()));
    options.Override(DataType()).IgnoreError();
    return internal::TransformAndApplyOptions(spec.base, std::move(options));
  }

  constexpr static auto json_binder = jb::Object(
      jb::Member("base",
                 [](auto is_loading, const auto& options, auto* obj, auto* j) {
                   return jb::Projection(&SpecData::base)(
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
        return ApplyOptions(*obj, std::move(base_options));
      }));

  static Result<IndexDomain<>> SpecGetDomain(const SpecData& spec) {
    return internal::GetEffectiveDomain(spec.base);
  }

  static Result<ChunkLayout> SpecGetChunkLayout(const SpecData& spec) {
    return internal::GetEffectiveChunkLayout(spec.base);
  }

  static Result<CodecSpec::Ptr> SpecGetCodec(const SpecData& spec) {
    return internal::GetEffectiveCodec(spec.base);
  }

  static Result<SharedArray<const void>> SpecGetFillValue(
      const SpecData& spec, IndexTransformView<> transform) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto adjusted_transform,
        tensorstore::ComposeOptionalTransforms(spec.base.transform, transform));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto fill_value,
        spec.base.driver_spec->GetFillValue(adjusted_transform));
    if (!fill_value.valid()) return {std::in_place};
    auto dtype = spec.schema.dtype();
    if (dtype == fill_value.dtype()) return fill_value;
    // Check if we can convert.
    auto converter = GetDataTypeConverter(fill_value.dtype(), dtype);
    if (!(converter.flags & DataTypeConversionFlags::kSupported)) {
      return {std::in_place};
    }
    return MakeCopy(fill_value, skip_repeated_elements, dtype);
  }

  static Future<internal::Driver::Handle> Open(
      internal::OpenTransactionPtr transaction,
      internal::RegisteredDriverOpener<BoundSpecData> spec,
      ReadWriteMode read_write_mode) {
    DataType target_dtype = spec->schema.dtype();
    if (!target_dtype.valid()) {
      return absl::InvalidArgumentError("dtype must be specified");
    }
    return MapFutureValue(
        InlineExecutor{},
        [target_dtype,
         read_write_mode](Driver::Handle handle) -> Result<Driver::Handle> {
          return MakeCastDriver(std::move(handle), target_dtype,
                                read_write_mode);
        },
        internal::OpenDriver(std::move(transaction), spec->base,
                             read_write_mode));
  }

  Result<IndexTransform<>> GetBoundSpecData(
      internal::OpenTransactionPtr transaction, BoundSpecData* spec,
      IndexTransformView<> transform) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        spec->base,
        base_driver_->GetBoundSpec(std::move(transaction), transform));
    spec->schema.Set(target_dtype_).IgnoreError();
    const DimensionIndex base_rank = base_driver_->rank();
    spec->schema.Set(RankConstraint{base_rank}).IgnoreError();
    return std::exchange(spec->base.transform, {});
  }

  Result<ChunkLayout> GetChunkLayout(IndexTransformView<> transform) override {
    return base_driver_->GetChunkLayout(transform);
  }

  Result<CodecSpec::Ptr> GetCodec() override {
    return base_driver_->GetCodec();
  }

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

  explicit CastDriver(Driver::Ptr base, DataType target_dtype,
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

  void Read(
      OpenTransactionPtr transaction, IndexTransform<> transform,
      AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver) override;

  void Write(
      OpenTransactionPtr transaction, IndexTransform<> transform,
      AnyFlowReceiver<Status, WriteChunk, IndexTransform<>> receiver) override;

  Future<IndexTransform<>> ResolveBounds(
      OpenTransactionPtr transaction, IndexTransform<> transform,
      ResolveBoundsOptions options) override {
    return base_driver_->ResolveBounds(
        std::move(transaction), std::move(transform), std::move(options));
  }

  Future<IndexTransform<>> Resize(OpenTransactionPtr transaction,
                                  IndexTransform<> transform,
                                  span<const Index> inclusive_min,
                                  span<const Index> exclusive_max,
                                  ResizeOptions options) override {
    return base_driver_->Resize(std::move(transaction), std::move(transform),
                                inclusive_min, exclusive_max,
                                std::move(options));
  }

  Driver::Ptr base_driver_;
  DataType target_dtype_;
  DataTypeConversionLookupResult input_conversion_;
  DataTypeConversionLookupResult output_conversion_;
};

// Implementation of `tensorstore::internal::ReadChunk::Impl` Poly interface.
struct ReadChunkImpl {
  CastDriver::Ptr self;
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

// Implementation of `tensorstore::internal::WriteChunk::Impl` Poly interface.
struct WriteChunkImpl {
  CastDriver::Ptr self;
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
                                        NDIterable::IterationLayoutView layout,
                                        span<const Index> write_end_position,
                                        Arena* arena) {
    return base(WriteChunk::EndWrite{}, chunk_transform, layout,
                write_end_position, arena);
  }
};

template <typename Chunk, typename ChunkImpl>
struct ChunkReceiverAdapter {
  CastDriver::Ptr self;
  AnyFlowReceiver<Status, Chunk, IndexTransform<>> base;
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

  void set_error(Status status) {
    tensorstore::execution::set_error(base, std::move(status));
  }

  void set_stopping() { tensorstore::execution::set_stopping(base); }
};

void CastDriver::Read(
    OpenTransactionPtr transaction, IndexTransform<> transform,
    AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver) {
  base_driver_->Read(std::move(transaction), std::move(transform),
                     ChunkReceiverAdapter<ReadChunk, ReadChunkImpl>{
                         Ptr(this), std::move(receiver)});
}

void CastDriver::Write(
    OpenTransactionPtr transaction, IndexTransform<> transform,
    AnyFlowReceiver<Status, WriteChunk, IndexTransform<>> receiver) {
  base_driver_->Write(std::move(transaction), std::move(transform),
                      ChunkReceiverAdapter<WriteChunk, WriteChunkImpl>{
                          Ptr(this), std::move(receiver)});
}

const internal::DriverRegistration<CastDriver> driver_registration;

}  // namespace

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
        return absl::InvalidArgumentError(
            StrCat("Read access requires unsupported ", source_dtype, " -> ",
                   target_dtype, " conversion"));
      }
      result.mode &= ~ReadWriteMode::read;
    }
  }
  if ((requested_mode & ReadWriteMode::write) == ReadWriteMode::write) {
    result.output = GetDataTypeConverter(target_dtype, source_dtype);
    if (!(result.output.flags & DataTypeConversionFlags::kSupported)) {
      if ((required_mode & ReadWriteMode::write) == ReadWriteMode::write) {
        return absl::InvalidArgumentError(
            StrCat("Write access requires unsupported ", target_dtype, " -> ",
                   source_dtype, " conversion"));
      }
      result.mode &= ~ReadWriteMode::write;
    }
  }
  if (result.mode == ReadWriteMode{}) {
    return absl::InvalidArgumentError(
        StrCat("Cannot convert ", source_dtype, " <-> ", target_dtype));
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
  base.driver =
      Driver::Ptr(new CastDriver(std::move(base.driver), target_dtype,
                                 conversions.input, conversions.output),
                  conversions.mode);
  return base;
}

}  // namespace internal
}  // namespace tensorstore
