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

namespace jb = tensorstore::internal::json_binding;

class CastDriver
    : public RegisteredDriver<CastDriver, /*Parent=*/internal::Driver> {
 public:
  constexpr static char id[] = "cast";

  template <template <typename> class MaybeBound>
  struct SpecT : public DriverConstraints {
    MaybeBound<TransformedDriverSpec<>> base;

    constexpr static auto ApplyMembers = [](auto& x, auto f) {
      return f(internal::BaseCast<internal::DriverConstraints>(x), x.base);
    };
  };

  using SpecData = SpecT<internal::ContextUnbound>;
  using BoundSpecData = SpecT<internal::ContextBound>;

  constexpr static auto json_binder = jb::Object(
      jb::Initialize([](auto* obj) -> Status {
        if (!obj->data_type.valid()) {
          return Status(absl::StatusCode::kInvalidArgument,
                        "Data type must be specified");
        }
        return absl::OkStatus();
      }),
      jb::Member(
          "base", [](auto is_loading, const auto& options, auto* obj, auto* j) {
            constexpr auto binder = jb::Projection(&SpecData::base);
            if constexpr (is_loading) {
              return binder(
                  is_loading,
                  DriverSpecFromJsonOptions{options, {DataType(), obj->rank}},
                  obj, j);
            } else {
              return binder(
                  is_loading,
                  DriverSpecToJsonOptions{options, {DataType(), obj->rank}},
                  obj, j);
            }
          }));

  using Ptr = Driver::PtrT<CastDriver>;

  static Status ConvertSpec(SpecData* spec, const SpecRequestOptions& options) {
    TENSORSTORE_ASSIGN_OR_RETURN(spec->base.driver_spec,
                                 spec->base.driver_spec->Convert(options));
    return absl::OkStatus();
  }

  static Future<internal::Driver::ReadWriteHandle> Open(
      internal::OpenTransactionPtr transaction,
      internal::RegisteredDriverOpener<BoundSpecData> spec,
      ReadWriteMode read_write_mode) {
    return MapFutureValue(
        InlineExecutor{},
        [target_data_type = spec->data_type, read_write_mode](
            Driver::ReadWriteHandle handle) -> Result<Driver::ReadWriteHandle> {
          return MakeCastDriver(std::move(handle), target_data_type,
                                read_write_mode);
        },
        internal::OpenDriver(std::move(transaction), spec->base,
                             read_write_mode));
  }

  Result<IndexTransformSpec> GetBoundSpecData(
      internal::OpenTransactionPtr transaction, BoundSpecData* spec,
      IndexTransformView<> transform) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        spec->base,
        base_driver_->GetBoundSpec(std::move(transaction), transform));
    spec->rank = base_driver_->rank();
    spec->data_type = target_data_type_;
    auto transform_spec = std::move(spec->base.transform_spec);
    spec->base.transform_spec = IndexTransformSpec(spec->rank);
    return transform_spec;
  }

  explicit CastDriver(Driver::Ptr base, DataType target_data_type,
                      DataTypeConversionLookupResult input_conversion,
                      DataTypeConversionLookupResult output_conversion)
      : base_driver_(std::move(base)),
        target_data_type_(target_data_type),
        input_conversion_(input_conversion),
        output_conversion_(output_conversion) {}

  DataType data_type() override { return target_data_type_; }
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
  DataType target_data_type_;
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
    return GetConvertedInputNDIterable(
        std::move(iterable), self->target_data_type_, self->input_conversion_);
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
        std::move(iterable), self->target_data_type_, self->output_conversion_);
  }

  Future<const void> operator()(WriteChunk::EndWrite,
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
    DataType source_data_type, DataType target_data_type,
    ReadWriteMode existing_mode, ReadWriteMode required_mode) {
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
    result.input = GetDataTypeConverter(source_data_type, target_data_type);
    if (!(result.input.flags & DataTypeConversionFlags::kSupported)) {
      if ((required_mode & ReadWriteMode::read) == ReadWriteMode::read) {
        return absl::InvalidArgumentError(
            StrCat("Read access requires unsupported ", source_data_type,
                   " -> ", target_data_type, " conversion"));
      }
      result.mode &= ~ReadWriteMode::read;
    }
  }
  if ((requested_mode & ReadWriteMode::write) == ReadWriteMode::write) {
    result.output = GetDataTypeConverter(target_data_type, source_data_type);
    if (!(result.output.flags & DataTypeConversionFlags::kSupported)) {
      if ((required_mode & ReadWriteMode::write) == ReadWriteMode::write) {
        return absl::InvalidArgumentError(
            StrCat("Write access requires unsupported ", target_data_type,
                   " -> ", source_data_type, " conversion"));
      }
      result.mode &= ~ReadWriteMode::write;
    }
  }
  if (result.mode == ReadWriteMode{}) {
    return absl::InvalidArgumentError(
        StrCat("Cannot convert ", source_data_type, " <-> ", target_data_type));
  }
  return result;
}

Result<Driver::ReadWriteHandle> MakeCastDriver(Driver::ReadWriteHandle base,
                                               DataType target_data_type,
                                               ReadWriteMode read_write_mode) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto conversions,
      GetCastDataTypeConversions(base.driver->data_type(), target_data_type,
                                 base.read_write_mode, read_write_mode));
  base.driver =
      Driver::Ptr(new CastDriver(std::move(base.driver), target_data_type,
                                 conversions.input, conversions.output));
  base.read_write_mode = conversions.mode;
  return base;
}

}  // namespace internal
}  // namespace tensorstore
