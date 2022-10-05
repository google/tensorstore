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

#include "tensorstore/driver/driver.h"

#include <assert.h>

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/rank.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/execution/sender_util.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/util/unit.h"

namespace tensorstore {
namespace internal {

Future<Driver::Handle> OpenDriver(TransformedDriverSpec spec,
                                  TransactionalOpenOptions&& options) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto open_transaction,
      internal::AcquireOpenTransactionPtrOrError(options.transaction));
  return internal::OpenDriver(std::move(open_transaction), std::move(spec),
                              std::move(options));
}

Future<Driver::Handle> OpenDriver(OpenTransactionPtr transaction,
                                  TransformedDriverSpec spec,
                                  OpenOptions&& options) {
  TENSORSTORE_RETURN_IF_ERROR(internal::TransformAndApplyOptions(
      spec,
      // Moves out just the `SpecOptions` base.  The members of the derived
      // class `OpenOptions`, like `context` and `read_write_mode`, are
      // retained.
      static_cast<SpecOptions&&>(options)));
  TENSORSTORE_RETURN_IF_ERROR(
      DriverSpecBindContext(spec.driver_spec, options.context));
  return internal::OpenDriver(std::move(transaction), std::move(spec),
                              options.read_write_mode);
}

Future<Driver::Handle> OpenDriver(OpenTransactionPtr transaction,
                                  TransformedDriverSpec bound_spec,
                                  ReadWriteMode read_write_mode) {
  DriverSpecPtr ptr = bound_spec.driver_spec;

  return MapFuture(
      InlineExecutor{},
      [bound_spec = std::move(bound_spec)](
          Result<Driver::Handle>& handle) mutable -> Result<Driver::Handle> {
        absl::Status status;
        if (!handle.ok()) {
          status = handle.status();
        } else if (bound_spec.transform.valid()) {
          auto composed_transform = tensorstore::ComposeTransforms(
              std::move(handle->transform), std::move(bound_spec.transform));
          if (composed_transform.ok()) {
            handle->transform = std::move(composed_transform).value();
          } else {
            status = composed_transform.status();
          }
        }

        /// On failure, annotate status with spec.
        if (!status.ok()) {
          status = tensorstore::MaybeAnnotateStatus(
              std::move(status),
              tensorstore::StrCat(
                  "Error opening ",
                  tensorstore::QuoteString(bound_spec.driver_spec->GetId()),
                  " driver"));
          auto spec_json = internal_json_binding::ToJson(bound_spec);
          if (spec_json.ok()) {
            AddStatusPayload(status, "tensorstore_spec",
                             absl::Cord(spec_json.value().dump()));
          }
          return status;
        }

        // Move handle out of the `Future`.
        return std::move(handle);
      },
      ptr->Open(std::move(transaction), read_write_mode));
}

Driver::~Driver() = default;

Result<TransformedDriverSpec> Driver::GetBoundSpec(
    internal::OpenTransactionPtr transaction, IndexTransformView<> transform) {
  return absl::UnimplementedError("JSON representation not supported");
}

Result<ChunkLayout> Driver::GetChunkLayout(IndexTransformView<> transform) {
  return {std::in_place};
}

Result<CodecSpec> Driver::GetCodec() { return CodecSpec{}; }

KvStore Driver::GetKvstore() { return {}; }

Result<SharedArray<const void>> Driver::GetFillValue(
    IndexTransformView<> transform) {
  return {std::in_place};
}

Result<DimensionUnitsVector> Driver::GetDimensionUnits() {
  return {std::in_place, this->rank()};
}

Future<IndexTransform<>> Driver::ResolveBounds(OpenTransactionPtr transaction,
                                               IndexTransform<> transform,
                                               ResolveBoundsOptions options) {
  assert(transform.output_rank() == rank());
  return std::move(transform);
}

void Driver::Read(internal::OpenTransactionPtr transaction,
                  IndexTransform<> transform, ReadChunkReceiver receiver) {
  execution::set_error(FlowSingleReceiver{std::move(receiver)},
                       absl::UnimplementedError("Reading not supported"));
}

void Driver::Write(internal::OpenTransactionPtr transaction,
                   IndexTransform<> transform, WriteChunkReceiver receiver) {
  execution::set_error(FlowSingleReceiver{std::move(receiver)},
                       absl::UnimplementedError("Writing not supported"));
}

Future<IndexTransform<>> Driver::Resize(OpenTransactionPtr transaction,
                                        IndexTransform<> transform,
                                        span<const Index> inclusive_min,
                                        span<const Index> exclusive_max,
                                        ResizeOptions options) {
  return absl::UnimplementedError("Resize not supported");
}

Result<ChunkLayout> GetChunkLayout(const Driver::Handle& handle) {
  assert(handle.driver);
  return handle.driver->GetChunkLayout(handle.transform);
}

Result<CodecSpec> GetCodec(const Driver::Handle& handle) {
  assert(handle.driver);
  return handle.driver->GetCodec();
}

Result<DimensionUnitsVector> GetDimensionUnits(const Driver::Handle& handle) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto units, handle.driver->GetDimensionUnits());
  return tensorstore::TransformOutputDimensionUnits(handle.transform,
                                                    std::move(units));
}

Result<Schema> GetSchema(const Driver::Handle& handle) {
  Schema schema;
  TENSORSTORE_RETURN_IF_ERROR(schema.Set(handle.driver->dtype()));
  TENSORSTORE_RETURN_IF_ERROR(schema.Set(handle.transform.domain()));
  {
    TENSORSTORE_ASSIGN_OR_RETURN(auto chunk_layout, GetChunkLayout(handle));
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(std::move(chunk_layout)));
  }
  {
    TENSORSTORE_ASSIGN_OR_RETURN(auto codec, GetCodec(handle));
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(std::move(codec)));
  }
  {
    TENSORSTORE_ASSIGN_OR_RETURN(auto fill_value, GetFillValue(handle));
    TENSORSTORE_RETURN_IF_ERROR(
        schema.Set(Schema::FillValue(std::move(fill_value))));
  }
  {
    TENSORSTORE_ASSIGN_OR_RETURN(auto dimension_units,
                                 GetDimensionUnits(handle));
    TENSORSTORE_RETURN_IF_ERROR(
        schema.Set(Schema::DimensionUnits(dimension_units)));
  }
  return schema;
}

KvStore GetKvstore(const DriverHandle& handle) {
  if (!handle.valid()) return {};
  auto kvs = handle.driver->GetKvstore();
  if (kvs.valid()) {
    kvs.transaction = handle.transaction;
  }
  return kvs;
}

Result<TransformedDriverSpec> GetTransformedDriverSpec(
    const DriverHandle& handle, SpecRequestOptions&& options) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto open_transaction,
      internal::AcquireOpenTransactionPtrOrError(handle.transaction));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto transformed_driver_spec,
      handle.driver->GetBoundSpec(std::move(open_transaction),
                                  handle.transform));
  // `ApplyContextBindingMode` and `TransformAndApplyOptions` are both
  // copy-on-write operations that will reset
  // `transformed_driver_spec.driver_spec` to a new copy if necessary, but in
  // this case, as there should only be a single reference, no copy will
  // actually be required.
  internal::ApplyContextBindingMode(transformed_driver_spec,
                                    options.context_binding_mode,
                                    /*default_mode=*/ContextBindingMode::strip);
  TENSORSTORE_RETURN_IF_ERROR(internal::TransformAndApplyOptions(
      transformed_driver_spec, std::move(options)));
  return transformed_driver_spec;
}

bool DriverHandleNonNullSerializer::Encode(serialization::EncodeSink& sink,
                                           const DriverHandle& value) {
  assert(value.driver);
  if (value.transaction != no_transaction) {
    sink.Fail(absl::InvalidArgumentError(
        "Cannot serialize TensorStore with bound transaction"));
    return false;
  }
  TransformedDriverSpec spec;
  TENSORSTORE_ASSIGN_OR_RETURN(
      spec, value.driver->GetBoundSpec(/*transaction=*/{}, value.transform),
      (sink.Fail(_), false));
  return serialization::Encode(sink, spec,
                               TransformedDriverSpecNonNullSerializer{}) &&
         serialization::Encode(
             sink, static_cast<int>(value.driver.read_write_mode()));
}

bool DriverHandleNonNullSerializer::Decode(serialization::DecodeSource& source,
                                           DriverHandle& value) {
  TransformedDriverSpec spec;
  int read_write_mode;
  if (!serialization::Decode(source, spec,
                             TransformedDriverSpecNonNullSerializer{}) ||
      !serialization::Decode(source, read_write_mode)) {
    return false;
  }
  read_write_mode =
      read_write_mode & static_cast<int>(ReadWriteMode::read_write);
  if (static_cast<ReadWriteMode>(read_write_mode) == ReadWriteMode::dynamic) {
    source.Fail(serialization::DecodeError("Expected valid ReadWriteMode"));
    return false;
  }
  TransactionalOpenOptions options;
  options.read_write_mode = static_cast<ReadWriteMode>(read_write_mode);
  TENSORSTORE_ASSIGN_OR_RETURN(
      value, internal::OpenDriver(std::move(spec), std::move(options)).result(),
      (source.Fail(_), false));
  return true;
}

bool DecodeDriverHandle(serialization::DecodeSource& source,
                        DriverHandle& value, DataType data_type_constraint,
                        DimensionIndex rank_constraint,
                        ReadWriteMode mode_constraint) {
  bool valid;
  if (!serialization::Decode(source, valid)) return false;
  if (!valid) return true;
  return DecodeNonNullDriverHandle(source, value, data_type_constraint,
                                   rank_constraint, mode_constraint);
}

bool DecodeNonNullDriverHandle(serialization::DecodeSource& source,
                               DriverHandle& value,
                               DataType data_type_constraint,
                               DimensionIndex rank_constraint,
                               ReadWriteMode mode_constraint) {
  DriverHandle temp;
  if (!serialization::Decode(source, temp, DriverHandleNonNullSerializer{})) {
    return false;
  }
  if ((data_type_constraint.valid() &&
       data_type_constraint != temp.driver->dtype()) ||
      (rank_constraint != dynamic_rank &&
       rank_constraint != temp.transform.input_rank()) ||
      ((temp.driver.read_write_mode() & mode_constraint) != mode_constraint)) {
    source.Fail(serialization::DecodeError(
        "data type, rank, or read-write mode mismatch"));
    return false;
  }
  value = std::move(temp);
  return true;
}

}  // namespace internal

namespace garbage_collection {
void GarbageCollection<internal::DriverHandle>::Visit(
    GarbageCollectionVisitor& visitor, const internal::DriverHandle& value) {
  garbage_collection::GarbageCollectionVisit(visitor, value.driver);
}
}  // namespace garbage_collection

}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal::DriverHandle,
    (tensorstore::serialization::MaybeNullSerializer<
        tensorstore::internal::DriverHandle,
        tensorstore::internal::DriverHandleNonNullSerializer,
        tensorstore::serialization::IsValid>()))

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal::Driver,
    tensorstore::garbage_collection::PolymorphicGarbageCollection<
        tensorstore::internal::Driver>)

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal::DriverPtr,
    tensorstore::garbage_collection::IndirectPointerGarbageCollection<
        tensorstore::internal::DriverPtr>)
