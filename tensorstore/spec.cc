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

#include "tensorstore/spec.h"

#include "tensorstore/context.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"

namespace tensorstore {

absl::Status Spec::Set(SpecConvertOptions&& options) {
  internal::ApplyContextBindingMode(
      *this, options.context_binding_mode,
      /*default_mode=*/ContextBindingMode::retain);
  TENSORSTORE_RETURN_IF_ERROR(
      internal::TransformAndApplyOptions(impl_, std::move(options)));
  if (options.context) {
    TENSORSTORE_RETURN_IF_ERROR(this->BindContext(options.context));
  }
  return absl::OkStatus();
}

Result<Schema> Spec::schema() const {
  return internal::GetEffectiveSchema(impl_);
}

Result<IndexDomain<>> Spec::domain() const {
  return internal::GetEffectiveDomain(impl_);
}

Result<ChunkLayout> Spec::chunk_layout() const {
  return internal::GetEffectiveChunkLayout(impl_);
}

Result<CodecSpec> Spec::codec() const {
  return internal::GetEffectiveCodec(impl_);
}

Result<SharedArray<const void>> Spec::fill_value() const {
  return internal::GetEffectiveFillValue(impl_);
}

Result<DimensionUnitsVector> Spec::dimension_units() const {
  return internal::GetEffectiveDimensionUnits(impl_);
}

kvstore::Spec Spec::kvstore() const {
  if (!impl_.driver_spec) return {};
  return impl_.driver_spec->GetKvstore();
}

ContextBindingState Spec::context_binding_state() const {
  return impl_.context_binding_state();
}

std::ostream& operator<<(std::ostream& os, const Spec& spec) {
  Spec copy = spec;
  copy.UnbindContext();
  JsonSerializationOptions options;
  options.preserve_bound_context_resources_ = true;
  auto json_result = copy.ToJson(options);
  if (!json_result.ok()) {
    os << "<unprintable spec: " << json_result.status() << ">";
  } else {
    os << json_result->dump();
  }
  return os;
}

bool operator==(const Spec& a, const Spec& b) {
  if (!a.valid() || !b.valid()) {
    return a.valid() == b.valid();
  }
  Spec a_unbound, b_unbound;
  {
    auto spec_builder = internal::ContextSpecBuilder::Make();
    // Track binding state, so that we don't compare equal if the binding state
    // is not the same.
    internal::SetRecordBindingState(spec_builder, true);
    a_unbound = a;
    a_unbound.UnbindContext(spec_builder);
    b_unbound = b;
    b_unbound.UnbindContext(spec_builder);
  }
  JsonSerializationOptions json_serialization_options;
  json_serialization_options.preserve_bound_context_resources_ = true;
  auto a_json = a_unbound.ToJson(json_serialization_options);
  auto b_json = b_unbound.ToJson(json_serialization_options);
  if (!a_json.ok() || !b_json.ok()) return false;
  return internal_json::JsonSame(*a_json, *b_json);
}

namespace jb = tensorstore::internal_json_binding;
TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    Spec,
    jb::Projection(&Spec::impl_, internal::TransformedDriverSpecJsonBinder))

Result<IndexTransform<>> Spec::GetTransformForIndexingOperation() const {
  if (impl_.transform.valid()) return impl_.transform;
  if (impl_.driver_spec) {
    const DimensionIndex rank = impl_.driver_spec->schema.rank();
    if (rank != dynamic_rank) {
      return IdentityTransform(rank);
    }
  }
  return absl::InvalidArgumentError(
      "Cannot perform indexing operations on Spec with unspecified rank");
}

Result<Spec> ApplyIndexTransform(IndexTransform<> transform, Spec spec) {
  if (!transform.valid()) return spec;
  if (spec.impl_.transform.valid()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        spec.impl_.transform, ComposeTransforms(std::move(spec.impl_.transform),
                                                std::move(transform)));
  } else {
    TENSORSTORE_RETURN_IF_ERROR(
        spec.Set(RankConstraint{transform.output_rank()}));
    spec.impl_.transform = std::move(transform);
  }
  return spec;
}

absl::Status Spec::BindContext(const Context& context) {
  return internal::DriverSpecBindContext(impl_.driver_spec, context);
}

void Spec::UnbindContext(const internal::ContextSpecBuilder& context_builder) {
  return internal::DriverSpecUnbindContext(impl_.driver_spec, context_builder);
}

void Spec::StripContext() {
  return internal::DriverSpecStripContext(impl_.driver_spec);
}

namespace internal {
Result<Spec> GetSpec(const DriverHandle& handle, SpecRequestOptions&& options) {
  Spec spec;
  TENSORSTORE_ASSIGN_OR_RETURN(
      internal_spec::SpecAccess::impl(spec),
      internal::GetTransformedDriverSpec(handle, std::move(options)));
  return spec;
}

bool SpecNonNullSerializer::Encode(serialization::EncodeSink& sink,
                                   const Spec& value) {
  return serialization::Encode(
      sink, internal_spec::SpecAccess::impl(value),
      internal::TransformedDriverSpecNonNullSerializer{});
}

bool SpecNonNullSerializer::Decode(serialization::DecodeSource& source,
                                   Spec& value) {
  return serialization::Decode(
      source, internal_spec::SpecAccess::impl(value),
      internal::TransformedDriverSpecNonNullSerializer{});
}

}  // namespace internal

namespace serialization {

bool Serializer<Spec>::Encode(EncodeSink& sink, const Spec& value) {
  return serialization::Encode(sink, internal_spec::SpecAccess::impl(value));
}

bool Serializer<Spec>::Decode(DecodeSource& source, Spec& value) {
  return serialization::Decode(source, internal_spec::SpecAccess::impl(value));
}

}  // namespace serialization

namespace garbage_collection {

void GarbageCollection<Spec>::Visit(GarbageCollectionVisitor& visitor,
                                    const Spec& value) {
  garbage_collection::GarbageCollectionVisit(
      visitor, internal_spec::SpecAccess::impl(value));
}
}  // namespace garbage_collection

}  // namespace tensorstore
