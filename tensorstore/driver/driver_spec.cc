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

#include "tensorstore/driver/driver_spec.h"

#include <assert.h>

#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/index_space/transform_broadcastable_array.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/data_type.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_registry.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/registry.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/util/unit.h"

namespace tensorstore {
namespace internal {

namespace jb = tensorstore::internal_json_binding;

DriverRegistry& GetDriverRegistry() {
  static internal::NoDestructor<DriverRegistry> registry;
  return *registry;
}

DriverSpec::~DriverSpec() = default;

Result<IndexDomain<>> DriverSpec::GetDomain() const { return schema.domain(); }

Result<ChunkLayout> DriverSpec::GetChunkLayout() const {
  return schema.chunk_layout();
}

Result<CodecSpec> DriverSpec::GetCodec() const { return schema.codec(); }

Result<SharedArray<const void>> DriverSpec::GetFillValue(
    IndexTransformView<> transform) const {
  auto fill_value = schema.fill_value();
  if (!fill_value.valid()) return {std::in_place};
  if (!transform.valid()) {
    return SharedArray<const void>(fill_value.shared_array_view());
  }
  return TransformOutputBroadcastableArray(transform, std::move(fill_value),
                                           schema.domain());
}

Result<DimensionUnitsVector> DriverSpec::GetDimensionUnits() const {
  return DimensionUnitsVector(schema.dimension_units());
}

kvstore::Spec DriverSpec::GetKvstore() const { return {}; }

absl::Status ApplyOptions(DriverSpec::Ptr& spec, SpecOptions&& options) {
  if (spec->use_count() != 1) spec = spec->Clone();
  return const_cast<DriverSpec&>(*spec).ApplyOptions(std::move(options));
}

namespace {
absl::Status MaybeDeriveTransform(TransformedDriverSpec& spec) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto domain, spec.driver_spec->GetDomain());
  if (domain.valid()) {
    spec.transform = IdentityTransform(domain);
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status TransformAndApplyOptions(TransformedDriverSpec& spec,
                                      SpecOptions&& options) {
  const bool should_get_transform =
      !spec.transform.valid() && options.domain().valid();
  TENSORSTORE_RETURN_IF_ERROR(
      options.TransformInputSpaceSchema(spec.transform));
  TENSORSTORE_RETURN_IF_ERROR(
      ApplyOptions(spec.driver_spec, std::move(options)));
  if (should_get_transform) {
    TENSORSTORE_RETURN_IF_ERROR(MaybeDeriveTransform(spec));
  }
  return absl::OkStatus();
}

Result<IndexDomain<>> GetEffectiveDomain(const TransformedDriverSpec& spec) {
  if (!spec.driver_spec) return {std::in_place};
  if (!spec.transform.valid()) {
    return spec.driver_spec->GetDomain();
  } else {
    return spec.transform.domain();
  }
}

Result<ChunkLayout> GetEffectiveChunkLayout(const TransformedDriverSpec& spec) {
  if (!spec.driver_spec) return {std::in_place};
  TENSORSTORE_ASSIGN_OR_RETURN(auto chunk_layout,
                               spec.driver_spec->GetChunkLayout());
  if (spec.transform.valid()) {
    TENSORSTORE_ASSIGN_OR_RETURN(chunk_layout,
                                 std::move(chunk_layout) | spec.transform);
  }
  return chunk_layout;
}

Result<SharedArray<const void>> GetEffectiveFillValue(
    const TransformedDriverSpec& spec) {
  if (!spec.driver_spec) return {std::in_place};
  return spec.driver_spec->GetFillValue(spec.transform);
}

Result<CodecSpec> GetEffectiveCodec(const TransformedDriverSpec& spec) {
  if (!spec.driver_spec) return {std::in_place};
  return spec.driver_spec->GetCodec();
}

Result<DimensionUnitsVector> GetEffectiveDimensionUnits(
    const TransformedDriverSpec& spec) {
  if (!spec.driver_spec) return {std::in_place};
  TENSORSTORE_ASSIGN_OR_RETURN(auto dimension_units,
                               spec.driver_spec->GetDimensionUnits());
  if (dimension_units.empty()) {
    if (const DimensionIndex rank = spec.driver_spec->schema.rank();
        rank != dynamic_rank) {
      dimension_units.resize(rank);
    }
  }
  if (spec.transform.valid()) {
    dimension_units = tensorstore::TransformOutputDimensionUnits(
        spec.transform, std::move(dimension_units));
  }
  return dimension_units;
}

Result<Schema> GetEffectiveSchema(const TransformedDriverSpec& spec) {
  if (!spec.driver_spec) return {std::in_place};
  Schema schema;
  TENSORSTORE_RETURN_IF_ERROR(schema.Set(spec.driver_spec->schema.dtype()));
  TENSORSTORE_RETURN_IF_ERROR(schema.Set(spec.driver_spec->schema.rank()));
  {
    TENSORSTORE_ASSIGN_OR_RETURN(auto domain, GetEffectiveDomain(spec));
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(domain));
  }
  {
    TENSORSTORE_ASSIGN_OR_RETURN(auto chunk_layout,
                                 GetEffectiveChunkLayout(spec));
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(std::move(chunk_layout)));
  }
  {
    TENSORSTORE_ASSIGN_OR_RETURN(auto codec, spec.driver_spec->GetCodec());
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(std::move(codec)));
  }
  {
    TENSORSTORE_ASSIGN_OR_RETURN(auto fill_value, GetEffectiveFillValue(spec));
    TENSORSTORE_RETURN_IF_ERROR(
        schema.Set(Schema::FillValue(std::move(fill_value))));
  }
  {
    TENSORSTORE_ASSIGN_OR_RETURN(auto dimension_units,
                                 GetEffectiveDimensionUnits(spec));
    TENSORSTORE_RETURN_IF_ERROR(
        schema.Set(Schema::DimensionUnits(dimension_units)));
  }
  return schema;
}

DimensionIndex GetRank(const TransformedDriverSpec& spec) {
  if (spec.transform.valid()) return spec.transform.input_rank();
  if (spec.driver_spec) return spec.driver_spec->schema.rank();
  return dynamic_rank;
}

namespace {
auto SchemaExcludingRankAndDtypeJsonBinder() {
  return jb::DefaultInitializedValue(
      [](auto is_loading, auto options, auto* obj, auto* j) {
        if constexpr (!is_loading) {
          // Set default dtype and rank to the actual dtype/rank to prevent them
          // from being included in json, since they are already included in the
          // separate `dtype` and `rank` members of the Spec.
          options.Set(obj->dtype());
          options.Set(obj->rank());
        }
        return jb::DefaultBinder<>(is_loading, options, obj, j);
      });
}
}  // namespace

TENSORSTORE_DEFINE_JSON_BINDER(
    TransformedDriverSpecJsonBinder,
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
      auto& registry = internal::GetDriverRegistry();
      return jb::NestedContextJsonBinder(jb::Object(
          jb::Member("driver",
                     jb::Projection(&TransformedDriverSpec::driver_spec,
                                    registry.KeyBinder())),
          jb::Projection(
              [](auto& obj) -> DriverSpec& {
                return const_cast<DriverSpec&>(*obj.driver_spec);
              },
              jb::Sequence(
                  jb::Initialize([](DriverSpec* x) {
                    x->context_binding_state_ = ContextBindingState::unbound;
                  }),
                  jb::Member("context",
                             jb::Projection(
                                 &DriverSpec::context_spec_,
                                 internal::ContextSpecDefaultableJsonBinder)),
                  jb::Member("schema",
                             jb::Projection<&DriverSpec::schema>(
                                 SchemaExcludingRankAndDtypeJsonBinder())),
                  jb::Member(
                      "dtype",
                      jb::GetterSetter([](auto& x) { return x.schema.dtype(); },
                                       [](auto& x, DataType value) {
                                         return x.schema.Set(value);
                                       },
                                       jb::ConstrainedDataTypeJsonBinder)))),
          jb::OptionalMember("transform",
                             jb::Projection(&TransformedDriverSpec::transform)),
          jb::OptionalMember(
              "rank",
              jb::GetterSetter(
                  [](const auto& obj) {
                    return obj.transform.valid()
                               ? static_cast<DimensionIndex>(dynamic_rank)
                               : static_cast<DimensionIndex>(
                                     obj.driver_spec->schema.rank());
                  },
                  [](const auto& obj, DimensionIndex rank) {
                    if (rank != dynamic_rank) {
                      if (obj.transform.valid()) {
                        if (obj.transform.input_rank() != rank) {
                          return absl::InvalidArgumentError(tensorstore::StrCat(
                              "Specified rank (", rank,
                              ") does not match input rank of transform (",
                              obj.transform.input_rank(), ")"));
                        }
                      } else {
                        TENSORSTORE_RETURN_IF_ERROR(
                            const_cast<DriverSpec&>(*obj.driver_spec)
                                .schema.Set(RankConstraint{rank}));
                      }
                    }
                    return absl::OkStatus();
                  },
                  jb::ConstrainedRankJsonBinder)),
          jb::Initialize([](auto* obj) {
            if (!obj->transform.valid()) return absl::OkStatus();
            return const_cast<DriverSpec&>(*obj->driver_spec)
                .schema.Set(RankConstraint{obj->transform.output_rank()});
          }),
          jb::Projection(&TransformedDriverSpec::driver_spec,
                         registry.RegisteredObjectBinder()),
          jb::Initialize([](auto* obj) {
            if (obj->transform.valid()) return absl::OkStatus();
            return MaybeDeriveTransform(*obj);
          })))(is_loading, options, obj, j);
    })

absl::Status DriverSpecBindContext(DriverSpecPtr& spec,
                                   const Context& context) {
  return internal::BindContextCopyOnWriteWithNestedContext(spec, context);
}

void DriverSpecUnbindContext(DriverSpecPtr& spec,
                             const ContextSpecBuilder& context_builder) {
  internal::UnbindContextCopyOnWriteWithNestedContext(spec, context_builder);
}

void DriverSpecStripContext(DriverSpecPtr& spec) {
  internal::StripContextCopyOnWriteWithNestedContext(spec);
}

using DriverSpecPtrNonNullDirectSerializer = serialization::RegistrySerializer<
    internal::IntrusivePtr<const internal::DriverSpec>>;

using DriverSpecPtrSerializer = serialization::IndirectPointerSerializer<
    internal::IntrusivePtr<const internal::DriverSpec>,
    DriverSpecPtrNonNullDirectSerializer>;

using DriverSpecPtrNonNullSerializer =
    serialization::NonNullIndirectPointerSerializer<
        internal::IntrusivePtr<const internal::DriverSpec>,
        DriverSpecPtrNonNullDirectSerializer>;

bool TransformedDriverSpecNonNullSerializer::Encode(
    serialization::EncodeSink& sink, const TransformedDriverSpec& value) {
  assert(value.driver_spec);
  return serialization::Encode(sink, value.driver_spec,
                               DriverSpecPtrNonNullSerializer()) &&
         serialization::Encode(sink, value.transform);
}

bool TransformedDriverSpecNonNullSerializer::Decode(
    serialization::DecodeSource& source, TransformedDriverSpec& value) {
  return serialization::Decode(source, value.driver_spec,
                               DriverSpecPtrNonNullSerializer()) &&
         serialization::Decode(source, value.transform);
}

}  // namespace internal

namespace serialization {

template serialization::Registry&
serialization::GetRegistry<internal::DriverSpecPtr>();

}  // namespace serialization

namespace garbage_collection {
void GarbageCollection<internal::DriverSpecPtr>::Visit(
    GarbageCollectionVisitor& visitor, const internal::DriverSpecPtr& value) {
  if (!value) return;
  visitor.Indirect<PolymorphicGarbageCollection<internal::DriverSpec>>(*value);
}
}  // namespace garbage_collection

}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal::DriverSpecPtr,
    tensorstore::internal::DriverSpecPtrSerializer())

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal::TransformedDriverSpec,
    tensorstore::serialization::ApplyMembersSerializer<
        tensorstore::internal::TransformedDriverSpec>())
