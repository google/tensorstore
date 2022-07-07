// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/schema.h"

#include <ostream>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/index_space/transform_broadcastable_array.h"
#include "tensorstore/internal/json/array.h"
#include "tensorstore/internal/json_binding/array.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/data_type.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/json_binding/unit.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/unit.h"

namespace tensorstore {

namespace {
namespace jb = tensorstore::internal_json_binding;

struct SchemaConstraintsData {
  IndexDomain<> domain_;
  ChunkLayout chunk_layout_;
  CodecSpec codec_;
  SharedArray<const void> fill_value_;
  DimensionUnitsVector dimension_units_;
};
}  // namespace

struct Schema::Impl : public SchemaConstraintsData {
  Impl() = default;
  Impl(const SchemaConstraintsData& data) : SchemaConstraintsData(data) {}
  std::atomic<size_t> ref_count_{0};
};

void intrusive_ptr_increment(Schema::Impl* p) {
  p->ref_count_.fetch_add(1, std::memory_order_acq_rel);
}

void intrusive_ptr_decrement(Schema::Impl* p) {
  if (p->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    delete p;
  }
}

namespace {

absl::Status ValidateRank(Schema& schema, const char* field_name,
                          DimensionIndex rank) {
  TENSORSTORE_RETURN_IF_ERROR(tensorstore::ValidateRank(rank));
  if (schema.rank_ != dynamic_rank && schema.rank_ != rank) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Rank specified by ", field_name, " (", rank,
        ") does not match existing rank specified by schema (", schema.rank_,
        ")"));
  }
  if (schema.impl_ && schema.impl_->fill_value_.valid() &&
      schema.impl_->fill_value_.rank() > rank) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Rank specified by ", field_name, " (", rank,
        ") is incompatible with existing fill_value of shape ",
        schema.impl_->fill_value_.shape()));
  }
  schema.rank_ = rank;
  return absl::OkStatus();
}

/// Merges constraints in `source` into `dest`.
///
/// \param source Source schema.
/// \param dest Destination schema.
absl::Status MergeSchemaInto(const Schema& source, Schema& dest) {
  // First merge constraints that are not part of `Impl`.
  TENSORSTORE_RETURN_IF_ERROR(dest.Set(source.rank()));
  TENSORSTORE_RETURN_IF_ERROR(dest.Set(source.dtype()));

  // If at most one of `source.impl_` and `dest.impl_` is allocated, we can
  // perform the merge trivially.
  if (!source.impl_) return absl::OkStatus();
  if (!dest.impl_) {
    dest.impl_ = source.impl_;
    return absl::OkStatus();
  }
  TENSORSTORE_RETURN_IF_ERROR(dest.Set(source.domain()));
  TENSORSTORE_RETURN_IF_ERROR(dest.Set(source.chunk_layout()));
  TENSORSTORE_RETURN_IF_ERROR(dest.Set(source.fill_value()));
  TENSORSTORE_RETURN_IF_ERROR(dest.Set(source.codec()));
  TENSORSTORE_RETURN_IF_ERROR(dest.Set(source.dimension_units()));
  return absl::OkStatus();
}

template <typename Wrapper, typename TempValue, auto MemberPointer,
          typename Binder>
auto ScalarMemberJsonBinder(Binder binder) {
  return [binder](auto is_loading, const JsonSerializationOptions& options,
                  auto* obj, ::nlohmann::json* j) {
    if constexpr (is_loading) {
      TempValue value;
      TENSORSTORE_RETURN_IF_ERROR(binder(is_loading, options, &value, j));
      return obj->Set(Wrapper{std::move(value)});
    } else {
      auto& value = (*obj->impl_).*MemberPointer;
      return binder(is_loading, options, &value, j);
    }
  };
}

struct LayoutJsonBinder {
  template <bool IsLoading>
  absl::Status operator()(
      std::integral_constant<bool, IsLoading> is_loading,
      std::conditional_t<IsLoading, const JsonSerializationOptions&,
                         JsonSerializationOptions>
          options,
      std::conditional_t<IsLoading, Schema, const Schema>* obj,
      ::nlohmann::json* j) const {
    ChunkLayout* chunk_layout_obj;
    if constexpr (is_loading) {
      if (j->is_discarded()) return absl::OkStatus();
      chunk_layout_obj = &obj->EnsureUniqueImpl().chunk_layout_;
    } else {
      chunk_layout_obj = &obj->impl_->chunk_layout_;
      options.Set(obj->rank());
    }
    TENSORSTORE_RETURN_IF_ERROR(
        jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>()(
            is_loading, options, chunk_layout_obj, j));
    if constexpr (is_loading) {
      TENSORSTORE_RETURN_IF_ERROR(obj->ValidateLayoutInternal());
    }
    return absl::OkStatus();
  }
};

auto JsonBinder() {
  return jb::Object(
      jb::Member("rank",
                 jb::Projection(&Schema::rank_, jb::ConstrainedRankJsonBinder)),
      jb::Member("dtype", jb::Projection(&Schema::dtype_,
                                         jb::ConstrainedDataTypeJsonBinder)),
      [](auto is_loading, const auto& options, auto* obj, auto* j) {
        if (!is_loading && !obj->impl_) return absl::OkStatus();
        return jb::Sequence(
            jb::Member(
                "domain",
                ScalarMemberJsonBinder<IndexDomain<>, IndexDomain<>,
                                       &Schema::Impl::domain_>(
                    jb::DefaultInitializedPredicate<jb::kNeverIncludeDefaults>(
                        [](auto* obj) { return !obj->valid(); }))),
            jb::Member("chunk_layout", LayoutJsonBinder{}),
            jb::Member(
                "codec",
                ScalarMemberJsonBinder<CodecSpec, CodecSpec,
                                       &Schema::Impl::codec_>(
                    jb::DefaultInitializedPredicate<jb::kNeverIncludeDefaults>(
                        [](auto* obj) { return !obj->valid(); }))),
            [](auto is_loading, const auto& options, auto* obj, auto* j) {
              DataType dtype = dtype_v<::nlohmann::json>;
              if constexpr (is_loading) {
                if (auto d = obj->dtype(); d.valid()) {
                  dtype = d;
                }
              }
              return jb::Member(
                  "fill_value",
                  ScalarMemberJsonBinder<Schema::FillValue,
                                         SharedArray<const void>,
                                         &Schema::Impl::fill_value_>(
                      jb::DefaultInitializedPredicate<
                          jb::kNeverIncludeDefaults>(
                          [](auto* obj) { return !obj->valid(); },
                          jb::NestedVoidArray(dtype))))(is_loading, options,
                                                        obj, j);
            },
            jb::Member(
                "dimension_units",
                ScalarMemberJsonBinder<Schema::DimensionUnits,
                                       DimensionUnitsVector,
                                       &Schema::Impl::dimension_units_>(
                    jb::DefaultInitializedPredicate<jb::kNeverIncludeDefaults>(
                        [](auto* obj) {
                          return obj->empty() ||
                                 std::none_of(obj->begin(), obj->end(),
                                              [](const auto& u) {
                                                return u.has_value();
                                              });
                        },
                        jb::Array(jb::Optional(jb::DefaultBinder<>,
                                               [] { return nullptr; }))))))  //
            (is_loading, options, obj, j);
      });
}

Result<IndexDomain<>> TransformInputDomain(IndexDomainView<> input_domain,
                                           IndexTransformView<> transform) {
  const DimensionIndex output_rank = transform.output_rank();
  assert(input_domain.rank() == transform.input_rank());
  DimensionSet seen_input_dims;
  IndexDomainBuilder output_domain_builder(output_rank);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const auto map = transform.output_index_maps()[output_dim];
    if (map.method() != OutputIndexMethod::single_input_dimension) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Cannot specify domain through a transform with a ", map.method(),
          " output index map"));
    }
    const DimensionIndex input_dim = map.input_dimension();
    if (seen_input_dims[input_dim]) {
      return absl::InvalidArgumentError(
          "Cannot specify domain through a transform with multiple output "
          "dimensions mapping to the same input dimension");
    }
    auto output_interval =
        input_domain[input_dim].optionally_implicit_interval();
    if (output_interval.interval() != IndexInterval::Infinite() &&
        std::abs(map.stride()) != 1) {
      return absl::InvalidArgumentError(
          "Cannot specify finite domain through a transform with "
          "non-unit-stride output index map");
    }
    seen_input_dims[input_dim] = true;
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_interval,
        GetAffineTransformRange(output_interval, map.offset(), map.stride()),
        tensorstore::MaybeAnnotateStatus(
            _, tensorstore::StrCat("Error computing range of output dimension ",
                                   output_dim, " from input dimension ",
                                   input_dim)));
    output_domain_builder.labels()[output_dim] =
        input_domain[input_dim].label();
    output_domain_builder.origin()[output_dim] = new_interval.inclusive_min();
    output_domain_builder.shape()[output_dim] = new_interval.size();
    output_domain_builder.implicit_lower_bounds()[output_dim] =
        new_interval.implicit_lower();
    output_domain_builder.implicit_upper_bounds()[output_dim] =
        new_interval.implicit_upper();
  }
  return output_domain_builder.Finalize();
}

template <typename T, typename U>
absl::Status MismatchError(const char* field_name, const T& existing_value,
                           const U& new_value) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Specified ", field_name, " (", new_value,
      ") does not match existing value (", existing_value, ")"));
}

}  // namespace

Schema::Impl& Schema::EnsureUniqueImpl() {
  if (!impl_) {
    impl_.reset(new Impl);
  } else if (impl_->ref_count_.load(std::memory_order_acquire) != 1) {
    impl_.reset(new Impl(static_cast<const SchemaConstraintsData&>(*impl_)));
  }
  return *impl_;
}

absl::Status Schema::Set(CodecSpec value) {
  if (!value.valid()) return absl::OkStatus();
  auto& impl = EnsureUniqueImpl();
  return impl.codec_.MergeFrom(std::move(value));
}

absl::Status Schema::Set(RankConstraint rank) {
  if (rank != dynamic_rank) {
    return ValidateRank(*this, "rank", rank);
  }
  return absl::OkStatus();
}

absl::Status Schema::Set(DataType value) {
  if (value.valid()) {
    if (dtype_.valid() && dtype_ != value) {
      return MismatchError("dtype", dtype_, value);
    }
    dtype_ = value;
  }
  return absl::OkStatus();
}

absl::Status Schema::Override(DataType value) {
  dtype_ = value;
  return absl::OkStatus();
}

namespace {
absl::Status ValidateFillValueForDomain(Schema::Impl& impl,
                                        IndexDomainView<> domain) {
  if (impl.fill_value_.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateShapeBroadcast(impl.fill_value_.shape(), domain.shape()),
        tensorstore::MaybeAnnotateStatus(
            _, "domain is incompatible with fill_value"));
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status Schema::Set(IndexDomain<> value) {
  if (value.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(ValidateRank(*this, "domain", value.rank()));
    auto& impl = EnsureUniqueImpl();
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_domain,
        tensorstore::MergeIndexDomains(impl.domain_, std::move(value)));
    TENSORSTORE_RETURN_IF_ERROR(ValidateFillValueForDomain(impl, new_domain));
    impl.domain_ = std::move(new_domain);
  }
  return absl::OkStatus();
}

absl::Status Schema::Override(IndexDomain<> value) {
  auto& impl = EnsureUniqueImpl();
  if (value.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(ValidateRank(*this, "domain", value.rank()));
    TENSORSTORE_RETURN_IF_ERROR(ValidateFillValueForDomain(impl, value));
  }
  impl.domain_ = std::move(value);
  return absl::OkStatus();
}

absl::Status Schema::Set(Shape value) {
  TENSORSTORE_RETURN_IF_ERROR(ValidateRank(*this, "shape", value.size()));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto domain, IndexDomainBuilder(value.size()).shape(value).Finalize());
  return Set(std::move(domain));
}

absl::Status Schema::Set(FillValue value) {
  if (!value.valid()) {
    return absl::OkStatus();
  }
  if (impl_ && impl_->domain_.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateShapeBroadcast(value.shape(), impl_->domain_.shape()),
        tensorstore::MaybeAnnotateStatus(
            _, "fill_value is incompatible with domain"));
  }
  auto unbroadcast = tensorstore::UnbroadcastArray(value.shared_array_view());
  if (rank_ != dynamic_rank && rank_ < unbroadcast.rank()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid fill_value for rank ", rank_, ": ", unbroadcast));
  }
  auto& impl = EnsureUniqueImpl();
  if (impl.fill_value_.valid()) {
    if (impl.fill_value_ != unbroadcast) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Specified fill_value (", unbroadcast,
                              ") does not match existing value in schema (",
                              impl.fill_value_, ")"));
    }
    return absl::OkStatus();
  }
  impl.fill_value_ = std::move(unbroadcast);
  return absl::OkStatus();
}

absl::Status Schema::Set(DimensionUnits value) {
  if (value.empty()) return absl::OkStatus();
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateRank(*this, "dimension_units", value.size()));
  auto& impl = EnsureUniqueImpl();
  return tensorstore::MergeDimensionUnits(impl.dimension_units_, value);
}

absl::Status Schema::Set(Schema value) { return MergeSchemaInto(value, *this); }

IndexDomain<> Schema::domain() const {
  if (!impl_) return {};
  return impl_->domain_;
}

ChunkLayout Schema::chunk_layout() const {
  if (!impl_) return {};
  return impl_->chunk_layout_;
}

Result<IndexTransform<>> Schema::GetTransformForIndexingOperation() const {
  if (rank_ == dynamic_rank) {
    if (impl_ && impl_->fill_value_.valid() && impl_->fill_value_.rank() > 0) {
      return absl::InvalidArgumentError(
          "Cannot apply dimension expression to schema constraints of "
          "unknown rank with non-scalar fill_value");
    }
    return {std::in_place};
  }
  if (impl_ && impl_->domain_.valid()) {
    return IdentityTransform(impl_->domain_);
  }
  return IdentityTransform(rank_);
}

CodecSpec Schema::codec() const {
  if (!impl_) return CodecSpec();
  return impl_->codec_;
}

Schema::FillValue Schema::fill_value() const {
  if (!impl_) return Schema::FillValue();
  return Schema::FillValue(impl_->fill_value_);
}

Schema::DimensionUnits Schema::dimension_units() const {
  if (!impl_) return DimensionUnits();
  return DimensionUnits(impl_->dimension_units_);
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(Schema, JsonBinder())

absl::Status Schema::TransformInputSpaceSchema(IndexTransformView<> transform) {
  if (!transform.valid()) return absl::OkStatus();
  const DimensionIndex rank = rank_;
  if (!RankConstraint::EqualOrUnspecified(rank, transform.input_rank())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Cannot inverse transform schema of rank ", rank,
        " by index transform of rank ", transform.input_rank(), " -> ",
        transform.output_rank()));
  }
  if (!impl_) {
    rank_ = transform.output_rank();
    return absl::OkStatus();
  }
  auto& impl = EnsureUniqueImpl();
  rank_ = transform.output_rank();

  TENSORSTORE_ASSIGN_OR_RETURN(
      impl.chunk_layout_,
      ApplyInverseIndexTransform(transform, std::move(impl.chunk_layout_)));

  // Transform domain and transform.
  if (impl.domain_.valid()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        impl.domain_, TransformInputDomain(impl.domain_, transform),
        tensorstore::MaybeAnnotateStatus(_, "Error transforming domain"));
  }

  // Transform fill value.
  if (impl.fill_value_.valid()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        impl.fill_value_,
        TransformInputBroadcastableArray(transform, impl.fill_value_),
        tensorstore::MaybeAnnotateStatus(_, "Error transforming fill_value"));
  }

  // Transform dimension units.
  if (!impl.dimension_units_.empty()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        impl.dimension_units_,
        TransformInputDimensionUnits(transform,
                                     std::move(impl.dimension_units_)),
        tensorstore::MaybeAnnotateStatus(_,
                                         "Error transforming dimension_units"));
  }
  return absl::OkStatus();
}

Result<Schema> ApplyIndexTransform(IndexTransform<> transform, Schema schema) {
  if (!transform.valid()) return schema;
  const DimensionIndex rank = schema.rank_;
  if (!RankConstraint::EqualOrUnspecified(rank, transform.output_rank())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Cannot transform schema of rank ", rank,
        " by index transform of rank ", transform.input_rank(), " -> ",
        transform.output_rank()));
  }
  if (!schema.impl_) {
    schema.rank_ = transform.input_rank();
    return schema;
  }
  auto& impl = schema.EnsureUniqueImpl();
  schema.rank_ = transform.input_rank();
  auto output_domain = std::move(impl.domain_);
  if (output_domain.valid()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform,
        PropagateBoundsToTransform(output_domain, std::move(transform)));
    impl.domain_ = transform.domain();
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      impl.chunk_layout_,
      ApplyIndexTransform(transform, std::move(impl.chunk_layout_)));
  if (impl.fill_value_.valid()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        impl.fill_value_,
        TransformOutputBroadcastableArray(
            transform, std::move(impl.fill_value_), output_domain),
        tensorstore::MaybeAnnotateStatus(_, "Error transforming fill_value"));
  }

  // Transform dimension units.
  if (!impl.dimension_units_.empty()) {
    impl.dimension_units_ = TransformOutputDimensionUnits(
        transform, std::move(impl.dimension_units_));
  }
  return schema;
}

bool operator==(const Schema::FillValue& a, const Schema::FillValue& b) {
  return a.valid() == b.valid() &&
         (!a.valid() || a.array_view() == b.array_view());
}

bool operator==(Schema::DimensionUnits a, Schema::DimensionUnits b) {
  return internal::RangesEqual(a, b);
}

std::ostream& operator<<(std::ostream& os, Schema::DimensionUnits u) {
  return os << tensorstore::DimensionUnitsToString(u);
}

namespace {
template <typename... T>
inline bool CompareEqualImpl(const Schema& a, const Schema& b) {
  return ((static_cast<T>(a) == static_cast<T>(b)) && ...);
}
}  // namespace

bool operator==(const Schema& a, const Schema& b) {
  return CompareEqualImpl<RankConstraint, DataType, IndexDomain<>, ChunkLayout,
                          Schema::FillValue, CodecSpec, Schema::DimensionUnits>(
      a, b);
}

std::ostream& operator<<(std::ostream& os, const Schema& schema) {
  auto json_result = schema.ToJson();
  if (!json_result.ok()) {
    return os << "<unprintable>";
  }
  return os << json_result->dump();
}

ChunkLayout& Schema::MutableLayoutInternal() {
  return EnsureUniqueImpl().chunk_layout_;
}

absl::Status Schema::ValidateLayoutInternal() {
  const DimensionIndex rank =
      impl_ ? impl_->chunk_layout_.rank() : dynamic_rank;
  if (rank == dynamic_rank) return absl::OkStatus();
  return ValidateRank(*this, "chunk_layout", rank);
}

namespace internal {
absl::Status ChooseReadWriteChunkGrid(MutableBoxView<> chunk_template,
                                      const Schema& schema) {
  if (!RankConstraint::EqualOrUnspecified(chunk_template.rank(),
                                          schema.rank())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Expected schema to have rank ", chunk_template.rank(),
        ", but received schema of rank: ", schema.rank()));
  }
  auto domain = schema.domain();
  BoxView<> box;
  if (domain.valid()) {
    box = domain.box();
  } else {
    box = BoxView(chunk_template.rank());
  }
  return internal::ChooseReadWriteChunkGrid(schema.chunk_layout(), box,
                                            chunk_template);
}

}  // namespace internal
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::Schema,
    tensorstore::serialization::JsonBindableSerializer<tensorstore::Schema>())
