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
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/index_space/transform_broadcastable_array.h"
#include "tensorstore/internal/data_type_json_binder.h"
#include "tensorstore/internal/dimension_indexed_json_binder.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_array.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

class Schema::Impl : public Schema::Builder {
 public:
  Impl() = default;
  Impl(const Schema::Builder& other) : Schema::Builder(other) {}
  std::atomic<size_t> ref_count_{0};
  DimensionIndex rank_;
};

void intrusive_ptr_increment(Schema::Impl* p) {
  p->ref_count_.fetch_add(1, std::memory_order_acq_rel);
}

void intrusive_ptr_decrement(Schema::Impl* p) {
  if (p->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    delete p;
  }
}

Result<Schema> Schema::Make(Builder builder) {
  DimensionIndex rank = dynamic_rank;
  if (builder.domain.valid()) {
    rank = builder.domain.rank();
    if (!IsRankExplicitlyConvertible(builder.domain.rank(),
                                     builder.chunk_layout.rank())) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Rank of chunk_layout (", builder.chunk_layout.rank(),
          ") does not match rank of domain (", builder.domain.rank(), ")"));
    }
    if (builder.fill_value.valid()) {
      TENSORSTORE_RETURN_IF_ERROR(
          ValidateShapeBroadcast(builder.fill_value.shape(),
                                 builder.domain.shape()),
          tensorstore::MaybeAnnotateStatus(_, "Invalid fill_value"));
    }
  } else {
    rank = builder.chunk_layout.rank();
  }
  Schema schema;
  schema.impl_.reset(new Impl);
  static_cast<Builder&>(*schema.impl_) = std::move(builder);
  schema.impl_->rank_ = rank;
  return schema;
}

DataType Schema::dtype() const { return impl_ ? impl_->dtype : DataType(); }

DimensionIndex Schema::rank() const {
  if (!impl_) return dynamic_rank;
  return impl_->rank_;
}

IndexDomain<> Schema::domain() const {
  return impl_ ? impl_->domain : IndexDomain<>();
}

ChunkLayout Schema::chunk_layout() const {
  return impl_ ? impl_->chunk_layout : ChunkLayout();
}

CodecSpec::Ptr Schema::codec() const {
  return impl_ ? impl_->codec : CodecSpec::Ptr();
}

Schema::FillValue Schema::fill_value() const {
  return impl_ ? FillValue(impl_->fill_value) : FillValue();
}

Result<IndexTransform<>> Schema::identity_transform() const {
  const DimensionIndex rank = this->rank();
  if (impl_->domain.valid()) return IdentityTransform(impl_->domain);
  if (rank != dynamic_rank) return IdentityTransform(rank);
  if (impl_->fill_value.valid() && impl_->fill_value.rank() > 0) {
    return absl::InvalidArgumentError(
        "Cannot apply dimension expression to schema of "
        "unknown rank with non-scalar fill_value");
  }
  return {std::in_place};
}

Result<Schema> ApplyIndexTransform(IndexTransform<> transform, Schema schema) {
  if (!schema.impl_ || !transform.valid()) {
    return schema;
  }
  const DimensionIndex rank = schema.rank();
  const DimensionIndex new_rank = transform.input_rank();
  if (!IsRankExplicitlyConvertible(rank, transform.output_rank())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Cannot transform schema of rank ", rank,
        " by index transform of rank ", transform.input_rank(), " -> ",
        transform.output_rank()));
  }
  if (schema.impl_->ref_count_.load(std::memory_order_acquire) != 1) {
    schema.impl_.reset(
        new Schema::Impl(static_cast<const Schema::Builder&>(*schema.impl_)));
  }
  auto& impl = *schema.impl_;
  auto output_domain = std::move(impl.domain);
  if (output_domain.valid()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform,
        PropagateBoundsToTransform(output_domain, std::move(transform)));
    impl.domain = transform.domain();
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      impl.chunk_layout,
      ApplyIndexTransform(transform, std::move(impl.chunk_layout)));
  TENSORSTORE_ASSIGN_OR_RETURN(
      impl.fill_value,
      TransformOutputBroadcastableArray(transform, std::move(impl.fill_value),
                                        output_domain));
  impl.rank_ = new_rank;
  return schema;
}

bool operator==(const Schema::FillValue& a, const Schema::FillValue& b) {
  return a.valid() == b.valid() &&
         (!a.valid() || a.array_view() == b.array_view());
}

bool operator==(const Schema& a, const Schema& b) {
  return a.dtype() == b.dtype() && a.domain() == b.domain() &&
         a.chunk_layout() == b.chunk_layout() && a.codec() == b.codec() &&
         a.fill_value() == b.fill_value();
}

std::ostream& operator<<(std::ostream& os, const Schema& schema) {
  auto json_result = schema.ToJson();
  if (!json_result.ok()) {
    return os << "<unprintable>";
  }
  return os << json_result->dump();
}

namespace {
namespace jb = tensorstore::internal_json_binding;

auto SchemaJsonBinder() {
  return jb::Object(
      jb::Member("dtype", jb::Projection(&Schema::Builder::dtype,
                                         jb::ConstrainedDataTypeJsonBinder)),
      jb::Member("domain", jb::Projection(&Schema::Builder::domain)),
      jb::Member("chunk_layout",
                 jb::Projection(&Schema::Builder::chunk_layout)),
      jb::Member("codec", jb::Projection(&Schema::Builder::codec)),
      [](auto is_loading, const auto& options, auto* obj, auto* j) {
        DataType dtype = dtype_v<::nlohmann::json>;
        if constexpr (is_loading) {
          if (auto d = obj->dtype; d.valid()) {
            dtype = d;
          }
        }
        return jb::Member(
            "fill_value",
            jb::DefaultPredicate<jb::kNeverIncludeDefaults>(
                [](auto* obj) {}, [](auto* obj) { return !obj->valid(); },
                jb::NestedVoidArray(dtype)))(is_loading, options,
                                             &obj->fill_value, j);
      });
}
}  // namespace

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    Schema,
    jb::Object([](auto is_loading, const auto& options, auto* obj, auto* j) {
      if constexpr (is_loading) {
        Schema::Builder builder;
        TENSORSTORE_RETURN_IF_ERROR(
            SchemaJsonBinder()(is_loading, options, &builder, j));
        TENSORSTORE_ASSIGN_OR_RETURN(*obj, Schema::Make(std::move(builder)));
        return absl::OkStatus();
      } else {
        if (!obj->impl_) return absl::OkStatus();
        return SchemaJsonBinder()(is_loading, options, obj->impl_.get(), j);
      }
    }))

}  // namespace tensorstore
