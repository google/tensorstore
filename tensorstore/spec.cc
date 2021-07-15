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

#include "tensorstore/driver/driver.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/internal/json.h"

namespace tensorstore {

Result<Spec> Spec::With(SpecOptions&& options) && {
  auto status = internal::TransformAndApplyOptions(impl_, std::move(options));
  if (!status.ok()) return status;
  return std::move(*this);
}

Result<Spec> Spec::With(SpecOptions&& options) const& {
  return Spec(*this).With(std::move(options));
}

absl::Status Spec::Set(SpecOptions&& options) {
  return internal::TransformAndApplyOptions(impl_, std::move(options));
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

Result<CodecSpec::Ptr> Spec::codec() const {
  return internal::GetEffectiveCodec(impl_);
}

Result<SharedArray<const void>> Spec::fill_value() const {
  return internal::GetEffectiveFillValue(impl_);
}

std::ostream& operator<<(std::ostream& os, const Spec& spec) {
  return os << ::nlohmann::json(spec).dump();
}

bool operator==(const Spec& a, const Spec& b) {
  auto result_a = a.ToJson(tensorstore::IncludeContext{true});
  auto result_b = b.ToJson(tensorstore::IncludeContext{true});
  if (!result_a || !result_b) return false;
  return *result_a == *result_b;
}

namespace jb = tensorstore::internal_json_binding;
TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    Spec,
    jb::Projection(&Spec::impl_, internal::TransformedDriverSpecJsonBinder))

Result<IndexTransform<>> Spec::GetTransformForIndexingOperation() const {
  if (impl_.transform.valid()) return impl_.transform;
  if (impl_.driver_spec) {
    const DimensionIndex rank = impl_.driver_spec->schema().rank();
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
    TENSORSTORE_ASSIGN_OR_RETURN(
        spec, std::move(spec).With(RankConstraint{transform.output_rank()}));
    spec.impl_.transform = std::move(transform);
  }
  return spec;
}

}  // namespace tensorstore
