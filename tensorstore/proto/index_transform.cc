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

#include "tensorstore/proto/index_transform.h"

#include <algorithm>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/json/array.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/proto/index_transform.pb.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

Result<IndexDomain<>> ParseIndexDomainFromProto(
    const ::tensorstore::proto::IndexDomain& proto,
    DimensionIndex rank_constraint) {
  // Maybe deduce rank.
  const DimensionIndex rank = [&]() -> DimensionIndex {
    if (proto.has_rank()) return proto.rank();
    if (proto.origin_size() > 0) return proto.origin_size();
    if (proto.shape_size() > 0) return proto.shape_size();
    return proto.labels_size();
  }();
  if (rank < 0 || rank > kMaxRank) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Expected rank to be in the range [0, ", kMaxRank,
                            "], but is: ", rank));
  }

  if (!RankConstraint::EqualOrUnspecified(rank_constraint, rank)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Expected rank to be ", rank_constraint, ", but is: ", rank));
  }

  if (proto.origin_size() > 0 && proto.origin_size() != rank) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Proto origin must include ", rank, " items"));
  }
  if (proto.shape_size() > 0 && proto.shape_size() != rank) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Proto shape must include ", rank, " items"));
  }
  if (proto.labels_size() > 0 && proto.labels_size() != rank) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Proto labels must include ", rank, " items"));
  }
  if (proto.implicit_lower_bound_size() > 0 &&
      proto.implicit_lower_bound_size() != rank) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Proto implicit_lower_bound must include ", rank, " items"));
  }
  if (proto.implicit_upper_bound_size() > 0 &&
      proto.implicit_upper_bound_size() != rank) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Proto implicit_upper_bound must include ", rank, " items"));
  }

  IndexDomainBuilder builder(rank);
  if (proto.origin_size() > 0) {
    std::copy(proto.origin().begin(), proto.origin().end(),
              builder.origin().begin());
    if (proto.implicit_lower_bound_size() > 0) {
      std::copy(proto.implicit_lower_bound().begin(),
                proto.implicit_lower_bound().end(),
                builder.implicit_lower_bounds().begin());
    }
  }
  if (proto.shape_size() > 0) {
    std::copy(proto.shape().begin(), proto.shape().end(),
              builder.shape().begin());
    if (proto.implicit_upper_bound_size() > 0) {
      std::copy(proto.implicit_upper_bound().begin(),
                proto.implicit_upper_bound().end(),
                builder.implicit_upper_bounds().begin());
    }
  }
  if (!proto.labels().empty()) {
    std::copy(proto.labels().begin(), proto.labels().end(),
              builder.labels().begin());
  }

  return builder.Finalize();
}

/// Parses an IndexTransform from a proto.
Result<IndexTransform<>> ParseIndexTransformFromProto(
    const proto::IndexTransform& proto, DimensionIndex input_rank_constraint,
    DimensionIndex output_rank_constraint) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto input_domain,
      ParseIndexDomainFromProto(proto.input_domain(), input_rank_constraint));

  const DimensionIndex rank = input_domain.rank();
  const DimensionIndex output_rank = [&]() -> DimensionIndex {
    if (proto.has_output_rank()) return proto.output_rank();
    if (proto.output_size() == 0) return rank;
    return proto.output_size();
  }();

  if (output_rank < 0 || output_rank > kMaxRank) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Expected output_rank to be in the range [0, ",
                            kMaxRank, "], but is: ", output_rank));
  }
  if (!RankConstraint::EqualOrUnspecified(output_rank_constraint,
                                          output_rank)) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Expected output_rank to be ",
                            output_rank_constraint, ", but is: ", output_rank));
  }

  IndexTransformBuilder builder(rank, output_rank);
  if (proto.output().empty() && output_rank == rank) {
    builder.output_identity_transform();
  } else {
    if (proto.output_size() != output_rank) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Proto output expected ", output_rank, " items"));
    }
    for (DimensionIndex output_dim = 0; output_dim < output_rank;
         ++output_dim) {
      const auto& output_proto = proto.output(output_dim);

      if (output_proto.has_index_array()) {
        const auto& array = output_proto.index_array();

        std::vector<tensorstore::Index> shape(array.shape().cbegin(),
                                              array.shape().cend());
        auto a = MakeCopy(
            tensorstore::Array(tensorstore::ElementPointer<const int64_t>(
                                   &array.data()[0], dtype_v<int64_t>),
                               shape));

        Result<IndexInterval> index_range = IndexInterval();
        if (output_proto.has_index_array_inclusive_min() &&
            output_proto.has_index_array_exclusive_max()) {
          index_range =
              IndexInterval::HalfOpen(output_proto.index_array_inclusive_min(),
                                      output_proto.index_array_exclusive_max());
        }
        builder.output_index_array(output_dim, output_proto.offset(),
                                   output_proto.stride(), a, index_range);

        continue;
      }
      if (output_proto.stride() == 0) {
        builder.output_constant(output_dim, output_proto.offset());
        continue;
      }
      builder.output_single_input_dimension(output_dim, output_proto.offset(),
                                            output_proto.stride(),
                                            output_proto.input_dimension());
    }
  }

  return builder.input_domain(input_domain).Finalize();
}

void EncodeToProto(proto::IndexDomain& proto,  // NOLINT
                   IndexDomainView<> d) {
  const DimensionIndex rank = d.rank();

  bool all_implicit_lower = true;
  bool all_implicit_upper = true;
  size_t implicit_lower_count = 0;
  size_t implicit_upper_count = 0;
  bool has_labels = false;
  for (DimensionIndex i = 0; i < rank; ++i) {
    implicit_lower_count += d.implicit_lower_bounds()[i];
    all_implicit_lower = all_implicit_lower && d.implicit_lower_bounds()[i] &&
                         (d[i].inclusive_min() == -kInfIndex);
    implicit_upper_count += d.implicit_upper_bounds()[i];
    all_implicit_upper = all_implicit_upper && d.implicit_upper_bounds()[i] &&
                         (d[i].exclusive_max() == (+kInfIndex + 1));
    has_labels |= !d.labels()[i].empty();
  }

  if (all_implicit_lower && all_implicit_upper && !has_labels) {
    proto.set_rank(rank);
  }
  for (DimensionIndex i = 0; i < rank; i++) {
    if (!all_implicit_lower) {
      proto.add_origin(d.origin()[i]);
      if (implicit_lower_count > 0) {
        proto.add_implicit_lower_bound(d.implicit_lower_bounds()[i]);
      }
    }
    if (!all_implicit_upper) {
      proto.add_shape(d.shape()[i]);
      if (implicit_upper_count > 0) {
        proto.add_implicit_upper_bound(d.implicit_upper_bounds()[i]);
      }
    }
    if (has_labels) {
      proto.add_labels(d.labels()[i]);
    }
  }
}

void EncodeToProto(proto::IndexTransform& proto,  // NOLINT
                   IndexTransformView<> t) {
  EncodeToProto(*proto.mutable_input_domain(), t.input_domain());
  const DimensionIndex input_rank = t.input_rank();

  bool all_identity = true;
  for (DimensionIndex i = 0; i < t.output_rank(); ++i) {
    const auto map = t.output_index_map(i);
    auto* out_proto = proto.add_output();
    if (map.offset() != 0) {
      out_proto->set_offset(map.offset());
      all_identity = false;
    }
    if (map.method() != OutputIndexMethod::constant) {
      out_proto->set_stride(map.stride());
      if (map.stride() != 1) all_identity = false;
    }
    switch (map.method()) {
      case OutputIndexMethod::constant:
        all_identity = false;
        break;
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex input_dim = map.input_dimension();
        out_proto->set_input_dimension(input_dim);
        if (input_dim != i) all_identity = false;
        break;
      }
      case OutputIndexMethod::array: {
        all_identity = false;

        const auto index_array_data = map.index_array();
        auto index_array =
            UnbroadcastArrayPreserveRank(index_array_data.array_ref());

        // Store the unbroadcast shape.
        auto* out_array = out_proto->mutable_index_array();
        for (Index size : index_array.shape()) {
          out_array->add_shape(size);
        }

        // If `index_array` contains values outside `index_range`, encode
        // `index_range` as well to avoid expanding the range.
        IndexInterval index_range = index_array_data.index_range();
        if (index_range != IndexInterval::Infinite() &&
            !ValidateIndexArrayBounds(index_range, index_array).ok()) {
          out_proto->set_index_array_inclusive_min(index_range.inclusive_min());
          out_proto->set_index_array_exclusive_max(index_range.exclusive_max());
        }

        // Store the index array data.
        IterateOverArrays(
            [&](const Index* value) { out_array->add_data(*value); }, c_order,
            index_array);
        break;
      }
    }
  }
  if (all_identity) {
    proto.clear_output();
  }
  if (t.output_rank() != input_rank && t.output_rank() == 0) {
    proto.set_output_rank(t.output_rank());
  }
}

}  // namespace tensorstore
