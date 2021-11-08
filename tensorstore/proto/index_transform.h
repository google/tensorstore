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

#ifndef TENSORSTORE_PROTO_INDEX_TRANSFORM_H_
#define TENSORSTORE_PROTO_INDEX_TRANSFORM_H_

#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/proto/index_transform.pb.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Parses an IndexTransform<> from a proto.
///
/// \param proto The IndexTransform proto representation.
/// \param input_rank_constraint Constraints on the input_rank.
/// \param output_rank_constraint Constraints on the output_rank.
Result<IndexTransform<>> ParseIndexTransformFromProto(
    const ::tensorstore::proto::IndexTransform& proto,
    DimensionIndex input_rank_constraint = dynamic_rank,
    DimensionIndex output_rank_constraint = dynamic_rank);

/// Parses an IndexDomain<> from a proto.
///
/// \param proto The IndexDomain proto representation.
/// \param rank_constraint Constraints on the rank.
Result<IndexDomain<>> ParseIndexDomainFromProto(
    const ::tensorstore::proto::IndexDomain& proto,
    DimensionIndex rank_constraint = dynamic_rank);

/// Encodes an index transform as IndexTransform proto.
///
/// \param proto[out] The IndexTransform proto representation.
/// \param t Index transform.
void EncodeToProto(::tensorstore::proto::IndexTransform& proto,  // NOLINT
                   IndexTransformView<> t);

/// Encodes an index domain as IndexDomain proto.
///
/// \param proto[out] The IndexDomain proto representation.
/// \param d Index domain.
void EncodeToProto(::tensorstore::proto::IndexDomain& proto,  // NOLINT
                   IndexDomainView<> d);

}  // namespace tensorstore

#endif
