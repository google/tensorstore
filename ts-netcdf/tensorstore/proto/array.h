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

#ifndef TENSORSTORE_PROTO_ARRAY_H_
#define TENSORSTORE_PROTO_ARRAY_H_

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/proto/array.pb.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

void EncodeToProtoImpl(::tensorstore::proto::Array& proto,
                       OffsetArrayView<const void> array);

/// Parses an Array<> from a proto.
///
/// \param proto The Schema proto representation.
Result<SharedArray<void, dynamic_rank, offset_origin>> ParseArrayFromProto(
    const ::tensorstore::proto::Array& proto,
    ArrayOriginKind origin_kind = offset_origin,
    DimensionIndex rank_constraint = dynamic_rank);

/// Encodes an Array as an Array proto.
///
/// \param proto[out] The Array proto representation.
/// \param schema The input Schema.
template <typename Element, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutCKind>
void EncodeToProto(
    ::tensorstore::proto::Array& proto,
    const Array<Shared<Element>, Rank, OriginKind, LayoutCKind>& value) {
  EncodeToProtoImpl(proto, value);
}

}  // namespace tensorstore

#endif  // TENSORSTORE_PROTO_ARRAY_H_
