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

#ifndef TENSORSTORE_PROTO_SCHEMA_H_
#define TENSORSTORE_PROTO_SCHEMA_H_

#include "absl/status/status.h"
#include "tensorstore/proto/schema.pb.h"
#include "tensorstore/schema.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Parses an Schema<> from a proto.
///
/// \param proto The Schema proto representation.
Result<Schema> ParseSchemaFromProto(const ::tensorstore::proto::Schema& proto);

/// Encodes a schema as a Schema proto.
///
/// \param proto[out] The Schema proto representation.
/// \param schema The input Schema.
void EncodeToProto(::tensorstore::proto::Schema& proto,  // NOLINT
                   const Schema& schema);

}  // namespace tensorstore

#endif
