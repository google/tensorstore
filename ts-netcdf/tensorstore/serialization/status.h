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

#ifndef TENSORSTORE_SERIALIZATION_STATUS_H_
#define TENSORSTORE_SERIALIZATION_STATUS_H_

#include "absl/status/status.h"
#include "tensorstore/serialization/fwd.h"

namespace tensorstore {
namespace serialization {

/// Serializer for `absl::Status` values that must not equal `absl::OkStatus()`.
struct ErrorStatusSerializer {
  [[nodiscard]] static bool Encode(EncodeSink& sink,
                                   const absl::Status& status);
  [[nodiscard]] static bool Decode(DecodeSource& source, absl::Status& status);
};

}  // namespace serialization
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(absl::Status)

#endif  // TENSORSTORE_SERIALIZATION_STATUS_H_
