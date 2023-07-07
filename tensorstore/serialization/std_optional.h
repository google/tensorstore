// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_SERIALIZATION_STD_OPTIONAL_H_
#define TENSORSTORE_SERIALIZATION_STD_OPTIONAL_H_

#include <optional>

#include "tensorstore/serialization/serialization.h"

namespace tensorstore {
namespace serialization {

template <typename T>
struct Serializer<std::optional<T>>
    : public OptionalSerializer<std::optional<T>> {};

}  // namespace serialization
}  // namespace tensorstore

#endif  // TENSORSTORE_SERIALIZATION_STD_OPTIONAL_H_
