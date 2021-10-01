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

#ifndef TENSORSTORE_SERIALIZATION_STD_MAP_H_
#define TENSORSTORE_SERIALIZATION_STD_MAP_H_

#include <map>

#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/std_pair.h"

namespace tensorstore {
namespace serialization {

template <typename Key, typename T, typename Compare, typename Allocator>
struct Serializer<std::map<Key, T, Compare, Allocator>>
    : public ContainerSerializer<std::map<Key, T, Compare, Allocator>,
                                 std::pair<Key, T>> {};

template <typename Key, typename T, typename Compare, typename Allocator>
struct Serializer<std::multimap<Key, T, Compare, Allocator>>
    : public ContainerSerializer<std::multimap<Key, T, Compare, Allocator>,
                                 std::pair<Key, T>> {};

}  // namespace serialization
}  // namespace tensorstore

#endif  // TENSORSTORE_SERIALIZATION_STD_MAP_H_
