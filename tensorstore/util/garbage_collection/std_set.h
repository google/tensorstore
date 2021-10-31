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

#ifndef TENSORSTORE_UTIL_GARBAGE_COLLECTION_STD_SET_H_
#define TENSORSTORE_UTIL_GARBAGE_COLLECTION_STD_SET_H_

#include <set>

#include "tensorstore/util/garbage_collection/garbage_collection.h"

namespace tensorstore {
namespace garbage_collection {

template <typename T, typename Compare, typename Allocator>
struct GarbageCollection<std::set<T, Compare, Allocator>>
    : public ContainerGarbageCollection<std::set<T, Compare, Allocator>, T> {};

template <typename T, typename Compare, typename Allocator>
struct GarbageCollection<std::multiset<T, Compare, Allocator>>
    : public ContainerGarbageCollection<std::multiset<T, Compare, Allocator>,
                                        T> {};

}  // namespace garbage_collection
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_GARBAGE_COLLECTION_STD_SET_H_
