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

#ifndef TENSORSTORE_INTERNAL_JSON_FWD_H_
#define TENSORSTORE_INTERNAL_JSON_FWD_H_

/// \file
/// Forward declaration of ::nlohmann::json type.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nlohmann {

template <typename T, typename SFINAE>
struct adl_serializer;

template <template <typename U, typename V, typename... Args>
          class ObjectType /* = std::map*/,
          template <typename U, typename... Args>
          class ArrayType /* = std::vector*/,
          class StringType /*= std::string*/, class BooleanType /* = bool*/,
          class NumberIntegerType /* = std::int64_t*/,
          class NumberUnsignedType /* = std::uint64_t*/,
          class NumberFloatType /* = double*/,
          template <typename U> class AllocatorType /* = std::allocator*/,
          template <typename T, typename SFINAE = void>
          class JSONSerializer /* = adl_serializer*/,
          class BinaryType /* = std::vector<std::uint8_t>*/>
class basic_json;

using json = basic_json<std::map, std::vector, std::string, bool, std::int64_t,
                        std::uint64_t, double, std::allocator, adl_serializer,
                        std::vector<std::uint8_t>>;

}  // namespace nlohmann

#endif  // TENSORSTORE_INTERNAL_JSON_FWD_H_
