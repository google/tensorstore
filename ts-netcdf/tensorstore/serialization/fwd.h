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

#ifndef TENSORSTORE_SERIALIZATION_FWD_H_
#define TENSORSTORE_SERIALIZATION_FWD_H_

/// \file
///
/// Forward declarations for serialization types.
///
/// This header is sufficient for declaring `Serializer` specializations.

namespace tensorstore {
namespace serialization {

class EncodeSink;
class DecodeSource;

/// Defines the default serialization for a given type `T`.
///
/// The extra `SFINAE` parameter must always be void and may be used to define
/// partial specializations.
template <typename T, typename SFINAE = void>
struct Serializer;

/// Polymorphic type registry, see `registry.h` for details.
class Registry;

template <typename Ptr>
Registry& GetRegistry();

/// Defines an explicit specialization of `Serializer<TYPE>` with static
/// out-of-line `Encode` and `Decode` methods.
///
/// This must be used at global scope (outside any namespaces).
///
/// This is intended to be used in a header file, to reduce the boilerplate
/// required to define a `Serializer` specialization.  In an associated ".cc"
/// file, the `Encode` and `Decode` methods can be defined manually, or can be
/// forwarded to an existing `Serializer`-like type using
/// `TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION`.
#define TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(TYPE)               \
  template <>                                                             \
  struct tensorstore::serialization::Serializer<TYPE> {                   \
    [[nodiscard]] static bool Encode(                                     \
        tensorstore::serialization::EncodeSink& sink, const TYPE& value); \
    [[nodiscard]] static bool Decode(                                     \
        tensorstore::serialization::DecodeSource& source, TYPE& value);   \
  };                                                                      \
  /**/

/// Defines `Serializer<TYPE>::Encode` and `Serialier<TYPE>::Decode` methods
/// that simply forward to `SERIALIZER`.
///
/// This must be used at global scope (outside any namespaces).
///
/// `SERIALIZER` should be an expression that evaluates to a `Serializer` object
/// with `Encode` and `Decode` methods.
///
/// This is intended to be used in a ".cc" file.
#define TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(TYPE, SERIALIZER)   \
  bool tensorstore::serialization::Serializer<TYPE>::Encode(             \
      tensorstore::serialization::EncodeSink& sink, const TYPE& value) { \
    return (SERIALIZER).Encode(sink, value);                             \
  }                                                                      \
  bool tensorstore::serialization::Serializer<TYPE>::Decode(             \
      tensorstore::serialization::DecodeSource& source, TYPE& value) {   \
    return (SERIALIZER).Decode(source, value);                           \
  }                                                                      \
  /**/

}  // namespace serialization
}  // namespace tensorstore

#endif  // TENSORSTORE_SERIALIZATION_SERIALIZATION_H_
