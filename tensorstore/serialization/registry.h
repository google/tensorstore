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

#ifndef TENSORSTORE_SERIALIZATION_REGISTRY_H_
#define TENSORSTORE_SERIALIZATION_REGISTRY_H_

/// \file
///
/// Support for serializing polymorphic types (i.e. derived types of a common
/// base with a virtual destructor) via a smart pointer to the base type.
///
/// Derived types must define a `constexpr static const char id[]` member
/// specifying a unique identifier and must be explicitly registered via a call
/// to `Register`.  The unique identifier need only be unique for a given base
/// class `Ptr` type, and is used in the serialized representation to indicate
/// the derived type.
///
/// Various smart pointers are supported: `std::unique_ptr`, `std::shared_ptr`,
/// and `tensorstore::internal::IntrusivePtr`.
///
/// This is used to implement serialization for TensorStore drivers and KvStore
/// drivers.

#include <string_view>
#include <type_traits>
#include <typeindex>
#include <typeinfo>

#include "absl/status/status.h"
#include "tensorstore/internal/heterogeneous_container.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/serialization.h"

namespace tensorstore {
namespace serialization {

/// Registry of derived types that can be serialized and deserialized for a
/// given smart pointer type `Ptr`.
class Registry {
 public:
  /// Registration info for a given `Derived` type.
  struct Entry {
    /// Encode function, `value` must be non-null `const Ptr*`.
    using Encode = bool (*)(EncodeSink& sink, const void* value);

    /// Decode function, `value` must be non-null `Ptr*`.
    using Decode = bool (*)(DecodeSource& source, void* value);

    /// Equal to `typeid(Derived)`.
    const std::type_info& type;

    /// Equal to `Derived::id`.
    std::string_view id;

    Encode encode;
    Decode decode;

    std::type_index type_index() const { return type; }
  };

  Registry();
  ~Registry();

  /// Adds a derived type to this registry.
  ///
  /// It is a fatal error if `entry.id` or `entry.type` is already registered.
  void Add(const Entry& entry);

  /// Encodes a value using the registry.
  ///
  /// \param sink Sink to use for encoding.
  /// \param value Must be non-null pointer to a `const Ptr` object.
  /// \param type Must equal `typeid(**static_cast<const Ptr*>(value))`.
  /// \returns `true` on success, `false` on error.  If `false` is returned, an
  ///     error status is also set on `sink`.
  [[nodiscard]] bool Encode(EncodeSink& sink, const void* value,
                            const std::type_info& type);

  /// Decodes a value using the registry.
  ///
  /// \param source Source to use for decoding.
  /// \param value Must be non-null pointer to a `Ptr` object.  On success, will
  ///     be reset to point to the decoded object.
  /// \returns `true` on success, `false` on error.  If `false` is returned, an
  ///     error status is also set on `sink`, unless the error is EOF.
  [[nodiscard]] bool Decode(DecodeSource& source, void* value);

 private:
  internal::HeterogeneousHashSet<const Entry*, std::string_view, &Entry::id>
      by_id_;
  internal::HeterogeneousHashSet<const Entry*, std::type_index,
                                 &Entry::type_index>
      by_type_;
};

/// Returns the global registry for a given smart pointer type `Ptr`.
///
/// To improve compilation efficiency, in a header file this may be declared
/// extern for a specific type:
///
///     extern template Registry& GetRegistry<internal::IntrusivePtr<MyBase>>();
///
/// Then in a single source file, it must be explicitly instantiated:
///
///     template Registry& GetRegistry<internal::IntrusivePtr<MyBase>>();
template <typename Ptr>
Registry& GetRegistry() {
  static internal::NoDestructor<Registry> registry;
  return *registry;
}

/// Registers `Derived` for serialization and deserialization via the smart
/// pointer type `Ptr` to a base class of `Derived`.
template <typename Ptr, typename Derived>
void Register() {
  using Base =
      std::remove_const_t<typename std::pointer_traits<Ptr>::element_type>;
  static_assert(std::has_virtual_destructor_v<Base>);
  static_assert(std::is_base_of_v<Base, Derived>);
  static const Registry::Entry entry{
      typeid(Derived),
      Derived::id,
      +[](EncodeSink& sink, const void* value) -> bool {
        return serialization::Encode(
            sink, *static_cast<const Derived*>(
                      static_cast<const Ptr*>(value)->get()));
      },
      +[](DecodeSource& source, void* value) -> bool {
        auto& ptr = *static_cast<Ptr*>(value);
        ptr.reset(new Derived);
        return serialization::Decode(
            source,
            *const_cast<Derived*>(static_cast<const Derived*>(ptr.get())));
      },
  };
  GetRegistry<Ptr>().Add(entry);
}

/// Serializer for base smart pointer type `Ptr` for registered types.
///
/// \tparam Ptr Smart pointer to base class with virtual destructor.
template <typename Ptr>
struct RegistrySerializer {
  [[nodiscard]] static bool Encode(EncodeSink& sink, const Ptr& value) {
    return GetRegistry<Ptr>().Encode(sink, &value, typeid(*value));
  }
  [[nodiscard]] static bool Decode(DecodeSource& source, Ptr& value) {
    return GetRegistry<Ptr>().Decode(source, &value);
  }
};

}  // namespace serialization
}  // namespace tensorstore

#endif  // TENSORSTORE_SERIALIZATION_REGISTRY_H_
